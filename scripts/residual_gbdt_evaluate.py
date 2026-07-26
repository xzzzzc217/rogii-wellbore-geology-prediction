"""Frozen leakage-safe OOF evaluation for the residual GBDT family.

Two protocols must independently pass:

* five folds grouped by exact typewell fingerprint;
* five spatial-cluster holdouts, with every validation typewell fingerprint
  removed from that fold's training wells.

The model never receives a well identifier or hidden TVT.  Hidden TVT is
isolated as an offline label after target-free features have been built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rogii.residual_gbdt import (  # noqa: E402
    RESIDUAL_FEATURE_NAMES,
    ResidualFeatureConfig,
    build_residual_features,
    sample_suffix_indices,
    typewell_fingerprint,
)
from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
)

RANDOM_SEED = 20_260_725
FOLD_COUNT = 5
MAX_TRAIN_ROWS_PER_WELL = 96
PREDICTION_RESIDUAL_CLIP = 80.0
BOOTSTRAP_REPLICATES = 10_000

# Frozen before OOF scoring.
MODEL_PARAMETERS: dict[str, Any] = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 600,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "rsm": 0.85,
    "random_seed": RANDOM_SEED,
    "thread_count": 8,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}

POOLED_DELTA_LIMIT = -0.10
REQUIRED_NONPOSITIVE_FOLDS = 4
MAXIMUM_FOLD_DELTA = 0.20
P95_DELTA_LIMIT = 0.25
WORST_DELTA_LIMIT = 1.00

BASELINE_METHODS = ("hold_last", "u_boundary", "slope_decay")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "residual_gbdt_oof.json",
    )
    parser.add_argument(
        "--max-wells",
        type=int,
        default=0,
        help="Smoke-test prefix of wells; zero requires the frozen full 773.",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _well_id(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


def _read_inputs(
    horizontal_path: Path,
    train_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    well_id = _well_id(horizontal_path)
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=[*HORIZONTAL_INPUT_COLUMNS, "TVT"],
    )
    truth = pd.to_numeric(horizontal.pop("TVT"), errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    horizontal = horizontal.loc[:, HORIZONTAL_INPUT_COLUMNS].copy()
    typewell = pd.read_csv(
        train_dir / f"{well_id}__typewell.csv",
        usecols=list(TYPEWELL_INPUT_COLUMNS),
    ).loc[:, TYPEWELL_INPUT_COLUMNS]
    return horizontal, typewell, truth


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    difference = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    if difference.size == 0 or not np.all(np.isfinite(difference)):
        raise ValueError("RMSE inputs must be finite and non-empty")
    return float(np.sqrt(np.mean(difference * difference)))


def _scan_wells(paths: list[Path], train_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    feature_config = ResidualFeatureConfig()
    started = time.perf_counter()
    for position, path in enumerate(paths, start=1):
        horizontal, typewell, truth = _read_inputs(path, train_dir)
        block = build_residual_features(
            horizontal,
            typewell,
            config=feature_config,
        )
        truth_hidden = truth[block.evaluation_mask]
        if not np.all(np.isfinite(truth_hidden)):
            raise ValueError(f"{_well_id(path)}: hidden TVT label is non-finite")
        labels = truth_hidden - block.anchor_tvt
        selected = sample_suffix_indices(
            len(truth_hidden),
            MAX_TRAIN_ROWS_PER_WELL,
        )
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(dtype=float)
        records.append(
            {
                "well_id": _well_id(path),
                "path": str(path),
                "fingerprint": typewell_fingerprint(typewell),
                "centroid_x": float(np.median(x)),
                "centroid_y": float(np.median(y)),
                "hidden_rows": int(len(truth_hidden)),
                "sample_features": block.features[selected],
                "sample_labels": labels[selected].astype(np.float32),
                "sample_weight": float(len(truth_hidden) / len(selected)),
            }
        )
        if position % 50 == 0 or position == len(paths):
            print(
                f"[feature scan {position:>3}/{len(paths)}] "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return records


def _relabel_spatial_clusters(
    labels: np.ndarray,
    coordinates: np.ndarray,
    *,
    fold_count: int = FOLD_COUNT,
) -> np.ndarray:
    ordering = sorted(
        range(fold_count),
        key=lambda label: (
            float(np.median(coordinates[labels == label, 0])),
            float(np.median(coordinates[labels == label, 1])),
        ),
    )
    mapping = {old: new for new, old in enumerate(ordering)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int16)


def build_fold_assignments(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Build deterministic fingerprint and spatial fold labels."""

    fingerprints = np.asarray([row["fingerprint"] for row in records], dtype=str)
    if len(np.unique(fingerprints)) < FOLD_COUNT:
        raise ValueError("fewer than five typewell fingerprint groups")
    fingerprint_folds = np.full(len(records), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=FOLD_COUNT)
    for fold, (_, validation) in enumerate(
        splitter.split(np.arange(len(records)), groups=fingerprints)
    ):
        fingerprint_folds[validation] = fold
    if np.any(fingerprint_folds < 0):
        raise AssertionError("fingerprint folds are incomplete")

    coordinates = np.asarray(
        [[row["centroid_x"], row["centroid_y"]] for row in records],
        dtype=float,
    )
    scaled = StandardScaler().fit_transform(coordinates)
    raw_spatial = KMeans(
        n_clusters=FOLD_COUNT,
        random_state=RANDOM_SEED,
        n_init=50,
    ).fit_predict(scaled)
    spatial_folds = _relabel_spatial_clusters(raw_spatial, coordinates)
    return {
        "typewell_fingerprint": fingerprint_folds,
        "spatial_cluster": spatial_folds,
    }


def _pooled_rmse(per_well_rmse: np.ndarray, rows: np.ndarray) -> float:
    return float(np.sqrt(np.sum(rows * per_well_rmse**2) / np.sum(rows)))


def _metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rmse = np.asarray([row["rmse"][method] for row in rows], dtype=float)
    weights = np.asarray([row["scored_rows"] for row in rows], dtype=float)
    worst = int(np.argmax(rmse))
    return {
        "pooled_rmse": _pooled_rmse(rmse, weights),
        "well_macro_rmse": float(np.mean(rmse)),
        "well_median_rmse": float(np.median(rmse)),
        "well_p90_rmse": float(np.quantile(rmse, 0.90)),
        "well_p95_rmse": float(np.quantile(rmse, 0.95)),
        "well_p99_rmse": float(np.quantile(rmse, 0.99)),
        "worst_well_rmse": float(rmse[worst]),
        "worst_well_id": str(rows[worst]["well_id"]),
        "scored_wells": int(len(rows)),
        "scored_rows": int(np.sum(weights)),
    }


def _metric_delta(model: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(model[key] - baseline[key])
        for key in (
            "pooled_rmse",
            "well_macro_rmse",
            "well_median_rmse",
            "well_p90_rmse",
            "well_p95_rmse",
            "well_p99_rmse",
            "worst_well_rmse",
        )
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    baseline_method: str,
    *,
    seed: int,
) -> dict[str, Any]:
    baseline = np.asarray(
        [row["rmse"][baseline_method] for row in rows],
        dtype=float,
    )
    model = np.asarray([row["rmse"]["residual_gbdt"] for row in rows], dtype=float)
    weights = np.asarray([row["scored_rows"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch_size = 512
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        count = min(batch_size, BOOTSTRAP_REPLICATES - start)
        sampled = rng.integers(0, len(rows), size=(count, len(rows)))
        sample_weights = weights[sampled]
        denominator = np.sum(sample_weights, axis=1)
        baseline_pooled = np.sqrt(
            np.sum(sample_weights * baseline[sampled] ** 2, axis=1) / denominator
        )
        model_pooled = np.sqrt(np.sum(sample_weights * model[sampled] ** 2, axis=1) / denominator)
        deltas[start : start + count] = model_pooled - baseline_pooled
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "mean": float(np.mean(deltas)),
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def _evaluate_protocol(
    protocol: str,
    records: list[dict[str, Any]],
    folds: np.ndarray,
    train_dir: Path,
    *,
    feature_indices: np.ndarray | None = None,
    feature_names: tuple[str, ...] = RESIDUAL_FEATURE_NAMES,
    required_nonpositive_folds: int = REQUIRED_NONPOSITIVE_FOLDS,
    bootstrap_seed: int | None = None,
    model_parameters: dict[str, Any] | None = None,
    residual_scale: float = 1.0,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    fingerprints = np.asarray([row["fingerprint"] for row in records], dtype=str)
    started = time.perf_counter()
    fold_training: list[dict[str, Any]] = []
    unique_folds = np.unique(folds)
    if not np.array_equal(unique_folds, np.arange(len(unique_folds))):
        raise ValueError(f"{protocol} folds must be contiguous and zero-based")
    if feature_indices is None:
        feature_indices = np.arange(len(RESIDUAL_FEATURE_NAMES), dtype=np.int64)
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    if len(feature_indices) != len(feature_names):
        raise ValueError("feature indices and feature names have different lengths")
    if model_parameters is None:
        model_parameters = MODEL_PARAMETERS
    if not 0.0 < residual_scale <= 1.0:
        raise ValueError("residual_scale must be in (0, 1]")

    for fold in unique_folds:
        validation_indices = np.flatnonzero(folds == fold)
        training_mask = folds != fold
        validation_fingerprints = set(fingerprints[validation_indices])
        # This is redundant for fingerprint folds and intentionally stricter
        # for spatial folds.
        training_mask &= np.asarray(
            [fingerprint not in validation_fingerprints for fingerprint in fingerprints],
            dtype=bool,
        )
        training_indices = np.flatnonzero(training_mask)
        if len(training_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(f"{protocol} fold {fold} is empty")

        train_x = np.concatenate(
            [records[index]["sample_features"][:, feature_indices] for index in training_indices],
            axis=0,
        )
        train_y = np.concatenate(
            [records[index]["sample_labels"] for index in training_indices],
            axis=0,
        )
        train_weight = np.concatenate(
            [
                np.full(
                    len(records[index]["sample_labels"]),
                    records[index]["sample_weight"],
                    dtype=np.float32,
                )
                for index in training_indices
            ]
        )
        training_pool = Pool(
            train_x,
            label=train_y,
            weight=train_weight,
            feature_names=list(feature_names),
        )
        model = CatBoostRegressor(**model_parameters)
        model.fit(training_pool)
        fold_training.append(
            {
                "fold": fold,
                "training_wells": int(len(training_indices)),
                "validation_wells": int(len(validation_indices)),
                "training_rows": int(len(train_y)),
                "validation_fingerprints": int(len(validation_fingerprints)),
                "training_fingerprint_overlap": 0,
            }
        )

        for position, index in enumerate(validation_indices, start=1):
            path = Path(records[index]["path"])
            horizontal, typewell, truth = _read_inputs(path, train_dir)
            block = build_residual_features(horizontal, typewell)
            truth_hidden = truth[block.evaluation_mask]
            raw_residual = np.asarray(
                model.predict(block.features[:, feature_indices]),
                dtype=float,
            )
            raw_residual = np.clip(
                raw_residual,
                -PREDICTION_RESIDUAL_CLIP,
                PREDICTION_RESIDUAL_CLIP,
            )
            residual = residual_scale * raw_residual
            prediction = block.anchor_tvt + residual
            output_rows.append(
                {
                    "well_id": records[index]["well_id"],
                    "fold": fold,
                    "scored_rows": int(len(truth_hidden)),
                    "rmse": {
                        "hold_last": _rmse(truth_hidden, block.hold_last),
                        "u_boundary": _rmse(truth_hidden, block.u_boundary),
                        "slope_decay": _rmse(truth_hidden, block.slope_decay),
                        "residual_gbdt": _rmse(truth_hidden, prediction),
                    },
                    "prediction": {
                        "residual_scale": residual_scale,
                        "mean_residual": float(np.mean(residual)),
                        "p95_abs_residual": float(np.quantile(np.abs(residual), 0.95)),
                        "max_abs_residual": float(np.max(np.abs(residual))),
                    },
                }
            )
            if position % 50 == 0 or position == len(validation_indices):
                print(
                    f"[{protocol} fold {fold} {position:>3}/{len(validation_indices)}] "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    output_rows.sort(key=lambda row: str(row["well_id"]))
    metrics = {
        method: _metrics(output_rows, method) for method in (*BASELINE_METHODS, "residual_gbdt")
    }
    baseline_method = min(
        BASELINE_METHODS,
        key=lambda method: (metrics[method]["pooled_rmse"], method),
    )
    delta = _metric_delta(metrics["residual_gbdt"], metrics[baseline_method])
    fold_results: list[dict[str, Any]] = []
    for fold in unique_folds:
        subset = [row for row in output_rows if row["fold"] == fold]
        baseline_fold = _metrics(subset, baseline_method)
        model_fold = _metrics(subset, "residual_gbdt")
        fold_results.append(
            {
                "fold": fold,
                "wells": len(subset),
                "baseline_pooled_rmse": baseline_fold["pooled_rmse"],
                "model_pooled_rmse": model_fold["pooled_rmse"],
                "pooled_delta": float(model_fold["pooled_rmse"] - baseline_fold["pooled_rmse"]),
            }
        )
    fold_deltas = np.asarray(
        [row["pooled_delta"] for row in fold_results],
        dtype=float,
    )
    bootstrap = _paired_bootstrap(
        output_rows,
        baseline_method,
        seed=(
            bootstrap_seed
            if bootstrap_seed is not None
            else RANDOM_SEED + (0 if protocol == "typewell_fingerprint" else 1)
        ),
    )
    fold_count = len(unique_folds)
    checks = {
        "pooled_delta_at_most_minus_0_10": bool(delta["pooled_rmse"] <= POOLED_DELTA_LIMIT),
        "well_macro_delta_negative": bool(delta["well_macro_rmse"] < 0.0),
        "bootstrap_ci95_upper_negative": bool(bootstrap["ci95_upper"] < 0.0),
        f"at_least_{required_nonpositive_folds}_of_{fold_count}_folds_nonpositive": bool(
            np.count_nonzero(fold_deltas <= 0.0) >= required_nonpositive_folds
        ),
        "no_fold_worse_than_plus_0_20": bool(float(np.max(fold_deltas)) <= MAXIMUM_FOLD_DELTA),
        "p95_not_worse_than_plus_0_25": bool(delta["well_p95_rmse"] <= P95_DELTA_LIMIT),
        "worst_not_worse_than_plus_1_00": bool(delta["worst_well_rmse"] <= WORST_DELTA_LIMIT),
    }
    return {
        "protocol": protocol,
        "baseline_selected_for_gate": baseline_method,
        "metrics": metrics,
        "delta_model_minus_baseline": delta,
        "paired_well_bootstrap": bootstrap,
        "fold_results": fold_results,
        "fold_training": fold_training,
        "gate": {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "nonpositive_folds": int(np.count_nonzero(fold_deltas <= 0.0)),
            "maximum_fold_delta": float(np.max(fold_deltas)),
        },
        "per_well": output_rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if args.max_wells < 0:
        raise ValueError("--max-wells cannot be negative")
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if args.max_wells:
        paths = paths[: args.max_wells]
    elif len(paths) != 773:
        raise ValueError(f"frozen full protocol expects 773 wells, found {len(paths)}")
    if len(paths) < 25:
        raise ValueError("at least 25 wells are required for five-fold evaluation")

    records = _scan_wells(paths, train_dir)
    assignments = build_fold_assignments(records)
    protocol_results = {
        name: _evaluate_protocol(name, records, folds, train_dir)
        for name, folds in assignments.items()
    }
    all_pass = all(result["gate"]["decision"] == "PASS" for result in protocol_results.values())
    report: dict[str, Any] = {
        "evaluation_contract": {
            "horizontal_columns": list(HORIZONTAL_INPUT_COLUMNS),
            "typewell_columns": list(TYPEWELL_INPUT_COLUMNS),
            "hidden_tvt_role": (
                "isolated after feature construction and used only as an offline label"
            ),
            "well_id_feature": False,
            "same_id_target_lookup": False,
            "formation_feature": False,
            "leaderboard_tuning": False,
            "validation_protocols": [
                "five typewell-fingerprint group folds",
                ("five spatial-cluster folds with validation fingerprints removed from training"),
            ],
        },
        "frozen_configuration": {
            "feature_config": asdict(ResidualFeatureConfig()),
            "feature_names": list(RESIDUAL_FEATURE_NAMES),
            "maximum_training_rows_per_well": MAX_TRAIN_ROWS_PER_WELL,
            "model_family": "CatBoostRegressor",
            "model_parameters": MODEL_PARAMETERS,
            "prediction_residual_clip": PREDICTION_RESIDUAL_CLIP,
            "fold_count": FOLD_COUNT,
            "random_seed": RANDOM_SEED,
            "gate": {
                "pooled_delta_limit": POOLED_DELTA_LIMIT,
                "required_nonpositive_folds": REQUIRED_NONPOSITIVE_FOLDS,
                "maximum_fold_delta": MAXIMUM_FOLD_DELTA,
                "p95_delta_limit": P95_DELTA_LIMIT,
                "worst_delta_limit": WORST_DELTA_LIMIT,
                "all_protocols_must_pass": True,
            },
        },
        "selection": {
            "wells": len(paths),
            "full_frozen_run": args.max_wells == 0,
        },
        "protocol_results": protocol_results,
        "decision": {
            "decision": "GO" if all_pass else "NO-GO",
            "reason": (
                "the one frozen model passed both independent OOF protocols"
                if all_pass
                else "the one frozen model failed at least one OOF protocol"
            ),
            "threshold_tuning_after_result": False,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output = args.json_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["artifact_sha256_before_hash_field"] = _sha256(output)
    output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"decision={report['decision']['decision']} runtime={report['runtime_seconds']:.1f}s",
        flush=True,
    )
    for name, result in protocol_results.items():
        delta = result["delta_model_minus_baseline"]["pooled_rmse"]
        print(
            f"{name}: gate={result['gate']['decision']} "
            f"baseline={result['baseline_selected_for_gate']} "
            f"pooled_delta={delta:+.6f}",
            flush=True,
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
