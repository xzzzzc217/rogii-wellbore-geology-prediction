"""Evaluate residual-v3 as an increment to the public model's grouped OOF track.

This is the decisive offline gate for packaging the residual overlay.  It
reconstructs the public Ridge/PF grouped-OOF trajectory from the published
feature cache and five published Trainer artifacts.  The frozen residual-v3
correction is then evaluated as an additive overlay under two independently
grouped protocols:

* exact typewell-fingerprint folds;
* the previously unscored-at-freeze seven-cluster spatial folds.

The script writes metrics only.  It never creates a competition submission.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from residual_gbdt_evaluate import (  # noqa: E402
    MODEL_PARAMETERS,
    PREDICTION_RESIDUAL_CLIP,
    _json_safe,
    _read_inputs,
    _scan_wells,
)
from residual_gbdt_invariant_evaluate import (  # noqa: E402
    INVARIANT_FEATURE_INDICES,
    INVARIANT_FEATURE_NAMES,
)
from residual_gbdt_shrink_evaluate import (  # noqa: E402
    RESIDUAL_SCALE,
    build_shrink_fold_assignments,
)

from rogii.residual_gbdt import build_residual_features  # noqa: E402

EXTERNAL_ROOT = PROJECT_ROOT / "artifacts" / "external" / "wellbore-geology-prediction-artifacts"
TRAINER_PATTERNS = (
    "lgbmregressor_trainer_20260526182612.pkl",
    "lgbmregressor_trainer_20260526190415.pkl",
    "lgbmregressor_trainer_20260526192806.pkl",
    "catboostregressor_trainer_20260526193740.pkl",
    "catboostregressor_trainer_20260526194838.pkl",
)
RIDGE_PARAMETERS = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True,
}
SELECTED_PROTOCOLS = (
    "typewell_fingerprint",
    "spatial_cluster_7_seed_20260727",
)

# Frozen before the competitive-overlay result is observed.
POOLED_DELTA_LIMIT = -0.02
MAXIMUM_FOLD_DELTA = 0.05
P95_DELTA_LIMIT = 0.10
WORST_DELTA_LIMIT = 0.50


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=EXTERNAL_ROOT,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "competitive_oof_overlay.json",
    )
    return parser


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _pooled(per_well: list[dict[str, Any]], key: str) -> float:
    rows = np.asarray([row["rows"] for row in per_well], dtype=float)
    rmse = np.asarray([row[key] for row in per_well], dtype=float)
    return float(np.sqrt(np.sum(rows * np.square(rmse)) / np.sum(rows)))


def _distribution(per_well: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in per_well], dtype=float)
    return {
        "macro": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "worst": float(np.max(values)),
    }


def _load_public_track(
    external_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    train_path = external_root / "train_oof_columns.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} is missing; run scripts/project_public_train_columns.py first"
        )
    frame = pd.read_parquet(
        train_path,
        columns=["well", "id", "last_known_tvt", "pf_ancc", "md_since", "target"],
        dtype_backend="pyarrow",
    )
    expected_rows = 3_783_989
    if len(frame) != expected_rows:
        raise ValueError(f"public feature cache has {len(frame)} rows, expected {expected_rows}")
    if frame["id"].duplicated().any():
        raise ValueError("public feature cache contains duplicate IDs")

    model_oof = np.empty((len(frame), len(TRAINER_PATTERNS)), dtype=np.float32)
    trainer_scores: dict[str, float] = {}
    for column, trainer_filename in enumerate(TRAINER_PATTERNS):
        path = external_root / Path(trainer_filename).with_suffix(".oof.npy")
        metadata_path = path.with_suffix(".json")
        if not path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"missing extracted OOF artifact for {trainer_filename}; "
                "run scripts/extract_trainer_oof.py first"
            )
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(prediction) != len(frame):
            raise ValueError(f"{trainer_filename}: OOF length mismatch")
        model_oof[:, column] = prediction
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trainer_scores[trainer_filename] = float(metadata["overall_score"])
        del prediction

    target = frame["target"].to_numpy(dtype=np.float64)
    groups, unique_wells = pd.factorize(frame["well"], sort=True)
    ridge_oof = np.empty(len(frame), dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for training, validation in splitter.split(model_oof, target, groups=groups):
        model = Ridge(**RIDGE_PARAMETERS)
        model.fit(model_oof[training], target[training])
        ridge_oof[validation] = model.predict(model_oof[validation])

    anchor = frame["last_known_tvt"].to_numpy(dtype=np.float64)
    pf_delta = frame["pf_ancc"].to_numpy(dtype=np.float64) - anchor
    ramp = 1.0 - np.exp(-np.maximum(frame["md_since"].to_numpy(dtype=np.float64), 0.0) / 85.0)
    track_delta = (0.91 * ridge_oof + 0.09 * pf_delta) * ramp
    frame["truth"] = anchor + target
    frame["competitive_oof"] = anchor + track_delta
    finite_values = frame[["truth", "competitive_oof"]].to_numpy(
        dtype=np.float64,
    )
    if not np.isfinite(finite_values).all():
        raise ValueError("reconstructed competitive OOF contains non-finite values")
    diagnostics = {
        "rows": int(len(frame)),
        "wells": int(len(unique_wells)),
        "trainer_scores": trainer_scores,
        "ridge_parameters": RIDGE_PARAMETERS,
        "ridge_pf_weights": {"ridge": 0.91, "pf": 0.09},
        "ramp_tau_md": 85.0,
        "pooled_rmse": _rmse(
            frame["truth"].to_numpy(dtype=float),
            frame["competitive_oof"].to_numpy(dtype=float),
        ),
    }
    base_by_well: dict[str, dict[str, np.ndarray]] = {}
    for well, group in frame.groupby("well", sort=False, observed=True):
        row_index = np.asarray(
            [int(str(value).rsplit("_", 1)[1]) for value in group["id"]],
            dtype=np.int32,
        )
        if np.any(np.diff(row_index) <= 0):
            raise ValueError(f"{well}: public OOF rows are not strictly increasing")
        base_by_well[str(well)] = {
            "row_index": row_index,
            "prediction": group["competitive_oof"].to_numpy(dtype=np.float64),
        }
    if len(base_by_well) != len(unique_wells):
        raise AssertionError("public OOF well mapping is incomplete")
    return base_by_well, diagnostics


def _evaluate_protocol(
    name: str,
    records: list[dict[str, Any]],
    folds: np.ndarray,
    train_dir: Path,
    base_by_well: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    fingerprints = np.asarray([row["fingerprint"] for row in records], dtype=str)
    unique_folds = np.unique(folds)
    output: list[dict[str, Any]] = []
    fold_training: list[dict[str, int]] = []
    started = time.perf_counter()

    for fold in unique_folds:
        validation_indices = np.flatnonzero(folds == fold)
        validation_fingerprints = set(fingerprints[validation_indices])
        training_mask = folds != fold
        training_mask &= np.asarray(
            [fingerprint not in validation_fingerprints for fingerprint in fingerprints],
            dtype=bool,
        )
        training_indices = np.flatnonzero(training_mask)
        train_x = np.concatenate(
            [
                records[index]["sample_features"][:, INVARIANT_FEATURE_INDICES]
                for index in training_indices
            ],
            axis=0,
        )
        train_y = np.concatenate([records[index]["sample_labels"] for index in training_indices])
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
        model = CatBoostRegressor(**MODEL_PARAMETERS)
        model.fit(
            Pool(
                train_x,
                label=train_y,
                weight=train_weight,
                feature_names=list(INVARIANT_FEATURE_NAMES),
            )
        )
        fold_training.append(
            {
                "fold": int(fold),
                "training_wells": int(len(training_indices)),
                "validation_wells": int(len(validation_indices)),
                "training_rows": int(len(train_y)),
            }
        )

        for index in validation_indices:
            record = records[index]
            path = Path(record["path"])
            horizontal, typewell, truth = _read_inputs(path, train_dir)
            block = build_residual_features(horizontal, typewell)
            raw = np.asarray(
                model.predict(block.features[:, INVARIANT_FEATURE_INDICES]),
                dtype=float,
            )
            correction = RESIDUAL_SCALE * np.clip(
                raw,
                -PREDICTION_RESIDUAL_CLIP,
                PREDICTION_RESIDUAL_CLIP,
            )
            hidden_rows = np.flatnonzero(block.evaluation_mask)
            base_rows = base_by_well.get(str(record["well_id"]))
            if base_rows is None:
                raise ValueError(f"{record['well_id']}: missing competitive OOF well")
            if not np.array_equal(
                base_rows["row_index"],
                hidden_rows.astype(np.int32),
            ):
                raise ValueError(f"{record['well_id']}: competitive OOF row mismatch")
            target = truth[block.evaluation_mask]
            base = base_rows["prediction"]
            overlay = base + correction
            output.append(
                {
                    "well": str(record["well_id"]),
                    "fold": int(fold),
                    "rows": int(len(target)),
                    "base_rmse": _rmse(target, base),
                    "overlay_rmse": _rmse(target, overlay),
                    "mean_correction": float(np.mean(correction)),
                    "p95_abs_correction": float(np.quantile(np.abs(correction), 0.95)),
                }
            )
        print(
            f"[{name} fold {int(fold)}] validation_wells={len(validation_indices)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    base_pooled = _pooled(output, "base_rmse")
    overlay_pooled = _pooled(output, "overlay_rmse")
    base_distribution = _distribution(output, "base_rmse")
    overlay_distribution = _distribution(output, "overlay_rmse")
    fold_results: list[dict[str, Any]] = []
    for fold in unique_folds:
        subset = [row for row in output if row["fold"] == int(fold)]
        base_fold = _pooled(subset, "base_rmse")
        overlay_fold = _pooled(subset, "overlay_rmse")
        fold_results.append(
            {
                "fold": int(fold),
                "wells": int(len(subset)),
                "base_pooled_rmse": base_fold,
                "overlay_pooled_rmse": overlay_fold,
                "delta": float(overlay_fold - base_fold),
            }
        )
    fold_deltas = np.asarray([row["delta"] for row in fold_results], dtype=float)
    checks = {
        "pooled_delta_at_most_minus_0_02": bool(overlay_pooled - base_pooled <= POOLED_DELTA_LIMIT),
        "at_least_80_percent_folds_nonpositive": bool(
            np.count_nonzero(fold_deltas <= 0.0) >= int(np.ceil(0.8 * len(fold_deltas)))
        ),
        "no_fold_worse_than_plus_0_05": bool(np.max(fold_deltas) <= MAXIMUM_FOLD_DELTA),
        "p95_not_worse_than_plus_0_10": bool(
            overlay_distribution["p95"] - base_distribution["p95"] <= P95_DELTA_LIMIT
        ),
        "worst_not_worse_than_plus_0_50": bool(
            overlay_distribution["worst"] - base_distribution["worst"] <= WORST_DELTA_LIMIT
        ),
    }
    output.sort(key=lambda row: row["well"])
    return {
        "protocol": name,
        "base_pooled_rmse": base_pooled,
        "overlay_pooled_rmse": overlay_pooled,
        "pooled_delta": float(overlay_pooled - base_pooled),
        "base_well_distribution": base_distribution,
        "overlay_well_distribution": overlay_distribution,
        "fold_results": fold_results,
        "fold_training": fold_training,
        "per_well": output,
        "gate": {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if len(paths) != 773:
        raise ValueError(f"expected 773 train wells, found {len(paths)}")

    base_by_well, public_diagnostics = _load_public_track(
        args.external_root.expanduser().resolve()
    )
    gc.collect()

    records = _scan_wells(paths, train_dir)
    assignments = build_shrink_fold_assignments(records)
    results = {
        name: _evaluate_protocol(
            name,
            records,
            assignments[name],
            train_dir,
            base_by_well,
        )
        for name in SELECTED_PROTOCOLS
    }
    all_pass = all(result["gate"]["decision"] == "PASS" for result in results.values())
    report = {
        "contract": {
            "writes_submission": False,
            "competitive_base": "published five-model grouped OOF Ridge/PF track",
            "overlay": "frozen spatial-invariant residual v3",
            "selected_protocols": list(SELECTED_PROTOCOLS),
            "residual_scale": RESIDUAL_SCALE,
            "thresholds_frozen_before_result": {
                "pooled_delta_limit": POOLED_DELTA_LIMIT,
                "maximum_fold_delta": MAXIMUM_FOLD_DELTA,
                "p95_delta_limit": P95_DELTA_LIMIT,
                "worst_delta_limit": WORST_DELTA_LIMIT,
            },
        },
        "public_track": public_diagnostics,
        "protocol_results": results,
        "decision": {
            "decision": "GO" if all_pass else "NO-GO",
            "all_protocols_pass": all_pass,
            "kaggle_submission_authorized": False,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output = args.json_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"decision={report['decision']['decision']} "
        f"competitive_base={public_diagnostics['pooled_rmse']:.6f} "
        f"runtime={report['runtime_seconds']:.1f}s"
    )
    for name, result in results.items():
        print(
            f"{name}: {result['gate']['decision']} "
            f"{result['base_pooled_rmse']:.6f} -> "
            f"{result['overlay_pooled_rmse']:.6f} "
            f"({result['pooled_delta']:+.6f})"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
