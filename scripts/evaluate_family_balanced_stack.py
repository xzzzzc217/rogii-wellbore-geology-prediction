"""Evaluate a preregistered family-balanced variant of the public OOF stack.

The published stack exposes three LightGBM and two CatBoost grouped-OOF
vectors as five separate Ridge inputs. This audit first averages each model
family and gives the positive Ridge one feature per family. Everything after
that collapse is held identical to the published Ridge/PF trajectory.

This is a secondary cross-fitted stacking audit. The base models are not
retrained inside the descriptive typewell and spatial partitions, so even a
PASS cannot directly authorize a competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rogii.family_stack import (  # noqa: E402
    DEFAULT_MODEL_FAMILIES,
    collapse_model_families,
    compose_ridge_pf_track,
)
from rogii.residual_gbdt import typewell_fingerprint  # noqa: E402
from rogii.trajectory import TYPEWELL_INPUT_COLUMNS  # noqa: E402
from rogii.u_continuity_gate import paired_well_bootstrap, pooled_rmse  # noqa: E402

EXTERNAL_ROOT = PROJECT_ROOT / "artifacts" / "external" / (
    "wellbore-geology-prediction-artifacts"
)
TRAINER_FILENAMES = (
    "lgbmregressor_trainer_20260526182612.pkl",
    "lgbmregressor_trainer_20260526190415.pkl",
    "lgbmregressor_trainer_20260526192806.pkl",
    "catboostregressor_trainer_20260526193740.pkl",
    "catboostregressor_trainer_20260526194838.pkl",
)
FAMILY_NAMES = ("lightgbm_mean_3", "catboost_mean_2")
RIDGE_PARAMETERS: dict[str, Any] = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True,
}
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_BASE_POOLED_RMSE = 10.370043517055203
META_FOLDS = 5
SPATIAL_FOLDS = 7
SPATIAL_SEED = 20_260_727
BOOTSTRAP_SEED = 20_260_730

# Frozen before the family-balanced target-scored result is observed.
POOLED_DELTA_LIMIT = -0.020
REQUIRED_TYPEWELL_NONPOSITIVE = 4
REQUIRED_SPATIAL_NONPOSITIVE = 6
MAXIMUM_PARTITION_DELTA = 0.050
P95_DELTA_LIMIT = 0.100
WORST_DELTA_LIMIT = 0.500


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
        default=PROJECT_ROOT / "artifacts" / "family_balanced_stack_oof.json",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_frame(external_root: Path) -> pd.DataFrame:
    path = external_root / "train_oof_columns.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing projected public train cache: {path}")
    frame = pd.read_parquet(
        path,
        columns=["well", "id", "last_known_tvt", "pf_ancc", "md_since", "target"],
        dtype_backend="pyarrow",
    )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"projected cache has {len(frame)} rows, expected {EXPECTED_ROWS}")
    if frame["id"].duplicated().any():
        raise ValueError("projected cache contains duplicate IDs")
    if frame["well"].nunique() != EXPECTED_WELLS:
        raise ValueError(
            f"projected cache has {frame['well'].nunique()} wells, expected {EXPECTED_WELLS}"
        )
    numeric = frame[["last_known_tvt", "pf_ancc", "md_since", "target"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("projected cache contains non-finite numeric values")
    return frame


def _load_model_oof(external_root: Path, rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = np.empty((rows, len(TRAINER_FILENAMES)), dtype=np.float32)
    metadata: dict[str, Any] = {}
    for column, filename in enumerate(TRAINER_FILENAMES):
        prediction_path = external_root / Path(filename).with_suffix(".oof.npy")
        metadata_path = prediction_path.with_suffix(".json")
        if not prediction_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"missing extracted trainer OOF: {filename}")
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != (rows,) or not np.isfinite(prediction).all():
            raise ValueError(f"{filename}: invalid extracted OOF vector")
        predictions[:, column] = prediction
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[filename] = {
            "overall_score": float(source["overall_score"]),
            "fold_scores": [float(value) for value in source["fold_scores"]],
            "oof_sha256": _sha256(prediction_path),
        }
    return predictions, metadata


def _relabel_spatial_clusters(labels: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    ordering = sorted(
        range(SPATIAL_FOLDS),
        key=lambda label: (
            float(np.median(coordinates[labels == label, 0])),
            float(np.median(coordinates[labels == label, 1])),
        ),
    )
    mapping = {old: new for new, old in enumerate(ordering)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int16)


def _build_robustness_partitions(
    wells: np.ndarray,
    train_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    fingerprints: list[str] = []
    coordinates: list[tuple[float, float]] = []
    started = time.perf_counter()
    for position, well in enumerate(wells, start=1):
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        typewell_path = train_dir / f"{well}__typewell.csv"
        if not horizontal_path.exists() or not typewell_path.exists():
            raise FileNotFoundError(f"{well}: official train files are missing")
        horizontal = pd.read_csv(horizontal_path, usecols=["X", "Y"])
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{well}: non-finite coordinates")
        typewell = pd.read_csv(typewell_path, usecols=list(TYPEWELL_INPUT_COLUMNS)).loc[
            :, TYPEWELL_INPUT_COLUMNS
        ]
        fingerprints.append(typewell_fingerprint(typewell))
        coordinates.append((float(np.median(x)), float(np.median(y))))
        if position % 100 == 0 or position == len(wells):
            print(
                f"[metadata {position:>3}/{len(wells)}] "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    fingerprint_array = np.asarray(fingerprints, dtype=str)
    typewell_folds = np.full(len(wells), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=META_FOLDS)
    for fold, (_, validation) in enumerate(
        splitter.split(np.arange(len(wells)), groups=fingerprint_array)
    ):
        typewell_folds[validation] = fold
    if np.any(typewell_folds < 0):
        raise AssertionError("typewell partition is incomplete")

    coordinate_array = np.asarray(coordinates, dtype=float)
    scaled = StandardScaler().fit_transform(coordinate_array)
    raw_spatial = KMeans(
        n_clusters=SPATIAL_FOLDS,
        random_state=SPATIAL_SEED,
        n_init=50,
    ).fit_predict(scaled)
    spatial_folds = _relabel_spatial_clusters(raw_spatial, coordinate_array)
    partitions = {
        "typewell_fingerprint_5": typewell_folds,
        "spatial_cluster_7_seed_20260727": spatial_folds,
    }
    diagnostics = {
        "unique_typewell_fingerprints": int(len(np.unique(fingerprint_array))),
        "coordinate_columns": ["X", "Y"],
        "hidden_tvt_used": False,
    }
    return partitions, fingerprint_array, diagnostics


def _crossfit_meta_models(
    model_oof: np.ndarray,
    family_oof: np.ndarray,
    target: np.ndarray,
    row_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    base_delta = np.full(len(target), np.nan, dtype=np.float64)
    candidate_delta = np.full(len(target), np.nan, dtype=np.float64)
    well_folds = np.full(int(np.max(row_groups)) + 1, -1, dtype=np.int16)
    fold_models: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=META_FOLDS)
    started = time.perf_counter()
    for fold, (training, validation) in enumerate(
        splitter.split(model_oof, target, groups=row_groups)
    ):
        base_model = Ridge(**RIDGE_PARAMETERS)
        candidate_model = Ridge(**RIDGE_PARAMETERS)
        base_model.fit(model_oof[training], target[training])
        candidate_model.fit(family_oof[training], target[training])
        base_delta[validation] = base_model.predict(model_oof[validation])
        candidate_delta[validation] = candidate_model.predict(family_oof[validation])
        validation_wells = np.unique(row_groups[validation])
        well_folds[validation_wells] = fold
        fold_models.append(
            {
                "fold": fold,
                "training_rows": int(len(training)),
                "validation_rows": int(len(validation)),
                "training_wells": int(len(np.unique(row_groups[training]))),
                "validation_wells": int(len(validation_wells)),
                "base_coefficients": base_model.coef_.tolist(),
                "base_intercept": float(base_model.intercept_),
                "candidate_coefficients": candidate_model.coef_.tolist(),
                "candidate_intercept": float(candidate_model.intercept_),
            }
        )
        print(
            f"[meta fold {fold}] train={len(training)} validation={len(validation)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if (
        not np.isfinite(base_delta).all()
        or not np.isfinite(candidate_delta).all()
        or np.any(well_folds < 0)
    ):
        raise AssertionError("meta cross-fitting is incomplete")
    return base_delta, candidate_delta, well_folds, fold_models


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "macro": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "worst": float(np.max(values)),
    }


def _partition_results(
    folds: np.ndarray,
    base_rmse: np.ndarray,
    candidate_rmse: np.ndarray,
    rows: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fold in np.unique(folds):
        selected = folds == fold
        base = pooled_rmse(base_rmse[selected], rows[selected])
        candidate = pooled_rmse(candidate_rmse[selected], rows[selected])
        results.append(
            {
                "fold": int(fold),
                "wells": int(np.count_nonzero(selected)),
                "rows": int(np.sum(rows[selected])),
                "base_pooled_rmse": base,
                "candidate_pooled_rmse": candidate,
                "delta": float(candidate - base),
            }
        )
    return results


def _evaluate(
    truth: np.ndarray,
    base_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    row_groups: np.ndarray,
    wells: np.ndarray,
    meta_folds: np.ndarray,
    partitions: dict[str, np.ndarray],
) -> dict[str, Any]:
    well_count = len(wells)
    rows = np.bincount(row_groups, minlength=well_count).astype(np.int64)
    base_squared = np.square(base_prediction - truth)
    candidate_squared = np.square(candidate_prediction - truth)
    base_rmse = np.sqrt(
        np.bincount(row_groups, weights=base_squared, minlength=well_count) / rows
    )
    candidate_rmse = np.sqrt(
        np.bincount(row_groups, weights=candidate_squared, minlength=well_count) / rows
    )
    base_pooled = pooled_rmse(base_rmse, rows)
    candidate_pooled = pooled_rmse(candidate_rmse, rows)
    base_distribution = _distribution(base_rmse)
    candidate_distribution = _distribution(candidate_rmse)
    bootstrap = paired_well_bootstrap(
        base_rmse,
        candidate_rmse,
        rows,
        seed=BOOTSTRAP_SEED,
    )
    all_partitions = {"published_group_kfold_5": meta_folds, **partitions}
    partition_results = {
        name: _partition_results(folds, base_rmse, candidate_rmse, rows)
        for name, folds in all_partitions.items()
    }
    typewell_deltas = np.asarray(
        [row["delta"] for row in partition_results["typewell_fingerprint_5"]]
    )
    spatial_deltas = np.asarray(
        [row["delta"] for row in partition_results["spatial_cluster_7_seed_20260727"]]
    )
    checks = {
        "pooled_delta_at_most_minus_0_020": bool(
            candidate_pooled - base_pooled <= POOLED_DELTA_LIMIT
        ),
        "well_macro_delta_negative": bool(
            candidate_distribution["macro"] - base_distribution["macro"] < 0.0
        ),
        "bootstrap_ci95_upper_negative": bool(bootstrap["ci95_upper"] < 0.0),
        "at_least_4_of_5_typewell_partitions_nonpositive": bool(
            np.count_nonzero(typewell_deltas <= 0.0) >= REQUIRED_TYPEWELL_NONPOSITIVE
        ),
        "at_least_6_of_7_spatial_partitions_nonpositive": bool(
            np.count_nonzero(spatial_deltas <= 0.0) >= REQUIRED_SPATIAL_NONPOSITIVE
        ),
        "no_partition_worse_than_plus_0_050": bool(
            max(float(np.max(typewell_deltas)), float(np.max(spatial_deltas)))
            <= MAXIMUM_PARTITION_DELTA
        ),
        "p95_not_worse_than_plus_0_100": bool(
            candidate_distribution["p95"] - base_distribution["p95"] <= P95_DELTA_LIMIT
        ),
        "worst_not_worse_than_plus_0_500": bool(
            candidate_distribution["worst"] - base_distribution["worst"]
            <= WORST_DELTA_LIMIT
        ),
    }
    per_well = [
        {
            "well": str(wells[index]),
            "rows": int(rows[index]),
            "base_rmse": float(base_rmse[index]),
            "candidate_rmse": float(candidate_rmse[index]),
            "delta": float(candidate_rmse[index] - base_rmse[index]),
            "meta_fold": int(meta_folds[index]),
            "typewell_fold": int(partitions["typewell_fingerprint_5"][index]),
            "spatial_fold": int(
                partitions["spatial_cluster_7_seed_20260727"][index]
            ),
        }
        for index in range(well_count)
    ]
    return {
        "base_pooled_rmse": base_pooled,
        "candidate_pooled_rmse": candidate_pooled,
        "pooled_delta": float(candidate_pooled - base_pooled),
        "base_well_distribution": base_distribution,
        "candidate_well_distribution": candidate_distribution,
        "well_distribution_delta": {
            key: float(candidate_distribution[key] - base_distribution[key])
            for key in base_distribution
        },
        "paired_well_bootstrap": bootstrap,
        "partition_results": partition_results,
        "per_well": per_well,
        "gate": {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "typewell_nonpositive_partitions": int(
                np.count_nonzero(typewell_deltas <= 0.0)
            ),
            "spatial_nonpositive_partitions": int(
                np.count_nonzero(spatial_deltas <= 0.0)
            ),
            "maximum_partition_delta": max(
                float(np.max(typewell_deltas)),
                float(np.max(spatial_deltas)),
            ),
        },
    }


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    external_root = args.external_root.expanduser().resolve()
    train_dir = args.data_root.expanduser().resolve() / "train"
    frame = _load_frame(external_root)
    row_groups, wells_index = pd.factorize(frame["well"], sort=True)
    wells = np.asarray(wells_index.astype(str), dtype=str)
    if len(wells) != EXPECTED_WELLS:
        raise AssertionError("well factorization is incomplete")

    model_oof, trainer_metadata = _load_model_oof(external_root, len(frame))
    family_oof = collapse_model_families(
        model_oof,
        families=DEFAULT_MODEL_FAMILIES,
    )
    target_delta = frame["target"].to_numpy(dtype=np.float64)
    base_delta, candidate_delta, meta_folds, fold_models = _crossfit_meta_models(
        model_oof,
        family_oof,
        target_delta,
        row_groups,
    )
    anchor = frame["last_known_tvt"].to_numpy(dtype=np.float64)
    pf_tvt = frame["pf_ancc"].to_numpy(dtype=np.float64)
    md_since = frame["md_since"].to_numpy(dtype=np.float64)
    truth = anchor + target_delta
    base_prediction = compose_ridge_pf_track(anchor, base_delta, pf_tvt, md_since)
    candidate_prediction = compose_ridge_pf_track(
        anchor,
        candidate_delta,
        pf_tvt,
        md_since,
    )
    partitions, fingerprints, partition_diagnostics = _build_robustness_partitions(
        wells,
        train_dir,
    )
    evaluation = _evaluate(
        truth,
        base_prediction,
        candidate_prediction,
        row_groups,
        wells,
        meta_folds,
        partitions,
    )
    if not np.isclose(
        evaluation["base_pooled_rmse"],
        EXPECTED_BASE_POOLED_RMSE,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "published competitive base failed exact reconstruction: "
            f"{evaluation['base_pooled_rmse']:.12f}"
        )

    report = {
        "contract": {
            "hypothesis": (
                "averaging correlated learners within model family prevents the "
                "three-versus-two model count from implicitly overweighting LightGBM"
            ),
            "candidate": (
                "positive Ridge over mean(3 LightGBM OOF), mean(2 CatBoost OOF)"
            ),
            "comparator": "positive Ridge over all five published grouped-OOF vectors",
            "ridge_parameters_unchanged": RIDGE_PARAMETERS,
            "ridge_pf_weights_unchanged": {"ridge": 0.91, "pf": 0.09},
            "ramp_tau_md_unchanged": 85.0,
            "meta_protocol": "published five-fold GroupKFold by well",
            "robustness_partitions": [
                "five exact typewell-fingerprint partitions",
                "seven X/Y spatial clusters, seed 20260727",
            ],
            "robustness_partitions_are_descriptive_not_nested_base_model_refits": True,
            "hidden_tvt_role": "stack fitting and offline scoring only",
            "well_id_feature": False,
            "leaderboard_tuning": False,
            "writes_submission": False,
            "thresholds_frozen_before_result": {
                "pooled_delta_limit": POOLED_DELTA_LIMIT,
                "well_macro_delta_limit": 0.0,
                "bootstrap_ci95_upper_limit": 0.0,
                "required_typewell_nonpositive": REQUIRED_TYPEWELL_NONPOSITIVE,
                "required_spatial_nonpositive": REQUIRED_SPATIAL_NONPOSITIVE,
                "maximum_partition_delta": MAXIMUM_PARTITION_DELTA,
                "p95_delta_limit": P95_DELTA_LIMIT,
                "worst_delta_limit": WORST_DELTA_LIMIT,
            },
            "pass_authorizes_submission": False,
            "threshold_tuning_after_result": False,
        },
        "source": {
            "rows": int(len(frame)),
            "wells": int(len(wells)),
            "trainers": trainer_metadata,
            "family_names": list(FAMILY_NAMES),
            "family_column_indices": DEFAULT_MODEL_FAMILIES,
            "partition_diagnostics": partition_diagnostics,
            "unique_typewell_fingerprints": int(len(np.unique(fingerprints))),
        },
        "meta_fold_models": fold_models,
        "evaluation": evaluation,
        "decision": {
            "decision": "GO" if evaluation["gate"]["decision"] == "PASS" else "NO-GO",
            "kaggle_submission_authorized": False,
            "reason": (
                "family balancing passed every frozen secondary-audit gate"
                if evaluation["gate"]["decision"] == "PASS"
                else "family balancing failed at least one frozen secondary-audit gate"
            ),
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
        f"{evaluation['base_pooled_rmse']:.6f} -> "
        f"{evaluation['candidate_pooled_rmse']:.6f} "
        f"({evaluation['pooled_delta']:+.6f}) "
        f"runtime={report['runtime_seconds']:.1f}s"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
