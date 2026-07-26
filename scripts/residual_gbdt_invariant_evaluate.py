"""Frozen spatial-invariant residual-model OOF evaluation.

This v2 experiment was defined after v1 showed a geographic generalization
failure.  It removes only absolute X/Y/Z position features and retains
relative trajectory geometry.  One model must pass all three protocols:

* five folds grouped by exact typewell fingerprint;
* five KMeans spatial holdouts using the original v1 seed;
* six KMeans spatial holdouts using a second frozen seed.

Validation typewell fingerprints are removed from every fold's training
wells.  The stricter third protocol is an added robustness check, not an
untouched confirmation set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from residual_gbdt_evaluate import (  # noqa: E402
    MAX_TRAIN_ROWS_PER_WELL,
    MAXIMUM_FOLD_DELTA,
    MODEL_PARAMETERS,
    P95_DELTA_LIMIT,
    POOLED_DELTA_LIMIT,
    PREDICTION_RESIDUAL_CLIP,
    RANDOM_SEED,
    WORST_DELTA_LIMIT,
    _evaluate_protocol,
    _json_safe,
    _relabel_spatial_clusters,
    _scan_wells,
    _sha256,
    build_fold_assignments,
)

from rogii.residual_gbdt import (  # noqa: E402
    RESIDUAL_FEATURE_NAMES,
    ResidualFeatureConfig,
)
from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
)

SECOND_SPATIAL_SEED = 20_260_726
SECOND_SPATIAL_FOLDS = 6
EXCLUDED_ABSOLUTE_POSITION_FEATURES = (
    "x_scaled",
    "y_scaled",
    "z_scaled",
    "anchor_x_scaled",
    "anchor_y_scaled",
    "anchor_z_scaled",
)
INVARIANT_FEATURE_NAMES = tuple(
    name for name in RESIDUAL_FEATURE_NAMES if name not in EXCLUDED_ABSOLUTE_POSITION_FEATURES
)
INVARIANT_FEATURE_INDICES = np.asarray(
    [RESIDUAL_FEATURE_NAMES.index(name) for name in INVARIANT_FEATURE_NAMES],
    dtype=np.int64,
)


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
        default=PROJECT_ROOT / "artifacts" / "residual_gbdt_invariant_oof.json",
    )
    parser.add_argument(
        "--max-wells",
        type=int,
        default=0,
        help="Smoke-test prefix of wells; zero requires the frozen full 773.",
    )
    return parser


def build_invariant_fold_assignments(
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build the three frozen v2 fold assignments."""

    assignments = build_fold_assignments(records)
    coordinates = np.asarray(
        [[row["centroid_x"], row["centroid_y"]] for row in records],
        dtype=float,
    )
    scaled = StandardScaler().fit_transform(coordinates)
    raw_second = KMeans(
        n_clusters=SECOND_SPATIAL_FOLDS,
        random_state=SECOND_SPATIAL_SEED,
        n_init=50,
    ).fit_predict(scaled)
    second = _relabel_spatial_clusters(
        raw_second,
        coordinates,
        fold_count=SECOND_SPATIAL_FOLDS,
    )
    assignments["spatial_cluster_6_seed_20260726"] = second
    return assignments


def _required_nonpositive_folds(folds: np.ndarray) -> int:
    return int(math.ceil(0.8 * len(np.unique(folds))))


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
    if len(paths) < 30:
        raise ValueError("at least 30 wells are required for six-fold evaluation")

    records = _scan_wells(paths, train_dir)
    assignments = build_invariant_fold_assignments(records)
    model_parameters = dict(MODEL_PARAMETERS)
    if args.max_wells:
        model_parameters["iterations"] = 20
    protocol_results = {
        name: _evaluate_protocol(
            name,
            records,
            folds,
            train_dir,
            feature_indices=INVARIANT_FEATURE_INDICES,
            feature_names=INVARIANT_FEATURE_NAMES,
            required_nonpositive_folds=_required_nonpositive_folds(folds),
            bootstrap_seed=SECOND_SPATIAL_SEED + protocol_index,
            model_parameters=model_parameters,
        )
        for protocol_index, (name, folds) in enumerate(assignments.items())
    }
    all_pass = all(result["gate"]["decision"] == "PASS" for result in protocol_results.values())
    report: dict[str, Any] = {
        "evaluation_contract": {
            "hypothesis": (
                "remove absolute X/Y/Z position while retaining relative trajectory geometry"
            ),
            "hypothesis_informed_by_v1_spatial_failure": True,
            "third_protocol_is_robustness_check_not_untouched_confirmation": True,
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
                "five spatial-cluster folds using seed 20260725",
                "six spatial-cluster folds using seed 20260726",
            ],
        },
        "frozen_configuration": {
            "feature_config": asdict(ResidualFeatureConfig()),
            "excluded_features": list(EXCLUDED_ABSOLUTE_POSITION_FEATURES),
            "feature_names": list(INVARIANT_FEATURE_NAMES),
            "maximum_training_rows_per_well": MAX_TRAIN_ROWS_PER_WELL,
            "model_family": "CatBoostRegressor",
            "model_parameters": model_parameters,
            "prediction_residual_clip": PREDICTION_RESIDUAL_CLIP,
            "random_seeds": [RANDOM_SEED, SECOND_SPATIAL_SEED],
            "gate": {
                "pooled_delta_limit": POOLED_DELTA_LIMIT,
                "required_nonpositive_fold_fraction": 0.8,
                "maximum_fold_delta": MAXIMUM_FOLD_DELTA,
                "p95_delta_limit": P95_DELTA_LIMIT,
                "worst_delta_limit": WORST_DELTA_LIMIT,
                "all_three_protocols_must_pass": True,
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
                "the one frozen spatial-invariant model passed all three OOF protocols"
                if all_pass
                else "the one frozen spatial-invariant model failed at least one OOF protocol"
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
