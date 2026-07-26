"""Frozen conservative residual-model OOF evaluation.

Spatial-invariant v2 improved aggregate geographic OOF but occasionally
overcorrected one shared region. This v3 hypothesis keeps the exact v2
features and model, then applies a fixed 0.20 scale to its predicted residual.
It must pass the three existing protocols plus a new seven-cluster spatial
protocol that had not been scored when the scale was frozen.
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
)
from residual_gbdt_invariant_evaluate import (  # noqa: E402
    EXCLUDED_ABSOLUTE_POSITION_FEATURES,
    INVARIANT_FEATURE_INDICES,
    INVARIANT_FEATURE_NAMES,
    SECOND_SPATIAL_SEED,
    build_invariant_fold_assignments,
)

from rogii.residual_gbdt import ResidualFeatureConfig  # noqa: E402
from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
)

RESIDUAL_SCALE = 0.20
CONFIRMATION_SPATIAL_SEED = 20_260_727
CONFIRMATION_SPATIAL_FOLDS = 7


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
        default=PROJECT_ROOT / "artifacts" / "residual_gbdt_shrink_oof.json",
    )
    parser.add_argument(
        "--max-wells",
        type=int,
        default=0,
        help="Smoke-test prefix of wells; zero requires the frozen full 773.",
    )
    return parser


def build_shrink_fold_assignments(
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build the four frozen v3 fold assignments."""

    assignments = build_invariant_fold_assignments(records)
    coordinates = np.asarray(
        [[row["centroid_x"], row["centroid_y"]] for row in records],
        dtype=float,
    )
    scaled = StandardScaler().fit_transform(coordinates)
    raw_confirmation = KMeans(
        n_clusters=CONFIRMATION_SPATIAL_FOLDS,
        random_state=CONFIRMATION_SPATIAL_SEED,
        n_init=50,
    ).fit_predict(scaled)
    assignments["spatial_cluster_7_seed_20260727"] = _relabel_spatial_clusters(
        raw_confirmation,
        coordinates,
        fold_count=CONFIRMATION_SPATIAL_FOLDS,
    )
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
    if len(paths) < 35:
        raise ValueError("at least 35 wells are required for seven-fold evaluation")

    records = _scan_wells(paths, train_dir)
    assignments = build_shrink_fold_assignments(records)
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
            bootstrap_seed=CONFIRMATION_SPATIAL_SEED + protocol_index,
            model_parameters=model_parameters,
            residual_scale=RESIDUAL_SCALE,
        )
        for protocol_index, (name, folds) in enumerate(assignments.items())
    }
    all_pass = all(result["gate"]["decision"] == "PASS" for result in protocol_results.values())
    report: dict[str, Any] = {
        "evaluation_contract": {
            "hypothesis": ("fixed 0.20 residual shrink prevents the v2 regional overcorrection"),
            "hypothesis_informed_by_v2_spatial_failure": True,
            "seven_cluster_protocol_unscored_when_scale_was_frozen": True,
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
                "seven spatial-cluster folds using seed 20260727",
            ],
        },
        "frozen_configuration": {
            "feature_config": asdict(ResidualFeatureConfig()),
            "excluded_features": list(EXCLUDED_ABSOLUTE_POSITION_FEATURES),
            "feature_names": list(INVARIANT_FEATURE_NAMES),
            "residual_scale": RESIDUAL_SCALE,
            "maximum_training_rows_per_well": MAX_TRAIN_ROWS_PER_WELL,
            "model_family": "CatBoostRegressor",
            "model_parameters": model_parameters,
            "prediction_residual_clip_before_scale": PREDICTION_RESIDUAL_CLIP,
            "random_seeds": [
                RANDOM_SEED,
                SECOND_SPATIAL_SEED,
                CONFIRMATION_SPATIAL_SEED,
            ],
            "gate": {
                "pooled_delta_limit": POOLED_DELTA_LIMIT,
                "required_nonpositive_fold_fraction": 0.8,
                "maximum_fold_delta": MAXIMUM_FOLD_DELTA,
                "p95_delta_limit": P95_DELTA_LIMIT,
                "worst_delta_limit": WORST_DELTA_LIMIT,
                "all_four_protocols_must_pass": True,
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
                "the frozen shrunken model passed all four OOF protocols"
                if all_pass
                else "the frozen shrunken model failed at least one OOF protocol"
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
