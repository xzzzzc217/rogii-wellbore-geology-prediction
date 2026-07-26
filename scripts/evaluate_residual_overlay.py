"""Qualify the frozen residual-v3 correction against competitive base trajectories.

The residual model was selected only with leakage-safe OOF protocols against
target-free baselines.  This script performs a separate deployment check:

1. exclude the three visible test wells and their typewell fingerprints;
2. train the frozen spatial-invariant CatBoost model on the remaining wells;
3. predict the already-frozen 0.20-shrunk residual for the excluded wells;
4. add that residual to one or more frozen competitive base submissions; and
5. read the excluded labels only for post-prediction scoring.

It never writes a submission file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from residual_gbdt_evaluate import (  # noqa: E402
    MAX_TRAIN_ROWS_PER_WELL,
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
from residual_gbdt_shrink_evaluate import RESIDUAL_SCALE  # noqa: E402

from rogii.residual_gbdt import build_residual_features, typewell_fingerprint  # noqa: E402

DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "artifacts"
    / "kaggle_runs"
    / "pf_exact128_v6d_v1"
    / "submission_sp45_learned_w0.60.csv",
    PROJECT_ROOT
    / "artifacts"
    / "kaggle_runs"
    / "pf_exact128_v6d_v1"
    / "submission_before_branch_hedge.csv",
    PROJECT_ROOT / "artifacts" / "kaggle_runs" / "pf_exact128_v6d_v1" / "submission.csv",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        help="Frozen id,tvt CSV to qualify; repeat for multiple candidates.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "residual_overlay_qualification.json",
    )
    return parser


def _split_id(value: str) -> tuple[str, int]:
    well, row = value.rsplit("_", 1)
    return well, int(row)


def _load_candidate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["id", "tvt"]:
        raise ValueError(f"{path}: expected id,tvt columns")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate IDs")
    frame["id"] = frame["id"].astype(str)
    frame["tvt"] = pd.to_numeric(frame["tvt"], errors="coerce")
    if not np.isfinite(frame["tvt"]).all():
        raise ValueError(f"{path}: non-finite prediction")
    split = frame["id"].map(_split_id)
    frame["well"] = split.map(lambda item: item[0])
    frame["row_idx"] = split.map(lambda item: item[1]).astype(int)
    return frame


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _pooled(per_well: list[dict[str, Any]], key: str) -> float:
    rows = np.asarray([row["rows"] for row in per_well], dtype=float)
    rmse = np.asarray([row[key] for row in per_well], dtype=float)
    return float(np.sqrt(np.sum(rows * np.square(rmse)) / np.sum(rows)))


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    candidate_paths = [
        path.expanduser().resolve() for path in (args.candidate or DEFAULT_CANDIDATES)
    ]
    candidates = {
        path.stem + f"_{index}": _load_candidate(path) for index, path in enumerate(candidate_paths)
    }
    reference_ids = next(iter(candidates.values()))["id"].tolist()
    for name, frame in candidates.items():
        if frame["id"].tolist() != reference_ids:
            raise ValueError(f"{name}: candidate ID order differs from reference")

    excluded_wells = sorted(next(iter(candidates.values()))["well"].unique())
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if len(paths) != 773:
        raise ValueError(f"expected 773 training wells, found {len(paths)}")

    records = _scan_wells(paths, train_dir)
    record_by_well = {str(row["well_id"]): row for row in records}
    missing = sorted(set(excluded_wells) - set(record_by_well))
    if missing:
        raise ValueError(f"candidate wells missing from train records: {missing}")
    excluded_fingerprints = {record_by_well[well]["fingerprint"] for well in excluded_wells}
    training_records = [
        row
        for row in records
        if row["well_id"] not in excluded_wells and row["fingerprint"] not in excluded_fingerprints
    ]
    train_x = np.concatenate(
        [row["sample_features"][:, INVARIANT_FEATURE_INDICES] for row in training_records],
        axis=0,
    )
    train_y = np.concatenate([row["sample_labels"] for row in training_records])
    train_weight = np.concatenate(
        [
            np.full(
                len(row["sample_labels"]),
                row["sample_weight"],
                dtype=np.float32,
            )
            for row in training_records
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

    residual_by_id: dict[str, float] = {}
    truth_by_id: dict[str, float] = {}
    correction_summary: dict[str, Any] = {}
    for well in excluded_wells:
        path = Path(record_by_well[well]["path"])
        horizontal, typewell, truth = _read_inputs(path, train_dir)
        if typewell_fingerprint(typewell) not in excluded_fingerprints:
            raise AssertionError("excluded typewell fingerprint changed during qualification")
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
        hidden_truth = truth[block.evaluation_mask]
        if not (len(hidden_rows) == len(correction) == len(hidden_truth)):
            raise AssertionError(f"{well}: correction length mismatch")
        for row_index, value, target in zip(hidden_rows, correction, hidden_truth, strict=True):
            row_id = f"{well}_{int(row_index)}"
            residual_by_id[row_id] = float(value)
            truth_by_id[row_id] = float(target)
        correction_summary[well] = {
            "rows": int(len(correction)),
            "mean": float(np.mean(correction)),
            "mean_abs": float(np.mean(np.abs(correction))),
            "p95_abs": float(np.quantile(np.abs(correction), 0.95)),
            "max_abs": float(np.max(np.abs(correction))),
        }

    results: dict[str, Any] = {}
    for name, frame in candidates.items():
        correction = frame["id"].map(residual_by_id)
        truth = frame["id"].map(truth_by_id)
        if correction.isna().any() or truth.isna().any():
            raise ValueError(f"{name}: missing correction or truth mapping")
        base = frame["tvt"].to_numpy(dtype=float)
        overlay = base + correction.to_numpy(dtype=float)
        target = truth.to_numpy(dtype=float)
        per_well: list[dict[str, Any]] = []
        for well in excluded_wells:
            mask = frame["well"].eq(well).to_numpy()
            base_rmse = _rmse(target[mask], base[mask])
            overlay_rmse = _rmse(target[mask], overlay[mask])
            per_well.append(
                {
                    "well": well,
                    "rows": int(mask.sum()),
                    "base_rmse": base_rmse,
                    "overlay_rmse": overlay_rmse,
                    "delta": float(overlay_rmse - base_rmse),
                }
            )
        base_pooled = _pooled(per_well, "base_rmse")
        overlay_pooled = _pooled(per_well, "overlay_rmse")
        results[name] = {
            "path": str(candidate_paths[int(name.rsplit("_", 1)[1])]),
            "base_pooled_rmse": base_pooled,
            "overlay_pooled_rmse": overlay_pooled,
            "delta": float(overlay_pooled - base_pooled),
            "per_well": per_well,
        }

    all_candidates_improve = all(row["delta"] < 0.0 for row in results.values())
    report = {
        "contract": {
            "writes_submission": False,
            "excluded_wells": excluded_wells,
            "excluded_typewell_fingerprints": sorted(excluded_fingerprints),
            "training_wells": int(len(training_records)),
            "training_rows": int(len(train_y)),
            "maximum_training_rows_per_well": MAX_TRAIN_ROWS_PER_WELL,
            "feature_names": list(INVARIANT_FEATURE_NAMES),
            "residual_scale": RESIDUAL_SCALE,
            "prediction_residual_clip_before_scale": PREDICTION_RESIDUAL_CLIP,
            "model_parameters": MODEL_PARAMETERS,
        },
        "correction": correction_summary,
        "candidates": results,
        "decision": {
            "incremental_demo_gate": "PASS" if all_candidates_improve else "FAIL",
            "all_frozen_candidates_improve": all_candidates_improve,
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
        f"incremental_demo_gate={report['decision']['incremental_demo_gate']} "
        f"training_wells={len(training_records)} runtime={report['runtime_seconds']:.1f}s"
    )
    for name, row in results.items():
        print(
            f"{name}: {row['base_pooled_rmse']:.6f} -> "
            f"{row['overlay_pooled_rmse']:.6f} ({row['delta']:+.6f})"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
