"""Confirm the frozen dip-rate HMM on all 741 untouched training wells.

The discovery parameters are imported unchanged.  The 32 discovery wells are
excluded exactly, and every remaining well is evaluated once.  The target-free
per-well HMM is parallelized only for runtime; no prediction parameter or gate
depends on confirmation truth.  This script never writes a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evaluate_dip_rate_hmm_discovery import (
    DEFAULT_CACHE,
    DEFAULT_DATA_ROOT,
    DISAGREEMENT_CLIP,
    DISCOVERY_FOLD,
    DISCOVERY_WELLS,
    OVERLAY_SCALE,
    ROOT,
    predict_hidden_hmm,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_confirmation"
WORKERS = 8
PROTOCOLS = (
    "spatial_cluster_7_seed_20260727",
    "typewell_fingerprint",
)
POOLED_DELTA_LIMIT = -0.05
WELL_WIN_FRACTION_LIMIT = 0.65


def _rmse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / max(int(rows), 1))


def _discovery_indices(metadata: dict[str, Any]) -> set[int]:
    eligible = [
        (index, str(item["well"]))
        for index, item in enumerate(metadata["per_well"])
        if int(item["folds"]["spatial_cluster_7_seed_20260727"])
        == DISCOVERY_FOLD
    ]
    eligible.sort(
        key=lambda item: hashlib.sha256(item[1].encode("utf-8")).hexdigest()
    )
    return {index for index, _ in eligible[:DISCOVERY_WELLS]}


def _evaluate_one(
    item: tuple[int, dict[str, Any]],
    *,
    data_root: Path,
    offsets: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    well_index, metadata = item
    well = str(metadata["well"])
    started = time.perf_counter()
    horizontal = pd.read_csv(data_root / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(data_root / "train" / f"{well}__typewell.csv")
    hmm_prediction, uncertainty = predict_hidden_hmm(horizontal, typewell)
    start = int(offsets[well_index])
    end = int(offsets[well_index + 1])
    well_base = np.asarray(base[start:end], dtype=float)
    well_truth = np.asarray(truth[start:end], dtype=float)
    if len(hmm_prediction) != len(well_base):
        raise RuntimeError(f"{well}: HMM/cache row mismatch")
    correction = OVERLAY_SCALE * np.clip(
        hmm_prediction - well_base,
        -DISAGREEMENT_CLIP,
        DISAGREEMENT_CLIP,
    )
    candidate = well_base + correction
    base_sse = float(np.sum(np.square(well_truth - well_base)))
    candidate_sse = float(np.sum(np.square(well_truth - candidate)))
    rows = len(well_base)
    base_rmse = _rmse(base_sse, rows)
    candidate_rmse = _rmse(candidate_sse, rows)
    return {
        "well": well,
        "rows": int(rows),
        "base_sse": base_sse,
        "candidate_sse": candidate_sse,
        "base_rmse": base_rmse,
        "candidate_rmse": candidate_rmse,
        "delta": candidate_rmse - base_rmse,
        "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
        "hmm_uncertainty_mean": float(np.mean(uncertainty)),
        "runtime_seconds": float(time.perf_counter() - started),
        **{f"fold_{protocol}": int(metadata["folds"][protocol]) for protocol in PROTOCOLS},
    }


def _protocol_summary(frame: pd.DataFrame, protocol: str) -> dict[str, Any]:
    fold_column = f"fold_{protocol}"
    fold_rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby(fold_column, sort=True):
        rows = int(group["rows"].sum())
        base_rmse = _rmse(float(group["base_sse"].sum()), rows)
        candidate_rmse = _rmse(float(group["candidate_sse"].sum()), rows)
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(len(group)),
                "rows": rows,
                "base_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "delta": candidate_rmse - base_rmse,
            }
        )
    total_rows = int(frame["rows"].sum())
    base_rmse = _rmse(float(frame["base_sse"].sum()), total_rows)
    candidate_rmse = _rmse(float(frame["candidate_sse"].sum()), total_rows)
    p95_delta = float(
        frame["candidate_rmse"].quantile(0.95)
        - frame["base_rmse"].quantile(0.95)
    )
    worst_delta = float(
        frame["candidate_rmse"].max() - frame["base_rmse"].max()
    )
    required_folds = 5 if len(fold_rows) >= 7 else len(fold_rows)
    improved_folds = int(sum(row["delta"] < 0.0 for row in fold_rows))
    checks = {
        "pooled_delta_at_most_minus_0_05": bool(
            candidate_rmse - base_rmse <= POOLED_DELTA_LIMIT
        ),
        "well_win_fraction_at_least_0_65": bool(
            (frame["delta"] <= 0.0).mean() >= WELL_WIN_FRACTION_LIMIT
        ),
        "p95_not_worse": bool(p95_delta <= 0.0),
        "worst_not_worse_than_plus_0_25": bool(worst_delta <= 0.25),
        "enough_folds_improve": bool(improved_folds >= required_folds),
    }
    return {
        "protocol": protocol,
        "fold_count": int(len(fold_rows)),
        "improved_folds": improved_folds,
        "base_rmse": base_rmse,
        "candidate_rmse": candidate_rmse,
        "pooled_delta": candidate_rmse - base_rmse,
        "well_win_fraction": float((frame["delta"] <= 0.0).mean()),
        "p95_well_delta": p95_delta,
        "worst_well_delta": worst_delta,
        "folds": fold_rows,
        "gate_checks": checks,
        "gate_passed": bool(all(checks.values())),
    }


def evaluate(data_root: Path, cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    metadata = json.loads(
        (cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8")
    )
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    excluded = _discovery_indices(metadata)
    items = [
        (index, row)
        for index, row in enumerate(metadata["per_well"])
        if index not in excluded
    ]
    if len(items) != len(metadata["per_well"]) - DISCOVERY_WELLS:
        raise AssertionError("confirmation/discovery partition mismatch")

    def worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        return _evaluate_one(
            item,
            data_root=data_root,
            offsets=offsets,
            base=base,
            truth=truth,
        )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for position, row in enumerate(executor.map(worker, items), start=1):
            rows.append(row)
            if position % 25 == 0 or position == len(items):
                print(
                    f"[{position:03d}/{len(items):03d}] latest={row['well']} "
                    f"delta={row['delta']:+.6f}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    protocols = {
        protocol: _protocol_summary(frame, protocol) for protocol in PROTOCOLS
    }
    runtime = float(time.perf_counter() - started)
    runtime_gate = bool(frame["runtime_seconds"].mean() <= 90.0)
    report = {
        "candidate": "target-free dip-rate HMM clipped overlay confirmation",
        "contract": {
            "writes_submission": False,
            "discovery_wells_excluded": DISCOVERY_WELLS,
            "confirmation_wells": int(len(frame)),
            "confirmation_selection": "all official training wells outside frozen discovery set",
            "prediction_parameters_imported_unchanged_from_discovery": True,
            "parallel_workers": WORKERS,
            "parallelism_changes_predictions": False,
            "target_free_runtime_inputs_only": True,
            "parameter_grid_searched": False,
        },
        "protocols": protocols,
        "mean_runtime_per_well": float(frame["runtime_seconds"].mean()),
        "wall_runtime_seconds": runtime,
        "runtime_gate_passed": runtime_gate,
        "decision": (
            "PASS"
            if runtime_gate and all(row["gate_passed"] for row in protocols.values())
            else "FAIL"
        ),
    }
    return report, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, detail = evaluate(args.data_root.resolve(), args.cache_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "dip_rate_hmm_confirmation_report.json"
    detail_path = args.output_dir / "dip_rate_hmm_confirmation_per_well.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(detail_path, index=False)
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
