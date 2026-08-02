"""Evaluate target-free cross-well GR/Typewell datum transfer.

The public inspiration calibrated its scale from one fixed visible test well.
This evaluator removes that transaction.  It computes a target-free raw datum
midpoint for every training well, then fits one no-intercept scale using only
outer-training wells.  It never writes a submission.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evaluate_functional_ridge import DEFAULT_CACHE, PROTOCOLS, ROOT, _protocol_gate

DEFAULT_DATA = ROOT / "data" / "competition"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gr_datum_transfer_oof"
DATUM_GRID = np.arange(-20.0, 20.0001, 0.5)
MINIMUM_MODE_SEPARATION = 6.0
SMOOTHING_WIDTH = 7
GAIN_MINIMUM = 0.2
GAIN_MAXIMUM = 5.0
OFFSET_ABS_MAXIMUM = 750.0
SCALE_MINIMUM = -0.25
SCALE_MAXIMUM = 0.25
CORRECTION_CLIP = 2.5
MATERIAL_DELTA = -0.015
RAW_SUCCESS_MINIMUM = 0.95


def _rmse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / max(int(rows), 1))


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _smooth(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .interpolate(limit_direction="both")
        .rolling(SMOOTHING_WIDTH, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tvt = _numeric(typewell, "TVT")
    gr = _numeric(typewell, "GR")
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if int(valid.sum()) < 2:
        raise ValueError("Typewell has fewer than two finite rows")
    order = np.argsort(tvt[valid], kind="stable")
    tvt = tvt[valid][order]
    gr = gr[valid][order]
    tvt, keep = np.unique(tvt, return_index=True)
    return tvt, gr[keep]


def _raw_datum_midpoint(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    base: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Compute the target-free raw datum midpoint for one hidden suffix."""
    tvt_input = _numeric(horizontal, "TVT_input")
    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input))
    if len(known) < 2 or len(hidden) != len(base):
        raise ValueError("visible/hidden rows do not match the cached base")
    if hidden[0] != known[-1] + 1:
        raise ValueError("TVT_input does not form one visible-prefix/hidden-suffix split")

    tw_tvt, tw_gr = _prepare_typewell(typewell)
    gr = _smooth(_numeric(horizontal, "GR"))
    if not np.isfinite(gr).all():
        raise ValueError("smoothed horizontal GR is not finite")
    target_gr = np.interp(
        tvt_input[known],
        tw_tvt,
        tw_gr,
        left=tw_gr[0],
        right=tw_gr[-1],
    )
    design = np.column_stack((gr[known], np.ones(len(known))))
    coefficients, *_ = np.linalg.lstsq(design, target_gr, rcond=None)
    gain, offset = (float(coefficients[0]), float(coefficients[1]))
    if not (GAIN_MINIMUM <= gain <= GAIN_MAXIMUM):
        raise ValueError("heel GR gain is outside the frozen guard")
    if abs(offset) > OFFSET_ABS_MAXIMUM:
        raise ValueError("heel GR offset is outside the frozen guard")

    calibrated_gr = gr[hidden] * gain + offset
    cost = np.empty(len(DATUM_GRID), dtype=float)
    for index, shift in enumerate(DATUM_GRID):
        expected = np.interp(
            np.asarray(base, dtype=float) + shift,
            tw_tvt,
            tw_gr,
            left=tw_gr[0],
            right=tw_gr[-1],
        )
        residual = calibrated_gr - expected
        center = float(np.median(residual))
        scale = float(np.median(np.abs(residual - center)) * 1.4826 + 1e-6)
        cost[index] = float(np.mean(np.clip(residual / scale, -5.0, 5.0) ** 2))
    first = int(np.argmin(cost))
    eligible = np.abs(DATUM_GRID - DATUM_GRID[first]) >= MINIMUM_MODE_SEPARATION
    if not bool(eligible.any()):
        raise ValueError("datum grid has no separated second mode")
    eligible_indices = np.flatnonzero(eligible)
    second = int(eligible_indices[int(np.argmin(cost[eligible]))])
    raw = 0.5 * float(DATUM_GRID[first] + DATUM_GRID[second])
    return raw, {
        "gain": gain,
        "offset": offset,
        "first_shift": float(DATUM_GRID[first]),
        "second_shift": float(DATUM_GRID[second]),
        "first_cost": float(cost[first]),
        "second_cost": float(cost[second]),
    }


def _fit_scale(raw: np.ndarray, targets: np.ndarray, rows: np.ndarray) -> float:
    raw = np.asarray(raw, dtype=float)
    targets = np.asarray(targets, dtype=float)
    weights = np.asarray(rows, dtype=float)
    denominator = float(np.sum(weights * raw * raw))
    if denominator <= 1e-12:
        return 0.0
    scale = float(np.sum(weights * raw * targets) / denominator)
    return float(np.clip(scale, SCALE_MINIMUM, SCALE_MAXIMUM))


def _gate(result: dict[str, Any], raw_success_fraction: float) -> dict[str, bool]:
    checks = _protocol_gate(result)
    checks["raw_signal_success_fraction"] = bool(
        raw_success_fraction >= RAW_SUCCESS_MINIMUM
    )
    checks["material_pooled_improvement"] = bool(
        result["pooled_delta"] <= MATERIAL_DELTA
    )
    return checks


def _evaluate_protocol(
    *,
    protocol: str,
    metadata: dict[str, Any],
    raw: np.ndarray,
    raw_success: np.ndarray,
    target_bias: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = np.asarray(
        [int(item["folds"][protocol]) for item in metadata["per_well"]],
        dtype=np.int64,
    )
    well_rows = np.diff(offsets).astype(np.int64)
    base_sse = 0.0
    candidate_sse = 0.0
    per_well: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []

    for fold in sorted(np.unique(folds).tolist()):
        train = np.flatnonzero(folds != fold)
        validation = np.flatnonzero(folds == fold)
        scale = _fit_scale(raw[train], target_bias[train], well_rows[train])
        fold_base_sse = 0.0
        fold_candidate_sse = 0.0
        fold_rows = 0
        for well_index in validation.tolist():
            start = int(offsets[well_index])
            end = int(offsets[well_index + 1])
            rows = end - start
            correction = float(np.clip(scale * raw[well_index], -CORRECTION_CLIP, CORRECTION_CLIP))
            well_truth = np.asarray(truth[start:end], dtype=float)
            well_base = np.asarray(base[start:end], dtype=float)
            well_candidate = well_base + correction
            well_base_sse = float(np.sum((well_base - well_truth) ** 2))
            well_candidate_sse = float(np.sum((well_candidate - well_truth) ** 2))
            well_base_rmse = _rmse(well_base_sse, rows)
            well_candidate_rmse = _rmse(well_candidate_sse, rows)
            base_sse += well_base_sse
            candidate_sse += well_candidate_sse
            fold_base_sse += well_base_sse
            fold_candidate_sse += well_candidate_sse
            fold_rows += rows
            per_well.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "well": str(metadata["per_well"][well_index]["well"]),
                    "rows": rows,
                    "raw_success": bool(raw_success[well_index]),
                    "raw_datum": float(raw[well_index]),
                    "fitted_scale": scale,
                    "correction": correction,
                    "target_bias_diagnostic": float(target_bias[well_index]),
                    "base_rmse": well_base_rmse,
                    "candidate_rmse": well_candidate_rmse,
                    "delta": well_candidate_rmse - well_base_rmse,
                }
            )
        fold_base_rmse = _rmse(fold_base_sse, fold_rows)
        fold_candidate_rmse = _rmse(fold_candidate_sse, fold_rows)
        fold_results.append(
            {
                "fold": int(fold),
                "train_wells": int(len(train)),
                "validation_wells": int(len(validation)),
                "rows": int(fold_rows),
                "fitted_scale": scale,
                "base_rmse": fold_base_rmse,
                "candidate_rmse": fold_candidate_rmse,
                "delta": fold_candidate_rmse - fold_base_rmse,
            }
        )

    frame = pd.DataFrame(per_well)
    total_rows = int(offsets[-1])
    base_rmse = _rmse(base_sse, total_rows)
    candidate_rmse = _rmse(candidate_sse, total_rows)
    result: dict[str, Any] = {
        "protocol": protocol,
        "fold_count": int(len(fold_results)),
        "base_rmse": base_rmse,
        "candidate_rmse": candidate_rmse,
        "pooled_delta": candidate_rmse - base_rmse,
        "nonpositive_well_fraction": float((frame["delta"] <= 0.0).mean()),
        "p95_well_delta": float(
            frame["candidate_rmse"].quantile(0.95)
            - frame["base_rmse"].quantile(0.95)
        ),
        "worst_well_delta": float(
            frame["candidate_rmse"].max() - frame["base_rmse"].max()
        ),
        "improved_folds": int(sum(row["delta"] < 0.0 for row in fold_results)),
        "folds": fold_results,
    }
    raw_success_fraction = float(np.mean(raw_success))
    result["gate_checks"] = _gate(result, raw_success_fraction)
    result["gate_passed"] = bool(all(result["gate_checks"].values()))
    return result, per_well


def evaluate(data_dir: Path, cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    started = time.time()
    metadata = json.loads(
        (cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8")
    )
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    well_count = len(offsets) - 1
    if well_count != len(metadata["per_well"]):
        raise RuntimeError("cache well count does not match metadata")
    if int(offsets[-1]) != len(base) or len(base) != len(truth):
        raise RuntimeError("cache arrays are inconsistent")

    raw = np.zeros(well_count, dtype=float)
    raw_success = np.zeros(well_count, dtype=bool)
    target_bias = np.zeros(well_count, dtype=float)
    raw_rows: list[dict[str, Any]] = []
    for well_index, item in enumerate(metadata["per_well"]):
        well = str(item["well"])
        start = int(offsets[well_index])
        end = int(offsets[well_index + 1])
        target_bias[well_index] = float(
            np.mean(
                np.asarray(truth[start:end], dtype=float)
                - np.asarray(base[start:end], dtype=float)
            )
        )
        horizontal_path = data_dir / "train" / f"{well}__horizontal_well.csv"
        typewell_path = data_dir / "train" / f"{well}__typewell.csv"
        status = "ok"
        diagnostics: dict[str, Any] = {}
        try:
            value, diagnostics = _raw_datum_midpoint(
                pd.read_csv(horizontal_path),
                pd.read_csv(typewell_path),
                np.asarray(base[start:end], dtype=float),
            )
            raw[well_index] = value
            raw_success[well_index] = True
        except Exception as exc:
            raw[well_index] = 0.0
            status = f"fallback_zero:{type(exc).__name__}:{exc}"
        raw_rows.append(
            {
                "well": well,
                "rows": end - start,
                "status": status,
                "raw_datum": float(raw[well_index]),
                "target_bias_diagnostic": float(target_bias[well_index]),
                **diagnostics,
            }
        )

    results: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        result, protocol_details = _evaluate_protocol(
            protocol=protocol,
            metadata=metadata,
            raw=raw,
            raw_success=raw_success,
            target_bias=target_bias,
            base=base,
            truth=truth,
            offsets=offsets,
        )
        results[protocol] = result
        details.extend(protocol_details)

    report = {
        "candidate": "target-free GR/Typewell datum transfer",
        "public_source_informed": True,
        "contract": {
            "writes_submission": False,
            "fixed_test_well_used": False,
            "expected_visible_rows_or_hash_used": False,
            "leaderboard_score_used_at_runtime": False,
            "raw_datum_uses_test_available_features_only": True,
            "scale_fit_on_outer_training_wells_only": True,
            "fit_intercept": False,
            "row_weighted_scale_fit": True,
            "scale_guard": [SCALE_MINIMUM, SCALE_MAXIMUM],
            "correction_clip": CORRECTION_CLIP,
        },
        "data": str(data_dir),
        "cache": str(cache_dir),
        "wells": int(well_count),
        "rows": int(len(base)),
        "raw_signal_success_wells": int(raw_success.sum()),
        "raw_signal_success_fraction": float(np.mean(raw_success)),
        "protocols": results,
        "decision": (
            "PASS" if all(row["gate_passed"] for row in results.values()) else "FAIL"
        ),
        "runtime_seconds": float(time.time() - started),
    }
    return report, pd.DataFrame(details), pd.DataFrame(raw_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, detail, raw = evaluate(args.data_dir.resolve(), args.cache_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "gr_datum_transfer_oof_report.json"
    detail_path = args.output_dir / "gr_datum_transfer_oof_per_well.csv"
    raw_path = args.output_dir / "gr_datum_transfer_raw_signal.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(detail_path, index=False)
    raw.to_csv(raw_path, index=False)
    print(json.dumps({"decision": report["decision"], "protocols": report["protocols"]}, indent=2))
    print(f"wrote {report_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {raw_path}")


if __name__ == "__main__":
    main()
