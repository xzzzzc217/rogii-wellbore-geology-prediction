"""Evaluate one preregistered target-free dip-rate HMM discovery candidate.

This is a bounded discovery test, not a submission builder.  A compact HMM
tracks two hidden states along each scored suffix: TVT position on the
typewell and local d(TVT+Z)/dMD rate.  Emissions use only hidden-test-available
GR and the typewell GR curve.  The HMM output supplies a frozen, clipped 0.20
correction to the competitive OOF base.

The discovery set is fixed before evaluation: the first 32 SHA-256-sorted
wells in spatial fold 0.  Failure stops the route without parameter tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "competition"
DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_discovery"

DISCOVERY_FOLD = 0
DISCOVERY_WELLS = 32
GRID_STEP = 0.50
RATE_STATES = 21
RATE_SPAN = 0.10
POSITION_BAND = 60.0
POSITION_SIGMA = 0.25
RATE_STAY_PROBABILITY = 0.96
OVERLAY_SCALE = 0.20
DISAGREEMENT_CLIP = 5.0


@njit(cache=True)
def _logadd(current: float, value: float) -> float:
    if current < -1.0e20:
        return value
    if value > current:
        return value + math.log1p(math.exp(current - value))
    return current + math.log1p(math.exp(value - current))


@njit(cache=True)
def _dip_rate_filter(
    emission: np.ndarray,
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    grid: np.ndarray,
    rates: np.ndarray,
    start_tvt: float,
    start_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-filter the joint TVT-position and U-rate state."""
    steps, positions = emission.shape
    rate_count = len(rates)
    log_state = np.empty((positions, rate_count), dtype=np.float64)
    for p in range(positions):
        position_term = -0.5 * ((grid[p] - start_tvt) / 0.75) ** 2
        for r in range(rate_count):
            rate_term = -0.5 * ((rates[r] - start_rate) / 0.01) ** 2
            log_state[p, r] = position_term + rate_term

    stay_log = math.log(RATE_STAY_PROBABILITY)
    neighbor_log = math.log((1.0 - RATE_STAY_PROBABILITY) / 2.0)
    predictions = np.empty(steps, dtype=np.float64)
    uncertainty = np.empty(steps, dtype=np.float64)
    after_rate = np.empty_like(log_state)
    current = np.empty_like(log_state)

    for t in range(steps):
        for p in range(positions):
            for r2 in range(rate_count):
                value = log_state[p, r2] + stay_log
                if r2 > 0:
                    value = _logadd(value, log_state[p, r2 - 1] + neighbor_log)
                if r2 + 1 < rate_count:
                    value = _logadd(value, log_state[p, r2 + 1] + neighbor_log)
                after_rate[p, r2] = value

        for p2 in range(positions):
            for r2 in range(rate_count):
                expected_delta = rates[r2] * delta_md[t] - delta_z[t]
                center = int(np.rint(expected_delta / GRID_STEP))
                value = -1.0e30
                for offset in range(-2, 3):
                    p1 = p2 - center - offset
                    if 0 <= p1 < positions:
                        actual_delta = (center + offset) * GRID_STEP
                        transition = -0.5 * (
                            (actual_delta - expected_delta) / POSITION_SIGMA
                        ) ** 2
                        value = _logadd(value, after_rate[p1, r2] + transition)
                current[p2, r2] = value + emission[t, p2]

        maximum = np.max(current)
        total = 0.0
        mean = 0.0
        second = 0.0
        for p in range(positions):
            position_mass = 0.0
            for r in range(rate_count):
                weight = math.exp(current[p, r] - maximum)
                position_mass += weight
                total += weight
            mean += position_mass * grid[p]
            second += position_mass * grid[p] * grid[p]
        mean /= total
        second /= total
        predictions[t] = mean
        uncertainty[t] = math.sqrt(max(second - mean * mean, 0.0))

        normalizer = maximum + math.log(total)
        for p in range(positions):
            for r in range(rate_count):
                log_state[p, r] = current[p, r] - normalizer

    return predictions, uncertainty


def _prepare_typewell(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(tvt) & np.isfinite(gr)
    tvt = tvt[valid]
    gr = gr[valid]
    order = np.argsort(tvt, kind="stable")
    tvt = tvt[order]
    gr = gr[order]
    unique, inverse = np.unique(tvt, return_inverse=True)
    if len(unique) != len(tvt):
        sums = np.bincount(inverse, weights=gr)
        counts = np.bincount(inverse)
        tvt = unique
        gr = sums / counts
    if len(tvt) < 10:
        raise ValueError("typewell has too few finite unique samples")
    return tvt, gr


def predict_hidden_hmm(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(
        limit_direction="both"
    ).to_numpy(dtype=float)
    tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(dtype=float)
    hidden = np.flatnonzero(~np.isfinite(tvt_input))
    if len(hidden) == 0 or hidden[0] == 0 or not np.all(np.diff(hidden) == 1):
        raise ValueError("horizontal well needs one non-empty contiguous hidden suffix")
    first = int(hidden[0])
    start = first - 1
    tw_tvt, tw_gr = _prepare_typewell(typewell)

    visible_tvt = tvt_input[:first]
    visible_gr = gr[:first]
    reference = np.interp(visible_tvt, tw_tvt, tw_gr)
    gr_sigma = float(
        np.clip(np.nanstd(visible_gr - reference), 10.0, 60.0)
    )
    tail_start = max(0, first - 31)
    tail_u = tvt_input[tail_start:first] + z[tail_start:first]
    tail_md = md[tail_start:first]
    valid_step = np.diff(tail_md) > 0.0
    if int(valid_step.sum()) >= 3:
        start_rate = float(np.median(np.diff(tail_u)[valid_step] / np.diff(tail_md)[valid_step]))
    else:
        start_rate = 0.0
    start_rate = float(np.clip(start_rate, -RATE_SPAN, RATE_SPAN))
    start_tvt = float(tvt_input[start])

    lower = max(float(tw_tvt[0] - 20.0), start_tvt - POSITION_BAND)
    upper = min(float(tw_tvt[-1] + 20.0), start_tvt + POSITION_BAND)
    if upper - lower < 10.0:
        lower = start_tvt - POSITION_BAND
        upper = start_tvt + POSITION_BAND
    grid = np.arange(lower, upper + GRID_STEP * 0.5, GRID_STEP, dtype=float)
    reference_grid = np.interp(grid, tw_tvt, tw_gr)
    hidden_gr = gr[hidden]
    emission = -0.5 * np.minimum(
        ((hidden_gr[:, None] - reference_grid[None, :]) / gr_sigma) ** 2,
        600.0,
    )
    previous_md = np.concatenate(([md[start]], md[hidden[:-1]]))
    previous_z = np.concatenate(([z[start]], z[hidden[:-1]]))
    delta_md = np.maximum(md[hidden] - previous_md, 1.0)
    delta_z = z[hidden] - previous_z
    rates = np.linspace(-RATE_SPAN, RATE_SPAN, RATE_STATES, dtype=float)
    prediction, uncertainty = _dip_rate_filter(
        emission.astype(np.float64),
        delta_md.astype(np.float64),
        delta_z.astype(np.float64),
        grid,
        rates,
        start_tvt,
        start_rate,
    )
    return prediction, uncertainty


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def evaluate(data_root: Path, cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    metadata = json.loads(
        (cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8")
    )
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    eligible = [
        (index, str(item["well"]))
        for index, item in enumerate(metadata["per_well"])
        if int(item["folds"]["spatial_cluster_7_seed_20260727"])
        == DISCOVERY_FOLD
    ]
    eligible.sort(
        key=lambda item: hashlib.sha256(item[1].encode("utf-8")).hexdigest()
    )
    selected = eligible[:DISCOVERY_WELLS]
    rows: list[dict[str, Any]] = []
    total_base_sse = 0.0
    total_candidate_sse = 0.0
    total_rows = 0

    for position, (well_index, well) in enumerate(selected, start=1):
        well_started = time.perf_counter()
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
        base_rmse = _rmse(well_truth, well_base)
        candidate_rmse = _rmse(well_truth, candidate)
        base_sse = float(np.sum(np.square(well_truth - well_base)))
        candidate_sse = float(np.sum(np.square(well_truth - candidate)))
        total_base_sse += base_sse
        total_candidate_sse += candidate_sse
        total_rows += len(well_base)
        rows.append(
            {
                "well": well,
                "rows": int(len(well_base)),
                "base_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "delta": candidate_rmse - base_rmse,
                "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
                "hmm_uncertainty_mean": float(np.mean(uncertainty)),
                "runtime_seconds": float(time.perf_counter() - well_started),
            }
        )
        print(
            f"[{position:02d}/{len(selected):02d}] {well} "
            f"delta={candidate_rmse - base_rmse:+.6f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    base_rmse = math.sqrt(total_base_sse / total_rows)
    candidate_rmse = math.sqrt(total_candidate_sse / total_rows)
    pooled_delta = candidate_rmse - base_rmse
    p95_delta = float(
        frame["candidate_rmse"].quantile(0.95)
        - frame["base_rmse"].quantile(0.95)
    )
    worst_delta = float(
        frame["candidate_rmse"].max() - frame["base_rmse"].max()
    )
    gate_checks = {
        "pooled_delta_at_most_minus_0_015": bool(pooled_delta <= -0.015),
        "well_win_fraction_at_least_0_55": bool((frame["delta"] <= 0.0).mean() >= 0.55),
        "p95_not_worse": bool(p95_delta <= 0.0),
        "worst_not_worse_than_plus_0_25": bool(worst_delta <= 0.25),
        "mean_runtime_per_well_at_most_90s": bool(frame["runtime_seconds"].mean() <= 90.0),
    }
    report = {
        "candidate": "target-free dip-rate HMM clipped overlay discovery",
        "contract": {
            "writes_submission": False,
            "discovery_fold": DISCOVERY_FOLD,
            "discovery_selection": "first 32 SHA256-sorted well IDs",
            "discovery_wells": DISCOVERY_WELLS,
            "grid_step": GRID_STEP,
            "rate_states": RATE_STATES,
            "rate_span": RATE_SPAN,
            "position_band": POSITION_BAND,
            "position_sigma": POSITION_SIGMA,
            "rate_stay_probability": RATE_STAY_PROBABILITY,
            "overlay_scale": OVERLAY_SCALE,
            "hmm_disagreement_clip": DISAGREEMENT_CLIP,
            "target_free_runtime_inputs_only": True,
            "public_same_well_and_contact_override_removed": True,
            "public_meta_learner_removed": True,
            "parameter_grid_searched": False,
        },
        "base_rmse": base_rmse,
        "candidate_rmse": candidate_rmse,
        "pooled_delta": pooled_delta,
        "well_win_fraction": float((frame["delta"] <= 0.0).mean()),
        "p95_well_delta": p95_delta,
        "worst_well_delta": worst_delta,
        "mean_runtime_per_well": float(frame["runtime_seconds"].mean()),
        "gate_checks": gate_checks,
        "decision": "PASS" if all(gate_checks.values()) else "FAIL",
        "runtime_seconds": float(time.perf_counter() - started),
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
    report_path = args.output_dir / "dip_rate_hmm_discovery_report.json"
    detail_path = args.output_dir / "dip_rate_hmm_discovery_per_well.csv"
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
