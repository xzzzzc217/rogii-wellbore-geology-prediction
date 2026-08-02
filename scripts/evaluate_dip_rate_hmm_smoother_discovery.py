"""Evaluate one frozen bidirectional smoother against the forward HMM scale2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "competition"
DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_smoother_discovery"

DISCOVERY_WELLS = 24
SELECTION_SALT = "smoother_v1:"
GRID_STEP = 0.50
RATE_STATES = 21
RATE_SPAN = 0.10
POSITION_BAND = 60.0
POSITION_SIGMA = 0.25
RATE_STAY_PROBABILITY = 0.96
OVERLAY_SCALE = 0.40
DISAGREEMENT_CLIP = 5.0
WORKERS = 4

EXPECTED_SELECTED_WELLS = (
    "91abc4c7",
    "d114d9c2",
    "a4719920",
    "8648abae",
    "044af7d1",
    "684a6fc1",
    "c03fba65",
    "5aa403d0",
    "b5aa9634",
    "529e88ca",
    "bb682ebd",
    "5cb483ac",
    "75cd5f11",
    "a4cdbb0f",
    "e4726608",
    "230eaaa3",
    "e03b45fd",
    "699ea575",
    "20096916",
    "0bbf5e67",
    "2fd68f7b",
    "55efee7f",
    "86454a6f",
    "c908edd0",
)


@njit(cache=False)
def _logadd(current: float, value: float) -> float:
    if current < -1.0e20:
        return value
    if value > current:
        return value + math.log1p(math.exp(current - value))
    return current + math.log1p(math.exp(value - current))


@njit(cache=False, nogil=True)
def _dip_rate_filter_and_smoother(
    emission: np.ndarray,
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    grid: np.ndarray,
    rates: np.ndarray,
    start_tvt: float,
    start_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward-filter and fixed-interval smoothed TVT means."""
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
    alpha = np.empty((steps, positions, rate_count), dtype=np.float32)
    forward = np.empty(steps, dtype=np.float64)
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
        for p in range(positions):
            position_mass = 0.0
            for r in range(rate_count):
                weight = math.exp(current[p, r] - maximum)
                position_mass += weight
                total += weight
            mean += position_mass * grid[p]
        forward[t] = mean / total
        normalizer = maximum + math.log(total)
        for p in range(positions):
            for r in range(rate_count):
                value = current[p, r] - normalizer
                log_state[p, r] = value
                alpha[t, p, r] = value

    smoothed = np.empty(steps, dtype=np.float64)
    beta_next = np.zeros((positions, rate_count), dtype=np.float64)
    beta_current = np.empty_like(beta_next)
    for t in range(steps - 1, -1, -1):
        if t < steps - 1:
            next_t = t + 1
            for p1 in range(positions):
                for r1 in range(rate_count):
                    value = -1.0e30
                    lower_rate = max(0, r1 - 1)
                    upper_rate = min(rate_count - 1, r1 + 1)
                    for r2 in range(lower_rate, upper_rate + 1):
                        rate_log = stay_log if r2 == r1 else neighbor_log
                        expected_delta = rates[r2] * delta_md[next_t] - delta_z[next_t]
                        center = int(np.rint(expected_delta / GRID_STEP))
                        for offset in range(-2, 3):
                            p2 = p1 + center + offset
                            if 0 <= p2 < positions:
                                actual_delta = (center + offset) * GRID_STEP
                                transition = -0.5 * (
                                    (actual_delta - expected_delta) / POSITION_SIGMA
                                ) ** 2
                                value = _logadd(
                                    value,
                                    rate_log
                                    + transition
                                    + emission[next_t, p2]
                                    + beta_next[p2, r2],
                                )
                    beta_current[p1, r1] = value
            beta_maximum = np.max(beta_current)
            for p in range(positions):
                for r in range(rate_count):
                    beta_next[p, r] = beta_current[p, r] - beta_maximum

        posterior_maximum = -1.0e30
        for p in range(positions):
            for r in range(rate_count):
                value = float(alpha[t, p, r]) + beta_next[p, r]
                if value > posterior_maximum:
                    posterior_maximum = value
        total = 0.0
        mean = 0.0
        for p in range(positions):
            position_mass = 0.0
            for r in range(rate_count):
                position_mass += math.exp(
                    float(alpha[t, p, r]) + beta_next[p, r] - posterior_maximum
                )
            total += position_mass
            mean += position_mass * grid[p]
        smoothed[t] = mean / total

    return forward, smoothed


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
        tvt = unique
        gr = np.bincount(inverse, weights=gr) / np.bincount(inverse)
    if len(tvt) < 10:
        raise ValueError("typewell has too few finite unique samples")
    return tvt, gr


def predict_hidden_filter_and_smoother(
    horizontal: pd.DataFrame, typewell: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=float)
    gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=float
    )
    hidden = np.flatnonzero(~np.isfinite(tvt_input))
    if len(hidden) == 0 or hidden[0] == 0 or not np.all(np.diff(hidden) == 1):
        raise ValueError("horizontal well needs one non-empty contiguous hidden suffix")
    first = int(hidden[0])
    start = first - 1
    tw_tvt, tw_gr = _prepare_typewell(typewell)
    reference = np.interp(tvt_input[:first], tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(gr[:first] - reference), 10.0, 60.0))
    tail_start = max(0, first - 31)
    tail_u = tvt_input[tail_start:first] + z[tail_start:first]
    tail_md = md[tail_start:first]
    valid_step = np.diff(tail_md) > 0.0
    if int(valid_step.sum()) >= 3:
        start_rate = float(
            np.median(np.diff(tail_u)[valid_step] / np.diff(tail_md)[valid_step])
        )
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
    emission = -0.5 * np.minimum(
        ((gr[hidden, None] - reference_grid[None, :]) / gr_sigma) ** 2,
        600.0,
    )
    previous_md = np.concatenate(([md[start]], md[hidden[:-1]]))
    previous_z = np.concatenate(([z[start]], z[hidden[:-1]]))
    delta_md = np.maximum(md[hidden] - previous_md, 1.0)
    delta_z = z[hidden] - previous_z
    rates = np.linspace(-RATE_SPAN, RATE_SPAN, RATE_STATES, dtype=float)
    return _dip_rate_filter_and_smoother(
        emission.astype(np.float64),
        delta_md.astype(np.float64),
        delta_z.astype(np.float64),
        grid,
        rates,
        start_tvt,
        start_rate,
    )


def _test_wells(sample_path: Path) -> set[str]:
    sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
    return set(sample["id"].astype(str).str.rsplit("_", n=1).str[0])


def _selection(metadata: dict[str, Any], sample_path: Path) -> list[tuple[int, str]]:
    fold0 = [
        str(item["well"])
        for item in metadata["per_well"]
        if int(item["folds"]["spatial_cluster_7_seed_20260727"]) == 0
    ]
    original_discovery = set(
        sorted(fold0, key=lambda well: hashlib.sha256(well.encode()).hexdigest())[:32]
    )
    test_wells = _test_wells(sample_path)
    eligible = [
        (index, str(item["well"]))
        for index, item in enumerate(metadata["per_well"])
        if str(item["well"]) not in original_discovery | test_wells
    ]
    eligible.sort(
        key=lambda item: hashlib.sha256(
            f"{SELECTION_SALT}{item[1]}".encode()
        ).hexdigest()
    )
    selected = eligible[:DISCOVERY_WELLS]
    if tuple(well for _, well in selected) != EXPECTED_SELECTED_WELLS:
        raise RuntimeError("smoother discovery selection differs from preregistration")
    return selected


def _evaluate_one(
    item: tuple[int, str],
    data_root: Path,
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
) -> dict[str, Any]:
    well_index, well = item
    started = time.perf_counter()
    horizontal = pd.read_csv(data_root / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(data_root / "train" / f"{well}__typewell.csv")
    forward, smoother = predict_hidden_filter_and_smoother(horizontal, typewell)
    start = int(offsets[well_index])
    end = int(offsets[well_index + 1])
    well_base = np.asarray(base[start:end], dtype=float)
    well_truth = np.asarray(truth[start:end], dtype=float)
    if len(forward) != len(well_base) or len(smoother) != len(well_base):
        raise RuntimeError(f"{well}: smoother/cache row mismatch")
    forward_candidate = well_base + OVERLAY_SCALE * np.clip(
        forward - well_base, -DISAGREEMENT_CLIP, DISAGREEMENT_CLIP
    )
    smoother_candidate = well_base + OVERLAY_SCALE * np.clip(
        smoother - well_base, -DISAGREEMENT_CLIP, DISAGREEMENT_CLIP
    )
    base_sse = float(np.sum(np.square(well_truth - well_base)))
    forward_sse = float(np.sum(np.square(well_truth - forward_candidate)))
    smoother_sse = float(np.sum(np.square(well_truth - smoother_candidate)))
    rows = len(well_base)
    base_rmse = math.sqrt(base_sse / rows)
    forward_rmse = math.sqrt(forward_sse / rows)
    smoother_rmse = math.sqrt(smoother_sse / rows)
    return {
        "well": well,
        "rows": rows,
        "base_sse": base_sse,
        "forward_scale2_sse": forward_sse,
        "smoother_scale2_sse": smoother_sse,
        "base_rmse": base_rmse,
        "forward_scale2_rmse": forward_rmse,
        "smoother_scale2_rmse": smoother_rmse,
        "smoother_delta_vs_forward_scale2": smoother_rmse - forward_rmse,
        "mean_abs_smoother_minus_forward": float(np.mean(np.abs(smoother - forward))),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def evaluate(
    data_root: Path, cache_dir: Path, sample_path: Path, output_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = json.loads(
        (cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8")
    )
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    selected = _selection(metadata, sample_path)

    rows = [_evaluate_one(selected[0], data_root, base, truth, offsets)]
    print(f"[01/{len(selected):02d}] {selected[0][1]}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_evaluate_one, item, data_root, base, truth, offsets): item
            for item in selected[1:]
        }
        completed = 1
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            completed += 1
            print(
                f"[{completed:02d}/{len(selected):02d}] {result['well']} "
                f"delta_vs_forward={result['smoother_delta_vs_forward_scale2']:+.6f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    total_rows = int(frame["rows"].sum())
    base_rmse = math.sqrt(float(frame["base_sse"].sum()) / total_rows)
    forward_rmse = math.sqrt(float(frame["forward_scale2_sse"].sum()) / total_rows)
    smoother_rmse = math.sqrt(float(frame["smoother_scale2_sse"].sum()) / total_rows)
    relative = frame["smoother_delta_vs_forward_scale2"].to_numpy(dtype=float)
    pooled_gain = forward_rmse - smoother_rmse
    gate_checks = {
        "pooled_gain_over_forward_scale2_at_least_0_02": pooled_gain >= 0.02,
        "well_win_fraction_at_least_0_60": float(np.mean(relative <= 0.0)) >= 0.60,
        "true_p95_well_delta_at_most_0_10": float(np.quantile(relative, 0.95)) <= 0.10,
        "worst_well_delta_at_most_0_50": float(np.max(relative)) <= 0.50,
        "mean_runtime_per_well_at_most_45s": float(frame["runtime_seconds"].mean())
        <= 45.0,
    }
    report = {
        "candidate": "target-free bidirectional dip-rate HMM smoother scale2",
        "contract": {
            "selected_wells": list(EXPECTED_SELECTED_WELLS),
            "original_hmm_discovery_wells_excluded": 32,
            "current_test_wells_excluded": sorted(_test_wells(sample_path)),
            "grid_step": GRID_STEP,
            "rate_states": RATE_STATES,
            "rate_span": RATE_SPAN,
            "position_band": POSITION_BAND,
            "position_sigma": POSITION_SIGMA,
            "rate_stay_probability": RATE_STAY_PROBABILITY,
            "overlay_scale": OVERLAY_SCALE,
            "hmm_disagreement_clip": DISAGREEMENT_CLIP,
            "target_free_runtime_inputs_only": True,
            "full_hidden_suffix_gr_used": True,
            "training_or_hidden_truth_used_for_prediction": False,
            "leaderboard_score_used": False,
            "probe_flags_used": False,
            "parameter_grid_searched": False,
            "writes_submission": False,
        },
        "wells": len(frame),
        "rows": total_rows,
        "base_rmse": base_rmse,
        "forward_scale2_rmse": forward_rmse,
        "smoother_scale2_rmse": smoother_rmse,
        "forward_scale2_delta_vs_base": forward_rmse - base_rmse,
        "smoother_scale2_delta_vs_base": smoother_rmse - base_rmse,
        "pooled_gain_over_forward_scale2": pooled_gain,
        "well_win_fraction_vs_forward_scale2": float(np.mean(relative <= 0.0)),
        "true_p95_well_delta_vs_forward_scale2": float(np.quantile(relative, 0.95)),
        "worst_well_delta_vs_forward_scale2": float(np.max(relative)),
        "mean_runtime_per_well": float(frame["runtime_seconds"].mean()),
        "mean_abs_smoother_minus_forward": float(
            frame["mean_abs_smoother_minus_forward"].mean()
        ),
        "gate_checks": gate_checks,
        "decision": "PASS" if all(gate_checks.values()) else "FAIL",
        "wall_runtime_seconds": float(time.perf_counter() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dip_rate_hmm_smoother_discovery_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frame.to_csv(output_dir / "dip_rate_hmm_smoother_discovery_per_well.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(
        args.data_root.resolve(),
        args.cache_dir.resolve(),
        args.sample.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
