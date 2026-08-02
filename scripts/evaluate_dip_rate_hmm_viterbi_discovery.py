"""Evaluate one frozen global Viterbi path against forward HMM scale2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dip_rate_hmm_discovery import (  # noqa: E402
    DISAGREEMENT_CLIP,
    GRID_STEP,
    OVERLAY_SCALE,
    POSITION_BAND,
    POSITION_SIGMA,
    RATE_SPAN,
    RATE_STATES,
    RATE_STAY_PROBABILITY,
    _prepare_typewell,
    predict_hidden_hmm,
)

DEFAULT_DATA_ROOT = ROOT / "data" / "competition"
DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_viterbi_discovery"

DISCOVERY_WELLS = 24
SELECTION_SALT = "viterbi_v1:"
WORKERS = 4
SMOOTHER_DISCOVERY_WELLS = {
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
}
EXPECTED_SELECTED_WELLS = (
    "c472c0b5",
    "357c12f9",
    "71642c7d",
    "c07fd0cd",
    "2e29e833",
    "173ed841",
    "da2d4d7c",
    "d9ae59b2",
    "d1ecf309",
    "0dd99dc5",
    "b7974e66",
    "9af1ffd0",
    "1a730ac2",
    "a3518960",
    "e14d641e",
    "3a0af893",
    "9c211c53",
    "eb934281",
    "5567d551",
    "f8f59188",
    "d0a43760",
    "35b30c7f",
    "ef04658e",
    "e5864244",
)


@njit(cache=False, nogil=True)
def _dip_rate_viterbi(
    emission: np.ndarray,
    delta_md: np.ndarray,
    delta_z: np.ndarray,
    grid: np.ndarray,
    rates: np.ndarray,
    start_tvt: float,
    start_rate: float,
) -> np.ndarray:
    """Return the single global max-product TVT path."""
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
    current = np.empty_like(log_state)
    back_offset = np.empty((steps, positions, rate_count), dtype=np.int8)
    back_rate_delta = np.empty((steps, positions, rate_count), dtype=np.int8)

    for t in range(steps):
        for p2 in range(positions):
            for r2 in range(rate_count):
                expected_delta = rates[r2] * delta_md[t] - delta_z[t]
                center = int(np.rint(expected_delta / GRID_STEP))
                best = -1.0e30
                best_offset = 0
                best_rate_delta = 0
                lower_rate = max(0, r2 - 1)
                upper_rate = min(rate_count - 1, r2 + 1)
                for r1 in range(lower_rate, upper_rate + 1):
                    rate_log = stay_log if r1 == r2 else neighbor_log
                    for offset in range(-2, 3):
                        p1 = p2 - center - offset
                        if 0 <= p1 < positions:
                            actual_delta = (center + offset) * GRID_STEP
                            transition = -0.5 * (
                                (actual_delta - expected_delta) / POSITION_SIGMA
                            ) ** 2
                            value = log_state[p1, r1] + rate_log + transition
                            if value > best:
                                best = value
                                best_offset = offset
                                best_rate_delta = r1 - r2
                current[p2, r2] = best + emission[t, p2]
                back_offset[t, p2, r2] = best_offset
                back_rate_delta[t, p2, r2] = best_rate_delta
        maximum = np.max(current)
        for p in range(positions):
            for r in range(rate_count):
                log_state[p, r] = current[p, r] - maximum

    best_position = 0
    best_rate = 0
    best_score = -1.0e30
    for p in range(positions):
        for r in range(rate_count):
            if log_state[p, r] > best_score:
                best_score = log_state[p, r]
                best_position = p
                best_rate = r

    path = np.empty(steps, dtype=np.float64)
    for t in range(steps - 1, -1, -1):
        path[t] = grid[best_position]
        offset = int(back_offset[t, best_position, best_rate])
        previous_rate = best_rate + int(
            back_rate_delta[t, best_position, best_rate]
        )
        expected_delta = rates[best_rate] * delta_md[t] - delta_z[t]
        center = int(np.rint(expected_delta / GRID_STEP))
        best_position = best_position - center - offset
        best_rate = previous_rate
    return path


def predict_hidden_viterbi(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> np.ndarray:
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=float)
    gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(dtype=float)
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
    return _dip_rate_viterbi(
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
    excluded = original_discovery | SMOOTHER_DISCOVERY_WELLS | _test_wells(sample_path)
    eligible = [
        (index, str(item["well"]))
        for index, item in enumerate(metadata["per_well"])
        if str(item["well"]) not in excluded
    ]
    eligible.sort(
        key=lambda item: hashlib.sha256(
            f"{SELECTION_SALT}{item[1]}".encode()
        ).hexdigest()
    )
    selected = eligible[:DISCOVERY_WELLS]
    if tuple(well for _, well in selected) != EXPECTED_SELECTED_WELLS:
        raise RuntimeError("Viterbi discovery selection differs from preregistration")
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
    forward, _ = predict_hidden_hmm(horizontal, typewell)
    viterbi = predict_hidden_viterbi(horizontal, typewell)
    start = int(offsets[well_index])
    end = int(offsets[well_index + 1])
    well_base = np.asarray(base[start:end], dtype=float)
    well_truth = np.asarray(truth[start:end], dtype=float)
    if len(forward) != len(well_base) or len(viterbi) != len(well_base):
        raise RuntimeError(f"{well}: Viterbi/cache row mismatch")
    forward_candidate = well_base + OVERLAY_SCALE * np.clip(
        forward - well_base, -DISAGREEMENT_CLIP, DISAGREEMENT_CLIP
    )
    viterbi_candidate = well_base + OVERLAY_SCALE * np.clip(
        viterbi - well_base, -DISAGREEMENT_CLIP, DISAGREEMENT_CLIP
    )
    base_sse = float(np.sum(np.square(well_truth - well_base)))
    forward_sse = float(np.sum(np.square(well_truth - forward_candidate)))
    viterbi_sse = float(np.sum(np.square(well_truth - viterbi_candidate)))
    rows = len(well_base)
    base_rmse = math.sqrt(base_sse / rows)
    forward_rmse = math.sqrt(forward_sse / rows)
    viterbi_rmse = math.sqrt(viterbi_sse / rows)
    return {
        "well": well,
        "rows": rows,
        "base_sse": base_sse,
        "forward_scale2_sse": forward_sse,
        "viterbi_scale2_sse": viterbi_sse,
        "base_rmse": base_rmse,
        "forward_scale2_rmse": forward_rmse,
        "viterbi_scale2_rmse": viterbi_rmse,
        "viterbi_delta_vs_forward_scale2": viterbi_rmse - forward_rmse,
        "mean_abs_viterbi_minus_forward": float(np.mean(np.abs(viterbi - forward))),
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
                f"delta_vs_forward={result['viterbi_delta_vs_forward_scale2']:+.6f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    total_rows = int(frame["rows"].sum())
    base_rmse = math.sqrt(float(frame["base_sse"].sum()) / total_rows)
    forward_rmse = math.sqrt(float(frame["forward_scale2_sse"].sum()) / total_rows)
    viterbi_rmse = math.sqrt(float(frame["viterbi_scale2_sse"].sum()) / total_rows)
    relative = frame["viterbi_delta_vs_forward_scale2"].to_numpy(dtype=float)
    pooled_gain = forward_rmse - viterbi_rmse
    gate_checks = {
        "pooled_gain_over_forward_scale2_at_least_0_02": pooled_gain >= 0.02,
        "well_win_fraction_at_least_0_60": float(np.mean(relative <= 0.0)) >= 0.60,
        "true_p95_well_delta_at_most_0_10": float(np.quantile(relative, 0.95)) <= 0.10,
        "worst_well_delta_at_most_0_50": float(np.max(relative)) <= 0.50,
        "mean_runtime_per_well_at_most_45s": float(frame["runtime_seconds"].mean())
        <= 45.0,
    }
    report = {
        "candidate": "target-free global Viterbi dip-rate HMM scale2",
        "contract": {
            "selected_wells": list(EXPECTED_SELECTED_WELLS),
            "original_hmm_discovery_wells_excluded": 32,
            "smoother_discovery_wells_excluded": 24,
            "current_test_wells_excluded": sorted(_test_wells(sample_path)),
            "grid_step": GRID_STEP,
            "rate_states": RATE_STATES,
            "rate_span": RATE_SPAN,
            "position_band": POSITION_BAND,
            "position_sigma": POSITION_SIGMA,
            "rate_stay_probability": RATE_STAY_PROBABILITY,
            "overlay_scale": OVERLAY_SCALE,
            "hmm_disagreement_clip": DISAGREEMENT_CLIP,
            "path_objective": "single global max-product path",
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
        "viterbi_scale2_rmse": viterbi_rmse,
        "pooled_gain_over_forward_scale2": pooled_gain,
        "well_win_fraction_vs_forward_scale2": float(np.mean(relative <= 0.0)),
        "true_p95_well_delta_vs_forward_scale2": float(np.quantile(relative, 0.95)),
        "worst_well_delta_vs_forward_scale2": float(np.max(relative)),
        "mean_runtime_per_well_seconds": float(frame["runtime_seconds"].mean()),
        "gate": {
            "decision": "PASS" if all(gate_checks.values()) else "FAIL",
            "checks": gate_checks,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dip_rate_hmm_viterbi_discovery_per_well.csv", index=False)
    (output_dir / "dip_rate_hmm_viterbi_discovery_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = evaluate(
        args.data_root.expanduser().resolve(),
        args.cache_dir.expanduser().resolve(),
        args.sample.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    return 0 if report["gate"]["decision"] in {"PASS", "FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
