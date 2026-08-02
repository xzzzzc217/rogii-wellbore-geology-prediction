"""Evaluate a parameter-free uncertainty fusion of forward and Viterbi HMM paths."""

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

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dip_rate_hmm_discovery import (  # noqa: E402
    DISAGREEMENT_CLIP,
    predict_hidden_hmm,
)
from evaluate_dip_rate_hmm_smoother_discovery import (  # noqa: E402
    EXPECTED_SELECTED_WELLS as SMOOTHER_DISCOVERY_WELLS,
)
from evaluate_dip_rate_hmm_viterbi_discovery import (  # noqa: E402
    EXPECTED_SELECTED_WELLS as VITERBI_DISCOVERY_WELLS,
)
from evaluate_dip_rate_hmm_viterbi_discovery import (  # noqa: E402
    predict_hidden_viterbi,
)

DEFAULT_DATA_ROOT = ROOT / "data" / "competition"
DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_viterbi_confidence_discovery"

DISCOVERY_WELLS = 32
SELECTION_SALT = "viterbi_confidence_v1:"
OVERLAY_SCALE = 0.40
WORKERS = 4
EXPECTED_SELECTED_WELLS = (
    "d41beaf2",
    "283269ac",
    "d90aa14c",
    "8b95d6d1",
    "995ff498",
    "f40cd091",
    "53f23031",
    "2c0c4a4e",
    "39b51819",
    "4cd1ac61",
    "a783cc24",
    "a3f71395",
    "47222616",
    "521a7819",
    "84815d5a",
    "53134831",
    "25050f63",
    "5def1ce5",
    "a645da9a",
    "fb03ae90",
    "877ed19c",
    "200d233d",
    "7b38844c",
    "511e1db0",
    "69e41fa6",
    "e97032ad",
    "3a7dd95d",
    "a0e92ed6",
    "d6c320d9",
    "656a067b",
    "57f05c51",
    "eafb863e",
)


def fuse_paths(
    forward: np.ndarray,
    uncertainty: np.ndarray,
    viterbi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forward_values = np.asarray(forward, dtype=np.float64)
    uncertainty_values = np.asarray(uncertainty, dtype=np.float64)
    viterbi_values = np.asarray(viterbi, dtype=np.float64)
    if not (
        forward_values.shape == uncertainty_values.shape == viterbi_values.shape
    ):
        raise ValueError("forward, uncertainty, and Viterbi paths must align")
    disagreement = viterbi_values - forward_values
    variance = np.square(np.maximum(uncertainty_values, 0.0))
    weight = variance / (variance + np.square(disagreement) + 1.0e-12)
    fused = forward_values + weight * disagreement
    if not np.isfinite(fused).all() or not np.isfinite(weight).all():
        raise ValueError("confidence fusion produced non-finite values")
    return fused, weight


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
    excluded = (
        original_discovery
        | set(SMOOTHER_DISCOVERY_WELLS)
        | set(VITERBI_DISCOVERY_WELLS)
        | _test_wells(sample_path)
    )
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
        raise RuntimeError("confidence-fusion selection differs from preregistration")
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
    forward, uncertainty = predict_hidden_hmm(horizontal, typewell)
    viterbi = predict_hidden_viterbi(horizontal, typewell)
    fused, weight = fuse_paths(forward, uncertainty, viterbi)
    start = int(offsets[well_index])
    end = int(offsets[well_index + 1])
    well_base = np.asarray(base[start:end], dtype=float)
    well_truth = np.asarray(truth[start:end], dtype=float)
    if len(fused) != len(well_base):
        raise RuntimeError(f"{well}: confidence-fusion/cache row mismatch")
    forward_candidate = well_base + OVERLAY_SCALE * np.clip(
        forward - well_base, -DISAGREEMENT_CLIP, DISAGREEMENT_CLIP
    )
    fused_candidate = well_base + OVERLAY_SCALE * np.clip(
        fused - well_base, -DISAGREEMENT_CLIP, DISAGREEMENT_CLIP
    )
    base_sse = float(np.sum(np.square(well_truth - well_base)))
    forward_sse = float(np.sum(np.square(well_truth - forward_candidate)))
    fused_sse = float(np.sum(np.square(well_truth - fused_candidate)))
    rows = len(well_base)
    base_rmse = math.sqrt(base_sse / rows)
    forward_rmse = math.sqrt(forward_sse / rows)
    fused_rmse = math.sqrt(fused_sse / rows)
    return {
        "well": well,
        "rows": rows,
        "base_sse": base_sse,
        "forward_scale2_sse": forward_sse,
        "fused_scale2_sse": fused_sse,
        "base_rmse": base_rmse,
        "forward_scale2_rmse": forward_rmse,
        "fused_scale2_rmse": fused_rmse,
        "fused_delta_vs_forward_scale2": fused_rmse - forward_rmse,
        "mean_fusion_weight": float(np.mean(weight)),
        "p90_fusion_weight": float(np.quantile(weight, 0.90)),
        "mean_abs_fused_minus_forward": float(np.mean(np.abs(fused - forward))),
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
                f"delta={result['fused_delta_vs_forward_scale2']:+.6f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    total_rows = int(frame["rows"].sum())
    base_rmse = math.sqrt(float(frame["base_sse"].sum()) / total_rows)
    forward_rmse = math.sqrt(float(frame["forward_scale2_sse"].sum()) / total_rows)
    fused_rmse = math.sqrt(float(frame["fused_scale2_sse"].sum()) / total_rows)
    relative = frame["fused_delta_vs_forward_scale2"].to_numpy(dtype=float)
    pooled_gain = forward_rmse - fused_rmse
    gate_checks = {
        "pooled_gain_over_forward_scale2_at_least_0_02": pooled_gain >= 0.02,
        "well_win_fraction_at_least_0_55": float(np.mean(relative <= 0.0)) >= 0.55,
        "true_p95_well_delta_at_most_0_10": float(np.quantile(relative, 0.95)) <= 0.10,
        "worst_well_delta_at_most_0_50": float(np.max(relative)) <= 0.50,
        "mean_runtime_per_well_at_most_45s": float(frame["runtime_seconds"].mean())
        <= 45.0,
    }
    report = {
        "candidate": "target-free uncertainty-fused Viterbi HMM scale2",
        "contract": {
            "selected_wells": list(EXPECTED_SELECTED_WELLS),
            "excluded_previous_hmm_discovery_wells": 80,
            "current_test_wells_excluded": sorted(_test_wells(sample_path)),
            "fusion_weight": "u^2 / (u^2 + (v-f)^2 + 1e-12)",
            "learned_parameters_or_thresholds": False,
            "overlay_scale": OVERLAY_SCALE,
            "hmm_disagreement_clip": DISAGREEMENT_CLIP,
            "target_free_runtime_inputs_only": True,
            "full_hidden_suffix_gr_used": True,
            "training_or_hidden_truth_used_for_prediction": False,
            "leaderboard_score_used": False,
            "probe_flags_used": False,
            "writes_submission": False,
        },
        "wells": len(frame),
        "rows": total_rows,
        "base_rmse": base_rmse,
        "forward_scale2_rmse": forward_rmse,
        "fused_scale2_rmse": fused_rmse,
        "pooled_gain_over_forward_scale2": pooled_gain,
        "well_win_fraction_vs_forward_scale2": float(np.mean(relative <= 0.0)),
        "true_p95_well_delta_vs_forward_scale2": float(np.quantile(relative, 0.95)),
        "worst_well_delta_vs_forward_scale2": float(np.max(relative)),
        "mean_fusion_weight": float(frame["mean_fusion_weight"].mean()),
        "mean_runtime_per_well_seconds": float(frame["runtime_seconds"].mean()),
        "gate": {
            "decision": "PASS" if all(gate_checks.values()) else "FAIL",
            "checks": gate_checks,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_dir / "dip_rate_hmm_viterbi_confidence_discovery_per_well.csv",
        index=False,
    )
    (output_dir / "dip_rate_hmm_viterbi_confidence_discovery_report.json").write_text(
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
