"""Evaluate one preregistered depth-affine scale for the frozen dip-rate HMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dip_rate_hmm_confirmation import _discovery_indices  # noqa: E402
from evaluate_dip_rate_hmm_discovery import DEFAULT_DATA_ROOT  # noqa: E402
from evaluate_hmm_scale2_functional_ridge_pls_additive import (  # noqa: E402
    HMM_MULTIPLIER,
    WORKERS,
    _hmm_correction,
)

DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_depth_affine"
PROTOCOLS = (
    "spatial_cluster_7_seed_20260727",
    "typewell_fingerprint",
)
MAX_ENDPOINT_MULTIPLIER = 2.3
FIXED_REFERENCE_MULTIPLIER = 2.0
MAX_FOLD_REGRESSION = 0.025


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rmse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / max(int(rows), 1))


def _test_wells(sample_path: Path) -> set[str]:
    sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
    return set(sample["id"].astype(str).str.rsplit("_", n=1).str[0])


def _quadratic_objective(beta: np.ndarray, xtx: np.ndarray, xtr: np.ndarray) -> float:
    return float(beta @ xtx @ beta - 2.0 * beta @ xtr)


def _bounded_two_coefficient_fit(
    xtx: np.ndarray,
    xtr: np.ndarray,
    upper: float = MAX_ENDPOINT_MULTIPLIER,
) -> np.ndarray:
    """Solve the exact two-variable box-constrained least-squares normal form."""

    matrix = np.asarray(xtx, dtype=float)
    vector = np.asarray(xtr, dtype=float)
    if matrix.shape != (2, 2) or vector.shape != (2,):
        raise ValueError("expected a 2x2 normal matrix and a length-2 vector")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("non-finite least-squares statistics")
    if upper <= 0.0:
        raise ValueError("upper bound must be positive")

    regularized = matrix + np.eye(2, dtype=float) * 1e-12
    candidates: list[np.ndarray] = []
    try:
        unconstrained = np.linalg.solve(regularized, vector)
    except np.linalg.LinAlgError:
        unconstrained = np.linalg.lstsq(regularized, vector, rcond=None)[0]
    if np.all((unconstrained >= 0.0) & (unconstrained <= upper)):
        candidates.append(np.asarray(unconstrained, dtype=float))

    for fixed_index in (0, 1):
        free_index = 1 - fixed_index
        denominator = float(regularized[free_index, free_index])
        for fixed_value in (0.0, float(upper)):
            candidate = np.zeros(2, dtype=float)
            candidate[fixed_index] = fixed_value
            numerator = vector[free_index] - matrix[free_index, fixed_index] * fixed_value
            candidate[free_index] = np.clip(
                numerator / max(denominator, 1e-12), 0.0, upper
            )
            candidates.append(candidate)

    for start in (0.0, float(upper)):
        for end in (0.0, float(upper)):
            candidates.append(np.asarray([start, end], dtype=float))
    return min(candidates, key=lambda beta: _quadratic_objective(beta, matrix, vector))


def _depth_basis(correction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(correction, dtype=float)
    depth = np.linspace(0.0, 1.0, len(values), dtype=float)
    return values * (1.0 - depth), values * depth


def _fit_endpoints(
    indices: np.ndarray,
    corrections: dict[int, np.ndarray],
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    xtx = np.zeros((2, 2), dtype=float)
    xtr = np.zeros(2, dtype=float)
    for index in indices.tolist():
        start = int(offsets[index])
        end = int(offsets[index + 1])
        residual = np.asarray(truth[start:end], dtype=float) - np.asarray(
            base[start:end], dtype=float
        )
        early, late = _depth_basis(corrections[index])
        xtx[0, 0] += float(early @ early)
        xtx[0, 1] += float(early @ late)
        xtx[1, 1] += float(late @ late)
        xtr[0] += float(early @ residual)
        xtr[1] += float(late @ residual)
    xtx[1, 0] = xtx[0, 1]
    return _bounded_two_coefficient_fit(xtx, xtr)


def _summarize_protocol(
    *,
    protocol: str,
    metadata: dict[str, Any],
    eligible: np.ndarray,
    corrections: dict[int, np.ndarray],
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = np.asarray(
        [int(item["folds"][protocol]) for item in metadata["per_well"]], dtype=int
    )
    fold_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    total_rows = 0
    total_base_sse = 0.0
    total_reference_sse = 0.0
    total_candidate_sse = 0.0

    for fold in sorted(np.unique(folds[eligible])):
        train = np.flatnonzero((folds != fold) & eligible)
        validation = np.flatnonzero((folds == fold) & eligible)
        endpoints = _fit_endpoints(train, corrections, base, truth, offsets)
        fold_base_sse = 0.0
        fold_reference_sse = 0.0
        fold_candidate_sse = 0.0
        fold_row_count = 0

        for index in validation.tolist():
            start = int(offsets[index])
            end = int(offsets[index + 1])
            rows = end - start
            well_base = np.asarray(base[start:end], dtype=float)
            well_truth = np.asarray(truth[start:end], dtype=float)
            scale1 = corrections[index]
            depth = np.linspace(0.0, 1.0, rows, dtype=float)
            multiplier = endpoints[0] * (1.0 - depth) + endpoints[1] * depth
            reference = well_base + FIXED_REFERENCE_MULTIPLIER * scale1
            candidate = well_base + multiplier * scale1
            base_sse = float(np.sum(np.square(well_truth - well_base)))
            reference_sse = float(np.sum(np.square(well_truth - reference)))
            candidate_sse = float(np.sum(np.square(well_truth - candidate)))
            fold_base_sse += base_sse
            fold_reference_sse += reference_sse
            fold_candidate_sse += candidate_sse
            fold_row_count += rows
            total_base_sse += base_sse
            total_reference_sse += reference_sse
            total_candidate_sse += candidate_sse
            total_rows += rows
            reference_rmse = _rmse(reference_sse, rows)
            candidate_rmse = _rmse(candidate_sse, rows)
            detail_rows.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "well": str(metadata["per_well"][index]["well"]),
                    "rows": int(rows),
                    "start_multiplier": float(endpoints[0]),
                    "end_multiplier": float(endpoints[1]),
                    "base_rmse": _rmse(base_sse, rows),
                    "hmm_scale2_rmse": reference_rmse,
                    "depth_affine_rmse": candidate_rmse,
                    "delta_vs_hmm_scale2": candidate_rmse - reference_rmse,
                }
            )

        reference_rmse = _rmse(fold_reference_sse, fold_row_count)
        candidate_rmse = _rmse(fold_candidate_sse, fold_row_count)
        fold_rows.append(
            {
                "fold": int(fold),
                "train_wells": int(len(train)),
                "validation_wells": int(len(validation)),
                "rows": int(fold_row_count),
                "start_multiplier": float(endpoints[0]),
                "end_multiplier": float(endpoints[1]),
                "base_rmse": _rmse(fold_base_sse, fold_row_count),
                "hmm_scale2_rmse": reference_rmse,
                "depth_affine_rmse": candidate_rmse,
                "delta_vs_hmm_scale2": candidate_rmse - reference_rmse,
                "delta_vs_base": candidate_rmse - _rmse(fold_base_sse, fold_row_count),
            }
        )

    detail = pd.DataFrame(detail_rows)
    base_rmse = _rmse(total_base_sse, total_rows)
    reference_rmse = _rmse(total_reference_sse, total_rows)
    candidate_rmse = _rmse(total_candidate_sse, total_rows)
    well_delta = detail["delta_vs_hmm_scale2"].to_numpy(dtype=float)
    improved_folds = int(sum(row["delta_vs_hmm_scale2"] < 0.0 for row in fold_rows))
    minimum_gain = 0.01 if protocol.startswith("spatial") else 0.005
    minimum_folds = 6 if protocol.startswith("spatial") else 4
    checks = {
        "additional_pooled_gain_reaches_threshold": bool(
            reference_rmse - candidate_rmse >= minimum_gain
        ),
        "enough_folds_improve_over_scale2": bool(improved_folds >= minimum_folds),
        "maximum_fold_regression_within_limit": bool(
            max(row["delta_vs_hmm_scale2"] for row in fold_rows)
            <= MAX_FOLD_REGRESSION
        ),
        "candidate_improves_base_in_every_fold": bool(
            all(row["delta_vs_base"] < 0.0 for row in fold_rows)
        ),
    }
    return (
        {
            "protocol": protocol,
            "fold_count": int(len(fold_rows)),
            "improved_folds_over_scale2": improved_folds,
            "base_rmse": base_rmse,
            "hmm_scale2_rmse": reference_rmse,
            "depth_affine_rmse": candidate_rmse,
            "hmm_scale2_delta_vs_base": reference_rmse - base_rmse,
            "depth_affine_delta_vs_base": candidate_rmse - base_rmse,
            "additional_gain_over_hmm_scale2": reference_rmse - candidate_rmse,
            "well_win_fraction_vs_hmm_scale2": float(np.mean(well_delta <= 0.0)),
            "true_p95_well_delta_vs_hmm_scale2": float(np.quantile(well_delta, 0.95)),
            "worst_well_delta_vs_hmm_scale2": float(np.max(well_delta)),
            "folds": fold_rows,
            "gate_checks": checks,
            "gate_passed": bool(all(checks.values())),
        },
        detail_rows,
    )


def evaluate(
    data_root: Path,
    cache_dir: Path,
    sample_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = cache_dir / "sequence_cache_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    test_wells = _test_wells(sample_path)
    test_mask = np.asarray(
        [str(item["well"]) in test_wells for item in metadata["per_well"]], dtype=bool
    )
    discovery = set(_discovery_indices(metadata))
    eligible = np.asarray(
        [
            index not in discovery and not test_mask[index]
            for index in range(len(metadata["per_well"]))
        ],
        dtype=bool,
    )
    if int(test_mask.sum()) == 0 or int(eligible.sum()) != 738:
        raise RuntimeError("expected exactly 738 non-discovery, non-test wells")

    items = [
        (index, metadata["per_well"][index])
        for index in np.flatnonzero(eligible).tolist()
    ]

    def worker(item: tuple[int, dict[str, Any]]) -> tuple[int, np.ndarray, float]:
        index, scale2, runtime = _hmm_correction(
            item, data_root=data_root, offsets=offsets, base=base
        )
        return index, scale2 / HMM_MULTIPLIER, runtime

    corrections: dict[int, np.ndarray] = {}
    runtimes: list[float] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for position, (index, correction, runtime) in enumerate(
            executor.map(worker, items), start=1
        ):
            corrections[index] = correction
            runtimes.append(runtime)
            if position % 25 == 0 or position == len(items):
                print(
                    f"[{position:03d}/{len(items):03d}] "
                    f"latest={metadata['per_well'][index]['well']}",
                    flush=True,
                )

    protocols: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        result, detail = _summarize_protocol(
            protocol=protocol,
            metadata=metadata,
            eligible=eligible,
            corrections=corrections,
            base=base,
            truth=truth,
            offsets=offsets,
        )
        protocols[protocol] = result
        detail_rows.extend(detail)

    full_fit = _fit_endpoints(
        np.flatnonzero(eligible), corrections, base, truth, offsets
    )
    report = {
        "candidate": "target-free depth-affine dip-rate HMM scale",
        "contract": {
            "writes_submission": False,
            "evaluation_wells": int(eligible.sum()),
            "hmm_discovery_wells_excluded_from_all_fit_and_validation": len(discovery),
            "current_test_wells_excluded_from_all_fit_and_validation": sorted(test_wells),
            "scale1_overlay": 0.2,
            "fixed_reference_multiplier": FIXED_REFERENCE_MULTIPLIER,
            "maximum_endpoint_multiplier": MAX_ENDPOINT_MULTIPLIER,
            "maximum_row_movement_ft": MAX_ENDPOINT_MULTIPLIER,
            "depth_breakpoint_or_grid_searched": False,
            "outer_fold_truth_isolation": True,
            "well_id_feature_used": False,
            "absolute_coordinates_used": False,
            "leaderboard_score_used": False,
            "probe_flags_used": False,
        },
        "input_sha256": {
            "metadata": _sha256(metadata_path),
            "sample": _sha256(sample_path),
        },
        "full_training_fit": {
            "start_multiplier": float(full_fit[0]),
            "end_multiplier": float(full_fit[1]),
        },
        "protocols": protocols,
        "mean_hmm_runtime_per_well": float(np.mean(runtimes)),
        "wall_runtime_seconds": float(time.perf_counter() - started),
        "decision": (
            "PASS" if all(item["gate_passed"] for item in protocols.values()) else "FAIL"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dip_rate_hmm_depth_affine_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(detail_rows).to_csv(
        output_dir / "dip_rate_hmm_depth_affine_per_well.csv", index=False
    )
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
