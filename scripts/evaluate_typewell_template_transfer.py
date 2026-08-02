"""Evaluate a bounded exact-Typewell nearest-well template correction.

For a query well whose Typewell fingerprint occurs on another training well,
the spatially nearest distinct well supplies a suffix TVT shape.  That shape is
datum-aligned at the last visible TVT, interpolated over normalized suffix
progress, clipped against the competitive OOF trajectory, and applied at one
frozen scale.  Query targets are used only after prediction for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "competition"
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "typewell_template_transfer"

OVERLAY_SCALE = 0.25
TEMPLATE_DELTA_CLIP = 5.0
MINIMUM_ELIGIBLE_WELLS = 20
MINIMUM_GROUPS = 3
MINIMUM_ELIGIBLE_GAIN = 0.10
MINIMUM_FULL_GAIN = 0.005
MINIMUM_WELL_WIN_FRACTION = 0.55
MAXIMUM_P95_WELL_DELTA = 0.10
MAXIMUM_WORST_WELL_DELTA = 0.50


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _interpolate_template(reference_relative: np.ndarray, query_rows: int) -> np.ndarray:
    reference = np.asarray(reference_relative, dtype=np.float64)
    if len(reference) == 0 or query_rows <= 0 or not np.isfinite(reference).all():
        raise ValueError("template inputs must be finite and non-empty")
    if len(reference) == 1:
        return np.full(query_rows, reference[0], dtype=np.float64)
    source_progress = np.linspace(0.0, 1.0, len(reference))
    query_progress = np.linspace(0.0, 1.0, query_rows)
    return np.interp(query_progress, source_progress, reference)


def _load_cache(cache_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    metadata = json.loads((cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8"))
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(cache_dir / "well_offsets.int64.npy", mmap_mode="r", allow_pickle=False)
    if len(metadata["per_well"]) + 1 != len(offsets):
        raise ValueError("cache metadata and offsets disagree")
    if int(offsets[-1]) != len(base) or len(base) != len(truth):
        raise ValueError("cache arrays and offsets disagree")
    return metadata, base, truth, offsets


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    train_dir = args.data_root.expanduser().resolve() / "train"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, base_store, truth_store, offsets = _load_cache(args.cache_dir.expanduser().resolve())

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata["per_well"]):
        groups[str(row["typewell_fingerprint"])].append(index)

    horizontal: dict[int, pd.DataFrame] = {}

    def load_horizontal(index: int) -> pd.DataFrame:
        if index not in horizontal:
            well = str(metadata["per_well"][index]["well"])
            horizontal[index] = pd.read_csv(
                train_dir / f"{well}__horizontal_well.csv",
                usecols=["TVT", "TVT_input"],
            )
        return horizontal[index]

    details: list[dict[str, Any]] = []
    for fingerprint, indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        for query_index in indices:
            query_meta = metadata["per_well"][query_index]
            query_xy = np.asarray(
                [query_meta["centroid_x"], query_meta["centroid_y"]],
                dtype=np.float64,
            )
            candidates = [index for index in indices if index != query_index]
            reference_index = min(
                candidates,
                key=lambda index: (
                    float(
                        np.linalg.norm(
                            query_xy
                            - np.asarray(
                                [
                                    metadata["per_well"][index]["centroid_x"],
                                    metadata["per_well"][index]["centroid_y"],
                                ],
                                dtype=np.float64,
                            )
                        )
                    ),
                    str(metadata["per_well"][index]["well"]),
                ),
            )
            query = load_horizontal(query_index)
            reference = load_horizontal(reference_index)
            query_hidden = query["TVT_input"].isna().to_numpy()
            reference_hidden = reference["TVT_input"].isna().to_numpy()
            query_visible_values = query.loc[~query_hidden, "TVT_input"].to_numpy(dtype=np.float64)
            reference_visible_values = reference.loc[
                ~reference_hidden, "TVT_input"
            ].to_numpy(dtype=np.float64)
            if not len(query_visible_values) or not len(reference_visible_values):
                raise ValueError("every well must expose a finite TVT prefix")
            query_anchor = float(query_visible_values[-1])
            reference_anchor = float(reference_visible_values[-1])
            reference_relative = (
                reference.loc[reference_hidden, "TVT"].to_numpy(dtype=np.float64)
                - reference_anchor
            )
            query_rows = int(query_hidden.sum())
            template = query_anchor + _interpolate_template(reference_relative, query_rows)
            start = int(offsets[query_index])
            stop = int(offsets[query_index + 1])
            base = np.asarray(base_store[start:stop], dtype=np.float64)
            truth = np.asarray(truth_store[start:stop], dtype=np.float64)
            if len(base) != query_rows or not np.isfinite(template).all():
                raise ValueError("template/cache row mismatch")
            correction = OVERLAY_SCALE * np.clip(
                template - base,
                -TEMPLATE_DELTA_CLIP,
                TEMPLATE_DELTA_CLIP,
            )
            prediction = base + correction
            base_rmse = _rmse(truth, base)
            candidate_rmse = _rmse(truth, prediction)
            reference_meta = metadata["per_well"][reference_index]
            distance = float(
                np.linalg.norm(
                    query_xy
                    - np.asarray(
                        [reference_meta["centroid_x"], reference_meta["centroid_y"]],
                        dtype=np.float64,
                    )
                )
            )
            details.append(
                {
                    "typewell_fingerprint": fingerprint,
                    "group_size": int(len(indices)),
                    "query_well": str(query_meta["well"]),
                    "reference_well": str(reference_meta["well"]),
                    "reference_distance_xy": distance,
                    "rows": query_rows,
                    "base_rmse": base_rmse,
                    "candidate_rmse": candidate_rmse,
                    "delta": candidate_rmse - base_rmse,
                    "base_sse": float(np.square(truth - base).sum()),
                    "candidate_sse": float(np.square(truth - prediction).sum()),
                    "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
                    "correction_max_abs": float(np.max(np.abs(correction))),
                }
            )

    detail = pd.DataFrame(details)
    detail_path = output_dir / "typewell_template_transfer_per_well.csv"
    detail.to_csv(detail_path, index=False)
    eligible_rows = int(detail["rows"].sum())
    eligible_base_rmse = float(np.sqrt(detail["base_sse"].sum() / eligible_rows))
    eligible_candidate_rmse = float(
        np.sqrt(detail["candidate_sse"].sum() / eligible_rows)
    )
    all_base_sse = float(np.square(np.asarray(truth_store) - np.asarray(base_store)).sum())
    all_candidate_sse = (
        all_base_sse - float(detail["base_sse"].sum()) + float(detail["candidate_sse"].sum())
    )
    all_rows = int(len(base_store))
    full_base_rmse = float(np.sqrt(all_base_sse / all_rows))
    full_candidate_rmse = float(np.sqrt(all_candidate_sse / all_rows))
    deltas = detail["delta"].to_numpy(dtype=float)
    checks = {
        "eligible_wells_at_least_20": len(detail) >= MINIMUM_ELIGIBLE_WELLS,
        "exact_typewell_groups_at_least_3": detail["typewell_fingerprint"].nunique()
        >= MINIMUM_GROUPS,
        "eligible_pooled_gain_at_least_0_10": eligible_base_rmse - eligible_candidate_rmse
        >= MINIMUM_ELIGIBLE_GAIN,
        "full_population_gain_at_least_0_005": full_base_rmse - full_candidate_rmse
        >= MINIMUM_FULL_GAIN,
        "well_win_fraction_at_least_0_55": float(np.mean(deltas < 0.0))
        >= MINIMUM_WELL_WIN_FRACTION,
        "p95_delta_at_most_0_10": float(np.quantile(deltas, 0.95))
        <= MAXIMUM_P95_WELL_DELTA,
        "worst_delta_at_most_0_50": float(np.max(deltas)) <= MAXIMUM_WORST_WELL_DELTA,
    }
    report = {
        "candidate": "exact-Typewell nearest-well template transfer",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "parameters": {
            "overlay_scale": OVERLAY_SCALE,
            "template_minus_base_clip_ft": TEMPLATE_DELTA_CLIP,
            "reference_selection": "nearest other well by centroid XY",
            "curve_alignment": "normalized suffix progress",
        },
        "contract": {
            "query_target_used_for_prediction": False,
            "different_reference_well_required": True,
            "exact_typewell_fingerprint_required": True,
            "fixed_test_well_or_shape_used": False,
            "leaderboard_or_probe_used": False,
            "unmatched_wells_keep_base": True,
        },
        "eligible_wells": int(len(detail)),
        "eligible_groups": int(detail["typewell_fingerprint"].nunique()),
        "eligible_rows": eligible_rows,
        "eligible_base_pooled_rmse": eligible_base_rmse,
        "eligible_candidate_pooled_rmse": eligible_candidate_rmse,
        "eligible_pooled_gain": eligible_base_rmse - eligible_candidate_rmse,
        "full_population_base_rmse": full_base_rmse,
        "full_population_candidate_rmse": full_candidate_rmse,
        "full_population_gain": full_base_rmse - full_candidate_rmse,
        "well_win_fraction": float(np.mean(deltas < 0.0)),
        "p95_well_delta": float(np.quantile(deltas, 0.95)),
        "worst_well_delta": float(np.max(deltas)),
        "gate_checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "writes_submission": False,
    }
    report_path = output_dir / "typewell_template_transfer_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    print(f"wrote {detail_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
