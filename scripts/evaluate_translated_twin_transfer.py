"""Audit strict cross-well translated twins against the competitive OOF base.

The query fingerprint uses only columns available for a competition test well.
Absolute MD/XYZ and TVT datum are removed before hashing.  A transfer is
evaluated only between different well IDs and only when the complete
horizontal and typewell fingerprints match.  Unmatched wells remain on the
competitive OOF base, so this script is a read-only discovery audit and never
writes a submission.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "translated_twin_transfer"

QUANTIZATION = 1e-4
MAX_PREFIX_ALIGNMENT_RESIDUAL = 0.02
MINIMUM_ELIGIBLE_WELLS = 5
MINIMUM_GROUPS = 3
MINIMUM_ELIGIBLE_ROWS = 10_000
MINIMUM_POOLED_GAIN = 0.20
MINIMUM_WELL_WIN_FRACTION = 0.80
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


def _quantized_bytes(values: np.ndarray, *, relative: bool) -> bytes:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    output = np.full(array.shape, np.iinfo(np.int64).min, dtype=np.int64)
    if finite.any():
        adjusted = array.copy()
        if relative:
            adjusted[finite] -= adjusted[finite][0]
        output[finite] = np.rint(adjusted[finite] / QUANTIZATION).astype(np.int64)
    return finite.astype(np.uint8).tobytes() + output.tobytes()


def _fingerprint_horizontal(frame: pd.DataFrame) -> str:
    required = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    if not set(required).issubset(frame.columns):
        missing = sorted(set(required) - set(frame.columns))
        raise ValueError(f"horizontal columns missing: {missing}")
    digest = hashlib.sha256()
    digest.update(np.asarray([len(frame)], dtype=np.int64).tobytes())
    for column in ("MD", "X", "Y", "Z"):
        digest.update(_quantized_bytes(frame[column].to_numpy(), relative=True))
    digest.update(_quantized_bytes(frame["GR"].to_numpy(), relative=False))
    digest.update(_quantized_bytes(frame["TVT_input"].to_numpy(), relative=True))
    return digest.hexdigest()


def _fingerprint_typewell(frame: pd.DataFrame) -> str:
    required = ["TVT", "GR"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"typewell columns missing: {sorted(set(required) - set(frame.columns))}")
    digest = hashlib.sha256()
    digest.update(np.asarray([len(frame)], dtype=np.int64).tobytes())
    digest.update(_quantized_bytes(frame["TVT"].to_numpy(), relative=True))
    digest.update(_quantized_bytes(frame["GR"].to_numpy(), relative=False))
    return digest.hexdigest()


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _load_cache(cache_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata = json.loads((cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8"))
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(cache_dir / "well_offsets.int64.npy", mmap_mode="r", allow_pickle=False)
    if len(metadata["per_well"]) + 1 != len(offsets) or int(offsets[-1]) != len(base):
        raise ValueError("competitive cache offsets are inconsistent")
    return metadata, base, offsets


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, base_store, offsets = _load_cache(args.cache_dir.expanduser().resolve())

    records: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in metadata["per_well"]:
        well = str(row["well"])
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        typewell_path = train_dir / f"{well}__typewell.csv"
        horizontal = pd.read_csv(
            horizontal_path,
            usecols=["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input"],
        )
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        hidden = horizontal["TVT_input"].isna().to_numpy()
        if not hidden.any() or not np.isfinite(horizontal.loc[hidden, "TVT"]).all():
            raise ValueError(f"{well}: invalid hidden suffix")
        fingerprint = f"{_fingerprint_horizontal(horizontal)}:{_fingerprint_typewell(typewell)}"
        groups[fingerprint].append(well)
        records[well] = {
            "horizontal": horizontal,
            "fingerprint": fingerprint,
            "hidden": hidden,
        }

    index_by_well = {str(row["well"]): index for index, row in enumerate(metadata["per_well"])}
    details: list[dict[str, Any]] = []
    for fingerprint, wells in sorted(groups.items()):
        if len(wells) < 2:
            continue
        ordered = sorted(wells)
        for query_well in ordered:
            reference_well = next(well for well in ordered if well != query_well)
            query = records[query_well]["horizontal"]
            reference = records[reference_well]["horizontal"]
            hidden = records[query_well]["hidden"]
            visible = ~hidden
            if len(query) != len(reference) or not visible.any():
                continue
            query_visible = query.loc[visible, "TVT_input"].to_numpy(dtype=np.float64)
            reference_visible = reference.loc[visible, "TVT"].to_numpy(dtype=np.float64)
            finite = np.isfinite(query_visible) & np.isfinite(reference_visible)
            if not finite.any():
                continue
            datum = float(np.median(query_visible[finite] - reference_visible[finite]))
            aligned_residual = query_visible[finite] - (reference_visible[finite] + datum)
            maximum_residual = float(np.max(np.abs(aligned_residual)))
            if maximum_residual > MAX_PREFIX_ALIGNMENT_RESIDUAL:
                continue
            transferred = reference.loc[hidden, "TVT"].to_numpy(dtype=np.float64) + datum
            truth = query.loc[hidden, "TVT"].to_numpy(dtype=np.float64)
            well_index = index_by_well[query_well]
            base = np.asarray(
                base_store[int(offsets[well_index]) : int(offsets[well_index + 1])],
                dtype=np.float64,
            )
            if len(base) != len(truth) or not np.isfinite(transferred).all():
                raise ValueError(f"{query_well}: transfer/cache row mismatch")
            base_rmse = _rmse(truth, base)
            transfer_rmse = _rmse(truth, transferred)
            details.append(
                {
                    "fingerprint": fingerprint,
                    "query_well": query_well,
                    "reference_well": reference_well,
                    "hidden_rows": int(len(truth)),
                    "visible_rows": int(finite.sum()),
                    "datum_shift_ft": datum,
                    "maximum_visible_alignment_residual_ft": maximum_residual,
                    "base_rmse": base_rmse,
                    "transfer_rmse": transfer_rmse,
                    "delta": transfer_rmse - base_rmse,
                    "base_sse": float(np.square(truth - base).sum()),
                    "transfer_sse": float(np.square(truth - transferred).sum()),
                }
            )

    detail = pd.DataFrame(details)
    detail_path = output_dir / "translated_twin_transfer_per_well.csv"
    detail.to_csv(detail_path, index=False)
    eligible_wells = int(len(detail))
    eligible_groups = int(detail["fingerprint"].nunique()) if eligible_wells else 0
    eligible_rows = int(detail["hidden_rows"].sum()) if eligible_wells else 0
    if eligible_wells:
        base_rmse = float(np.sqrt(detail["base_sse"].sum() / eligible_rows))
        transfer_rmse = float(np.sqrt(detail["transfer_sse"].sum() / eligible_rows))
        deltas = detail["delta"].to_numpy(dtype=float)
        win_fraction = float(np.mean(deltas < 0.0))
        p95_delta = float(np.quantile(deltas, 0.95))
        worst_delta = float(np.max(deltas))
    else:
        base_rmse = transfer_rmse = win_fraction = p95_delta = worst_delta = math.nan

    checks = {
        "eligible_wells_at_least_5": eligible_wells >= MINIMUM_ELIGIBLE_WELLS,
        "independent_groups_at_least_3": eligible_groups >= MINIMUM_GROUPS,
        "eligible_rows_at_least_10000": eligible_rows >= MINIMUM_ELIGIBLE_ROWS,
        "pooled_gain_at_least_0_20": bool(base_rmse - transfer_rmse >= MINIMUM_POOLED_GAIN),
        "well_win_fraction_at_least_0_80": bool(win_fraction >= MINIMUM_WELL_WIN_FRACTION),
        "p95_delta_at_most_0_10": bool(p95_delta <= MAXIMUM_P95_WELL_DELTA),
        "worst_delta_at_most_0_50": bool(worst_delta <= MAXIMUM_WORST_WELL_DELTA),
    }
    report = {
        "candidate": "translated cross-well twin transfer",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract": {
            "distinct_well_only": True,
            "query_match_uses_targets": False,
            "absolute_coordinates_used": False,
            "fixed_test_well_or_shape_used": False,
            "leaderboard_or_probe_used": False,
            "quantization": QUANTIZATION,
            "maximum_visible_prefix_alignment_residual_ft": MAX_PREFIX_ALIGNMENT_RESIDUAL,
            "unmatched_wells_keep_base": True,
        },
        "all_training_wells": int(len(records)),
        "raw_fingerprint_groups_with_multiple_wells": int(sum(len(v) > 1 for v in groups.values())),
        "eligible_wells": eligible_wells,
        "eligible_fingerprint_groups": eligible_groups,
        "eligible_hidden_rows": eligible_rows,
        "eligible_base_pooled_rmse": base_rmse,
        "eligible_transfer_pooled_rmse": transfer_rmse,
        "eligible_pooled_gain": base_rmse - transfer_rmse,
        "well_win_fraction": win_fraction,
        "p95_well_delta": p95_delta,
        "worst_well_delta": worst_delta,
        "gate_checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "writes_submission": False,
    }
    report_path = output_dir / "translated_twin_transfer_report.json"
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
