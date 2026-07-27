"""Build the private, leakage-safe cache for the GPU sequence-model pilot.

The cache contains only training suffix rows.  It combines target-free,
spatial-invariant sequence features with the already reconstructed competitive
Ridge/PF OOF trajectory.  Well IDs are retained only in metadata so validation
can keep complete wells together; they are never encoded as model features.

This script writes experiment artifacts only.  It never creates a Kaggle
submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_competitive_oof_overlay import (  # noqa: E402
    EXTERNAL_ROOT,
    _load_public_track,
)
from residual_gbdt_evaluate import _read_inputs, _scan_wells  # noqa: E402
from residual_gbdt_invariant_evaluate import (  # noqa: E402
    INVARIANT_FEATURE_INDICES,
    INVARIANT_FEATURE_NAMES,
)
from residual_gbdt_shrink_evaluate import build_shrink_fold_assignments  # noqa: E402

from rogii.residual_gbdt import build_residual_features  # noqa: E402

FLOAT16_LIMIT = 60_000.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=EXTERNAL_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sequence_pilot_cache_v1",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if len(paths) != 773:
        raise ValueError(f"expected 773 training wells, found {len(paths)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache directory: {output_dir}")

    base_by_well, public_diagnostics = _load_public_track(
        args.external_root.expanduser().resolve()
    )
    records = _scan_wells(paths, train_dir)
    assignments = build_shrink_fold_assignments(records)
    total_rows = int(public_diagnostics["rows"])
    feature_count = len(INVARIANT_FEATURE_INDICES)

    feature_path = output_dir / "features.float16.npy"
    base_path = output_dir / "competitive_base.float32.npy"
    truth_path = output_dir / "truth.float32.npy"
    feature_store = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, feature_count),
    )
    base_store = np.lib.format.open_memmap(
        base_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows,),
    )
    truth_store = np.lib.format.open_memmap(
        truth_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows,),
    )

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    clipped_values = 0
    missing_values = 0
    cursor = 0
    per_well: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        well_id = str(record["well_id"])
        horizontal, typewell, full_truth = _read_inputs(Path(record["path"]), train_dir)
        block = build_residual_features(horizontal, typewell)
        base_rows = base_by_well.get(well_id)
        if base_rows is None:
            raise ValueError(f"{well_id}: competitive OOF trajectory is missing")
        if not np.array_equal(
            base_rows["row_index"],
            block.hidden_indices.astype(np.int32),
        ):
            raise ValueError(f"{well_id}: hidden row mapping does not match competitive OOF")

        features = block.features[:, INVARIANT_FEATURE_INDICES].astype(np.float32, copy=False)
        finite = np.isfinite(features)
        clipped_values += int(np.count_nonzero(finite & (np.abs(features) > FLOAT16_LIMIT)))
        missing_values += int(np.count_nonzero(~finite))
        features = np.where(
            finite,
            np.clip(features, -FLOAT16_LIMIT, FLOAT16_LIMIT),
            np.nan,
        )
        truth = np.asarray(full_truth[block.evaluation_mask], dtype=np.float32)
        base = np.asarray(base_rows["prediction"], dtype=np.float32)
        if len(features) != len(truth) or len(base) != len(truth):
            raise ValueError(f"{well_id}: feature/target/base length mismatch")
        if not np.isfinite(truth).all() or not np.isfinite(base).all():
            raise ValueError(f"{well_id}: truth or competitive base is non-finite")

        stop = cursor + len(truth)
        feature_store[cursor:stop] = features.astype(np.float16)
        base_store[cursor:stop] = base
        truth_store[cursor:stop] = truth
        offsets[index] = cursor
        cursor = stop
        offsets[index + 1] = cursor
        per_well.append(
            {
                "well": well_id,
                "rows": int(len(truth)),
                "typewell_fingerprint": str(record["fingerprint"]),
                "centroid_x": float(record["centroid_x"]),
                "centroid_y": float(record["centroid_y"]),
                "folds": {
                    name: int(values[index])
                    for name, values in assignments.items()
                },
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(
                f"[cache {index + 1:3d}/{len(records)}] rows={cursor:,} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if cursor != total_rows:
        raise AssertionError(f"wrote {cursor} rows, expected {total_rows}")
    feature_store.flush()
    base_store.flush()
    truth_store.flush()
    del feature_store, base_store, truth_store
    np.save(output_dir / "well_offsets.int64.npy", offsets, allow_pickle=False)

    report = {
        "contract": {
            "version": 1,
            "writes_submission": False,
            "training_rows_only": True,
            "well_id_encoded_as_feature": False,
            "absolute_position_features_removed": True,
            "target_feature_leakage": False,
            "feature_dtype": "float16",
            "float16_clip": FLOAT16_LIMIT,
        },
        "rows": total_rows,
        "wells": len(records),
        "features": list(INVARIANT_FEATURE_NAMES),
        "feature_count": feature_count,
        "missing_feature_values": missing_values,
        "clipped_feature_values": clipped_values,
        "competitive_base": public_diagnostics,
        "per_well": per_well,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metadata_path = output_dir / "sequence_cache_metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [
        feature_path,
        base_path,
        truth_path,
        output_dir / "well_offsets.int64.npy",
        metadata_path,
    ]
    manifest = {
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "rows": total_rows,
        "wells": len(records),
        "feature_count": feature_count,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output_dir} rows={total_rows:,} features={feature_count} "
        f"runtime={time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
