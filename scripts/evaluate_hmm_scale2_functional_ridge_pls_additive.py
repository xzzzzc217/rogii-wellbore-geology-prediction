"""Evaluate a frozen additive HMM scale2 and Functional Ridge/PLS overlay."""

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
from evaluate_dip_rate_hmm_confirmation import _discovery_indices
from evaluate_dip_rate_hmm_discovery import (
    DEFAULT_DATA_ROOT,
    DISAGREEMENT_CLIP,
    predict_hidden_hmm,
)
from evaluate_dip_rate_hmm_discovery import (
    OVERLAY_SCALE as HMM_SCALE1,
)
from evaluate_functional_ridge import CURVE_POINTS, RESIDUAL_CLIP
from evaluate_functional_ridge_augmented import _augmented_signature
from evaluate_functional_ridge_pls_ensemble import (
    PLS_WEIGHT,
    RIDGE_WEIGHT,
)
from evaluate_functional_ridge_pls_ensemble import (
    _model as functional_model,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hmm_scale2_functional_ridge_pls_additive"
PROTOCOLS = (
    "spatial_cluster_7_seed_20260727",
    "typewell_fingerprint",
)
HMM_MULTIPLIER = 2.0
FUNCTIONAL_OVERLAY_SCALE = 0.20
FUNCTIONAL_ADDITIVE_WEIGHT = 1.0
WORKERS = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rmse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / max(int(rows), 1))


def _test_wells(sample_path: Path) -> set[str]:
    sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
    return set(sample["id"].astype(str).str.rsplit("_", n=1).str[0])


def _curve(residual: np.ndarray) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    clipped = np.clip(values, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return np.interp(
        np.linspace(0.0, 1.0, CURVE_POINTS),
        np.linspace(0.0, 1.0, len(clipped)),
        clipped,
    ).astype(np.float32)


def _hmm_correction(
    item: tuple[int, dict[str, Any]],
    *,
    data_root: Path,
    offsets: np.ndarray,
    base: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    index, metadata = item
    well = str(metadata["well"])
    started = time.perf_counter()
    horizontal = pd.read_csv(data_root / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(data_root / "train" / f"{well}__typewell.csv")
    hmm_prediction, _ = predict_hidden_hmm(horizontal, typewell)
    start = int(offsets[index])
    end = int(offsets[index + 1])
    well_base = np.asarray(base[start:end], dtype=float)
    if len(hmm_prediction) != len(well_base):
        raise RuntimeError(f"{well}: HMM/cache row mismatch")
    scale1 = HMM_SCALE1 * np.clip(
        hmm_prediction - well_base,
        -DISAGREEMENT_CLIP,
        DISAGREEMENT_CLIP,
    )
    return index, HMM_MULTIPLIER * scale1, float(time.perf_counter() - started)


def _summarize_protocol(
    *,
    protocol: str,
    metadata: dict[str, Any],
    eligible: np.ndarray,
    excluded_test: np.ndarray,
    signatures: np.ndarray,
    curves: np.ndarray,
    hmm_corrections: dict[int, np.ndarray],
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = np.asarray(
        [int(item["folds"][protocol]) for item in metadata["per_well"]], dtype=int
    )
    fold_rows: list[dict[str, Any]] = []
    per_well: list[dict[str, Any]] = []
    total_hmm_sse = 0.0
    total_additive_sse = 0.0
    total_base_sse = 0.0
    total_rows = 0
    for fold in sorted(np.unique(folds[eligible])):
        train = np.flatnonzero((folds != fold) & ~excluded_test)
        validation = np.flatnonzero((folds == fold) & eligible)
        model = functional_model()
        model.fit(signatures[train], curves[train])
        predictions = np.asarray(model.predict(signatures[validation]), dtype=float)
        fold_base_sse = 0.0
        fold_hmm_sse = 0.0
        fold_additive_sse = 0.0
        fold_row_count = 0
        for position, index in enumerate(validation.tolist()):
            start = int(offsets[index])
            end = int(offsets[index + 1])
            rows = end - start
            x = np.linspace(0.0, 1.0, rows)
            functional = np.interp(
                x,
                np.linspace(0.0, 1.0, CURVE_POINTS),
                predictions[position],
            )
            functional = FUNCTIONAL_OVERLAY_SCALE * np.clip(
                functional, -RESIDUAL_CLIP, RESIDUAL_CLIP
            )
            well_base = np.asarray(base[start:end], dtype=float)
            well_truth = np.asarray(truth[start:end], dtype=float)
            hmm = well_base + hmm_corrections[index]
            additive = hmm + FUNCTIONAL_ADDITIVE_WEIGHT * functional
            base_sse = float(np.sum(np.square(well_truth - well_base)))
            hmm_sse = float(np.sum(np.square(well_truth - hmm)))
            additive_sse = float(np.sum(np.square(well_truth - additive)))
            fold_base_sse += base_sse
            fold_hmm_sse += hmm_sse
            fold_additive_sse += additive_sse
            fold_row_count += rows
            total_base_sse += base_sse
            total_hmm_sse += hmm_sse
            total_additive_sse += additive_sse
            total_rows += rows
            hmm_rmse = _rmse(hmm_sse, rows)
            additive_rmse = _rmse(additive_sse, rows)
            per_well.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "well": str(metadata["per_well"][index]["well"]),
                    "rows": int(rows),
                    "base_rmse": _rmse(base_sse, rows),
                    "hmm_scale2_rmse": hmm_rmse,
                    "additive_rmse": additive_rmse,
                    "delta_vs_hmm_scale2": additive_rmse - hmm_rmse,
                    "hmm_correction_rms": float(
                        np.sqrt(np.mean(np.square(hmm_corrections[index])))
                    ),
                    "functional_correction_rms": float(
                        np.sqrt(np.mean(np.square(functional)))
                    ),
                }
            )
        fold_hmm_rmse = _rmse(fold_hmm_sse, fold_row_count)
        fold_additive_rmse = _rmse(fold_additive_sse, fold_row_count)
        fold_rows.append(
            {
                "fold": int(fold),
                "train_wells": int(len(train)),
                "validation_wells": int(len(validation)),
                "rows": int(fold_row_count),
                "base_rmse": _rmse(fold_base_sse, fold_row_count),
                "hmm_scale2_rmse": fold_hmm_rmse,
                "additive_rmse": fold_additive_rmse,
                "delta_vs_hmm_scale2": fold_additive_rmse - fold_hmm_rmse,
            }
        )
    detail = pd.DataFrame(per_well)
    base_rmse = _rmse(total_base_sse, total_rows)
    hmm_rmse = _rmse(total_hmm_sse, total_rows)
    additive_rmse = _rmse(total_additive_sse, total_rows)
    well_delta = detail["delta_vs_hmm_scale2"].to_numpy(dtype=float)
    improved_folds = int(sum(row["delta_vs_hmm_scale2"] < 0.0 for row in fold_rows))
    required_folds = 5
    minimum_gain = 0.01 if protocol.startswith("spatial") else 0.005
    checks = {
        "additional_pooled_gain_reaches_threshold": bool(
            hmm_rmse - additive_rmse >= minimum_gain
        ),
        "enough_folds_improve_over_hmm_scale2": bool(improved_folds >= required_folds),
    }
    return (
        {
            "protocol": protocol,
            "fold_count": int(len(fold_rows)),
            "improved_folds_over_hmm_scale2": improved_folds,
            "base_rmse": base_rmse,
            "hmm_scale2_rmse": hmm_rmse,
            "additive_rmse": additive_rmse,
            "hmm_scale2_delta_vs_base": hmm_rmse - base_rmse,
            "additive_delta_vs_base": additive_rmse - base_rmse,
            "additional_gain_over_hmm_scale2": hmm_rmse - additive_rmse,
            "well_win_fraction_vs_hmm_scale2": float(np.mean(well_delta <= 0.0)),
            "true_p95_well_delta_vs_hmm_scale2": float(np.quantile(well_delta, 0.95)),
            "worst_well_delta_vs_hmm_scale2": float(np.max(well_delta)),
            "folds": fold_rows,
            "gate_checks": checks,
            "gate_passed": bool(all(checks.values())),
        },
        per_well,
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
    features = np.load(cache_dir / "features.float16.npy", mmap_mode="r")
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    well_count = len(metadata["per_well"])
    test_wells = _test_wells(sample_path)
    test_mask = np.asarray(
        [str(item["well"]) in test_wells for item in metadata["per_well"]], dtype=bool
    )
    discovery = _discovery_indices(metadata)
    eligible = np.asarray(
        [index not in discovery and not test_mask[index] for index in range(well_count)],
        dtype=bool,
    )
    if int(test_mask.sum()) == 0 or int(eligible.sum()) != 738:
        raise RuntimeError("expected exactly 738 non-discovery, non-test confirmation wells")

    signature_rows = []
    curves = np.empty((well_count, CURVE_POINTS), dtype=np.float32)
    for index in range(well_count):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        signature_rows.append(
            _augmented_signature(np.asarray(features[start:end], dtype=np.float32))
        )
        curves[index] = _curve(
            np.asarray(truth[start:end], dtype=float)
            - np.asarray(base[start:end], dtype=float)
        )
    signatures = np.stack(signature_rows).astype(np.float32, copy=False)

    items = [
        (index, metadata["per_well"][index]) for index in np.flatnonzero(eligible).tolist()
    ]

    def worker(item: tuple[int, dict[str, Any]]) -> tuple[int, np.ndarray, float]:
        return _hmm_correction(item, data_root=data_root, offsets=offsets, base=base)

    corrections: dict[int, np.ndarray] = {}
    runtimes = []
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
            excluded_test=test_mask,
            signatures=signatures,
            curves=curves,
            hmm_corrections=corrections,
            base=base,
            truth=truth,
            offsets=offsets,
        )
        protocols[protocol] = result
        detail_rows.extend(detail)

    report = {
        "candidate": "target-free HMM scale2 plus fixed Functional Ridge-PLS additive overlay",
        "contract": {
            "writes_submission": False,
            "evaluation_wells": int(eligible.sum()),
            "hmm_discovery_wells_excluded_from_validation": len(discovery),
            "current_test_wells_excluded_from_training_and_validation": sorted(test_wells),
            "hmm_effective_overlay_scale": 0.4,
            "hmm_maximum_movement_ft": 2.0,
            "functional_ridge_weight": RIDGE_WEIGHT,
            "functional_pls_weight": PLS_WEIGHT,
            "functional_overlay_scale": FUNCTIONAL_OVERLAY_SCALE,
            "functional_residual_clip_ft": RESIDUAL_CLIP,
            "functional_additive_weight": FUNCTIONAL_ADDITIVE_WEIGHT,
            "weight_grid_searched": False,
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
        "protocols": protocols,
        "mean_hmm_runtime_per_well": float(np.mean(runtimes)),
        "wall_runtime_seconds": float(time.perf_counter() - started),
        "decision": (
            "PASS" if all(result["gate_passed"] for result in protocols.values()) else "FAIL"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hmm_scale2_functional_ridge_pls_additive_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(detail_rows).to_csv(
        output_dir / "hmm_scale2_functional_ridge_pls_additive_per_well.csv",
        index=False,
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
