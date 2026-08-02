"""Evaluate a target-free well-functional residual model out of fold.

Each well is represented by five fixed quantile samples of the 51 target-free
sequence features.  A multi-output Ridge model predicts a 64-point residual
curve, learned only from wells outside the held-out outer fold.  The curve is
interpolated back to row resolution and applied as a frozen 0.20 overlay on the
competitive OOF base.

This script is an OOF evaluator only.  It never writes a Kaggle submission.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts" / "sequence_pilot_cache_v1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "functional_ridge_oof"

SIGNATURE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
CURVE_POINTS = 64
RIDGE_ALPHA = 100.0
OVERLAY_SCALE = 0.20
RESIDUAL_CLIP = 20.0
PROTOCOLS = (
    "spatial_cluster_7_seed_20260727",
    "typewell_fingerprint",
)


def _rmse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / max(int(rows), 1))


def _signature(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("well features must be a non-empty matrix")
    indices = np.rint(
        np.asarray(SIGNATURE_QUANTILES) * (len(features) - 1)
    ).astype(np.int64)
    return np.asarray(features[indices], dtype=np.float32).reshape(-1)


def _curve(residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 1 or len(residual) == 0:
        raise ValueError("well residual must be a non-empty vector")
    clipped = np.clip(residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return np.interp(
        np.linspace(0.0, 1.0, CURVE_POINTS),
        np.linspace(0.0, 1.0, len(clipped)),
        clipped,
    ).astype(np.float32)


def _model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def _protocol_gate(result: dict[str, Any]) -> dict[str, bool]:
    required_folds = 5 if result["fold_count"] >= 7 else result["fold_count"]
    return {
        "pooled_improves": bool(result["pooled_delta"] < 0.0),
        "p95_not_worse": bool(result["p95_well_delta"] <= 0.0),
        "worst_not_worse_than_plus_0_25": bool(result["worst_well_delta"] <= 0.25),
        "well_win_rate_at_least_0_45": bool(
            result["nonpositive_well_fraction"] >= 0.45
        ),
        "enough_folds_improve": bool(result["improved_folds"] >= required_folds),
    }


def _evaluate_protocol(
    *,
    protocol: str,
    metadata: dict[str, Any],
    signatures: np.ndarray,
    curves: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = np.asarray(
        [int(item["folds"][protocol]) for item in metadata["per_well"]],
        dtype=np.int64,
    )
    base_sse = 0.0
    candidate_sse = 0.0
    per_well: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []

    for fold in sorted(np.unique(folds).tolist()):
        train = np.flatnonzero(folds != fold)
        validation = np.flatnonzero(folds == fold)
        model = _model()
        model.fit(signatures[train], curves[train])
        predicted_curves = np.asarray(model.predict(signatures[validation]), dtype=float)
        fold_base_sse = 0.0
        fold_candidate_sse = 0.0
        fold_rows = 0

        for position, well_index in enumerate(validation.tolist()):
            start = int(offsets[well_index])
            end = int(offsets[well_index + 1])
            rows = end - start
            correction = np.interp(
                np.linspace(0.0, 1.0, rows),
                np.linspace(0.0, 1.0, CURVE_POINTS),
                predicted_curves[position],
            )
            correction = OVERLAY_SCALE * np.clip(
                correction,
                -RESIDUAL_CLIP,
                RESIDUAL_CLIP,
            )
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
            item = metadata["per_well"][well_index]
            per_well.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "well": str(item["well"]),
                    "rows": rows,
                    "base_rmse": well_base_rmse,
                    "candidate_rmse": well_candidate_rmse,
                    "delta": well_candidate_rmse - well_base_rmse,
                    "correction_rms": float(np.sqrt(np.mean(correction**2))),
                    "correction_max_abs": float(np.max(np.abs(correction))),
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
    result["gate_checks"] = _protocol_gate(result)
    result["gate_passed"] = bool(all(result["gate_checks"].values()))
    return result, per_well


def evaluate(cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.time()
    metadata = json.loads(
        (cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8")
    )
    features = np.load(cache_dir / "features.float16.npy", mmap_mode="r")
    base = np.load(cache_dir / "competitive_base.float32.npy", mmap_mode="r")
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "well_offsets.int64.npy")
    well_count = len(offsets) - 1
    if well_count != len(metadata["per_well"]):
        raise RuntimeError("cache well count does not match metadata")
    if int(offsets[-1]) != len(features) or len(base) != len(truth) or len(base) != len(features):
        raise RuntimeError("cache row arrays are inconsistent")

    signature_rows: list[np.ndarray] = []
    curves = np.empty((well_count, CURVE_POINTS), dtype=np.float32)
    for well_index in range(well_count):
        start = int(offsets[well_index])
        end = int(offsets[well_index + 1])
        signature_rows.append(
            _signature(np.asarray(features[start:end], dtype=np.float32))
        )
        curves[well_index] = _curve(
            np.asarray(truth[start:end], dtype=float)
            - np.asarray(base[start:end], dtype=float)
        )
    signatures = np.stack(signature_rows).astype(np.float32, copy=False)

    results: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        result, protocol_details = _evaluate_protocol(
            protocol=protocol,
            metadata=metadata,
            signatures=signatures,
            curves=curves,
            base=base,
            truth=truth,
            offsets=offsets,
        )
        results[protocol] = result
        details.extend(protocol_details)

    report = {
        "candidate": "target-free well-functional residual Ridge",
        "contract": {
            "writes_submission": False,
            "target_features_used": False,
            "well_id_encoded": False,
            "absolute_coordinates_used": False,
            "outer_fold_truth_isolation": True,
            "signature_quantiles": list(SIGNATURE_QUANTILES),
            "curve_points": CURVE_POINTS,
            "ridge_alpha": RIDGE_ALPHA,
            "overlay_scale": OVERLAY_SCALE,
            "residual_clip": RESIDUAL_CLIP,
            "feature_count": int(features.shape[1]),
            "signature_width": int(signatures.shape[1]),
        },
        "cache": str(cache_dir),
        "wells": int(well_count),
        "rows": int(len(features)),
        "protocols": results,
        "decision": "PASS" if all(row["gate_passed"] for row in results.values()) else "FAIL",
        "runtime_seconds": float(time.time() - started),
    }
    return report, pd.DataFrame(details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, detail = evaluate(args.cache_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "functional_ridge_oof_report.json"
    detail_path = args.output_dir / "functional_ridge_oof_per_well.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(detail_path, index=False)
    print(json.dumps({"decision": report["decision"], "protocols": report["protocols"]}, indent=2))
    print(f"wrote {report_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
