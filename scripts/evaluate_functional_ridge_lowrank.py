"""Evaluate one preregistered low-rank Functional Ridge candidate.

Response PCA is fitted only on the residual curves of outer-training wells.
The same target-free well signatures used by the plain Functional Ridge then
predict eight PCA coefficients.  Held-out curves are reconstructed and applied
with the frozen 0.20 overlay.  This evaluator never writes a submission.
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
from evaluate_functional_ridge import (
    CURVE_POINTS,
    DEFAULT_CACHE,
    OVERLAY_SCALE,
    PROTOCOLS,
    RESIDUAL_CLIP,
    ROOT,
    _curve,
    _model,
    _protocol_gate,
    _signature,
)
from sklearn.decomposition import PCA

DEFAULT_OUTPUT = ROOT / "artifacts" / "functional_ridge_lowrank_oof"
PCA_COMPONENTS = 8
DOMINANCE_MARGIN = 0.002
PLAIN_RIDGE_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0276029,
    "typewell_fingerprint": -0.0187671,
}


def _rmse(sse: float, rows: int) -> float:
    return math.sqrt(float(sse) / max(int(rows), 1))


def _fit_predict_curves(
    train_signatures: np.ndarray,
    train_curves: np.ndarray,
    validation_signatures: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit training-only response PCA and return reconstructed held-out curves."""
    if len(train_signatures) != len(train_curves):
        raise ValueError("training signatures and curves must have equal rows")
    if len(train_curves) <= PCA_COMPONENTS:
        raise ValueError("too few training wells for frozen PCA dimension")
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    train_scores = pca.fit_transform(np.asarray(train_curves, dtype=float))
    model = _model()
    model.fit(train_signatures, train_scores)
    validation_scores = np.asarray(model.predict(validation_signatures), dtype=float)
    reconstructed = np.asarray(pca.inverse_transform(validation_scores), dtype=float)
    return reconstructed, float(np.sum(pca.explained_variance_ratio_))


def _deployment_gate(protocol: str, result: dict[str, Any]) -> dict[str, bool]:
    checks = _protocol_gate(result)
    checks["beats_plain_ridge_by_margin"] = bool(
        result["pooled_delta"]
        <= PLAIN_RIDGE_REFERENCE[protocol] - DOMINANCE_MARGIN
    )
    return checks


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
        predicted_curves, explained_variance = _fit_predict_curves(
            signatures[train],
            curves[train],
            signatures[validation],
        )
        fold_base_sse = 0.0
        fold_candidate_sse = 0.0
        fold_rows = 0

        for position, well_index in enumerate(validation.tolist()):
            start = int(offsets[well_index])
            end = int(offsets[well_index + 1])
            rows = end - start
            correction = OVERLAY_SCALE * np.clip(
                np.interp(
                    np.linspace(0.0, 1.0, rows),
                    np.linspace(0.0, 1.0, CURVE_POINTS),
                    predicted_curves[position],
                ),
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
                "response_pca_explained_variance": explained_variance,
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
        "plain_ridge_reference_delta": PLAIN_RIDGE_REFERENCE[protocol],
        "delta_vs_plain_ridge": (
            candidate_rmse - base_rmse - PLAIN_RIDGE_REFERENCE[protocol]
        ),
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
    result["gate_checks"] = _deployment_gate(protocol, result)
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
    if (
        int(offsets[-1]) != len(features)
        or len(base) != len(truth)
        or len(base) != len(features)
    ):
        raise RuntimeError("cache row arrays are inconsistent")

    signatures = np.empty(
        (well_count, features.shape[1] * 5),
        dtype=np.float32,
    )
    curves = np.empty((well_count, CURVE_POINTS), dtype=np.float32)
    for well_index in range(well_count):
        start = int(offsets[well_index])
        end = int(offsets[well_index + 1])
        signatures[well_index] = _signature(
            np.asarray(features[start:end], dtype=np.float32)
        )
        curves[well_index] = _curve(
            np.asarray(truth[start:end], dtype=float)
            - np.asarray(base[start:end], dtype=float)
        )

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
        "candidate": "target-free low-rank well-functional residual Ridge",
        "contract": {
            "writes_submission": False,
            "target_features_used": False,
            "well_id_encoded": False,
            "absolute_coordinates_used": False,
            "outer_fold_truth_isolation": True,
            "response_pca_fit_on_outer_training_only": True,
            "response_pca_components": PCA_COMPONENTS,
            "curve_points": CURVE_POINTS,
            "ridge_alpha": 100.0,
            "overlay_scale": OVERLAY_SCALE,
            "residual_clip": RESIDUAL_CLIP,
            "dominance_margin_vs_plain_ridge": DOMINANCE_MARGIN,
        },
        "cache": str(cache_dir),
        "wells": int(well_count),
        "rows": int(len(features)),
        "protocols": results,
        "decision": (
            "PASS" if all(row["gate_passed"] for row in results.values()) else "FAIL"
        ),
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
    report_path = args.output_dir / "functional_ridge_lowrank_oof_report.json"
    detail_path = args.output_dir / "functional_ridge_lowrank_oof_per_well.csv"
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
