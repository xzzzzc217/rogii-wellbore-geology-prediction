"""Evaluate one preregistered nonlinear well-functional residual model.

The candidate uses the exact target-free well signatures and 64-point residual
curves of the passing Functional Ridge experiment.  Within every outer fold,
median imputation and standardization are fitted on training wells only.  An
RBF Kernel Ridge model then predicts the held-out curves.  Its RBF bandwidth is
derived only from the median positive pairwise squared distance among that
outer fold's training wells.

This is an OOF evaluator only.  It never writes a Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_functional_ridge as plain
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_kernel_ridge_oof"
KERNEL_ALPHA = 10.0
OVERLAY_SCALE = plain.OVERLAY_SCALE
DOMINANCE_MARGIN = 0.005
PLAIN_RIDGE_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0276029,
    "typewell_fingerprint": -0.0187671,
}


class MedianBandwidthKernelRidge:
    """RBF Kernel Ridge with a training-only median-distance bandwidth."""

    def __init__(self, alpha: float = KERNEL_ALPHA) -> None:
        self.alpha = float(alpha)
        self.gamma_: float | None = None
        self.model_: KernelRidge | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> MedianBandwidthKernelRidge:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("kernel model needs at least two training wells")
        distances = pairwise_distances(x, metric="sqeuclidean")
        positive = distances[np.triu_indices(len(x), k=1)]
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if len(positive) == 0:
            raise ValueError("training signatures have no positive finite distances")
        median_squared_distance = float(np.median(positive))
        self.gamma_ = 1.0 / median_squared_distance
        self.model_ = KernelRidge(
            alpha=self.alpha,
            kernel="rbf",
            gamma=self.gamma_,
        )
        self.model_.fit(x, np.asarray(y, dtype=float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("kernel model is not fitted")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


def _model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MedianBandwidthKernelRidge(alpha=KERNEL_ALPHA),
    )


def _apply_deployment_gate(report: dict[str, Any]) -> None:
    for protocol, result in report["protocols"].items():
        result["plain_ridge_reference_delta"] = PLAIN_RIDGE_REFERENCE[protocol]
        result["delta_vs_plain_ridge"] = (
            result["pooled_delta"] - PLAIN_RIDGE_REFERENCE[protocol]
        )
        result["gate_checks"]["beats_plain_ridge_by_margin"] = bool(
            result["pooled_delta"]
            <= PLAIN_RIDGE_REFERENCE[protocol] - DOMINANCE_MARGIN
        )
        result["gate_passed"] = bool(all(result["gate_checks"].values()))
    report["decision"] = (
        "PASS"
        if all(result["gate_passed"] for result in report["protocols"].values())
        else "FAIL"
    )


def evaluate(cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    original_model = plain._model
    try:
        plain._model = _model
        report, detail = plain.evaluate(cache_dir)
    finally:
        plain._model = original_model

    report["candidate"] = "target-free well-functional residual RBF Kernel Ridge"
    contract = report["contract"]
    contract.pop("ridge_alpha", None)
    contract.update(
        {
            "model": "RBF Kernel Ridge",
            "kernel_alpha": KERNEL_ALPHA,
            "kernel_gamma_rule": "inverse median positive squared distance",
            "kernel_gamma_fit_on_outer_training_wells_only": True,
            "dominance_margin_vs_plain_ridge": DOMINANCE_MARGIN,
        }
    )
    _apply_deployment_gate(report)
    return report, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=plain.DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, detail = evaluate(args.cache_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "functional_kernel_ridge_oof_report.json"
    detail_path = args.output_dir / "functional_kernel_ridge_oof_per_well.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(detail_path, index=False)
    print(
        json.dumps(
            {"decision": report["decision"], "protocols": report["protocols"]},
            indent=2,
        )
    )
    print(f"wrote {report_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
