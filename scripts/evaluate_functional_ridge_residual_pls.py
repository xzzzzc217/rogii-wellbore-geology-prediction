"""Evaluate one preregistered sequential Functional Ridge residual-PLS model.

The stable plain Functional Ridge first predicts the complete residual curve.
An eight-component PLS model is then fitted to the remaining in-training curve
residual using the augmented target-free signature.  At inference the frozen
0.25 second-stage correction is added to Ridge.  Both stages are fitted only
on outer-training wells.  This evaluator never writes a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_functional_ridge as plain
import numpy as np
import pandas as pd
from evaluate_functional_pls import _model as pls_model
from evaluate_functional_ridge_augmented import _augmented_signature
from sklearn.base import BaseEstimator, RegressorMixin

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_ridge_residual_pls_oof"
BASE_SIGNATURE_WIDTH = 51 * 5
RESIDUAL_PLS_WEIGHT = 0.25
DOMINANCE_MARGIN = 0.002
RIDGE_PLS_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0352086425,
    "typewell_fingerprint": -0.0230585034,
}
PLAIN_RIDGE_MODEL = plain._model


class RidgeResidualPLS(RegressorMixin, BaseEstimator):
    """Sequential curve model with a frozen PLS residual boost."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeResidualPLS:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] <= BASE_SIGNATURE_WIDTH:
            raise ValueError("residual PLS requires the globally augmented signature")
        if y.ndim != 2 or len(y) != len(x):
            raise ValueError("target curves must be a matching two-dimensional matrix")
        self.ridge_model_ = PLAIN_RIDGE_MODEL()
        self.ridge_model_.fit(x[:, :BASE_SIGNATURE_WIDTH], y)
        ridge_in_sample = np.asarray(
            self.ridge_model_.predict(x[:, :BASE_SIGNATURE_WIDTH]),
            dtype=float,
        )
        self.pls_residual_model_ = pls_model()
        self.pls_residual_model_.fit(x, y - ridge_in_sample)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ridge_prediction = np.asarray(
            self.ridge_model_.predict(x[:, :BASE_SIGNATURE_WIDTH]),
            dtype=float,
        )
        residual_prediction = np.asarray(
            self.pls_residual_model_.predict(x),
            dtype=float,
        )
        return ridge_prediction + RESIDUAL_PLS_WEIGHT * residual_prediction


def _model() -> RidgeResidualPLS:
    return RidgeResidualPLS()


def _apply_deployment_gate(report: dict[str, Any]) -> None:
    for protocol, result in report["protocols"].items():
        reference = RIDGE_PLS_REFERENCE[protocol]
        result["ridge_pls_reference_delta"] = reference
        result["delta_vs_ridge_pls"] = result["pooled_delta"] - reference
        result["gate_checks"]["beats_ridge_pls_by_margin"] = bool(
            result["pooled_delta"] <= reference - DOMINANCE_MARGIN
        )
        result["gate_passed"] = bool(all(result["gate_checks"].values()))
    report["decision"] = (
        "PASS"
        if all(result["gate_passed"] for result in report["protocols"].values())
        else "FAIL"
    )


def evaluate(cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    original_signature = plain._signature
    original_model = plain._model
    try:
        plain._signature = _augmented_signature
        plain._model = _model
        report, detail = plain.evaluate(cache_dir)
    finally:
        plain._signature = original_signature
        plain._model = original_model

    report["candidate"] = "target-free sequential Functional Ridge residual PLS"
    contract = report["contract"]
    contract.pop("ridge_alpha", None)
    contract.update(
        {
            "model": "plain Ridge plus supervised residual PLS",
            "residual_pls_weight": RESIDUAL_PLS_WEIGHT,
            "weight_grid_searched": False,
            "ridge_fit_on_outer_training_wells_only": True,
            "pls_target": "training curve minus in-sample Ridge curve",
            "pls_fit_on_outer_training_wells_only": True,
            "signature": "five fixed samples plus mean, std, and relative slope",
            "dominance_margin_vs_ridge_pls": DOMINANCE_MARGIN,
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
    report_path = args.output_dir / "functional_ridge_residual_pls_oof_report.json"
    detail_path = args.output_dir / "functional_ridge_residual_pls_oof_per_well.csv"
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
