"""Evaluate one conservative fixed-weight Functional Ridge/PLS ensemble.

The stable plain Functional Ridge receives 75% curve weight.  The stronger but
one-fold-unstable supervised Functional PLS receives 25%.  The weight is fixed
before this evaluator is run; no grid is searched.  Both component models are
fit only on outer-training wells.  This script never writes a submission.
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

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_ridge_pls_ensemble_oof"
BASE_SIGNATURE_WIDTH = 51 * 5
RIDGE_WEIGHT = 0.75
PLS_WEIGHT = 0.25
DOMINANCE_MARGIN = 0.002
PLAIN_RIDGE_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0276029,
    "typewell_fingerprint": -0.0187671,
}
PLAIN_RIDGE_MODEL = plain._model


class FixedRidgePLSEnsemble(RegressorMixin, BaseEstimator):
    """Training-only curve ensemble with frozen 75/25 component weights."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> FixedRidgePLSEnsemble:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] <= BASE_SIGNATURE_WIDTH:
            raise ValueError("ensemble requires the globally augmented signature")
        self.ridge_model_ = PLAIN_RIDGE_MODEL()
        self.pls_model_ = pls_model()
        self.ridge_model_.fit(x[:, :BASE_SIGNATURE_WIDTH], y)
        self.pls_model_.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ridge_prediction = np.asarray(
            self.ridge_model_.predict(x[:, :BASE_SIGNATURE_WIDTH]),
            dtype=float,
        )
        pls_prediction = np.asarray(self.pls_model_.predict(x), dtype=float)
        return RIDGE_WEIGHT * ridge_prediction + PLS_WEIGHT * pls_prediction


def _model() -> FixedRidgePLSEnsemble:
    return FixedRidgePLSEnsemble()


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
    original_signature = plain._signature
    original_model = plain._model
    try:
        plain._signature = _augmented_signature
        plain._model = _model
        report, detail = plain.evaluate(cache_dir)
    finally:
        plain._signature = original_signature
        plain._model = original_model

    report["candidate"] = "target-free fixed 75/25 Functional Ridge/PLS ensemble"
    contract = report["contract"]
    contract.pop("ridge_alpha", None)
    contract.update(
        {
            "model": "fixed curve-level Ridge/PLS ensemble",
            "plain_ridge_weight": RIDGE_WEIGHT,
            "supervised_pls_weight": PLS_WEIGHT,
            "weight_grid_searched": False,
            "both_models_fit_on_outer_training_wells_only": True,
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
    report_path = args.output_dir / "functional_ridge_pls_ensemble_oof_report.json"
    detail_path = args.output_dir / "functional_ridge_pls_ensemble_oof_per_well.csv"
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
