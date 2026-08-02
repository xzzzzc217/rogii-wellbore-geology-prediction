"""Evaluate one preregistered target-free distance-gated Ridge/PLS ensemble.

PLS receives 50% curve weight for signatures inside the outer-training
distribution and 25% for out-of-distribution signatures.  The OOD threshold is
the 95th percentile of training-well leave-one-out nearest-neighbor distances
after training-only median imputation and standardization.  No validation truth
is used by the gate.  This evaluator never writes a Kaggle submission.
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
from evaluate_functional_ridge_pls_ensemble import (
    BASE_SIGNATURE_WIDTH,
    PLAIN_RIDGE_MODEL,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_ridge_pls_distance_gate_oof"
IN_DOMAIN_PLS_WEIGHT = 0.50
OOD_PLS_WEIGHT = 0.25
TRAINING_NN_QUANTILE = 0.95
DOMINANCE_MARGIN = 0.003
FIXED_25_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.03520864252152123,
    "typewell_fingerprint": -0.023058503367586525,
}


class DistanceGatedRidgePLSEnsemble(RegressorMixin, BaseEstimator):
    """Curve ensemble whose PLS weight depends only on training support."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> DistanceGatedRidgePLSEnsemble:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] <= BASE_SIGNATURE_WIDTH:
            raise ValueError("distance gate requires the augmented signature")
        self.ridge_model_ = PLAIN_RIDGE_MODEL()
        self.pls_model_ = pls_model()
        self.ridge_model_.fit(x[:, :BASE_SIGNATURE_WIDTH], y)
        self.pls_model_.fit(x, y)

        self.distance_imputer_ = SimpleImputer(strategy="median")
        self.distance_scaler_ = StandardScaler()
        imputed = self.distance_imputer_.fit_transform(x)
        self.distance_train_ = self.distance_scaler_.fit_transform(imputed)
        distances = pairwise_distances(self.distance_train_, metric="euclidean")
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        if not np.isfinite(nearest).all():
            raise ValueError("training nearest-neighbor distances are non-finite")
        self.ood_threshold_ = float(np.quantile(nearest, TRAINING_NN_QUANTILE))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ridge_prediction = np.asarray(
            self.ridge_model_.predict(x[:, :BASE_SIGNATURE_WIDTH]),
            dtype=float,
        )
        pls_prediction = np.asarray(self.pls_model_.predict(x), dtype=float)
        standardized = self.distance_scaler_.transform(
            self.distance_imputer_.transform(x)
        )
        nearest = np.min(
            pairwise_distances(standardized, self.distance_train_, metric="euclidean"),
            axis=1,
        )
        self.last_pls_weights_ = np.where(
            nearest <= self.ood_threshold_,
            IN_DOMAIN_PLS_WEIGHT,
            OOD_PLS_WEIGHT,
        )
        weights = self.last_pls_weights_[:, None]
        return (1.0 - weights) * ridge_prediction + weights * pls_prediction


def _model() -> DistanceGatedRidgePLSEnsemble:
    return DistanceGatedRidgePLSEnsemble()


def _apply_deployment_gate(report: dict[str, Any]) -> None:
    for protocol, result in report["protocols"].items():
        result["fixed_25_reference_delta"] = FIXED_25_REFERENCE[protocol]
        result["delta_vs_fixed_25"] = (
            result["pooled_delta"] - FIXED_25_REFERENCE[protocol]
        )
        result["gate_checks"]["beats_fixed_25_by_margin"] = bool(
            result["pooled_delta"]
            <= FIXED_25_REFERENCE[protocol] - DOMINANCE_MARGIN
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

    report["candidate"] = "target-free distance-gated Functional Ridge/PLS"
    contract = report["contract"]
    contract.pop("ridge_alpha", None)
    contract.update(
        {
            "model": "training-support distance-gated Ridge/PLS curve ensemble",
            "in_domain_pls_weight": IN_DOMAIN_PLS_WEIGHT,
            "ood_pls_weight": OOD_PLS_WEIGHT,
            "training_nn_quantile": TRAINING_NN_QUANTILE,
            "distance_preprocessing_fit_on_outer_training_wells_only": True,
            "validation_truth_used_by_gate": False,
            "weight_or_quantile_grid_searched": False,
            "dominance_margin_vs_fixed_25": DOMINANCE_MARGIN,
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
    report_path = args.output_dir / "functional_ridge_pls_distance_gate_report.json"
    detail_path = args.output_dir / "functional_ridge_pls_distance_gate_per_well.csv"
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
