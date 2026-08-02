"""Evaluate one preregistered nonlinear Functional ExtraTrees candidate.

The model receives the globally augmented, target-free well signature and
predicts the complete 64-point residual curve with one conservative
multi-output ExtraTrees regressor.  All imputation and tree fitting happen
inside each outer-training fold.  The forest parameters and 0.20 overlay are
frozen before evaluation; this script never writes a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_functional_ridge as plain
import pandas as pd
from evaluate_functional_ridge_augmented import _augmented_signature
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_extra_trees_oof"
N_ESTIMATORS = 384
MAX_FEATURES = 0.50
MIN_SAMPLES_LEAF = 8
RANDOM_STATE = 20260802
DOMINANCE_MARGIN = 0.003
RIDGE_PLS_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0352086425,
    "typewell_fingerprint": -0.0230585034,
}


def _model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=MAX_FEATURES,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            bootstrap=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )


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

    report["candidate"] = "target-free nonlinear Functional ExtraTrees"
    contract = report["contract"]
    contract.pop("ridge_alpha", None)
    contract.update(
        {
            "model": "multi-output ExtraTreesRegressor",
            "n_estimators": N_ESTIMATORS,
            "max_features": MAX_FEATURES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "bootstrap": False,
            "random_state": RANDOM_STATE,
            "hyperparameter_search_performed": False,
            "augmented_signature": True,
            "tree_fit_on_outer_training_wells_only": True,
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
    report_path = args.output_dir / "functional_extra_trees_oof_report.json"
    detail_path = args.output_dir / "functional_extra_trees_oof_per_well.csv"
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
