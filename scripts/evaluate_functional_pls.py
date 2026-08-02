"""Evaluate one preregistered supervised low-rank functional PLS model.

The candidate uses the fixed globally augmented target-free well signature and
predicts the 64-point residual curve with eight-component PLSRegression.  PLS
components are fitted only on outer-training wells, so response compression is
supervised without exposing held-out curves.  This evaluator never writes a
Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_functional_ridge as plain
import pandas as pd
from evaluate_functional_ridge_augmented import _augmented_signature
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_pls_oof"
PLS_COMPONENTS = 8
PLS_MAX_ITER = 500
PLS_TOLERANCE = 1e-6
DOMINANCE_MARGIN = 0.003
PLAIN_RIDGE_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0276029,
    "typewell_fingerprint": -0.0187671,
}


def _model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        PLSRegression(
            n_components=PLS_COMPONENTS,
            scale=True,
            max_iter=PLS_MAX_ITER,
            tol=PLS_TOLERANCE,
        ),
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
    original_signature = plain._signature
    original_model = plain._model
    try:
        plain._signature = _augmented_signature
        plain._model = _model
        report, detail = plain.evaluate(cache_dir)
    finally:
        plain._signature = original_signature
        plain._model = original_model

    report["candidate"] = "target-free supervised low-rank Functional PLS"
    contract = report["contract"]
    contract.pop("ridge_alpha", None)
    contract.update(
        {
            "model": "PLSRegression",
            "pls_components": PLS_COMPONENTS,
            "pls_max_iter": PLS_MAX_ITER,
            "pls_tolerance": PLS_TOLERANCE,
            "pls_scale": True,
            "pls_fit_on_outer_training_wells_only": True,
            "signature": "five fixed samples plus mean, std, and relative slope",
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
    report_path = args.output_dir / "functional_pls_oof_report.json"
    detail_path = args.output_dir / "functional_pls_oof_per_well.csv"
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
