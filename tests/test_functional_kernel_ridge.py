from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_kernel_ridge import (  # noqa: E402
    DOMINANCE_MARGIN,
    PLAIN_RIDGE_REFERENCE,
    MedianBandwidthKernelRidge,
    _apply_deployment_gate,
    _model,
)


def test_median_bandwidth_kernel_model_fits_multioutput_curves() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    y = np.column_stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]])
    model = MedianBandwidthKernelRidge(alpha=10.0).fit(x, y)
    prediction = model.predict(x[:2])
    assert prediction.shape == (2, 2)
    assert model.gamma_ is not None
    assert np.isfinite(model.gamma_) and model.gamma_ > 0.0
    assert np.isfinite(prediction).all()


def test_kernel_model_obeys_sklearn_pipeline_fitted_contract() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, np.nan], [0.0, 2.0], [2.0, 2.0]])
    y = np.column_stack([np.nan_to_num(x[:, 0]) + np.nan_to_num(x[:, 1])] * 2)
    pipeline = _model().fit(x, y)
    prediction = pipeline.predict(x[:2])
    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()


def test_deployment_gate_requires_margin_over_plain_ridge() -> None:
    protocols = {}
    for protocol, reference in PLAIN_RIDGE_REFERENCE.items():
        protocols[protocol] = {
            "pooled_delta": reference - DOMINANCE_MARGIN + 0.001,
            "gate_checks": {"base_gate": True},
            "gate_passed": True,
        }
    report = {"protocols": protocols, "decision": "PASS"}
    _apply_deployment_gate(report)
    assert report["decision"] == "FAIL"
    assert all(
        not row["gate_checks"]["beats_plain_ridge_by_margin"]
        for row in report["protocols"].values()
    )
