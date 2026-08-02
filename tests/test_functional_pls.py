from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_pls import (  # noqa: E402
    DOMINANCE_MARGIN,
    PLAIN_RIDGE_REFERENCE,
    PLS_COMPONENTS,
    _apply_deployment_gate,
    _model,
)


def test_functional_pls_pipeline_predicts_multioutput_curve() -> None:
    rng = np.random.default_rng(20260802)
    x = rng.normal(size=(32, 12))
    y = rng.normal(size=(32, 16))
    x[0, 0] = np.nan
    model = _model().fit(x, y)
    prediction = model.predict(x[:3])
    assert PLS_COMPONENTS < min(x.shape)
    assert prediction.shape == (3, 16)
    assert np.isfinite(prediction).all()


def test_functional_pls_gate_requires_plain_ridge_dominance() -> None:
    protocols = {
        protocol: {
            "pooled_delta": reference - DOMINANCE_MARGIN + 0.0001,
            "gate_checks": {"base_gate": True},
            "gate_passed": True,
        }
        for protocol, reference in PLAIN_RIDGE_REFERENCE.items()
    }
    report = {"protocols": protocols, "decision": "PASS"}
    _apply_deployment_gate(report)
    assert report["decision"] == "FAIL"
