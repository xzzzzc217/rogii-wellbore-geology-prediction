from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_ridge import SIGNATURE_QUANTILES  # noqa: E402
from evaluate_functional_ridge_augmented import (  # noqa: E402
    DOMINANCE_MARGIN,
    PLAIN_RIDGE_REFERENCE,
    _apply_deployment_gate,
    _augmented_signature,
)


def test_augmented_signature_appends_mean_std_and_slope_per_feature() -> None:
    position = np.linspace(0.0, 1.0, 11)
    features = np.column_stack([2.0 * position + 1.0, -3.0 * position + 4.0])
    signature = _augmented_signature(features)
    base_width = len(SIGNATURE_QUANTILES) * features.shape[1]
    assert signature.shape == (base_width + 3 * features.shape[1],)
    np.testing.assert_allclose(signature[base_width : base_width + 2], [2.0, 2.5])
    np.testing.assert_allclose(signature[-2:], [2.0, -3.0], atol=1e-6)


def test_augmented_deployment_gate_requires_both_protocol_margins() -> None:
    protocols = {}
    for protocol, reference in PLAIN_RIDGE_REFERENCE.items():
        protocols[protocol] = {
            "pooled_delta": reference - DOMINANCE_MARGIN - 0.001,
            "gate_checks": {"base_gate": True},
            "gate_passed": True,
        }
    report = {"protocols": protocols, "decision": "FAIL"}
    _apply_deployment_gate(report)
    assert report["decision"] == "PASS"
    report["protocols"]["typewell_fingerprint"]["pooled_delta"] += 0.01
    _apply_deployment_gate(report)
    assert report["decision"] == "FAIL"
