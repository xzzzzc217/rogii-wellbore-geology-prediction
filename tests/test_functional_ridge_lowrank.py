from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_ridge_lowrank import (  # noqa: E402
    DOMINANCE_MARGIN,
    PCA_COMPONENTS,
    PLAIN_RIDGE_REFERENCE,
    _deployment_gate,
    _fit_predict_curves,
)


def test_lowrank_predictor_returns_reconstructed_curves() -> None:
    rng = np.random.default_rng(20260802)
    signatures = rng.normal(size=(40, 12))
    latent = rng.normal(size=(40, PCA_COMPONENTS))
    basis = rng.normal(size=(PCA_COMPONENTS, 64))
    curves = latent @ basis
    predicted, explained = _fit_predict_curves(
        signatures[:32],
        curves[:32],
        signatures[32:],
    )
    assert predicted.shape == (8, 64)
    assert np.isfinite(predicted).all()
    assert 0.0 < explained <= 1.0 + 1e-12


def test_gate_requires_margin_over_plain_ridge() -> None:
    protocol = "typewell_fingerprint"
    passing = {
        "fold_count": 5,
        "pooled_delta": PLAIN_RIDGE_REFERENCE[protocol] - DOMINANCE_MARGIN - 1e-6,
        "p95_well_delta": -0.01,
        "worst_well_delta": 0.20,
        "nonpositive_well_fraction": 0.46,
        "improved_folds": 5,
    }
    assert all(_deployment_gate(protocol, passing).values())
    failing = dict(passing, pooled_delta=PLAIN_RIDGE_REFERENCE[protocol] - 0.001)
    assert not _deployment_gate(protocol, failing)["beats_plain_ridge_by_margin"]
