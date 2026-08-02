from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_ridge import (  # noqa: E402
    CURVE_POINTS,
    RESIDUAL_CLIP,
    SIGNATURE_QUANTILES,
    _curve,
    _protocol_gate,
    _signature,
)


def test_signature_uses_fixed_well_quantiles() -> None:
    features = np.arange(11 * 3, dtype=float).reshape(11, 3)
    signature = _signature(features)
    expected_indices = np.rint(
        np.asarray(SIGNATURE_QUANTILES) * (len(features) - 1)
    ).astype(int)
    np.testing.assert_array_equal(signature, features[expected_indices].reshape(-1))


def test_curve_clips_and_resamples_residual() -> None:
    residual = np.asarray([-100.0, 0.0, 100.0])
    curve = _curve(residual)
    assert curve.shape == (CURVE_POINTS,)
    assert float(curve.min()) >= -RESIDUAL_CLIP
    assert float(curve.max()) <= RESIDUAL_CLIP
    assert float(curve[0]) == -RESIDUAL_CLIP
    assert float(curve[-1]) == RESIDUAL_CLIP


def test_protocol_gate_requires_stable_outer_fold_improvement() -> None:
    passing = {
        "fold_count": 7,
        "pooled_delta": -0.02,
        "p95_well_delta": -0.01,
        "worst_well_delta": 0.20,
        "nonpositive_well_fraction": 0.46,
        "improved_folds": 5,
    }
    assert all(_protocol_gate(passing).values())
    failing = dict(passing, improved_folds=4)
    assert not all(_protocol_gate(failing).values())


def test_signature_width_is_not_assumed_by_the_evaluator_source() -> None:
    source = (SCRIPTS / "evaluate_functional_ridge.py").read_text(encoding="utf-8")
    assert "np.stack(signature_rows)" in source
    assert "features.shape[1] * len(SIGNATURE_QUANTILES)" not in source
