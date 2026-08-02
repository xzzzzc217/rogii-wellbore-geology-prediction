from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_ridge_pls_ensemble import (  # noqa: E402
    BASE_SIGNATURE_WIDTH,
    PLS_WEIGHT,
    RIDGE_WEIGHT,
    FixedRidgePLSEnsemble,
)


def test_fixed_ensemble_weights_are_conservative_and_normalized() -> None:
    assert RIDGE_WEIGHT == 0.75
    assert PLS_WEIGHT == 0.25
    assert RIDGE_WEIGHT + PLS_WEIGHT == 1.0


def test_fixed_ensemble_fits_both_multioutput_components() -> None:
    rng = np.random.default_rng(20260802)
    x = rng.normal(size=(32, BASE_SIGNATURE_WIDTH + 12))
    y = rng.normal(size=(32, 16))
    prediction = FixedRidgePLSEnsemble().fit(x, y).predict(x[:3])
    assert prediction.shape == (3, 16)
    assert np.isfinite(prediction).all()
