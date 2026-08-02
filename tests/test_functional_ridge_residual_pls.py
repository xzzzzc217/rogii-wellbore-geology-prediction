from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_ridge_residual_pls import (  # noqa: E402
    BASE_SIGNATURE_WIDTH,
    RESIDUAL_PLS_WEIGHT,
    RidgeResidualPLS,
)


def test_residual_pls_weight_is_frozen() -> None:
    assert RESIDUAL_PLS_WEIGHT == 0.25


def test_residual_pls_predicts_finite_curve() -> None:
    rng = np.random.default_rng(20260802)
    x = rng.normal(size=(36, BASE_SIGNATURE_WIDTH + 12))
    y = rng.normal(size=(36, 16))
    prediction = RidgeResidualPLS().fit(x, y).predict(x[:4])
    assert prediction.shape == (4, 16)
    assert np.isfinite(prediction).all()
