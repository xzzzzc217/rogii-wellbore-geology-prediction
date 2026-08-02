from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_functional_extra_trees import (  # noqa: E402
    MAX_FEATURES,
    MIN_SAMPLES_LEAF,
    N_ESTIMATORS,
    RANDOM_STATE,
    _model,
)


def test_extra_trees_contract_is_frozen() -> None:
    assert N_ESTIMATORS == 384
    assert MAX_FEATURES == 0.50
    assert MIN_SAMPLES_LEAF == 8
    assert RANDOM_STATE == 20260802


def test_extra_trees_predicts_finite_multioutput_curve() -> None:
    rng = np.random.default_rng(20260802)
    x = rng.normal(size=(48, 12))
    x[0, 0] = np.nan
    y = rng.normal(size=(48, 7))
    prediction = _model().fit(x, y).predict(x[:5])
    assert prediction.shape == (5, 7)
    assert np.isfinite(prediction).all()
