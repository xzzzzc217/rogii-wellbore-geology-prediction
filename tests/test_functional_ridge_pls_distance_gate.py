from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_functional_ridge_pls_distance_gate import (  # noqa: E402
    BASE_SIGNATURE_WIDTH,
    IN_DOMAIN_PLS_WEIGHT,
    OOD_PLS_WEIGHT,
    TRAINING_NN_QUANTILE,
    DistanceGatedRidgePLSEnsemble,
)


def test_distance_gate_parameters_are_frozen() -> None:
    assert IN_DOMAIN_PLS_WEIGHT == 0.50
    assert OOD_PLS_WEIGHT == 0.25
    assert TRAINING_NN_QUANTILE == 0.95


def test_distance_gate_downweights_far_signature_without_truth() -> None:
    rng = np.random.default_rng(20260802)
    x = rng.normal(scale=0.1, size=(32, BASE_SIGNATURE_WIDTH + 12))
    y = rng.normal(size=(32, 16))
    model = DistanceGatedRidgePLSEnsemble().fit(x, y)
    query = np.vstack([x[0], np.full(x.shape[1], 100.0)])
    prediction = model.predict(query)
    assert prediction.shape == (2, 16)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(
        model.last_pls_weights_,
        [IN_DOMAIN_PLS_WEIGHT, OOD_PLS_WEIGHT],
    )
