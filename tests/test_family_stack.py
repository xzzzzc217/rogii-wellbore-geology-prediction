from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.family_stack import (  # noqa: E402
    collapse_model_families,
    compose_ridge_pf_track,
)


def test_collapse_model_families_averages_each_frozen_family() -> None:
    predictions = np.asarray(
        [
            [1.0, 4.0, 7.0, 10.0, 14.0],
            [2.0, 5.0, 8.0, 12.0, 16.0],
        ],
        dtype=np.float32,
    )

    collapsed = collapse_model_families(predictions)

    assert collapsed.dtype == np.float32
    assert np.array_equal(collapsed, np.asarray([[4.0, 12.0], [5.0, 14.0]]))


def test_collapse_model_families_requires_exact_nonoverlapping_coverage() -> None:
    predictions = np.ones((4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="cover every"):
        collapse_model_families(predictions, families=((0, 1, 2), (3,)))
    with pytest.raises(ValueError, match="unique"):
        collapse_model_families(predictions, families=((0, 1, 2), (2, 3, 4)))
    with pytest.raises(ValueError, match="finite"):
        broken = predictions.copy()
        broken[0, 0] = np.nan
        collapse_model_families(broken)


def test_compose_ridge_pf_track_matches_frozen_formula() -> None:
    anchor = np.asarray([100.0, 100.0, 100.0])
    ridge_delta = np.asarray([20.0, 20.0, 20.0])
    pf_tvt = np.asarray([110.0, 110.0, 110.0])
    md_since = np.asarray([0.0, 85.0, 170.0])

    prediction = compose_ridge_pf_track(anchor, ridge_delta, pf_tvt, md_since)

    blended_delta = 0.91 * 20.0 + 0.09 * 10.0
    expected = anchor + blended_delta * (1.0 - np.exp(-md_since / 85.0))
    assert np.allclose(prediction, expected)
    assert prediction[0] == 100.0


def test_compose_ridge_pf_track_rejects_invalid_contract() -> None:
    values = np.ones(3)

    with pytest.raises(ValueError, match="same shape"):
        compose_ridge_pf_track(values, values[:2], values, values)
    with pytest.raises(ValueError, match="sum to one"):
        compose_ridge_pf_track(values, values, values, values, ridge_weight=0.8)
    with pytest.raises(ValueError, match="positive"):
        compose_ridge_pf_track(values, values, values, values, ramp_tau_md=0.0)
