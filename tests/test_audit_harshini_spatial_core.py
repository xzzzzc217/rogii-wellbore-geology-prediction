from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_harshini_spatial_core import (  # noqa: E402
    _aggregate,
    _fold_labels,
    _restricted_pickle_load,
)


class _UnsafePickle:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return eval, ("40 + 2",)


def test_restricted_pickle_loader_rejects_arbitrary_globals() -> None:
    payload = pickle.dumps(_UnsafePickle())

    with pytest.raises(pickle.UnpicklingError, match="blocked pickle global"):
        _restricted_pickle_load(io.BytesIO(payload))


def test_aggregate_uses_inverse_squared_reliability() -> None:
    predictions = {
        "a": np.array([1.0, 3.0]),
        "b": np.array([5.0, 7.0]),
    }
    reliability = {"a": 1.0, "b": 2.0}
    weights = np.array([1.0 / (1.0 + 1e-6), 1.0 / (4.0 + 1e-6)])
    weights /= weights.sum()

    actual = _aggregate(predictions, reliability)

    expected = weights[0] * predictions["a"] + weights[1] * predictions["b"]
    np.testing.assert_allclose(actual, expected)


def test_fold_labels_are_deterministic_and_complete() -> None:
    centroids = {
        f"well-{index:02d}": np.array([float(index % 5), float(index // 5)])
        for index in range(20)
    }

    first = _fold_labels(centroids, clusters=4, seed=20260727)
    second = _fold_labels(centroids, clusters=4, seed=20260727)

    assert first == second
    assert set(first) == set(centroids)
    assert set(first.values()) == {0, 1, 2, 3}
