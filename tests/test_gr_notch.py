from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.gr_notch import notch_strongest_periodic_component  # noqa: E402


def test_removes_strongest_periodic_bin_and_preserves_missing_positions() -> None:
    rows = 256
    index = np.arange(rows, dtype=float)
    values = 80.0 + 12.0 * np.sin(2.0 * np.pi * 8.0 * index / rows)
    values[17] = np.nan

    filtered, diagnostics = notch_strongest_periodic_component(values)

    assert diagnostics.applied is True
    assert diagnostics.peak_bin == 8
    assert diagnostics.removed_bins == (7, 8, 9)
    assert np.isnan(filtered[17])
    finite = np.isfinite(values)
    assert np.std(filtered[finite]) < 0.05 * np.std(values[finite])
    assert diagnostics.rms_move > 1.0


def test_short_and_constant_series_are_deterministic_no_ops() -> None:
    short = np.linspace(1.0, 2.0, 32)
    short_filtered, short_diagnostics = notch_strongest_periodic_component(short)
    assert np.array_equal(short_filtered, short)
    assert short_diagnostics.reason == "too_few_finite_rows"

    constant = np.full(128, 7.5)
    first, first_diagnostics = notch_strongest_periodic_component(constant)
    second, second_diagnostics = notch_strongest_periodic_component(constant)
    assert np.array_equal(first, constant)
    assert np.array_equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics.applied is False


def test_transform_has_no_truth_or_identifier_input() -> None:
    parameters = set(inspect.signature(notch_strongest_periodic_component).parameters)
    assert parameters == {"gr", "min_finite", "notch_radius"}

    with pytest.raises(TypeError):
        notch_strongest_periodic_component(
            np.arange(128, dtype=float),
            truth=np.arange(128, dtype=float),  # type: ignore[call-arg]
        )


def test_rejects_invalid_shapes_and_controls() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        notch_strongest_periodic_component(np.ones((4, 4)))
    with pytest.raises(ValueError, match="at least two"):
        notch_strongest_periodic_component(np.ones(64), min_finite=1)
    with pytest.raises(ValueError, match="cannot be negative"):
        notch_strongest_periodic_component(np.ones(64), notch_radius=-1)
