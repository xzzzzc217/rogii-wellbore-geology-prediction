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

from rogii.u_continuity import apply_u_continuity  # noqa: E402


def test_exact_guard640_formula_and_only_hidden_suffix_changes() -> None:
    prediction = np.array([700.0, 701.0, 905.0, 906.0])
    md = np.array([0.0, 10.0, 20.0, 40.0])
    z = np.array([100.0, 101.0, 102.0, 103.0])
    tvt_input = np.array([900.0, 901.0, np.nan, np.nan])

    refined, diagnostics = apply_u_continuity(
        prediction,
        md,
        z,
        tvt_input,
        cap=8.0,
        tau=240.0,
    )

    gap = (905.0 + 102.0) - (901.0 + 101.0)
    expected_move = -gap * np.exp(-np.array([10.0, 30.0]) / 240.0)
    assert np.array_equal(refined[:2], prediction[:2])
    assert refined[2:] == pytest.approx(prediction[2:] + expected_move)
    assert diagnostics.boundary_gap_u_before == pytest.approx(gap)
    assert diagnostics.first_move == pytest.approx(expected_move[0])
    assert diagnostics.boundary_gap_u_after == pytest.approx(gap + expected_move[0])
    assert np.array_equal(prediction, np.array([700.0, 701.0, 905.0, 906.0]))


@pytest.mark.parametrize("gap", [20.0, -20.0])
def test_cap_sign_first_point_and_exponential_decay(gap: float) -> None:
    md = np.array([100.0, 101.0, 121.0, 361.0])
    z = np.zeros(4)
    tvt_input = np.array([1000.0, 1000.0, np.nan, np.nan])
    prediction = np.array([1000.0, 1000.0, 1000.0 + gap, 1000.0 + gap])

    refined, diagnostics = apply_u_continuity(
        prediction,
        md,
        z,
        tvt_input,
        cap=8.0,
        tau=240.0,
    )

    signed_cap = np.clip(gap, -8.0, 8.0)
    first = -signed_cap * np.exp(-20.0 / 240.0)
    last = -signed_cap * np.exp(-260.0 / 240.0)
    assert refined[2] - prediction[2] == pytest.approx(first)
    assert refined[3] - prediction[3] == pytest.approx(last)
    assert abs(first) <= 8.0
    assert abs(last) < abs(first)
    assert diagnostics.max_abs_move == pytest.approx(abs(first))


def test_requires_official_contiguous_hidden_suffix() -> None:
    prediction = np.arange(5, dtype=float)
    md = np.arange(5, dtype=float)
    z = np.zeros(5)

    with pytest.raises(ValueError, match="contiguous hidden suffix"):
        apply_u_continuity(
            prediction,
            md,
            z,
            np.array([1.0, np.nan, 2.0, np.nan, np.nan]),
        )
    with pytest.raises(ValueError, match="immediately follow"):
        apply_u_continuity(
            prediction,
            md,
            z,
            np.array([1.0, 2.0, np.nan, np.nan, 3.0]),
        )


def test_transform_has_no_ground_truth_input() -> None:
    parameters = set(inspect.signature(apply_u_continuity).parameters)
    assert parameters == {"prediction", "md", "z", "tvt_input", "cap", "tau"}

    with pytest.raises(TypeError):
        apply_u_continuity(
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, np.nan]),
            truth=np.array([1.0, 2.0]),  # type: ignore[call-arg]
        )
