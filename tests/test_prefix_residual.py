from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prefix_residual_evaluate import (  # noqa: E402
    CONFIRMATION_CLUSTER_RANKS,
    _discovery_gate,
    _spatial_split,
)

from rogii.prefix_residual import (  # noqa: E402
    PREFIX_RESIDUAL_CANDIDATES,
    PrefixResidualCandidate,
    apply_prefix_residual,
    candidate_by_name,
    estimate_visible_prefix_residual,
    prefix_residual_correction,
)


def test_frozen_candidate_family_is_small_unique_and_bounded() -> None:
    names = [candidate.name for candidate in PREFIX_RESIDUAL_CANDIDATES]

    assert len(names) == 4
    assert len(set(names)) == len(names)
    assert all(candidate.correction_cap <= 8.0 for candidate in PREFIX_RESIDUAL_CANDIDATES)
    assert all(candidate.decay_tau_md <= 480.0 for candidate in PREFIX_RESIDUAL_CANDIDATES)
    assert candidate_by_name(names[0]) is PREFIX_RESIDUAL_CANDIDATES[0]
    with pytest.raises(KeyError):
        candidate_by_name("not-a-candidate")


def test_level_correction_is_robust_bounded_and_decays() -> None:
    prefix_md = np.arange(100.0, 228.0)
    residual = np.full(128, 6.0)
    residual[::11] = 1_000.0
    hidden_md = np.arange(228.0, 1_229.0)
    candidate = candidate_by_name("level128_cap4_tau480")

    correction, diagnostics = prefix_residual_correction(
        hidden_md,
        227.0,
        prefix_md,
        residual,
        candidate,
    )

    assert diagnostics["median_level"] == pytest.approx(6.0)
    assert correction[0] < 4.0
    assert correction[0] > 3.9
    assert np.max(np.abs(correction)) <= 4.0
    assert np.all(np.diff(correction) < 0.0)
    assert correction[-1] < correction[0] * 0.13


def test_slope_is_clipped_and_extrapolation_stops_at_frozen_horizon() -> None:
    prefix_md = np.arange(0.0, 128.0)
    residual = -3.0 + 0.2 * (prefix_md - prefix_md[-1])
    candidate = candidate_by_name("level_slope128_cap4_tau480")
    estimate = estimate_visible_prefix_residual(prefix_md, residual, candidate)

    assert estimate["slope"] == pytest.approx(candidate.maximum_abs_slope)

    hidden_md = np.array([128.0, 247.0, 248.0, 500.0])
    correction, _ = prefix_residual_correction(
        hidden_md,
        127.0,
        prefix_md,
        residual,
        candidate,
    )
    decay = np.exp(-(hidden_md - 127.0) / candidate.decay_tau_md)
    undececayed = correction / decay
    assert undececayed[1] == pytest.approx(undececayed[2])
    assert undececayed[2] == pytest.approx(undececayed[3])


def test_application_cannot_depend_on_hidden_truth() -> None:
    baseline = np.linspace(2_000.0, 2_010.0, 100)
    hidden_md = np.arange(500.0, 600.0)
    prefix_md = np.arange(372.0, 500.0)
    prefix_residual = np.linspace(1.5, 2.0, 128)
    candidate = candidate_by_name("level64_cap4_tau240")

    first, first_diagnostics = apply_prefix_residual(
        baseline,
        hidden_md,
        499.0,
        prefix_md,
        prefix_residual,
        candidate,
    )
    unrelated_hidden_truth = np.linspace(-1e6, 1e6, 100)
    second, second_diagnostics = apply_prefix_residual(
        baseline,
        hidden_md,
        499.0,
        prefix_md,
        prefix_residual,
        candidate,
    )

    assert np.array_equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert unrelated_hidden_truth.shape == first.shape


def test_invalid_prefix_and_hidden_shapes_are_rejected() -> None:
    candidate = PrefixResidualCandidate(
        name="test",
        residual_window=8,
        correction_cap=2.0,
        decay_tau_md=100.0,
    )
    with pytest.raises(ValueError, match="eight finite"):
        estimate_visible_prefix_residual(
            np.arange(7.0),
            np.ones(7),
            candidate,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        prefix_residual_correction(
            np.array([10.0, 9.0]),
            8.0,
            np.arange(8.0),
            np.ones(8),
            candidate,
        )


def test_spatial_discovery_confirmation_split_is_deterministic_and_disjoint() -> None:
    rng = np.random.default_rng(20260725)
    centers = np.array(
        [
            [-50.0, -20.0],
            [-30.0, 30.0],
            [-10.0, -30.0],
            [10.0, 30.0],
            [30.0, -30.0],
            [50.0, 20.0],
        ]
    )
    centroids = np.vstack([center + rng.normal(0.0, 0.3, size=(8, 2)) for center in centers])
    paths = [Path(f"{index:04d}__horizontal_well.csv") for index in range(len(centroids))]

    first, first_report = _spatial_split(paths, centroids)
    second, second_report = _spatial_split(paths, centroids)
    confirmation = np.flatnonzero(np.isin(first, CONFIRMATION_CLUSTER_RANKS))
    discovery = np.flatnonzero(~np.isin(first, CONFIRMATION_CLUSTER_RANKS))

    assert np.array_equal(first, second)
    assert first_report == second_report
    assert not set(confirmation) & set(discovery)
    assert sorted([*confirmation, *discovery]) == list(range(len(paths)))
    assert {row["role"] for row in first_report["cluster_summary"]} == {
        "discovery",
        "confirmation",
    }


def test_discovery_gate_rejects_small_or_spatially_unstable_gain() -> None:
    bootstrap = {"ci95_upper": -0.001}
    stable_clusters = [
        {"pooled_delta": -0.03},
        {"pooled_delta": -0.02},
        {"pooled_delta": -0.01},
        {"pooled_delta": 0.01},
    ]
    passing_delta = {
        "pooled_rmse": -0.03,
        "well_macro_rmse": -0.01,
        "well_p95_rmse": 0.1,
        "worst_well_rmse": 0.5,
    }

    assert _discovery_gate(passing_delta, bootstrap, stable_clusters)["decision"] == "PASS"
    too_small = dict(passing_delta, pooled_rmse=-0.019)
    assert _discovery_gate(too_small, bootstrap, stable_clusters)["decision"] == "FAIL"
    unstable = [
        {"pooled_delta": -0.10},
        {"pooled_delta": -0.10},
        {"pooled_delta": 0.04},
        {"pooled_delta": 0.04},
    ]
    assert _discovery_gate(passing_delta, bootstrap, unstable)["decision"] == "FAIL"
