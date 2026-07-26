from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
    TrajectoryConfig,
    beam_viterbi_trajectory,
    calibrate_heel_gr,
    hold_last_prediction,
    interpolate_typewell_gr,
    predict_trajectory,
)


def _synthetic_well(
    *,
    rows: int = 180,
    visible_rows: int = 75,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    md = np.arange(rows, dtype=float)
    z = -1_000.0 + 0.018 * md + 0.15 * np.sin(md / 37.0)
    true_tvt = 2_000.0 + 0.030 * md + 0.25 * np.sin(md / 31.0)
    typewell_tvt = np.arange(1_970.0, 2_035.0, 0.20)
    typewell_gr = (
        82.0
        + 19.0 * np.sin((typewell_tvt - 1_970.0) / 1.9)
        + 8.0 * np.sin((typewell_tvt - 1_970.0) / 0.61)
    )
    reference = np.interp(true_tvt, typewell_tvt, typewell_gr)
    horizontal_gr = 1.35 * reference + 7.5
    tvt_input = true_tvt.copy()
    tvt_input[visible_rows:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 500_000.0 + md,
            "Y": 3_000_000.0 + 0.25 * md,
            "Z": z,
            "GR": horizontal_gr,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell, true_tvt


def _small_config() -> TrajectoryConfig:
    return TrajectoryConfig(
        heel_window=60,
        calibration_min_rows=20,
        slope_window=20,
        pf_particles=36,
        pf_seeds=(0, 1, 2),
        beam_width=8,
        maximum_tvt_departure=40.0,
    )


def test_interpolation_and_robust_heel_gain_offset_calibration() -> None:
    typewell_tvt = np.linspace(100.0, 140.0, 161)
    typewell_gr = 70.0 + 15.0 * np.sin(typewell_tvt / 3.0)
    known_tvt = np.linspace(104.0, 136.0, 120)
    reference = interpolate_typewell_gr(known_tvt, typewell_tvt, typewell_gr)
    horizontal_gr = 1.7 * reference - 12.0
    horizontal_gr[::19] += 70.0

    calibration = calibrate_heel_gr(
        horizontal_gr,
        known_tvt,
        typewell_tvt,
        typewell_gr,
        config=TrajectoryConfig(
            heel_window=120,
            calibration_min_rows=20,
            minimum_gr_scale=1.0,
        ),
    )

    assert calibration.status == "ok"
    assert calibration.rows == 120
    assert calibration.gain == pytest.approx(1.7, abs=0.03)
    assert calibration.offset == pytest.approx(-12.0, abs=1.0)
    assert calibration.correlation > 0.70
    outside = interpolate_typewell_gr(
        np.array([99.0, 120.0, 141.0]),
        typewell_tvt,
        typewell_gr,
    )
    assert np.isnan(outside[[0, 2]]).all()
    assert np.isfinite(outside[1])


def test_hold_last_fills_only_missing_values() -> None:
    values = np.array([10.0, 11.0, 12.5, np.nan, np.nan])
    assert np.array_equal(
        hold_last_prediction(values),
        np.array([10.0, 11.0, 12.5, 12.5, 12.5]),
    )
    with pytest.raises(ValueError, match="visible TVT"):
        hold_last_prediction(np.array([np.nan, np.nan]))


def test_prediction_is_deterministic_and_improves_synthetic_hold() -> None:
    horizontal, typewell, truth = _synthetic_well()
    config = _small_config()

    first = predict_trajectory(horizontal, typewell, config=config)
    second = predict_trajectory(horizontal, typewell, config=config)

    assert np.array_equal(first.prediction, second.prediction)
    assert (
        first.diagnostics["particle_filter"]["seed_weights"]
        == second.diagnostics["particle_filter"]["seed_weights"]
    )
    assert sum(first.diagnostics["particle_filter"]["seed_weights"]) == pytest.approx(1.0)
    mask = first.evaluation_mask
    hold_rmse = np.sqrt(np.mean((first.components["hold_last"][mask] - truth[mask]) ** 2))
    final_rmse = np.sqrt(np.mean((first.prediction[mask] - truth[mask]) ** 2))
    assert final_rmse < hold_rmse
    assert np.isfinite(first.prediction).all()


def test_beam_path_uses_constrained_typewell_grid_moves() -> None:
    horizontal, typewell, _ = _synthetic_well()
    config = _small_config()
    prediction, diagnostics = beam_viterbi_trajectory(
        horizontal,
        typewell,
        config=config,
    )
    hidden = horizontal["TVT_input"].isna().to_numpy()
    hidden_prediction = prediction[hidden]
    grid = typewell["TVT"].to_numpy(dtype=float)
    indices = np.array([int(np.argmin(np.abs(grid - value))) for value in hidden_prediction])

    assert diagnostics["status"] == "ok"
    assert np.allclose(hidden_prediction, grid[indices])
    assert np.max(np.abs(np.diff(indices))) <= max(abs(move) for move in config.beam_index_moves)
    assert diagnostics["states_evaluated"] > len(hidden_prediction)


def test_train_only_extra_columns_cannot_change_prediction() -> None:
    horizontal, typewell, truth = _synthetic_well()
    config = _small_config()
    with_train_only = horizontal.copy()
    with_train_only["TVT"] = truth
    with_train_only["EGFDU"] = np.linspace(-9_000.0, 9_000.0, len(horizontal))
    richer_typewell = typewell.copy()
    richer_typewell["Geology"] = "LEAK_SENTINEL"

    clean = predict_trajectory(
        horizontal.loc[:, HORIZONTAL_INPUT_COLUMNS],
        typewell.loc[:, TYPEWELL_INPUT_COLUMNS],
        config=config,
    )
    rich = predict_trajectory(with_train_only, richer_typewell, config=config)

    assert np.array_equal(clean.prediction, rich.prediction)
    assert rich.diagnostics["horizontal_columns_used"] == list(HORIZONTAL_INPUT_COLUMNS)
    assert rich.diagnostics["typewell_columns_used"] == list(TYPEWELL_INPUT_COLUMNS)
    assert rich.diagnostics["same_id_lookup"] is False
    assert rich.diagnostics["train_target_used"] is False
    assert rich.diagnostics["formation_columns_used"] is False
    assert rich.diagnostics["leaderboard_bias_used"] is False
    assert rich.diagnostics["precomputed_predictions_used"] is False


def test_short_typewell_returns_audited_hold_last_fallback() -> None:
    horizontal, _, _ = _synthetic_well()
    typewell = pd.DataFrame({"TVT": [2_000.0], "GR": [80.0]})
    result = predict_trajectory(horizontal, typewell, config=_small_config())
    hidden = result.evaluation_mask

    assert result.diagnostics["status"] == "fallback"
    assert result.diagnostics["fallback_reason"] == ("typewell_has_fewer_than_two_finite_rows")
    assert np.unique(result.prediction[hidden]).size == 1
    assert result.diagnostics["tracker_weight"] == 0.0


def test_non_suffix_missing_tvt_input_is_rejected() -> None:
    horizontal, typewell, _ = _synthetic_well()
    horizontal.loc[90, "TVT_input"] = 2_001.0
    with pytest.raises(ValueError, match="contiguous suffix"):
        predict_trajectory(horizontal, typewell, config=_small_config())
