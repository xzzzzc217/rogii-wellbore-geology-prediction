from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_targetfree_selector import (  # noqa: E402
    ALL_METHODS,
    CORE_FEATURES,
    EXPANDED_GR_FEATURES,
    _pooled_rmse,
    _signed_log1p,
    extract_features,
    make_protocol_folds,
)


def _record() -> dict:
    return {
        "well_id": "well-a",
        "scored_rows": 80,
        "rmse": {method: float(index + 1) for index, method in enumerate(ALL_METHODS)},
        "diagnostics": {
            "target_free": True,
            "formation_columns_used": False,
            "leaderboard_bias_used": False,
            "precomputed_predictions_used": False,
            "same_id_lookup": False,
            "train_target_used": False,
            "row_count": 100,
            "hidden_rows": 80,
            "typewell_tvt_min": 1000.0,
            "typewell_tvt_max": 1100.0,
            "anchor_tvt": 1040.0,
            "final_tvt_min": 1035.0,
            "final_tvt_max": 1055.0,
            "calibration_trust": 0.8,
            "calibration": {
                "correlation": 0.7,
                "gain": 0.9,
                "residual_scale": 2.0,
            },
            "particle_filter": {
                "tvt_slope_prior": 0.02,
                "posterior_std_median": 1.2,
                "posterior_std_p90": 2.3,
                "average_log_evidence": [-1.0, -2.0],
                "resamples": [3, 4],
                "seed_weights": [0.25, 0.75],
            },
            "beam_viterbi": {
                "average_cost": 1.1,
                "terminal_margin": 4.0,
            },
            "hidden_gr_comparable_rows": 60,
            "hidden_gr_rmse_by_component": {
                method: float(index + 2) for index, method in enumerate(ALL_METHODS)
            },
        },
    }


def _geometry() -> dict[str, float]:
    return {
        "centroid_x": 10.0,
        "centroid_y": 20.0,
        "x_span": 3.0,
        "y_span": 4.0,
        "xy_end_to_end": 5.0,
        "xy_path_length_per_row": 0.1,
    }


def test_core_and_expanded_features_have_exact_allow_list() -> None:
    core = extract_features(_record(), _geometry(), include_hidden_gr=False)
    expanded = extract_features(_record(), _geometry(), include_hidden_gr=True)

    assert set(core) == set(CORE_FEATURES)
    assert set(expanded) == set(CORE_FEATURES + EXPANDED_GR_FEATURES)
    assert not any("hidden_gr" in name for name in core)


def test_feature_extraction_is_independent_of_tvt_error_labels() -> None:
    original = _record()
    changed = copy.deepcopy(original)
    changed["rmse"] = {method: 10_000.0 + index for index, method in enumerate(ALL_METHODS)}

    assert extract_features(
        original, _geometry(), include_hidden_gr=False
    ) == extract_features(changed, _geometry(), include_hidden_gr=False)
    assert extract_features(
        original, _geometry(), include_hidden_gr=True
    ) == extract_features(changed, _geometry(), include_hidden_gr=True)


def test_spatial_and_shuffled_folds_are_complete_and_disjoint() -> None:
    rng = np.random.default_rng(123)
    centroids = np.vstack(
        [
            rng.normal(loc=(cluster * 20.0, cluster * -10.0), scale=0.2, size=(8, 2))
            for cluster in range(5)
        ]
    )
    protocols = make_protocol_folds(centroids)

    for folds in protocols.values():
        validation = np.concatenate([fold[1] for fold in folds])
        assert sorted(validation.tolist()) == list(range(len(centroids)))
        for train, held_out in folds:
            assert not np.intersect1d(train, held_out).size


def test_pooled_rmse_is_row_weighted() -> None:
    mse = np.asarray([[1.0, 9.0], [16.0, 4.0]])
    rows = np.asarray([1.0, 3.0])
    indices = np.asarray([0, 1])
    choices = np.asarray([0, 1])

    assert _pooled_rmse(mse, rows, indices, choices) == np.sqrt(13.0 / 4.0)


def test_signed_log_transform_preserves_order_and_sign() -> None:
    values = np.asarray([-100.0, -1.0, 0.0, 2.0, 200.0])
    transformed = _signed_log1p(values)

    assert np.all(np.diff(transformed) > 0)
    assert np.array_equal(np.sign(transformed), np.sign(values))
