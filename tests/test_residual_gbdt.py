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
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from residual_gbdt_evaluate import build_fold_assignments  # noqa: E402
from residual_gbdt_invariant_evaluate import (  # noqa: E402
    EXCLUDED_ABSOLUTE_POSITION_FEATURES,
    INVARIANT_FEATURE_INDICES,
    INVARIANT_FEATURE_NAMES,
    build_invariant_fold_assignments,
)
from residual_gbdt_shrink_evaluate import (  # noqa: E402
    CONFIRMATION_SPATIAL_FOLDS,
    RESIDUAL_SCALE,
    build_shrink_fold_assignments,
)

from rogii.residual_gbdt import (  # noqa: E402
    RESIDUAL_FEATURE_NAMES,
    ResidualFeatureConfig,
    build_residual_features,
    sample_suffix_indices,
    typewell_fingerprint,
)


def _synthetic_inputs(
    *,
    rows: int = 180,
    visible_rows: int = 72,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    md = np.arange(rows, dtype=float)
    x = 2_980_000.0 + 0.85 * md
    y = 1_070_000.0 + 0.40 * md
    z = -9_200.0 - 0.08 * md + 0.2 * np.sin(md / 31.0)
    truth = 11_200.0 + 0.025 * md + 0.35 * np.sin(md / 27.0)
    type_tvt = np.arange(11_170.0, 11_240.0, 0.25)
    type_gr = 85.0 + 15.0 * np.sin(type_tvt / 2.7) + 4.0 * np.cos(type_tvt / 0.8)
    gr = 1.2 * np.interp(truth, type_tvt, type_gr) - 8.0
    gr[100:112] = np.nan
    tvt_input = truth.copy()
    tvt_input[visible_rows:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": type_tvt, "GR": type_gr})
    return horizontal, typewell, truth


def test_features_use_only_test_available_columns_and_anchor_boundary() -> None:
    horizontal, typewell, truth = _synthetic_inputs()
    rich_horizontal = horizontal.copy()
    rich_horizontal["TVT"] = truth
    rich_horizontal["ANCC"] = np.linspace(-1_000.0, 1_000.0, len(horizontal))
    rich_typewell = typewell.copy()
    rich_typewell["Geology"] = "LEAK_SENTINEL"

    clean = build_residual_features(horizontal, typewell)
    rich = build_residual_features(rich_horizontal, rich_typewell)

    assert clean.features.shape == (len(horizontal) - 72, len(RESIDUAL_FEATURE_NAMES))
    assert clean.boundary_features.shape == (1, len(RESIDUAL_FEATURE_NAMES))
    assert np.array_equal(clean.features, rich.features, equal_nan=True)
    assert np.array_equal(clean.boundary_features, rich.boundary_features, equal_nan=True)
    assert rich.diagnostics["train_target_used"] is False
    assert rich.diagnostics["well_id_encoded"] is False
    assert rich.diagnostics["formation_columns_used"] is False
    assert rich.diagnostics["same_id_lookup"] is False
    assert rich.anchor_index == 71
    assert clean.boundary_features[0, RESIDUAL_FEATURE_NAMES.index("dmd")] == 0.0
    assert np.all(clean.evaluation_mask[72:])
    assert not np.any(clean.evaluation_mask[:72])


def test_simple_baselines_follow_the_visible_boundary_definitions() -> None:
    horizontal, typewell, truth = _synthetic_inputs()
    block = build_residual_features(horizontal, typewell)
    anchor_tvt = truth[71]
    anchor_u = anchor_tvt + horizontal.loc[71, "Z"]
    hidden_z = horizontal.loc[72:, "Z"].to_numpy()

    assert np.all(block.hold_last == anchor_tvt)
    assert np.allclose(block.u_boundary, anchor_u - hidden_z)
    assert block.slope_decay[0] == pytest.approx(anchor_tvt, abs=0.05)
    assert np.max(np.abs(block.slope_decay - anchor_tvt)) <= 40.0


def test_hidden_truth_mutation_cannot_change_features() -> None:
    horizontal, typewell, truth = _synthetic_inputs()
    first = horizontal.copy()
    first["TVT"] = truth
    second = horizontal.copy()
    second["TVT"] = truth
    second.loc[72:, "TVT"] += np.linspace(-50_000.0, 50_000.0, len(second) - 72)

    first_block = build_residual_features(first, typewell)
    second_block = build_residual_features(second, typewell)

    assert np.array_equal(first_block.features, second_block.features, equal_nan=True)
    assert np.array_equal(
        first_block.boundary_features,
        second_block.boundary_features,
        equal_nan=True,
    )


def test_sampling_is_deterministic_bounded_and_includes_endpoints() -> None:
    first = sample_suffix_indices(10_052, 96)
    second = sample_suffix_indices(10_052, 96)
    assert np.array_equal(first, second)
    assert len(first) == 96
    assert first[0] == 0
    assert first[-1] == 10_051
    assert np.all(np.diff(first) > 0)
    assert np.array_equal(sample_suffix_indices(4, 96), np.arange(4))
    with pytest.raises(ValueError):
        sample_suffix_indices(0, 96)


def test_typewell_fingerprint_ignores_geology_but_detects_numeric_change() -> None:
    _, typewell, _ = _synthetic_inputs()
    richer = typewell.copy()
    richer["Geology"] = "A"
    changed = typewell.copy()
    changed.loc[5, "GR"] += 0.01

    assert typewell_fingerprint(typewell) == typewell_fingerprint(richer)
    assert typewell_fingerprint(typewell) != typewell_fingerprint(changed)


def test_non_suffix_mask_is_rejected() -> None:
    horizontal, typewell, _ = _synthetic_inputs()
    horizontal.loc[120, "TVT_input"] = 11_203.0
    with pytest.raises(ValueError, match="contiguous suffix"):
        build_residual_features(
            horizontal,
            typewell,
            config=ResidualFeatureConfig(),
        )


def test_frozen_fold_assignments_are_deterministic_and_complete() -> None:
    records = []
    for index in range(30):
        cluster = index % 5
        records.append(
            {
                "fingerprint": f"type-{index // 2:02d}",
                "centroid_x": float(cluster * 100 + index / 100),
                "centroid_y": float((cluster % 2) * 100 + index / 200),
            }
        )

    first = build_fold_assignments(records)
    second = build_fold_assignments(records)

    assert first.keys() == second.keys()
    for protocol in first:
        assert np.array_equal(first[protocol], second[protocol])
        assert set(first[protocol]) == set(range(5))

    fingerprint_folds = first["typewell_fingerprint"]
    for fingerprint in {row["fingerprint"] for row in records}:
        indices = [index for index, row in enumerate(records) if row["fingerprint"] == fingerprint]
        assert len(set(fingerprint_folds[indices])) == 1


def test_invariant_profile_removes_only_absolute_xyz_features() -> None:
    retained = tuple(RESIDUAL_FEATURE_NAMES[index] for index in INVARIANT_FEATURE_INDICES)
    assert retained == INVARIANT_FEATURE_NAMES
    assert not set(EXCLUDED_ABSOLUTE_POSITION_FEATURES) & set(retained)
    assert len(retained) == len(RESIDUAL_FEATURE_NAMES) - 6
    assert {"dx", "dy", "dz", "anchor_tvt_scaled", "anchor_u_scaled"} <= set(retained)


def test_invariant_fold_assignments_add_deterministic_six_cluster_protocol() -> None:
    records = []
    for index in range(60):
        cluster = index % 6
        records.append(
            {
                "fingerprint": f"type-{index // 2:02d}",
                "centroid_x": float(cluster * 100 + index / 100),
                "centroid_y": float((cluster % 3) * 100 + index / 200),
            }
        )

    first = build_invariant_fold_assignments(records)
    second = build_invariant_fold_assignments(records)

    assert first.keys() == second.keys()
    assert set(first) == {
        "typewell_fingerprint",
        "spatial_cluster",
        "spatial_cluster_6_seed_20260726",
    }
    for protocol in first:
        assert np.array_equal(first[protocol], second[protocol])
    assert set(first["spatial_cluster_6_seed_20260726"]) == set(range(6))


def test_shrink_fold_assignments_add_deterministic_seven_cluster_protocol() -> None:
    records = []
    for index in range(70):
        cluster = index % CONFIRMATION_SPATIAL_FOLDS
        records.append(
            {
                "fingerprint": f"type-{index // 2:02d}",
                "centroid_x": float(cluster * 100 + index / 100),
                "centroid_y": float((cluster % 3) * 100 + index / 200),
            }
        )

    first = build_shrink_fold_assignments(records)
    second = build_shrink_fold_assignments(records)

    assert RESIDUAL_SCALE == 0.20
    assert first.keys() == second.keys()
    assert set(first) == {
        "typewell_fingerprint",
        "spatial_cluster",
        "spatial_cluster_6_seed_20260726",
        "spatial_cluster_7_seed_20260727",
    }
    for protocol in first:
        assert np.array_equal(first[protocol], second[protocol])
    assert set(first["spatial_cluster_7_seed_20260727"]) == set(range(7))
