"""Leakage-safe row features for a residual GBDT trajectory model.

The model target is the residual of true ``U = TVT + Z`` from the simple
hold-last-TVT path:

``(TVT + Z) - (last_visible_TVT + Z) == TVT - last_visible_TVT``.

Only fields available in the hidden test are accepted.  Full suffix TVT is
kept outside this module and may be used by an evaluator only as a label.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from rogii.trajectory import (
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
    TrajectoryConfig,
    calibrate_heel_gr,
    interpolate_typewell_gr,
)

RESIDUAL_FEATURE_NAMES = (
    "dmd",
    "log1p_dmd",
    "suffix_progress",
    "log1p_suffix_rows",
    "visible_fraction",
    "x_scaled",
    "y_scaled",
    "z_scaled",
    "anchor_x_scaled",
    "anchor_y_scaled",
    "anchor_z_scaled",
    "dx",
    "dy",
    "dz",
    "dxy",
    "dx_per_dmd",
    "dy_per_dmd",
    "dz_per_dmd",
    "local_dx_per_dmd",
    "local_dy_per_dmd",
    "local_dz_per_dmd",
    "anchor_tvt_scaled",
    "anchor_u_scaled",
    "visible_tvt_slope_short",
    "visible_tvt_slope_long",
    "visible_u_slope_short",
    "visible_u_slope_long",
    "visible_tvt_slope_gap",
    "slope_decay_delta",
    "gr",
    "gr_missing",
    "gr_delta_anchor",
    "gr_local_diff_per_md",
    "gr_roll16_mean",
    "gr_roll16_std",
    "gr_roll64_mean",
    "gr_roll64_std",
    "visible_gr_mean",
    "visible_gr_std",
    "visible_gr_missing_fraction",
    "heel_gain",
    "heel_offset_scaled",
    "heel_residual_scale",
    "heel_correlation",
    "type_gr_anchor",
    "type_gr_u_hold",
    "type_gr_slope",
    "gr_minus_type_anchor",
    "gr_minus_type_u_hold",
    "gr_minus_type_slope",
    "type_gradient_anchor",
    "type_gradient_u_hold",
    "type_gradient_slope",
    "type_match_delta",
    "type_match_abs_error",
    "type_lower_distance",
    "type_upper_distance",
)


@dataclass(frozen=True)
class ResidualFeatureConfig:
    """Frozen target-free feature and baseline controls."""

    short_slope_rows: int = 64
    long_slope_rows: int = 256
    maximum_abs_visible_slope: float = 0.20
    slope_decay_tau_md: float = 240.0
    slope_decay_cap: float = 40.0
    calibration_rows: int = 512
    calibration_min_rows: int = 20
    type_gradient_half_width: float = 2.0
    type_match_offsets: tuple[float, ...] = (
        -40.0,
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        0.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
    )

    def __post_init__(self) -> None:
        if self.short_slope_rows < 2 or self.long_slope_rows < self.short_slope_rows:
            raise ValueError("visible slope windows are invalid")
        if self.maximum_abs_visible_slope <= 0:
            raise ValueError("maximum_abs_visible_slope must be positive")
        if self.slope_decay_tau_md <= 0 or self.slope_decay_cap <= 0:
            raise ValueError("slope-decay controls must be positive")
        if self.calibration_rows < 2 or self.calibration_min_rows < 2:
            raise ValueError("calibration row controls must be at least two")
        if self.type_gradient_half_width <= 0 or not self.type_match_offsets:
            raise ValueError("typewell feature controls are invalid")
        if 0.0 not in self.type_match_offsets:
            raise ValueError("type_match_offsets must include zero")


@dataclass(frozen=True)
class ResidualFeatureBlock:
    """Target-free suffix features and simple visible-prefix baselines."""

    features: np.ndarray
    boundary_features: np.ndarray
    evaluation_mask: np.ndarray
    hidden_indices: np.ndarray
    hold_last: np.ndarray
    u_boundary: np.ndarray
    slope_decay: np.ndarray
    anchor_index: int
    anchor_tvt: float
    anchor_u: float
    diagnostics: dict[str, Any]


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"required column is missing: {name}")
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tvt = _numeric_column(typewell, "TVT")
    gr = _numeric_column(typewell, "GR")
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if int(valid.sum()) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    tvt = tvt[valid]
    gr = gr[valid]
    order = np.argsort(tvt, kind="stable")
    tvt = tvt[order]
    gr = gr[order]
    unique, first, counts = np.unique(tvt, return_index=True, return_counts=True)
    if np.any(counts > 1):
        gr = np.add.reduceat(gr, first) / counts
        tvt = unique
    return tvt.astype(float, copy=False), gr.astype(float, copy=False)


def typewell_fingerprint(typewell: pd.DataFrame) -> str:
    """Return a stable fingerprint of the test-available typewell columns."""

    tvt, gr = _prepare_typewell(typewell.loc[:, TYPEWELL_INPUT_COLUMNS])
    canonical = np.column_stack((tvt, gr)).astype("<f8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _suffix_boundary(tvt_input: np.ndarray) -> tuple[np.ndarray, int, int]:
    finite = np.isfinite(tvt_input)
    missing = ~finite
    if not np.any(missing):
        raise ValueError("TVT_input has no masked suffix")
    first_hidden = int(np.flatnonzero(missing)[0])
    if first_hidden == 0:
        raise ValueError("TVT_input has no visible prefix")
    if np.any(finite[first_hidden:]):
        raise ValueError("TVT_input missing values must form one contiguous suffix")
    return missing, first_hidden, first_hidden - 1


def _linear_slope(x: np.ndarray, y: np.ndarray, rows: int, clip: float) -> float:
    start = max(0, len(x) - rows)
    x = np.asarray(x[start:], dtype=float)
    y = np.asarray(y[start:], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return 0.0
    x = x[valid]
    y = y[valid]
    centered_x = x - float(np.mean(x))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        return 0.0
    slope = float(np.dot(centered_x, y - float(np.mean(y))) / denominator)
    return float(np.clip(slope, -clip, clip))


def _median_step_slope(x: np.ndarray, y: np.ndarray, rows: int, clip: float) -> float:
    start = max(0, len(x) - rows)
    dx = np.diff(np.asarray(x[start:], dtype=float))
    dy = np.diff(np.asarray(y[start:], dtype=float))
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-9)
    if not np.any(valid):
        return 0.0
    slope = float(np.median(dy[valid] / dx[valid]))
    return float(np.clip(slope, -clip, clip))


def _trailing_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    cleaned = np.where(finite, values, 0.0)
    counts = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    totals = np.concatenate(([0.0], np.cumsum(cleaned, dtype=float)))
    squares = np.concatenate(([0.0], np.cumsum(cleaned * cleaned, dtype=float)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    count = counts[end] - counts[start]
    total = totals[end] - totals[start]
    square = squares[end] - squares[start]
    mean = np.full(len(values), np.nan, dtype=float)
    std = np.full(len(values), np.nan, dtype=float)
    valid = count > 0
    mean[valid] = total[valid] / count[valid]
    variance = np.zeros(len(values), dtype=float)
    variance[valid] = square[valid] / count[valid] - mean[valid] ** 2
    std[valid] = np.sqrt(np.maximum(variance[valid], 0.0))
    return mean, std


def _step_derivative(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    delta_value = np.diff(values)
    delta_md = np.diff(md)
    valid = np.isfinite(delta_value) & np.isfinite(delta_md) & (np.abs(delta_md) > 1e-9)
    result[1:][valid] = delta_value[valid] / delta_md[valid]
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-9),
    )
    return result


def _interp(
    query: np.ndarray,
    typewell_tvt: np.ndarray,
    calibrated_typewell_gr: np.ndarray,
) -> np.ndarray:
    shape = np.asarray(query).shape
    flat = np.asarray(query, dtype=float).reshape(-1)
    result = interpolate_typewell_gr(flat, typewell_tvt, calibrated_typewell_gr)
    return np.asarray(result, dtype=float).reshape(shape)


def _type_gradient(
    query: np.ndarray,
    typewell_tvt: np.ndarray,
    calibrated_typewell_gr: np.ndarray,
    half_width: float,
) -> np.ndarray:
    upper = _interp(query + half_width, typewell_tvt, calibrated_typewell_gr)
    lower = _interp(query - half_width, typewell_tvt, calibrated_typewell_gr)
    return (upper - lower) / (2.0 * half_width)


def _local_type_match(
    query_tvt: np.ndarray,
    observed_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    calibrated_typewell_gr: np.ndarray,
    offsets: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    offset_array = np.asarray(offsets, dtype=float)
    candidates = query_tvt[:, None] + offset_array[None, :]
    candidate_gr = _interp(candidates, typewell_tvt, calibrated_typewell_gr)
    distance = np.abs(candidate_gr - observed_gr[:, None])
    usable = np.isfinite(distance)
    safe_distance = np.where(usable, distance, np.inf)
    best = np.argmin(safe_distance, axis=1)
    row = np.arange(len(query_tvt))
    best_error = safe_distance[row, best]
    best_delta = offset_array[best].astype(float, copy=True)
    invalid = ~np.isfinite(observed_gr) | ~np.isfinite(best_error)
    best_error[invalid] = np.nan
    best_delta[invalid] = np.nan
    return best_delta, best_error


def sample_suffix_indices(row_count: int, maximum_rows: int) -> np.ndarray:
    """Return deterministic, evenly spaced suffix indices including both ends."""

    if row_count <= 0 or maximum_rows <= 0:
        raise ValueError("row_count and maximum_rows must be positive")
    if row_count <= maximum_rows:
        return np.arange(row_count, dtype=np.int64)
    indices = np.rint(np.linspace(0, row_count - 1, maximum_rows)).astype(np.int64)
    indices = np.unique(indices)
    if indices[0] != 0 or indices[-1] != row_count - 1:
        raise AssertionError("deterministic sample must include both suffix ends")
    return indices


def build_residual_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    config: ResidualFeatureConfig | None = None,
) -> ResidualFeatureBlock:
    """Build target-free features for one masked horizontal-well suffix."""

    cfg = config or ResidualFeatureConfig()
    horizontal_model = horizontal.loc[:, HORIZONTAL_INPUT_COLUMNS]
    typewell_model = typewell.loc[:, TYPEWELL_INPUT_COLUMNS]

    md = _numeric_column(horizontal_model, "MD")
    x = _numeric_column(horizontal_model, "X")
    y = _numeric_column(horizontal_model, "Y")
    z = _numeric_column(horizontal_model, "Z")
    gr = _numeric_column(horizontal_model, "GR")
    tvt_input = _numeric_column(horizontal_model, "TVT_input")
    if not (len(md) == len(x) == len(y) == len(z) == len(gr) == len(tvt_input)):
        raise ValueError("horizontal columns have inconsistent lengths")
    if len(md) < 3:
        raise ValueError("horizontal well has too few rows")

    evaluation_mask, first_hidden, anchor_index = _suffix_boundary(tvt_input)
    hidden_indices = np.flatnonzero(evaluation_mask)
    query_indices = np.concatenate(([anchor_index], hidden_indices))
    anchor_tvt = float(tvt_input[anchor_index])
    anchor_z = float(z[anchor_index])
    anchor_u = anchor_tvt + anchor_z
    if not all(math.isfinite(value) for value in (anchor_tvt, anchor_z, anchor_u)):
        raise ValueError("visible boundary TVT/Z is non-finite")

    visible_slice = slice(0, first_hidden)
    short_tvt_slope = _median_step_slope(
        md[visible_slice],
        tvt_input[visible_slice],
        cfg.short_slope_rows,
        cfg.maximum_abs_visible_slope,
    )
    long_tvt_slope = _linear_slope(
        md[visible_slice],
        tvt_input[visible_slice],
        cfg.long_slope_rows,
        cfg.maximum_abs_visible_slope,
    )
    visible_u = tvt_input[visible_slice] + z[visible_slice]
    short_u_slope = _median_step_slope(
        md[visible_slice],
        visible_u,
        cfg.short_slope_rows,
        2.0,
    )
    long_u_slope = _linear_slope(
        md[visible_slice],
        visible_u,
        cfg.long_slope_rows,
        2.0,
    )

    gr_roll16_mean, gr_roll16_std = _trailing_mean_std(gr, 16)
    gr_roll64_mean, gr_roll64_std = _trailing_mean_std(gr, 64)
    gr_derivative = _step_derivative(gr, md)
    x_derivative = _step_derivative(x, md)
    y_derivative = _step_derivative(y, md)
    z_derivative = _step_derivative(z, md)
    visible_gr = gr[visible_slice]
    finite_visible_gr = visible_gr[np.isfinite(visible_gr)]
    visible_gr_mean = float(np.mean(finite_visible_gr)) if finite_visible_gr.size else float("nan")
    visible_gr_std = float(np.std(finite_visible_gr)) if finite_visible_gr.size else float("nan")
    visible_gr_missing = float(1.0 - finite_visible_gr.size / max(1, len(visible_gr)))
    anchor_gr = (
        float(gr[anchor_index]) if math.isfinite(float(gr[anchor_index])) else visible_gr_mean
    )

    typewell_tvt, typewell_gr = _prepare_typewell(typewell_model)
    calibration_cfg = TrajectoryConfig(
        heel_window=cfg.calibration_rows,
        calibration_min_rows=cfg.calibration_min_rows,
    )
    calibration = calibrate_heel_gr(
        gr[visible_slice],
        tvt_input[visible_slice],
        typewell_tvt,
        typewell_gr,
        config=calibration_cfg,
    )
    calibrated_typewell_gr = calibration.gain * typewell_gr + calibration.offset

    query_md = md[query_indices]
    query_x = x[query_indices]
    query_y = y[query_indices]
    query_z = z[query_indices]
    query_gr = gr[query_indices]
    dmd = query_md - md[anchor_index]
    dx = query_x - x[anchor_index]
    dy = query_y - y[anchor_index]
    dz = query_z - anchor_z
    horizon_md = max(float(md[-1] - md[anchor_index]), 1.0)
    suffix_progress = np.clip(dmd / horizon_md, 0.0, 1.0)
    slope_decay_delta = (
        short_tvt_slope
        * cfg.slope_decay_tau_md
        * (1.0 - np.exp(-np.maximum(dmd, 0.0) / cfg.slope_decay_tau_md))
    )
    slope_decay_delta = np.clip(
        slope_decay_delta,
        -cfg.slope_decay_cap,
        cfg.slope_decay_cap,
    )

    u_hold_candidate_tvt = anchor_u - query_z
    slope_candidate_tvt = anchor_tvt + slope_decay_delta
    type_gr_anchor = _interp(
        np.full(len(query_indices), anchor_tvt),
        typewell_tvt,
        calibrated_typewell_gr,
    )
    type_gr_u_hold = _interp(u_hold_candidate_tvt, typewell_tvt, calibrated_typewell_gr)
    type_gr_slope = _interp(slope_candidate_tvt, typewell_tvt, calibrated_typewell_gr)
    type_gradient_anchor = _type_gradient(
        np.full(len(query_indices), anchor_tvt),
        typewell_tvt,
        calibrated_typewell_gr,
        cfg.type_gradient_half_width,
    )
    type_gradient_u_hold = _type_gradient(
        u_hold_candidate_tvt,
        typewell_tvt,
        calibrated_typewell_gr,
        cfg.type_gradient_half_width,
    )
    type_gradient_slope = _type_gradient(
        slope_candidate_tvt,
        typewell_tvt,
        calibrated_typewell_gr,
        cfg.type_gradient_half_width,
    )
    type_match_delta, type_match_error = _local_type_match(
        slope_candidate_tvt,
        query_gr,
        typewell_tvt,
        calibrated_typewell_gr,
        cfg.type_match_offsets,
    )

    dmd_safe = np.where(np.abs(dmd) > 1e-9, dmd, np.nan)
    columns = (
        dmd,
        np.log1p(np.maximum(dmd, 0.0)),
        suffix_progress,
        np.full(len(query_indices), np.log1p(len(hidden_indices))),
        np.full(len(query_indices), first_hidden / len(md)),
        query_x / 1_000_000.0,
        query_y / 1_000_000.0,
        query_z / 10_000.0,
        np.full(len(query_indices), x[anchor_index] / 1_000_000.0),
        np.full(len(query_indices), y[anchor_index] / 1_000_000.0),
        np.full(len(query_indices), anchor_z / 10_000.0),
        dx,
        dy,
        dz,
        np.hypot(dx, dy),
        _safe_ratio(dx, dmd_safe),
        _safe_ratio(dy, dmd_safe),
        _safe_ratio(dz, dmd_safe),
        x_derivative[query_indices],
        y_derivative[query_indices],
        z_derivative[query_indices],
        np.full(len(query_indices), anchor_tvt / 10_000.0),
        np.full(len(query_indices), anchor_u / 10_000.0),
        np.full(len(query_indices), short_tvt_slope),
        np.full(len(query_indices), long_tvt_slope),
        np.full(len(query_indices), short_u_slope),
        np.full(len(query_indices), long_u_slope),
        np.full(len(query_indices), short_tvt_slope - long_tvt_slope),
        slope_decay_delta,
        query_gr,
        (~np.isfinite(query_gr)).astype(float),
        query_gr - anchor_gr,
        gr_derivative[query_indices],
        gr_roll16_mean[query_indices],
        gr_roll16_std[query_indices],
        gr_roll64_mean[query_indices],
        gr_roll64_std[query_indices],
        np.full(len(query_indices), visible_gr_mean),
        np.full(len(query_indices), visible_gr_std),
        np.full(len(query_indices), visible_gr_missing),
        np.full(len(query_indices), calibration.gain),
        np.full(len(query_indices), calibration.offset / 100.0),
        np.full(len(query_indices), calibration.residual_scale),
        np.full(len(query_indices), calibration.correlation),
        type_gr_anchor,
        type_gr_u_hold,
        type_gr_slope,
        query_gr - type_gr_anchor,
        query_gr - type_gr_u_hold,
        query_gr - type_gr_slope,
        type_gradient_anchor,
        type_gradient_u_hold,
        type_gradient_slope,
        type_match_delta,
        type_match_error,
        slope_candidate_tvt - (typewell_tvt[0] if len(typewell_tvt) else np.nan),
        (typewell_tvt[-1] if len(typewell_tvt) else np.nan) - slope_candidate_tvt,
    )
    features_with_boundary = np.column_stack(columns)
    if features_with_boundary.shape[1] != len(RESIDUAL_FEATURE_NAMES):
        raise AssertionError("feature-name and feature-matrix width mismatch")
    features_with_boundary[~np.isfinite(features_with_boundary)] = np.nan
    features_with_boundary = features_with_boundary.astype(np.float32, copy=False)

    hidden_z = z[hidden_indices]
    hold_last = np.full(len(hidden_indices), anchor_tvt, dtype=float)
    u_boundary = anchor_u - hidden_z
    slope_decay = anchor_tvt + slope_decay_delta[1:]
    diagnostics = {
        "feature_config": asdict(cfg),
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "horizontal_columns_used": list(HORIZONTAL_INPUT_COLUMNS),
        "typewell_columns_used": list(TYPEWELL_INPUT_COLUMNS),
        "train_target_used": False,
        "well_id_encoded": False,
        "formation_columns_used": False,
        "same_id_lookup": False,
        "rows": int(len(md)),
        "visible_rows": int(first_hidden),
        "hidden_rows": int(len(hidden_indices)),
        "anchor_index": int(anchor_index),
        "anchor_tvt": anchor_tvt,
        "anchor_u": anchor_u,
        "visible_tvt_slope_short": short_tvt_slope,
        "visible_tvt_slope_long": long_tvt_slope,
        "heel_calibration": calibration.to_dict(),
    }
    return ResidualFeatureBlock(
        features=features_with_boundary[1:],
        boundary_features=features_with_boundary[:1],
        evaluation_mask=evaluation_mask,
        hidden_indices=hidden_indices,
        hold_last=hold_last,
        u_boundary=u_boundary,
        slope_decay=slope_decay,
        anchor_index=anchor_index,
        anchor_tvt=anchor_tvt,
        anchor_u=anchor_u,
        diagnostics=diagnostics,
    )
