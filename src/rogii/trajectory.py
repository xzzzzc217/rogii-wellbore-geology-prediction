"""Deterministic, target-free trajectory baselines for ROGII.

The predictor deliberately accepts only columns that are present at hidden-test
inference time.  It follows the hidden suffix in the stratigraphic coordinate
``U = TVT + Z`` with two independent sequence trackers:

* a deterministic multi-seed particle filter; and
* a beam-limited Viterbi tracker.

Both trackers compare horizontal-well GR against linearly interpolated typewell
GR after a robust gain/offset calibration on the visible heel.  No training
well, formation/contact column, identifier lookup, leaderboard correction, or
precomputed prediction is consulted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

HORIZONTAL_INPUT_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_INPUT_COLUMNS = ("TVT", "GR")


@dataclass(frozen=True)
class TrajectoryConfig:
    """Controls for the deterministic target-free trajectory predictor."""

    heel_window: int = 900
    calibration_min_rows: int = 40
    calibration_huber_c: float = 2.5
    calibration_iterations: int = 5
    gain_min: float = 0.25
    gain_max: float = 4.0
    minimum_gr_scale: float = 10.0
    slope_window: int = 40
    maximum_abs_tvt_slope: float = 0.20

    pf_particles: int = 64
    pf_seeds: tuple[int, ...] = tuple(range(8))
    pf_initial_u_std: float = 4.5
    pf_initial_slope_std: float = 0.010
    pf_u_noise: float = 0.005
    pf_slope_noise: float = 0.002
    pf_slope_reversion: float = 0.998
    pf_observation_strength: float = 1.0
    pf_gr_scale_multiplier: float = 3.0
    pf_resample_threshold: float = 0.55
    pf_likelihood_clip: float = 12.0
    pf_seed_temperature: float = 5.0
    pf_resample_u_jitter: float = 0.10
    pf_resample_slope_jitter: float = 0.001

    beam_width: int = 10
    beam_index_moves: tuple[int, ...] = (-2, -1, 0, 1, 2)
    beam_move_penalty: float = 20.0
    beam_observation_strength: float = 1.0
    beam_likelihood_clip: float = 12.0
    beam_smoothing_window: int = 5

    pf_blend_weight: float = 0.95
    minimum_tracker_weight: float = 0.85
    maximum_tracker_weight: float = 0.95
    maximum_tvt_departure: float = 90.0

    def __post_init__(self) -> None:
        if self.heel_window <= 0 or self.slope_window <= 1:
            raise ValueError("heel_window and slope_window must be positive")
        if self.calibration_min_rows < 2:
            raise ValueError("calibration_min_rows must be at least two")
        if not 0 < self.gain_min <= self.gain_max:
            raise ValueError("gain bounds are invalid")
        if self.pf_particles < 2 or not self.pf_seeds:
            raise ValueError("the particle filter needs particles and at least one seed")
        if len(set(self.pf_seeds)) != len(self.pf_seeds):
            raise ValueError("pf_seeds must be unique")
        if not 0.0 <= self.pf_resample_threshold <= 1.0:
            raise ValueError("pf_resample_threshold must be in [0, 1]")
        if not 0.0 <= self.pf_slope_reversion < 1.0:
            raise ValueError("pf_slope_reversion must be in [0, 1)")
        if self.pf_gr_scale_multiplier <= 0:
            raise ValueError("pf_gr_scale_multiplier must be positive")
        if self.pf_seed_temperature <= 0:
            raise ValueError("pf_seed_temperature must be positive")
        if self.beam_width < 2 or not self.beam_index_moves:
            raise ValueError("beam_width and beam_index_moves are invalid")
        if 0 not in self.beam_index_moves:
            raise ValueError("beam_index_moves must include zero")
        if not 0.0 <= self.pf_blend_weight <= 1.0:
            raise ValueError("pf_blend_weight must be in [0, 1]")
        if not 0.0 <= self.minimum_tracker_weight <= 1.0:
            raise ValueError("minimum_tracker_weight must be in [0, 1]")
        if not self.minimum_tracker_weight <= self.maximum_tracker_weight <= 1.0:
            raise ValueError("maximum_tracker_weight must be in [minimum, 1]")
        if self.maximum_tvt_departure <= 0:
            raise ValueError("maximum_tvt_departure must be positive")


@dataclass(frozen=True)
class HeelCalibration:
    """Robust mapping from typewell GR to horizontal-well GR."""

    gain: float
    offset: float
    residual_scale: float
    correlation: float
    rows: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass
class TrajectoryResult:
    """Full-row predictions, component paths, and auditable diagnostics."""

    prediction: np.ndarray
    evaluation_mask: np.ndarray
    components: dict[str, np.ndarray]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedInputs:
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_input: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    mask: np.ndarray
    first_hidden: int
    anchor_index: int


@dataclass(frozen=True)
class _PFRun:
    u: np.ndarray
    posterior_std: np.ndarray
    total_log_evidence: float
    observed_rows: int
    resamples: int


@dataclass(frozen=True)
class _BeamRun:
    u: np.ndarray
    average_cost: float
    terminal_margin: float
    states_evaluated: int


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"required column is missing: {name}")
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, sorted typewell arrays with duplicate TVT rows averaged."""

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

    unique_tvt, first, counts = np.unique(tvt, return_index=True, return_counts=True)
    if np.any(counts > 1):
        gr = np.add.reduceat(gr, first) / counts
        tvt = unique_tvt
    return tvt.astype(float, copy=False), gr.astype(float, copy=False)


def interpolate_typewell_gr(
    tvt: np.ndarray | list[float] | float,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate typewell GR without extrapolating past its support."""

    query = np.asarray(tvt, dtype=float)
    if len(typewell_tvt) < 2:
        return np.full(query.shape, np.nan, dtype=float)
    return np.interp(query, typewell_tvt, typewell_gr, left=np.nan, right=np.nan)


def _robust_scale(values: np.ndarray, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(floor)
    center = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - center)))
    return float(max(floor, scale if math.isfinite(scale) else floor))


def calibrate_heel_gr(
    horizontal_gr: np.ndarray,
    known_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    config: TrajectoryConfig | None = None,
) -> HeelCalibration:
    """Fit ``horizontal_GR ~= gain * typewell_GR(TVT) + offset`` robustly."""

    cfg = config or TrajectoryConfig()
    horizontal_gr = np.asarray(horizontal_gr, dtype=float)
    known_tvt = np.asarray(known_tvt, dtype=float)
    if horizontal_gr.shape != known_tvt.shape:
        raise ValueError("horizontal_gr and known_tvt must have the same shape")

    reference = interpolate_typewell_gr(known_tvt, typewell_tvt, typewell_gr)
    valid = np.isfinite(horizontal_gr) & np.isfinite(reference)
    indices = np.flatnonzero(valid)
    if indices.size > cfg.heel_window:
        indices = indices[-cfg.heel_window :]
    if indices.size < cfg.calibration_min_rows:
        return HeelCalibration(
            1.0,
            0.0,
            cfg.minimum_gr_scale,
            0.0,
            int(indices.size),
            "too_few_rows",
        )

    x = reference[indices]
    y = horizontal_gr[indices]
    if float(np.std(x)) < 1e-8:
        offset = float(np.median(y) - np.median(x))
        scale = _robust_scale(y - (x + offset), cfg.minimum_gr_scale)
        return HeelCalibration(1.0, offset, scale, 0.0, int(indices.size), "flat_reference")

    design = np.column_stack((x, np.ones_like(x)))
    weights = np.ones_like(x)
    gain = 1.0
    offset = float(np.median(y) - np.median(x))
    for _ in range(cfg.calibration_iterations):
        root_weights = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root_weights[:, None],
            y * root_weights,
            rcond=None,
        )
        gain = float(np.clip(coefficients[0], cfg.gain_min, cfg.gain_max))
        offset = float(np.average(y - gain * x, weights=weights))
        residual = y - (gain * x + offset)
        scale = _robust_scale(residual, cfg.minimum_gr_scale)
        standardized = np.abs(residual) / scale
        weights = np.ones_like(standardized)
        tail = standardized > cfg.calibration_huber_c
        weights[tail] = cfg.calibration_huber_c / standardized[tail]

    residual = y - (gain * x + offset)
    scale = _robust_scale(residual, cfg.minimum_gr_scale)
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    if not math.isfinite(correlation):
        correlation = 0.0
    return HeelCalibration(gain, offset, scale, correlation, int(indices.size), "ok")


def hold_last_prediction(tvt_input: np.ndarray, fallback: float | None = None) -> np.ndarray:
    """Fill missing TVT rows with the last finite visible value."""

    values = np.asarray(tvt_input, dtype=float)
    output = values.copy()
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size:
        anchor = float(values[finite[-1]])
    elif fallback is not None and math.isfinite(float(fallback)):
        anchor = float(fallback)
    else:
        raise ValueError("hold-last prediction needs a visible TVT or finite fallback")
    output[~np.isfinite(output)] = anchor
    return output


def _prepare_inputs(horizontal: pd.DataFrame, typewell: pd.DataFrame) -> _PreparedInputs:
    # Extract only this explicit allow-list. Extra train columns are never read.
    md = _numeric_column(horizontal, "MD")
    x = _numeric_column(horizontal, "X")
    y = _numeric_column(horizontal, "Y")
    z = _numeric_column(horizontal, "Z")
    gr = _numeric_column(horizontal, "GR")
    tvt_input = _numeric_column(horizontal, "TVT_input")
    if not (len(md) == len(x) == len(y) == len(z) == len(gr) == len(tvt_input)):
        raise ValueError("horizontal columns have inconsistent lengths")
    if len(md) == 0:
        raise ValueError("horizontal frame is empty")
    if not np.all(np.isfinite(md)) or not np.all(np.isfinite(z)):
        raise ValueError("MD and Z must be finite")
    if np.any(np.diff(md) <= 0):
        raise ValueError("MD must be strictly increasing")

    mask = ~np.isfinite(tvt_input)
    hidden = np.flatnonzero(mask)
    if hidden.size == 0:
        first_hidden = len(md)
        anchor_index = len(md) - 1
    else:
        first_hidden = int(hidden[0])
        if not np.all(mask[first_hidden:]):
            raise ValueError("TVT_input missing rows must form one contiguous suffix")
        if first_hidden == 0:
            raise ValueError("a finite TVT_input prefix is required")
        anchor_index = first_hidden - 1
        if not np.isfinite(tvt_input[anchor_index]):
            raise ValueError("the row before the hidden suffix must have finite TVT_input")

    typewell_tvt, typewell_gr = _prepare_typewell(typewell)
    return _PreparedInputs(
        md=md,
        x=x,
        y=y,
        z=z,
        gr=gr,
        tvt_input=tvt_input,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        mask=mask,
        first_hidden=first_hidden,
        anchor_index=anchor_index,
    )


def _fit_local_slope(md: np.ndarray, values: np.ndarray, end: int, window: int) -> float:
    start = max(0, end - window + 1)
    x = md[start : end + 1]
    y = values[start : end + 1]
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return 0.0
    x = x[valid]
    y = y[valid]
    x_center = x - float(np.mean(x))
    denominator = float(np.dot(x_center, x_center))
    if denominator <= 0:
        return 0.0
    slope = float(np.dot(x_center, y - float(np.mean(y))) / denominator)
    residual = y - (float(np.mean(y)) + slope * x_center)
    scale = _robust_scale(residual, 1e-3)
    keep = np.abs(residual - float(np.median(residual))) <= 3.5 * scale
    if int(keep.sum()) >= 3 and not np.all(keep):
        kept_x = x[keep]
        kept_y = y[keep]
        kept_center = kept_x - float(np.mean(kept_x))
        denominator = float(np.dot(kept_center, kept_center))
        if denominator > 0:
            slope = float(np.dot(kept_center, kept_y - float(np.mean(kept_y))) / denominator)
    return slope


def _median_local_increment_slope(
    md: np.ndarray,
    values: np.ndarray,
    end: int,
    window: int,
) -> float:
    start = max(0, end - window + 1)
    delta_md = np.diff(md[start : end + 1])
    delta_value = np.diff(values[start : end + 1])
    valid = np.isfinite(delta_md) & np.isfinite(delta_value) & (delta_md > 0)
    if int(valid.sum()) < 3:
        return _fit_local_slope(md, values, end, window)
    return float(np.median(delta_value[valid] / delta_md[valid]))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    positions = (float(rng.random()) + np.arange(count, dtype=float)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _interpolate_missing(values: np.ndarray, fallback: float) -> np.ndarray:
    """Interpolate internal/end gaps in a one-dimensional observed signal."""

    output = np.asarray(values, dtype=float).copy()
    finite = np.flatnonzero(np.isfinite(output))
    if finite.size == 0:
        output.fill(float(fallback))
    elif finite.size == 1:
        output.fill(float(output[finite[0]]))
    elif finite.size < len(output):
        missing = np.flatnonzero(~np.isfinite(output))
        output[missing] = np.interp(missing, finite, output[finite])
    return output


def _observation_log_likelihood(
    u: np.ndarray,
    z: float,
    observed_gr: float,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    calibration: HeelCalibration,
    *,
    strength: float,
    clip: float,
    scale_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(observed_gr):
        return np.zeros_like(u), np.zeros_like(u, dtype=bool)
    candidate_tvt = u - z
    reference_gr = interpolate_typewell_gr(candidate_tvt, typewell_tvt, typewell_gr)
    valid = np.isfinite(reference_gr)
    standardized = np.full_like(u, clip)
    standardized[valid] = (
        observed_gr - (calibration.gain * reference_gr[valid] + calibration.offset)
    ) / (calibration.residual_scale * scale_multiplier)
    standardized = np.clip(standardized, -clip, clip)
    return -0.5 * strength * standardized**2, valid


def _particle_filter_once(
    inputs: _PreparedInputs,
    calibration: HeelCalibration,
    *,
    anchor_u: float,
    initial_u_slope: float,
    tvt_slope_prior: float,
    seed: int,
    config: TrajectoryConfig,
) -> _PFRun:
    suffix = np.arange(inputs.first_hidden, len(inputs.md))
    count = len(suffix)
    rng = np.random.default_rng(seed)
    particles = anchor_u + rng.normal(0.0, config.pf_initial_u_std, config.pf_particles)
    slopes = initial_u_slope + rng.normal(0.0, config.pf_initial_slope_std, config.pf_particles)
    weights = np.full(config.pf_particles, 1.0 / config.pf_particles)
    output = np.empty(count, dtype=float)
    spread = np.empty(count, dtype=float)
    evidence = 0.0
    observed_rows = 0
    resamples = 0
    previous_index = inputs.anchor_index
    tracking_gr = _interpolate_missing(
        inputs.gr,
        float(np.mean(inputs.typewell_gr) * calibration.gain + calibration.offset),
    )

    for output_index, row_index in enumerate(suffix):
        delta_md = float(inputs.md[row_index] - inputs.md[previous_index])
        slopes = config.pf_slope_reversion * slopes + rng.normal(
            0.0,
            config.pf_slope_noise * math.sqrt(delta_md),
            config.pf_particles,
        )
        slopes = np.clip(
            slopes,
            -config.maximum_abs_tvt_slope,
            config.maximum_abs_tvt_slope,
        )
        particles = (
            particles
            + slopes * delta_md
            + rng.normal(
                0.0,
                config.pf_u_noise * math.sqrt(delta_md),
                config.pf_particles,
            )
        )

        log_likelihood, valid = _observation_log_likelihood(
            particles,
            float(inputs.z[row_index]),
            float(tracking_gr[row_index]),
            inputs.typewell_tvt,
            inputs.typewell_gr,
            calibration,
            strength=config.pf_observation_strength,
            clip=config.pf_likelihood_clip,
            scale_multiplier=config.pf_gr_scale_multiplier,
        )
        if np.any(valid):
            log_prior = np.log(np.maximum(weights, np.finfo(float).tiny))
            combined = log_prior + log_likelihood
            maximum = float(np.max(combined))
            normalizer = maximum + math.log(float(np.exp(combined - maximum).sum()))
            weights = np.exp(combined - normalizer)
            evidence += normalizer
            observed_rows += 1

        mean_u = float(np.dot(weights, particles))
        output[output_index] = mean_u
        spread[output_index] = math.sqrt(
            max(0.0, float(np.dot(weights, (particles - mean_u) ** 2)))
        )

        effective_size = 1.0 / float(np.dot(weights, weights))
        if effective_size < config.pf_resample_threshold * config.pf_particles:
            selected = _systematic_resample(weights, rng)
            particles = particles[selected] + rng.normal(
                0.0,
                config.pf_resample_u_jitter,
                config.pf_particles,
            )
            slopes = slopes[selected] + rng.normal(
                0.0,
                config.pf_resample_slope_jitter,
                config.pf_particles,
            )
            weights.fill(1.0 / config.pf_particles)
            resamples += 1
        previous_index = row_index

    return _PFRun(output, spread, float(evidence), observed_rows, resamples)


def particle_filter_trajectory(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    config: TrajectoryConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the deterministic multi-seed ``U=TVT+Z`` particle filter.

    Returns a full-row TVT path and PF-specific diagnostics. Visible rows are
    copied from ``TVT_input``.
    """

    cfg = config or TrajectoryConfig()
    inputs = _prepare_inputs(horizontal, typewell)
    if inputs.first_hidden == len(inputs.md):
        return inputs.tvt_input.copy(), {
            "seeds": list(cfg.pf_seeds),
            "seed_weights": [1.0 / len(cfg.pf_seeds)] * len(cfg.pf_seeds),
            "status": "no_hidden_rows",
        }
    if len(inputs.typewell_tvt) < 2:
        fallback = hold_last_prediction(inputs.tvt_input)
        return fallback, {
            "seeds": list(cfg.pf_seeds),
            "seed_weights": [1.0 / len(cfg.pf_seeds)] * len(cfg.pf_seeds),
            "status": "fallback_typewell_unavailable",
        }

    calibration = calibrate_heel_gr(
        inputs.gr[: inputs.first_hidden],
        inputs.tvt_input[: inputs.first_hidden],
        inputs.typewell_tvt,
        inputs.typewell_gr,
        config=cfg,
    )
    anchor_u = float(inputs.tvt_input[inputs.anchor_index] + inputs.z[inputs.anchor_index])
    prefix_u = inputs.tvt_input + inputs.z
    initial_u_slope = _median_local_increment_slope(
        inputs.md, prefix_u, inputs.anchor_index, cfg.slope_window
    )
    tvt_slope_prior = float(
        np.clip(
            _fit_local_slope(
                inputs.md,
                inputs.tvt_input,
                inputs.anchor_index,
                cfg.slope_window,
            ),
            -cfg.maximum_abs_tvt_slope,
            cfg.maximum_abs_tvt_slope,
        )
    )
    runs = [
        _particle_filter_once(
            inputs,
            calibration,
            anchor_u=anchor_u,
            initial_u_slope=initial_u_slope,
            tvt_slope_prior=tvt_slope_prior,
            seed=seed,
            config=cfg,
        )
        for seed in cfg.pf_seeds
    ]
    evidence = np.array([run.total_log_evidence for run in runs], dtype=float)
    centered = (evidence - float(np.max(evidence))) / cfg.pf_seed_temperature
    seed_weights = np.exp(np.clip(centered, -60.0, 0.0))
    seed_weights /= float(seed_weights.sum())
    u = np.average(np.stack([run.u for run in runs]), axis=0, weights=seed_weights)
    within_seed_variance = np.average(
        np.stack([run.posterior_std**2 for run in runs]),
        axis=0,
        weights=seed_weights,
    )
    between_seed_variance = np.average(
        (np.stack([run.u for run in runs]) - u[None, :]) ** 2,
        axis=0,
        weights=seed_weights,
    )
    total_std = np.sqrt(np.maximum(0.0, within_seed_variance + between_seed_variance))

    output = inputs.tvt_input.copy()
    output[inputs.mask] = u - inputs.z[inputs.mask]
    diagnostics = {
        "status": "ok",
        "seeds": [int(seed) for seed in cfg.pf_seeds],
        "seed_weights": [float(value) for value in seed_weights],
        "total_log_evidence": [float(value) for value in evidence],
        "average_log_evidence": [
            float(run.total_log_evidence / max(run.observed_rows, 1)) for run in runs
        ],
        "resamples": [int(run.resamples) for run in runs],
        "posterior_std_median": float(np.median(total_std)),
        "posterior_std_p90": float(np.quantile(total_std, 0.90)),
        "initial_u_slope": initial_u_slope,
        "tvt_slope_prior": tvt_slope_prior,
        "calibration": calibration.to_dict(),
    }
    return output, diagnostics


def _beam_tracker_from_inputs(
    inputs: _PreparedInputs,
    calibration: HeelCalibration,
    *,
    anchor_u: float,
    initial_u_slope: float,
    tvt_slope_prior: float,
    config: TrajectoryConfig,
) -> _BeamRun:
    suffix = np.arange(inputs.first_hidden, len(inputs.md))
    anchor_tvt = anchor_u - float(inputs.z[inputs.anchor_index])
    start_index = int(np.argmin(np.abs(inputs.typewell_tvt - anchor_tvt)))
    previous_typewell_index = np.array([start_index], dtype=int)
    previous_cost = np.array([0.0], dtype=float)
    histories_typewell_index: list[np.ndarray] = []
    histories_parent: list[np.ndarray] = []
    states_evaluated = 0
    fallback_gr = float(np.mean(inputs.typewell_gr) * calibration.gain + calibration.offset)
    tracking_gr = _interpolate_missing(inputs.gr, fallback_gr)
    window = int(config.beam_smoothing_window)
    if window > 1 and len(tracking_gr) >= window:
        if window % 2 == 0:
            window += 1
        half = window // 2
        kernel = np.arange(1, half + 2, dtype=float)
        kernel = np.concatenate((kernel, kernel[-2::-1]))
        kernel /= float(kernel.sum())
        padded = np.pad(tracking_gr, (half, half), mode="edge")
        tracking_gr = np.convolve(padded, kernel, mode="valid")

    moves = np.asarray(config.beam_index_moves, dtype=int)
    for row_index in suffix:
        parent = np.repeat(np.arange(len(previous_typewell_index)), len(moves))
        repeated_index = np.repeat(previous_typewell_index, len(moves))
        candidate_move = np.tile(moves, len(previous_typewell_index))
        candidate_typewell_index = repeated_index + candidate_move
        valid = (candidate_typewell_index >= 0) & (
            candidate_typewell_index < len(inputs.typewell_tvt)
        )
        safe_index = np.clip(
            candidate_typewell_index,
            0,
            len(inputs.typewell_tvt) - 1,
        )
        expected_gr = calibration.gain * inputs.typewell_gr[safe_index] + calibration.offset
        standardized = (float(tracking_gr[row_index]) - expected_gr) / calibration.residual_scale
        observation_cost = config.beam_observation_strength * np.minimum(
            standardized**2,
            config.beam_likelihood_clip**2,
        )
        movement_cost = config.beam_move_penalty * np.abs(candidate_move)
        candidate_cost = np.repeat(previous_cost, len(moves)) + observation_cost + movement_cost
        candidate_cost[~valid] = np.inf
        states_evaluated += len(candidate_cost)

        order = np.argsort(candidate_cost, kind="stable")
        selected: list[int] = []
        occupied: set[int] = set()
        for candidate_index in order:
            if not math.isfinite(float(candidate_cost[candidate_index])):
                continue
            key = int(candidate_typewell_index[candidate_index])
            if key in occupied:
                continue
            occupied.add(key)
            selected.append(int(candidate_index))
            if len(selected) == config.beam_width:
                break
        if not selected:
            selected = [int(order[0])]

        chosen = np.asarray(selected, dtype=int)
        previous_typewell_index = candidate_typewell_index[chosen]
        previous_cost = candidate_cost[chosen]
        histories_typewell_index.append(previous_typewell_index.copy())
        histories_parent.append(parent[chosen].copy())

    terminal_order = np.argsort(previous_cost, kind="stable")
    path_index = int(terminal_order[0])
    path_typewell_index = np.empty(len(suffix), dtype=int)
    for step in range(len(suffix) - 1, -1, -1):
        path_typewell_index[step] = histories_typewell_index[step][path_index]
        path_index = int(histories_parent[step][path_index])
    path = inputs.typewell_tvt[path_typewell_index] + inputs.z[inputs.first_hidden :]
    margin = (
        float(previous_cost[terminal_order[1]] - previous_cost[terminal_order[0]])
        if len(terminal_order) > 1
        else 0.0
    )
    return _BeamRun(
        path,
        float(previous_cost[terminal_order[0]] / max(len(suffix), 1)),
        margin,
        states_evaluated,
    )


def beam_viterbi_trajectory(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    config: TrajectoryConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the constrained beam/Viterbi GR tracker in ``U=TVT+Z`` space."""

    cfg = config or TrajectoryConfig()
    inputs = _prepare_inputs(horizontal, typewell)
    if inputs.first_hidden == len(inputs.md):
        return inputs.tvt_input.copy(), {"status": "no_hidden_rows"}
    if len(inputs.typewell_tvt) < 2:
        return hold_last_prediction(inputs.tvt_input), {"status": "fallback_typewell_unavailable"}

    calibration = calibrate_heel_gr(
        inputs.gr[: inputs.first_hidden],
        inputs.tvt_input[: inputs.first_hidden],
        inputs.typewell_tvt,
        inputs.typewell_gr,
        config=cfg,
    )
    anchor_u = float(inputs.tvt_input[inputs.anchor_index] + inputs.z[inputs.anchor_index])
    initial_u_slope = _median_local_increment_slope(
        inputs.md,
        inputs.tvt_input + inputs.z,
        inputs.anchor_index,
        cfg.slope_window,
    )
    tvt_slope_prior = float(
        np.clip(
            _fit_local_slope(
                inputs.md,
                inputs.tvt_input,
                inputs.anchor_index,
                cfg.slope_window,
            ),
            -cfg.maximum_abs_tvt_slope,
            cfg.maximum_abs_tvt_slope,
        )
    )
    run = _beam_tracker_from_inputs(
        inputs,
        calibration,
        anchor_u=anchor_u,
        initial_u_slope=initial_u_slope,
        tvt_slope_prior=tvt_slope_prior,
        config=cfg,
    )
    output = inputs.tvt_input.copy()
    output[inputs.mask] = run.u - inputs.z[inputs.mask]
    return output, {
        "status": "ok",
        "average_cost": run.average_cost,
        "terminal_margin": run.terminal_margin,
        "states_evaluated": run.states_evaluated,
        "initial_u_slope": initial_u_slope,
        "tvt_slope_prior": tvt_slope_prior,
        "calibration": calibration.to_dict(),
    }


def _path_gr_rmse(
    tvt: np.ndarray,
    gr: np.ndarray,
    mask: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    calibration: HeelCalibration,
) -> tuple[float | None, int]:
    reference = interpolate_typewell_gr(tvt[mask], typewell_tvt, typewell_gr)
    observed = gr[mask]
    valid = np.isfinite(reference) & np.isfinite(observed)
    if not np.any(valid):
        return None, 0
    predicted = calibration.gain * reference[valid] + calibration.offset
    rmse = float(np.sqrt(np.mean((observed[valid] - predicted) ** 2)))
    return rmse, int(valid.sum())


def predict_trajectory(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    config: TrajectoryConfig | None = None,
) -> TrajectoryResult:
    """Predict the hidden TVT suffix using only hidden-test-available inputs."""

    cfg = config or TrajectoryConfig()
    inputs = _prepare_inputs(horizontal, typewell)
    fallback_value = float(np.median(inputs.typewell_tvt)) if len(inputs.typewell_tvt) else None
    hold = hold_last_prediction(inputs.tvt_input, fallback=fallback_value)
    components = {"hold_last": hold.copy()}

    base_diagnostics: dict[str, Any] = {
        "target_free": True,
        "horizontal_columns_used": list(HORIZONTAL_INPUT_COLUMNS),
        "typewell_columns_used": list(TYPEWELL_INPUT_COLUMNS),
        "same_id_lookup": False,
        "train_target_used": False,
        "formation_columns_used": False,
        "leaderboard_bias_used": False,
        "precomputed_predictions_used": False,
        "row_count": int(len(inputs.md)),
        "visible_rows": int((~inputs.mask).sum()),
        "hidden_rows": int(inputs.mask.sum()),
        "first_hidden_position": (
            int(inputs.first_hidden) if inputs.first_hidden < len(inputs.md) else None
        ),
    }
    if not np.any(inputs.mask):
        base_diagnostics.update(
            {
                "status": "no_hidden_rows",
                "fallback_reason": None,
                "tracker_weight": 0.0,
            }
        )
        return TrajectoryResult(hold, inputs.mask.copy(), components, base_diagnostics)

    if len(inputs.typewell_tvt) < 2:
        base_diagnostics.update(
            {
                "status": "fallback",
                "fallback_reason": "typewell_has_fewer_than_two_finite_rows",
                "tracker_weight": 0.0,
            }
        )
        return TrajectoryResult(hold, inputs.mask.copy(), components, base_diagnostics)

    calibration = calibrate_heel_gr(
        inputs.gr[: inputs.first_hidden],
        inputs.tvt_input[: inputs.first_hidden],
        inputs.typewell_tvt,
        inputs.typewell_gr,
        config=cfg,
    )
    if calibration.status == "too_few_rows":
        base_diagnostics.update(
            {
                "status": "fallback",
                "fallback_reason": "insufficient_heel_rows_for_gr_calibration",
                "tracker_weight": 0.0,
                "calibration": calibration.to_dict(),
            }
        )
        return TrajectoryResult(hold, inputs.mask.copy(), components, base_diagnostics)

    pf, pf_diagnostics = particle_filter_trajectory(horizontal, typewell, config=cfg)
    beam, beam_diagnostics = beam_viterbi_trajectory(horizontal, typewell, config=cfg)
    components["particle_filter"] = pf.copy()
    components["beam_viterbi"] = beam.copy()

    tracker = hold.copy()
    tracker[inputs.mask] = (
        cfg.pf_blend_weight * pf[inputs.mask] + (1.0 - cfg.pf_blend_weight) * beam[inputs.mask]
    )
    calibration_row_trust = min(1.0, calibration.rows / max(cfg.heel_window * 0.5, 1.0))
    correlation_trust = float(np.clip((calibration.correlation - 0.10) / 0.65, 0.0, 1.0))
    trust = calibration_row_trust * correlation_trust
    tracker_weight = (
        cfg.minimum_tracker_weight
        + (cfg.maximum_tracker_weight - cfg.minimum_tracker_weight) * trust
    )
    final = hold.copy()
    final[inputs.mask] = (
        tracker_weight * tracker[inputs.mask] + (1.0 - tracker_weight) * hold[inputs.mask]
    )
    lower = float(hold[inputs.mask][0] - cfg.maximum_tvt_departure)
    upper = float(hold[inputs.mask][0] + cfg.maximum_tvt_departure)
    support_lower = float(inputs.typewell_tvt[0])
    support_upper = float(inputs.typewell_tvt[-1])
    lower = max(lower, support_lower)
    upper = min(upper, support_upper)
    if lower <= upper:
        final[inputs.mask] = np.clip(final[inputs.mask], lower, upper)
    if not np.all(np.isfinite(final)):
        final = hold.copy()
        status = "fallback"
        fallback_reason = "non_finite_tracker_output"
        tracker_weight = 0.0
    else:
        status = "ok"
        fallback_reason = None
    components["tracker_blend"] = tracker
    components["final"] = final.copy()

    gr_rmse: dict[str, float | None] = {}
    gr_rows = 0
    for name, path in components.items():
        rmse, rows = _path_gr_rmse(
            path,
            inputs.gr,
            inputs.mask,
            inputs.typewell_tvt,
            inputs.typewell_gr,
            calibration,
        )
        gr_rmse[name] = rmse
        gr_rows = max(gr_rows, rows)

    base_diagnostics.update(
        {
            "status": status,
            "fallback_reason": fallback_reason,
            "anchor_tvt": float(inputs.tvt_input[inputs.anchor_index]),
            "anchor_u": float(
                inputs.tvt_input[inputs.anchor_index] + inputs.z[inputs.anchor_index]
            ),
            "typewell_tvt_min": float(inputs.typewell_tvt[0]),
            "typewell_tvt_max": float(inputs.typewell_tvt[-1]),
            "calibration": calibration.to_dict(),
            "calibration_trust": trust,
            "tracker_weight": tracker_weight,
            "pf_blend_weight": cfg.pf_blend_weight,
            "particle_filter": pf_diagnostics,
            "beam_viterbi": beam_diagnostics,
            "hidden_gr_comparable_rows": gr_rows,
            "hidden_gr_rmse_by_component": gr_rmse,
            "final_tvt_min": float(np.min(final[inputs.mask])),
            "final_tvt_max": float(np.max(final[inputs.mask])),
        }
    )
    return TrajectoryResult(final, inputs.mask.copy(), components, base_diagnostics)
