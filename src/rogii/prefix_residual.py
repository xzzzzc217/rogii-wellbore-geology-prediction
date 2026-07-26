"""Inference-time visible-prefix residual calibration for trajectory predictions.

The layer is deliberately small and target-free.  A caller first creates a
pseudo-hidden suffix wholly inside the finite ``TVT_input`` prefix, predicts it
with the unchanged trajectory baseline, and supplies the resulting visible
residuals here.  Hidden ``TVT`` truth is neither needed nor accepted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PREFIX_CALIBRATION_ROWS = 256


@dataclass(frozen=True)
class PrefixResidualCandidate:
    """One predeclared bounded, decayed residual correction."""

    name: str
    residual_window: int
    correction_cap: float
    decay_tau_md: float
    include_slope: bool = False
    maximum_abs_slope: float = 0.0
    slope_horizon_md: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name cannot be empty")
        if self.residual_window < 8:
            raise ValueError("residual_window must be at least eight rows")
        if self.correction_cap <= 0.0:
            raise ValueError("correction_cap must be positive")
        if self.decay_tau_md <= 0.0:
            raise ValueError("decay_tau_md must be positive")
        if self.include_slope:
            if self.maximum_abs_slope <= 0.0 or self.slope_horizon_md <= 0.0:
                raise ValueError("slope candidates need positive slope controls")
        elif self.maximum_abs_slope != 0.0 or self.slope_horizon_md != 0.0:
            raise ValueError("level-only candidates cannot set slope controls")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


# Frozen before discovery scoring.  This is the complete candidate family.
PREFIX_RESIDUAL_CANDIDATES = (
    PrefixResidualCandidate(
        name="level64_cap4_tau240",
        residual_window=64,
        correction_cap=4.0,
        decay_tau_md=240.0,
    ),
    PrefixResidualCandidate(
        name="level128_cap4_tau480",
        residual_window=128,
        correction_cap=4.0,
        decay_tau_md=480.0,
    ),
    PrefixResidualCandidate(
        name="level128_cap8_tau480",
        residual_window=128,
        correction_cap=8.0,
        decay_tau_md=480.0,
    ),
    PrefixResidualCandidate(
        name="level_slope128_cap4_tau480",
        residual_window=128,
        correction_cap=4.0,
        decay_tau_md=480.0,
        include_slope=True,
        maximum_abs_slope=0.010,
        slope_horizon_md=120.0,
    ),
)


def _finite_tail(
    md: np.ndarray,
    residual: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(md, dtype=float)
    y = np.asarray(residual, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("prefix md and residual must be equal-length 1D arrays")
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 8:
        raise ValueError("at least eight finite prefix residual rows are required")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("finite prefix MD must be strictly increasing")
    return x[-window:], y[-window:]


def _robust_boundary_line(md: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    """Estimate residual level at the last MD and a robust local slope.

    The tail is divided into up to eight contiguous blocks.  Medians suppress
    row-level spikes, after which a short Huber reweighting estimates the line.
    """

    block_count = min(8, max(2, len(md) // 8))
    blocks = np.array_split(np.arange(len(md)), block_count)
    x = np.asarray([float(np.median(md[index])) for index in blocks])
    y = np.asarray([float(np.median(residual[index])) for index in blocks])
    boundary = float(md[-1])
    centered = x - boundary
    design = np.column_stack((np.ones_like(centered), centered))
    weights = np.ones_like(y)
    coefficients = np.array([float(np.median(y)), 0.0])
    for _ in range(5):
        root = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root[:, None],
            y * root,
            rcond=None,
        )
        errors = y - design @ coefficients
        center = float(np.median(errors))
        scale = 1.4826 * float(np.median(np.abs(errors - center)))
        scale = max(scale, 1e-6)
        standardized = np.abs(errors - center) / scale
        weights = np.ones_like(standardized)
        tail = standardized > 2.5
        weights[tail] = 2.5 / standardized[tail]
    return float(coefficients[0]), float(coefficients[1])


def estimate_visible_prefix_residual(
    prefix_md: np.ndarray,
    prefix_residual: np.ndarray,
    candidate: PrefixResidualCandidate,
) -> dict[str, float | int]:
    """Estimate a robust boundary level and optional slope from visible rows."""

    md, residual = _finite_tail(
        prefix_md,
        prefix_residual,
        window=candidate.residual_window,
    )
    median_level = float(np.median(residual))
    line_level, line_slope = _robust_boundary_line(md, residual)
    if candidate.include_slope:
        level = line_level
        slope = float(
            np.clip(
                line_slope,
                -candidate.maximum_abs_slope,
                candidate.maximum_abs_slope,
            )
        )
    else:
        level = median_level
        slope = 0.0
    return {
        "rows": int(len(md)),
        "boundary_md": float(md[-1]),
        "median_level": median_level,
        "line_level": line_level,
        "raw_line_slope": line_slope,
        "level": float(level),
        "slope": float(slope),
    }


def prefix_residual_correction(
    hidden_md: np.ndarray,
    boundary_md: float,
    prefix_md: np.ndarray,
    prefix_residual: np.ndarray,
    candidate: PrefixResidualCandidate,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return a bounded correction vector for the actual hidden suffix."""

    query_md = np.asarray(hidden_md, dtype=float)
    if query_md.ndim != 1 or query_md.size == 0:
        raise ValueError("hidden_md must be a non-empty 1D array")
    if not np.all(np.isfinite(query_md)) or np.any(np.diff(query_md) <= 0.0):
        raise ValueError("hidden_md must be finite and strictly increasing")
    if not math.isfinite(float(boundary_md)) or query_md[0] <= float(boundary_md):
        raise ValueError("boundary_md must precede the hidden suffix")

    estimate = estimate_visible_prefix_residual(prefix_md, prefix_residual, candidate)
    distance = np.maximum(query_md - float(boundary_md), 0.0)
    slope_distance = np.minimum(distance, candidate.slope_horizon_md)
    raw = float(estimate["level"]) + float(estimate["slope"]) * slope_distance
    bounded = np.clip(raw, -candidate.correction_cap, candidate.correction_cap)
    correction = bounded * np.exp(-distance / candidate.decay_tau_md)
    if not np.all(np.isfinite(correction)):
        raise ValueError("prefix residual correction is non-finite")
    diagnostics = dict(estimate)
    diagnostics.update(
        {
            "maximum_abs_correction": float(np.max(np.abs(correction))),
            "first_correction": float(correction[0]),
            "last_correction": float(correction[-1]),
        }
    )
    return correction, diagnostics


def apply_prefix_residual(
    baseline_hidden: np.ndarray,
    hidden_md: np.ndarray,
    boundary_md: float,
    prefix_md: np.ndarray,
    prefix_residual: np.ndarray,
    candidate: PrefixResidualCandidate,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply one candidate without accepting any hidden target values."""

    baseline = np.asarray(baseline_hidden, dtype=float)
    if baseline.ndim != 1 or baseline.shape != np.asarray(hidden_md).shape:
        raise ValueError("baseline_hidden and hidden_md must have the same 1D shape")
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_hidden must be finite")
    correction, diagnostics = prefix_residual_correction(
        hidden_md,
        boundary_md,
        prefix_md,
        prefix_residual,
        candidate,
    )
    prediction = baseline + correction
    if not np.all(np.isfinite(prediction)):
        raise ValueError("corrected prediction is non-finite")
    return prediction, diagnostics


def candidate_by_name(name: str) -> PrefixResidualCandidate:
    """Return one member of the frozen family."""

    matches = [candidate for candidate in PREFIX_RESIDUAL_CANDIDATES if candidate.name == name]
    if len(matches) != 1:
        raise KeyError(name)
    return matches[0]
