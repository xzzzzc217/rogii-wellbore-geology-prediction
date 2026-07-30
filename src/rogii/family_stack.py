"""Small, target-agnostic helpers for the family-balanced OOF stack audit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_MODEL_FAMILIES: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4))


def collapse_model_families(
    model_predictions: np.ndarray,
    *,
    families: Sequence[Sequence[int]] = DEFAULT_MODEL_FAMILIES,
) -> np.ndarray:
    """Average correlated model columns inside each preregistered family."""

    predictions = np.asarray(model_predictions)
    if predictions.ndim != 2:
        raise ValueError("model_predictions must be a two-dimensional array")
    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("model_predictions must be non-empty")
    if not np.isfinite(predictions).all():
        raise ValueError("model_predictions must be finite")
    if not families:
        raise ValueError("at least one model family is required")

    seen: set[int] = set()
    columns: list[np.ndarray] = []
    for family in families:
        indices = tuple(int(index) for index in family)
        if not indices:
            raise ValueError("model families cannot be empty")
        if min(indices) < 0 or max(indices) >= predictions.shape[1]:
            raise ValueError("model family index is out of bounds")
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError("model family indices must be unique")
        seen.update(indices)
        columns.append(
            np.mean(
                predictions[:, indices],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
        )
    if seen != set(range(predictions.shape[1])):
        raise ValueError("model families must cover every prediction column exactly once")
    return np.column_stack(columns)


def compose_ridge_pf_track(
    anchor_tvt: np.ndarray,
    ridge_delta: np.ndarray,
    pf_tvt: np.ndarray,
    md_since: np.ndarray,
    *,
    ridge_weight: float = 0.91,
    pf_weight: float = 0.09,
    ramp_tau_md: float = 85.0,
) -> np.ndarray:
    """Apply the published Ridge/PF blend and measured-depth ramp."""

    anchor = np.asarray(anchor_tvt, dtype=np.float64)
    ridge = np.asarray(ridge_delta, dtype=np.float64)
    pf = np.asarray(pf_tvt, dtype=np.float64)
    distance = np.asarray(md_since, dtype=np.float64)
    if anchor.ndim != 1:
        raise ValueError("track inputs must be one-dimensional")
    if not (anchor.shape == ridge.shape == pf.shape == distance.shape):
        raise ValueError("track inputs must have the same shape")
    if anchor.size == 0 or not np.isfinite(
        np.column_stack((anchor, ridge, pf, distance))
    ).all():
        raise ValueError("track inputs must be finite and non-empty")
    if ridge_weight < 0.0 or pf_weight < 0.0:
        raise ValueError("track weights cannot be negative")
    if not np.isclose(ridge_weight + pf_weight, 1.0):
        raise ValueError("track weights must sum to one")
    if not np.isfinite(ramp_tau_md) or ramp_tau_md <= 0.0:
        raise ValueError("ramp_tau_md must be finite and positive")

    ramp = 1.0 - np.exp(-np.maximum(distance, 0.0) / ramp_tau_md)
    blended_delta = ridge_weight * ridge + pf_weight * (pf - anchor)
    return anchor + blended_delta * ramp
