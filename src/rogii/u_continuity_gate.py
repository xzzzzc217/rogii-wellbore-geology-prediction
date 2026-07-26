"""Pre-frozen deployment gate for the full U-continuity evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_724
SPATIAL_CLUSTERS = 5
SPATIAL_RANDOM_STATE = 20_260_724
SPATIAL_N_INIT = 50

POOLED_DELTA_LIMIT = -0.020
SPATIAL_REQUIRED_NONPOSITIVE = 4
SPATIAL_MAX_WORSE_LIMIT = 0.020


def pooled_rmse(rmse: np.ndarray, rows: np.ndarray) -> float:
    """Reconstruct pooled RMSE from per-well RMSE and row counts."""

    values = np.asarray(rmse, dtype=float)
    weights = np.asarray(rows, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
        raise ValueError("rmse and rows must be equal-length 1D arrays")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("rmse must be finite and non-empty")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("rows must be finite and positive")
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def paired_well_bootstrap(
    baseline_rmse: np.ndarray,
    corrected_rmse: np.ndarray,
    rows: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap wells as paired units and return a percentile CI for pooled delta."""

    baseline = np.asarray(baseline_rmse, dtype=float)
    corrected = np.asarray(corrected_rmse, dtype=float)
    weights = np.asarray(rows, dtype=float)
    if baseline.shape != corrected.shape or baseline.shape != weights.shape:
        raise ValueError("paired arrays must have the same shape")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    batch_size = 512
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        sampled = rng.integers(0, baseline.size, size=(count, baseline.size))
        sampled_rows = weights[sampled]
        denominator = np.sum(sampled_rows, axis=1)
        baseline_pooled = np.sqrt(
            np.sum(sampled_rows * baseline[sampled] ** 2, axis=1) / denominator
        )
        corrected_pooled = np.sqrt(
            np.sum(sampled_rows * corrected[sampled] ** 2, axis=1) / denominator
        )
        deltas[start : start + count] = corrected_pooled - baseline_pooled

    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def deployment_decision(
    *,
    pooled_delta: float,
    macro_delta: float,
    bootstrap_ci_upper: float,
    cluster_deltas: list[float],
) -> dict[str, Any]:
    """Apply the gate frozen before the full-run result was available."""

    if len(cluster_deltas) != SPATIAL_CLUSTERS:
        raise ValueError(f"deployment gate requires exactly {SPATIAL_CLUSTERS} clusters")
    nonpositive = int(np.count_nonzero(np.asarray(cluster_deltas) <= 0.0))
    maximum = float(np.max(cluster_deltas))
    checks = {
        "pooled_delta_at_most_minus_0_020": bool(pooled_delta <= POOLED_DELTA_LIMIT),
        "well_macro_delta_negative": bool(macro_delta < 0.0),
        "bootstrap_ci95_upper_negative": bool(bootstrap_ci_upper < 0.0),
        "at_least_4_of_5_spatial_clusters_nonpositive": bool(
            nonpositive >= SPATIAL_REQUIRED_NONPOSITIVE
        ),
        "no_spatial_cluster_worse_than_plus_0_020": bool(maximum <= SPATIAL_MAX_WORSE_LIMIT),
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "spatial_nonpositive_clusters": nonpositive,
        "spatial_maximum_delta": maximum,
        "threshold_tuning_after_result": False,
    }
