"""Honest discovery/confirmation evaluation of visible-prefix residual calibration.

The protocol is frozen in this file:

* six target-free spatial clusters from X/Y centroids;
* clusters ranked 1 and 4 are untouched confirmation blocks;
* four predeclared corrections are evaluated on discovery;
* at most one passing candidate is frozen by discovery pooled RMSE;
* confirmation is then scored exactly once with that frozen candidate.

Prediction caches contain inference-time arrays only.  Hidden ``TVT`` is loaded
separately and only after a split enters its scoring phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rogii.prefix_residual import (  # noqa: E402
    PREFIX_CALIBRATION_ROWS,
    PREFIX_RESIDUAL_CANDIDATES,
    apply_prefix_residual,
)
from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
    TrajectoryConfig,
    predict_trajectory,
)

CACHE_SCHEMA = 1
SPATIAL_CLUSTERS = 6
SPATIAL_RANDOM_STATE = 20_260_725
SPATIAL_N_INIT = 50
CONFIRMATION_CLUSTER_RANKS = (1, 4)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_725

# Frozen discovery gate.  The effect must be material, stable, and tail-safe.
DISCOVERY_POOLED_DELTA_LIMIT = -0.020
DISCOVERY_REQUIRED_NONPOSITIVE_CLUSTERS = 3
DISCOVERY_MAX_CLUSTER_DELTA = 0.030
DISCOVERY_P95_DELTA_LIMIT = 0.250
DISCOVERY_WORST_DELTA_LIMIT = 1.000

# Frozen confirmation/deployment gate.  Both untouched spatial blocks must agree.
CONFIRMATION_POOLED_DELTA_LIMIT = -0.020
CONFIRMATION_REQUIRED_NONPOSITIVE_CLUSTERS = 2
CONFIRMATION_MAX_CLUSTER_DELTA = 0.020
CONFIRMATION_P95_DELTA_LIMIT = 0.250
CONFIRMATION_WORST_DELTA_LIMIT = 1.000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "prefix_residual_cache",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "prefix_residual_evaluation.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-cache", action="store_true")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _config_fingerprint(config: TrajectoryConfig) -> str:
    payload = json.dumps(_json_safe(asdict(config)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _well_id(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


def _cache_path(cache_dir: Path, well_id: str) -> Path:
    return cache_dir / f"prefix_residual_{well_id}.npz"


def _cache_is_valid(
    path: Path,
    *,
    well_id: str,
    config_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    required = {
        "cache_schema",
        "well_id",
        "config_fingerprint",
        "hidden_indices",
        "hidden_md",
        "baseline_hidden",
        "boundary_md",
        "prefix_md",
        "prefix_residual",
        "centroid_x",
        "centroid_y",
    }
    try:
        with np.load(path, allow_pickle=False) as cache:
            if not required.issubset(cache.files):
                return False
            if int(cache["cache_schema"]) != CACHE_SCHEMA:
                return False
            if str(cache["well_id"]) != well_id:
                return False
            if str(cache["config_fingerprint"]) != config_fingerprint:
                return False
            hidden_md = np.asarray(cache["hidden_md"], dtype=float)
            baseline = np.asarray(cache["baseline_hidden"], dtype=float)
            prefix_md = np.asarray(cache["prefix_md"], dtype=float)
            residual = np.asarray(cache["prefix_residual"], dtype=float)
            return bool(
                hidden_md.ndim == 1
                and hidden_md.size > 0
                and hidden_md.shape == baseline.shape
                and prefix_md.shape == residual.shape
                and prefix_md.size == PREFIX_CALIBRATION_ROWS
                and np.all(np.isfinite(baseline))
                and np.all(np.isfinite(residual))
            )
    except (OSError, ValueError, KeyError):
        return False


def _build_one_cache(payload: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    horizontal_text, typewell_text, cache_text, fingerprint, force = payload
    horizontal_path = Path(horizontal_text)
    typewell_path = Path(typewell_text)
    cache_path = Path(cache_text)
    well_id = _well_id(horizontal_path)
    started = time.perf_counter()
    if not force and _cache_is_valid(
        cache_path,
        well_id=well_id,
        config_fingerprint=fingerprint,
    ):
        with np.load(cache_path, allow_pickle=False) as cache:
            return {
                "well_id": well_id,
                "cache_hit": True,
                "runtime_seconds": float(time.perf_counter() - started),
                "hidden_rows": int(len(cache["hidden_md"])),
                "prefix_rows": int(len(cache["prefix_md"])),
                "centroid": [
                    float(cache["centroid_x"]),
                    float(cache["centroid_y"]),
                ],
            }

    # This worker never reads the hidden TVT target column.
    horizontal = pd.read_csv(horizontal_path, usecols=list(HORIZONTAL_INPUT_COLUMNS))
    horizontal = horizontal.loc[:, HORIZONTAL_INPUT_COLUMNS].copy()
    typewell = pd.read_csv(typewell_path, usecols=list(TYPEWELL_INPUT_COLUMNS))
    typewell = typewell.loc[:, TYPEWELL_INPUT_COLUMNS].copy()
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    hidden = np.flatnonzero(~np.isfinite(tvt_input))
    if hidden.size == 0:
        raise ValueError(f"{well_id}: official TVT_input has no hidden suffix")
    first_hidden = int(hidden[0])
    if not np.all(~np.isfinite(tvt_input[first_hidden:])):
        raise ValueError(f"{well_id}: official hidden mask is not one suffix")
    if first_hidden <= PREFIX_CALIBRATION_ROWS:
        raise ValueError(f"{well_id}: visible prefix is too short for pseudo-suffix calibration")

    config = TrajectoryConfig()
    baseline_result = predict_trajectory(horizontal, typewell, config=config)
    hidden_mask = baseline_result.evaluation_mask
    hidden_indices = np.flatnonzero(hidden_mask)
    hidden_md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=float)[hidden_mask]
    baseline_hidden = np.asarray(baseline_result.prediction, dtype=float)[hidden_mask]

    # Truncate at the official boundary so the pseudo run cannot consume any
    # future hidden GR.  Only the final 256 finite TVT_input rows are masked.
    prefix_start = first_hidden - PREFIX_CALIBRATION_ROWS
    prefix_model = horizontal.iloc[:first_hidden].copy()
    prefix_model.loc[prefix_start:, "TVT_input"] = np.nan
    prefix_result = predict_trajectory(prefix_model, typewell, config=config)
    prefix_mask = prefix_result.evaluation_mask
    prefix_truth = tvt_input[:first_hidden][prefix_mask]
    prefix_prediction = np.asarray(prefix_result.prediction, dtype=float)[prefix_mask]
    prefix_md = pd.to_numeric(prefix_model["MD"], errors="coerce").to_numpy(dtype=float)[
        prefix_mask
    ]
    prefix_residual = prefix_truth - prefix_prediction
    if len(prefix_residual) != PREFIX_CALIBRATION_ROWS:
        raise AssertionError(f"{well_id}: unexpected pseudo-suffix length")
    if not np.all(np.isfinite(prefix_residual)):
        raise ValueError(f"{well_id}: non-finite pseudo-suffix residual")

    x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{well_id}: X/Y must be finite for the target-free spatial split")
    boundary_md = float(horizontal["MD"].iloc[first_hidden - 1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        cache_schema=np.asarray(CACHE_SCHEMA),
        well_id=np.asarray(well_id),
        config_fingerprint=np.asarray(fingerprint),
        hidden_indices=hidden_indices.astype(np.int32),
        hidden_md=hidden_md.astype(np.float64),
        baseline_hidden=baseline_hidden.astype(np.float64),
        boundary_md=np.asarray(boundary_md),
        prefix_md=prefix_md.astype(np.float64),
        prefix_residual=prefix_residual.astype(np.float64),
        centroid_x=np.asarray(float(np.mean(x))),
        centroid_y=np.asarray(float(np.mean(y))),
    )
    os.replace(temporary, cache_path)
    return {
        "well_id": well_id,
        "cache_hit": False,
        "runtime_seconds": float(time.perf_counter() - started),
        "hidden_rows": int(len(hidden_md)),
        "prefix_rows": int(len(prefix_md)),
        "centroid": [float(np.mean(x)), float(np.mean(y))],
    }


def _build_caches(
    paths: Iterable[Path],
    *,
    train_dir: Path,
    cache_dir: Path,
    config_fingerprint: str,
    workers: int,
    force: bool,
    phase: str,
) -> list[dict[str, Any]]:
    selected = list(paths)
    payloads = [
        (
            str(path),
            str(train_dir / f"{_well_id(path)}__typewell.csv"),
            str(_cache_path(cache_dir, _well_id(path))),
            config_fingerprint,
            force,
        )
        for path in selected
    ]
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_build_one_cache, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 10 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started
                hits = sum(bool(row["cache_hit"]) for row in results)
                print(
                    f"[{phase} cache {completed:>3}/{len(futures)}] "
                    f"hits={hits} built={completed - hits} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return sorted(results, key=lambda row: str(row["well_id"]))


def _read_centroids(paths: list[Path]) -> np.ndarray:
    centroids = np.empty((len(paths), 2), dtype=float)
    for index, path in enumerate(paths):
        frame = pd.read_csv(path, usecols=["X", "Y"])
        coordinates = frame.loc[:, ["X", "Y"]].apply(pd.to_numeric, errors="coerce")
        values = coordinates.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{_well_id(path)}: non-finite X/Y")
        centroids[index] = np.mean(values, axis=0)
        if (index + 1) % 100 == 0 or index + 1 == len(paths):
            print(f"[centroids {index + 1:>3}/{len(paths)}]", flush=True)
    return centroids


def _spatial_split(paths: list[Path], centroids: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = StandardScaler().fit_transform(centroids)
    raw = KMeans(
        n_clusters=SPATIAL_CLUSTERS,
        random_state=SPATIAL_RANDOM_STATE,
        n_init=SPATIAL_N_INIT,
    ).fit_predict(scaled)
    cluster_order = sorted(
        range(SPATIAL_CLUSTERS),
        key=lambda cluster: tuple(np.mean(centroids[raw == cluster], axis=0)),
    )
    remap = {cluster: rank for rank, cluster in enumerate(cluster_order)}
    labels = np.asarray([remap[int(label)] for label in raw], dtype=int)
    clusters = []
    for cluster in range(SPATIAL_CLUSTERS):
        indices = np.flatnonzero(labels == cluster)
        role = "confirmation" if cluster in CONFIRMATION_CLUSTER_RANKS else "discovery"
        clusters.append(
            {
                "cluster": cluster,
                "role": role,
                "wells": int(len(indices)),
                "centroid_x": float(np.mean(centroids[indices, 0])),
                "centroid_y": float(np.mean(centroids[indices, 1])),
                "x_min": float(np.min(centroids[indices, 0])),
                "x_max": float(np.max(centroids[indices, 0])),
                "y_min": float(np.min(centroids[indices, 1])),
                "y_max": float(np.max(centroids[indices, 1])),
                "well_ids": [_well_id(paths[index]) for index in indices],
            }
        )
    return labels, {
        "method": "StandardScaler(X/Y centroid) + KMeans; labels sorted by raw X/Y centroid",
        "random_state": SPATIAL_RANDOM_STATE,
        "n_init": SPATIAL_N_INIT,
        "clusters": SPATIAL_CLUSTERS,
        "confirmation_cluster_ranks": list(CONFIRMATION_CLUSTER_RANKS),
        "cluster_summary": clusters,
    }


def _load_cache(cache_dir: Path, well_id: str) -> dict[str, np.ndarray | float]:
    path = _cache_path(cache_dir, well_id)
    with np.load(path, allow_pickle=False) as cache:
        return {
            "hidden_indices": np.asarray(cache["hidden_indices"], dtype=int),
            "hidden_md": np.asarray(cache["hidden_md"], dtype=float),
            "baseline_hidden": np.asarray(cache["baseline_hidden"], dtype=float),
            "boundary_md": float(cache["boundary_md"]),
            "prefix_md": np.asarray(cache["prefix_md"], dtype=float),
            "prefix_residual": np.asarray(cache["prefix_residual"], dtype=float),
        }


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _score_split(
    paths: list[Path],
    labels: np.ndarray,
    selected_indices: np.ndarray,
    *,
    cache_dir: Path,
    candidate_names: tuple[str, ...],
    phase: str,
) -> list[dict[str, Any]]:
    candidates = {
        candidate.name: candidate
        for candidate in PREFIX_RESIDUAL_CANDIDATES
        if candidate.name in candidate_names
    }
    if set(candidates) != set(candidate_names):
        raise KeyError("candidate_names must come from the frozen family")
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(selected_indices, start=1):
        path = paths[int(index)]
        well_id = _well_id(path)
        cache = _load_cache(cache_dir, well_id)

        # Hidden truth enters only here, after the phase and candidate list have
        # been frozen by the caller.  It is never passed to either predictor.
        truth_full = pd.read_csv(path, usecols=["TVT"])["TVT"]
        truth_all = pd.to_numeric(truth_full, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        hidden_indices = np.asarray(cache["hidden_indices"], dtype=int)
        truth = truth_all[hidden_indices]
        valid = np.isfinite(truth)
        if not np.all(valid):
            truth = truth[valid]
        baseline = np.asarray(cache["baseline_hidden"], dtype=float)[valid]
        hidden_md = np.asarray(cache["hidden_md"], dtype=float)[valid]
        method_rmse = {"baseline": _rmse(truth, baseline)}
        correction_diagnostics: dict[str, Any] = {}
        for name, candidate in candidates.items():
            prediction, diagnostics = apply_prefix_residual(
                baseline,
                hidden_md,
                float(cache["boundary_md"]),
                np.asarray(cache["prefix_md"], dtype=float),
                np.asarray(cache["prefix_residual"], dtype=float),
                candidate,
            )
            method_rmse[name] = _rmse(truth, prediction)
            correction_diagnostics[name] = diagnostics
        rows.append(
            {
                "well_id": well_id,
                "cluster": int(labels[int(index)]),
                "scored_rows": int(len(truth)),
                "rmse": method_rmse,
                "correction_diagnostics": correction_diagnostics,
            }
        )
        if position % 100 == 0 or position == len(selected_indices):
            print(f"[{phase} score {position:>3}/{len(selected_indices)}]", flush=True)
    return rows


def _metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rmse = np.asarray([row["rmse"][method] for row in rows], dtype=float)
    weights = np.asarray([row["scored_rows"] for row in rows], dtype=float)
    worst = int(np.argmax(rmse))
    return {
        "pooled_rmse": float(np.sqrt(np.sum(weights * rmse**2) / np.sum(weights))),
        "well_macro_rmse": float(np.mean(rmse)),
        "well_median_rmse": float(np.median(rmse)),
        "well_p90_rmse": float(np.quantile(rmse, 0.90)),
        "well_p95_rmse": float(np.quantile(rmse, 0.95)),
        "well_p99_rmse": float(np.quantile(rmse, 0.99)),
        "worst_well_id": str(rows[worst]["well_id"]),
        "worst_well_rmse": float(rmse[worst]),
        "scored_wells": int(len(rows)),
        "scored_rows": int(np.sum(weights)),
    }


def _delta(corrected: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "pooled_rmse",
        "well_macro_rmse",
        "well_median_rmse",
        "well_p90_rmse",
        "well_p95_rmse",
        "well_p99_rmse",
        "worst_well_rmse",
    )
    return {key: float(corrected[key] - baseline[key]) for key in keys}


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    method: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline = np.asarray([row["rmse"]["baseline"] for row in rows], dtype=float)
    corrected = np.asarray([row["rmse"][method] for row in rows], dtype=float)
    weights = np.asarray([row["scored_rows"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    batch = 512
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        sampled = rng.integers(0, len(rows), size=(count, len(rows)))
        sampled_weights = weights[sampled]
        denominator = np.sum(sampled_weights, axis=1)
        baseline_pooled = np.sqrt(
            np.sum(sampled_weights * baseline[sampled] ** 2, axis=1) / denominator
        )
        corrected_pooled = np.sqrt(
            np.sum(sampled_weights * corrected[sampled] ** 2, axis=1) / denominator
        )
        deltas[start : start + count] = corrected_pooled - baseline_pooled
    return {
        "replicates": replicates,
        "seed": seed,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def _cluster_results(
    rows: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    output = []
    for cluster in sorted({int(row["cluster"]) for row in rows}):
        cluster_rows = [row for row in rows if int(row["cluster"]) == cluster]
        baseline = _metrics(cluster_rows, "baseline")
        corrected = _metrics(cluster_rows, method)
        output.append(
            {
                "cluster": cluster,
                "wells": len(cluster_rows),
                "scored_rows": baseline["scored_rows"],
                "baseline_pooled_rmse": baseline["pooled_rmse"],
                "corrected_pooled_rmse": corrected["pooled_rmse"],
                "pooled_delta": float(corrected["pooled_rmse"] - baseline["pooled_rmse"]),
                "macro_delta": float(corrected["well_macro_rmse"] - baseline["well_macro_rmse"]),
            }
        )
    return output


def _discovery_gate(
    delta: dict[str, float],
    bootstrap: dict[str, Any],
    cluster_results: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster_deltas = np.asarray(
        [row["pooled_delta"] for row in cluster_results],
        dtype=float,
    )
    checks = {
        "pooled_delta_at_most_minus_0_020": bool(
            delta["pooled_rmse"] <= DISCOVERY_POOLED_DELTA_LIMIT
        ),
        "macro_delta_negative": bool(delta["well_macro_rmse"] < 0.0),
        "bootstrap_ci95_upper_negative": bool(bootstrap["ci95_upper"] < 0.0),
        "at_least_3_of_4_discovery_clusters_nonpositive": bool(
            np.count_nonzero(cluster_deltas <= 0.0) >= DISCOVERY_REQUIRED_NONPOSITIVE_CLUSTERS
        ),
        "no_discovery_cluster_worse_than_plus_0_030": bool(
            np.max(cluster_deltas) <= DISCOVERY_MAX_CLUSTER_DELTA
        ),
        "p95_not_worse_than_plus_0_250": bool(delta["well_p95_rmse"] <= DISCOVERY_P95_DELTA_LIMIT),
        "worst_not_worse_than_plus_1_000": bool(
            delta["worst_well_rmse"] <= DISCOVERY_WORST_DELTA_LIMIT
        ),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "nonpositive_clusters": int(np.count_nonzero(cluster_deltas <= 0.0)),
        "maximum_cluster_delta": float(np.max(cluster_deltas)),
    }


def _confirmation_gate(
    delta: dict[str, float],
    bootstrap: dict[str, Any],
    cluster_results: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster_deltas = np.asarray(
        [row["pooled_delta"] for row in cluster_results],
        dtype=float,
    )
    checks = {
        "pooled_delta_at_most_minus_0_020": bool(
            delta["pooled_rmse"] <= CONFIRMATION_POOLED_DELTA_LIMIT
        ),
        "macro_delta_negative": bool(delta["well_macro_rmse"] < 0.0),
        "bootstrap_ci95_upper_negative": bool(bootstrap["ci95_upper"] < 0.0),
        "both_confirmation_clusters_nonpositive": bool(
            np.count_nonzero(cluster_deltas <= 0.0) >= CONFIRMATION_REQUIRED_NONPOSITIVE_CLUSTERS
        ),
        "no_confirmation_cluster_worse_than_plus_0_020": bool(
            np.max(cluster_deltas) <= CONFIRMATION_MAX_CLUSTER_DELTA
        ),
        "p95_not_worse_than_plus_0_250": bool(
            delta["well_p95_rmse"] <= CONFIRMATION_P95_DELTA_LIMIT
        ),
        "worst_not_worse_than_plus_1_000": bool(
            delta["worst_well_rmse"] <= CONFIRMATION_WORST_DELTA_LIMIT
        ),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "nonpositive_clusters": int(np.count_nonzero(cluster_deltas <= 0.0)),
        "maximum_cluster_delta": float(np.max(cluster_deltas)),
    }


def _summarize_method(
    rows: list[dict[str, Any]],
    method: str,
    *,
    phase: str,
) -> dict[str, Any]:
    baseline = _metrics(rows, "baseline")
    corrected = _metrics(rows, method)
    delta = _delta(corrected, baseline)
    bootstrap = _paired_bootstrap(rows, method)
    clusters = _cluster_results(rows, method)
    summary = {
        "baseline": baseline,
        "corrected": corrected,
        "delta_corrected_minus_baseline": delta,
        "paired_well_bootstrap_pooled_delta": bootstrap,
        "spatial_cluster_holdouts": clusters,
    }
    if phase == "discovery":
        summary["gate"] = _discovery_gate(delta, bootstrap, clusters)
    elif phase == "confirmation":
        summary["gate"] = _confirmation_gate(delta, bootstrap, clusters)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if len(paths) != 773:
        raise ValueError(f"frozen protocol expects 773 wells, found {len(paths)}")
    cache_dir = args.cache_dir.expanduser().resolve()
    output_path = args.json_output.expanduser().resolve()

    print("Reading target-free X/Y centroids for the frozen spatial split.", flush=True)
    centroids = _read_centroids(paths)
    labels, split_report = _spatial_split(paths, centroids)
    confirmation_mask = np.isin(labels, CONFIRMATION_CLUSTER_RANKS)
    discovery_indices = np.flatnonzero(~confirmation_mask)
    confirmation_indices = np.flatnonzero(confirmation_mask)
    if len(set(discovery_indices) & set(confirmation_indices)):
        raise AssertionError("discovery and confirmation overlap")
    if len(discovery_indices) + len(confirmation_indices) != len(paths):
        raise AssertionError("spatial split is incomplete")

    config = TrajectoryConfig()
    fingerprint = _config_fingerprint(config)
    print(
        f"Frozen split: discovery={len(discovery_indices)}, "
        f"confirmation={len(confirmation_indices)}.",
        flush=True,
    )
    discovery_cache = _build_caches(
        [paths[index] for index in discovery_indices],
        train_dir=train_dir,
        cache_dir=cache_dir,
        config_fingerprint=fingerprint,
        workers=args.workers,
        force=args.force_cache,
        phase="discovery",
    )
    candidate_names = tuple(candidate.name for candidate in PREFIX_RESIDUAL_CANDIDATES)
    discovery_rows = _score_split(
        paths,
        labels,
        discovery_indices,
        cache_dir=cache_dir,
        candidate_names=candidate_names,
        phase="discovery",
    )
    discovery_results = {
        name: _summarize_method(discovery_rows, name, phase="discovery") for name in candidate_names
    }
    passing = [
        name for name in candidate_names if discovery_results[name]["gate"]["decision"] == "PASS"
    ]
    selected_name = (
        min(
            passing,
            key=lambda name: (
                discovery_results[name]["delta_corrected_minus_baseline"]["pooled_rmse"],
                name,
            ),
        )
        if passing
        else None
    )
    print(f"Frozen discovery selection: {selected_name or 'none (all failed)'}.", flush=True)

    report: dict[str, Any] = {
        "evaluation_contract": {
            "model_horizontal_columns": list(HORIZONTAL_INPUT_COLUMNS),
            "model_typewell_columns": list(TYPEWELL_INPUT_COLUMNS),
            "prefix_calibration": (
                "last 256 finite TVT_input rows are pseudo-masked; pseudo frame is "
                "truncated at the official boundary before predict_trajectory"
            ),
            "hidden_target_role": (
                "TVT is absent from prediction caches and loaded only in the scoring phase"
            ),
            "same_id_lookup": False,
            "formation_feature": False,
            "train_target_feature": False,
            "leaderboard_bias": False,
            "threshold_tuning_after_confirmation": False,
        },
        "protocol": {
            "wells": len(paths),
            "workers": args.workers,
            "prefix_calibration_rows": PREFIX_CALIBRATION_ROWS,
            "trajectory_config": _json_safe(asdict(config)),
            "trajectory_config_fingerprint": fingerprint,
            "spatial_split": split_report,
            "discovery_wells": len(discovery_indices),
            "confirmation_wells": len(confirmation_indices),
            "candidate_family_frozen_before_discovery": [
                candidate.to_dict() for candidate in PREFIX_RESIDUAL_CANDIDATES
            ],
            "selection_rule": (
                "among discovery gate passers, select the most negative pooled delta; "
                "candidate name is the deterministic tie-break"
            ),
            "discovery_gate": {
                "pooled_delta_limit": DISCOVERY_POOLED_DELTA_LIMIT,
                "required_nonpositive_clusters": DISCOVERY_REQUIRED_NONPOSITIVE_CLUSTERS,
                "maximum_cluster_delta": DISCOVERY_MAX_CLUSTER_DELTA,
                "p95_delta_limit": DISCOVERY_P95_DELTA_LIMIT,
                "worst_delta_limit": DISCOVERY_WORST_DELTA_LIMIT,
                "bootstrap_ci95_upper_must_be_negative": True,
                "macro_delta_must_be_negative": True,
            },
            "confirmation_gate": {
                "pooled_delta_limit": CONFIRMATION_POOLED_DELTA_LIMIT,
                "required_nonpositive_clusters": (CONFIRMATION_REQUIRED_NONPOSITIVE_CLUSTERS),
                "maximum_cluster_delta": CONFIRMATION_MAX_CLUSTER_DELTA,
                "p95_delta_limit": CONFIRMATION_P95_DELTA_LIMIT,
                "worst_delta_limit": CONFIRMATION_WORST_DELTA_LIMIT,
                "bootstrap_ci95_upper_must_be_negative": True,
                "macro_delta_must_be_negative": True,
            },
        },
        "cache": {
            "schema": CACHE_SCHEMA,
            "directory": str(cache_dir),
            "contains_hidden_tvt_truth": False,
            "discovery": {
                "wells": len(discovery_cache),
                "hits": sum(bool(row["cache_hit"]) for row in discovery_cache),
                "built": sum(not bool(row["cache_hit"]) for row in discovery_cache),
            },
        },
        "discovery": {
            "candidate_results": discovery_results,
            "passing_candidates": passing,
            "frozen_candidate": selected_name,
            "per_well": discovery_rows,
        },
        "confirmation": None,
        "full_summary": None,
        "decision": {
            "decision": "NO-GO",
            "reason": "no candidate passed the frozen discovery gate",
            "frozen_candidate": selected_name,
            "confirmation_evaluations": 0,
        },
    }

    if selected_name is not None:
        confirmation_cache = _build_caches(
            [paths[index] for index in confirmation_indices],
            train_dir=train_dir,
            cache_dir=cache_dir,
            config_fingerprint=fingerprint,
            workers=args.workers,
            force=args.force_cache,
            phase="confirmation",
        )
        confirmation_rows = _score_split(
            paths,
            labels,
            confirmation_indices,
            cache_dir=cache_dir,
            candidate_names=(selected_name,),
            phase="confirmation",
        )
        confirmation_summary = _summarize_method(
            confirmation_rows,
            selected_name,
            phase="confirmation",
        )
        full_rows = sorted(
            [
                *[
                    {
                        **row,
                        "rmse": {
                            "baseline": row["rmse"]["baseline"],
                            selected_name: row["rmse"][selected_name],
                        },
                        "correction_diagnostics": {
                            selected_name: row["correction_diagnostics"][selected_name]
                        },
                    }
                    for row in discovery_rows
                ],
                *confirmation_rows,
            ],
            key=lambda row: str(row["well_id"]),
        )
        full_summary = _summarize_method(full_rows, selected_name, phase="full")
        confirmation_pass = confirmation_summary["gate"]["decision"] == "PASS"
        report["cache"]["confirmation"] = {
            "wells": len(confirmation_cache),
            "hits": sum(bool(row["cache_hit"]) for row in confirmation_cache),
            "built": sum(not bool(row["cache_hit"]) for row in confirmation_cache),
        }
        report["confirmation"] = {
            "frozen_candidate": selected_name,
            "candidate_family_members_scored": 1,
            "summary": confirmation_summary,
            "per_well": confirmation_rows,
        }
        report["full_summary"] = {
            "frozen_candidate": selected_name,
            "summary": full_summary,
            "per_well": full_rows,
        }
        report["decision"] = {
            "decision": "GO" if confirmation_pass else "NO-GO",
            "reason": (
                "frozen candidate passed discovery and untouched confirmation gates"
                if confirmation_pass
                else "frozen candidate failed the untouched confirmation gate"
            ),
            "frozen_candidate": selected_name,
            "confirmation_evaluations": 1,
        }

    report["runtime_seconds"] = float(time.perf_counter() - started)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["artifact_sha256"] = _sha256(output_path)
    # Rewrite once to include the hash of the content before this field.  This is
    # explicitly labelled by the field name in the final report.
    output_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Decision={report['decision']['decision']} "
        f"candidate={selected_name} runtime={report['runtime_seconds']:.1f}s",
        flush=True,
    )
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
