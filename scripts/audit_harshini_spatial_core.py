"""Audit the Harshini public notebook's fixed core on strict spatial holdouts.

The public notebook reports leave-one-well-out surface features, but its random
well folds can still use formation surfaces from nearby validation-region
wells. This audit removes the complete spatial validation cluster from the
surface point cloud before reconstructing the candidate's fixed imputation/PF
blend. It deliberately stops before the boosted residual ensemble: a failure
of this cheaper fixed core is enough to reject the expensive route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
DATUM_WINDOWS = (("full", None), ("late2", 1500), ("late", 500), ("last", 150))
PR_WINDOW = 500
SHRINK = 0.60
CLIP = 80.0
PRODUCTION_WEIGHT = 0.55
PF_WEIGHT = 0.45
WARMUP_TAU = 85.0


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Load array/dict caches while rejecting arbitrary pickle globals."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")
        return super().find_class(module, name)


def _restricted_pickle_load(handle: BinaryIO) -> Any:
    return _NumpyOnlyUnpickler(handle).load()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--imputation-cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "public_scout_20260727_late"
            / "harshini_f_output"
            / "imp_cache.pkl"
        ),
    )
    parser.add_argument(
        "--pf-cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "public_scout_20260727_late"
            / "harshini_f_output"
            / "pf_train_cache.pkl"
        ),
    )
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        default=None,
        help="Evaluate one or more fold IDs; omit to evaluate every fold.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_start(frame: pd.DataFrame) -> int:
    missing = frame["TVT_input"].isna().to_numpy()
    return int(np.argmax(missing)) if missing.any() else len(frame)


def _horizontal_paths(train_dir: Path) -> dict[str, Path]:
    suffix = "__horizontal_well.csv"
    return {
        path.name.removesuffix(suffix): path
        for path in sorted(train_dir.glob(f"*{suffix}"))
    }


def _load_field(
    paths: dict[str, Path],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    dict[str, cKDTree],
    dict[str, np.ndarray],
]:
    points: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    trees: dict[str, cKDTree] = {}
    centroids: dict[str, np.ndarray] = {}
    usecols = ["X", "Y", *FORMATIONS]
    for index, (well_id, path) in enumerate(paths.items(), start=1):
        frame = pd.read_csv(path, usecols=usecols)
        step = np.arange(0, len(frame), 20)
        values = {
            formation: frame[formation].to_numpy(dtype=float)[step]
            for formation in FORMATIONS
        }
        valid = np.logical_and.reduce(
            [np.isfinite(values[formation]) for formation in FORMATIONS]
        )
        if np.count_nonzero(valid) < 5:
            continue
        x = frame["X"].to_numpy(dtype=float)[step][valid]
        y = frame["Y"].to_numpy(dtype=float)[step][valid]
        surface = {formation: values[formation][valid] for formation in FORMATIONS}
        points[well_id] = (x, y, surface)
        trees[well_id] = cKDTree(np.c_[x, y])
        centroids[well_id] = np.array([float(np.mean(x)), float(np.mean(y))])
        if index % 150 == 0:
            print(f"field {index}/{len(paths)}", flush=True)
    return points, trees, centroids


def _fold_labels(
    centroids: dict[str, np.ndarray],
    clusters: int,
    seed: int,
) -> dict[str, int]:
    well_ids = sorted(centroids)
    coordinates = np.stack([centroids[well_id] for well_id in well_ids])
    standardized = StandardScaler().fit_transform(coordinates)
    raw = KMeans(n_clusters=clusters, random_state=seed, n_init=20).fit_predict(
        standardized
    )
    raw_centers = np.stack(
        [coordinates[raw == cluster].mean(axis=0) for cluster in range(clusters)]
    )
    order = sorted(
        range(clusters),
        key=lambda cluster: (raw_centers[cluster, 0], raw_centers[cluster, 1]),
    )
    remap = {cluster: fold for fold, cluster in enumerate(order)}
    return {
        well_id: remap[int(label)]
        for well_id, label in zip(well_ids, raw, strict=True)
    }


def _linear_trends(
    allowed: list[str],
    points: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
) -> tuple[float, float, dict[str, np.ndarray]]:
    all_x = np.concatenate([points[well_id][0] for well_id in allowed])
    all_y = np.concatenate([points[well_id][1] for well_id in allowed])
    x0 = float(np.mean(all_x))
    y0 = float(np.mean(all_y))
    trends: dict[str, np.ndarray] = {}
    for formation in FORMATIONS:
        values = np.concatenate([points[well_id][2][formation] for well_id in allowed])
        design = np.c_[all_x - x0, all_y - y0, np.ones(len(all_x))]
        trends[formation] = np.linalg.lstsq(design, values, rcond=None)[0]
    return x0, y0, trends


def _impute_location(
    x_query: np.ndarray,
    y_query: np.ndarray,
    *,
    allowed: list[str],
    points: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    trees: dict[str, cKDTree],
    centroids: dict[str, np.ndarray],
    global_trend: tuple[float, float, dict[str, np.ndarray]],
    nearest_wells: int = 8,
    candidates: int = 25,
    length_scale: float = 600.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    center_x = float(np.mean(x_query))
    center_y = float(np.mean(y_query))
    ranked = sorted(
        allowed,
        key=lambda well_id: float(
            np.sum((centroids[well_id] - np.array([center_x, center_y])) ** 2)
        ),
    )[:candidates]
    if len(ranked) < 3:
        raise ValueError("fewer than three source wells after spatial exclusion")

    plane: dict[str, np.ndarray | None] = {}
    plane_x = np.concatenate([points[well_id][0] for well_id in ranked])
    plane_y = np.concatenate([points[well_id][1] for well_id in ranked])
    for formation in FORMATIONS:
        values = np.concatenate(
            [points[well_id][2][formation] for well_id in ranked]
        )
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 50:
            plane[formation] = None
            continue
        design = np.c_[
            plane_x[valid] - center_x,
            plane_y[valid] - center_y,
            np.ones(np.count_nonzero(valid)),
        ]
        plane[formation] = np.linalg.lstsq(design, values[valid], rcond=None)[0]

    global_x, global_y, global_coefficients = global_trend

    def trend(formation: str, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        coefficients = plane[formation]
        if coefficients is not None:
            return (
                coefficients[0] * (np.asarray(x) - center_x)
                + coefficients[1] * (np.asarray(y) - center_y)
                + coefficients[2]
            )
        fallback = global_coefficients[formation]
        return (
            fallback[0] * (np.asarray(x) - global_x)
            + fallback[1] * (np.asarray(y) - global_y)
            + fallback[2]
        )

    query = np.c_[x_query, y_query]
    distances = np.empty((len(ranked), len(query)))
    indices = np.empty((len(ranked), len(query)), dtype=int)
    for source_index, well_id in enumerate(ranked):
        distance, point_index = trees[well_id].query(query, k=1)
        distances[source_index] = distance
        indices[source_index] = point_index
    order = np.argsort(distances, axis=0)[:nearest_wells]
    output = {
        formation: np.empty(len(query), dtype=float) for formation in FORMATIONS
    }
    nearest_distance = np.empty(len(query), dtype=float)
    for query_index in range(len(query)):
        source_indices = order[:, query_index]
        source_wells = [ranked[index] for index in source_indices]
        point_indices = [indices[index, query_index] for index in source_indices]
        distance = distances[source_indices, query_index]
        nearest_distance[query_index] = float(np.min(distance))
        weights = 1.0 / (distance**2 + length_scale**2)
        for formation in FORMATIONS:
            residuals = np.array(
                [
                    points[well_id][2][formation][point_index]
                    - trend(
                        formation,
                        points[well_id][0][point_index],
                        points[well_id][1][point_index],
                    )
                    for well_id, point_index in zip(
                        source_wells, point_indices, strict=True
                    )
                ]
            )
            valid = np.isfinite(residuals)
            residual = (
                float(np.sum(residuals[valid] * weights[valid]) / np.sum(weights[valid]))
                if np.any(valid)
                else 0.0
            )
            output[formation][query_index] = residual + trend(
                formation, x_query[query_index], y_query[query_index]
            )
    return output, nearest_distance


def _aggregate(predictions: dict[str, np.ndarray], reliability: dict[str, float]) -> np.ndarray:
    keys = [key for key in predictions if key in reliability]
    matrix = np.stack([predictions[key] for key in keys])
    weights = np.array([1.0 / (reliability[key] ** 2 + 1e-6) for key in keys])
    weights /= np.sum(weights)
    return np.sum(matrix * weights[:, None], axis=0)


def _rebuild_imputation(
    frame: pd.DataFrame,
    prediction_start: int,
    *,
    allowed: list[str],
    points: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    trees: dict[str, cKDTree],
    centroids: dict[str, np.ndarray],
    global_trend: tuple[float, float, dict[str, np.ndarray]],
) -> tuple[np.ndarray, float]:
    evaluation, nearest = _impute_location(
        frame["X"].to_numpy(dtype=float)[prediction_start:],
        frame["Y"].to_numpy(dtype=float)[prediction_start:],
        allowed=allowed,
        points=points,
        trees=trees,
        centroids=centroids,
        global_trend=global_trend,
    )
    prefix, _ = _impute_location(
        frame["X"].to_numpy(dtype=float)[:prediction_start],
        frame["Y"].to_numpy(dtype=float)[:prediction_start],
        allowed=allowed,
        points=points,
        trees=trees,
        centroids=centroids,
        global_trend=global_trend,
    )
    z_evaluation = frame["Z"].to_numpy(dtype=float)[prediction_start:]
    z_prefix = frame["Z"].to_numpy(dtype=float)[:prediction_start]
    known_tvt = frame["TVT_input"].to_numpy(dtype=float)[:prediction_start]
    near_prefix = slice(-min(PR_WINDOW, prediction_start), None)
    predictions: dict[str, np.ndarray] = {}
    reliability: dict[str, float] = {}
    for formation in FORMATIONS:
        datum_residual = known_tvt + z_prefix - prefix[formation]
        for tag, window in DATUM_WINDOWS:
            selected = (
                slice(None)
                if window is None
                else slice(-min(window, prediction_start), None)
            )
            datum = float(np.nanmedian(datum_residual[selected]))
            key = f"{formation}_{tag}"
            predictions[key] = -z_evaluation + evaluation[formation] + datum
            fitted_prefix = -z_prefix[near_prefix] + prefix[formation][near_prefix] + datum
            reliability[key] = float(
                np.sqrt(np.nanmean((known_tvt[near_prefix] - fitted_prefix) ** 2))
            )
    return _aggregate(predictions, reliability), float(np.mean(nearest))


def _sum_squared_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum((prediction - truth) ** 2))


def _evaluate_fold(
    fold: int,
    *,
    paths: dict[str, Path],
    imputation_cache: dict[str, Any],
    pf_cache: dict[str, Any],
    points: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    trees: dict[str, cKDTree],
    centroids: dict[str, np.ndarray],
    fold_labels: dict[str, int],
) -> dict[str, Any]:
    validation_wells = sorted(
        set(imputation_cache)
        & set(pf_cache)
        & {well_id for well_id, label in fold_labels.items() if label == fold}
    )
    excluded = {
        well_id for well_id, label in fold_labels.items() if label == fold
    }
    allowed = sorted(set(points) - excluded)
    global_trend = _linear_trends(allowed, points)
    totals = {
        "flat": 0.0,
        "pf": 0.0,
        "loo_core": 0.0,
        "spatial_imputation": 0.0,
        "spatial_core": 0.0,
    }
    row_count = 0
    nearest_distances: list[float] = []
    started = time.monotonic()
    usecols = ["MD", "X", "Y", "Z", "TVT_input", "TVT"]
    for index, well_id in enumerate(validation_wells, start=1):
        frame = pd.read_csv(paths[well_id], usecols=usecols).sort_values("MD")
        prediction_start = _prediction_start(frame)
        truth = frame["TVT"].to_numpy(dtype=float)[prediction_start:]
        flat = np.full(len(truth), frame["TVT_input"].iloc[prediction_start - 1])
        md = frame["MD"].to_numpy(dtype=float)
        md_since = md[prediction_start:] - md[prediction_start - 1]
        ramp = 1.0 - np.exp(-np.maximum(md_since, 0.0) / WARMUP_TAU)
        pf = pf_cache[well_id]
        pf_mean = (
            np.asarray(pf["s3"], dtype=float)
            + np.asarray(pf["s5"], dtype=float)
            + np.asarray(pf["s8"], dtype=float)
        ) / 3.0
        if len(pf_mean) != len(truth):
            raise ValueError(f"{well_id}: PF length does not match truth")

        spatial_imputation, nearest = _rebuild_imputation(
            frame,
            prediction_start,
            allowed=allowed,
            points=points,
            trees=trees,
            centroids=centroids,
            global_trend=global_trend,
        )
        nearest_distances.append(nearest)
        spatial_delta = SHRINK * np.clip(spatial_imputation - flat, -CLIP, CLIP)
        pf_delta = pf_mean - flat
        spatial_core_delta = (
            PRODUCTION_WEIGHT * spatial_delta + PF_WEIGHT * pf_delta
        )

        cached = imputation_cache[well_id]
        cached_imputation = _aggregate(cached["P"], cached["PR"])
        loo_delta = SHRINK * np.clip(cached_imputation - flat, -CLIP, CLIP)
        loo_core_delta = PRODUCTION_WEIGHT * loo_delta + PF_WEIGHT * pf_delta

        predictions = {
            "flat": flat,
            "pf": flat + ramp * pf_delta,
            "loo_core": flat + ramp * loo_core_delta,
            "spatial_imputation": flat + ramp * spatial_delta,
            "spatial_core": flat + ramp * spatial_core_delta,
        }
        for name, prediction in predictions.items():
            totals[name] += _sum_squared_error(prediction, truth)
        row_count += len(truth)
        if index % 10 == 0 or index == len(validation_wells):
            elapsed = time.monotonic() - started
            print(
                f"fold {fold}: {index}/{len(validation_wells)} wells "
                f"[{elapsed:.0f}s]",
                flush=True,
            )
    scores = {
        name: float(np.sqrt(sum_squared_error / row_count))
        for name, sum_squared_error in totals.items()
    }
    return {
        "fold": fold,
        "validation_wells": len(validation_wells),
        "excluded_field_wells": len(excluded),
        "source_field_wells": len(allowed),
        "rows": row_count,
        "mean_nearest_source_distance": float(np.mean(nearest_distances)),
        "scores": scores,
        "spatial_core_delta_vs_flat": scores["spatial_core"] - scores["flat"],
        "spatial_core_delta_vs_loo_core": scores["spatial_core"] - scores["loo_core"],
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    args = _parser().parse_args()
    train_dir = args.data_root / "train"
    paths = _horizontal_paths(train_dir)
    with args.imputation_cache.open("rb") as handle:
        imputation_cache = _restricted_pickle_load(handle)
    with args.pf_cache.open("rb") as handle:
        pf_cache = _restricted_pickle_load(handle)
    print(
        f"loaded imputation={len(imputation_cache)} PF={len(pf_cache)} wells",
        flush=True,
    )
    points, trees, centroids = _load_field(paths)
    labels = _fold_labels(centroids, args.clusters, args.seed)
    folds = list(range(args.clusters)) if args.fold is None else sorted(set(args.fold))
    if any(fold < 0 or fold >= args.clusters for fold in folds):
        raise ValueError(f"folds must be in [0, {args.clusters})")
    results = [
        _evaluate_fold(
            fold,
            paths=paths,
            imputation_cache=imputation_cache,
            pf_cache=pf_cache,
            points=points,
            trees=trees,
            centroids=centroids,
            fold_labels=labels,
        )
        for fold in folds
    ]
    total_rows = sum(result["rows"] for result in results)
    pooled = {}
    for name in results[0]["scores"]:
        pooled[name] = float(
            np.sqrt(
                sum(
                    result["scores"][name] ** 2 * result["rows"] for result in results
                )
                / total_rows
            )
        )
    report = {
        "protocol": {
            "name": "strict_spatial_cluster_surface_exclusion",
            "clusters": args.clusters,
            "seed": args.seed,
            "folds_evaluated": folds,
            "surface_source_exclusion": "all wells in validation spatial cluster",
            "fixed_core": {
                "imputation_shrink": SHRINK,
                "imputation_clip": CLIP,
                "production_weight": PRODUCTION_WEIGHT,
                "pf_weight": PF_WEIGHT,
                "warmup_tau": WARMUP_TAU,
            },
        },
        "inputs": {
            "imputation_cache": str(args.imputation_cache),
            "imputation_cache_sha256": _sha256(args.imputation_cache),
            "pf_cache": str(args.pf_cache),
            "pf_cache_sha256": _sha256(args.pf_cache),
            "field_wells": len(points),
            "candidate_wells": len(set(imputation_cache) & set(pf_cache)),
        },
        "folds": results,
        "pooled_scores": pooled,
        "pooled_spatial_core_delta_vs_flat": pooled["spatial_core"] - pooled["flat"],
        "pooled_spatial_core_delta_vs_loo_core": (
            pooled["spatial_core"] - pooled["loo_core"]
        ),
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
