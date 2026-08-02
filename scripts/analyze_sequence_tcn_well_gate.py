"""Cross-fit a target-free well gate for the frozen fold-0 TCN pilot.

The TCN checkpoint and overlay scale are frozen.  This script only asks
whether target-free summaries can identify wells where applying that overlay
is likely to help.  Gate predictions are strictly out-of-spatial-subcluster:
the held-out wells' labels are never used to fit their gate.

This remains an offline diagnostic.  It does not train a new sequence model,
build a competition submission, or authorize one directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sequence_tcn_pilot import (
    PRIMARY_SCALE,
    PROTOCOL,
    VALIDATION_FOLD,
    SequenceResidualTCN,
    _load_cache,
    _partition,
    _predict_well,
    _rmse,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 20_260_729
OUTER_SPATIAL_FOLDS = 5
PCA_COMPONENTS = 12
LOGISTIC_C = 0.5
PROBABILITY_THRESHOLD = 0.5
MINIMUM_SELECTED_WELL_FRACTION = 0.20
MINIMUM_SELECTED_WIN_FRACTION = 0.55


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spatial_subclusters(
    metadata_rows: list[dict[str, Any]],
    validation_wells: np.ndarray,
) -> np.ndarray:
    """Cluster fingerprint groups so duplicate type wells share an outer fold."""
    grouped: dict[str, list[int]] = {}
    for local_index, well_index in enumerate(validation_wells):
        fingerprint = metadata_rows[int(well_index)]["typewell_fingerprint"]
        grouped.setdefault(fingerprint, []).append(local_index)
    fingerprints = sorted(grouped)
    group_centroids = np.asarray(
        [
            [
                np.mean(
                    [
                        metadata_rows[int(validation_wells[index])]["centroid_x"]
                        for index in grouped[fingerprint]
                    ]
                ),
                np.mean(
                    [
                        metadata_rows[int(validation_wells[index])]["centroid_y"]
                        for index in grouped[fingerprint]
                    ]
                ),
            ]
            for fingerprint in fingerprints
        ],
        dtype=np.float64,
    )
    group_centroids = StandardScaler().fit_transform(group_centroids)
    group_folds = KMeans(
        n_clusters=OUTER_SPATIAL_FOLDS,
        random_state=RANDOM_SEED,
        n_init=20,
    ).fit_predict(group_centroids)
    result = np.empty(len(validation_wells), dtype=np.int64)
    for fingerprint, fold in zip(fingerprints, group_folds, strict=True):
        result[grouped[fingerprint]] = int(fold)
    return result


def _target_free_summary(
    normalized_features: np.ndarray,
    base: np.ndarray,
    raw: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Summarize only model inputs and predictions; no target is accepted."""
    values = np.asarray(normalized_features, dtype=np.float64)
    base_values = np.asarray(base, dtype=np.float64)
    raw_values = np.asarray(raw, dtype=np.float64)
    feature_count = values.shape[1]
    names = [f"input_mean_{index:02d}" for index in range(feature_count)]
    names += [f"input_std_{index:02d}" for index in range(feature_count)]
    summary = [
        *np.mean(values, axis=0),
        *np.std(values, axis=0),
        math.log1p(len(raw_values)),
        float(np.mean(base_values)),
        float(np.std(base_values)),
        float(np.mean(raw_values)),
        float(np.std(raw_values)),
        float(np.mean(np.abs(raw_values))),
        float(np.quantile(np.abs(raw_values), 0.90)),
        float(np.mean(raw_values > 0.0)),
    ]
    names += [
        "log_rows",
        "base_mean",
        "base_std",
        "raw_mean",
        "raw_std",
        "raw_abs_mean",
        "raw_abs_p90",
        "raw_positive_fraction",
    ]
    for fraction, label in ((0.20, "head20"), (0.20, "tail20")):
        count = max(1, int(math.ceil(len(raw_values) * fraction)))
        section = raw_values[:count] if label == "head20" else raw_values[-count:]
        summary.extend([float(np.mean(section)), float(np.std(section))])
        names.extend([f"raw_{label}_mean", f"raw_{label}_std"])
    return np.asarray(summary, dtype=np.float64), names


def _gate_pipeline(sample_count: int, feature_count: int) -> Pipeline:
    components = min(PCA_COMPONENTS, feature_count, max(1, sample_count - 1))
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=components, random_state=RANDOM_SEED)),
            (
                "classifier",
                LogisticRegression(
                    C=LOGISTIC_C,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _metrics(
    frame: pd.DataFrame,
    *,
    candidate_mask: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(candidate_mask, dtype=bool)
    candidate_sse = np.where(mask, frame["overlay_sse"], frame["base_sse"])
    candidate_rmse = np.where(mask, frame["overlay_rmse"], frame["base_rmse"])
    base_pooled = math.sqrt(float(frame["base_sse"].sum() / frame["rows"].sum()))
    pooled = math.sqrt(float(candidate_sse.sum() / frame["rows"].sum()))
    return {
        "pooled_rmse": pooled,
        "pooled_delta": pooled - base_pooled,
        "macro_well_rmse": float(np.mean(candidate_rmse)),
        "p95_well_delta": float(
            np.quantile(candidate_rmse, 0.95) - np.quantile(frame["base_rmse"], 0.95)
        ),
        "worst_well_delta": float(np.max(candidate_rmse) - np.max(frame["base_rmse"])),
        "nonpositive_well_fraction": float(np.mean(candidate_rmse <= frame["base_rmse"])),
    }


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    cache_dir = args.cache_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    pilot_report_path = args.pilot_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cache = _load_cache(cache_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["feature_names"] != cache["metadata"]["features"]:
        raise ValueError("checkpoint and cache feature names differ")
    center = np.asarray(checkpoint["center"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    model = SequenceResidualTCN(len(center)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, validation_wells = _partition(cache["metadata"], smoke_wells=0)
    outer_folds = _spatial_subclusters(
        cache["metadata"]["per_well"],
        validation_wells,
    )
    summaries: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    summary_names: list[str] | None = None
    offsets = cache["offsets"]
    for progress, well_index in enumerate(validation_wells, start=1):
        start = int(offsets[well_index])
        stop = int(offsets[well_index + 1])
        features = np.asarray(cache["features"][start:stop], dtype=np.float32)
        normalized = np.clip((features - center) / scale, -8.0, 8.0)
        normalized[~np.isfinite(normalized)] = 0.0
        base = np.asarray(cache["base"][start:stop], dtype=np.float32)
        truth = np.asarray(cache["truth"][start:stop], dtype=np.float32)
        raw = _predict_well(model, features, center, scale, device)
        overlay = base + PRIMARY_SCALE * raw
        summary, names = _target_free_summary(normalized, base, raw)
        if summary_names is None:
            summary_names = names
        elif names != summary_names:
            raise RuntimeError("gate feature schema changed between wells")
        summaries.append(summary)
        rows.append(
            {
                "well": cache["metadata"]["per_well"][well_index]["well"],
                "typewell_fingerprint": cache["metadata"]["per_well"][well_index][
                    "typewell_fingerprint"
                ],
                "outer_spatial_fold": int(outer_folds[progress - 1]),
                "rows": int(stop - start),
                "base_sse": float(np.sum(np.square(truth - base), dtype=np.float64)),
                "overlay_sse": float(np.sum(np.square(truth - overlay), dtype=np.float64)),
                "base_rmse": _rmse(truth, base),
                "overlay_rmse": _rmse(truth, overlay),
            }
        )
        if progress % 25 == 0 or progress == len(validation_wells):
            print(
                f"[infer {progress:3d}/{len(validation_wells)}] "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    matrix = np.vstack(summaries)
    labels = (frame["overlay_rmse"] <= frame["base_rmse"]).to_numpy(dtype=np.int64)
    probabilities = np.full(len(frame), np.nan, dtype=np.float64)
    fold_details: list[dict[str, Any]] = []
    for fold in range(OUTER_SPATIAL_FOLDS):
        test = outer_folds == fold
        train = ~test
        if len(np.unique(labels[train])) < 2:
            probabilities[test] = float(np.mean(labels[train]))
        else:
            pipeline = _gate_pipeline(int(train.sum()), matrix.shape[1])
            row_weights = np.sqrt(
                frame.loc[train, "rows"].to_numpy(dtype=np.float64)
                / float(np.median(frame.loc[train, "rows"]))
            )
            pipeline.fit(
                matrix[train],
                labels[train],
                classifier__sample_weight=row_weights,
            )
            probabilities[test] = pipeline.predict_proba(matrix[test])[:, 1]
        fold_details.append(
            {
                "fold": fold,
                "training_wells": int(train.sum()),
                "test_wells": int(test.sum()),
                "training_win_fraction": float(np.mean(labels[train])),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("outer cross-fit did not predict every validation well")

    selected = probabilities >= PROBABILITY_THRESHOLD
    frame["overlay_delta"] = frame["overlay_rmse"] - frame["base_rmse"]
    frame["gate_probability"] = probabilities
    frame["gate_selected"] = selected
    frame["gated_rmse"] = np.where(
        selected,
        frame["overlay_rmse"],
        frame["base_rmse"],
    )
    frame["gated_delta"] = frame["gated_rmse"] - frame["base_rmse"]

    base_pooled = math.sqrt(float(frame["base_sse"].sum() / frame["rows"].sum()))
    ungated_metrics = _metrics(frame, candidate_mask=np.ones(len(frame), dtype=bool))
    gated_metrics = _metrics(frame, candidate_mask=selected)
    selected_win_fraction = float(np.mean(labels[selected])) if selected.any() else 0.0
    selected_well_fraction = float(np.mean(selected))
    improving_folds = 0
    for detail in fold_details:
        fold_mask = outer_folds == detail["fold"]
        fold_metrics = _metrics(
            frame.loc[fold_mask].reset_index(drop=True),
            candidate_mask=selected[fold_mask],
        )
        detail["selected_wells"] = int(np.sum(selected[fold_mask]))
        detail["selected_win_fraction"] = (
            float(np.mean(labels[fold_mask][selected[fold_mask]]))
            if np.any(selected[fold_mask])
            else 0.0
        )
        detail["gated_pooled_delta"] = fold_metrics["pooled_delta"]
        improving_folds += int(fold_metrics["pooled_delta"] < 0.0)

    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    pilot_primary = pilot_report["validation"]["scale_results"][f"{PRIMARY_SCALE:.2f}"]
    reproduction = {
        "base_pooled_absolute_error": abs(
            base_pooled - pilot_report["validation"]["base_pooled_rmse"]
        ),
        "overlay_pooled_absolute_error": abs(
            ungated_metrics["pooled_rmse"] - pilot_primary["pooled_rmse"]
        ),
    }
    checks = {
        "frozen_pilot_reproduced_to_5e_5": bool(max(reproduction.values()) <= 5e-5),
        "pooled_delta_at_most_minus_0_02": bool(gated_metrics["pooled_delta"] <= -0.02),
        "p95_not_worse_than_plus_0_10": bool(gated_metrics["p95_well_delta"] <= 0.10),
        "worst_not_worse_than_plus_0_50": bool(gated_metrics["worst_well_delta"] <= 0.50),
        "at_least_half_wells_nonpositive": bool(gated_metrics["nonpositive_well_fraction"] >= 0.50),
        "at_least_20pct_wells_selected": bool(
            selected_well_fraction >= MINIMUM_SELECTED_WELL_FRACTION
        ),
        "selected_well_win_fraction_at_least_55pct": bool(
            selected_win_fraction >= MINIMUM_SELECTED_WIN_FRACTION
        ),
        "every_spatial_fold_selects_a_well": bool(
            all(detail["selected_wells"] > 0 for detail in fold_details)
        ),
        "at_least_four_of_five_spatial_folds_improve": bool(improving_folds >= 4),
    }
    report = {
        "contract": {
            "writes_submission": False,
            "trains_sequence_model": False,
            "checkpoint_frozen": True,
            "overlay_scale": PRIMARY_SCALE,
            "gate_inputs_target_free": True,
            "absolute_coordinates_used_as_gate_features": False,
            "absolute_coordinates_used_only_for_outer_spatial_partition": True,
            "outer_protocol": "five_kmeans_spatial_subclusters_grouped_by_typewell_fingerprint",
            "outer_predictions_cross_fitted": True,
            "gate_model": {
                "standardization": True,
                "pca_components": PCA_COMPONENTS,
                "logistic_c": LOGISTIC_C,
                "class_weight": "balanced",
                "probability_threshold": PROBABILITY_THRESHOLD,
                "hyperparameters_tuned_on_validation_labels": False,
            },
        },
        "source": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "pilot_report_sha256": _sha256(pilot_report_path),
            "cache_protocol": PROTOCOL,
            "validation_fold": VALIDATION_FOLD,
        },
        "runtime": {
            "device": str(device),
            "seconds": float(time.perf_counter() - started),
        },
        "gate_features": {
            "count": len(summary_names or []),
            "names": summary_names,
        },
        "validation": {
            "wells": int(len(frame)),
            "rows": int(frame["rows"].sum()),
            "base_pooled_rmse": base_pooled,
            "ungated_overlay": ungated_metrics,
            "gated_overlay": gated_metrics,
            "selected_wells": int(np.sum(selected)),
            "selected_well_fraction": selected_well_fraction,
            "selected_win_fraction": selected_win_fraction,
            "outer_folds_improving": improving_folds,
            "outer_folds": fold_details,
            "pilot_reproduction": reproduction,
        },
        "decision": {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "full_multifold_authorized": all(checks.values()),
            "kaggle_submission_authorized": False,
        },
    }
    report_path = output_dir / "sequence_tcn_well_gate_report.json"
    detail_path = output_dir / "sequence_tcn_well_gate_per_well.csv"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame.to_csv(detail_path, index=False)
    print(
        f"decision={report['decision']['decision']} "
        f"selected={int(np.sum(selected))}/{len(selected)} "
        f"selected_win={selected_win_fraction:.3f} "
        f"gated_delta={gated_metrics['pooled_delta']:+.6f}",
        flush=True,
    )
    print(f"wrote {report_path} and {detail_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
