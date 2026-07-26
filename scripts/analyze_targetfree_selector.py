"""Honest OOF evaluation of target-free per-well trajectory selectors.

The feature matrix is built from an explicit inference-time allow-list. TVT truth
and the per-method TVT errors are used only as supervised labels and for offline
scoring after each validation fold has been predicted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 20260724
N_FOLDS = 5

BASELINE_METHOD = "final"
ELIGIBLE_METHODS = ("final", "particle_filter", "tracker_blend")
ALL_METHODS = (
    "final",
    "particle_filter",
    "tracker_blend",
    "beam_viterbi",
    "hold_last",
)

# The decision gate is fixed before looking at selector results.
MIN_POOLED_IMPROVEMENT = 0.10
MIN_IMPROVED_FOLDS = 4
PRIMARY_FEATURE_SET = "core"
PRIMARY_MODEL = "ridge_delta"

CORE_FEATURES = (
    "centroid_x",
    "centroid_y",
    "x_span",
    "y_span",
    "xy_end_to_end",
    "xy_path_length_per_row",
    "log1p_row_count",
    "log1p_hidden_rows",
    "hidden_fraction",
    "typewell_tvt_span",
    "anchor_typewell_fraction",
    "calibration_correlation",
    "calibration_gain",
    "calibration_residual_scale",
    "calibration_trust",
    "tvt_slope_prior",
    "pf_posterior_std_median",
    "pf_posterior_std_p90",
    "pf_log_evidence_mean",
    "pf_log_evidence_range",
    "pf_resamples_per_hidden_row",
    "pf_effective_seed_count",
    "beam_average_cost",
    "beam_terminal_margin",
    "predicted_final_tvt_span",
)

EXPANDED_GR_FEATURES = (
    "hidden_gr_comparable_fraction",
    "log1p_hidden_gr_error_final",
    "log1p_hidden_gr_error_particle_filter",
    "log1p_hidden_gr_error_tracker_blend",
    "log1p_hidden_gr_error_beam_viterbi",
    "log1p_hidden_gr_error_hold_last",
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "ridge_delta": {
        "alpha": 20.0,
        "fit_intercept": True,
        "target": "signed_log1p(method_mse_minus_final_mse)",
    },
    "extra_trees_delta": {
        "n_estimators": 256,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "max_features": 0.75,
        "bootstrap": False,
        "random_state": RANDOM_SEED,
        "target": "signed_log1p(method_mse_minus_final_mse)",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed target-free adaptive selectors with honest OOF folds."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "target_free_eval_773.json",
        help="Full target-free per-well evaluation JSON.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
        help="Competition root containing train horizontal-well files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "targetfree_selector_cv.json",
        help="Output JSON artifact.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "targetfree_selector_cv.md",
        help="Output Markdown report.",
    )
    return parser


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _finite_array(value: Any) -> np.ndarray:
    if not isinstance(value, list):
        return np.asarray([], dtype=float)
    array = np.asarray([_finite_float(item) for item in value], dtype=float)
    return array[np.isfinite(array)]


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_xy_geometry(train_dir: Path, well_ids: list[str]) -> dict[str, dict[str, float]]:
    """Load only X/Y columns and return inference-time well geometry."""

    geometry: dict[str, dict[str, float]] = {}
    for well_id in well_ids:
        path = train_dir / f"{well_id}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["X", "Y"])
        x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            raise ValueError(f"{well_id}: no finite X/Y rows")
        xv = x[valid]
        yv = y[valid]
        dx = np.diff(xv)
        dy = np.diff(yv)
        geometry[well_id] = {
            "centroid_x": float(np.mean(xv)),
            "centroid_y": float(np.mean(yv)),
            "x_span": float(np.ptp(xv)),
            "y_span": float(np.ptp(yv)),
            "xy_end_to_end": float(np.hypot(xv[-1] - xv[0], yv[-1] - yv[0])),
            "xy_path_length_per_row": float(
                np.sum(np.hypot(dx, dy)) / max(len(xv) - 1, 1)
            ),
        }
    return geometry


def extract_features(
    record: dict[str, Any],
    geometry: dict[str, float],
    *,
    include_hidden_gr: bool,
) -> dict[str, float]:
    """Extract an explicit target-free feature allow-list from one well.

    Deliberately does not inspect ``record["rmse"]``. The hidden-GR fields compare
    observed horizontal-well GR against typewell GR and are not TVT truth errors.
    """

    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{record.get('well_id')}: missing diagnostics")
    if diagnostics.get("target_free") is not True:
        raise ValueError(f"{record.get('well_id')}: diagnostics are not target-free")
    forbidden_flags = (
        "formation_columns_used",
        "leaderboard_bias_used",
        "precomputed_predictions_used",
        "same_id_lookup",
        "train_target_used",
    )
    if any(bool(diagnostics.get(name)) for name in forbidden_flags):
        raise ValueError(f"{record.get('well_id')}: unsafe diagnostic flag is active")

    row_count = _finite_float(diagnostics.get("row_count"))
    hidden_rows = _finite_float(diagnostics.get("hidden_rows"))
    typewell_min = _finite_float(diagnostics.get("typewell_tvt_min"))
    typewell_max = _finite_float(diagnostics.get("typewell_tvt_max"))
    typewell_span = typewell_max - typewell_min
    anchor_tvt = _finite_float(diagnostics.get("anchor_tvt"))
    final_min = _finite_float(diagnostics.get("final_tvt_min"))
    final_max = _finite_float(diagnostics.get("final_tvt_max"))

    calibration = diagnostics.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {}
    particle_filter = diagnostics.get("particle_filter")
    if not isinstance(particle_filter, dict):
        particle_filter = {}
    beam = diagnostics.get("beam_viterbi")
    if not isinstance(beam, dict):
        beam = {}

    evidence = _finite_array(particle_filter.get("average_log_evidence"))
    resamples = _finite_array(particle_filter.get("resamples"))
    seed_weights = _finite_array(particle_filter.get("seed_weights"))
    seed_square_sum = float(np.sum(seed_weights**2)) if seed_weights.size else math.nan

    features = {
        **{name: _finite_float(geometry.get(name)) for name in geometry},
        "log1p_row_count": float(np.log1p(max(row_count, 0.0))),
        "log1p_hidden_rows": float(np.log1p(max(hidden_rows, 0.0))),
        "hidden_fraction": hidden_rows / row_count if row_count > 0 else math.nan,
        "typewell_tvt_span": typewell_span,
        "anchor_typewell_fraction": (
            (anchor_tvt - typewell_min) / typewell_span if typewell_span > 0 else math.nan
        ),
        "calibration_correlation": _finite_float(calibration.get("correlation")),
        "calibration_gain": _finite_float(calibration.get("gain")),
        "calibration_residual_scale": _finite_float(calibration.get("residual_scale")),
        "calibration_trust": _finite_float(diagnostics.get("calibration_trust")),
        "tvt_slope_prior": _finite_float(particle_filter.get("tvt_slope_prior")),
        "pf_posterior_std_median": _finite_float(
            particle_filter.get("posterior_std_median")
        ),
        "pf_posterior_std_p90": _finite_float(particle_filter.get("posterior_std_p90")),
        "pf_log_evidence_mean": (
            float(np.mean(evidence)) if evidence.size else math.nan
        ),
        "pf_log_evidence_range": (
            float(np.ptp(evidence)) if evidence.size else math.nan
        ),
        "pf_resamples_per_hidden_row": (
            float(np.sum(resamples)) / hidden_rows
            if resamples.size and hidden_rows > 0
            else math.nan
        ),
        "pf_effective_seed_count": (
            1.0 / seed_square_sum if seed_square_sum > 0 else math.nan
        ),
        "beam_average_cost": _finite_float(beam.get("average_cost")),
        "beam_terminal_margin": _finite_float(beam.get("terminal_margin")),
        "predicted_final_tvt_span": final_max - final_min,
    }

    if include_hidden_gr:
        comparable_rows = _finite_float(diagnostics.get("hidden_gr_comparable_rows"))
        features["hidden_gr_comparable_fraction"] = (
            comparable_rows / hidden_rows if hidden_rows > 0 else math.nan
        )
        gr_errors = diagnostics.get("hidden_gr_rmse_by_component")
        if not isinstance(gr_errors, dict):
            gr_errors = {}
        for method in ALL_METHODS:
            value = _finite_float(gr_errors.get(method))
            features[f"log1p_hidden_gr_error_{method}"] = (
                float(np.log1p(max(value, 0.0))) if math.isfinite(value) else math.nan
            )

    expected = set(CORE_FEATURES)
    if include_hidden_gr:
        expected.update(EXPANDED_GR_FEATURES)
    if set(features) != expected:
        missing = sorted(expected - set(features))
        extra = sorted(set(features) - expected)
        raise AssertionError(f"feature schema mismatch; missing={missing}, extra={extra}")
    return features


def _make_model(model_name: str) -> Pipeline:
    if model_name == "ridge_delta":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=20.0, fit_intercept=True)),
            ]
        )
    if model_name == "extra_trees_delta":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=256,
                        max_depth=5,
                        min_samples_leaf=20,
                        max_features=0.75,
                        bootstrap=False,
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise KeyError(model_name)


def make_protocol_folds(
    centroids: np.ndarray,
    *,
    random_seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build fixed shuffled and label-free spatial-cluster folds."""

    indices = np.arange(len(centroids))
    random_folds = [
        (train, validation)
        for train, validation in KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        ).split(indices)
    ]

    scaled = StandardScaler().fit_transform(centroids)
    raw_labels = KMeans(
        n_clusters=n_folds,
        random_state=random_seed,
        n_init=32,
    ).fit_predict(scaled)
    cluster_order = sorted(
        range(n_folds),
        key=lambda cluster: tuple(np.mean(centroids[raw_labels == cluster], axis=0)),
    )
    remap = {cluster: fold for fold, cluster in enumerate(cluster_order)}
    spatial_labels = np.asarray([remap[label] for label in raw_labels], dtype=int)
    spatial_folds = [
        (indices[spatial_labels != fold], indices[spatial_labels == fold])
        for fold in range(n_folds)
    ]
    return {
        "shuffled_well": random_folds,
        "spatial_cluster": spatial_folds,
    }


def _pooled_rmse(
    method_mse: np.ndarray,
    scored_rows: np.ndarray,
    indices: np.ndarray,
    choices: np.ndarray,
) -> float:
    selected_mse = method_mse[indices, choices]
    return float(np.sqrt(np.average(selected_mse, weights=scored_rows[indices])))


def _choice_counts(choices: np.ndarray, methods: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(methods[int(choice)] for choice in choices)
    return {method: int(counts.get(method, 0)) for method in methods}


def evaluate_configuration(
    x: np.ndarray,
    method_mse: np.ndarray,
    scored_rows: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate one fixed selector with fold-local preprocessing and fitting."""

    transformed_deltas = _signed_log1p(method_mse[:, 1:] - method_mse[:, [0]])
    oof_choices = np.zeros(len(x), dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (train, validation) in enumerate(folds):
        model = _make_model(model_name)
        train_weight = scored_rows[train] / float(np.mean(scored_rows[train]))
        model.fit(
            x[train],
            transformed_deltas[train],
            model__sample_weight=train_weight,
        )
        predicted_delta = np.asarray(model.predict(x[validation]), dtype=float)
        if predicted_delta.ndim == 1:
            predicted_delta = predicted_delta[:, None]
        scores = np.column_stack([np.zeros(len(validation)), predicted_delta])
        choices = np.argmin(scores, axis=1)
        oof_choices[validation] = choices

        baseline_choices = np.zeros(len(validation), dtype=int)
        baseline = _pooled_rmse(
            method_mse,
            scored_rows,
            validation,
            baseline_choices,
        )
        selected = _pooled_rmse(
            method_mse,
            scored_rows,
            validation,
            choices,
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "train_wells": int(len(train)),
                "validation_wells": int(len(validation)),
                "validation_rows": int(np.sum(scored_rows[validation])),
                "baseline_pooled_rmse": baseline,
                "selector_pooled_rmse": selected,
                "delta_rmse": selected - baseline,
                "choice_counts": _choice_counts(choices, ELIGIBLE_METHODS),
            }
        )

    all_indices = np.arange(len(x))
    baseline_choices = np.zeros(len(x), dtype=int)
    baseline = _pooled_rmse(
        method_mse,
        scored_rows,
        all_indices,
        baseline_choices,
    )
    selected = _pooled_rmse(
        method_mse,
        scored_rows,
        all_indices,
        oof_choices,
    )
    improved_folds = sum(row["delta_rmse"] < 0 for row in fold_rows)
    return {
        "baseline_pooled_rmse": baseline,
        "selector_pooled_rmse": selected,
        "delta_rmse": selected - baseline,
        "relative_delta_percent": 100.0 * (selected - baseline) / baseline,
        "improved_folds": int(improved_folds),
        "total_folds": len(fold_rows),
        "choice_counts": _choice_counts(oof_choices, ELIGIBLE_METHODS),
        "folds": fold_rows,
    }


def _oracle_summary(
    mse: np.ndarray,
    rows: np.ndarray,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    choices = np.argmin(mse, axis=1)
    indices = np.arange(len(mse))
    return {
        "pooled_rmse": _pooled_rmse(mse, rows, indices, choices),
        "choice_counts": _choice_counts(choices, methods),
    }


def _gate(protocols: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    passed = True
    for protocol_name, result in protocols.items():
        pooled_improvement = -float(result["delta_rmse"])
        protocol_pass = (
            pooled_improvement >= MIN_POOLED_IMPROVEMENT
            and int(result["improved_folds"]) >= MIN_IMPROVED_FOLDS
        )
        checks[protocol_name] = {
            "pooled_improvement": pooled_improvement,
            "required_pooled_improvement": MIN_POOLED_IMPROVEMENT,
            "improved_folds": int(result["improved_folds"]),
            "required_improved_folds": MIN_IMPROVED_FOLDS,
            "passed": protocol_pass,
        }
        passed = passed and protocol_pass
    return {"passed": passed, "protocol_checks": checks}


def analyze(payload: dict[str, Any], geometry: dict[str, dict[str, float]]) -> dict[str, Any]:
    records = payload.get("per_well")
    if not isinstance(records, list) or not records:
        raise ValueError("input JSON has no per_well records")
    records = sorted(records, key=lambda row: str(row["well_id"]))
    well_ids = [str(record["well_id"]) for record in records]
    if len(well_ids) != len(set(well_ids)):
        raise ValueError("well IDs are not unique")
    if set(well_ids) != set(geometry):
        raise ValueError("geometry well IDs do not match evaluator well IDs")

    rows = np.asarray([int(record["scored_rows"]) for record in records], dtype=float)
    rmse_all = np.asarray(
        [
            [_finite_float(record["rmse"].get(method)) for method in ALL_METHODS]
            for record in records
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(rmse_all)) or np.any(rmse_all < 0):
        raise ValueError("method RMSE label matrix is incomplete or invalid")
    mse_all = rmse_all**2
    eligible_indices = [ALL_METHODS.index(method) for method in ELIGIBLE_METHODS]
    eligible_mse = mse_all[:, eligible_indices]

    feature_matrices: dict[str, np.ndarray] = {}
    feature_names: dict[str, tuple[str, ...]] = {
        "core": CORE_FEATURES,
        "expanded_hidden_gr": CORE_FEATURES + EXPANDED_GR_FEATURES,
    }
    for feature_set, names in feature_names.items():
        include_hidden_gr = feature_set == "expanded_hidden_gr"
        extracted = [
            extract_features(
                record,
                geometry[str(record["well_id"])],
                include_hidden_gr=include_hidden_gr,
            )
            for record in records
        ]
        feature_matrices[feature_set] = np.asarray(
            [[row[name] for name in names] for row in extracted],
            dtype=float,
        )

    centroids = np.asarray(
        [
            [geometry[well_id]["centroid_x"], geometry[well_id]["centroid_y"]]
            for well_id in well_ids
        ],
        dtype=float,
    )
    protocol_folds = make_protocol_folds(centroids)

    configurations: list[dict[str, Any]] = []
    for feature_set, x in feature_matrices.items():
        for model_name in MODEL_SPECS:
            protocol_results = {
                protocol_name: evaluate_configuration(
                    x,
                    eligible_mse,
                    rows,
                    folds,
                    model_name=model_name,
                )
                for protocol_name, folds in protocol_folds.items()
            }
            configurations.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "is_primary": (
                        feature_set == PRIMARY_FEATURE_SET and model_name == PRIMARY_MODEL
                    ),
                    "protocols": protocol_results,
                    "gate": _gate(protocol_results),
                }
            )

    primary = next(item for item in configurations if item["is_primary"])
    baseline = float(
        np.sqrt(np.average(eligible_mse[:, 0], weights=rows))
    )
    return {
        "contract": {
            "random_seed": RANDOM_SEED,
            "n_folds": N_FOLDS,
            "baseline_method": BASELINE_METHOD,
            "eligible_selector_methods": list(ELIGIBLE_METHODS),
            "oracle_only_methods": [
                method for method in ALL_METHODS if method not in ELIGIBLE_METHODS
            ],
            "primary_selector": {
                "feature_set": PRIMARY_FEATURE_SET,
                "model": PRIMARY_MODEL,
            },
            "go_gate": {
                "minimum_pooled_rmse_improvement_each_protocol": (
                    MIN_POOLED_IMPROVEMENT
                ),
                "minimum_improved_folds_each_protocol": MIN_IMPROVED_FOLDS,
                "recommendation_uses_primary_only": True,
            },
            "feature_exclusions": [
                "TVT truth",
                "per-method TVT RMSE/SSE",
                "any hidden-target-derived quantity",
                "well ID encodings",
            ],
            "hidden_gr_note": (
                "Expanded features use observed hidden horizontal-well GR versus "
                "typewell GR diagnostics; they do not use hidden TVT."
            ),
            "folds": {
                "shuffled_well": (
                    "5-fold KFold, shuffled with fixed seed; splits are by well"
                ),
                "spatial_cluster": (
                    "5 fixed KMeans clusters of standardized X/Y centroids; each "
                    "cluster is held out in turn"
                ),
            },
        },
        "data": {
            "wells": len(well_ids),
            "scored_rows": int(np.sum(rows)),
            "well_ids_sha256": hashlib.sha256(
                "\n".join(well_ids).encode("utf-8")
            ).hexdigest(),
        },
        "feature_sets": {
            name: {
                "feature_count": len(features),
                "features": list(features),
            }
            for name, features in feature_names.items()
        },
        "models": MODEL_SPECS,
        "baseline": {
            "method": BASELINE_METHOD,
            "pooled_rmse": baseline,
        },
        "oracle_ceiling": {
            "eligible_methods": {
                "methods": list(ELIGIBLE_METHODS),
                **_oracle_summary(eligible_mse, rows, ELIGIBLE_METHODS),
            },
            "all_methods": {
                "methods": list(ALL_METHODS),
                **_oracle_summary(mse_all, rows, ALL_METHODS),
            },
        },
        "configurations": configurations,
        "recommendation": {
            "decision": "GO" if primary["gate"]["passed"] else "NO-GO",
            "basis": (
                "Only the predeclared core + ridge_delta primary selector controls "
                "the decision; alternative models/features are sensitivity checks."
            ),
            "primary_gate": primary["gate"],
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline"]["pooled_rmse"]
    eligible_oracle = result["oracle_ceiling"]["eligible_methods"]
    all_oracle = result["oracle_ceiling"]["all_methods"]
    decision = result["recommendation"]["decision"]
    lines = [
        "# Target-free adaptive selector: frozen OOF evaluation",
        "",
        "## Decision",
        "",
        f"**{decision}.** The deployment decision is controlled only by the "
        "predeclared `core + ridge_delta` selector. The ExtraTrees model and "
        "hidden-GR feature expansion are sensitivity checks, not a model search.",
        "",
        "The GO gate was frozen before evaluation: the same primary selector must "
        f"improve pooled RMSE by at least {MIN_POOLED_IMPROVEMENT:.2f} and improve "
        f"at least {MIN_IMPROVED_FOLDS}/{N_FOLDS} folds under both shuffled-well "
        "and spatial-cluster holdout.",
        "",
        "## Leakage contract",
        "",
        "- Features are an explicit allow-list of X/Y well geometry and target-free "
        "calibration, PF, beam, masking-horizon, and predicted-trajectory diagnostics.",
        "- The expanded feature set additionally uses observed hidden-well GR versus "
        "typewell GR diagnostics. GR is an inference-time input; hidden TVT is not used.",
        "- TVT truth and per-method TVT RMSE/SSE are labels and offline scoring values "
        "only. They are never present in the feature matrix.",
        "- Well IDs are joins only and are not encoded as features.",
        "- Every imputer, scaler, and regressor is fitted inside its training fold.",
        "",
        "## Fixed evaluation design",
        "",
        f"- Wells: {result['data']['wells']}; scored rows: "
        f"{result['data']['scored_rows']:,}.",
        "- Selector candidates: `final`, `particle_filter`, `tracker_blend`. "
        "`beam_viterbi` and `hold_last` are included only in the oracle ceiling "
        "because their frozen pooled baselines are much weaker.",
        "- Shuffled protocol: fixed-seed 5-fold well KFold.",
        "- Spatial protocol: fixed 5-cluster KMeans on standardized X/Y centroids, "
        "holding out one complete cluster at a time.",
        "- Models: Ridge(alpha=20) and ExtraTrees(256 trees, depth 5, leaf 20), "
        "with no tuned threshold or post-hoc abstention margin.",
        "",
        "## Baseline and oracle ceiling",
        "",
        "| Method | Pooled RMSE | Choice counts |",
        "|---|---:|---|",
        f"| Always final | {_fmt(baseline)} | final={result['data']['wells']} |",
        f"| Oracle over eligible 3 | {_fmt(eligible_oracle['pooled_rmse'])} | "
        f"`{eligible_oracle['choice_counts']}` |",
        f"| Oracle over all 5 | {_fmt(all_oracle['pooled_rmse'])} | "
        f"`{all_oracle['choice_counts']}` |",
        "",
        "Oracle rows use held-out TVT truth and show only the theoretical ceiling; "
        "they are not deployable selectors.",
        "",
        "## OOF summary",
        "",
        "| Feature set | Model | Protocol | Selector | Delta vs final | "
        "Improved folds | Choices (final / PF / tracker) | Gate |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for configuration in result["configurations"]:
        for protocol_name, protocol in configuration["protocols"].items():
            counts = protocol["choice_counts"]
            gate = configuration["gate"]["protocol_checks"][protocol_name]["passed"]
            lines.append(
                f"| {configuration['feature_set']} | {configuration['model']} | "
                f"{protocol_name} | {_fmt(protocol['selector_pooled_rmse'])} | "
                f"{protocol['delta_rmse']:+.4f} | "
                f"{protocol['improved_folds']}/{protocol['total_folds']} | "
                f"{counts['final']} / {counts['particle_filter']} / "
                f"{counts['tracker_blend']} | {'pass' if gate else 'fail'} |"
            )

    primary = next(item for item in result["configurations"] if item["is_primary"])
    lines.extend(
        [
            "",
            "## Primary selector per-fold results",
            "",
        ]
    )
    for protocol_name, protocol in primary["protocols"].items():
        lines.extend(
            [
                f"### {protocol_name}",
                "",
                "| Fold | Validation wells | Rows | Final | Selector | Delta | "
                "Choices (final / PF / tracker) |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for fold in protocol["folds"]:
            counts = fold["choice_counts"]
            lines.append(
                f"| {fold['fold']} | {fold['validation_wells']} | "
                f"{fold['validation_rows']:,} | "
                f"{_fmt(fold['baseline_pooled_rmse'])} | "
                f"{_fmt(fold['selector_pooled_rmse'])} | "
                f"{fold['delta_rmse']:+.4f} | "
                f"{counts['final']} / {counts['particle_filter']} / "
                f"{counts['tracker_blend']} |"
            )
        lines.append("")

    if decision == "GO":
        conclusion = (
            "The primary selector cleared the frozen materiality and consistency "
            "gate in both protocols. It is eligible for a separate implementation "
            "review; this analysis itself does not modify or submit a notebook."
        )
    else:
        conclusion = (
            "The primary selector did not clear the same frozen gate in both "
            "protocols. Do not deploy or submit an adaptive selector from this "
            "experiment, and do not retune thresholds on these 773 labels."
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            conclusion,
            "",
            "Reproduce with:",
            "",
            "```powershell",
            "uv run python scripts/analyze_targetfree_selector.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    input_path = args.input.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload.get("per_well")
    if not isinstance(records, list):
        raise ValueError("input JSON has no per_well list")
    well_ids = sorted(str(record["well_id"]) for record in records)
    geometry = load_xy_geometry(data_root / "train", well_ids)
    result = analyze(payload, geometry)
    result["provenance"] = {
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "geometry_source": str((data_root / "train").resolve()),
        "geometry_columns_loaded": ["X", "Y"],
        "geometry_sha256": hashlib.sha256(
            json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }

    output_path = args.output.expanduser().resolve()
    markdown_path = args.markdown_output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["recommendation"]["decision"],
                "baseline": result["baseline"]["pooled_rmse"],
                "output": str(output_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
