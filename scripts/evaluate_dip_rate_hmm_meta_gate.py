"""Cross-fit a target-free per-well scale gate for the frozen dip-rate HMM overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "dip_rate_hmm_confirmation"
    / "dip_rate_hmm_confirmation_per_well.csv"
)
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_meta_gate_discovery"
SEED = 20260802

FEATURE_COLUMNS = (
    "log1p_rows",
    "correction_rms",
    "log1p_hmm_uncertainty_mean",
    "uncertainty_to_correction_ratio",
)
PROTOCOLS = {
    "spatial_cluster_7_seed_20260727": "fold_spatial_cluster_7_seed_20260727",
    "typewell_fingerprint": "fold_typewell_fingerprint",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_wells(sample_path: Path) -> set[str]:
    sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
    return set(sample["id"].astype(str).str.rsplit("_", n=1).str[0])


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log1p_rows"] = np.log1p(out["rows"].to_numpy(dtype=float))
    out["log1p_hmm_uncertainty_mean"] = np.log1p(
        out["hmm_uncertainty_mean"].to_numpy(dtype=float)
    )
    out["uncertainty_to_correction_ratio"] = (
        out["hmm_uncertainty_mean"].to_numpy(dtype=float)
        / np.maximum(out["correction_rms"].to_numpy(dtype=float), 1e-6)
    )
    correction_sse = (
        out["rows"].to_numpy(dtype=float)
        * np.square(out["correction_rms"].to_numpy(dtype=float))
    )
    cross = (
        out["candidate_sse"].to_numpy(dtype=float)
        - out["base_sse"].to_numpy(dtype=float)
        - correction_sse
    ) / 2.0
    out["correction_sse"] = correction_sse
    out["base_correction_cross"] = cross
    out["optimal_multiplier"] = np.clip(
        -cross / np.maximum(correction_sse, 1e-12), 0.0, 1.0
    )
    return out


def _model(name: str) -> Any:
    if name == "ridge_1":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "ridge_10":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest_leaf20":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=20,
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "extra_trees_leaf20":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=20,
            random_state=SEED,
            n_jobs=-1,
        )
    raise KeyError(name)


def _candidate_sse(frame: pd.DataFrame, multiplier: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(multiplier, dtype=float), 0.0, 1.0)
    return (
        frame["base_sse"].to_numpy(dtype=float)
        + 2.0 * alpha * frame["base_correction_cross"].to_numpy(dtype=float)
        + np.square(alpha) * frame["correction_sse"].to_numpy(dtype=float)
    )


def _crossfit_multiplier(frame: pd.DataFrame, fold_column: str, name: str) -> np.ndarray:
    if name == "constant_full":
        return np.ones(len(frame), dtype=float)
    if name == "constant_half":
        return np.full(len(frame), 0.5, dtype=float)
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    target = frame["optimal_multiplier"].to_numpy(dtype=float)
    result = np.full(len(frame), np.nan, dtype=float)
    folds = frame[fold_column].to_numpy(dtype=int)
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        estimator = _model(name)
        estimator.fit(features[train_mask], target[train_mask])
        result[valid_mask] = np.clip(estimator.predict(features[valid_mask]), 0.0, 1.0)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite cross-fit multiplier for {name}/{fold_column}")
    return result


def _summarize(
    frame: pd.DataFrame,
    fold_column: str,
    name: str,
    multiplier: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate_sse = _candidate_sse(frame, multiplier)
    rows = frame["rows"].to_numpy(dtype=float)
    base_sse = frame["base_sse"].to_numpy(dtype=float)
    base_rmse = np.sqrt(base_sse / rows)
    candidate_rmse = np.sqrt(candidate_sse / rows)
    well_delta = candidate_rmse - base_rmse
    folds = frame[fold_column].to_numpy(dtype=int)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        total_rows = float(rows[mask].sum())
        fold_base = float(np.sqrt(base_sse[mask].sum() / total_rows))
        fold_candidate = float(np.sqrt(candidate_sse[mask].sum() / total_rows))
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(mask.sum()),
                "rows": int(total_rows),
                "base_rmse": fold_base,
                "candidate_rmse": fold_candidate,
                "delta": fold_candidate - fold_base,
            }
        )
    total_rows = float(rows.sum())
    pooled_base = float(np.sqrt(base_sse.sum() / total_rows))
    pooled_candidate = float(np.sqrt(candidate_sse.sum() / total_rows))
    detail = frame.loc[:, ["well", "rows", fold_column]].copy()
    detail["candidate"] = name
    detail["multiplier"] = multiplier
    detail["base_rmse"] = base_rmse
    detail["candidate_rmse"] = candidate_rmse
    detail["delta"] = well_delta
    summary = {
        "candidate": name,
        "base_rmse": pooled_base,
        "candidate_rmse": pooled_candidate,
        "pooled_delta": pooled_candidate - pooled_base,
        "fold_count": len(fold_rows),
        "improved_folds": int(sum(row["delta"] < 0.0 for row in fold_rows)),
        "well_win_fraction": float(np.mean(well_delta <= 0.0)),
        "p95_well_delta": float(np.quantile(well_delta, 0.95)),
        "worst_well_delta": float(np.max(well_delta)),
        "mean_multiplier": float(np.mean(multiplier)),
        "minimum_multiplier": float(np.min(multiplier)),
        "maximum_multiplier": float(np.max(multiplier)),
        "folds": fold_rows,
    }
    return summary, detail


def evaluate(input_path: Path, sample_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = pd.read_csv(input_path, dtype={"well": "string"})
    excluded_test_wells = sorted(_test_wells(sample_path))
    overlap = sorted(set(raw["well"].astype(str)) & set(excluded_test_wells))
    frame = raw.loc[~raw["well"].astype(str).isin(excluded_test_wells)].copy()
    frame = _prepare(frame).reset_index(drop=True)
    if not overlap:
        raise RuntimeError("expected current test wells to be explicitly excluded")
    if set(frame["well"].astype(str)) & set(excluded_test_wells):
        raise RuntimeError("test-well exclusion failed")

    names = (
        "constant_full",
        "constant_half",
        "ridge_1",
        "ridge_10",
        "random_forest_leaf20",
        "extra_trees_leaf20",
    )
    protocols: dict[str, Any] = {}
    detail_frames: list[pd.DataFrame] = []
    for protocol, fold_column in PROTOCOLS.items():
        summaries = []
        for name in names:
            multiplier = _crossfit_multiplier(frame, fold_column, name)
            summary, detail = _summarize(frame, fold_column, name, multiplier)
            summary["protocol"] = protocol
            summaries.append(summary)
            detail["protocol"] = protocol
            detail_frames.append(detail)
        protocols[protocol] = summaries

    spatial = protocols["spatial_cluster_7_seed_20260727"]
    eligible = [row for row in spatial if row["improved_folds"] == row["fold_count"]]
    selected = min(eligible, key=lambda row: row["candidate_rmse"])
    selected_name = str(selected["candidate"])
    typewell = next(
        row for row in protocols["typewell_fingerprint"] if row["candidate"] == selected_name
    )
    checks = {
        "spatial_all_folds_improve": selected["improved_folds"] == selected["fold_count"],
        "typewell_all_folds_improve": typewell["improved_folds"] == typewell["fold_count"],
        "typewell_pooled_delta_at_most_minus_0_05": typewell["pooled_delta"] <= -0.05,
        "typewell_p95_not_worse": typewell["p95_well_delta"] <= 0.0,
        "typewell_worst_not_worse_than_plus_0_25": typewell["worst_well_delta"] <= 0.25,
        "typewell_well_win_fraction_at_least_0_65": typewell["well_win_fraction"] >= 0.65,
    }
    report = {
        "candidate": "target-free dip-rate HMM learned per-well scale gate",
        "input_sha256": _sha256(input_path),
        "sample_sha256": _sha256(sample_path),
        "input_wells": int(len(raw)),
        "excluded_current_test_wells": overlap,
        "evaluation_wells": int(len(frame)),
        "features": list(FEATURE_COLUMNS),
        "target": "per-well SSE-optimal multiplier clipped to [0,1]",
        "test_well_targets_used": False,
        "well_id_feature_used": False,
        "absolute_coordinates_used": False,
        "leaderboard_score_used": False,
        "probe_flags_used": False,
        "protocols": protocols,
        "spatial_selected_candidate": selected_name,
        "confirmation_checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "writes_submission": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dip_rate_hmm_meta_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.concat(detail_frames, ignore_index=True).to_csv(
        output_dir / "dip_rate_hmm_meta_gate_per_well.csv", index=False
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.input.resolve(), args.sample.resolve(), args.output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
