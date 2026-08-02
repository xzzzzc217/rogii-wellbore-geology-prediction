"""Confirm a frozen 2x multiplier for the target-free dip-rate HMM correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "dip_rate_hmm_confirmation"
    / "dip_rate_hmm_confirmation_per_well.csv"
)
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_scale2_confirmation"
MULTIPLIER = 2.0
PROTOCOLS = {
    "spatial_cluster_7_seed_20260727": "fold_spatial_cluster_7_seed_20260727",
    "typewell_fingerprint": "fold_typewell_fingerprint",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_wells(sample_path: Path) -> set[str]:
    sample = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
    return set(sample["id"].astype(str).str.rsplit("_", n=1).str[0])


def _candidate_sse(frame: pd.DataFrame, multiplier: float) -> np.ndarray:
    correction_sse = (
        frame["rows"].to_numpy(dtype=float)
        * np.square(frame["correction_rms"].to_numpy(dtype=float))
    )
    cross = (
        frame["candidate_sse"].to_numpy(dtype=float)
        - frame["base_sse"].to_numpy(dtype=float)
        - correction_sse
    ) / 2.0
    return (
        frame["base_sse"].to_numpy(dtype=float)
        + 2.0 * multiplier * cross
        + multiplier**2 * correction_sse
    )


def _summary(frame: pd.DataFrame, fold_column: str, multiplier: float) -> dict[str, Any]:
    rows = frame["rows"].to_numpy(dtype=float)
    base_sse = frame["base_sse"].to_numpy(dtype=float)
    candidate_sse = _candidate_sse(frame, multiplier)
    if np.any(candidate_sse < -1e-8):
        raise RuntimeError("derived negative SSE")
    candidate_sse = np.maximum(candidate_sse, 0.0)
    base_rmse = np.sqrt(base_sse / rows)
    candidate_rmse = np.sqrt(candidate_sse / rows)
    well_delta = candidate_rmse - base_rmse
    folds = frame[fold_column].to_numpy(dtype=int)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_row_count = float(rows[mask].sum())
        fold_base = float(np.sqrt(base_sse[mask].sum() / fold_row_count))
        fold_candidate = float(np.sqrt(candidate_sse[mask].sum() / fold_row_count))
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(mask.sum()),
                "rows": int(fold_row_count),
                "base_rmse": fold_base,
                "candidate_rmse": fold_candidate,
                "delta": fold_candidate - fold_base,
            }
        )
    total_rows = float(rows.sum())
    pooled_base = float(np.sqrt(base_sse.sum() / total_rows))
    pooled_candidate = float(np.sqrt(candidate_sse.sum() / total_rows))
    return {
        "multiplier": multiplier,
        "base_rmse": pooled_base,
        "candidate_rmse": pooled_candidate,
        "pooled_delta": pooled_candidate - pooled_base,
        "fold_count": len(fold_rows),
        "improved_folds": int(sum(row["delta"] < 0.0 for row in fold_rows)),
        "well_win_fraction": float(np.mean(well_delta <= 0.0)),
        "p95_well_delta": float(np.quantile(well_delta, 0.95)),
        "worst_well_delta": float(np.max(well_delta)),
        "folds": fold_rows,
    }


def evaluate(input_path: Path, sample_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = pd.read_csv(input_path, dtype={"well": "string"})
    test_wells = _test_wells(sample_path)
    excluded = sorted(set(raw["well"].astype(str)) & test_wells)
    frame = raw.loc[~raw["well"].astype(str).isin(test_wells)].copy().reset_index(drop=True)
    if not excluded or set(frame["well"].astype(str)) & test_wells:
        raise RuntimeError("current test wells were not cleanly excluded")

    protocols: dict[str, Any] = {}
    for protocol, fold_column in PROTOCOLS.items():
        scale1 = _summary(frame, fold_column, 1.0)
        scale2 = _summary(frame, fold_column, MULTIPLIER)
        scale2["improvement_over_scale1"] = (
            scale1["candidate_rmse"] - scale2["candidate_rmse"]
        )
        protocols[protocol] = {"scale1": scale1, "scale2": scale2}

    spatial = protocols["spatial_cluster_7_seed_20260727"]["scale2"]
    typewell = protocols["typewell_fingerprint"]["scale2"]
    checks = {
        "spatial_all_folds_improve_vs_base": spatial["improved_folds"]
        == spatial["fold_count"],
        "typewell_all_folds_improve_vs_base": typewell["improved_folds"]
        == typewell["fold_count"],
        "spatial_pooled_delta_at_most_minus_0_20": spatial["pooled_delta"] <= -0.20,
        "typewell_pooled_delta_at_most_minus_0_20": typewell["pooled_delta"] <= -0.20,
        "spatial_better_than_scale1_by_at_least_0_04": spatial[
            "improvement_over_scale1"
        ]
        >= 0.04,
        "typewell_better_than_scale1_by_at_least_0_04": typewell[
            "improvement_over_scale1"
        ]
        >= 0.04,
    }
    report = {
        "candidate": "target-free dip-rate HMM rank-seeking scale2 overlay",
        "input_sha256": _sha256(input_path),
        "sample_sha256": _sha256(sample_path),
        "input_wells": int(len(raw)),
        "excluded_current_test_wells": excluded,
        "evaluation_wells": int(len(frame)),
        "base_overlay_scale": 0.2,
        "correction_multiplier": MULTIPLIER,
        "effective_overlay_scale": 0.4,
        "maximum_final_movement_ft": 2.0,
        "test_well_targets_used": False,
        "well_id_feature_used": False,
        "absolute_coordinates_used": False,
        "leaderboard_score_used": False,
        "probe_flags_used": False,
        "true_p95_quantile": 0.95,
        "protocols": protocols,
        "gate_checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "writes_submission": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dip_rate_hmm_scale2_confirmation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
