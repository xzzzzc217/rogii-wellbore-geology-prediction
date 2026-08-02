"""Confirm the preregistered 2.3x dip-rate HMM correction multiplier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from evaluate_dip_rate_hmm_scale2_confirmation import (
    PROTOCOLS,
    _summary,
    _test_wells,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "dip_rate_hmm_confirmation"
    / "dip_rate_hmm_confirmation_per_well.csv"
)
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dip_rate_hmm_scale23_confirmation"
MULTIPLIER = 2.3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(input_path: Path, sample_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = pd.read_csv(input_path, dtype={"well": "string"})
    test_wells = _test_wells(sample_path)
    excluded = sorted(set(raw["well"].astype(str)) & test_wells)
    frame = raw.loc[~raw["well"].astype(str).isin(test_wells)].copy().reset_index(drop=True)
    if not excluded or set(frame["well"].astype(str)) & test_wells:
        raise RuntimeError("current test wells were not cleanly excluded")

    protocols: dict[str, Any] = {}
    for protocol, fold_column in PROTOCOLS.items():
        scale2 = _summary(frame, fold_column, 2.0)
        scale23 = _summary(frame, fold_column, MULTIPLIER)
        scale23["improvement_over_scale2"] = (
            scale2["candidate_rmse"] - scale23["candidate_rmse"]
        )
        protocols[protocol] = {"scale2": scale2, "scale23": scale23}

    spatial = protocols["spatial_cluster_7_seed_20260727"]["scale23"]
    typewell = protocols["typewell_fingerprint"]["scale23"]
    checks = {
        "spatial_all_folds_improve_vs_base": spatial["improved_folds"]
        == spatial["fold_count"],
        "typewell_all_folds_improve_vs_base": typewell["improved_folds"]
        == typewell["fold_count"],
        "spatial_pooled_delta_at_most_minus_0_24": spatial["pooled_delta"] <= -0.24,
        "typewell_pooled_delta_at_most_minus_0_24": typewell["pooled_delta"] <= -0.24,
        "spatial_better_than_scale2_by_at_least_0_01": spatial[
            "improvement_over_scale2"
        ]
        >= 0.01,
        "typewell_better_than_scale2_by_at_least_0_01": typewell[
            "improvement_over_scale2"
        ]
        >= 0.01,
    }
    report = {
        "candidate": "target-free dip-rate HMM rank-seeking scale2.3 overlay",
        "input_sha256": _sha256(input_path),
        "sample_sha256": _sha256(sample_path),
        "input_wells": int(len(raw)),
        "excluded_current_test_wells": excluded,
        "evaluation_wells": int(len(frame)),
        "base_overlay_scale": 0.2,
        "correction_multiplier": MULTIPLIER,
        "effective_overlay_scale": 0.46,
        "maximum_final_movement_ft": 2.3,
        "test_well_targets_used": False,
        "well_id_feature_used": False,
        "absolute_coordinates_used": False,
        "leaderboard_score_used": False,
        "probe_flags_used": False,
        "true_p95_quantile": 0.95,
        "protocols": protocols,
        "gate_checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "execution_authorized": False,
        "writes_submission": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dip_rate_hmm_scale23_confirmation_report.json").write_text(
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
