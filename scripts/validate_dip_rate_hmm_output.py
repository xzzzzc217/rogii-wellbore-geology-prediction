"""Validate a completed dip-rate HMM Kaggle Notebook output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
REQUIRED_FILES = (
    "submission.csv",
    "submission_before_dip_rate_hmm.csv",
    "dip_rate_hmm_per_well.csv",
    "dip_rate_hmm_deploy_report.json",
    "submission_audit.json",
)
REQUIRED_BOOLEAN_FLAGS = {
    "target_free_runtime_inputs_only": True,
    "training_or_hidden_truth_used_for_prediction": False,
    "well_id_encoded_as_feature": False,
    "absolute_coordinates_used": False,
    "fixed_test_well_used": False,
    "fixed_visible_rows_used": False,
    "expected_visible_hash_used": False,
    "leaderboard_score_used_for_selection": False,
    "probe_flags_used": False,
    "same_well_or_contact_override_in_overlay": False,
    "meta_learner_in_overlay": False,
    "parameter_grid_searched": False,
    "high_risk_candidate": True,
}
EXPECTED_CONFIRMATION = {
    "wells": 741,
    "pooled_delta": -0.15675641408341612,
    "spatial_improved_folds": 7,
    "spatial_fold_count": 7,
    "typewell_improved_folds": 5,
    "typewell_fold_count": 5,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_submission(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": "string"})
    if frame.columns.tolist() != ["id", "tvt"]:
        raise ValueError(f"{label}: expected exact id,tvt columns")
    if not frame["id"].is_unique:
        raise ValueError(f"{label}: duplicate IDs")
    values = pd.to_numeric(frame["tvt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: non-finite predictions")
    frame["tvt"] = values
    return frame


def _close(actual: float, expected: float, label: str, tolerance: float = 1e-8) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


def validate(run_dir: Path, sample_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing HMM output files: {missing}")

    final_path = run_dir / "submission.csv"
    before_path = run_dir / "submission_before_dip_rate_hmm.csv"
    final = _load_submission(final_path, "final")
    before = _load_submission(before_path, "before")
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    if sample.columns.tolist() != ["id", "tvt"]:
        raise ValueError("sample: expected exact id,tvt columns")
    if len(final) != len(sample) or len(before) != len(sample):
        raise ValueError("HMM output row count differs from official sample")
    sample_ids = sample["id"].astype(str)
    if not final["id"].astype(str).equals(sample_ids):
        raise ValueError("final ID order differs from official sample")
    if not before["id"].astype(str).equals(sample_ids):
        raise ValueError("before ID order differs from official sample")

    report = json.loads(
        (run_dir / "dip_rate_hmm_deploy_report.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (run_dir / "submission_audit.json").read_text(encoding="utf-8")
    )
    per_well = pd.read_csv(run_dir / "dip_rate_hmm_per_well.csv")
    final_sha = _sha256(final_path)
    before_sha = _sha256(before_path)
    if report.get("submission_sha256") != final_sha:
        raise ValueError("HMM report final SHA mismatch")
    if report.get("before_submission_sha256") != before_sha:
        raise ValueError("HMM report before SHA mismatch")
    if audit.get("sha256_submission_csv") != final_sha:
        raise ValueError("parent final audit SHA mismatch")
    if int(report.get("rows", -1)) != len(final):
        raise ValueError("HMM report row count mismatch")
    if int(report.get("test_wells", -1)) != len(per_well):
        raise ValueError("HMM report test-well count mismatch")

    for key, expected in REQUIRED_BOOLEAN_FLAGS.items():
        if report.get(key) is not expected:
            raise ValueError(f"HMM contract flag mismatch: {key}")
    if report.get("confirmation_decision") != "FAIL_tail_gates":
        raise ValueError("HMM confirmation risk disclosure mismatch")
    confirmation = report.get("confirmation", {})
    for key, expected in EXPECTED_CONFIRMATION.items():
        actual = confirmation.get(key)
        if isinstance(expected, float):
            _close(float(actual), expected, f"confirmation.{key}", 1e-12)
        elif actual != expected:
            raise ValueError(f"confirmation.{key}: {actual} != {expected}")

    delta = final["tvt"].to_numpy(dtype=float) - before["tvt"].to_numpy(dtype=float)
    max_movement = float(np.max(np.abs(delta)))
    if max_movement > 1.0000001:
        raise ValueError(f"HMM movement cap exceeded: {max_movement}")
    changed_rows = int((np.abs(delta) > 1e-12).sum())
    if int(report.get("changed_rows", -1)) != changed_rows:
        raise ValueError("HMM report changed-row count mismatch")
    _close(float(report["correction_mean"]), float(np.mean(delta)), "correction_mean")
    _close(
        float(report["correction_rms"]),
        float(np.sqrt(np.mean(np.square(delta)))),
        "correction_rms",
    )
    _close(float(report["correction_minimum"]), float(np.min(delta)), "correction_minimum")
    _close(float(report["correction_maximum"]), float(np.max(delta)), "correction_maximum")

    detail = pd.DataFrame(
        {
            "well": final["id"].astype(str).str.rsplit("_", n=1).str[0],
            "delta": delta,
        }
    )
    calculated = (
        detail.groupby("well", sort=True)["delta"]
        .agg(
            rows="size",
            correction_mean="mean",
            correction_minimum="min",
            correction_maximum="max",
        )
        .reset_index()
    )
    rms = (
        detail.assign(square=np.square(detail["delta"]))
        .groupby("well", sort=True)["square"]
        .mean()
        .pow(0.5)
        .rename("correction_rms")
        .reset_index()
    )
    calculated = calculated.merge(rms, on="well", validate="one_to_one")
    required_columns = {
        "well",
        "rows",
        "correction_mean",
        "correction_rms",
        "correction_minimum",
        "correction_maximum",
        "hmm_uncertainty_mean",
    }
    if not required_columns.issubset(per_well.columns):
        raise ValueError("HMM per-well report columns are incomplete")
    observed = per_well.sort_values("well").reset_index(drop=True)
    calculated = calculated.sort_values("well").reset_index(drop=True)
    if observed["well"].astype(str).tolist() != calculated["well"].tolist():
        raise ValueError("HMM per-well identifiers mismatch final submission")
    if observed["rows"].astype(int).tolist() != calculated["rows"].astype(int).tolist():
        raise ValueError("HMM per-well row counts mismatch final submission")
    for column in (
        "correction_mean",
        "correction_rms",
        "correction_minimum",
        "correction_maximum",
    ):
        if not np.allclose(
            observed[column].to_numpy(dtype=float),
            calculated[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(f"HMM per-well {column} mismatch")
    if not np.isfinite(observed["hmm_uncertainty_mean"].to_numpy(dtype=float)).all():
        raise ValueError("HMM per-well uncertainty contains non-finite values")

    return {
        "decision": "PASS",
        "run_dir": str(run_dir),
        "rows": int(len(final)),
        "test_wells": int(len(per_well)),
        "changed_rows": changed_rows,
        "maximum_abs_movement": max_movement,
        "before_sha256": before_sha,
        "submission_sha256": final_sha,
        "confirmation_decision_disclosed": report["confirmation_decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.run_dir, args.sample.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
