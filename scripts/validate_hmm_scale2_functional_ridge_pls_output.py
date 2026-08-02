"""Validate hidden output from the conditional HMM2 plus Ridge/PLS notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from validate_dip_rate_hmm_output import DEFAULT_SAMPLE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": "string"})
    if list(frame.columns) != ["id", "tvt"]:
        raise ValueError(f"unexpected submission schema: {path}")
    if not frame["id"].is_unique:
        raise ValueError(f"submission IDs are not unique: {path}")
    values = pd.to_numeric(frame["tvt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"submission contains non-finite predictions: {path}")
    return frame


def _values(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["tvt"], errors="raise").to_numpy(dtype=float)


def _require_flag(report: dict[str, Any], key: str, expected: bool) -> None:
    if report.get(key) is not expected:
        raise ValueError(f"report flag mismatch: {key}")


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-8):
        raise ValueError(f"numeric report mismatch: {label}")


def _validate_delta_report(
    delta: np.ndarray,
    report: dict[str, Any],
    *,
    prefix: str,
) -> None:
    calculated = {
        "correction_mean": float(np.mean(delta)),
        "correction_rms": float(np.sqrt(np.mean(delta**2))),
        "correction_minimum": float(np.min(delta)),
        "correction_maximum": float(np.max(delta)),
    }
    for key, expected in calculated.items():
        _close(float(report.get(key)), expected, f"{prefix}.{key}")


def _validate_per_well(
    ids: pd.Series,
    delta: np.ndarray,
    observed: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    detail = pd.DataFrame(
        {"well": ids.astype(str).str.rsplit("_", n=1).str[0], "delta": delta}
    )
    calculated = (
        detail.groupby("well", sort=True)["delta"]
        .agg(rows="size", mean="mean", minimum="min", maximum="max")
        .reset_index()
    )
    rms = (
        detail.assign(square=np.square(detail["delta"]))
        .groupby("well", sort=True)["square"]
        .mean()
        .pow(0.5)
        .rename("rms")
        .reset_index()
    )
    calculated = calculated.merge(rms, on="well", validate="one_to_one")
    observed = observed.sort_values("well").reset_index(drop=True)
    calculated = calculated.sort_values("well").reset_index(drop=True)
    if observed["well"].astype(str).tolist() != calculated["well"].tolist():
        raise ValueError(f"{prefix} per-well identifiers mismatch")
    if observed["rows"].astype(int).tolist() != calculated["rows"].tolist():
        raise ValueError(f"{prefix} per-well row counts mismatch")
    mapping = (
        {
            "correction_mean": "mean",
            "correction_rms": "rms",
            "correction_minimum": "minimum",
            "correction_maximum": "maximum",
        }
        if prefix == "HMM"
        else {
            "mean_correction": "mean",
            "rms_correction": "rms",
            "minimum_correction": "minimum",
            "maximum_correction": "maximum",
        }
    )
    for observed_column, calculated_column in mapping.items():
        if observed_column not in observed:
            raise ValueError(f"{prefix} per-well column missing: {observed_column}")
        if not np.allclose(
            observed[observed_column].to_numpy(dtype=float),
            calculated[calculated_column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(f"{prefix} per-well values mismatch: {observed_column}")


def validate(run_dir: Path, sample_path: Path) -> dict[str, object]:
    paths = {
        "final": run_dir / "submission.csv",
        "parent": run_dir / "submission_before_dip_rate_hmm.csv",
        "hmm": run_dir / "submission_after_dip_rate_hmm_scale2.csv",
        "functional_before": (
            run_dir / "submission_before_functional_ridge_pls_ensemble.csv"
        ),
        "hmm_report": run_dir / "dip_rate_hmm_deploy_report.json",
        "hmm_per_well": run_dir / "dip_rate_hmm_per_well.csv",
        "functional_report": (
            run_dir / "functional_ridge_pls_ensemble_deploy_report.json"
        ),
        "functional_per_well": (
            run_dir / "functional_ridge_pls_ensemble_per_well.csv"
        ),
        "audit": run_dir / "submission_audit.json",
        "combined_report": (
            run_dir / "hmm_scale2_functional_ridge_pls_combined_report.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing combined output files: {missing}")

    sample = _frame(sample_path)
    frames = {
        key: _frame(paths[key])
        for key in ("final", "parent", "hmm", "functional_before")
    }
    sample_ids = sample["id"].astype(str)
    for key, frame in frames.items():
        if len(frame) != len(sample):
            raise ValueError(f"row count differs from official sample: {key}")
        if not frame["id"].astype(str).equals(sample_ids):
            raise ValueError(f"ID order differs from official sample: {key}")

    parent_values = _values(frames["parent"])
    hmm_values = _values(frames["hmm"])
    functional_before_values = _values(frames["functional_before"])
    final_values = _values(frames["final"])
    if not np.array_equal(hmm_values, functional_before_values):
        raise ValueError("Functional overlay did not start from HMM intermediate")
    hmm_delta = hmm_values - parent_values
    functional_delta = final_values - functional_before_values
    if float(np.max(np.abs(hmm_delta))) > 2.0000001:
        raise ValueError("HMM scale2 movement exceeds 2 ft")

    hmm_report = json.loads(paths["hmm_report"].read_text(encoding="utf-8"))
    functional_report = json.loads(
        paths["functional_report"].read_text(encoding="utf-8")
    )
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    combined = json.loads(paths["combined_report"].read_text(encoding="utf-8"))
    hmm_sha = _sha256(paths["hmm"])
    parent_sha = _sha256(paths["parent"])
    functional_before_sha = _sha256(paths["functional_before"])
    final_sha = _sha256(paths["final"])
    expected_sha = {
        "hmm report parent": (hmm_report.get("before_submission_sha256"), parent_sha),
        "hmm report intermediate": (hmm_report.get("submission_sha256"), hmm_sha),
        "functional report before": (
            functional_report.get("before_submission_sha256"),
            functional_before_sha,
        ),
        "functional report final": (
            functional_report.get("submission_sha256"),
            final_sha,
        ),
        "audit final": (audit.get("sha256_submission_csv"), final_sha),
        "combined HMM intermediate": (
            combined.get("hmm_intermediate_sha256"),
            hmm_sha,
        ),
        "combined final": (combined.get("submission_sha256"), final_sha),
    }
    for label, (reported, observed) in expected_sha.items():
        if str(reported) != observed:
            raise ValueError(f"SHA mismatch: {label}")

    for key in (
        "fixed_test_well_used",
        "fixed_visible_rows_used",
        "expected_visible_hash_used",
        "leaderboard_score_used_for_selection",
        "probe_flags_used",
    ):
        _require_flag(hmm_report, key, False)
        _require_flag(functional_report, key, False)
    _require_flag(hmm_report, "confirmation_test_wells_excluded", True)
    _require_flag(hmm_report, "target_free_runtime_inputs_only", True)
    _require_flag(functional_report, "target_free_features_only", True)
    _require_flag(functional_report, "cache_training_rows_only", True)
    hmm_parameters = hmm_report.get("parameters", {})
    functional_parameters = functional_report.get("parameters", {})
    exact_parameters = {
        "HMM overlay scale": (hmm_parameters.get("overlay_scale"), 0.4),
        "HMM movement cap": (
            hmm_parameters.get("maximum_final_movement"),
            2.0,
        ),
        "Ridge weight": (functional_parameters.get("ridge_weight"), 0.75),
        "PLS weight": (functional_parameters.get("pls_weight"), 0.25),
        "Functional scale": (functional_parameters.get("overlay_scale"), 0.2),
    }
    for label, (observed, expected) in exact_parameters.items():
        if float(observed) != expected:
            raise ValueError(f"parameter mismatch: {label}")
    if functional_parameters.get("weight_grid_searched") is not False:
        raise ValueError("Functional component searched a weight grid")

    changed_hmm = int((np.abs(hmm_delta) > 1e-12).sum())
    changed_functional = int((np.abs(functional_delta) > 1e-12).sum())
    if changed_hmm != int(hmm_report.get("changed_rows", -1)):
        raise ValueError("HMM changed-row count mismatch")
    if changed_functional != int(functional_report.get("changed_rows", -1)):
        raise ValueError("Functional changed-row count mismatch")
    _validate_delta_report(hmm_delta, hmm_report, prefix="HMM")
    _validate_delta_report(functional_delta, functional_report, prefix="Functional")
    hmm_per_well = pd.read_csv(paths["hmm_per_well"])
    functional_per_well = pd.read_csv(paths["functional_per_well"])
    if int(hmm_per_well["rows"].sum()) != len(sample):
        raise ValueError("HMM per-well row total mismatch")
    if int(functional_per_well["rows"].sum()) != len(sample):
        raise ValueError("Functional per-well row total mismatch")
    _validate_per_well(
        frames["final"]["id"], hmm_delta, hmm_per_well, prefix="HMM"
    )
    _validate_per_well(
        frames["final"]["id"],
        functional_delta,
        functional_per_well,
        prefix="Functional",
    )

    if combined.get("authorization_status") != "PREPARED_NOT_AUTHORIZED":
        raise ValueError("combined authorization status mismatch")
    _require_flag(combined, "runtime_reads_leaderboard_score", False)
    _require_flag(combined, "runtime_uses_fixed_test_well", False)
    _require_flag(combined, "runtime_uses_fixed_visible_rows", False)
    _require_flag(combined, "runtime_uses_expected_visible_hash", False)
    _require_flag(combined, "probe_flags_used", False)
    oof = combined.get("preregistered_additive_oof", {})
    if int(oof.get("spatial_improved_folds_over_hmm_scale2", -1)) != 6:
        raise ValueError("combined spatial OOF fold count mismatch")
    if int(oof.get("typewell_improved_folds_over_hmm_scale2", -1)) != 5:
        raise ValueError("combined Typewell OOF fold count mismatch")
    if oof.get("tail_risk_disclosed") is not True:
        raise ValueError("combined tail-risk disclosure is missing")

    return {
        "status": "PASS",
        "rows": len(sample),
        "wells": int(len(hmm_per_well)),
        "hmm_changed_rows": changed_hmm,
        "hmm_max_abs_movement_ft": float(np.max(np.abs(hmm_delta))),
        "functional_changed_rows": changed_functional,
        "functional_correction_rms_ft": float(
            np.sqrt(np.mean(functional_delta**2))
        ),
        "hmm_intermediate_sha256": hmm_sha,
        "submission_sha256": final_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    result = validate(args.run_dir.resolve(), args.sample.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
