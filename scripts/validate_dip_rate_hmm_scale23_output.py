"""Validate a future hidden output from the conditional scale2.3 HMM notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_dip_rate_hmm_output import DEFAULT_SAMPLE, validate

EXPECTED_CONFIRMATION = {
    "wells": 738,
    "pooled_delta": -0.256847265382099,
    "spatial_improved_folds": 7,
    "spatial_fold_count": 7,
    "typewell_improved_folds": 5,
    "typewell_fold_count": 5,
}


def validate_scale23(run_dir: Path, sample_path: Path) -> dict[str, object]:
    result = validate(
        run_dir,
        sample_path,
        maximum_abs_movement=2.3,
        confirmation_decision="PASS_scale23_pooled_all_folds_rank_gate_tail_risk",
        expected_confirmation=EXPECTED_CONFIRMATION,
    )
    report = json.loads(
        (run_dir / "dip_rate_hmm_deploy_report.json").read_text(encoding="utf-8")
    )
    required = {
        "high_risk_candidate": True,
        "scale23_rank_gate_passed": True,
        "confirmation_test_wells_excluded": True,
    }
    for key, expected in required.items():
        if report.get(key) is not expected:
            raise ValueError(f"scale2.3 report flag mismatch: {key}")
    if float(report.get("true_p95_quantile", -1.0)) != 0.95:
        raise ValueError("scale2.3 report true p95 contract mismatch")
    parameters = report.get("parameters", {})
    if float(parameters.get("overlay_scale", -1.0)) != 0.46:
        raise ValueError("scale2.3 overlay scale mismatch")
    if float(parameters.get("maximum_final_movement", -1.0)) != 2.3:
        raise ValueError("scale2.3 maximum movement contract mismatch")
    result["scale23_contract"] = "PASS"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    result = validate_scale23(args.run_dir.resolve(), args.sample.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
