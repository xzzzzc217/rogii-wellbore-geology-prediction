from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_dip_rate_hmm_output import (  # noqa: E402
    EXPECTED_CONFIRMATION,
    REQUIRED_BOOLEAN_FLAGS,
    validate,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, movement: float = 0.5) -> tuple[Path, Path]:
    ids = ["well-a_1", "well-a_2", "well-b_1", "well-b_2"]
    sample = pd.DataFrame({"id": ids, "tvt": [0.0] * 4})
    before = pd.DataFrame({"id": ids, "tvt": [10.0, 11.0, 20.0, 21.0]})
    final = before.copy()
    final["tvt"] = final["tvt"] + movement
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    before_path = tmp_path / "submission_before_dip_rate_hmm.csv"
    final_path = tmp_path / "submission.csv"
    before.to_csv(before_path, index=False)
    final.to_csv(final_path, index=False)
    per_well = pd.DataFrame(
        {
            "well": ["well-a", "well-b"],
            "rows": [2, 2],
            "correction_mean": [movement, movement],
            "correction_rms": [abs(movement), abs(movement)],
            "correction_minimum": [movement, movement],
            "correction_maximum": [movement, movement],
            "hmm_uncertainty_mean": [1.0, 2.0],
        }
    )
    per_well.to_csv(tmp_path / "dip_rate_hmm_per_well.csv", index=False)
    report = {
        **REQUIRED_BOOLEAN_FLAGS,
        "confirmation_decision": "FAIL_tail_gates",
        "confirmation": EXPECTED_CONFIRMATION,
        "rows": 4,
        "test_wells": 2,
        "changed_rows": 4,
        "correction_mean": movement,
        "correction_rms": abs(movement),
        "correction_minimum": movement,
        "correction_maximum": movement,
        "before_submission_sha256": _sha(before_path),
        "submission_sha256": _sha(final_path),
    }
    (tmp_path / "dip_rate_hmm_deploy_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (tmp_path / "submission_audit.json").write_text(
        json.dumps({"sha256_submission_csv": _sha(final_path)}), encoding="utf-8"
    )
    return tmp_path, sample_path


def test_validator_accepts_consistent_output(tmp_path: Path) -> None:
    run_dir, sample_path = _fixture(tmp_path)
    result = validate(run_dir, sample_path)
    assert result["decision"] == "PASS"
    assert result["rows"] == 4
    assert result["maximum_abs_movement"] == pytest.approx(0.5)


def test_validator_rejects_movement_above_cap(tmp_path: Path) -> None:
    run_dir, sample_path = _fixture(tmp_path, movement=1.1)
    with pytest.raises(ValueError, match="movement cap exceeded"):
        validate(run_dir, sample_path)
