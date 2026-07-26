from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_submission import main as audit_submission_main  # noqa: E402

from rogii.submission_guard import audit_submission, write_submission_report  # noqa: E402


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "id": [
                "aaaabbbb_2",
                "aaaabbbb_3",
                "aaaabbbb_4",
                "ccccdddd_5",
                "ccccdddd_6",
            ],
            "tvt": [0.0] * 5,
        }
    ).to_csv(path, index=False)
    return path


def _write_submission(path: Path, ids: list[object], tvt: list[object]) -> Path:
    pd.DataFrame({"id": ids, "tvt": tvt}).to_csv(path, index=False)
    return path


def _codes(report: dict[str, object], severity: str | None = None) -> set[str]:
    issues = report["issues"]
    assert isinstance(issues, list)
    return {
        str(issue["code"]) for issue in issues if severity is None or issue["severity"] == severity
    }


def test_valid_submission_has_exact_alignment_and_per_well_statistics(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_2", "aaaabbbb_3", "aaaabbbb_4", "ccccdddd_5", "ccccdddd_6"],
        [10_000.0, 10_001.0, 10_002.0, 11_000.0, 11_001.0],
    )

    report = audit_submission(candidate, sample_path)

    assert report["structural_valid"] is True
    assert report["issue_summary"]["by_severity"].get("error", 0) == 0
    assert report["structure"]["id_order_exact"] is True
    assert report["structure"]["all_tvt_values_finite_numeric"] is True
    assert report["coverage_by_well"]["aaaabbbb"]["coverage_exact"] is True
    assert report["coverage_by_well"]["aaaabbbb"]["submission_indices_contiguous"] is True
    stats = report["statistics"]["per_well"]["aaaabbbb"]
    assert stats["range"]["min"] == 10_000.0
    assert stats["range"]["max"] == 10_002.0
    assert stats["adjacent_jumps"]["max_absolute_jump"] == 1.0
    assert stats["is_constant_trajectory"] is False


@pytest.mark.parametrize(
    ("columns", "code"),
    [
        ({"tvt": [1.0], "id": ["aaaabbbb_2"]}, "submission_columns_mismatch"),
        (
            {"id": ["aaaabbbb_2"], "tvt": [1.0], "debug": [1]},
            "submission_columns_mismatch",
        ),
    ],
)
def test_columns_must_be_exact_and_ordered(
    tmp_path: Path,
    columns: dict[str, list[object]],
    code: str,
) -> None:
    sample = _write_submission(tmp_path / "sample.csv", ["aaaabbbb_2"], [0.0])
    candidate = tmp_path / "submission.csv"
    pd.DataFrame(columns).to_csv(candidate, index=False)

    report = audit_submission(candidate, sample)

    assert report["structural_valid"] is False
    assert code in _codes(report, "error")


def test_duplicate_missing_extra_and_ordered_ids_are_rejected(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_3", "aaaabbbb_2", "aaaabbbb_2", "ccccdddd_5", "ffffffff_9"],
        [10_001.0, 10_000.0, 10_000.0, 11_000.0, 12_000.0],
    )

    report = audit_submission(candidate, sample_path)
    codes = _codes(report, "error")

    assert report["structural_valid"] is False
    assert "submission_ids_not_unique" in codes
    assert "submission_id_coverage_mismatch" in codes
    assert "submission_id_order_mismatch" in codes
    assert "submission_well_coverage_mismatch" in codes
    assert report["structure"]["missing_expected_id_count"] == 2
    assert report["structure"]["extra_submission_id_count"] == 2
    assert report["structure"]["all_submission_ids_belong_to_sample"] is False


def test_row_count_malformed_id_and_non_contiguous_indices_are_errors(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_2", "aaaabbbb_4", "bad-id", "ccccdddd_5"],
        [10_000.0, 10_002.0, 10_003.0, 11_000.0],
    )

    report = audit_submission(candidate, sample_path)
    codes = _codes(report, "error")

    assert "submission_row_count_mismatch" in codes
    assert "submission_ids_malformed" in codes
    assert "submission_row_indices_not_contiguous" in codes
    assert report["coverage_by_well"]["aaaabbbb"]["submission_indices_contiguous"] is False


def test_non_numeric_nan_and_infinite_predictions_are_errors(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_2", "aaaabbbb_3", "aaaabbbb_4", "ccccdddd_5", "ccccdddd_6"],
        ["not-a-number", "", "inf", 11_000.0, 11_001.0],
    )

    report = audit_submission(candidate, sample_path)

    assert "submission_tvt_non_finite" in _codes(report, "error")
    assert report["structure"]["tvt_non_finite_count"] == 3
    assert report["structure"]["tvt_missing_or_non_numeric_count"] == 2
    assert report["structure"]["tvt_infinite_count"] == 1


def test_statistical_anomalies_warn_but_do_not_break_strict_structure(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_2", "aaaabbbb_3", "aaaabbbb_4", "ccccdddd_5", "ccccdddd_6"],
        [10_000.0, 10_100.0, 30_000.0, 7_000.0, 7_000.0],
    )

    report = audit_submission(candidate, sample_path, max_adjacent_jump=25.0)
    warnings = _codes(report, "warning")

    assert report["structural_valid"] is True
    assert "tvt_outside_plausible_range" in warnings
    assert "large_adjacent_tvt_jumps" in warnings
    assert "constant_well_trajectories" in warnings
    assert report["statistics"]["per_well"]["ccccdddd"]["is_constant_trajectory"] is True
    assert (
        report["statistics"]["per_well"]["aaaabbbb"]["adjacent_jumps"]["above_threshold_count"] == 2
    )


def test_sample_fallback_is_reported_for_overall_and_each_well(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_2", "aaaabbbb_3", "aaaabbbb_4", "ccccdddd_5", "ccccdddd_6"],
        [0.0] * 5,
    )

    report = audit_submission(candidate, sample_path)
    warnings = _codes(report, "warning")

    assert report["structural_valid"] is True
    assert "constant_submission" in warnings
    assert "constant_well_trajectories" in warnings
    assert "submission_matches_sample_fallback" in warnings
    assert "well_trajectories_match_sample_fallback" in warnings
    assert report["statistics"]["overall"]["fallback_vs_sample"]["matching_fraction"] == 1.0


def test_non_contiguous_sample_reference_is_rejected(tmp_path: Path) -> None:
    sample = _write_submission(
        tmp_path / "sample.csv",
        ["aaaabbbb_2", "aaaabbbb_4"],
        [0.0, 0.0],
    )
    candidate = _write_submission(
        tmp_path / "submission.csv",
        ["aaaabbbb_2", "aaaabbbb_4"],
        [10_000.0, 10_002.0],
    )

    report = audit_submission(candidate, sample)

    assert "sample_row_indices_not_contiguous" in _codes(report, "error")
    assert "submission_row_indices_not_contiguous" in _codes(report, "error")
    assert report["structural_valid"] is False


def test_report_writer_emits_strict_json_without_nan(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    empty = _write_submission(tmp_path / "submission.csv", [], [])
    report = audit_submission(empty, sample_path)
    output = write_submission_report(report, tmp_path / "nested" / "audit.json")

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["submission_path"] == str(empty.resolve())
    assert loaded["statistics"]["overall"]["range"]["min"] is None


def test_cli_strict_fails_only_for_structural_errors(
    tmp_path: Path,
    sample_path: Path,
) -> None:
    ids = ["aaaabbbb_2", "aaaabbbb_3", "aaaabbbb_4", "ccccdddd_5", "ccccdddd_6"]
    warning_only = _write_submission(tmp_path / "warning_only.csv", ids, [0.0] * 5)
    invalid = _write_submission(
        tmp_path / "invalid.csv",
        list(reversed(ids)),
        [10_000.0, 10_001.0, 10_002.0, 11_000.0, 11_001.0],
    )

    warning_exit = audit_submission_main(
        [
            str(warning_only),
            "--sample",
            str(sample_path),
            "--output",
            str(tmp_path / "warning.json"),
            "--strict",
        ]
    )
    error_exit = audit_submission_main(
        [
            str(invalid),
            "--sample",
            str(sample_path),
            "--output",
            str(tmp_path / "invalid.json"),
            "--strict",
        ]
    )

    assert warning_exit == 0
    assert error_exit == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tvt_min": 5.0, "tvt_max": 5.0}, "tvt_min"),
        ({"max_adjacent_jump": 0.0}, "max_adjacent_jump"),
        ({"fallback_fraction": 1.1}, "fallback_fraction"),
    ],
)
def test_invalid_guard_configuration_is_rejected(
    tmp_path: Path,
    sample_path: Path,
    kwargs: dict[str, float],
    message: str,
) -> None:
    candidate = _write_submission(tmp_path / "submission.csv", [], [])
    with pytest.raises(ValueError, match=message):
        audit_submission(candidate, sample_path, **kwargs)
