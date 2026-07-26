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

from rogii.data_audit import (  # noqa: E402
    audit_dataset,
    detect_evaluation_zone,
    discover_well_files,
    parse_submission_id,
    summarize_frame,
    write_json_report,
)


def _horizontal(well_id: str, *, train: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "MD": [1000.0, 1001.0, 1002.0, 1003.0],
            "X": [10.0, 11.0, 12.0, 13.0],
            "Y": [20.0, 20.5, 21.0, 21.5],
            "Z": [-5000.0, -5000.1, -5000.2, -5000.3],
            "GR": [80.0, None, 82.0, 83.0],
            "TVT_input": [10.0, 10.5, None, None],
        }
    )
    if train:
        frame["TVT"] = [10.0, 10.5, 11.0, 11.5]
    return frame


def _typewell() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TVT": [0.0, 1.0, 2.0],
            "GR": [70.0, 75.0, 80.0],
            "Geology": ["A", "A", "B"],
        }
    )


def _write_competition_tree(root: Path) -> tuple[str, str]:
    train_well = "abcdef01"
    test_well = "1234abcd"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    _horizontal(train_well, train=True).to_csv(
        root / "train" / f"{train_well}__horizontal_well.csv", index=False
    )
    _typewell().to_csv(root / "train" / f"{train_well}__typewell.csv", index=False)
    _horizontal(test_well, train=False).to_csv(
        root / "test" / f"{test_well}__horizontal_well.csv", index=False
    )
    _typewell().drop(columns="Geology").to_csv(
        root / "test" / f"{test_well}__typewell__Typewell20001.csv", index=False
    )
    pd.DataFrame({"id": [f"{test_well}_2", f"{test_well}_3"], "tvt": [0.0, 0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return train_well, test_well


def test_discovery_pairs_current_and_legacy_typewell_names(tmp_path: Path) -> None:
    train_well, test_well = _write_competition_tree(tmp_path)

    discovery = discover_well_files(tmp_path)

    assert len(discovery.wells) == 2
    by_key = {(well.split, well.well_id): well for well in discovery.wells}
    assert by_key[("train", train_well)].is_complete_pair
    assert by_key[("test", test_well)].is_complete_pair
    assert by_key[("test", test_well)].embedded_typewell_ids == ("Typewell20001",)
    assert not [issue for issue in discovery.issues if issue["severity"] == "error"]


def test_discovery_reports_orphans_and_ignored_csvs(tmp_path: Path) -> None:
    _write_competition_tree(tmp_path)
    orphan = "deadbeef"
    _typewell().to_csv(tmp_path / "train" / f"{orphan}__typewell.csv", index=False)
    pd.DataFrame({"other": [1]}).to_csv(tmp_path / "notes.csv", index=False)

    discovery = discover_well_files(tmp_path)

    codes = {issue["code"] for issue in discovery.issues}
    assert "horizontal_missing" in codes
    assert [path.name for path in discovery.ignored_csvs] == ["notes.csv"]


def test_evaluation_zone_and_frame_summary() -> None:
    frame = pd.DataFrame(
        {
            "MD": [1.0, 2.0, 3.0, 4.0],
            "TVT": [5.0, 6.0, 7.0, 8.0],
            "TVT_input": [5.0, 6.0, None, None],
            "GR": [10.0, None, float("inf"), 13.0],
        }
    )

    zone = detect_evaluation_zone(frame)
    summary = summarize_frame(frame)

    assert zone["source"] == "TVT_input_missing"
    assert zone["evaluation_row_count"] == 2
    assert zone["first_evaluation_position"] == 2
    assert zone["blocks"] == [{"start": 2, "end": 3}]
    assert zone["is_contiguous"] is True
    assert zone["is_suffix"] is True
    assert zone["tvt_available_in_evaluation_rows"] == 2
    assert zone["tvt_input_mismatch_rows"] == 0
    assert summary["missingness"]["GR"]["missing_count"] == 1
    assert summary["numeric_ranges"]["GR"]["finite_count"] == 2
    assert summary["numeric_ranges"]["GR"]["infinite_count"] == 1
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    ("raw", "well_id", "row_index"),
    [
        ("abcdef01_0", "abcdef01", 0),
        ("well_with_underscores_42", "well_with_underscores", 42),
    ],
)
def test_parse_submission_id(raw: str, well_id: str, row_index: int) -> None:
    parsed = parse_submission_id(raw)
    assert parsed.well_id == well_id
    assert parsed.row_index == row_index


@pytest.mark.parametrize("raw", [None, "", "abcdef01", "_2", "abcdef01_-1", " abc_1"])
def test_parse_submission_id_rejects_malformed_values(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_submission_id(raw)


def test_end_to_end_audit_writes_json_and_checks_id_coverage(tmp_path: Path) -> None:
    _, test_well = _write_competition_tree(tmp_path)
    output = tmp_path / "reports" / "audit.json"

    report = audit_dataset(tmp_path)
    written = write_json_report(report, output)
    loaded = json.loads(written.read_text(encoding="utf-8"))

    assert loaded["splits"]["train"]["well_count"] == 1
    assert loaded["splits"]["test"]["well_count"] == 1
    assert loaded["splits"]["train"]["aggregate"]["horizontal"]["row_count"] == 4
    assert loaded["splits"]["test"]["evaluation_zone"]["evaluation_row_count"] == 2
    assert loaded["sample_submission"]["id_checks"]["exact_evaluation_coverage"] is True
    assert loaded["sample_submission"]["id_checks"]["referenced_wells"] == [test_well]
    assert loaded["issue_summary"]["by_severity"].get("error", 0) == 0


def test_sample_id_audit_flags_unknown_and_missing_evaluation_rows(tmp_path: Path) -> None:
    _, test_well = _write_competition_tree(tmp_path)
    pd.DataFrame({"id": [f"{test_well}_2", "ffffffff_999"], "tvt": [0.0, 0.0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )

    report = audit_dataset(tmp_path)
    checks = report["sample_submission"]["id_checks"]
    codes = {issue["code"] for issue in report["issues"]}

    assert checks["unknown_well_count"] == 1
    assert checks["missing_expected_evaluation_id_count"] == 1
    assert checks["exact_evaluation_coverage"] is False
    assert "sample_ids_unknown_well" in codes
    assert "sample_ids_missing_evaluation_rows" in codes
