from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "kaggle_notebooks" / "functional_ridge_cap40_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-functional-ridge-cap40-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_combined_builder_is_prepared_but_not_authorized() -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_functional_ridge_cap40_v1.py"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    branch_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_parent_index") == 53
    )
    branch_code = _source(branch_cell)
    assert branch_code.count("_BH_STRENGTH = 0.62") == 1
    assert branch_code.count("_BH_CAP = 4.00") == 1
    assert "_BH_STRENGTH = 0.60" not in branch_code
    assert "_BH_CAP = 2.00" not in branch_code

    contract = next(
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_cap40_combined_contract")
    )
    contract_code = _source(contract)
    assert "candidate_selection_public_score_informed" in contract_code
    assert "runtime_reads_leaderboard_score" in contract_code
    assert "runtime_uses_fixed_test_well" in contract_code
    assert "functional_ridge_cap40_combined_report.json" in contract_code
    build = notebook["metadata"]["functional_ridge_cap40_build"]
    assert build["authorization_status"] == "PREPARED_NOT_AUTHORIZED"
    assert build["candidate_selection_public_score_informed"] is True
    assert build["runtime_reads_leaderboard_score"] is False
    assert metadata["id"] == "heralces/rogii-functional-ridge-cap40-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False

    all_code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    prohibited = (
        "af43d97ab5b76f7f0bbbc5c72fb6e9a829e939c48582fc4130144340547360a2",
        "_EX_EXPECTED_WELL",
        "_EX_EXPECTED_ROWS",
        "_EX_EXPECTED_BASE_SHA",
    )
    assert not [fragment for fragment in prohibited if fragment in all_code]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
