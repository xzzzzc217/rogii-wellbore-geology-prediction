from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_pls_cap40_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-functional-ridge-pls-cap40-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_ridge_pls_cap40_builder_is_prepared_not_authorized() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_functional_ridge_pls_cap40_v1.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    branch = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_parent_index") == 53
    )
    contract = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get(
            "functional_ridge_pls_cap40_combined_contract"
        )
    )
    assert branch.count("_BH_STRENGTH = 0.62") == 1
    assert branch.count("_BH_CAP = 4.00") == 1
    assert "functional_ridge_pls_ensemble_deploy_report.json" in contract
    assert "functional_ridge_pls_cap40_combined_report.json" in contract
    assert "get('ridge_weight') != 0.75" in contract
    assert "get('pls_weight') != 0.25" in contract
    assert "runtime_reads_leaderboard_score" in contract
    build = notebook["metadata"]["functional_ridge_pls_cap40_build"]
    assert build["authorization_status"] == "PREPARED_NOT_AUTHORIZED"
    assert build["ridge_weight"] == 0.75
    assert build["pls_weight"] == 0.25
    assert build["weight_grid_searched"] is False
    assert metadata["id"] == "heralces/rogii-functional-ridge-pls-cap40-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False

    all_code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    prohibited = (
        "af43d97ab5b76f7f0bbbc5c72fb6e9a829e939c48582fc4130144340547360a2",
        "d4d7e368c940d84cdc1504fcad18809675f18499b0efd6e089011b21febb7da3",
        "_EX_EXPECTED_WELL",
        "_EX_EXPECTED_ROWS",
        "_EX_EXPECTED_BASE_SHA",
    )
    assert not [fragment for fragment in prohibited if fragment in all_code]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
