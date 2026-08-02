from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_pls_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-functional-ridge-pls-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_builder_installs_fixed_ridge_pls_ensemble_contract() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_functional_ridge_pls_v1.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    overlay = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_pls_overlay_code")
    )
    contract = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_pls_final_contract")
    )
    bootstrap = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_source_bootstrap")
    )
    assert "_FR_RIDGE_WEIGHT = 0.75" in overlay
    assert "_FR_PLS_WEIGHT = 0.25" in overlay
    assert "_FR_PLS_COMPONENTS = 8" in overlay
    assert "def _fr_augmented_signature" in overlay
    assert "_FrPLSRegression" in overlay
    assert "_fr_ridge_model.fit(_fr_signatures, _fr_curves)" in overlay
    assert "_fr_pls_model.fit(_fr_augmented_signatures, _fr_curves)" in overlay
    assert "_FR_RIDGE_WEIGHT * _fr_ridge_curve" in overlay
    assert "'weight_grid_searched': False" in overlay
    assert "'pooled_delta': -0.0352086425" in overlay
    assert "'pooled_delta': -0.0230585034" in overlay
    assert "functional_ridge_pls_ensemble_deploy_report.json" in overlay
    assert "functional_ridge_pls_ensemble_deploy_report.json" in contract
    assert "get('ridge_weight') != 0.75" in contract
    assert "get('pls_weight') != 0.25" in contract
    assert bootstrap.count("newline='\\n'") == 3
    assert "leaderboard_score_used_for_selection" in overlay
    assert "fixed_test_well_used" in overlay
    assert "expected_visible_hash_used" in overlay
    assert metadata["id"] == "heralces/rogii-functional-ridge-pls-v1"
    assert metadata["code_file"] == NOTEBOOK.name
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    build = notebook["metadata"]["functional_ridge_pls_build"]
    assert build["ridge_weight"] == 0.75
    assert build["pls_weight"] == 0.25
    assert build["weight_grid_searched"] is False

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
