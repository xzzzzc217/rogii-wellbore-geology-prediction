from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "kaggle_notebooks" / "functional_ridge_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-functional-ridge-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_builder_installs_audited_overlay_after_branch() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_functional_ridge_v1.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    markers = [cell.get("metadata", {}) for cell in notebook["cells"]]
    branch_position = next(
        index
        for index, marker in enumerate(markers)
        if marker.get("functional_ridge_parent_index") == 53
    )
    overlay_position = next(
        index
        for index, marker in enumerate(markers)
        if marker.get("functional_ridge_overlay_code")
    )
    audit_position = next(
        index
        for index, marker in enumerate(markers)
        if marker.get("functional_ridge_parent_index") == 54
    )
    contract_position = next(
        index
        for index, marker in enumerate(markers)
        if marker.get("functional_ridge_final_contract")
    )

    assert branch_position < overlay_position < audit_position < contract_position
    overlay = _source(notebook["cells"][overlay_position])
    assert "_FR_OVERLAY_SCALE = 0.20" in overlay
    assert "_FR_RIDGE_ALPHA = 100.0" in overlay
    assert "target_free_features_only" in overlay
    assert "deployment_base_matches_oof_base_exactly" in overlay
    assert "leaderboard_score_used_for_selection" in overlay
    assert "fixed_test_well_used" in overlay
    assert "expected_visible_hash_used" in overlay
    assert "functional_ridge_deploy_report.json" in overlay
    assert "submission_before_functional_ridge.csv" in overlay
    assert "set(_fr_corrections) != set(_fr_ids)" in overlay
    assert "_fr_metadata.get('contract', {})" in overlay
    assert "_fr_cache_contract.get('absolute_position_features_removed')" in overlay
    assert "_fr_cache_contract.get('training_rows_only')" in overlay
    assert "_fr_cache_contract.get('target_feature_leakage')" in overlay
    assert "_fr_cache_contract.get('well_id_encoded_as_feature')" in overlay
    assert "_fr_metadata.get('absolute_position_features_removed')" not in overlay

    all_code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    assert "RUN_OVERLAP_DRY_RUN_PROBE = False" in all_code
    assert "DEMO = _demo_well()" in all_code
    assert notebook["metadata"]["functional_ridge_build"]["target_free_overlay"] is True
    assert (
        notebook["metadata"]["functional_ridge_build"][
            "leaderboard_score_used_for_overlay_selection"
        ]
        is False
    )
    assert metadata["id"] == "heralces/rogii-functional-ridge-v1"
    assert "heralces/rogii-private-sequence-pilot-cache-v1" in metadata["dataset_sources"]
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
