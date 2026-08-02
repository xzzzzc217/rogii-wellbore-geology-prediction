from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "kaggle_notebooks" / "branch_cap25_targetfree_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-pf-branch-cap25-targetfree-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_builder_removes_fixed_test_transaction() -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_branch_cap25_targetfree_v1.py"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    code = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert code.count("_BH_CAP = 2.50") == 1
    assert "_BH_CAP = 2.00" not in code
    assert "RUN_OVERLAP_DRY_RUN_PROBE = False" in code
    assert "DEMO = _demo_well()" in code
    assert "branch_cap25_targetfree_report.json" in code
    assert '"fixed_test_well_used": False' in code
    assert '"expected_visible_hash_used": False' in code

    prohibited = (
        "_EX_EXPECTED_WELL",
        "_EX_EXPECTED_ROWS",
        "_EX_EXPECTED_BASE_SHA",
        "_EX_EXTRA_SHIFT",
        "source_public_score",
        "00e12e8b",
        "00bbac68",
        "b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a",
    )
    assert not [fragment for fragment in prohibited if fragment in code]
    assert notebook["metadata"]["branch_cap25_targetfree_build"][
        "leaderboard_score_used_at_runtime"
    ] is False
    assert metadata["id"] == "heralces/rogii-pf-branch-cap25-targetfree-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
