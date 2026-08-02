from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "kaggle_notebooks" / "a28_runtime_fast_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-a28-runtime-fast-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_builder_creates_pruned_dynamic_a28_notebook() -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_a28_runtime_fast_v1.py"),
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

    assert "SP45_SELECTOR_N_PARTICLES = 500" in code
    assert "SP45_SELECTOR_N_SEEDS = 128" in code
    assert "_A28H_W_SP45 = 0.62" in code
    assert "_A28H_W_LEARNED = 0.38" in code
    assert "a28_runtime_fast_final_report.json" in code
    assert '"source_score_used_for_selection": False' in code
    assert '"leaderboard_bias_applied": False' in code
    assert '"fixed_test_well_used": False' in code
    assert "RUN_OVERLAP_DRY_RUN_PROBE = False" in code
    assert "DEMO = _demo_well()" in code

    code_cell_starts = [
        _source(cell).lstrip().splitlines()[0]
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and _source(cell).strip()
    ]
    excluded_stage_starts = (
        "# Visible-prefix calibration overlay.",
        "# Optional saved-model correction on top of the current trajectory submission.",
        "# Guarded PF seed-branch midpoint hedge computed entirely in this notebook.",
        "# Full-stack bimodal CV ablation.",
    )
    assert not [
        fragment for fragment in excluded_stage_starts if fragment in code_cell_starts
    ]

    prohibited = (
        "14151",
        "00e12e8b",
        "00bbac68",
        "6.433",
        "source_public_score",
        "Q0522",
        "b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a",
    )
    assert not [fragment for fragment in prohibited if fragment in code]
    assert notebook["metadata"]["a28_runtime_fast_build"][
        "source_score_used_for_selection"
    ] is False
    assert notebook["metadata"]["a28_runtime_fast_build"][
        "visible_shape_constants_used"
    ] is False
    assert metadata["id"] == "heralces/rogii-a28-runtime-fast-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
