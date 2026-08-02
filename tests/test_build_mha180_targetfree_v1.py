from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "kaggle_notebooks" / "mha180_targetfree_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-mha180-targetfree-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_builder_removes_probe_and_bias_cells() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_mha180_targetfree_v1.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )

    assert "PF_BIMODAL_STATS" in code
    assert "1.8, 0.22, 3.0, 40.0, 4.0" in code
    assert "mha180_targetfree_report.json" in code
    assert "_BC_SHIFT" not in code
    assert "hidden public test via the probe canary" not in code
    assert "PROBE v4 ARMED" not in code
    assert "+100000" not in code
    assert notebook["metadata"]["mha180_targetfree_build"][
        "leaderboard_derived_bias_used"
    ] is False
    assert '"leaderboard_bias_applied": False' in code
    assert '"probe_flags_used": False' in code
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
