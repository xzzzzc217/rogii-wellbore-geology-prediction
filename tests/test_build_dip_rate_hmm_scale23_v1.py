from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_dip_rate_hmm_scale23_v1.py"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "dip_rate_hmm_scale23_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-dip-rate-hmm-scale23-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_scale23_builder_is_local_only_and_target_free() -> None:
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "heralces/rogii-dip-rate-hmm-scale23-v1"
    assert metadata["is_private"] is True
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "OVERLAY_SCALE = 0.46" in source
    assert "PASS_scale23_pooled_all_folds_rank_gate_tail_risk" in source
    assert "scale23_rank_gate_passed" in source
    assert "> 2.3000001" in source
    for forbidden in ("00e12e8b", "00bbac68", "000d7d20", "14151"):
        assert forbidden not in source
    build = notebook["metadata"]["dip_rate_hmm_build"]
    assert build["variant"] == "scale23_rank_seeking_conditional"
    assert build["execution_authorized"] is False
    assert build["effective_overlay_scale"] == 0.46
