from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_dip_rate_hmm_scale2_v1.py"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "dip_rate_hmm_scale2_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-dip-rate-hmm-scale2-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_scale2_builder_contract() -> None:
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "heralces/rogii-dip-rate-hmm-scale2-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    all_source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "OVERLAY_SCALE = 0.40" in all_source
    assert "PASS_pooled_all_folds_rank_gate_tail_risk" in all_source
    assert "scale2_rank_gate_passed" in all_source
    assert "confirmation_test_wells_excluded" in all_source
    assert "> 2.0000001" in all_source
    assert "RUN_OVERLAP_DRY_RUN_PROBE = False" in all_source
    assert "DEMO = _demo_well()" in all_source
    for forbidden in ("00e12e8b", "00bbac68", "000d7d20", "14151"):
        assert forbidden not in all_source
    build = notebook["metadata"]["dip_rate_hmm_build"]
    assert build["variant"] == "scale2_rank_seeking"
    assert build["confirmation_current_test_wells_excluded"] is True
    assert build["effective_overlay_scale"] == 0.4
    assert build["maximum_final_movement_ft"] == 2.0
