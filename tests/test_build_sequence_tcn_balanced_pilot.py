from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "kaggle_notebooks" / "sequence_tcn_balanced_pilot_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-sequence-tcn-balanced-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_builder_freezes_uniform_well_sampling_contract() -> None:
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "build_sequence_tcn_balanced_pilot_notebook.py"
            ),
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
    contract = notebook["metadata"]["rogii_sequence_tcn_balanced_pilot"]

    assert "'--well-sampling', 'uniform'" in code
    assert "sequence_tcn_balanced_pilot_report.json" in code
    assert "sequence_tcn_balanced_pilot.pt" in code
    assert "submission.csv" not in code
    assert contract["well_sampling"] == "uniform"
    assert contract["training_steps"] == 4_000
    assert contract["primary_overlay_scale"] == 0.20
    assert contract["validation_fold"] == 0
    assert contract["writes_submission"] is False
    assert metadata["id"] == "heralces/rogii-sequence-tcn-well-balanced-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
