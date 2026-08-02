from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_dip_rate_hmm_v1 as builder  # noqa: E402


def test_core_source_is_exact_and_complete() -> None:
    core = builder._core_source()
    for name in builder.CORE_CONSTANTS | builder.CORE_FUNCTIONS:
        assert name in core
    assert "@njit(cache=True, nogil=True)" in core
    notebook_core = builder._notebook_core_source()
    assert "@njit(cache=True" not in notebook_core
    assert notebook_core.count("@njit(cache=False") == 2


def test_overlay_is_dynamic_target_free_and_risk_disclosed() -> None:
    overlay = builder.source(builder._overlay_cell())
    assert "high_risk_candidate" in overlay
    assert "target_free_runtime_inputs_only" in overlay
    assert "confirmation_decision" in overlay
    assert "leaderboard_score_used_for_selection" in overlay
    assert "DISAGREEMENT_CLIP" in overlay
    for forbidden in ("00e12e8b", "00bbac68", "000d7d20", "14151"):
        assert forbidden not in overlay


def test_builder_writes_private_notebook() -> None:
    builder.main()
    notebook = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    metadata = json.loads(builder.METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "heralces/rogii-dip-rate-hmm-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    sources = [builder.source(cell) for cell in notebook["cells"]]
    assert any("dip-rate HMM deployment report" in value for value in sources)
    assert any("dip-rate HMM final contract: PASS" in value for value in sources)
