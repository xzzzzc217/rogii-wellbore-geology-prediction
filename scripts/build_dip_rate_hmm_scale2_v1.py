"""Build the independent rank-seeking 2x dip-rate HMM overlay notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_dip_rate_hmm_v1 as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "dip_rate_hmm_scale2_v1"
OUTPUT = OUTPUT_DIR / "rogii-dip-rate-hmm-scale2-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"
KERNEL_ID = "heralces/rogii-dip-rate-hmm-scale2-v1"


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected one {label} fragment, found {value.count(old)}")
    return value.replace(old, new)


def _cell_source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _marked_cell(notebook: dict[str, Any], marker: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get(marker)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one notebook cell marked {marker}, found {len(matches)}")
    return matches[0]


def main() -> None:
    base.OUTPUT_DIR = OUTPUT_DIR
    base.OUTPUT = OUTPUT
    base.METADATA = METADATA
    base.main()

    notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))
    overlay = _marked_cell(notebook, "dip_rate_hmm_overlay_code")
    code = _cell_source(overlay)
    replacements = (
        ("OVERLAY_SCALE = 0.20", "OVERLAY_SCALE = 0.40", "overlay scale"),
        ("> 1.0000001", "> 2.0000001", "overlay movement cap"),
        (
            "'confirmation_decision': 'FAIL_tail_gates'",
            "'confirmation_decision': 'PASS_pooled_all_folds_rank_gate_tail_risk'",
            "confirmation decision",
        ),
        (
            "'high_risk_candidate': True,",
            (
                "'high_risk_candidate': True,\n"
                "    'scale2_rank_gate_passed': True,\n"
                "    'confirmation_test_wells_excluded': True,\n"
                "    'true_p95_quantile': 0.95,"
            ),
            "scale2 risk flags",
        ),
        (
            "'well_win_fraction_0.6383_below_0.65',\n"
            "        'worst_well_delta_0.6836_above_0.25',",
            "'well_win_fraction_0.5786_tail_risk',\n"
            "        'worst_well_delta_1.8142_tail_risk',",
            "tail disclosure",
        ),
        (
            "'Untouched confirmation improved pooled RMSE by 0.156756 and all 12 '\n"
            "        'spatial/Typewell folds; competition ranking uses pooled RMSE.'",
            "'Scale2 confirmation excluding current test wells improved pooled RMSE '\n"
            "        'by 0.244037 and all 12 spatial/Typewell folds; competition '\n"
            "        'ranking uses pooled RMSE, while well-level tail risk is disclosed.'",
            "rank rationale",
        ),
        ("'wells': 741", "'wells': 738", "confirmation well count"),
        (
            "'pooled_delta': -0.15675641408341612",
            "'pooled_delta': -0.24403719075116825",
            "confirmation pooled delta",
        ),
        (
            "'well_win_fraction': 0.6383265856950068",
            "'well_win_fraction': 0.5785907859078591",
            "confirmation win fraction",
        ),
        (
            "'p95_well_delta': -0.12914116104679962",
            "'p95_well_delta': 1.3359175014266658",
            "true p95 delta",
        ),
        (
            "'worst_well_delta': 0.6836268403655126",
            "'worst_well_delta': 1.8142252473707021",
            "worst-well delta",
        ),
    )
    for old, new, label in replacements:
        code = _replace_once(code, old, new, label)
    overlay["source"] = code

    contract = _marked_cell(notebook, "dip_rate_hmm_final_contract")
    contract["source"] = _replace_once(
        _cell_source(contract),
        "> 1.0000001",
        "> 2.0000001",
        "final movement cap",
    )

    intro = _marked_cell(notebook, "dip_rate_hmm_overlay")
    intro["source"] = (
        "# ROGII target-free dip-rate HMM scale2 v1\n\n"
        "Independent high-risk rank-seeking validation on the exact 6.477 parent. "
        "After excluding every current test-well ID from confirmation labels, the "
        "frozen 2x correction improved pooled RMSE by 0.244037 and every one of 12 "
        "spatial/Typewell folds. True p95 and worst-well deltas are positive, so this "
        "candidate is explicitly tail-risky and cannot alter final selection without "
        "a strictly better terminal Kaggle score."
    )

    build = notebook.setdefault("metadata", {}).setdefault("dip_rate_hmm_build", {})
    build.update(
        {
            "variant": "scale2_rank_seeking",
            "base_overlay_scale": 0.2,
            "correction_multiplier": 2.0,
            "effective_overlay_scale": 0.4,
            "maximum_final_movement_ft": 2.0,
            "confirmation_wells": 738,
            "confirmation_current_test_wells_excluded": True,
            "confirmation_pooled_delta": -0.24403719075116825,
            "confirmation_all_spatial_typewell_folds_improve": True,
            "confirmation_tail_gate_passed": False,
            "true_p95_well_delta": 1.3359175014266658,
            "worst_well_delta": 1.8142252473707021,
        }
    )

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metadata.update(
        {
            "id": KERNEL_ID,
            "title": "ROGII Dip Rate HMM Scale2 v1",
            "code_file": OUTPUT.name,
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched scale2 notebook {OUTPUT}")
    print(f"patched scale2 metadata {METADATA}")


if __name__ == "__main__":
    main()
