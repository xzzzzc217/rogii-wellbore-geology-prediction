"""Build the local-only preregistered 2.3x dip-rate HMM notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_dip_rate_hmm_scale2_v1 as scale2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "dip_rate_hmm_scale23_v1"
OUTPUT = OUTPUT_DIR / "rogii-dip-rate-hmm-scale23-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"
KERNEL_ID = "heralces/rogii-dip-rate-hmm-scale23-v1"


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected one {label} fragment, found {value.count(old)}")
    return value.replace(old, new)


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _marked_cell(notebook: dict[str, Any], marker: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get(marker)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one notebook cell marked {marker}, found {len(matches)}")
    return matches[0]


def main() -> None:
    scale2.OUTPUT_DIR = OUTPUT_DIR
    scale2.OUTPUT = OUTPUT
    scale2.METADATA = METADATA
    scale2.KERNEL_ID = KERNEL_ID
    scale2.main()

    notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))
    overlay = _marked_cell(notebook, "dip_rate_hmm_overlay_code")
    code = _source(overlay)
    replacements = (
        ("OVERLAY_SCALE = 0.40", "OVERLAY_SCALE = 0.46", "overlay scale"),
        ("> 2.0000001", "> 2.3000001", "overlay movement cap"),
        (
            "'confirmation_decision': 'PASS_pooled_all_folds_rank_gate_tail_risk'",
            "'confirmation_decision': 'PASS_scale23_pooled_all_folds_rank_gate_tail_risk'",
            "confirmation decision",
        ),
        (
            "'scale2_rank_gate_passed': True",
            "'scale23_rank_gate_passed': True",
            "rank gate flag",
        ),
        (
            "'well_win_fraction_0.5786_tail_risk'",
            "'well_win_fraction_0.5623_tail_risk'",
            "win-risk disclosure",
        ),
        (
            "'worst_well_delta_1.8142_tail_risk'",
            "'worst_well_delta_2.0891_tail_risk'",
            "worst-risk disclosure",
        ),
        (
            "'Scale2 confirmation excluding current test wells improved pooled RMSE '\n"
            "        'by 0.244037 and all 12 spatial/Typewell folds; competition '\n"
            "        'ranking uses pooled RMSE, while well-level tail risk is disclosed.'",
            "'Scale2.3 confirmation excluding current test wells improved pooled RMSE '\n"
            "        'by 0.256847 and all 12 spatial/Typewell folds; competition '\n"
            "        'ranking uses pooled RMSE, while well-level tail risk is disclosed.'",
            "rank rationale",
        ),
        (
            "'pooled_delta': -0.24403719075116825",
            "'pooled_delta': -0.256847265382099",
            "pooled delta",
        ),
        (
            "'well_win_fraction': 0.5785907859078591",
            "'well_win_fraction': 0.5623306233062331",
            "well win fraction",
        ),
        (
            "'p95_well_delta': 1.3359175014266658",
            "'p95_well_delta': 1.553767945219925",
            "true p95",
        ),
        (
            "'worst_well_delta': 1.8142252473707021",
            "'worst_well_delta': 2.089131786592704",
            "worst-well delta",
        ),
    )
    for old, new, label in replacements:
        code = _replace_once(code, old, new, label)
    overlay["source"] = code

    contract = _marked_cell(notebook, "dip_rate_hmm_final_contract")
    contract["source"] = _replace_once(
        _source(contract),
        "> 2.0000001",
        "> 2.3000001",
        "final movement cap",
    )
    intro = _marked_cell(notebook, "dip_rate_hmm_overlay")
    intro["source"] = (
        "# ROGII target-free dip-rate HMM scale2.3 v1\n\n"
        "Local-only preregistered rank candidate on the exact 6.477 parent. "
        "All current test-well IDs were excluded from label-based confirmation. "
        "The fixed 2.3x correction improved pooled RMSE by 0.256847 and every "
        "spatial/Typewell fold, but true p95 and worst-well risk increased. This "
        "Notebook must not be pushed unless scale2 first scores strictly below 6.477."
    )
    build = notebook.setdefault("metadata", {}).setdefault("dip_rate_hmm_build", {})
    build.update(
        {
            "variant": "scale23_rank_seeking_conditional",
            "correction_multiplier": 2.3,
            "effective_overlay_scale": 0.46,
            "maximum_final_movement_ft": 2.3,
            "confirmation_pooled_delta": -0.256847265382099,
            "true_p95_well_delta": 1.553767945219925,
            "worst_well_delta": 2.089131786592704,
            "execution_authorized": False,
            "authorization_condition": "scale2_public_score_strictly_below_6.477",
        }
    )
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metadata.update(
        {
            "id": KERNEL_ID,
            "title": "ROGII Dip Rate HMM Scale2.3 v1",
            "code_file": OUTPUT.name,
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote local-only scale2.3 notebook {OUTPUT}")


if __name__ == "__main__":
    main()
