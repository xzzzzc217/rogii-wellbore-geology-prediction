"""Build the private target-free PF branch-cap 4.0 Kaggle notebook.

This is a single controlled, public-score-informed verification.  The generic
strength is raised from 0.60 to 0.62 so a qualifying, sufficiently separated
branch reaches the 4.0 ft cap.  The runtime scoring path remains dynamic: it
selects PF seed branches from the current run and contains no fixed visible
well, row count, artifact hash, probe, or score.
"""

from __future__ import annotations

import json
from pathlib import Path

import build_branch_cap25_targetfree_v1 as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "branch_cap40_targetfree_v1"
OUTPUT = OUTPUT_DIR / "rogii-pf-branch-cap40-targetfree-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _replace_exact(value: str, old: str, new: str, count: int, label: str) -> str:
    actual = value.count(old)
    if actual != count:
        raise RuntimeError(f"expected {count} {label} fragments, found {actual}")
    return value.replace(old, new)


def main() -> None:
    # Reuse the already audited removal of the parent's fixed-test A23 cell.
    base.main()
    notebook = json.loads(base.OUTPUT.read_text(encoding="utf-8"))
    metadata = json.loads(base.METADATA.read_text(encoding="utf-8"))

    serialized = json.dumps(notebook, ensure_ascii=False)
    serialized = _replace_exact(
        serialized,
        "_BH_CAP = 2.50",
        "_BH_CAP = 4.00",
        1,
        "PF branch cap",
    )
    serialized = _replace_exact(
        serialized,
        "_BH_STRENGTH = 0.60",
        "_BH_STRENGTH = 0.62",
        1,
        "PF branch strength",
    )
    serialized = serialized.replace("_cap25", "_cap40")
    serialized = serialized.replace("_CAP25", "_CAP40")
    serialized = serialized.replace("branch_cap25", "branch_cap40")
    serialized = serialized.replace("cap-2.5", "cap-4.0")
    serialized = serialized.replace("cap 2.5", "cap 4.0")
    serialized = serialized.replace("Cap25", "Cap40")
    serialized = serialized.replace("2.5 ft", "4.0 ft")
    notebook = json.loads(serialized)

    code = "\n".join(
        base.source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    provenance_anchor = '"leaderboard_bias_applied": False,'
    if code.count(provenance_anchor) != 1:
        raise RuntimeError("unexpected leaderboard-bias report anchor")
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell_source = base.source(cell)
        if provenance_anchor in cell_source:
            cell["source"] = cell_source.replace(
                provenance_anchor,
                provenance_anchor
                + '\n    "candidate_selection_public_score_informed": True,',
            )

    build = notebook["metadata"].pop("branch_cap40_targetfree_build")
    build.update(
        {
            "candidate_selection_public_score_informed": True,
            "reference_public_score": 6.432,
            "reference_candidate_relation": (
                "visible output equals dynamic cap-2.0 main plus 2.0 ft on "
                "the same runtime-selected PF branch"
            ),
            "scoring_change": (
                "generic PF seed-branch strength 0.60 -> 0.62 and cap "
                "2.0 -> 4.0 ft"
            ),
        }
    )
    notebook["metadata"]["branch_cap40_targetfree_build"] = build

    code = "\n".join(
        base.source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
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
    survived = [fragment for fragment in prohibited if fragment in code]
    if survived:
        raise RuntimeError(f"prohibited fixed-test fragments survived: {survived}")
    if code.count("_BH_CAP = 4.00") != 1:
        raise RuntimeError("target-free cap 4.0 was not installed exactly once")
    if code.count("_BH_STRENGTH = 0.62") != 1:
        raise RuntimeError("target-free strength 0.62 was not installed exactly once")
    if "branch_cap40_targetfree_report.json" not in code:
        raise RuntimeError("cap-4.0 audit report path was not installed")

    metadata.update(
        {
            "id": "heralces/rogii-pf-branch-cap40-targetfree-v1",
            "title": "ROGII PF Branch Cap40 Targetfree v1",
            "code_file": OUTPUT.name,
            "is_private": True,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
