"""Build the local-only Guard640 v4b notebook from the immutable v4 artifact.

The only scoring-path change is the model-package global p95 disable threshold:
25.0 -> 1.0e9. Identity strings and audit filenames are updated so a future
Kaggle dry-run cannot be mistaken for v4.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "kaggle_notebooks" / "pf_guard640_v4" / "rogii-guard640-v4.ipynb"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "pf_guard640_v4b"
OUTPUT = OUTPUT_DIR / "rogii-guard640-v4b.ipynb"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _replace_once(cells: list[dict], old: str, new: str) -> None:
    matches: list[int] = []
    for index, cell in enumerate(cells):
        value = _source(cell)
        count = value.count(old)
        if count:
            if count != 1:
                raise RuntimeError(f"expected one occurrence per matching cell: {old!r}")
            matches.append(index)
            cell["source"] = value.replace(old, new, 1)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one notebook occurrence of {old!r}, got {matches}")


def main() -> None:
    notebook = json.loads(V4.read_text(encoding="utf-8"))
    cells = copy.deepcopy(notebook["cells"])

    # The sole behavioral change. The public 6.40 source uses the same value.
    _replace_once(
        cells,
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0",
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9",
    )
    _replace_once(
        cells,
        "assert MODEL_PACKAGE_DIFF_P95_DISABLE == 25.0",
        "assert MODEL_PACKAGE_DIFF_P95_DISABLE == 1.0e9",
    )

    # Non-behavioral identity changes keep provenance and run artifacts distinct.
    _replace_once(cells, "# ROGII Guard640 v4", "# ROGII Guard640 v4b")
    _replace_once(
        cells,
        "'derived_kernel': 'heralces/rogii-guard640-v4'",
        "'derived_kernel': 'heralces/rogii-guard640-v4b'",
    )
    _replace_once(
        cells,
        "guard640_v4_final_audit.json",
        "guard640_v4b_final_audit.json",
    )

    metadata = copy.deepcopy(notebook.get("metadata", {}))
    v4_metadata = copy.deepcopy(metadata.pop("guard640_v4", {}))
    v4_metadata["variant_parent"] = "heralces/rogii-guard640-v4"
    v4_metadata.setdefault("model_package", {})["diff_p95_disable"] = 1.0e9
    metadata["guard640_v4b"] = v4_metadata

    notebook["cells"] = cells
    notebook["metadata"] = metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
