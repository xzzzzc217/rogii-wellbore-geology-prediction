"""Build a private A28-only Kaggle notebook from the audited public frontier run.

The public source finishes with a later A28/Q0522 pair experiment.  This
builder retains the source through its A28 audit cell only, so the competition
contract file is the run-local PF-1.3 blend at SP45 weight 0.62.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "artifacts"
    / "public_scout_20260726"
    / "frontier_blend"
    / "rogii-public-frontier-blend-research-visuals.ipynb"
)
PARENT_METADATA = PARENT.with_name("kernel-metadata.json")
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "frontier_a28_v1"
OUTPUT = OUTPUT_DIR / "rogii-frontier-a28-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"

# Cell 55 creates the A28 w0.62 trajectory, cell 56 audits submission.csv, and
# cell 57 prints a compact provenance summary.  Later cells create other
# experiments and therefore must not be retained.
LAST_RETAINED_CELL = 57


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    parent = json.loads(parent_bytes)
    parent_metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))

    if len(parent["cells"]) <= LAST_RETAINED_CELL:
        raise RuntimeError("public frontier notebook is missing the audited A28 cells")
    if "A28 hidden-safe PF-1.3 blend boundary reconstruction" not in source(
        parent["cells"][55]
    ):
        raise RuntimeError("public frontier A28 selection cell changed")
    if "Final submission audit" not in source(parent["cells"][56]):
        raise RuntimeError("public frontier final audit cell changed")
    if "A28 compact run summary" not in source(parent["cells"][57]):
        raise RuntimeError("public frontier compact summary cell changed")

    notebook = copy.deepcopy(parent)
    notebook["cells"] = notebook["cells"][: LAST_RETAINED_CELL + 1]
    notebook["cells"][0] = {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            "# ROGII Frontier A28 w0.62\n\n"
            "Private verification copy of the audited public A28 PF-1.3 boundary. "
            "The scoring path is retained byte-for-byte through the A28 final "
            "audit; later Q0522 and pair-blend experiments are excluded."
        ),
    }
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        cell.setdefault("metadata", {})["frontier_a28_parent_index"] = index

    notebook.setdefault("metadata", {})["frontier_a28_build"] = {
        "source_kernel": parent_metadata["id"],
        "source_notebook_sha256": sha256_bytes(parent_bytes),
        "last_retained_parent_cell": LAST_RETAINED_CELL,
        "final_candidate": "A28 PF-1.3 blend w_sp45=0.62",
        "source_reported_public_score": 6.433,
        "later_experiments_excluded": ["DYNQ0522", "A28_Q0522_PAIRBLEND_W050"],
    }

    metadata = {
        "id": "heralces/rogii-frontier-a28-v1",
        "title": "ROGII Frontier A28 v1",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["geology", "ensembling"],
        "dataset_sources": parent_metadata.get("dataset_sources", []),
        "kernel_sources": parent_metadata.get("kernel_sources", []),
        "competition_sources": parent_metadata.get("competition_sources", []),
        "model_sources": parent_metadata.get("model_sources", []),
    }

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
