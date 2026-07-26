"""Static validator for the exact-public no-hedge child notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "rogii-pf-public-nohedge.ipynb"
PARENT = (
    ROOT.parent
    / "pf_public_reproduction"
    / "rogii-pf-public-reproduction.ipynb"
)


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> int:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    child = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(child["cells"]) == len(parent["cells"]) + 1
    assert child["cells"][:-1] == parent["cells"]
    final = child["cells"][-1]
    source = _source(final)
    assert final["cell_type"] == "code"
    assert "PF_PUBLIC_NOHEDGE_FINAL_OVERRIDE" in source
    assert "_nohedge_shutil.copyfile(_nohedge_source, _nohedge_output)" in source
    assert "'submission_before_branch_hedge.csv'" in source
    assert "'submission.csv'" in source
    assert "sample_submission.csv" in source
    assert "np.isfinite" in source
    assert "TVT" not in source
    print(
        f"validated {NOTEBOOK}: {len(parent['cells'])} byte-identical parent "
        "cells plus one final no-hedge override"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
