"""Build a private target-free MHA180 midpoint-hedge Kaggle notebook.

The public parent contains a leaderboard-canary-derived global bias and a
flag-gated probe cell. Both are excluded. The retained scoring change uses
only run-local particle-filter branch statistics and skips wells already
handled by the guarded train/test overlap transaction.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = (
    ROOT
    / "artifacts"
    / "public_scout_20260727_late"
    / "youill0317_rogii-mha180sep3-public-repro"
)
PARENT = PARENT_DIR / "rogii-mha180sep3-public-repro.ipynb"
PARENT_METADATA = PARENT_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "mha180_targetfree_v1"
OUTPUT = OUTPUT_DIR / "rogii-mha180-targetfree-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"

EXPECTED_PARENT_SHA256 = "ad08227bbaa2a742a6e4edcc2879c813230c414205d0a8008885c8a899d7b0d2"
PREFIX_LAST_CELL = 39
BIAS_CELL = 40
MIDHEDGE_CELL = 41


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _audit_cell() -> dict[str, object]:
    code = """
# Target-free final contract and provenance report.
import hashlib as _tf_hashlib
import json as _tf_json
from pathlib import Path as _TfPath

import numpy as _tf_np
import pandas as _tf_pd

_TF_WORK = _TfPath("/kaggle/working") if _TfPath("/kaggle/working").exists() else _TfPath(".")
_TF_SUB = _TF_WORK / "submission.csv"
_TF_SAMPLE = _GOLD_DATA / "sample_submission.csv"

_tf_sub = _tf_pd.read_csv(_TF_SUB)
_tf_sample = _tf_pd.read_csv(_TF_SAMPLE)
if list(_tf_sub.columns) != ["id", "tvt"]:
    raise RuntimeError("unexpected submission columns")
if len(_tf_sub) != len(_tf_sample):
    raise RuntimeError("submission row count does not match sample")
if not _tf_sub["id"].astype(str).equals(_tf_sample["id"].astype(str)):
    raise RuntimeError("submission IDs do not match sample order")
if not _tf_np.isfinite(_tf_sub["tvt"].to_numpy(dtype=float)).all():
    raise RuntimeError("submission contains non-finite predictions")

_tf_hash = _tf_hashlib.sha256(_TF_SUB.read_bytes()).hexdigest()
_tf_report = {
    "candidate": "MHA180 target-free midpoint hedge",
    "source_kernel": "youill0317/rogii-mha180sep3-public-repro",
    "leaderboard_canary_used": False,
    "leaderboard_derived_bias_used": False,
    "leaderboard_bias_applied": False,
    "probe_cells_retained": False,
    "probe_flags_used": False,
    "parameters": {
        "alpha": float(_MH_ALPHA),
        "minor_mass_minimum": float(_MH_MINMASS),
        "separation_minimum": float(_MH_SEPLO),
        "separation_maximum": float(_MH_SEPHI),
        "shift_cap": float(_MH_CAP),
    },
    "pf_stat_wells": int(len(_stats)),
    "qualified_wells": sorted(str(well_id) for well_id in _shifts),
    "qualified_well_count": int(len(_shifts)),
    "skipped_overlap_wells": sorted(str(well_id) for well_id in _skip),
    "moved_rows_by_well": {
        str(well_id): int(len(rows)) for well_id, (_, rows) in _shifts.items()
    },
    "shifts_by_well": {
        str(well_id): float(shift) for well_id, (shift, _) in _shifts.items()
    },
    "rows": int(len(_tf_sub)),
    "id_order_matches_sample": True,
    "finite_predictions": True,
    "submission_sha256": _tf_hash,
}
(_TF_WORK / "mha180_targetfree_report.json").write_text(
    _tf_json.dumps(_tf_report, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print("MHA180 target-free final audit:", _tf_report, flush=True)
""".strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"mha180_targetfree_audit": True},
        "outputs": [],
        "source": code,
    }


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    parent_sha256 = sha256_bytes(parent_bytes)
    if parent_sha256 != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"public MHA180 parent changed: {parent_sha256} != {EXPECTED_PARENT_SHA256}"
        )
    parent = json.loads(parent_bytes)
    parent_metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))
    if parent_metadata.get("id") != "youill0317/rogii-mha180sep3-public-repro":
        raise RuntimeError("unexpected public parent kernel")
    if "hidden public test via the probe canary" not in source(
        parent["cells"][BIAS_CELL]
    ):
        raise RuntimeError("expected leaderboard-derived bias cell was not found")
    midhedge = source(parent["cells"][MIDHEDGE_CELL])
    required_midhedge_fragments = (
        "_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP",
        "1.8, 0.22, 3.0, 40.0, 4.0",
        "PF_BIMODAL_STATS",
        "override_status.csv",
    )
    if any(fragment not in midhedge for fragment in required_midhedge_fragments):
        raise RuntimeError("public midpoint-hedge cell changed")

    notebook = copy.deepcopy(parent)
    retained_indices = [*range(PREFIX_LAST_CELL + 1), MIDHEDGE_CELL]
    notebook["cells"] = [copy.deepcopy(parent["cells"][index]) for index in retained_indices]
    notebook["cells"][0] = {
        "cell_type": "markdown",
        "metadata": {"mha180_targetfree_parent_index": 0},
        "source": (
            "# ROGII MHA180 target-free midpoint hedge\n\n"
            "Private controlled verification derived from public Notebook "
            "`youill0317/rogii-mha180sep3-public-repro`. The run-local PF "
            "midpoint hedge is retained. The leaderboard-canary-derived global "
            "bias, all probe cells, and later leaderboard diagnostics are "
            "excluded. No public score is assumed."
        ),
    }
    notebook["cells"][1]["source"] = (
        'import os\n'
        'os.environ["ROGII_GOLD_PROFILE"] = "conservative"\n'
        'os.environ["ROGII_PROBE"] = "0"\n'
    )
    for output_index, (cell, parent_index) in enumerate(
        zip(notebook["cells"], retained_indices, strict=True)
    ):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        cell.setdefault("metadata", {})["mha180_targetfree_parent_index"] = parent_index
        cell["metadata"]["mha180_targetfree_output_index"] = output_index
    notebook["cells"].append(_audit_cell())

    retained_code = "\n".join(
        source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    prohibited = (
        "_BC_SHIFT",
        "hidden public test via the probe canary",
        "PROBE v4 ARMED",
        "+100000",
    )
    if any(fragment in retained_code for fragment in prohibited):
        raise RuntimeError("a prohibited leaderboard-derived cell survived the build")

    notebook.setdefault("metadata", {})["mha180_targetfree_build"] = {
        "source_kernel": parent_metadata["id"],
        "source_notebook_sha256": parent_sha256,
        "retained_parent_cells": retained_indices,
        "excluded_parent_cells": [BIAS_CELL, *range(MIDHEDGE_CELL + 1, len(parent["cells"]))],
        "leaderboard_canary_used": False,
        "leaderboard_derived_bias_used": False,
        "final_candidate": "run-local PF midpoint hedge alpha=1.8 sep=[3,40] cap=4",
    }

    metadata = {
        "id": "heralces/rogii-mha180-target-free-v1",
        "title": "ROGII MHA180 Target-Free v1",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["geology"],
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
