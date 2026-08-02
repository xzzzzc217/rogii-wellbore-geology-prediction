"""Build the private target-free PF branch-cap 2.5 Kaggle notebook.

The public parent appends an A23 transaction that writes a specific visible
test well, expected row count, expected artifact hash, and leaderboard score
into the scoring path. This builder removes that transaction. It changes only
the generic, run-local PF seed-branch cap from 2.0 ft to 2.5 ft.
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
    / "public_scout_20260728"
    / "mtoshidesu_rogii-test-another-approch-2nd"
)
PARENT = PARENT_DIR / "rogii-test-another-approch-2nd.ipynb"
PARENT_METADATA = PARENT_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "branch_cap25_targetfree_v1"
OUTPUT = OUTPUT_DIR / "rogii-pf-branch-cap25-targetfree-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"

EXPECTED_PARENT_SHA256 = (
    "15e5636714bcb07c4ad16a71140e7f6cb21b75646706d8348c370fbf86a17a2f"
)
BRANCH_CELL = 53
PARENT_A23_CELL = 54
PARENT_FINAL_AUDIT_CELL = 55
RETAINED_PARENT_CELLS = [*range(1, BRANCH_CELL + 1), PARENT_FINAL_AUDIT_CELL]


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _start_cell() -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"branch_cap25_start": True},
        "outputs": [],
        "source": (
            "import time as _cap25_time\n"
            "_CAP25_START_TIME = _cap25_time.time()\n"
        ),
    }


def _audit_cell() -> dict[str, object]:
    code = """
# Target-free cap-2.5 final contract and provenance report.
import hashlib as _cap25_hashlib
import json as _cap25_json
from pathlib import Path as _Cap25Path

import numpy as _cap25_np
import pandas as _cap25_pd

_CAP25_WORK = (
    _Cap25Path("/kaggle/working")
    if _Cap25Path("/kaggle/working").exists()
    else _Cap25Path(".")
)
_CAP25_SUB = _CAP25_WORK / "submission.csv"
_CAP25_BEFORE = _CAP25_WORK / "submission_before_branch_hedge.csv"
_CAP25_BRANCH = _CAP25_WORK / "pf_seed_branch_hedge_report.csv"
_CAP25_DATA = _Cap25Path(
    getattr(
        CFG,
        "DATA",
        getattr(
            CFG,
            "dataset_path",
            "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        ),
    )
)
_CAP25_SAMPLE = _CAP25_DATA / "sample_submission.csv"

for _cap25_required in (_CAP25_SUB, _CAP25_BEFORE, _CAP25_BRANCH, _CAP25_SAMPLE):
    if not _cap25_required.exists():
        raise RuntimeError(f"missing cap-2.5 contract input: {_cap25_required}")

_cap25_sub = _cap25_pd.read_csv(_CAP25_SUB, dtype={"id": "string"})
_cap25_before = _cap25_pd.read_csv(_CAP25_BEFORE, dtype={"id": "string"})
_cap25_sample = _cap25_pd.read_csv(_CAP25_SAMPLE, dtype={"id": "string"})
_cap25_branch = _cap25_pd.read_csv(_CAP25_BRANCH)
if list(_cap25_sub.columns) != ["id", "tvt"]:
    raise RuntimeError("unexpected submission columns")
if list(_cap25_before.columns) != ["id", "tvt"]:
    raise RuntimeError("unexpected pre-branch columns")
if len(_cap25_sub) != len(_cap25_sample):
    raise RuntimeError("submission row count does not match sample")
if not _cap25_sub["id"].astype(str).equals(_cap25_sample["id"].astype(str)):
    raise RuntimeError("submission IDs do not match sample order")
if not _cap25_sub["id"].is_unique:
    raise RuntimeError("submission IDs are not unique")
if not _cap25_before["id"].astype(str).equals(_cap25_sub["id"].astype(str)):
    raise RuntimeError("pre-branch IDs do not match final submission")

_cap25_values = _cap25_pd.to_numeric(
    _cap25_sub["tvt"],
    errors="coerce",
).to_numpy(dtype=float)
_cap25_before_values = _cap25_pd.to_numeric(
    _cap25_before["tvt"],
    errors="coerce",
).to_numpy(dtype=float)
if not _cap25_np.isfinite(_cap25_values).all():
    raise RuntimeError("submission contains non-finite predictions")
if not _cap25_np.isfinite(_cap25_before_values).all():
    raise RuntimeError("pre-branch submission contains non-finite predictions")

_cap25_required_columns = {"well", "reason", "shift", "moved_rows"}
if not _cap25_required_columns.issubset(_cap25_branch.columns):
    raise RuntimeError("branch report is missing required columns")
_cap25_applied = _cap25_branch[
    _cap25_branch["reason"].astype(str).str.lower().eq("applied")
].copy()
_cap25_delta = _cap25_values - _cap25_before_values
_cap25_changed = _cap25_np.abs(_cap25_delta) > 1e-10
_cap25_expected_moved = int(
    _cap25_pd.to_numeric(
        _cap25_applied["moved_rows"],
        errors="raise",
    ).sum()
)
if int(_cap25_changed.sum()) != _cap25_expected_moved:
    raise RuntimeError("changed-row count does not match branch report")

_cap25_wells = _cap25_sub["id"].astype(str).str.rsplit("_", n=1).str[0]
for _, _cap25_row in _cap25_applied.iterrows():
    _cap25_well = str(_cap25_row["well"])
    _cap25_shift = float(_cap25_row["shift"])
    _cap25_moved = int(_cap25_row["moved_rows"])
    _cap25_mask = _cap25_wells.eq(_cap25_well).to_numpy()
    _cap25_matches = _cap25_np.isclose(
        _cap25_delta[_cap25_mask],
        _cap25_shift,
        atol=1e-8,
        rtol=0.0,
    )
    if int(_cap25_matches.sum()) != _cap25_moved:
        raise RuntimeError(f"branch delta audit failed for {_cap25_well}")

_cap25_hash = _cap25_hashlib.sha256(_CAP25_SUB.read_bytes()).hexdigest()
_cap25_report = {
    "candidate": "target-free PF seed-branch cap 2.5",
    "source_kernel": "mtoshidesu/rogii-test-another-approch-2nd",
    "leaderboard_canary_used": False,
    "leaderboard_bias_applied": False,
    "probe_flags_used": False,
    "fixed_test_well_used": False,
    "expected_visible_hash_used": False,
    "parameters": {
        "strength": float(_BH_STRENGTH),
        "minor_mass_minimum": float(_BH_MIN_MASS),
        "separation_minimum": float(_BH_SEP_LOW),
        "separation_maximum": float(_BH_SEP_HIGH),
        "shift_cap": float(_BH_CAP),
    },
    "branch_stat_well_count": int(len(_cap25_branch)),
    "applied_well_count": int(len(_cap25_applied)),
    "applied_wells": sorted(_cap25_applied["well"].astype(str).tolist()),
    "applied_shifts": {
        str(row["well"]): float(row["shift"])
        for _, row in _cap25_applied.iterrows()
    },
    "moved_rows_by_well": {
        str(row["well"]): int(row["moved_rows"])
        for _, row in _cap25_applied.iterrows()
    },
    "changed_rows": int(_cap25_changed.sum()),
    "mean_abs_move": float(_cap25_np.mean(_cap25_np.abs(_cap25_delta))),
    "max_abs_move": float(_cap25_np.max(_cap25_np.abs(_cap25_delta))),
    "rows": int(len(_cap25_sub)),
    "id_order_matches_sample": True,
    "ids_unique": True,
    "finite_predictions": True,
    "submission_sha256": _cap25_hash,
    "elapsed_seconds": float(_cap25_time.time() - _CAP25_START_TIME),
}
(_CAP25_WORK / "branch_cap25_targetfree_report.json").write_text(
    _cap25_json.dumps(_cap25_report, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print("Target-free cap-2.5 final audit:", _cap25_report, flush=True)
""".strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"branch_cap25_targetfree_audit": True},
        "outputs": [],
        "source": code,
    }


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected one {label} fragment, found {value.count(old)}")
    return value.replace(old, new)


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    parent_sha256 = sha256_bytes(parent_bytes)
    if parent_sha256 != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"public A23 parent changed: {parent_sha256} != {EXPECTED_PARENT_SHA256}"
        )
    parent = json.loads(parent_bytes)
    parent_metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))
    if parent_metadata.get("id") != "mtoshidesu/rogii-test-another-approch-2nd":
        raise RuntimeError("unexpected public parent kernel")
    if "_EX_EXPECTED_WELL = '00e12e8b'" not in source(
        parent["cells"][PARENT_A23_CELL]
    ):
        raise RuntimeError("expected fixed-well A23 transaction was not found")

    notebook = copy.deepcopy(parent)
    cells = [
        copy.deepcopy(parent["cells"][index])
        for index in RETAINED_PARENT_CELLS
    ]
    for cell, parent_index in zip(cells, RETAINED_PARENT_CELLS, strict=True):
        cell_source = source(cell)
        if parent_index == 7:
            cell_source = _replace_once(
                cell_source,
                "RUN_OVERLAP_DRY_RUN_PROBE = True",
                "RUN_OVERLAP_DRY_RUN_PROBE = False",
                "dry-run probe toggle",
            )
        elif parent_index == 36:
            cell_source = _replace_once(
                cell_source,
                (
                    'DEMO = "00bbac68" if '
                    '(CFG.DATA/"train"/"00bbac68__horizontal_well.csv").exists() '
                    "else _demo_well()"
                ),
                "DEMO = _demo_well()",
                "fixed demo well",
            )
        elif parent_index == BRANCH_CELL:
            cell_source = _replace_once(
                cell_source,
                "_BH_CAP = 2.00",
                "_BH_CAP = 2.50",
                "PF branch cap",
            )
        cell["source"] = cell_source
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        cell.setdefault("metadata", {})[
            "branch_cap25_targetfree_parent_index"
        ] = parent_index

    notebook["cells"] = [
        {
            "cell_type": "markdown",
            "metadata": {"branch_cap25_targetfree": True},
            "source": (
                "# ROGII target-free PF seed-branch cap 2.5\n\n"
                "Private controlled verification derived from public Notebook "
                "`mtoshidesu/rogii-test-another-approch-2nd`. The fixed-well "
                "A23 transaction, expected visible row/hash checks, and "
                "leaderboard-score assumptions are excluded. The only scoring "
                "change is the generic run-local PF seed-branch cap from "
                "2.0 ft to 2.5 ft."
            ),
        },
        _start_cell(),
        *cells,
        _audit_cell(),
    ]

    retained_code = "\n".join(
        source(cell)
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
    survived = [fragment for fragment in prohibited if fragment in retained_code]
    if survived:
        raise RuntimeError(f"prohibited fixed-test fragments survived: {survived}")
    if retained_code.count("_BH_CAP = 2.50") != 1:
        raise RuntimeError("target-free cap 2.5 was not installed exactly once")

    notebook.setdefault("metadata", {})["branch_cap25_targetfree_build"] = {
        "source_kernel": parent_metadata["id"],
        "source_notebook_sha256": parent_sha256,
        "retained_parent_cells": RETAINED_PARENT_CELLS,
        "excluded_parent_cells": [
            0,
            PARENT_A23_CELL,
            *range(PARENT_FINAL_AUDIT_CELL + 1, len(parent["cells"])),
        ],
        "fixed_test_well_used": False,
        "expected_visible_hash_used": False,
        "leaderboard_score_used_at_runtime": False,
        "scoring_change": "generic PF seed-branch cap 2.0 -> 2.5 ft",
    }

    metadata = {
        "id": "heralces/rogii-pf-branch-cap25-targetfree-v1",
        "title": "ROGII PF Branch Cap25 Targetfree v1",
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
