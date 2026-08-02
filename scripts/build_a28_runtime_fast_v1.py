"""Build a runtime-safe A28 notebook from the successful exact public core.

The earlier A28 notebook doubled both selector particles and seeds, then ran
visible-prefix, model-package, and branch-hedge stages whose outputs A28
ultimately replaced.  Its hidden Code Submission exceeded nine hours.

This build retains the already successful exact core only through creation of
``sp45_projection_submission.csv`` and ``learned_trajectory_submission.csv``.
It then applies the frozen 0.62/0.38 A28 blend and stops.  No leaderboard
score, visible row count, output hash, or fixed test well is used at runtime.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "kaggle_notebooks"
    / "pf_public_reproduction"
    / "rogii-pf-public-reproduction.ipynb"
)
PARENT_METADATA = PARENT.with_name("kernel-metadata.json")
A28_TEMPLATE = (
    ROOT
    / "kaggle_notebooks"
    / "frontier_a28_v1"
    / "rogii-frontier-a28-v1.ipynb"
)
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "a28_runtime_fast_v1"
OUTPUT = OUTPUT_DIR / "rogii-a28-runtime-fast-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"

EXPECTED_PARENT_SHA256 = (
    "8966d3e406c8732546360bee9e5856c9dfa3774be24fca6010aaf0b246be0dab"
)
EXPECTED_A28_TEMPLATE_SHA256 = (
    "096c13f8cd1459635cafaF05c569974e438a54373188eae154e04a133673a558".lower()
)
LAST_CORE_CELL = 46
A28_TEMPLATE_CELL = 55
FINAL_AUDIT_TEMPLATE_CELL = 54


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected one {label} fragment, found {value.count(old)}")
    return value.replace(old, new)


def _start_cell() -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"a28_runtime_fast_start": True},
        "outputs": [],
        "source": (
            "import time as _a28fast_time\n"
            "_A28_FAST_START_TIME = _a28fast_time.time()\n"
        ),
    }


def _final_report_cell() -> dict[str, object]:
    code = """
# Immutable A28 runtime-fast contract after the generic submission audit.
import hashlib as _a28f_hashlib
import json as _a28f_json
from pathlib import Path as _A28FPath

import numpy as _a28f_np
import pandas as _a28f_pd

_A28F_WORK = (
    _A28FPath("/kaggle/working")
    if _A28FPath("/kaggle/working").exists()
    else _A28FPath(".")
)
_A28F_SUB = _A28F_WORK / "submission.csv"
_A28F_REPORT = _A28F_WORK / "a28_runtime_fast_report.json"
_a28f_report = _a28f_json.loads(_A28F_REPORT.read_text(encoding="utf-8"))
_a28f_sub = _a28f_pd.read_csv(_A28F_SUB, dtype={"id": "string"})
_a28f_values = _a28f_pd.to_numeric(
    _a28f_sub["tvt"],
    errors="coerce",
).to_numpy(dtype="float64")
_a28f_sha = _a28f_hashlib.sha256(_A28F_SUB.read_bytes()).hexdigest()

if list(_a28f_sub.columns) != ["id", "tvt"]:
    raise RuntimeError("A28 runtime-fast submission has unexpected columns")
if not _a28f_sub["id"].is_unique:
    raise RuntimeError("A28 runtime-fast submission IDs are not unique")
if not _a28f_np.isfinite(_a28f_values).all():
    raise RuntimeError("A28 runtime-fast submission contains non-finite values")
if int(_a28f_report["rows"]) != len(_a28f_sub):
    raise RuntimeError("A28 runtime-fast report row count differs from submission")
if _a28f_report["selected_file_sha"] != _a28f_sha:
    raise RuntimeError("A28 runtime-fast report hash differs from submission")

_a28f_final = dict(_a28f_report)
_a28f_final.update(
    {
        "candidate": "A28 runtime-fast exact-core PF-1.3 w0.62",
        "source_score_used_for_selection": False,
        "leaderboard_bias_applied": False,
        "probe_flags_used": False,
        "fixed_test_well_used": False,
        "expected_visible_hash_used": False,
        "visible_shape_constants_used": False,
        "exact_core_hidden_submission_previously_completed": True,
        "selector_particles": int(SP45_SELECTOR_N_PARTICLES),
        "selector_seeds": int(SP45_SELECTOR_N_SEEDS),
        "post_blend_visible_prefix_executed": False,
        "post_blend_model_package_executed": False,
        "post_blend_branch_hedge_executed": False,
        "rows": int(len(_a28f_sub)),
        "id_order_matches_sample": True,
        "ids_unique": True,
        "finite_predictions": True,
        "submission_sha256": _a28f_sha,
        "elapsed_seconds": float(_a28fast_time.time() - _A28_FAST_START_TIME),
    }
)
(_A28F_WORK / "a28_runtime_fast_final_report.json").write_text(
    _a28f_json.dumps(_a28f_final, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print("A28 runtime-fast final contract:", _a28f_final, flush=True)
""".strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"a28_runtime_fast_final_report": True},
        "outputs": [],
        "source": code,
    }


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    if sha256_bytes(parent_bytes) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("exact public parent changed")
    template_bytes = A28_TEMPLATE.read_bytes()
    if sha256_bytes(template_bytes) != EXPECTED_A28_TEMPLATE_SHA256:
        raise RuntimeError("A28 selection template changed")

    parent = json.loads(parent_bytes)
    parent_metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))
    template = json.loads(template_bytes)
    if "wrote final submission.csv from" not in source(parent["cells"][LAST_CORE_CELL]):
        raise RuntimeError("exact parent core boundary changed")
    if "A28 hidden-safe PF-1.3 blend boundary reconstruction" not in source(
        template["cells"][A28_TEMPLATE_CELL]
    ):
        raise RuntimeError("A28 selection template is missing")
    if "Final submission audit" not in source(
        parent["cells"][FINAL_AUDIT_TEMPLATE_CELL]
    ):
        raise RuntimeError("exact final audit template is missing")

    retained = [
        copy.deepcopy(parent["cells"][index])
        for index in range(1, LAST_CORE_CELL + 1)
    ]
    for cell, parent_index in zip(
        retained,
        range(1, LAST_CORE_CELL + 1),
        strict=True,
    ):
        cell_source = source(cell)
        if parent_index == 7:
            cell_source = _replace_once(
                cell_source,
                "RUN_OVERLAP_DRY_RUN_PROBE = True",
                "RUN_OVERLAP_DRY_RUN_PROBE = False",
                "overlap probe toggle",
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
                "fixed visualization well",
            )
        cell["source"] = cell_source
        cell.setdefault("metadata", {})["a28_runtime_fast_parent_index"] = parent_index

    a28_cell = copy.deepcopy(template["cells"][A28_TEMPLATE_CELL])
    a28_source = source(a28_cell)
    a28_source = _replace_once(
        a28_source,
        "# A28 hidden-safe PF-1.3 blend boundary reconstruction.",
        "# A28 runtime-fast exact-core PF-1.3 blend boundary reconstruction.",
        "A28 heading",
    )
    a28_source = _replace_once(
        a28_source,
        "'source_public_score': 6.433,",
        "'source_score_used_for_selection': False,",
        "source score field",
    )
    a28_source = _replace_once(
        a28_source,
        "hidden-safe PF-1.3 boundary; ",
        "runtime-fast exact-core PF-1.3 boundary; ",
        "strategy label",
    )
    a28_source = _replace_once(
        a28_source,
        "a28_pf13_blend_w062_report.json",
        "a28_runtime_fast_report.json",
        "report filename",
    )
    a28_source = _replace_once(
        a28_source,
        "'visible_shape_constants_used': False,",
        (
            "'visible_shape_constants_used': False,\n"
            "    'leaderboard_bias_applied': False,\n"
            "    'probe_flags_used': False,\n"
            "    'fixed_test_well_used': False,\n"
            "    'expected_visible_hash_used': False,"
        ),
        "runtime safety fields",
    )
    a28_cell["source"] = a28_source
    a28_cell["metadata"] = {"a28_runtime_fast_selection": True}

    audit_cell = copy.deepcopy(parent["cells"][FINAL_AUDIT_TEMPLATE_CELL])
    audit_cell["metadata"] = {"a28_runtime_fast_submission_audit": True}

    notebook = copy.deepcopy(parent)
    notebook["cells"] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "# ROGII A28 runtime-fast v1\n\n"
                "The hidden-proven exact core stops immediately after producing "
                "SP45 and learned trajectories, then applies the frozen 0.62/0.38 "
                "A28 blend. Redundant post-blend stages are excluded."
            ),
        },
        _start_cell(),
        *retained,
        a28_cell,
        audit_cell,
        _final_report_cell(),
    ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    notebook.setdefault("metadata", {})["a28_runtime_fast_build"] = {
        "source_kernel": parent_metadata["id"],
        "source_notebook_sha256": EXPECTED_PARENT_SHA256,
        "last_retained_parent_cell": LAST_CORE_CELL,
        "selector_particles": 500,
        "selector_seeds": 128,
        "final_candidate": "A28 PF-1.3 blend w_sp45=0.62",
        "source_score_used_for_selection": False,
        "visible_shape_constants_used": False,
        "fixed_test_well_used": False,
        "leaderboard_bias_applied": False,
        "pruned_post_blend_stages": [
            "same-well overlap diagnostics",
            "visible-prefix calibration",
            "model-package correction",
            "PF branch hedge",
            "full-stack CV ablation",
        ],
    }

    metadata = {
        "id": "heralces/rogii-a28-runtime-fast-v1",
        "title": "ROGII A28 Runtime Fast v1",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["geology", "particle-filter", "ensembling"],
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
