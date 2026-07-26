"""Fail-closed validator for the local-only Guard640 v4b notebook."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
NOTEBOOK = ROOT / "rogii-guard640-v4b.ipynb"
V4_NOTEBOOK = PROJECT / "kaggle_notebooks" / "pf_guard640_v4" / "rogii-guard640-v4.ipynb"
METADATA = ROOT / "kernel-metadata.json"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _literal_assignments(source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def _normalized_v4b_source(value: str) -> str:
    return (
        value.replace(
            "MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9",
            "MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0",
        )
        .replace(
            "assert MODEL_PACKAGE_DIFF_P95_DISABLE == 1.0e9",
            "assert MODEL_PACKAGE_DIFF_P95_DISABLE == 25.0",
        )
        .replace("# ROGII Guard640 v4b", "# ROGII Guard640 v4")
        .replace(
            "'derived_kernel': 'heralces/rogii-guard640-v4b'",
            "'derived_kernel': 'heralces/rogii-guard640-v4'",
        )
        .replace("guard640_v4b_final_audit.json", "guard640_v4_final_audit.json")
    )


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    v4_notebook = json.loads(V4_NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    v4_cells = v4_notebook["cells"]
    if len(cells) != len(v4_cells):
        raise AssertionError(f"cell count differs from v4: {len(cells)} != {len(v4_cells)}")

    # Prove that the scoring implementation differs from v4 only by the p95 value.
    for index, (cell, v4_cell) in enumerate(zip(cells, v4_cells, strict=True)):
        if cell.get("cell_type") != v4_cell.get("cell_type"):
            raise AssertionError(f"cell type differs from v4 at {index}")
        if _normalized_v4b_source(_source(cell)) != _source(v4_cell):
            raise AssertionError(f"unapproved source difference from v4 at cell {index}")
        if cell.get("metadata", {}) != v4_cell.get("metadata", {}):
            raise AssertionError(f"cell metadata differs from v4 at {index}")

    code_cells = [(i, cell) for i, cell in enumerate(cells) if cell["cell_type"] == "code"]
    for index, cell in code_cells:
        compile(_source(cell), f"{NOTEBOOK.name}:cell_{index}", "exec")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale execution state")

    control_matches = [
        _source(cell)
        for _, cell in code_cells
        if "SUBMISSION_PROFILE = 'guard640_v4'" in _source(cell)
    ]
    if len(control_matches) != 1:
        raise AssertionError(f"expected one control cell, got {len(control_matches)}")
    control = control_matches[0]
    values = _literal_assignments(control)
    runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(control, f"{NOTEBOOK.name}:control", "exec"), runtime)

    expected_literals = {
        "SUBMISSION_PROFILE": "guard640_v4",
        "NOTEBOOK_HARD_BUDGET_SEC": 27000.0,
        "NOTEBOOK_FINAL_RESERVE_SEC": 1800.0,
        "MODEL_PACKAGE_MIN_BUDGET_SEC": 7200.0,
        "SP45_SELECTOR_N_SEEDS": 96,
        "ALLOW_SAME_ID_TRAIN_LOOKUP": False,
        "RUN_GUARDED_OVERLAP_OVERRIDE": False,
        "RUN_OVERLAP_DRY_RUN_PROBE": False,
        "RUN_EXACT_MATCH_RECOVERY": False,
        "RUN_PUBLIC_OVERLAP_CALIBRATION": False,
        "APPLY_GLOBAL_LB_BIAS": False,
        "RUN_PF_BRANCH_HEDGE": True,
        "RUN_U_CONTINUITY": True,
        "VISIBLE_PREFIX_ALPHA_MULTIPLIER": 0.35,
        "PF_BRANCH_HEDGE_CAP": 3.0,
        "U_CONTINUITY_CAP": 8.0,
        "U_CONTINUITY_TAU": 240.0,
        "MODEL_PACKAGE_REQUIRE": True,
        "MODEL_PACKAGE_ALLOW_AUTO_SEARCH": False,
        "MODEL_PACKAGE_DIFF_P95_DISABLE": 1.0e9,
    }
    for name, expected in expected_literals.items():
        if values.get(name) != expected:
            raise AssertionError(f"{name}: expected {expected!r}, got {values.get(name)!r}")

    expected_runtime = {
        "RUN_VISIBLE_PREFIX_CALIBRATION": True,
        "VISIBLE_PREFIX_PROFILE": "balanced",
        "VISIBLE_PREFIX_FINAL_SELECTION": "profile",
        "RUN_MODEL_PACKAGE_CORRECTION": True,
        "MODEL_PACKAGE_GATED_MAX_WEIGHT": 0.49,
        "MODEL_PACKAGE_GATED_SCALE": 192.0,
        "SELECTOR_PF_SEEDS": 96,
    }
    for name, expected in expected_runtime.items():
        if runtime.get(name) != expected:
            raise AssertionError(
                f"runtime {name}: expected {expected!r}, got {runtime.get(name)!r}"
            )

    all_source = "\n".join(_source(cell) for cell in cells)
    if all_source.count("MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9") != 1:
        raise AssertionError("expected exactly one v4b p95 assignment")
    if "_gate = float(_gmax) / (1.0 + (_diff / _scale) ** 2)" not in all_source:
        raise AssertionError("audited row-wise disagreement gate changed")
    max_gate_move = (
        float(runtime["MODEL_PACKAGE_GATED_MAX_WEIGHT"])
        * float(runtime["MODEL_PACKAGE_GATED_SCALE"])
        / 2.0
    )
    if not math.isclose(max_gate_move, 47.04, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"unexpected analytical gate-move bound: {max_gate_move}")

    forbidden_fragments = (
        "RUN_GUARDED_OVERLAP_OVERRIDE = True",
        "RUN_OVERLAP_DRY_RUN_PROBE = True",
        "ALLOW_SAME_ID_TRAIN_LOOKUP = True",
        "RUN_EXACT_MATCH_RECOVERY = True",
        "RUN_PUBLIC_OVERLAP_CALIBRATION = True",
        "APPLY_GLOBAL_LB_BIAS = True",
        "GLOBAL_TVT_BIAS",
        "submission_before_public_overlap.csv",
        "Public-test overlap calibration",
        "ROGII_PROBE",
        "LB PROBE",
        "canary",
        "EXPECTED_SOURCE",
        "EXPECTED_PRED",
        "14151",
        "00bbac68",
        "PF_SEEDS = 64",
        "PF_SEEDS = 128",
        "alpha * 1.30",
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0",
    )
    for fragment in forbidden_fragments:
        if fragment in all_source:
            raise AssertionError(f"forbidden source fragment present: {fragment}")

    for index, cell in code_cells:
        tree = ast.parse(_source(cell))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.fullmatch(r"[0-9a-f]{8}", node.value.lower())
            ):
                raise AssertionError(f"fixed eight-hex well ID in cell {index}: {node.value}")

    hidden_contract = [
        (index, _source(cell))
        for index, cell in code_cells
        if "# Final hidden-set contract transaction." in _source(cell)
    ]
    if len(hidden_contract) != 1:
        raise AssertionError(f"expected one dynamic hidden contract, got {len(hidden_contract)}")
    hidden_index, hidden_source = hidden_contract[0]
    if "'derived_kernel': 'heralces/rogii-guard640-v4b'" not in hidden_source:
        raise AssertionError("v4b hidden-contract attribution mismatch")
    if (
        "len(_hc_reread) != len(_hc_sample)" not in hidden_source
        or "_hc_sample_ids" not in hidden_source
    ):
        raise AssertionError("hidden contract is not dynamically sample-derived")
    later_writers = [
        index for index, cell in code_cells if index > hidden_index and ".to_csv(" in _source(cell)
    ]
    if later_writers:
        raise AssertionError(f"dynamic contract is not the last CSV writer: {later_writers}")

    variant_metadata = notebook.get("metadata", {}).get("guard640_v4b", {})
    if variant_metadata.get("variant_parent") != "heralces/rogii-guard640-v4":
        raise AssertionError("v4b parent metadata mismatch")
    if variant_metadata.get("model_package", {}).get("diff_p95_disable") != 1.0e9:
        raise AssertionError("v4b p95 metadata mismatch")

    expected_datasets = {
        "phongnguyn23021656/koolbox-offline",
        "fleongg/rogii-claude-models-pub",
        "ravaghi/wellbore-geology-prediction-artifacts",
        "pilkwang/rogii-model-package",
    }
    if set(metadata.get("dataset_sources", [])) != expected_datasets:
        raise AssertionError("kernel metadata dataset sources changed")
    if metadata.get("id") != "heralces/rogii-guard640-v4b":
        raise AssertionError("unexpected Kaggle kernel id")
    if metadata.get("code_file") != NOTEBOOK.name:
        raise AssertionError("kernel metadata code_file mismatch")
    if metadata.get("is_private") is not True or metadata.get("enable_internet") is not False:
        raise AssertionError("kernel must be private with internet disabled")

    print(
        json.dumps(
            {
                "status": "ok",
                "cells": len(cells),
                "code_cells_compiled": len(code_cells),
                "behavioral_diff_from_v4": {"MODEL_PACKAGE_DIFF_P95_DISABLE": [25.0, 1.0e9]},
                "model_package": {
                    "max_weight": runtime["MODEL_PACKAGE_GATED_MAX_WEIGHT"],
                    "scale": runtime["MODEL_PACKAGE_GATED_SCALE"],
                    "analytical_max_abs_move_ft": max_gate_move,
                },
                "unsafe_gates_enabled": 0,
                "dynamic_sample_contract": True,
                "runtime_guards_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
