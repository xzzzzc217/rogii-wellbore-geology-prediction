"""Fail-closed static validator for the unpushed Guard640 v4 notebook."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "rogii-guard640-v4.ipynb"
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


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [(i, cell) for i, cell in enumerate(cells) if cell["cell_type"] == "code"]

    for index, cell in code_cells:
        compile(_source(cell), f"{NOTEBOOK.name}:cell_{index}", "exec")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale execution state")
        if "papermill" in cell.get("metadata", {}) or "execution" in cell.get("metadata", {}):
            raise AssertionError(f"cell {index} contains stale execution metadata")

    control_matches = [
        (index, _source(cell))
        for index, cell in code_cells
        if "SUBMISSION_PROFILE = 'guard640_v4'" in _source(cell)
    ]
    if len(control_matches) != 1:
        raise AssertionError(f"expected one control cell, found {len(control_matches)}")
    _, control = control_matches[0]
    values = _literal_assignments(control)
    runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(control, f"{NOTEBOOK.name}:control", "exec"), runtime)

    expected_literals = {
        "SUBMISSION_PROFILE": "guard640_v4",
        "NOTEBOOK_HARD_BUDGET_SEC": 7.5 * 3600.0,
        "NOTEBOOK_FINAL_RESERVE_SEC": 0.5 * 3600.0,
        "SP45_SELECTOR_N_SEEDS": 96,
        "ALLOW_SAME_ID_TRAIN_LOOKUP": False,
        "RUN_GUARDED_OVERLAP_OVERRIDE": False,
        "RUN_OVERLAP_DRY_RUN_PROBE": False,
        "RUN_EXACT_MATCH_RECOVERY": False,
        "RUN_PUBLIC_OVERLAP_CALIBRATION": False,
        "APPLY_GLOBAL_LB_BIAS": False,
        "RUN_PF_BRANCH_HEDGE": True,
        "RUN_U_CONTINUITY": True,
        "VISIBLE_PREFIX_INCLUDE_PF": True,
        "VISIBLE_PREFIX_CAL_SEEDS": 12,
        "VISIBLE_PREFIX_FINAL_SEEDS": 24,
        "VISIBLE_PREFIX_PARTICLES": 250,
        "VISIBLE_PREFIX_ALPHA_MULTIPLIER": 0.35,
        "PF_BRANCH_HEDGE_CAP": 3.0,
        "U_CONTINUITY_CAP": 8.0,
        "U_CONTINUITY_TAU": 240.0,
        "MODEL_PACKAGE_REQUIRE": True,
        "MODEL_PACKAGE_ALLOW_AUTO_SEARCH": False,
        "MODEL_PACKAGE_DIFF_P95_DISABLE": 25.0,
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
    required_fragments = (
        "alpha * float(globals().get('VISIBLE_PREFIX_ALPHA_MULTIPLIER', 0.35))",
        "swid=wid",
        "swid = wid",
        "self_wid=swid",
        "self_wid=wid",
        "ALLOW_SAME_ID_TRAIN_LOOKUP",
        "_gold_contact_candidate",
        "RUN_PUBLIC_OVERLAP_CALIBRATION",
        "Model package skipped by runtime guard",
        "_BH_CAP = float(globals().get('PF_BRANCH_HEDGE_CAP', 3.0))",
        "_UC_CAP = float(globals().get('U_CONTINUITY_CAP', 8.0))",
        "_UC_TAU = float(globals().get('U_CONTINUITY_TAU', 240.0))",
        "hidden sample_submission.csv",
        "ID-set mismatch",
        "id_order_matches_hidden_sample",
        "submission_sha256",
    )
    for fragment in required_fragments:
        if fragment not in all_source:
            raise AssertionError(f"required source fragment missing: {fragment}")

    forbidden_fragments = (
        "RUN_GUARDED_OVERLAP_OVERRIDE = True",
        "RUN_OVERLAP_DRY_RUN_PROBE = True",
        "ALLOW_SAME_ID_TRAIN_LOOKUP = True",
        "RUN_EXACT_MATCH_RECOVERY = True",
        "RUN_PUBLIC_OVERLAP_CALIBRATION = True",
        "APPLY_GLOBAL_LB_BIAS = True",
        "GLOBAL_TVT_BIAS",
        "_OV_WEIGHT =",
        "submission_before_public_overlap.csv",
        "Public-test overlap calibration",
        "ROGII_PROBE",
        "LB PROBE",
        "canary",
        "expected source prediction",
        "EXPECTED_SOURCE",
        "EXPECTED_PRED",
        "14151",
        "00bbac68",
        "fig_overview",
        "fig_results",
        "_VIZ_READY",
        "Visual integrity manifest",
        "PF_SEEDS = 64",
        "PF_SEEDS = 128",
        "SP45_SELECTOR_N_SEEDS = 64",
        "SP45_SELECTOR_N_SEEDS = 128",
        "alpha * 1.30",
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9",
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 1e9",
    )
    for fragment in forbidden_fragments:
        if fragment in all_source:
            raise AssertionError(f"forbidden source fragment present: {fragment}")

    p95_assignments = re.findall(
        r"^MODEL_PACKAGE_DIFF_P95_DISABLE\s*=\s*([^\n#]+)",
        all_source,
        flags=re.MULTILINE,
    )
    if p95_assignments != ["25.0"]:
        raise AssertionError(
            "expected exactly one model-package p95 disable assignment at 25.0; "
            f"got {p95_assignments}"
        )

    hidden_contract_matches = [
        (index, _source(cell))
        for index, cell in code_cells
        if "# Final hidden-set contract transaction." in _source(cell)
    ]
    if len(hidden_contract_matches) != 1:
        raise AssertionError(
            f"expected one dynamic hidden contract, found {len(hidden_contract_matches)}"
        )
    hidden_contract_index, hidden_contract_source = hidden_contract_matches[0]
    if "'derived_kernel': 'heralces/rogii-guard640-v4'" not in hidden_contract_source:
        raise AssertionError("dynamic hidden contract attribution mismatch")
    if (
        "len(_hc_reread) != len(_hc_sample)" not in hidden_contract_source
        or "_hc_sample_ids" not in hidden_contract_source
    ):
        raise AssertionError("hidden contract is not dynamically sample-derived")
    later_csv_writers = [
        index
        for index, cell in code_cells
        if index > hidden_contract_index and ".to_csv(" in _source(cell)
    ]
    if later_csv_writers:
        raise AssertionError(
            "dynamic hidden contract must be the last CSV writer; "
            f"later writers={later_csv_writers}"
        )
    final_audit_indices = [
        index
        for index, cell in code_cells
        if "# Final immutable scoring audit after the dynamic sample transaction." in _source(cell)
    ]
    if len(final_audit_indices) != 1 or final_audit_indices[0] <= hidden_contract_index:
        raise AssertionError("final immutable audit must follow the dynamic sample writer")

    # Reject literal test-well IDs while allowing dynamic hashes and attribution IDs.
    for index, cell in code_cells:
        tree = ast.parse(_source(cell))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.fullmatch(r"[0-9a-f]{8}", node.value.lower()):
                    raise AssertionError(f"fixed eight-hex well ID in cell {index}: {node.value}")

    if "papermill" in notebook.get("metadata", {}):
        raise AssertionError("notebook contains stale papermill metadata")
    guard_meta = notebook.get("metadata", {}).get("guard640_v4", {})
    if guard_meta.get("derived_from") != "fleongg/rogii-codex-public-6-40-guard":
        raise AssertionError("notebook provenance metadata mismatch")

    expected_datasets = {
        "phongnguyn23021656/koolbox-offline",
        "fleongg/rogii-claude-models-pub",
        "ravaghi/wellbore-geology-prediction-artifacts",
        "pilkwang/rogii-model-package",
    }
    if set(metadata.get("dataset_sources", [])) != expected_datasets:
        raise AssertionError("kernel metadata dataset_sources differ from the audited set")
    if metadata.get("id") != "heralces/rogii-guard640-v4":
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
                "profile": runtime["SUBMISSION_PROFILE"],
                "selector_pf_seeds": runtime["SP45_SELECTOR_N_SEEDS"],
                "learned_pf_seeds": 96,
                "visible_prefix": {
                    "alpha_multiplier": runtime["VISIBLE_PREFIX_ALPHA_MULTIPLIER"],
                    "cal_seeds": runtime["VISIBLE_PREFIX_CAL_SEEDS"],
                    "final_seeds": runtime["VISIBLE_PREFIX_FINAL_SEEDS"],
                    "particles": runtime["VISIBLE_PREFIX_PARTICLES"],
                },
                "model_package": {
                    "max_weight": runtime["MODEL_PACKAGE_GATED_MAX_WEIGHT"],
                    "scale": runtime["MODEL_PACKAGE_GATED_SCALE"],
                },
                "generalized_branch_hedge": runtime["RUN_PF_BRANCH_HEDGE"],
                "u_continuity": runtime["RUN_U_CONTINUITY"],
                "unsafe_gates_enabled": 0,
                "estimated_hidden_runtime_hours": [5.5, 7.2],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
