"""Fail-closed static validator for the local-only Clean128 v5 notebook."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
NOTEBOOK = ROOT / "rogii-pf-clean128-v5.ipynb"
METADATA = ROOT / "kernel-metadata.json"
SAFE_SOURCE = PROJECT / "kaggle_notebooks" / "pf_safe_v3" / "rogii-pf-safe-v3.ipynb"
EXACT_SOURCE = (
    PROJECT / "kaggle_notebooks" / "pf_public_reproduction" / "rogii-pf-public-reproduction.ipynb"
)
BUILDER = PROJECT / "scripts" / "build_pf_clean128_v5.py"

TRANSFORMED_SOURCE_INDICES = {12, 27, 34, 44}
DIRECT_EXACT_INDICES = {
    9,
    10,
    11,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
    31,
    33,
    34,
    37,
    38,
    40,
    46,
}


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def literal_assignments(value: str) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in ast.parse(value).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return assignments


def class_literal(value: str, class_name: str, attribute: str) -> object:
    for node in ast.parse(value).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == attribute:
                return ast.literal_eval(statement.value)
    raise AssertionError(f"{class_name}.{attribute} literal assignment not found")


def without_contact_helper(value: str) -> str:
    start = value.index("def tvt_from_contacts(")
    end = value.index("def load_well(", start)
    return value[:start] + value[end:]


def mapped_cells(cells: list[dict]) -> dict[int, dict]:
    mapped: dict[int, dict] = {}
    for cell in cells:
        index = cell.get("metadata", {}).get("clean128_source_index")
        if index is None:
            continue
        index = int(index)
        if index in mapped:
            raise AssertionError(f"duplicate clean128 source index: {index}")
        mapped[index] = cell
    return mapped


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    safe = json.loads(SAFE_SOURCE.read_text(encoding="utf-8"))
    exact = json.loads(EXACT_SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [(index, cell) for index, cell in enumerate(cells) if cell["cell_type"] == "code"]

    for index, cell in code_cells:
        compile(source(cell), f"{NOTEBOOK.name}:cell_{index}", "exec")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale execution state")

    controls = [
        source(cell)
        for _, cell in code_cells
        if "SUBMISSION_PROFILE = 'clean128_v5'" in source(cell)
    ]
    if len(controls) != 1:
        raise AssertionError(f"expected one Clean128 control cell, got {len(controls)}")
    control = controls[0]
    values = literal_assignments(control)
    runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(control, f"{NOTEBOOK.name}:control", "exec"), runtime)

    expected_values = {
        "SUBMISSION_PROFILE": "clean128_v5",
        "NOTEBOOK_HARD_BUDGET_SEC": 27000.0,
        "NOTEBOOK_FINAL_RESERVE_SEC": 1800.0,
        "LEARNED_STAGE_MIN_BUDGET_SEC": 7200.0,
        "SP45_RIDGE_MODEL_WEIGHT": 0.30,
        "SP45_SELECTOR_WEIGHT": 0.70,
        "SP45_SELECTOR_N_PARTICLES": 500,
        "SP45_SELECTOR_N_SEEDS": 128,
        "SP45_PROJECTION_DEGREE": 3,
        "SP45_PROJECTION_BLEND_WEIGHT": 0.75,
        "SP45_BLEND_WEIGHT": 0.60,
        "SP45_BLEND_CANDIDATE_WEIGHTS": (0.60,),
    }
    for name, expected in expected_values.items():
        if values.get(name) != expected:
            raise AssertionError(f"{name}: expected {expected!r}, got {values.get(name)!r}")

    prohibited_controls = (
        "ALLOW_SAME_ID_TRAIN_LOOKUP",
        "RUN_GUARDED_OVERLAP_OVERRIDE",
        "RUN_OVERLAP_DRY_RUN_PROBE",
        "RUN_EXACT_MATCH_RECOVERY",
        "RUN_PUBLIC_OVERLAP_CALIBRATION",
        "RUN_VISIBLE_PREFIX_CALIBRATION",
        "RUN_MODEL_PACKAGE_CORRECTION",
        "RUN_PF_BRANCH_HEDGE",
        "RUN_U_CONTINUITY",
        "APPLY_GLOBAL_LB_BIAS",
        "MODEL_PACKAGE_ALLOW_AUTO_SEARCH",
        "MODEL_PACKAGE_REQUIRE",
    )
    for name in prohibited_controls:
        if values.get(name) is not False or runtime.get(name) is not False:
            raise AssertionError(f"prohibited control is not immutable false: {name}")
    if runtime.get("SELECTOR_PF_SEEDS") != 128:
        raise AssertionError("selector runtime seed count is not 128")

    mapped = mapped_cells(cells)
    expected_indices = {
        9,
        10,
        11,
        12,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        33,
        34,
        37,
        38,
        40,
        41,
        43,
        44,
        46,
    }
    if set(mapped) != expected_indices:
        raise AssertionError(
            f"unexpected source-cell map: missing={sorted(expected_indices - set(mapped))}, "
            f"extra={sorted(set(mapped) - expected_indices)}"
        )

    # Every untransformed core cell remains byte-identical to the audited Safe
    # v3 source. The majority also remains byte-identical to exact public 6.451.
    for index, cell in mapped.items():
        if index not in TRANSFORMED_SOURCE_INDICES:
            if source(cell) != source(safe["cells"][index]):
                raise AssertionError(f"unapproved Safe-v3 source drift at source cell {index}")
    for index in DIRECT_EXACT_INDICES:
        if source(mapped[index]) != source(exact["cells"][index]):
            raise AssertionError(f"exact-6.451 core drift at source cell {index}")

    # The only functional differences in these two exact cells are the required
    # query-well exclusions. A comment-only delta in PP has identical AST.
    for index, marker in ((15, "swid=wid"), (41, "swid = wid")):
        if marker not in source(mapped[index]):
            raise AssertionError(f"query-well exclusion missing at source cell {index}")
    if ast.dump(ast.parse(source(mapped[43]))) != ast.dump(ast.parse(source(exact["cells"][43]))):
        raise AssertionError("learned trajectory PP math differs from exact 6.451")
    expected_no_contact = without_contact_helper(source(exact["cells"][12]))
    if source(mapped[12]) != expected_no_contact:
        raise AssertionError("selector definitions differ beyond removing the contact helper")

    learned_cfg = next(
        source(cell)
        for cell in mapped.values()
        if "class CFG:" in source(cell) and "PF_SEEDS" in source(cell)
    )
    if class_literal(learned_cfg, "CFG", "PF_SEEDS") != 128:
        raise AssertionError("learned trajectory PF seed count is not 128")

    all_code = "\n".join(source(cell) for _, cell in code_cells)
    if "tvt_from_contacts" in all_code or "load_well(wid, 'train')" in all_code:
        raise AssertionError("same-ID/contact scoring path is still callable")
    if "if wid in train_wells" in all_code or "tvt_phys" in all_code:
        raise AssertionError("same-ID physical substitution remains in selector scoring")
    if "swid=wid if is_train else None" in all_code:
        raise AssertionError("compact learned feature path can retain the query well")
    if "swid = wid if is_train else None" in all_code:
        raise AssertionError("learned feature path can retain the query well")
    if all_code.count("self_wid=swid") < 8:
        raise AssertionError("not all train-derived spatial imputer calls carry self_wid")
    if "_find_precomputed_learned_submission" in all_code:
        raise AssertionError("ID-matched precomputed learned submission path is present")
    if "train_stack(" in all_code:
        raise AssertionError("unbounded train-from-scratch scoring path is present")

    required_clean_fragments = (
        "n_seeds=int(globals().get('SELECTOR_PF_SEEDS', SP45_SELECTOR_N_SEEDS))",
        "SP45_PROJECTION_DEGREE",
        "SP45_PROJECTION_BLEND_WEIGHT",
        "build_features(test_wids, 'test', is_train=False)",
        "meta_test = np.mean([model.predict(xt) for model in models], axis=0)",
        "test_pred = make_prediction(test_df, meta_test, None)",
        "float(w_sp45) * merged['tvt_sp45']",
        "+ w_learned * merged['tvt_learned']",
        "Learned stage skipped by 7.5h budget guard",
        "restoring audited SP45 projection",
        "budget_last_visible_fallback",
    )
    for fragment in required_clean_fragments:
        if fragment not in all_code:
            raise AssertionError(f"required clean scoring fragment missing: {fragment}")

    forbidden_implementation_fragments = (
        "_gold_calibrate_well",
        "_gold_profile_output",
        "submission_gold_prefix",
        "MODEL_PACKAGE_GATED_MAX_WEIGHT",
        "submission_before_model_package",
        "model_package_correction_report",
        "_BH_STRENGTH",
        "_BH_CAP",
        "submission_before_branch_hedge",
        "_UC_CAP",
        "_UC_TAU",
        "u_continuity_report",
        "_OV_WEIGHT",
        "submission_before_public_overlap",
        "GLOBAL_TVT_BIAS",
        "ROGII_PROBE",
        "EXPECTED_SOURCE",
        "EXPECTED_PRED",
    )
    for fragment in forbidden_implementation_fragments:
        if fragment in all_code:
            raise AssertionError(f"forbidden correction implementation present: {fragment}")

    for index, cell in code_cells:
        for node in ast.walk(ast.parse(source(cell))):
            if isinstance(node, ast.Constant) and node.value == 14151:
                raise AssertionError(f"fixed visible row count in cell {index}")
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if re.fullmatch(r"[0-9a-fA-F]{8}", node.value):
                raise AssertionError(f"fixed eight-hex well ID in cell {index}: {node.value}")
            if re.fullmatch(r"[0-9a-fA-F]{64}", node.value):
                raise AssertionError(f"fixed prediction/source SHA in cell {index}")

    contracts = [
        (index, source(cell))
        for index, cell in code_cells
        if "# Final hidden-set contract transaction." in source(cell)
    ]
    if len(contracts) != 1:
        raise AssertionError(f"expected one dynamic hidden contract, got {len(contracts)}")
    contract_index, contract = contracts[0]
    if "'derived_kernel': 'heralces/rogii-pf-clean128-v5'" not in contract:
        raise AssertionError("hidden contract attribution mismatch")
    if (
        "len(_hc_reread) != len(_hc_sample)" not in contract
        or "_hc_sample_ids" not in contract
        or "_hc_missing" not in contract
        or "_hc_extra" not in contract
    ):
        raise AssertionError("hidden contract is not dynamically sample-derived")
    later_csv_writers = [
        index for index, cell in code_cells if index > contract_index and ".to_csv(" in source(cell)
    ]
    if later_csv_writers:
        raise AssertionError(f"dynamic contract is not the last CSV writer: {later_csv_writers}")

    variant = notebook.get("metadata", {}).get("clean128_v5", {})
    expected_variant = {
        "selector_pf_seeds": 128,
        "learned_pf_seeds": 128,
        "sp45_weight": 0.60,
        "learned_weight": 0.40,
        "query_well_self_exclusion": True,
        "late_correction_cells_present": False,
        "hard_budget_sec": 27000.0,
        "final_reserve_sec": 1800.0,
        "learned_stage_min_budget_sec": 7200.0,
    }
    for name, expected in expected_variant.items():
        if variant.get(name) != expected:
            raise AssertionError(f"notebook metadata {name} mismatch")
    if variant.get("projection") != {"degree": 3, "blend_weight": 0.75}:
        raise AssertionError("projection metadata mismatch")

    expected_datasets = {
        "phongnguyn23021656/koolbox-offline",
        "fleongg/rogii-claude-models-pub",
        "ravaghi/wellbore-geology-prediction-artifacts",
    }
    if set(metadata.get("dataset_sources", [])) != expected_datasets:
        raise AssertionError("kernel metadata dataset sources changed")
    if metadata.get("id") != "heralces/rogii-pf-clean128-v5":
        raise AssertionError("unexpected Kaggle kernel id")
    if metadata.get("code_file") != NOTEBOOK.name:
        raise AssertionError("kernel metadata code_file mismatch")
    if metadata.get("is_private") is not True or metadata.get("enable_internet") is not False:
        raise AssertionError("kernel must be private with internet disabled")
    if not BUILDER.exists():
        raise AssertionError("reproducible builder is missing")

    notebook_sha = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "ok",
                "cells": len(cells),
                "code_cells_compiled": len(code_cells),
                "notebook_sha256": notebook_sha,
                "exact_clean_configuration": {
                    "selector_pf_seeds": 128,
                    "learned_pf_seeds": 128,
                    "sp45_weight": 0.60,
                    "learned_weight": 0.40,
                    "projection_degree": 3,
                    "projection_blend_weight": 0.75,
                },
                "query_well_self_exclusion": True,
                "late_correction_implementation_cells": 0,
                "unsafe_gates_enabled": 0,
                "dynamic_sample_contract": True,
                "contract_is_last_csv_writer": True,
                "budget": {
                    "hard_hours": 7.5,
                    "final_reserve_hours": 0.5,
                    "learned_stage_reserve_hours": 2.0,
                    "sp45_fallback": True,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
