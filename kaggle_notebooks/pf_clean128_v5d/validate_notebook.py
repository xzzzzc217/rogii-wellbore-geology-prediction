"""Fail-closed validator for deterministic Clean128 v5d."""

from __future__ import annotations

import ast
import contextlib
import difflib
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
NOTEBOOK = ROOT / "rogii-pf-clean128-v5d.ipynb"
METADATA = ROOT / "kernel-metadata.json"
PARENT = PROJECT / "kaggle_notebooks" / "pf_clean128_v5" / "rogii-pf-clean128-v5.ipynb"
BUILDER = PROJECT / "scripts" / "build_pf_clean128_v5d.py"
TRAIN_DIR = PROJECT / "data" / "competition" / "train"

TOUCHED_SOURCE_INDICES = {15, 37, 38, 41}
SCORING_BRANCHES = (
    "sp45_train_ancc",
    "sp45_train_z",
    "sp45_test_ancc",
    "sp45_test_z",
    "learned_train_ancc",
    "learned_train_z",
    "learned_test_ancc",
    "learned_test_z",
)


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


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def is_njit(function: ast.FunctionDef) -> bool:
    for decorator in function.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if dotted_name(target) == "njit":
            return True
    return False


def random_calls(node: ast.AST) -> list[ast.Call]:
    calls = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = dotted_name(child.func)
        if name and name.startswith("np.random."):
            calls.append(child)
    return calls


def first_statement_is_seed(function: ast.FunctionDef, argument: str) -> bool:
    if not function.body:
        return False
    statement = function.body[0]
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    return (
        dotted_name(call.func) == "np.random.seed"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == argument
    )


def stable_seed_independent(well_id: str, branch: str) -> int:
    payload = f"rogii-pf-clean128-v5d\0{well_id}\0{branch}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % 2147483646 + 1


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [(index, cell) for index, cell in enumerate(cells) if cell["cell_type"] == "code"]

    for index, cell in code_cells:
        compile(source(cell), f"{NOTEBOOK.name}:cell_{index}", "exec")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale execution state")

    seed_cells = [
        (index, cell)
        for index, cell in code_cells
        if cell.get("metadata", {}).get("clean128_stage") == "deterministic_seed_control"
    ]
    if len(seed_cells) != 1:
        raise AssertionError(f"expected one stable-seed control cell, got {len(seed_cells)}")
    _, seed_cell = seed_cells[0]
    seed_source = source(seed_cell)
    if "hash(" in seed_source or "_pf_seed_hashlib.sha256(payload)" not in seed_source:
        raise AssertionError("seed derivation is not based solely on stable SHA-256")
    seed_runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(seed_source, f"{NOTEBOOK.name}:seed_control", "exec"), seed_runtime)
    stable_seed = seed_runtime["_stable_pf_seed"]
    if seed_runtime.get("PF_DETERMINISTIC_SEEDING") is not True:
        raise AssertionError("deterministic PF control is not enabled")
    if seed_runtime.get("PF_SEED_NAMESPACE") != "rogii-pf-clean128-v5d":
        raise AssertionError("unexpected PF seed namespace")

    well_ids = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in TRAIN_DIR.glob("*__horizontal_well.csv")
    )
    if len(well_ids) != 773:
        raise AssertionError(f"expected 773 mounted train wells, got {len(well_ids)}")
    seed_map = {
        (well_id, branch): int(stable_seed(well_id, branch))
        for well_id in well_ids
        for branch in SCORING_BRANCHES
    }
    if any(seed < 1 or seed > 2147483646 for seed in seed_map.values()):
        raise AssertionError("stable PF seed is outside the signed-31-bit range")
    if len(set(seed_map.values())) != len(seed_map):
        duplicates = [seed for seed, count in Counter(seed_map.values()).items() if count > 1]
        raise AssertionError(f"stable PF seed collision on mounted wells: {duplicates[:5]}")
    for key, seed in seed_map.items():
        if seed != stable_seed_independent(*key):
            raise AssertionError(f"seed does not match independent SHA-256 derivation: {key}")
    reverse_map = {key: int(stable_seed(*key)) for key in reversed(list(seed_map))}
    if reverse_map != seed_map:
        raise AssertionError("seed mapping depends on traversal order")

    controls = [
        source(cell)
        for _, cell in code_cells
        if "SUBMISSION_PROFILE = 'clean128_v5d'" in source(cell)
    ]
    if len(controls) != 1:
        raise AssertionError(f"expected one v5d control cell, got {len(controls)}")
    control = controls[0]
    parent_control = next(
        source(cell)
        for cell in parent["cells"]
        if "SUBMISSION_PROFILE = 'clean128_v5'" in source(cell)
    )
    if control.replace("clean128_v5d", "clean128_v5") != parent_control:
        raise AssertionError("v5d changed a clean control other than profile identity")
    control_values = literal_assignments(control)
    expected_clean = {
        "SP45_SELECTOR_N_SEEDS": 128,
        "SP45_BLEND_WEIGHT": 0.60,
        "SP45_BLEND_CANDIDATE_WEIGHTS": (0.60,),
        "SP45_PROJECTION_DEGREE": 3,
        "SP45_PROJECTION_BLEND_WEIGHT": 0.75,
        "NOTEBOOK_HARD_BUDGET_SEC": 27000.0,
        "NOTEBOOK_FINAL_RESERVE_SEC": 1800.0,
        "LEARNED_STAGE_MIN_BUDGET_SEC": 7200.0,
    }
    for name, expected in expected_clean.items():
        if control_values.get(name) != expected:
            raise AssertionError(f"clean configuration changed: {name}")
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
    )
    for name in prohibited_controls:
        if control_values.get(name) is not False:
            raise AssertionError(f"prohibited clean control changed: {name}")

    # Remove the one inserted helper and prove every unrelated parent cell is
    # byte-identical. The remaining differences are identity/audit text plus
    # four explicitly marked deterministic-repair source cells.
    child_without_helper = [
        cell
        for cell in cells
        if cell.get("metadata", {}).get("clean128_stage") != "deterministic_seed_control"
    ]
    if len(child_without_helper) != len(parent["cells"]):
        raise AssertionError("v5d cell count differs from v5 beyond the seed helper")
    changed_pairs: list[tuple[dict, dict]] = []
    for parent_cell, child_cell in zip(parent["cells"], child_without_helper, strict=True):
        if parent_cell.get("cell_type") != child_cell.get("cell_type"):
            raise AssertionError("v5d changed a parent cell type")
        if source(parent_cell) != source(child_cell):
            changed_pairs.append((parent_cell, child_cell))
    changed_source_indices = {
        int(child.get("metadata", {}).get("clean128_source_index"))
        for _, child in changed_pairs
        if child.get("metadata", {}).get("clean128_source_index") is not None
    }
    if changed_source_indices != TOUCHED_SOURCE_INDICES:
        raise AssertionError(
            f"unexpected scoring-source changes from v5: {sorted(changed_source_indices)}"
        )
    if len(changed_pairs) != 10:
        raise AssertionError(f"unexpected total changed parent cells: {len(changed_pairs)}")

    # A line-level allowlist makes accidental mathematical edits in the four
    # touched core cells fail even if their metadata is forged.
    delta_tokens = (
        "seed",
        "noise",
        "_resamp",
        "_pf_ancc",
        "_pf_z",
        "run_pf_ancc",
        "run_pf_z",
        "np.random",
        "np.linspace",
        "np2",
        "PF_RESAMP",
        "RESAMP",
        "pf_a",
        "pf_z",
        "_stable_pf_seed",
        "diagnostic_ancc",
        "for k in range(N):",
    )
    for parent_cell, child_cell in changed_pairs:
        source_index = child_cell.get("metadata", {}).get("clean128_source_index")
        if source_index not in TOUCHED_SOURCE_INDICES:
            continue
        delta = difflib.ndiff(
            source(parent_cell).splitlines(),
            source(child_cell).splitlines(),
        )
        unexpected = [
            line
            for line in delta
            if line[:2] in {"+ ", "- "}
            and not any(token in line for token in delta_tokens)
            and line[2:].strip() != ")"
        ]
        if unexpected:
            raise AssertionError(
                f"non-seeding source delta in source cell {source_index}: {unexpected[:3]}"
            )

    njit_random_functions: list[tuple[int, ast.FunctionDef, list[ast.Call]]] = []
    all_random_calls: list[tuple[int, ast.Call, str]] = []
    njit_ranges: dict[int, list[tuple[int, int]]] = {}
    resamplers: list[tuple[int, ast.FunctionDef]] = []
    for cell_index, cell in code_cells:
        tree = ast.parse(source(cell))
        cell_njit_ranges = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "_resamp":
                resamplers.append((cell_index, node))
            calls = random_calls(node)
            if is_njit(node) and calls:
                njit_random_functions.append((cell_index, node, calls))
                cell_njit_ranges.append((node.lineno, node.end_lineno or node.lineno))
        njit_ranges[cell_index] = cell_njit_ranges
        for call in random_calls(tree):
            name = dotted_name(call.func)
            assert name is not None
            all_random_calls.append((cell_index, call, name))

    name_counts = Counter(function.name for _, function, _ in njit_random_functions)
    if name_counts != Counter({"_pf_ancc": 2, "_pf_z": 2, "_pf_lik_allseeds": 1}):
        raise AssertionError(f"unexpected random Numba kernels: {dict(name_counts)}")
    if len(resamplers) != 2:
        raise AssertionError(f"expected two deterministic resamplers, got {len(resamplers)}")
    for cell_index, function in resamplers:
        if random_calls(function):
            raise AssertionError(f"_resamp still owns implicit RNG in cell {cell_index}")
        argument_names = [argument.arg for argument in function.args.args]
        if argument_names[-3:] != ["u0", "pos_noise", "aux_noise"]:
            raise AssertionError("_resamp does not consume parent-owned random draws")

    for cell_index, function, calls in njit_random_functions:
        if function.name in {"_pf_ancc", "_pf_z"}:
            if function.args.args[-1].arg != "seed":
                raise AssertionError(f"{function.name} lacks an explicit seed parameter")
            if not first_statement_is_seed(function, "seed"):
                raise AssertionError(
                    f"{function.name} in cell {cell_index} draws before explicit seeding"
                )
        elif function.name == "_pf_lik_allseeds":
            seed_loop = next(
                (
                    node
                    for node in function.body
                    if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "s"
                ),
                None,
            )
            if seed_loop is None or not seed_loop.body:
                raise AssertionError("likelihood PF seed loop missing")
            first = seed_loop.body[0]
            if (
                not isinstance(first, ast.Expr)
                or not isinstance(first.value, ast.Call)
                or dotted_name(first.value.func) != "np.random.seed"
            ):
                raise AssertionError("likelihood PF random loop is not seeded first")
            loop_start = seed_loop.lineno
            loop_end = seed_loop.end_lineno or loop_start
            if any(not (loop_start <= call.lineno <= loop_end) for call in calls):
                raise AssertionError("likelihood PF has a random draw outside its seeded loop")

    # Outside Numba, only selector's explicitly seeded Generator construction
    # may touch np.random.
    for cell_index, call, name in all_random_calls:
        inside_njit = any(start <= call.lineno <= end for start, end in njit_ranges[cell_index])
        if not inside_njit and name != "np.random.default_rng":
            raise AssertionError(
                f"unseeded np.random call outside a seeded Numba kernel: "
                f"cell={cell_index}, line={call.lineno}, call={name}"
            )
        if name == "np.random.default_rng":
            if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
                raise AssertionError("selector default_rng does not receive its explicit seed")

    all_code = "\n".join(source(cell) for _, cell in code_cells)
    if "hash(" in all_code:
        raise AssertionError("process-randomized Python hash() is present")
    expected_seed_fragments = (
        "seed_branch='sp45_train' if is_train else 'sp45_test'",
        "_stable_pf_seed(wid,seed_branch+'_ancc')",
        "_stable_pf_seed(wid,seed_branch+'_z')",
        "seed_branch = 'learned_train' if is_train else 'learned_test'",
        "_stable_pf_seed(wid, seed_branch + '_ancc')",
        "_stable_pf_seed(wid, seed_branch + '_z')",
    )
    for fragment in expected_seed_fragments:
        if fragment not in all_code:
            raise AssertionError(f"per-well scoring seed fragment missing: {fragment}")

    # Every wrapper call has an explicit seed; every direct kernel call either
    # forwards that seed or is a fixed JIT warm-up.
    wrapper_calls = []
    kernel_calls = []
    resample_calls = []
    for cell_index, cell in code_cells:
        tree = ast.parse(source(cell))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted_name(node.func)
            if name in {"run_pf_ancc", "run_pf_z"}:
                wrapper_calls.append((cell_index, name, node))
            elif name in {"_pf_ancc", "_pf_z"}:
                kernel_calls.append((cell_index, name, node))
            elif name == "_resamp":
                resample_calls.append((cell_index, node))
            elif name == "hash":
                raise AssertionError(f"Python hash() call in cell {cell_index}")
    if len(wrapper_calls) != 5:
        raise AssertionError(f"unexpected single-PF wrapper call count: {len(wrapper_calls)}")
    for cell_index, name, call in wrapper_calls:
        seed_keywords = [keyword for keyword in call.keywords if keyword.arg == "seed"]
        if len(seed_keywords) != 1:
            raise AssertionError(f"{name} call lacks explicit seed in cell {cell_index}")
    if Counter(name for _, name, _ in kernel_calls) != Counter({"_pf_ancc": 4, "_pf_z": 4}):
        raise AssertionError("unexpected direct single-PF kernel call graph")
    for cell_index, name, call in kernel_calls:
        if not call.args:
            raise AssertionError(f"{name} call has no arguments in cell {cell_index}")
        last = call.args[-1]
        if not (
            (isinstance(last, ast.Name) and last.id == "seed") or isinstance(last, ast.Constant)
        ):
            raise AssertionError(f"{name} call does not forward/fix a seed in cell {cell_index}")
    if len(resample_calls) != 4 or any(len(call.args) != 9 for _, call in resample_calls):
        raise AssertionError("resampler call graph does not pass all parent-owned draws")

    forbidden_corrections = (
        "_gold_calibrate_well",
        "MODEL_PACKAGE_GATED_MAX_WEIGHT",
        "_BH_STRENGTH",
        "_UC_CAP",
        "submission_before_public_overlap",
        "GLOBAL_TVT_BIAS",
    )
    for fragment in forbidden_corrections:
        if fragment in all_code:
            raise AssertionError(f"forbidden clean correction appeared: {fragment}")

    contracts = [
        (index, source(cell))
        for index, cell in code_cells
        if "# Final hidden-set contract transaction." in source(cell)
    ]
    if len(contracts) != 1:
        raise AssertionError(f"expected one dynamic contract, got {len(contracts)}")
    contract_index, contract = contracts[0]
    if "'derived_kernel': 'heralces/rogii-pf-clean128-v5d'" not in contract:
        raise AssertionError("v5d contract attribution mismatch")
    later_writers = [
        index for index, cell in code_cells if index > contract_index and ".to_csv(" in source(cell)
    ]
    if later_writers:
        raise AssertionError(f"dynamic contract is not the last CSV writer: {later_writers}")

    variant = notebook.get("metadata", {}).get("clean128_v5d", {})
    if variant.get("variant_parent") != "heralces/rogii-pf-clean128-v5":
        raise AssertionError("v5d parent metadata mismatch")
    if variant.get("behavioral_delta") != "deterministic single-PF seeding only":
        raise AssertionError("v5d behavioral-delta metadata mismatch")
    if set(variant.get("seeded_single_pf_branches", [])) != set(SCORING_BRANCHES):
        raise AssertionError("v5d seeded-branch metadata mismatch")
    for name, expected in {
        "selector_pf_seeds": 128,
        "learned_pf_seeds": 128,
        "sp45_weight": 0.60,
        "learned_weight": 0.40,
        "late_correction_cells_present": False,
        "hard_budget_sec": 27000.0,
    }.items():
        if variant.get(name) != expected:
            raise AssertionError(f"v5d clean metadata changed: {name}")

    expected_datasets = {
        "phongnguyn23021656/koolbox-offline",
        "fleongg/rogii-claude-models-pub",
        "ravaghi/wellbore-geology-prediction-artifacts",
    }
    if set(metadata.get("dataset_sources", [])) != expected_datasets:
        raise AssertionError("v5d kernel dataset sources changed")
    if metadata.get("id") != "heralces/rogii-pf-clean128-v5d":
        raise AssertionError("unexpected v5d Kaggle kernel id")
    if metadata.get("code_file") != NOTEBOOK.name:
        raise AssertionError("v5d kernel code_file mismatch")
    if metadata.get("is_private") is not True or metadata.get("enable_internet") is not False:
        raise AssertionError("v5d kernel must be private with internet disabled")
    if not BUILDER.exists():
        raise AssertionError("v5d reproducible builder is missing")

    notebook_sha = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "ok",
                "cells": len(cells),
                "code_cells_compiled": len(code_cells),
                "notebook_sha256": notebook_sha,
                "parent": "heralces/rogii-pf-clean128-v5",
                "behavioral_delta": "deterministic single-PF seeding only",
                "random_numba_kernels": dict(name_counts),
                "resamplers_with_internal_rng": 0,
                "single_pf_wrapper_calls_explicitly_seeded": len(wrapper_calls),
                "direct_single_pf_kernel_calls_seeded": len(kernel_calls),
                "mounted_seed_pairs_audited": len(seed_map),
                "mounted_seed_collisions": 0,
                "python_hash_calls": 0,
                "thread_order_independent_seed_mapping": True,
                "clean128_configuration_unchanged": True,
                "unsafe_gates_enabled": 0,
                "dynamic_sample_contract": True,
                "contract_is_last_csv_writer": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
