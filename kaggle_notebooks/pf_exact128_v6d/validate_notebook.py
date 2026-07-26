"""Fail-closed validator for local-only deterministic Exact128 v6d."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
NOTEBOOK = ROOT / "rogii-exact128-v6d.ipynb"
METADATA = ROOT / "kernel-metadata.json"
PARENT = (
    PROJECT / "kaggle_notebooks" / "pf_public_reproduction" / "rogii-pf-public-reproduction.ipynb"
)
BUILDER = PROJECT / "scripts" / "build_pf_exact128_v6d.py"
TRAIN_DIR = PROJECT / "data" / "competition" / "train"

EXPECTED_RETAINED_INDICES = {
    7,
    8,
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
    50,
    51,
    52,
    53,
}
EXPECTED_DELTA_CATEGORIES = {
    7: "immutable_control_and_safety",
    8: "runtime_budget_bridge",
    12: "remove_contact_helper",
    15: "query_self_exclusion_and_deterministic_rng",
    27: "remove_same_id_selector_and_add_budget_fallback",
    37: "deterministic_rng",
    38: "deterministic_warmup",
    41: "query_self_exclusion_and_deterministic_rng",
    44: "strict_inference_and_budget_fallback",
    51: "remove_contact_candidates_self_exclude_and_budget_guard",
    52: "model_package_runtime_guard",
    53: "immutable_generalized_hedge_gate",
}
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
    payload = f"rogii-exact128-v6d\0{well_id}\0{branch}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % 2147483646 + 1


def parent_cell(child_cells: list[dict], index: int) -> dict:
    matches = [
        cell
        for cell in child_cells
        if cell.get("metadata", {}).get("exact128_parent_index") == index
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one retained parent cell {index}, got {len(matches)}")
    return matches[0]


def assert_control(control: str) -> dict[str, object]:
    runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(control, f"{NOTEBOOK.name}:control", "exec"), runtime)
    expected = {
        "SUBMISSION_PROFILE": "exact128_v6d",
        "SP45_SELECTOR_N_SEEDS": 128,
        "SELECTOR_PF_SEEDS": 128,
        "SP45_SELECTOR_N_PARTICLES": 500,
        "SP45_PROJECTION_DEGREE": 3,
        "SP45_PROJECTION_BLEND_WEIGHT": 0.75,
        "SP45_BLEND_WEIGHT": 0.60,
        "VISIBLE_PREFIX_PROFILE": "balanced",
        "VISIBLE_PREFIX_FINAL_SELECTION": "profile",
        "VISIBLE_PREFIX_CUT_FRACS": (0.50, 0.65, 0.75),
        "VISIBLE_PREFIX_CAL_SEEDS": 24,
        "VISIBLE_PREFIX_FINAL_SEEDS": 48,
        "VISIBLE_PREFIX_PARTICLES": 350,
        "RUN_VISIBLE_PREFIX_CALIBRATION": True,
        "RUN_MODEL_PACKAGE_CORRECTION": True,
        "MODEL_PACKAGE_GATED_MAX_WEIGHT": 0.00425,
        "MODEL_PACKAGE_GATED_SCALE": 6.0,
        "MODEL_PACKAGE_DIFF_P95_DISABLE": 25.0,
        "RUN_PF_BRANCH_HEDGE": True,
        "RUN_U_CONTINUITY": False,
        "NOTEBOOK_HARD_BUDGET_SEC": 27000.0,
        "NOTEBOOK_FINAL_RESERVE_SEC": 1800.0,
        "LEARNED_STAGE_MIN_BUDGET_SEC": 7200.0,
        "MODEL_PACKAGE_MIN_BUDGET_SEC": 7200.0,
    }
    for name, value in expected.items():
        if runtime.get(name) != value:
            raise AssertionError(f"retained configuration mismatch: {name}")
    unsafe = (
        "ALLOW_SAME_ID_TRAIN_LOOKUP",
        "RUN_GUARDED_OVERLAP_OVERRIDE",
        "RUN_OVERLAP_DRY_RUN_PROBE",
        "RUN_EXACT_MATCH_RECOVERY",
        "RUN_PUBLIC_OVERLAP_CALIBRATION",
        "APPLY_GLOBAL_LB_BIAS",
        "MODEL_PACKAGE_ALLOW_AUTO_SEARCH",
    )
    for name in unsafe:
        if runtime.get(name) is not False:
            raise AssertionError(f"unsafe control is not immutable-off: {name}")
    return runtime


def main() -> None:
    notebook_bytes_before = NOTEBOOK.read_bytes()
    notebook_sha_before = hashlib.sha256(notebook_bytes_before).hexdigest()
    notebook = json.loads(notebook_bytes_before)
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [(index, cell) for index, cell in enumerate(cells) if cell["cell_type"] == "code"]

    for index, cell in code_cells:
        compile(source(cell), f"{NOTEBOOK.name}:cell_{index}", "exec")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale execution state")

    retained_indices = {
        int(cell.get("metadata", {}).get("exact128_parent_index"))
        for cell in cells
        if cell.get("metadata", {}).get("exact128_parent_index") is not None
    }
    if retained_indices != EXPECTED_RETAINED_INDICES:
        raise AssertionError(f"retained parent cells changed: {sorted(retained_indices)}")
    changed_indices = {
        index
        for index in EXPECTED_RETAINED_INDICES
        if source(parent_cell(cells, index)) != source(parent["cells"][index])
    }
    if changed_indices != set(EXPECTED_DELTA_CATEGORIES):
        raise AssertionError(f"unexpected parent source deltas: {sorted(changed_indices)}")
    for index, category in EXPECTED_DELTA_CATEGORIES.items():
        actual = parent_cell(cells, index).get("metadata", {}).get("exact128_delta")
        if actual != category:
            raise AssertionError(f"delta category mismatch for parent cell {index}")
    for index in EXPECTED_RETAINED_INDICES - set(EXPECTED_DELTA_CATEGORIES):
        if source(parent_cell(cells, index)) != source(parent["cells"][index]):
            raise AssertionError(f"unrelated parent scoring cell {index} is not byte-identical")

    control = source(parent_cell(cells, 7))
    assert_control(control)

    seed_cells = [
        cell
        for cell in cells
        if cell.get("metadata", {}).get("exact128_stage") == "deterministic_seed_control"
    ]
    if len(seed_cells) != 1:
        raise AssertionError(f"expected one deterministic seed helper, got {len(seed_cells)}")
    seed_source = source(seed_cells[0])
    if "hash(" in seed_source or "_pf_seed_hashlib.sha256(payload)" not in seed_source:
        raise AssertionError("seed helper is not solely stable SHA-256")
    seed_runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(seed_source, f"{NOTEBOOK.name}:seed", "exec"), seed_runtime)
    if seed_runtime.get("PF_SEED_NAMESPACE") != "rogii-exact128-v6d":
        raise AssertionError("unexpected deterministic seed namespace")
    if seed_runtime.get("PF_DETERMINISTIC_SEEDING") is not True:
        raise AssertionError("deterministic PF seeding is not enabled")
    stable_seed = seed_runtime["_stable_pf_seed"]

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
    if len(set(seed_map.values())) != len(seed_map):
        raise AssertionError("stable PF seed collision on mounted wells")
    if any(seed < 1 or seed > 2147483646 for seed in seed_map.values()):
        raise AssertionError("stable PF seed outside signed-31-bit range")
    for key, seed in seed_map.items():
        if seed != stable_seed_independent(*key):
            raise AssertionError(f"seed differs from independent derivation: {key}")
    if {key: int(stable_seed(*key)) for key in reversed(list(seed_map))} != seed_map:
        raise AssertionError("stable seed mapping depends on traversal order")

    njit_random_functions: list[tuple[int, ast.FunctionDef, list[ast.Call]]] = []
    njit_ranges: dict[int, list[tuple[int, int]]] = {}
    all_random_calls: list[tuple[int, ast.Call, str]] = []
    resamplers: list[tuple[int, ast.FunctionDef]] = []
    for cell_index, cell in code_cells:
        tree = ast.parse(source(cell))
        ranges = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "_resamp":
                resamplers.append((cell_index, node))
            calls = random_calls(node)
            if is_njit(node) and calls:
                njit_random_functions.append((cell_index, node, calls))
                ranges.append((node.lineno, node.end_lineno or node.lineno))
        njit_ranges[cell_index] = ranges
        for call in random_calls(tree):
            name = dotted_name(call.func)
            assert name is not None
            all_random_calls.append((cell_index, call, name))

    name_counts = Counter(function.name for _, function, _ in njit_random_functions)
    expected_random = Counter({"_pf_ancc": 2, "_pf_z": 2, "_pf_lik_allseeds": 1})
    if name_counts != expected_random:
        raise AssertionError(f"unexpected random Numba kernels: {dict(name_counts)}")
    if len(resamplers) != 2:
        raise AssertionError(f"expected two resamplers, got {len(resamplers)}")
    for cell_index, function in resamplers:
        if random_calls(function):
            raise AssertionError(f"_resamp still owns implicit RNG in cell {cell_index}")
        args = [argument.arg for argument in function.args.args]
        if args[-3:] != ["u0", "pos_noise", "aux_noise"]:
            raise AssertionError("_resamp lacks parent-owned random draw arguments")

    for cell_index, function, calls in njit_random_functions:
        if function.name in {"_pf_ancc", "_pf_z"}:
            if function.args.args[-1].arg != "seed":
                raise AssertionError(f"{function.name} lacks explicit seed parameter")
            if not first_statement_is_seed(function, "seed"):
                raise AssertionError(f"{function.name} in cell {cell_index} draws before seeding")
        elif function.name == "_pf_lik_allseeds":
            loop = next(
                (
                    node
                    for node in function.body
                    if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "s"
                ),
                None,
            )
            if loop is None or not loop.body:
                raise AssertionError("likelihood PF seeded loop missing")
            first = loop.body[0]
            if (
                not isinstance(first, ast.Expr)
                or not isinstance(first.value, ast.Call)
                or dotted_name(first.value.func) != "np.random.seed"
            ):
                raise AssertionError("likelihood PF loop does not seed first")
            if any(
                not (loop.lineno <= call.lineno <= (loop.end_lineno or loop.lineno))
                for call in calls
            ):
                raise AssertionError("likelihood PF random call escaped seeded loop")

    for cell_index, call, name in all_random_calls:
        inside_njit = any(start <= call.lineno <= end for start, end in njit_ranges[cell_index])
        if not inside_njit and name != "np.random.default_rng":
            raise AssertionError(
                f"unseeded random call outside Numba: cell={cell_index}, call={name}"
            )
        if name == "np.random.default_rng":
            if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
                raise AssertionError("selector default_rng lacks explicit seed argument")

    wrapper_calls = []
    kernel_calls = []
    resample_calls = []
    selector_rng_calls = []
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
            elif name == "run_particle_filter":
                selector_rng_calls.append((cell_index, node))
            elif name == "hash":
                raise AssertionError(f"process-randomized Python hash() in cell {cell_index}")
    if len(wrapper_calls) != 5:
        raise AssertionError(f"unexpected single-PF wrapper call count: {len(wrapper_calls)}")
    for cell_index, name, call in wrapper_calls:
        if len([keyword for keyword in call.keywords if keyword.arg == "seed"]) != 1:
            raise AssertionError(f"{name} call lacks explicit seed in cell {cell_index}")
    if Counter(name for _, name, _ in kernel_calls) != Counter({"_pf_ancc": 4, "_pf_z": 4}):
        raise AssertionError("unexpected direct single-PF kernel call graph")
    for cell_index, name, call in kernel_calls:
        last = call.args[-1]
        if not (
            (isinstance(last, ast.Name) and last.id == "seed") or isinstance(last, ast.Constant)
        ):
            raise AssertionError(f"{name} does not forward/fix a seed in cell {cell_index}")
    if len(resample_calls) != 4 or any(len(call.args) != 9 for _, call in resample_calls):
        raise AssertionError("resampler does not consume all parent-owned draws")
    if len(selector_rng_calls) != 2:
        raise AssertionError("unexpected explicit-generator selector call graph")
    for cell_index, call in selector_rng_calls:
        if len([keyword for keyword in call.keywords if keyword.arg == "seed"]) != 1:
            raise AssertionError(
                f"selector particle filter lacks explicit seed in cell {cell_index}"
            )

    all_code = "\n".join(source(cell) for _, cell in code_cells)
    required_seed_fragments = (
        "seed_branch='sp45_train' if is_train else 'sp45_test'",
        "_stable_pf_seed(wid,seed_branch+'_ancc')",
        "_stable_pf_seed(wid,seed_branch+'_z')",
        "seed_branch = 'learned_train' if is_train else 'learned_test'",
        "_stable_pf_seed(wid, seed_branch + '_ancc')",
        "_stable_pf_seed(wid, seed_branch + '_z')",
    )
    for fragment in required_seed_fragments:
        if fragment not in all_code:
            raise AssertionError(f"stable scoring seed fragment missing: {fragment}")

    sp45_builder = source(parent_cell(cells, 15))
    learned_builder = source(parent_cell(cells, 41))
    visible_prefix = source(parent_cell(cells, 51))
    if "swid=wid if is_train else None" in sp45_builder or "swid=wid" not in sp45_builder:
        raise AssertionError("SP45 test feature builder does not self-exclude query well")
    if "swid = wid if is_train else None" in learned_builder or "swid = wid" not in learned_builder:
        raise AssertionError("learned test feature builder does not self-exclude query well")
    for fragment in (
        "fi.impute(xy, self_wid=wid)",
        "di.impute(xy, self_wid=wid)",
        "# Same-ID contact candidates are intentionally absent.",
        "_GOLD_CONTACT_OVERRIDE = False",
        "_gold_budget_stopped = False",
    ):
        if fragment not in visible_prefix:
            raise AssertionError(f"sanitized visible-prefix fragment missing: {fragment}")

    forbidden_scoring_fragments = (
        "def tvt_from_contacts",
        "_gold_contact_candidate",
        "_gold_tvt_from_contacts",
        "_gold_best_contact_reconstruction",
        "_gold_reapply_guarded_contact_override",
        "load_well(wid, 'train')",
        "overlap_layer_probe",
        "guarded_overlap_override_report",
        "submission_before_public_overlap",
        "RUN_PUBLIC_OVERLAP_CALIBRATION = True",
        "GLOBAL_TVT_BIAS",
        "U_CONTINUITY_CAP",
        "# Target-free post-composition U-continuity fade.",
    )
    for fragment in forbidden_scoring_fragments:
        if fragment in all_code:
            raise AssertionError(f"forbidden scoring fragment present: {fragment}")
    for fixed_count in ("14151", "14_151", "4301", "4_301"):
        if fixed_count in all_code:
            raise AssertionError(f"fixed visible row count present: {fixed_count}")
    if re.search(r"\b(?:fingerprint|canary)\b", all_code, flags=re.IGNORECASE):
        raise AssertionError("fingerprint/canary behavior present in code")
    for _, cell in code_cells:
        for node in ast.walk(ast.parse(source(cell))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.fullmatch(r"[0-9a-f]{8}", node.value)
            ):
                raise AssertionError(f"fixed well ID literal present: {node.value}")

    model_package = source(parent_cell(cells, 52))
    if not model_package.endswith(source(parent["cells"][52])):
        raise AssertionError("model-package math changed beyond the prepended runtime guard")
    for fragment in (
        "MODEL_PACKAGE_GATED_MAX_WEIGHT",
        "MODEL_PACKAGE_GATED_SCALE",
        "MODEL_PACKAGE_DIFF_P95_DISABLE",
        "_disabled = _disable_limit is not None and _p95 > float(_disable_limit)",
        "Model package skipped by runtime guard",
    ):
        if fragment not in model_package:
            raise AssertionError(f"model-package guard fragment missing: {fragment}")

    hedge = source(parent_cell(cells, 53))
    for fragment in (
        "_BH_STRENGTH = 0.60",
        "_BH_MIN_MASS = 0.25",
        "_BH_SEP_LOW = 4.00",
        "_BH_SEP_HIGH = 40.00",
        "_BH_CAP = 2.00",
        "RUN_PF_BRANCH_HEDGE",
        "PF_SEED_BRANCH_STATS",
    ):
        if fragment not in hedge:
            raise AssertionError(f"exact generalized hedge fragment missing: {fragment}")
    if "_BH_SKIP_EXISTING = True" in hedge or "gold_contact_override_report" in hedge:
        raise AssertionError("branch hedge retains a same-well/contact skip route")

    removed_parent_indices = set(notebook["metadata"]["exact128_v6d"]["removed_parent_indices"])
    required_removed = {13, 32, 35, 36, 39, 42, 45, 47, 48, 49, 54, 55, 56, 57}
    if not required_removed.issubset(removed_parent_indices):
        raise AssertionError("unsafe/diagnostic parent cells were not all removed")

    contracts = [
        (index, source(cell))
        for index, cell in code_cells
        if "# Final hidden-set contract transaction." in source(cell)
    ]
    if len(contracts) != 1:
        raise AssertionError(f"expected one dynamic contract, got {len(contracts)}")
    contract_index, contract = contracts[0]
    if "'derived_kernel': 'heralces/rogii-exact128-v6d'" not in contract:
        raise AssertionError("dynamic contract attribution mismatch")
    later_writers = [
        index for index, cell in code_cells if index > contract_index and ".to_csv(" in source(cell)
    ]
    if later_writers:
        raise AssertionError(f"dynamic contract is not the last CSV writer: {later_writers}")

    variant = notebook.get("metadata", {}).get("exact128_v6d", {})
    expected_variant = {
        "derived_from": "arnavsalkade/rogii-public-6-451-base",
        "query_well_self_exclusion": True,
        "u_continuity": False,
        "hard_budget_sec": 27000.0,
        "final_reserve_sec": 1800.0,
        "deterministic_seed_namespace": "rogii-exact128-v6d",
        "authorized_state": "local_build_only",
    }
    for name, expected in expected_variant.items():
        if variant.get(name) != expected:
            raise AssertionError(f"variant metadata mismatch: {name}")
    if set(variant.get("seeded_single_pf_branches", [])) != set(SCORING_BRANCHES):
        raise AssertionError("variant seeded-branch metadata mismatch")
    if set(variant.get("changed_parent_indices", [])) != set(EXPECTED_DELTA_CATEGORIES):
        raise AssertionError("variant changed-cell metadata mismatch")
    if set(variant.get("retained_parent_indices", [])) != EXPECTED_RETAINED_INDICES:
        raise AssertionError("variant retained-cell metadata mismatch")

    expected_datasets = {
        "phongnguyn23021656/koolbox-offline",
        "fleongg/rogii-claude-models-pub",
        "ravaghi/wellbore-geology-prediction-artifacts",
        "pilkwang/rogii-model-package",
    }
    if set(metadata.get("dataset_sources", [])) != expected_datasets:
        raise AssertionError("kernel dataset sources mismatch")
    if metadata.get("id") != "heralces/rogii-exact128-v6d":
        raise AssertionError("unexpected Kaggle kernel id")
    if metadata.get("code_file") != NOTEBOOK.name:
        raise AssertionError("kernel code_file mismatch")
    if metadata.get("is_private") is not True or metadata.get("enable_internet") is not False:
        raise AssertionError("kernel must be private with internet disabled")
    if not BUILDER.exists():
        raise AssertionError("idempotent builder is missing")

    # Running the local builder again must reproduce the exact notebook bytes.
    subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook_bytes_after = NOTEBOOK.read_bytes()
    if notebook_bytes_after != notebook_bytes_before:
        raise AssertionError("builder is not byte-idempotent")

    print(
        json.dumps(
            {
                "status": "ok",
                "cells": len(cells),
                "code_cells_compiled": len(code_cells),
                "notebook_sha256": notebook_sha_before,
                "parent": "arnavsalkade/rogii-public-6-451-base",
                "retained_parent_cells": len(EXPECTED_RETAINED_INDICES),
                "byte_identical_parent_cells": len(
                    EXPECTED_RETAINED_INDICES - set(EXPECTED_DELTA_CATEGORIES)
                ),
                "allowlisted_parent_cell_deltas": {
                    str(index): category
                    for index, category in sorted(EXPECTED_DELTA_CATEGORIES.items())
                },
                "random_numba_kernels": dict(name_counts),
                "resamplers_with_internal_rng": 0,
                "single_pf_wrapper_calls_explicitly_seeded": len(wrapper_calls),
                "direct_single_pf_kernel_calls_seeded": len(kernel_calls),
                "mounted_seed_pairs_audited": len(seed_map),
                "mounted_seed_collisions": 0,
                "thread_order_independent_seed_mapping": True,
                "query_well_self_exclusion": True,
                "same_id_contact_scoring_routes": 0,
                "overlap_probe_or_public_overlap_routes": 0,
                "fixed_well_ids_or_row_counts": 0,
                "dynamic_sample_contract": True,
                "contract_is_last_csv_writer": True,
                "builder_byte_idempotent": True,
                "ready_state": "ordinary_visible_data_run_only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
