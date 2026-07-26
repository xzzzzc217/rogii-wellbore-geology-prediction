"""Static, fail-closed validator for the unpushed Safe v3 notebook."""

from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "rogii-pf-safe-v3.ipynb"
METADATA = ROOT / "kernel-metadata.json"


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
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"cell {index} contains stale execution state")

    control_index = next(
        index
        for index, cell in code_cells
        if "SUBMISSION_PROFILE = 'safe_v3_prefix'" in "".join(cell.get("source", []))
    )
    control = "".join(cells[control_index]["source"])
    values = _literal_assignments(control)
    all_source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    default_runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(control, f"{NOTEBOOK.name}:control", "exec"), default_runtime)
    optional_control = control.replace(
        "SUBMISSION_PROFILE = 'safe_v3_prefix'",
        "SUBMISSION_PROFILE = 'safe_v3_prefix_modelpkg'",
        1,
    )
    optional_runtime: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(
            compile(optional_control, f"{NOTEBOOK.name}:optional_control", "exec"),
            optional_runtime,
        )

    expected_literals = {
        "SUBMISSION_PROFILE": "safe_v3_prefix",
        "ALLOW_SAME_ID_TRAIN_LOOKUP": False,
        "RUN_GUARDED_OVERLAP_OVERRIDE": False,
        "RUN_OVERLAP_DRY_RUN_PROBE": False,
        "RUN_EXACT_MATCH_RECOVERY": False,
        "RUN_PF_BRANCH_HEDGE": False,
        "APPLY_GLOBAL_LB_BIAS": False,
        "MODEL_PACKAGE_ALLOW_AUTO_SEARCH": False,
        "MODEL_PACKAGE_REQUIRE": False,
        "VISIBLE_PREFIX_INCLUDE_PF": True,
        "VISIBLE_PREFIX_CAL_SEEDS": 12,
        "VISIBLE_PREFIX_FINAL_SEEDS": 24,
        "VISIBLE_PREFIX_PARTICLES": 250,
    }
    for name, expected in expected_literals.items():
        actual = values.get(name)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")

    expected_default_runtime = {
        "SUBMISSION_PROFILE": "safe_v3_prefix",
        "RUN_VISIBLE_PREFIX_CALIBRATION": True,
        "VISIBLE_PREFIX_FINAL_SELECTION": "profile",
        "RUN_MODEL_PACKAGE_CORRECTION": False,
        "RUN_GUARDED_OVERLAP_OVERRIDE": False,
    }
    expected_optional_runtime = {
        "SUBMISSION_PROFILE": "safe_v3_prefix_modelpkg",
        "RUN_VISIBLE_PREFIX_CALIBRATION": True,
        "VISIBLE_PREFIX_FINAL_SELECTION": "profile",
        "RUN_MODEL_PACKAGE_CORRECTION": True,
        "MODEL_PACKAGE_GATED_MAX_WEIGHT": 0.49,
        "MODEL_PACKAGE_GATED_SCALE": 192.0,
        "RUN_GUARDED_OVERLAP_OVERRIDE": False,
    }
    for name, expected in expected_default_runtime.items():
        if default_runtime.get(name) != expected:
            raise AssertionError(
                f"default runtime {name}: expected {expected!r}, got {default_runtime.get(name)!r}"
            )
    for name, expected in expected_optional_runtime.items():
        if optional_runtime.get(name) != expected:
            raise AssertionError(
                f"optional runtime {name}: expected {expected!r}, "
                f"got {optional_runtime.get(name)!r}"
            )

    required_source_fragments = (
        "'safe_v3_prefix': dict(",
        "'safe_v3_prefix_modelpkg': dict(",
        "run_visible_prefix_calibration=True",
        "run_model_package_correction=False",
        "run_model_package_correction=True",
        "model_package_gated_max_weight=0.49",
        "model_package_gated_scale=192.0",
        "swid = wid",
        "self_wid=swid",
        "self_wid=wid",
        "_GOLD_CONTACT_OVERRIDE",
        "_gold_budget_stopped",
        "Model package skipped by runtime guard",
        "_safety_controls = {",
    )
    for fragment in required_source_fragments:
        if fragment not in all_source:
            raise AssertionError(f"required source fragment missing: {fragment}")

    forbidden_fragments = (
        "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'",
        "MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.00425",
        "RUN_PF_BRANCH_HEDGE = True",
        "ALLOW_SAME_ID_TRAIN_LOOKUP = True",
        "RUN_GUARDED_OVERLAP_OVERRIDE = True",
        "RUN_OVERLAP_DRY_RUN_PROBE = True",
        "APPLY_GLOBAL_LB_BIAS = True",
        "_OV_WEIGHT =",
        "submission_before_public_overlap.csv",
        "_UC_CAP =",
        "_BH_STRENGTH = 1.00",
        "00bbac68",
        "canary",
        "expected_sha256",
    )
    for fragment in forbidden_fragments:
        if fragment in all_source:
            raise AssertionError(f"forbidden source fragment present: {fragment}")

    expected_datasets = {
        "phongnguyn23021656/koolbox-offline",
        "fleongg/rogii-claude-models-pub",
        "ravaghi/wellbore-geology-prediction-artifacts",
        "pilkwang/rogii-model-package",
    }
    if set(metadata["dataset_sources"]) != expected_datasets:
        raise AssertionError("kernel metadata dataset_sources differ from the audited set")
    if metadata["id"] != "heralces/rogii-pf-safe-v3":
        raise AssertionError("unexpected Kaggle kernel id")
    if metadata["code_file"] != NOTEBOOK.name:
        raise AssertionError("kernel metadata code_file mismatch")
    if metadata["is_private"] is not True or metadata["enable_internet"] is not False:
        raise AssertionError("kernel must be private with internet disabled")

    print(
        json.dumps(
            {
                "status": "ok",
                "code_cells_compiled": len(code_cells),
                "default_profile": values["SUBMISSION_PROFILE"],
                "visible_prefix": {
                    "cal_seeds": values["VISIBLE_PREFIX_CAL_SEEDS"],
                    "final_seeds": values["VISIBLE_PREFIX_FINAL_SEEDS"],
                    "particles": values["VISIBLE_PREFIX_PARTICLES"],
                },
                "unsafe_gates_enabled": 0,
                "metadata_datasets": len(expected_datasets),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
