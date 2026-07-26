"""Build deterministic sanitized Exact128 v6d from the public-6.451 source.

This builder deliberately does not push or execute a Kaggle kernel.  It keeps
the public notebook's 128/128 PF, projection/blend, balanced visible-prefix,
0.00425/6 model-package gate, and generalized branch hedge.  Its only scoring
deltas are explicit safety sanitization, runtime fail-closed guards, and the
proven deterministic single-PF RNG repair reused from Clean128 v5d.
"""

# ruff: noqa: E501

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

from build_pf_clean128_v5 import (
    LEARNED_INFERENCE_SOURCE,
    remove_contact_helper,
    sanitize_selector_cell,
)
from build_pf_clean128_v5d import (
    SEED_HELPER_SOURCE,
    transform_compact_pf,
    transform_learned_build,
    transform_learned_warmup,
    transform_spaced_pf,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "kaggle_notebooks" / "pf_public_reproduction" / "rogii-pf-public-reproduction.ipynb"
CONTRACT_PARENT = ROOT / "kaggle_notebooks" / "pf_clean128_v5d" / "rogii-pf-clean128-v5d.ipynb"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "pf_exact128_v6d"
OUTPUT = OUTPUT_DIR / "rogii-exact128-v6d.ipynb"

# Read-only train/CV/visual cells and every same-well probe/override cell are
# absent.  The retained parent cells are recorded individually in notebook
# metadata so the validator can prove all unmodified scoring cells byte-for-byte.
RETAIN_PARENT_INDICES = (
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
)

DELTA_CATEGORIES = {
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


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def replace_once(value: str, old: str, new: str, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return value.replace(old, new, 1)


def code_cell(value: str, *, stage: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"exact128_stage": stage},
        "outputs": [],
        "source": value,
    }


def markdown_cell(value: str, *, stage: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"exact128_stage": stage},
        "source": value,
    }


def find_cell(cells: list[dict], fragment: str) -> dict:
    matches = [cell for cell in cells if fragment in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell containing {fragment!r}, got {len(matches)}")
    return matches[0]


def remove_top_level_functions(value: str, names: set[str]) -> str:
    """Remove named top-level functions without touching surrounding math."""

    tree = ast.parse(value)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    found = {node.name for node in matches}
    if found != names:
        raise RuntimeError(f"contact-function removal mismatch: {sorted(found)} != {sorted(names)}")
    lines = value.splitlines(keepends=True)
    for node in sorted(matches, key=lambda item: item.lineno, reverse=True):
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        del lines[start:end]
        while start < len(lines) and not lines[start].strip():
            del lines[start]
        lines.insert(start, "\n")
    return "".join(lines)


def transform_control(value: str) -> str:
    value = replace_once(
        value,
        "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'",
        "SUBMISSION_PROFILE = 'exact128_v6d'",
        "profile identity",
    )
    preset = """PROFILE_PRESETS = {
    'exact128_v6d': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_guarded_overlap_override=False,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=True,
        model_package_gated_max_weight=0.00425,
        model_package_gated_scale=6.0,
    ),
}

if SUBMISSION_PROFILE not in PROFILE_PRESETS:"""
    value, count = re.subn(
        r"PROFILE_PRESETS = \{.*?\n\}\n\s*\nif SUBMISSION_PROFILE not in PROFILE_PRESETS:",
        preset,
        value,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("profile preset replacement failed")

    value = replace_once(
        value,
        "_profile = PROFILE_PRESETS[SUBMISSION_PROFILE]\n",
        """_profile = PROFILE_PRESETS[SUBMISSION_PROFILE]

import time as _exact128_control_time
NOTEBOOK_STARTED_AT = _exact128_control_time.time()
NOTEBOOK_HARD_BUDGET_SEC = 27000.0
NOTEBOOK_FINAL_RESERVE_SEC = 1800.0
LEARNED_STAGE_MIN_BUDGET_SEC = 7200.0
MODEL_PACKAGE_MIN_BUDGET_SEC = 7200.0
""",
        "hard budget controls",
    )
    value = replace_once(
        value,
        "RUN_OVERLAP_DRY_RUN_PROBE = True",
        "RUN_OVERLAP_DRY_RUN_PROBE = False",
        "disable overlap probe",
    )
    value = replace_once(
        value,
        "RUN_GUARDED_OVERLAP_OVERRIDE = bool(_profile['run_guarded_overlap_override'])",
        "RUN_GUARDED_OVERLAP_OVERRIDE = False",
        "disable guarded overlap",
    )
    value = replace_once(
        value,
        "MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0\n",
        """MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0

# Immutable full-stack safety and composition controls.
ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_PUBLIC_OVERLAP_CALIBRATION = False
RUN_PF_BRANCH_HEDGE = True
RUN_U_CONTINUITY = False
APPLY_GLOBAL_LB_BIAS = False
""",
        "immutable safety controls",
    )
    value += """

# Fail closed before any scoring stage starts.
assert SUBMISSION_PROFILE == 'exact128_v6d'
assert SP45_SELECTOR_N_SEEDS == SELECTOR_PF_SEEDS == 128
assert SP45_SELECTOR_N_PARTICLES == 500
assert SP45_PROJECTION_DEGREE == 3
assert SP45_PROJECTION_BLEND_WEIGHT == 0.75
assert SP45_BLEND_WEIGHT == 0.60
assert VISIBLE_PREFIX_PROFILE == 'balanced'
assert VISIBLE_PREFIX_FINAL_SELECTION == 'profile'
assert VISIBLE_PREFIX_CUT_FRACS == (0.50, 0.65, 0.75)
assert (
    VISIBLE_PREFIX_CAL_SEEDS,
    VISIBLE_PREFIX_FINAL_SEEDS,
    VISIBLE_PREFIX_PARTICLES,
) == (24, 48, 350)
assert RUN_VISIBLE_PREFIX_CALIBRATION is True
assert RUN_MODEL_PACKAGE_CORRECTION is True
assert MODEL_PACKAGE_GATED_MAX_WEIGHT == 0.00425
assert MODEL_PACKAGE_GATED_SCALE == 6.0
assert MODEL_PACKAGE_DIFF_P95_DISABLE == 25.0
assert RUN_PF_BRANCH_HEDGE is True
assert RUN_U_CONTINUITY is False
assert not any((
    ALLOW_SAME_ID_TRAIN_LOOKUP,
    RUN_GUARDED_OVERLAP_OVERRIDE,
    RUN_OVERLAP_DRY_RUN_PROBE,
    RUN_EXACT_MATCH_RECOVERY,
    RUN_PUBLIC_OVERLAP_CALIBRATION,
    APPLY_GLOBAL_LB_BIAS,
    MODEL_PACKAGE_ALLOW_AUTO_SEARCH,
))
print('exact128 retained stack:', {
    'selector_pf': (SP45_SELECTOR_N_SEEDS, SP45_SELECTOR_N_PARTICLES),
    'learned_pf_seeds': 128,
    'projection': (SP45_PROJECTION_DEGREE, SP45_PROJECTION_BLEND_WEIGHT),
    'blend': (SP45_BLEND_WEIGHT, 1.0 - SP45_BLEND_WEIGHT),
    'visible_prefix': (
        VISIBLE_PREFIX_PROFILE,
        VISIBLE_PREFIX_CAL_SEEDS,
        VISIBLE_PREFIX_FINAL_SEEDS,
        VISIBLE_PREFIX_PARTICLES,
    ),
    'model_package': (
        MODEL_PACKAGE_GATED_MAX_WEIGHT,
        MODEL_PACKAGE_GATED_SCALE,
        MODEL_PACKAGE_DIFF_P95_DISABLE,
    ),
    'branch_hedge': RUN_PF_BRANCH_HEDGE,
    'u_continuity': RUN_U_CONTINUITY,
})
"""
    return value


def transform_bridge(value: str) -> str:
    return (
        value
        + """

os.environ['ROGII_GOLD_HARD_BUDGET_SEC'] = str(float(NOTEBOOK_HARD_BUDGET_SEC))
os.environ['ROGII_GOLD_FINAL_RESERVE_SEC'] = str(
    float(NOTEBOOK_FINAL_RESERVE_SEC + MODEL_PACKAGE_MIN_BUDGET_SEC)
)
"""
    )


def transform_visible_prefix(value: str) -> str:
    value = replace_once(
        value,
        "# It runs AFTER the final blend and guarded contact override.",
        "# It runs after the final clean blend; all same-well contact routes are removed.",
        "visible-prefix provenance",
    )
    value = replace_once(
        value,
        "_GOLD_CONTACT_OVERRIDE = _gold_os.environ.get('ROGII_GOLD_CONTACT_OVERRIDE', '0') == '1'",
        """_GOLD_CONTACT_OVERRIDE = False
if _gold_os.environ.get('ROGII_GOLD_CONTACT_OVERRIDE', '0') != '0':
    raise RuntimeError('same-well contact override must remain disabled')""",
        "visible-prefix contact gate",
    )
    value = replace_once(
        value,
        "_GOLD_FINAL_SELECTION = _gold_os.environ.get('ROGII_GOLD_FINAL_SELECTION', 'self_verified_anchor').strip().lower()\n",
        """_GOLD_FINAL_SELECTION = _gold_os.environ.get('ROGII_GOLD_FINAL_SELECTION', 'self_verified_anchor').strip().lower()
_GOLD_HARD_BUDGET_SEC = float(
    _gold_os.environ.get('ROGII_GOLD_HARD_BUDGET_SEC', str(7.5 * 3600.0))
)
_GOLD_FINAL_RESERVE_SEC = float(
    _gold_os.environ.get('ROGII_GOLD_FINAL_RESERVE_SEC', str(2.5 * 3600.0))
)
""",
        "visible-prefix budget controls",
    )
    value = value.replace("fi.impute(xy, self_wid=None)", "fi.impute(xy, self_wid=wid)")
    value = value.replace("di.impute(xy, self_wid=None)", "di.impute(xy, self_wid=wid)")
    value = replace_once(
        value,
        "    out.update(_gold_contact_candidate(wid, hw_masked, data_dir))\n",
        "    # Same-ID contact candidates are intentionally absent.\n",
        "remove visible-prefix contact candidate",
    )
    value = replace_once(
        value,
        """        if _GOLD_CONTACT_OVERRIDE:
            _profile_sub = _gold_reapply_guarded_contact_override(_profile_sub[['id', 'tvt']], _GOLD_DATA)
        else:
            _profile_sub = _profile_sub[['id', 'tvt']].copy()
""",
        "        _profile_sub = _profile_sub[['id', 'tvt']].copy()\n",
        "remove visible-prefix contact reapply",
    )
    value = remove_top_level_functions(
        value,
        {
            "_gold_contact_ref_priority",
            "_gold_tvt_from_contacts",
            "_gold_contact_candidate",
            "_gold_best_contact_reconstruction",
            "_gold_reapply_guarded_contact_override",
        },
    )
    value = replace_once(
        value,
        " or best_name.startswith('contact_')",
        "",
        "remove dormant contact candidate class",
    )
    value = replace_once(
        value,
        """    _gold_wells = list(_gold_base['well'].drop_duplicates())[:_GOLD_MAX_WELLS]
    for _wi, _wid in enumerate(_gold_wells, 1):
""",
        """    _gold_wells = list(_gold_base['well'].drop_duplicates())[:_GOLD_MAX_WELLS]
    _gold_budget_stopped = False
    _gold_budget_stop_after_wells = None
    for _wi, _wid in enumerate(_gold_wells, 1):
        _gold_total_elapsed = _gold_time.time() - float(
            globals().get('NOTEBOOK_STARTED_AT', _gold_t0)
        )
        if _gold_total_elapsed >= (
            _GOLD_HARD_BUDGET_SEC - _GOLD_FINAL_RESERVE_SEC
        ):
            _gold_budget_stopped = True
            _gold_budget_stop_after_wells = _wi - 1
            print(
                'Visible-prefix budget guard: keeping the audited base '
                'for remaining wells.',
                flush=True,
            )
            break
""",
        "visible-prefix per-well budget guard",
    )
    value = replace_once(
        value,
        "        contact_override_enabled=bool(_GOLD_CONTACT_OVERRIDE),\n",
        """        contact_override_enabled=False,
        query_well_self_exclusion=True,
        budget_guard_stopped=bool(_gold_budget_stopped),
        budget_guard_completed_wells=_gold_budget_stop_after_wells,
        hard_budget_sec=float(_GOLD_HARD_BUDGET_SEC),
        downstream_reserve_sec=float(_GOLD_FINAL_RESERVE_SEC),
""",
        "visible-prefix audit fields",
    )
    compile(value, "visible_prefix_sanitized", "exec")
    return value


def transform_model_package(value: str) -> str:
    prefix = """# Runtime start guard for the exact public model-package layer.
import time as _mp_budget_time
_mp_total_elapsed = _mp_budget_time.time() - float(
    globals().get('NOTEBOOK_STARTED_AT', _mp_budget_time.time())
)
_mp_hard_budget = float(globals().get('NOTEBOOK_HARD_BUDGET_SEC', 7.5 * 3600.0))
_mp_final_reserve = float(globals().get('NOTEBOOK_FINAL_RESERVE_SEC', 0.5 * 3600.0))
_mp_min_budget = float(globals().get('MODEL_PACKAGE_MIN_BUDGET_SEC', 2.0 * 3600.0))
if (
    bool(globals().get('RUN_MODEL_PACKAGE_CORRECTION', False))
    and _mp_total_elapsed
    > (_mp_hard_budget - _mp_final_reserve - _mp_min_budget)
):
    print(
        'Model package skipped by runtime guard; keeping visible-prefix submission.csv.',
        flush=True,
    )
    RUN_MODEL_PACKAGE_CORRECTION = False

"""
    return prefix + value


def transform_branch_hedge(value: str) -> str:
    start = value.index("_bh_skip = set()")
    end = value.index("_bh_rows = []", start)
    value = value[:start] + "_bh_skip = set()\n" + value[end:]
    first_line, body = value.split("\n", 1)
    indented = "\n".join(("    " + line if line else "") for line in body.splitlines())
    return (
        first_line
        + "\nif not bool(globals().get('RUN_PF_BRANCH_HEDGE', False)):\n"
        + "    print('PF seed-branch midpoint hedge skipped.')\n"
        + "else:\n"
        + indented
        + "\n"
    )


SAFETY_AUDIT_SOURCE = """# Exact128 v6d scoring-path safety audit before the dynamic contract.
import json as _ea_json
import time as _ea_time
from pathlib import Path as _EAPath

import numpy as _ea_np
import pandas as _ea_pd

_EA_WORK = _EAPath('/kaggle/working') if _EAPath('/kaggle/working').exists() else _EAPath('.')
_EA_SUB = _EA_WORK / 'submission.csv'
_ea_unsafe = {
    'allow_same_id_train_lookup': bool(ALLOW_SAME_ID_TRAIN_LOOKUP),
    'run_guarded_overlap_override': bool(RUN_GUARDED_OVERLAP_OVERRIDE),
    'run_overlap_dry_run_probe': bool(RUN_OVERLAP_DRY_RUN_PROBE),
    'run_exact_match_recovery': bool(RUN_EXACT_MATCH_RECOVERY),
    'run_public_overlap_calibration': bool(RUN_PUBLIC_OVERLAP_CALIBRATION),
    'run_u_continuity': bool(RUN_U_CONTINUITY),
    'apply_global_lb_bias': bool(APPLY_GLOBAL_LB_BIAS),
    'model_package_auto_search': bool(MODEL_PACKAGE_ALLOW_AUTO_SEARCH),
}
if any(_ea_unsafe.values()):
    raise RuntimeError(f'unsafe exact128 control enabled: {_ea_unsafe}')
if not RUN_VISIBLE_PREFIX_CALIBRATION:
    raise RuntimeError('balanced visible-prefix layer must remain enabled')
if not RUN_PF_BRANCH_HEDGE:
    raise RuntimeError('generalized branch hedge must remain enabled')
if not _EA_SUB.exists():
    raise FileNotFoundError('full stack did not produce submission.csv')
_ea_frame = _ea_pd.read_csv(_EA_SUB, dtype={'id': 'string'})
_ea_values = _ea_pd.to_numeric(_ea_frame['tvt'], errors='coerce').to_numpy(dtype='float64')
if list(_ea_frame.columns) != ['id', 'tvt']:
    raise RuntimeError(f'unexpected full-stack columns: {list(_ea_frame.columns)}')
if _ea_frame['id'].isna().any() or _ea_frame['id'].duplicated().any():
    raise RuntimeError('full-stack IDs must be non-null and unique')
if not _ea_np.isfinite(_ea_values).all():
    raise RuntimeError('full-stack TVT contains NaN or infinity')
_EA_AUDIT = {
    'profile': SUBMISSION_PROFILE,
    'rows_before_contract': int(len(_ea_frame)),
    'selector_pf_seeds': int(SP45_SELECTOR_N_SEEDS),
    'learned_pf_seeds': int(CFG.PF_SEEDS),
    'sp45_weight': float(SP45_BLEND_WEIGHT),
    'learned_weight': float(1.0 - SP45_BLEND_WEIGHT),
    'projection_degree': int(SP45_PROJECTION_DEGREE),
    'projection_blend_weight': float(SP45_PROJECTION_BLEND_WEIGHT),
    'visible_prefix': {
        'profile': VISIBLE_PREFIX_PROFILE,
        'cal_seeds': int(VISIBLE_PREFIX_CAL_SEEDS),
        'final_seeds': int(VISIBLE_PREFIX_FINAL_SEEDS),
        'particles': int(VISIBLE_PREFIX_PARTICLES),
        'cut_fracs': list(VISIBLE_PREFIX_CUT_FRACS),
    },
    'model_package': {
        'initially_enabled': True,
        'final_runtime_state': bool(RUN_MODEL_PACKAGE_CORRECTION),
        'max_weight': float(MODEL_PACKAGE_GATED_MAX_WEIGHT),
        'scale': float(MODEL_PACKAGE_GATED_SCALE),
        'p95_disable': float(MODEL_PACKAGE_DIFF_P95_DISABLE),
    },
    'generalized_branch_hedge': True,
    'u_continuity': False,
    'unsafe_controls': _ea_unsafe,
    'query_well_self_exclusion': True,
    'deterministic_pf_seeding': bool(PF_DETERMINISTIC_SEEDING),
    'pf_seed_namespace': PF_SEED_NAMESPACE,
    'elapsed_sec_before_contract': float(_ea_time.time() - NOTEBOOK_STARTED_AT),
    'hard_budget_sec': float(NOTEBOOK_HARD_BUDGET_SEC),
    'final_reserve_sec': float(NOTEBOOK_FINAL_RESERVE_SEC),
}
(_EA_WORK / 'exact128_v6d_scoring_path_audit.json').write_text(
    _ea_json.dumps(_EA_AUDIT, indent=2, sort_keys=True),
    encoding='utf-8',
)
print('Exact128 v6d scoring-path audit:', _ea_json.dumps(_EA_AUDIT, indent=2), flush=True)
"""


def transform_parent_cell(index: int, value: str) -> str:
    if index == 7:
        return transform_control(value)
    if index == 8:
        return transform_bridge(value)
    if index == 12:
        return remove_contact_helper(value)
    if index == 15:
        value = value.replace("swid=wid if is_train else None", "swid=wid")
        return transform_compact_pf(value)
    if index == 27:
        return sanitize_selector_cell(value)
    if index == 37:
        return transform_spaced_pf(value)
    if index == 38:
        return transform_learned_warmup(value)
    if index == 41:
        value = value.replace("swid = wid if is_train else None", "swid = wid")
        return transform_learned_build(value)
    if index == 44:
        return LEARNED_INFERENCE_SOURCE
    if index == 51:
        return transform_visible_prefix(value)
    if index == 52:
        return transform_model_package(value)
    if index == 53:
        return transform_branch_hedge(value)
    return value


def main() -> None:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    contract_parent = json.loads(CONTRACT_PARENT.read_text(encoding="utf-8"))
    parent_cells = parent["cells"]

    cells = [
        markdown_cell(
            """# ROGII Exact128 v6d

Local-only deterministic and sanitized derivative of
`arnavsalkade/rogii-public-6-451-base`. The public title score is provenance,
not a score claim for this candidate.

The retained stack is the public 128/128 PF anchor and learned branch, degree-3
projection at 0.75, 0.60/0.40 blend, original balanced visible-prefix
24/48/350 settings, exact conservative model-package gate 0.00425/6 with its
25-foot p95 disable guard, and the generalized PF branch hedge. U-continuity,
same-ID/contact, overlap probes or calibration, global LB bias, fixed IDs,
fixed row counts, expected hashes, fingerprints, and canaries are absent from
the scoring path.
""",
            stage="provenance",
        ),
        markdown_cell(
            """## Fail-closed scoring transaction

Every query well is excluded from both train-derived spatial feature builders.
All formerly unseeded single-PF Numba kernels receive stable SHA-256
well/branch seeds. Selector, learned, visible-prefix, and model-package stages
have hard-budget fallbacks. The mounted runtime sample submission is the only
authority for final IDs, row count, and order.
""",
            stage="flow",
        ),
    ]

    for index in RETAIN_PARENT_INDICES:
        cell = copy.deepcopy(parent_cells[index])
        cell_source = transform_parent_cell(index, source(cell))
        metadata = cell.setdefault("metadata", {})
        metadata.pop("execution", None)
        metadata.pop("papermill", None)
        metadata["exact128_parent_index"] = index
        metadata["exact128_stage"] = "public_scoring_core"
        if index in DELTA_CATEGORIES:
            metadata["exact128_delta"] = DELTA_CATEGORIES[index]
        cell["source"] = cell_source
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        cells.append(cell)

        if index == 8:
            seed_source = SEED_HELPER_SOURCE.replace(
                "rogii-pf-clean128-v5d",
                "rogii-exact128-v6d",
            ).replace(
                "PF seed derivation.",
                "Exact128 v6d PF seed derivation.",
            )
            cells.append(code_cell(seed_source, stage="deterministic_seed_control"))

    cells.extend(
        [
            markdown_cell(
                """## Immutable full-stack safety audit

This audit verifies the retained exact parameters and all prohibited controls
before the runtime-sample contract becomes the last prediction writer.
""",
                stage="audit",
            ),
            code_cell(SAFETY_AUDIT_SOURCE, stage="audit"),
            markdown_cell(
                """## Hidden-rerun submission contract

The mounted sample submission is authoritative. No fixed ID, row count,
prediction hash, visible-test fingerprint, or canary is stored.
""",
                stage="contract",
            ),
        ]
    )

    contract = copy.deepcopy(
        find_cell(contract_parent["cells"], "# Final hidden-set contract transaction.")
    )
    contract_source = source(contract).replace(
        "'derived_kernel': 'heralces/rogii-pf-clean128-v5d'",
        "'derived_kernel': 'heralces/rogii-exact128-v6d'",
        1,
    )
    contract["source"] = contract_source
    contract["execution_count"] = None
    contract["outputs"] = []
    contract["metadata"] = {"exact128_stage": "contract"}
    cells.append(contract)

    final_audit = copy.deepcopy(
        find_cell(contract_parent["cells"], "# Final immutable scoring audit")
    )
    final_source = (
        source(final_audit)
        .replace(
            "clean128_v5d_final_audit.json",
            "exact128_v6d_final_audit.json",
            1,
        )
        .replace("Clean128 v5d final audit:", "Exact128 v6d final audit:", 1)
    )
    final_audit["source"] = final_source
    final_audit["execution_count"] = None
    final_audit["outputs"] = []
    final_audit["metadata"] = {"exact128_stage": "final_audit"}
    cells.append(final_audit)

    metadata = copy.deepcopy(parent.get("metadata", {}))
    metadata.pop("papermill", None)
    metadata["exact128_v6d"] = {
        "derived_from": "arnavsalkade/rogii-public-6-451-base",
        "source_snapshot": "2026-07-24",
        "source_public_title_score": 6.451,
        "retained_parent_indices": list(RETAIN_PARENT_INDICES),
        "changed_parent_indices": sorted(DELTA_CATEGORIES),
        "removed_parent_indices": sorted(
            set(range(len(parent_cells))) - set(RETAIN_PARENT_INDICES)
        ),
        "delta_categories": {
            str(index): category for index, category in sorted(DELTA_CATEGORIES.items())
        },
        "selector_pf": {"seeds": 128, "particles": 500},
        "learned_pf_seeds": 128,
        "projection": {"degree": 3, "blend_weight": 0.75},
        "clean_blend": {"sp45": 0.60, "learned": 0.40},
        "visible_prefix": {
            "profile": "balanced",
            "cal_seeds": 24,
            "final_seeds": 48,
            "particles": 350,
            "cut_fracs": [0.50, 0.65, 0.75],
        },
        "model_package": {
            "max_weight": 0.00425,
            "scale": 6.0,
            "diff_p95_disable": 25.0,
        },
        "branch_hedge": {
            "enabled": True,
            "strength": 0.60,
            "min_mass": 0.25,
            "separation": [4.0, 40.0],
            "cap": 2.0,
        },
        "u_continuity": False,
        "query_well_self_exclusion": True,
        "hard_budget_sec": 27000.0,
        "final_reserve_sec": 1800.0,
        "deterministic_seed_namespace": "rogii-exact128-v6d",
        "seeded_single_pf_branches": [
            "sp45_train_ancc",
            "sp45_train_z",
            "sp45_test_ancc",
            "sp45_test_z",
            "learned_train_ancc",
            "learned_train_z",
            "learned_test_ancc",
            "learned_test_z",
        ],
        "authorized_state": "local_build_only",
    }

    output = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": parent.get("nbformat", 4),
        "nbformat_minor": parent.get("nbformat_minor", 5),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
