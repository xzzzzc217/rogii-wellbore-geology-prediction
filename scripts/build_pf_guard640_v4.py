"""Build the unpushed Guard640 v4 notebook from the downloaded public source."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = (
    ROOT
    / "data"
    / "public_notebooks"
    / "rogii-codex-public-6-40-guard"
    / "rogii-codex-public-6-40-guard.ipynb"
)
SAFE_V3 = ROOT / "kaggle_notebooks" / "pf_safe_v3" / "rogii-pf-safe-v3.ipynb"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "pf_guard640_v4"
OUTPUT = OUTPUT_DIR / "rogii-guard640-v4.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value


def find_cell(cells: list[dict], fragment: str) -> dict:
    matches = [cell for cell in cells if fragment in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell containing {fragment!r}, got {len(matches)}")
    return matches[0]


def safe_replacement(safe_cells: list[dict], fragment: str) -> dict:
    return copy.deepcopy(find_cell(safe_cells, fragment))


def main() -> None:
    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    safe = json.loads(SAFE_V3.read_text(encoding="utf-8"))
    cells = copy.deepcopy(upstream["cells"])
    safe_cells = safe["cells"]

    # Reuse the already audited fail-closed implementations for the high-risk paths.
    replacements = {
        "# Profile choices:": safe_replacement(safe_cells, "SUBMISSION_PROFILE = 'safe_v3_prefix'"),
        "sample = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')": safe_replacement(
            safe_cells, "sample = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')"
        ),
        "def build_well(hw_path, tw_path, is_train, likpf_map=None):": safe_replacement(
            safe_cells, "swid = wid"
        ),
        "if not bool(globals().get('RUN_GUARDED_OVERLAP_OVERRIDE'": safe_replacement(
            safe_cells,
            "if (not bool(globals().get('RUN_GUARDED_OVERLAP_OVERRIDE'",
        ),
        "# Visible-prefix calibration overlay.": safe_replacement(
            safe_cells, "_gold_budget_stopped"
        ),
        "# Optional saved-model correction on top": safe_replacement(
            safe_cells, "Model package skipped by runtime guard"
        ),
        "# Guarded PF seed-branch midpoint hedge": safe_replacement(
            safe_cells, "# Guarded PF seed-branch midpoint hedge"
        ),
        "# Final submission audit:": safe_replacement(safe_cells, "_safety_controls = {"),
    }
    for index, cell in enumerate(cells):
        cell_source = source(cell)
        for fragment, replacement in replacements.items():
            if fragment in cell_source:
                cells[index] = copy.deepcopy(replacement)
                break

    # Configure one immutable v4 profile.
    control = find_cell(cells, "SUBMISSION_PROFILE = 'safe_v3_prefix'")
    control_source = source(control)
    control_source = control_source.replace(
        "# Two auditable candidates; the package-free profile is deliberately the default.",
        "# Guard640 v4: one audited profile derived from the public 6.40 guard.",
        1,
    )
    control_source = control_source.replace(
        "SUBMISSION_PROFILE = 'safe_v3_prefix'",
        "SUBMISSION_PROFILE = 'guard640_v4'",
        1,
    )
    profile_block = """PROFILE_PRESETS = {
    'guard640_v4': dict(
        visible_prefix_profile='balanced',
        visible_prefix_final_selection='profile',
        visible_prefix_cut_fracs=(0.50, 0.65, 0.75),
        sp45_blend_weight=0.60,
        run_visible_prefix_calibration=True,
        run_bimodal_detector=False,
        run_vp_bimodal_guard=False,
        run_model_package_correction=True,
        model_package_gated_max_weight=0.49,
        model_package_gated_scale=192.0,
    ),
}

if SUBMISSION_PROFILE not in PROFILE_PRESETS:"""
    control_source, count = re.subn(
        r"PROFILE_PRESETS = \{.*?\n\}\n\s*\nif SUBMISSION_PROFILE not in PROFILE_PRESETS:",
        profile_block,
        control_source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("could not replace profile preset block")

    control_source = control_source.replace(
        "SP45_SELECTOR_N_SEEDS = 64", "SP45_SELECTOR_N_SEEDS = 96", 1
    )
    control_source = (
        control_source.replace(
            "NOTEBOOK_HARD_BUDGET_SEC = 7.5 * 3600.0",
            "NOTEBOOK_HARD_BUDGET_SEC = 27000.0",
            1,
        )
        .replace(
            "NOTEBOOK_FINAL_RESERVE_SEC = 0.5 * 3600.0",
            "NOTEBOOK_FINAL_RESERVE_SEC = 1800.0",
            1,
        )
        .replace(
            "MODEL_PACKAGE_MIN_BUDGET_SEC = 2.0 * 3600.0",
            "MODEL_PACKAGE_MIN_BUDGET_SEC = 7200.0",
            1,
        )
    )
    control_source = control_source.replace(
        "RUN_PF_BRANCH_HEDGE = False",
        "RUN_PF_BRANCH_HEDGE = True\n"
        "RUN_U_CONTINUITY = True\n"
        "RUN_PUBLIC_OVERLAP_CALIBRATION = False\n"
        "VISIBLE_PREFIX_ALPHA_MULTIPLIER = 0.35\n"
        "PF_BRANCH_HEDGE_CAP = 3.0\n"
        "U_CONTINUITY_CAP = 8.0\n"
        "U_CONTINUITY_TAU = 240.0",
        1,
    )
    control_source = control_source.replace(
        "# Safe-v3 immutable safety gates. Profiles cannot override these.",
        "# Guard640 v4 immutable safety gates. Profiles cannot override these.",
        1,
    )
    control_source = control_source.replace(
        "MODEL_PACKAGE_REQUIRE = False", "MODEL_PACKAGE_REQUIRE = True", 1
    )
    control_source = control_source.replace(
        "# Fail closed if an unsafe path is accidentally re-enabled before a push.",
        "# Fail closed if an unsafe path is accidentally re-enabled before a push.",
        1,
    )
    assert_start = control_source.index(
        "# Fail closed if an unsafe path is accidentally re-enabled before a push."
    )
    control_source = (
        control_source[:assert_start]
        + """# Fail closed if an unsafe path is accidentally re-enabled before a push.
assert SUBMISSION_PROFILE == 'guard640_v4'
assert RUN_VISIBLE_PREFIX_CALIBRATION is True
assert VISIBLE_PREFIX_FINAL_SELECTION == 'profile'
assert (
    VISIBLE_PREFIX_CAL_SEEDS,
    VISIBLE_PREFIX_FINAL_SEEDS,
    VISIBLE_PREFIX_PARTICLES,
) == (12, 24, 250)
assert VISIBLE_PREFIX_ALPHA_MULTIPLIER == 0.35
assert ALLOW_SAME_ID_TRAIN_LOOKUP is False
assert RUN_GUARDED_OVERLAP_OVERRIDE is False
assert RUN_OVERLAP_DRY_RUN_PROBE is False
assert RUN_EXACT_MATCH_RECOVERY is False
assert RUN_PUBLIC_OVERLAP_CALIBRATION is False
assert APPLY_GLOBAL_LB_BIAS is False
assert RUN_PF_BRANCH_HEDGE is True
assert RUN_U_CONTINUITY is True
assert RUN_MODEL_PACKAGE_CORRECTION is True
assert MODEL_PACKAGE_ALLOW_AUTO_SEARCH is False
assert MODEL_PACKAGE_REQUIRE is True
assert MODEL_PACKAGE_GATED_MAX_WEIGHT == 0.49
assert MODEL_PACKAGE_GATED_SCALE == 192.0
assert MODEL_PACKAGE_DIFF_P95_DISABLE == 25.0
assert SP45_SELECTOR_N_SEEDS == 96
"""
    )
    set_source(control, control_source)

    # Correct inherited prose so it describes the safe v4 route rather than the
    # disabled same-well/contact experiment in the public source.
    prediction_flow = find_cell(cells, "## Prediction Flow")
    set_source(
        prediction_flow,
        """## Prediction flow

1. Build the ridge/PF artifact trajectory and the physical/PF selector
   trajectory with 96 seeds each.
2. Form the SP45-style anchor and apply the robust MD projection.
3. Build and blend the learned trajectory branch, excluding the query well from
   every train-derived feature path.
4. Apply visible-prefix calibration at 12/24 seeds and 250 particles with the
   correction amplitude multiplied by 0.35.
5. Apply the gated model package when the runtime reserve permits, followed by
   the generalized PF branch hedge and target-free U-continuity cap.
6. Re-align `submission.csv` dynamically to the mounted
   `sample_submission.csv` and write the final safety/contract audit.

Same-ID/contact reconstruction, exact-overlap train-TVT calibration, overlap
probing, and global leaderboard bias are immutable-off.
""",
    )
    contact_reconstruction = find_cell(cells, "## Guarded Contact Reconstruction")
    set_source(
        contact_reconstruction,
        """## Leakage controls

The inherited same-well contact reconstruction helper is retained only as
dead, fail-closed reference code. `ALLOW_SAME_ID_TRAIN_LOOKUP` and
`RUN_GUARDED_OVERLAP_OVERRIDE` are both asserted false, and query-well
self-exclusion is enforced in every train-derived feature path.
""",
    )
    contact_override = find_cell(cells, "## Guarded Same-Well Contact Override")
    set_source(
        contact_override,
        """## Disabled same-well helper

This inherited helper cannot write predictions in Guard640 v4. Its entry gate
requires both unsafe controls to be true, while the immutable control cell and
the final safety audit require both controls to remain false.
""",
    )

    # Apply the intended 0.35 visible-prefix amplitude and query-well exclusion.
    vp_cell = find_cell(cells, "_gold_budget_stopped")
    vp_source = source(vp_cell)
    vp_source = vp_source.replace(
        "alpha * 1.30",
        "alpha * float(globals().get('VISIBLE_PREFIX_ALPHA_MULTIPLIER', 0.35))",
    )
    set_source(vp_cell, vp_source)

    # Ensure both PF stages use 96 seeds and every learned feature path excludes query well.
    for cell in cells:
        value = source(cell)
        value = value.replace("PF_SEEDS = 128", "PF_SEEDS = 96")
        value = value.replace("PF_SEEDS = 64", "PF_SEEDS = 96")
        value = value.replace("swid=wid if is_train else None", "swid=wid")
        value = value.replace("swid = wid if is_train else None", "swid = wid")
        set_source(cell, value)

    # Parameterize the generalized branch hedge and retain the 6.40 U-continuity layer.
    hedge = find_cell(cells, "# Guarded PF seed-branch midpoint hedge")
    hedge_source = source(hedge).replace(
        "_BH_CAP = 2.00",
        "_BH_CAP = float(globals().get('PF_BRANCH_HEDGE_CAP', 3.0))",
    )
    set_source(hedge, hedge_source)

    continuity = find_cell(cells, "# Target-free post-composition U-continuity fade.")
    continuity_source = source(continuity)
    continuity_source = continuity_source.replace(
        "_UC_CAP = 8.000000",
        "_UC_CAP = float(globals().get('U_CONTINUITY_CAP', 8.0))",
    ).replace(
        "_UC_TAU = 240.000000",
        "_UC_TAU = float(globals().get('U_CONTINUITY_TAU', 240.0))",
    )
    continuity_source = (
        "if not bool(globals().get('RUN_U_CONTINUITY', False)):\n"
        "    print('U-continuity refinement skipped.')\n"
        "else:\n"
        + "\n".join(("    " + line if line else "") for line in continuity_source.splitlines())
        + "\n"
    )
    set_source(continuity, continuity_source)

    model_package = find_cell(cells, "Model package skipped by runtime guard")
    model_package_source = source(model_package).replace(
        "# Optional saved-model correction on top of the current trajectory submission.\n"
        "# The default profile never enters this branch. The optional profile fails closed when\n"
        "# less than the reserved engineering allowance remains; this is a start-time guard,\n"
        "# not a guarantee about an unbenchmarked external package.",
        "# Gated saved-model correction on top of the current trajectory submission.\n"
        "# Guard640 v4 enters this branch only when the reserved engineering allowance remains.\n"
        "# If the runtime guard fires, the fully audited visible-prefix submission is preserved.",
        1,
    )
    set_source(model_package, model_package_source)

    # The final audit treats the generalized branch hedge as an intended layer, not a leak.
    audit = find_cell(cells, "_safety_controls = {")
    audit_source = source(audit)
    audit_source = audit_source.replace(
        "'run_exact_match_recovery', 'run_pf_branch_hedge', 'apply_global_lb_bias',",
        "'run_exact_match_recovery', 'apply_global_lb_bias',",
    )
    audit_source = audit_source.replace(
        "if not _safety_controls['run_visible_prefix_calibration']:\n"
        "    raise RuntimeError('safe v3 requires visible-prefix calibration')",
        "if not _safety_controls['run_visible_prefix_calibration']:\n"
        "    raise RuntimeError('Guard640 v4 requires visible-prefix calibration')\n"
        "if not _safety_controls['run_pf_branch_hedge']:\n"
        "    raise RuntimeError('Guard640 v4 requires the generalized PF branch hedge')\n"
        "if bool(globals().get('RUN_PUBLIC_OVERLAP_CALIBRATION', True)):\n"
        "    raise RuntimeError('public exact-overlap calibration must stay disabled')\n"
        "if not bool(globals().get('RUN_U_CONTINUITY', False)):\n"
        "    raise RuntimeError('Guard640 v4 requires U-continuity')",
    )
    set_source(audit, audit_source)

    # Remove public exact-overlap train-TVT logic, all fixed-ID demo cells, and read-only visuals.
    remove_fragments = (
        "## Exact-overlap public calibration",
        "# Public-test overlap calibration.",
        "# Read-only probe for same-well overlap layers.",
        "fold_scores_df = pd.DataFrame(fold_scores)",
        "def fig_overview(wid):",
        'DEMO = "00bbac68"',
        "if CFG.SHOW_FIGS and DEMO:",
        "def fig_results():",
        "# Read-only visualization setup.",
        "## Atlas I",
        "## Atlas II",
        "## Atlas III",
        "## Atlas IV",
        "## Atlas V",
        "## Visual integrity manifest",
        "# VISUALS",
        "_VIZ_READY",
    )
    cells = [
        cell for cell in cells if not any(fragment in source(cell) for fragment in remove_fragments)
    ]
    # Remove the visual-section separator, but retain the hidden contract heading.
    cells = [
        cell
        for cell in cells
        if not (cell.get("cell_type") == "markdown" and source(cell).strip() == "---")
    ]

    # Replace title with explicit provenance and the exact safety contract.
    provenance = """# ROGII Guard640 v4

Private, unpushed derivative of Fleong Guo's public
`fleongg/rogii-codex-public-6-40-guard`, downloaded 2026-07-24.
The public title score `6.40` is attribution only and is not claimed for this
modified 96-seed candidate.

Retained clean layers: visible-prefix alpha multiplier 0.35, model-package gate
0.49 / scale 192, generalized PF branch hedge, and target-free U-continuity
with cap 8 / tau 240. Same-ID/contact lookup, public exact-overlap train-TVT
calibration, overlap probing, global LB bias, fixed test IDs, fixed row counts,
and expected prediction hashes are prohibited. The final contract derives row
count, ID set, and order from the mounted runtime sample submission.
"""
    set_source(cells[0], provenance)

    diagnostics = find_cell(cells, "## Diagnostics and Output Files")
    set_source(
        diagnostics,
        """## Diagnostics and output

`submission.csv` is the only scoring artifact. The run also records the
dynamic submission contract, safety controls, visible-prefix calibration,
model-package disagreement gate, generalized branch hedge, U-continuity moves,
stage timing, and the latest fully audited fallback.

The active profile is `guard640_v4`; same-ID/contact and exact-overlap routes
        remain disabled throughout inference.
""",
    )
    closing_note = find_cell(cells, "## Closing Note")
    set_source(
        closing_note,
        """## Closing note

This candidate keeps the public guard's generalized, target-free trajectory
layers while removing its competition-specific overlap routes and read-only
visual workload. Profile names describe the mathematical layer allowed to
change the final trajectory.
""",
    )

    # Attribute the final dynamic audit to the actual public base and this
    # derivative. No fixed expected row count, ID, or prediction hash is stored.
    hidden_contract = find_cell(cells, "# Final hidden-set contract transaction.")
    hidden_contract_source = source(hidden_contract)
    hidden_contract_source = hidden_contract_source.replace(
        "'source_notebook': 'beicicc/rogii-mha160-sep3-r2-20260720',\n"
        "    'source_public_score': None,\n"
        "    'source_version': 1,",
        "'source_notebook': 'fleongg/rogii-codex-public-6-40-guard',\n"
        "    'derived_kernel': 'heralces/rogii-guard640-v4',\n"
        "    'source_public_title_score': 6.40,\n"
        "    'source_snapshot': '2026-07-24',",
        1,
    )
    if "'derived_kernel': 'heralces/rogii-guard640-v4'" not in hidden_contract_source:
        raise RuntimeError("could not update hidden-contract attribution")
    set_source(hidden_contract, hidden_contract_source)

    final_audit = find_cell(cells, "# Final immutable scoring audit")
    final_audit_source = source(final_audit).replace(
        "# Final immutable scoring audit, before read-only visualizations.",
        "# Final immutable scoring audit after the dynamic sample transaction.",
        1,
    )
    final_audit_source = final_audit_source.replace(
        "lowest_frontier_lab_final_audit.json",
        "guard640_v4_final_audit.json",
    )
    set_source(final_audit, final_audit_source)

    # Clear all stale public execution state and add build metadata.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        metadata = cell.setdefault("metadata", {})
        metadata.pop("execution", None)
        metadata.pop("papermill", None)
    notebook_metadata = copy.deepcopy(upstream.get("metadata", {}))
    notebook_metadata.pop("papermill", None)
    notebook_metadata["guard640_v4"] = {
        "derived_from": "fleongg/rogii-codex-public-6-40-guard",
        "public_title_score": 6.40,
        "same_id_enabled": False,
        "public_overlap_enabled": False,
        "selector_pf_seeds": 96,
        "learned_pf_seeds": 96,
        "visible_prefix": {"cal_seeds": 12, "final_seeds": 24, "particles": 250},
        "model_package": {"max_weight": 0.49, "scale": 192.0},
        "branch_hedge": True,
        "u_continuity": {"cap": 8.0, "tau": 240.0},
    }
    upstream["cells"] = cells
    upstream["metadata"] = notebook_metadata

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(upstream, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
