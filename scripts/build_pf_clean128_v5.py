"""Build the local-only sanitized 128-seed clean-anchor candidate.

The scoring core is copied from the audited Safe v3 derivative of the exact
public-6.451 notebook. Only the clean anchor is retained: 128-seed selector PF,
the 128-seed learned trajectory, the existing SP45 projection, and the fixed
0.60 SP45 / 0.40 learned blend. All later correction cells are omitted.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_notebooks" / "pf_safe_v3" / "rogii-pf-safe-v3.ipynb"
CONTRACT_SOURCE = ROOT / "kaggle_notebooks" / "pf_guard640_v4" / "rogii-guard640-v4.ipynb"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "pf_clean128_v5"
OUTPUT = OUTPUT_DIR / "rogii-pf-clean128-v5.ipynb"

# Only cells needed by the exact clean anchor. Diagnostic CV, figures, contact,
# visible-prefix, model-package, hedge, continuity, and late-ablation cells are
# deliberately absent.
SOURCE_CELL_INDICES = (
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
)


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def code_cell(value: str, *, stage: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"clean128_stage": stage},
        "outputs": [],
        "source": value,
    }


def markdown_cell(value: str, *, stage: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"clean128_stage": stage},
        "source": value,
    }


def find_cell(cells: list[dict], fragment: str) -> dict:
    matches = [cell for cell in cells if fragment in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell containing {fragment!r}, got {len(matches)}")
    return matches[0]


CONTROL_SOURCE = """# Clean128 v5 has one immutable scoring profile.
SUBMISSION_PROFILE = 'clean128_v5'

import time as _clean_control_time
NOTEBOOK_STARTED_AT = _clean_control_time.time()
NOTEBOOK_HARD_BUDGET_SEC = 27000.0
NOTEBOOK_FINAL_RESERVE_SEC = 1800.0
LEARNED_STAGE_MIN_BUDGET_SEC = 7200.0

# Data and audited artifact roots.
COMPETITION_DATA_ROOT = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts'
KOOLBOX_OFFLINE_ROOTS = (
    '/kaggle/input/datasets/phongnguyn23021656/koolbox-offline',
    '/kaggle/input/koolbox-offline',
    '/kaggle/input/pm-125564438-at-07-07-2026-07-28-00',
)
LEARNED_MODEL_ROOTS = (
    '/kaggle/input/datasets/fleongg/rogii-claude-models-pub',
    '/kaggle/input/rogii-claude-models-pub',
)

# Exact public-6.451 clean-anchor configuration.
SP45_RIDGE_MODEL_WEIGHT = 0.30
SP45_SELECTOR_WEIGHT = 0.70
SP45_SELECTOR_N_PARTICLES = 500
SP45_SELECTOR_N_SEEDS = 128
SELECTOR_PF_SEEDS = SP45_SELECTOR_N_SEEDS
SELECTOR_PF_RETURN_STD = False
SP45_PROJECTION_DEGREE = 3
SP45_PROJECTION_BLEND_WEIGHT = 0.75
SP45_BLEND_WEIGHT = 0.60
SP45_BLEND_CANDIDATE_WEIGHTS = (0.60,)

# Optional diagnostics stay off on the scoring run.
RUN_CV_REPORT = False
RUN_FULL_STACK_CV_ABLATION = False
CV_N_WELLS = 250
CV_ABLATION_N_WELLS = 250
CV_N_SPLITS = 5
CV_SEED = 0
CV_SELECTOR_PF_SEEDS = 24
ABLATION_VP_POSTERIOR_TRUST_ONLY = False

# Exact selector configuration. The optional bimodal correction remains off.
RUN_BIMODAL_DETECTOR = False
RUN_BIMODAL_SELECTOR_HEDGE = False
BIMODAL_DZ_RANGE = 20.0
BIMODAL_DZ_STEP = 0.5
BUNDLE_MIN = 10.0
BUNDLE_MAX = 20.0
BIMODAL_J_RATIO_EPS = 0.15
SCAN_MIN_SEP = 8.0
BIMODAL_TEMP = 0.75
RUN_ADAPTIVE_TEMP = True
T_MIN = 0.25
T_MAX = 4.0
RUN_PREFIX_TRUST_GATE = True
TRUST_MIN_PREFIX_ROWS = 60
TRUST_FALLBACK = 0.0
TRUST_KAPPA = 1.0
USE_STRUCTURAL_P_FALLBACK = False
BIMODAL_FORCE_MIDPOINT = False
BIMODAL_TRIGGER_MIN_MEDIAN_DIFF = 12.0
BIMODAL_TRIGGER_MIN_P90_DIFF = 18.0
BIMODAL_TRIGGER_MIN_BIG_DIFF_FRAC = 0.18
BIMODAL_TRIGGER_BIG_DIFF_THRESHOLD = 15.0
BIMODAL_MIN_VALID_GR_ROWS = 80

RUN_HEEL_CALIBRATION = True
RUN_HEEL_LOCALIZATION_REPORT = False
RUN_HEEL_ABLATION_GRID = False
HEEL_MIN_ROWS = 40
HEEL_LOCALIZATION_TOLERANCE = 2.0
HEEL_ALPHA_MIN = 0.25
HEEL_ALPHA_MAX = 4.0
HEEL_BETA_ABS_MAX = 500.0
RUN_GR_FFT_DENOISE = False
RUN_SEQ_MATCHER = False

# Every prohibited correction is immutable-off. Its implementation cell is
# also absent from this notebook.
ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_EXACT_MATCH_RECOVERY = False
RUN_PUBLIC_OVERLAP_CALIBRATION = False
RUN_VISIBLE_PREFIX_CALIBRATION = False
RUN_MODEL_PACKAGE_CORRECTION = False
RUN_PF_BRANCH_HEDGE = False
RUN_U_CONTINUITY = False
APPLY_GLOBAL_LB_BIAS = False
MODEL_PACKAGE_ALLOW_AUTO_SEARCH = False
MODEL_PACKAGE_REQUIRE = False

assert SUBMISSION_PROFILE == 'clean128_v5'
assert SP45_SELECTOR_N_SEEDS == SELECTOR_PF_SEEDS == 128
assert SP45_BLEND_CANDIDATE_WEIGHTS == (0.60,)
assert SP45_BLEND_WEIGHT == 0.60
assert SP45_PROJECTION_DEGREE == 3
assert SP45_PROJECTION_BLEND_WEIGHT == 0.75
assert not any((
    ALLOW_SAME_ID_TRAIN_LOOKUP,
    RUN_GUARDED_OVERLAP_OVERRIDE,
    RUN_OVERLAP_DRY_RUN_PROBE,
    RUN_EXACT_MATCH_RECOVERY,
    RUN_PUBLIC_OVERLAP_CALIBRATION,
    RUN_VISIBLE_PREFIX_CALIBRATION,
    RUN_MODEL_PACKAGE_CORRECTION,
    RUN_PF_BRANCH_HEDGE,
    RUN_U_CONTINUITY,
    APPLY_GLOBAL_LB_BIAS,
    MODEL_PACKAGE_ALLOW_AUTO_SEARCH,
    MODEL_PACKAGE_REQUIRE,
))
print('submission profile:', SUBMISSION_PROFILE)
print('clean anchor:', {
    'selector_pf_seeds': SP45_SELECTOR_N_SEEDS,
    'learned_pf_seeds': 128,
    'sp45_weight': SP45_BLEND_WEIGHT,
    'projection_degree': SP45_PROJECTION_DEGREE,
    'projection_weight': SP45_PROJECTION_BLEND_WEIGHT,
})
"""


LEARNED_INFERENCE_SOURCE = """def _find_models():
    \"\"\"Find the audited pre-trained trajectory boosters from explicit roots.\"\"\"
    roots = [Path(p) for p in globals().get('LEARNED_MODEL_ROOTS', ()) if str(p).strip()]
    for root in roots:
        if (root / 'features.json').exists() and list(root.glob('lgb*.pkl')):
            return root
    return None


def main():
    import json
    import joblib

    t0 = time.time()
    train_wids = sorted(
        p.stem.replace('__horizontal_well', '')
        for p in (CFG.DATA / 'train').glob('*__horizontal_well.csv')
    )
    test_wids = sorted(
        p.stem.replace('__horizontal_well', '')
        for p in (CFG.DATA / 'test').glob('*__horizontal_well.csv')
    )
    if CFG.N_TRAIN_WELLS:
        train_wids = train_wids[:CFG.N_TRAIN_WELLS]
    print(f'train wells: {len(train_wids)} | test wells: {len(test_wids)}')
    init_imputers(train_wids)

    # Dynamic hidden-test features. build_well passes each query well as
    # self_wid, excluding it from every train-derived spatial prior.
    print('building lik-PF + features (test)...', flush=True)
    likpf_test = build_likpf(test_wids, 'test')
    test_df = add_likpf_features(
        build_features(test_wids, 'test', is_train=False),
        likpf_test,
    ).reset_index(drop=True)

    models_dir = _find_models()
    if models_dir is None:
        raise RuntimeError('audited learned model bundle is unavailable')
    print(f'INFERENCE mode - loading models from {models_dir}', flush=True)
    feats = json.loads((models_dir / 'features.json').read_text())
    models = [joblib.load(path) for path in sorted(models_dir.glob('lgb*.pkl'))]
    if not models:
        raise RuntimeError('audited learned model bundle contains no boosters')
    for column in feats:
        if column not in test_df.columns:
            test_df[column] = 0.0
    xt = test_df[feats].values.astype(np.float32)
    meta_test = np.mean([model.predict(xt) for model in models], axis=0)

    test_pred = make_prediction(test_df, meta_test, None)
    sub = pd.read_csv(CFG.DATA / 'sample_submission.csv')
    prediction_by_id = dict(zip(test_df['id'], test_pred, strict=True))
    mapped = sub['id'].map(prediction_by_id)
    if mapped.isna().any():
        raise RuntimeError(f'learned trajectory missed {int(mapped.isna().sum())} sample IDs')
    sub['tvt'] = mapped.astype(float)
    if not np.isfinite(sub['tvt'].to_numpy(dtype=float)).all():
        raise RuntimeError('learned trajectory produced non-finite values')
    sub.to_csv(CFG.OUT / 'submission.csv', index=False)
    print(f'submission.csv written ({len(sub)} rows) in {time.time() - t0:.0f}s')
    return sub, None


_clean_sp45_fallback = CFG.OUT / 'sp45_projection_submission.csv'
_clean_elapsed = time.time() - float(NOTEBOOK_STARTED_AT)
_clean_remaining = float(NOTEBOOK_HARD_BUDGET_SEC) - _clean_elapsed
_clean_required = float(NOTEBOOK_FINAL_RESERVE_SEC) + float(LEARNED_STAGE_MIN_BUDGET_SEC)
if _clean_remaining < _clean_required:
    print(
        'Learned stage skipped by 7.5h budget guard; retaining SP45 projection.',
        flush=True,
    )
    sub = pd.read_csv(_clean_sp45_fallback)
    sub.to_csv(CFG.OUT / 'submission.csv', index=False)
    cv_final = None
else:
    try:
        sub, cv_final = main()
    except Exception as _clean_learned_error:
        print(
            'Learned stage failed; restoring audited SP45 projection:',
            _clean_learned_error,
            flush=True,
        )
        sub = pd.read_csv(_clean_sp45_fallback)
        sub.to_csv(CFG.OUT / 'submission.csv', index=False)
        cv_final = None
sub.head()
"""


SAFETY_AUDIT_SOURCE = """# Clean128 scoring-path audit before the dynamic sample transaction.
import json as _ca_json
import time as _ca_time
from pathlib import Path as _CAPath

import numpy as _ca_np
import pandas as _ca_pd

_CA_WORK = _CAPath('/kaggle/working') if _CAPath('/kaggle/working').exists() else _CAPath('.')
_CA_SUB = _CA_WORK / 'submission.csv'
_ca_controls = {
    'allow_same_id_train_lookup': bool(ALLOW_SAME_ID_TRAIN_LOOKUP),
    'run_guarded_overlap_override': bool(RUN_GUARDED_OVERLAP_OVERRIDE),
    'run_overlap_dry_run_probe': bool(RUN_OVERLAP_DRY_RUN_PROBE),
    'run_exact_match_recovery': bool(RUN_EXACT_MATCH_RECOVERY),
    'run_public_overlap_calibration': bool(RUN_PUBLIC_OVERLAP_CALIBRATION),
    'run_visible_prefix_calibration': bool(RUN_VISIBLE_PREFIX_CALIBRATION),
    'run_model_package_correction': bool(RUN_MODEL_PACKAGE_CORRECTION),
    'run_pf_branch_hedge': bool(RUN_PF_BRANCH_HEDGE),
    'run_u_continuity': bool(RUN_U_CONTINUITY),
    'apply_global_lb_bias': bool(APPLY_GLOBAL_LB_BIAS),
}
if any(_ca_controls.values()):
    raise RuntimeError(f'prohibited clean128 control enabled: {_ca_controls}')
if not _CA_SUB.exists():
    raise FileNotFoundError('clean anchor did not produce submission.csv')
_ca_frame = _ca_pd.read_csv(_CA_SUB, dtype={'id': 'string'})
_ca_values = _ca_pd.to_numeric(_ca_frame['tvt'], errors='coerce').to_numpy(dtype='float64')
if list(_ca_frame.columns) != ['id', 'tvt']:
    raise RuntimeError(f'unexpected clean-anchor columns: {list(_ca_frame.columns)}')
if _ca_frame['id'].isna().any() or _ca_frame['id'].duplicated().any():
    raise RuntimeError('clean-anchor IDs must be non-null and unique')
if not _ca_np.isfinite(_ca_values).all():
    raise RuntimeError('clean-anchor TVT contains NaN or infinity')
_CA_AUDIT = {
    'profile': SUBMISSION_PROFILE,
    'rows_before_contract': int(len(_ca_frame)),
    'selector_pf_seeds': int(SP45_SELECTOR_N_SEEDS),
    'learned_pf_seeds': int(CFG.PF_SEEDS),
    'sp45_weight': float(SP45_BLEND_WEIGHT),
    'learned_weight': float(1.0 - SP45_BLEND_WEIGHT),
    'projection_degree': int(SP45_PROJECTION_DEGREE),
    'projection_blend_weight': float(SP45_PROJECTION_BLEND_WEIGHT),
    'prohibited_controls': _ca_controls,
    'query_well_self_exclusion': True,
    'elapsed_sec_before_contract': float(_ca_time.time() - NOTEBOOK_STARTED_AT),
    'hard_budget_sec': float(NOTEBOOK_HARD_BUDGET_SEC),
    'final_reserve_sec': float(NOTEBOOK_FINAL_RESERVE_SEC),
}
(_CA_WORK / 'clean128_v5_scoring_path_audit.json').write_text(
    _ca_json.dumps(_CA_AUDIT, indent=2, sort_keys=True),
    encoding='utf-8',
)
print('Clean128 scoring-path audit:', _ca_json.dumps(_CA_AUDIT, indent=2), flush=True)
"""


def sanitize_selector_cell(value: str) -> str:
    # The public demo can load a same-ID train well and substitute its contact
    # trajectory. Remove that branch entirely; hidden and visible runs now share
    # the same test-only selector path.
    start = value.index("train_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'train'")
    end = value.index("    # 128-seed likelihood-weighted PF ensemble")
    replacement = """test_hw_files = sorted(
    glob.glob(str(CFG.dataset_path / 'test' / '*__horizontal_well.csv'))
)
test_wells = [os.path.basename(path).split('__')[0] for path in test_hw_files]

rows = []
bimodal_report_rows = []
PF_SEED_BRANCH_STATS = {}
_selector_budget_stopped = False
for i, wid in enumerate(test_wells):
    print(f'\\nProcessing {i + 1}/{len(test_wells)}: {wid}...')
    hw_te, tw_te = load_well(wid, 'test')

    # Leave two hours for the learned trajectory and 30 minutes for final
    # contract/audit. Remaining wells fall back to their last visible TVT.
    _selector_elapsed = time.time() - float(NOTEBOOK_STARTED_AT)
    _selector_cutoff = (
        float(NOTEBOOK_HARD_BUDGET_SEC)
        - float(NOTEBOOK_FINAL_RESERVE_SEC)
        - float(LEARNED_STAGE_MIN_BUDGET_SEC)
    )
    if _selector_elapsed >= _selector_cutoff:
        _selector_budget_stopped = True
        _known = pd.to_numeric(hw_te['TVT_input'], errors='coerce').dropna()
        if len(_known) == 0:
            raise RuntimeError(f'no visible TVT fallback for {wid}')
        _fallback_value = float(_known.iloc[-1])
        ws = sample[sample['well'] == wid]
        for _, row in ws.iterrows():
            rows.append({'id': row['id'], 'tvt': _fallback_value})
        bimodal_report_rows.append({
            'well': wid,
            'selector_code': -1,
            'selector_variant': 'budget_last_visible_fallback',
            'n_eval': float(len(ws)),
            'z_span': float('nan'),
            'bimodal_active': False,
        })
        print(f'  Selector budget fallback added {len(ws)} rows', flush=True)
        continue

    selector_code, selector_variant, selector_n_eval, selector_z_span = (
        selector_well_code(hw_te)
    )
    tw_ref = tw_te

"""
    value = value[:start] + replacement + value[end:]
    value = value.replace(
        """        ridx = int(row['row_idx'])
        if tvt_phys is not None:
            tvt_val = float(tvt_phys.iloc[ridx])
        else:
            tvt_val = float(tvt_selector[ridx])
        rows.append({'id': row['id'], 'tvt': tvt_val})
""",
        """        ridx = int(row['row_idx'])
        rows.append({'id': row['id'], 'tvt': float(tvt_selector[ridx])})
""",
        1,
    )
    if "tvt_phys" in value or "load_well(wid, 'train')" in value:
        raise RuntimeError("same-ID selector branch was not fully removed")
    return value


def remove_contact_helper(value: str) -> str:
    start = value.index("def tvt_from_contacts(")
    end = value.index("def load_well(", start)
    return value[:start] + value[end:]


def main() -> None:
    source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    contract_notebook = json.loads(CONTRACT_SOURCE.read_text(encoding="utf-8"))
    source_cells = source_notebook["cells"]

    cells = [
        markdown_cell(
            """# ROGII PF Clean128 v5

Local-only sanitized derivative of the exact public-6.451 clean anchor.
The public title score is attribution, not a score claim for this candidate.

The only scoring layers are the 128-seed selector/ridge SP45 anchor, its robust
MD projection, 128-seed learned trajectory inference from the audited mounted
boosters, and a fixed 0.60 SP45 / 0.40 learned blend. Same-ID/contact,
overlap/exact-overlap, visible-prefix, model-package, branch hedge,
U-continuity, global bias, fixed IDs/row counts/hashes, and canaries are absent
or immutable-off.
""",
            stage="provenance",
        ),
        markdown_cell(
            """## Scoring transaction

1. Build the selector/ridge trajectory with 128 PF seeds.
2. Apply the existing degree-3, weight-0.75 robust projection.
3. Build dynamic learned features with 128 PF seeds while excluding the query
   well from every train-derived spatial prior.
4. Blend projected SP45 and learned trajectories at exactly 0.60 / 0.40.
5. Fail closed through a sample-derived contract and final immutable audit.

The 7.5-hour guard preserves the projected SP45 trajectory if the learned
stage lacks its two-hour reserve or fails.
""",
            stage="flow",
        ),
        code_cell(CONTROL_SOURCE, stage="control"),
    ]

    for index in SOURCE_CELL_INDICES:
        cell = copy.deepcopy(source_cells[index])
        metadata = cell.setdefault("metadata", {})
        metadata["clean128_source_index"] = index
        metadata["clean128_stage"] = "exact_clean_core"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)

        value = source(cell)
        if index == 12:
            value = remove_contact_helper(value)
        elif index == 27:
            value = sanitize_selector_cell(value)
        elif index == 34:
            value = value.replace("PF_SEEDS = 64", "PF_SEEDS = 128", 1)
            if "PF_SEEDS = 128" not in value:
                raise RuntimeError("learned PF seed replacement failed")
        elif index == 44:
            value = LEARNED_INFERENCE_SOURCE
        cell["source"] = value
        cells.append(cell)

    cells.extend(
        [
            markdown_cell(
                """## Immutable clean scoring audit

The following transaction verifies that all prohibited controls remain false
and that the clean anchor is structurally valid before the runtime sample
contract becomes the last prediction writer.
""",
                stage="audit",
            ),
            code_cell(SAFETY_AUDIT_SOURCE, stage="audit"),
            markdown_cell(
                """## Hidden-rerun submission contract

The mounted sample submission is authoritative for row count, ID set, and
order. No fixed ID, row count, prediction hash, or visible-test fingerprint is
stored in this notebook.
""",
                stage="contract",
            ),
        ]
    )

    contract = copy.deepcopy(
        find_cell(contract_notebook["cells"], "# Final hidden-set contract transaction.")
    )
    contract_source = source(contract)
    contract_source = contract_source.replace(
        "'source_notebook': 'fleongg/rogii-codex-public-6-40-guard',",
        "'source_notebook': 'arnavsalkade/rogii-public-6-451-base',",
        1,
    )
    contract_source = contract_source.replace(
        "'derived_kernel': 'heralces/rogii-guard640-v4',",
        "'derived_kernel': 'heralces/rogii-pf-clean128-v5',",
        1,
    )
    contract_source = contract_source.replace(
        "'source_public_title_score': 6.40,",
        "'source_public_title_score': 6.451,",
        1,
    )
    contract["source"] = contract_source
    contract["execution_count"] = None
    contract["outputs"] = []
    contract["metadata"] = {"clean128_stage": "contract"}
    cells.append(contract)

    final_audit = copy.deepcopy(
        find_cell(contract_notebook["cells"], "# Final immutable scoring audit")
    )
    final_source = source(final_audit)
    final_source = final_source.replace(
        "guard640_v4_final_audit.json",
        "clean128_v5_final_audit.json",
        1,
    ).replace(
        "Lowest-public frontier-lab final audit:",
        "Clean128 v5 final audit:",
        1,
    )
    final_audit["source"] = final_source
    final_audit["execution_count"] = None
    final_audit["outputs"] = []
    final_audit["metadata"] = {"clean128_stage": "final_audit"}
    cells.append(final_audit)

    metadata = copy.deepcopy(source_notebook.get("metadata", {}))
    metadata.pop("papermill", None)
    metadata["clean128_v5"] = {
        "derived_from": "arnavsalkade/rogii-public-6-451-base",
        "sanitized_base": "heralces/rogii-pf-safe-v3",
        "selector_pf_seeds": 128,
        "learned_pf_seeds": 128,
        "sp45_weight": 0.60,
        "learned_weight": 0.40,
        "projection": {"degree": 3, "blend_weight": 0.75},
        "query_well_self_exclusion": True,
        "late_correction_cells_present": False,
        "hard_budget_sec": 27000.0,
        "final_reserve_sec": 1800.0,
        "learned_stage_min_budget_sec": 7200.0,
    }
    output_notebook = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": source_notebook.get("nbformat", 4),
        "nbformat_minor": source_notebook.get("nbformat_minor", 5),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output_notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
