"""Build one private high-risk dip-rate HMM controlled-verification notebook.

The parent is the exact current 6.477 full stack.  The appended HMM code is
extracted verbatim from the already confirmed evaluator, so the prediction
parameters cannot drift between OOF evidence and hidden inference.  The run is
explicitly marked high risk because confirmation passed pooled/all-fold gates
but failed the preregistered well-win and worst-tail gates.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT / "kaggle_notebooks" / "pf_public_reproduction"
PARENT = PARENT_DIR / "rogii-pf-public-reproduction.ipynb"
PARENT_METADATA = PARENT_DIR / "kernel-metadata.json"
HMM_EVALUATOR = ROOT / "scripts" / "evaluate_dip_rate_hmm_discovery.py"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "dip_rate_hmm_v1"
OUTPUT = OUTPUT_DIR / "rogii-dip-rate-hmm-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"

CONFIG_CELL = 7
DEMO_CELL = 36
BRANCH_CELL = 53
FINAL_AUDIT_CELL = 54
CORE_CONSTANTS = {
    "GRID_STEP",
    "RATE_STATES",
    "RATE_SPAN",
    "POSITION_BAND",
    "POSITION_SIGMA",
    "RATE_STAY_PROBABILITY",
    "OVERLAY_SCALE",
    "DISAGREEMENT_CLIP",
}
CORE_FUNCTIONS = {
    "_logadd",
    "_dip_rate_filter",
    "_prepare_typewell",
    "predict_hidden_hmm",
}


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected one {label} fragment, found {value.count(old)}")
    return value.replace(old, new)


def _code_cell(code: str, marker: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {marker: True},
        "outputs": [],
        "source": code.strip(),
    }


def _core_source() -> str:
    text = HMM_EVALUATOR.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    selected: list[tuple[int, str]] = []
    found_constants: set[str] = set()
    found_functions: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & CORE_CONSTANTS:
                selected.append(
                    (node.lineno, "\n".join(lines[node.lineno - 1 : node.end_lineno]))
                )
                found_constants.update(names & CORE_CONSTANTS)
        elif isinstance(node, ast.FunctionDef) and node.name in CORE_FUNCTIONS:
            first_line = min(
                [node.lineno, *[decorator.lineno for decorator in node.decorator_list]]
            )
            selected.append(
                (first_line, "\n".join(lines[first_line - 1 : node.end_lineno]))
            )
            found_functions.add(node.name)
    if found_constants != CORE_CONSTANTS or found_functions != CORE_FUNCTIONS:
        raise RuntimeError(
            f"HMM core extraction incomplete: constants={found_constants} "
            f"functions={found_functions}"
        )
    return "\n\n".join(value for _, value in sorted(selected))


def _notebook_core_source() -> str:
    core = _core_source()
    if core.count("@njit(cache=True") != 2:
        raise RuntimeError("expected exactly two cached HMM decorators")
    return core.replace("@njit(cache=True", "@njit(cache=False")


def _start_cell() -> dict[str, object]:
    return _code_cell(
        """
import time as _dh_time
_DH_START_TIME = _dh_time.time()
""",
        "dip_rate_hmm_start",
    )


def _overlay_cell() -> dict[str, object]:
    core = _notebook_core_source()
    code = f'''
# Frozen target-free dip-rate HMM high-risk controlled verification.
import hashlib as _dh_hashlib
import json as _dh_json
import math
from pathlib import Path as _DhPath

import numpy as np
import pandas as pd
from numba import njit

{core}

_DH_WORK = _DhPath('/kaggle/working') if _DhPath('/kaggle/working').exists() else _DhPath('.')
_DH_SUBMISSION = _DH_WORK / 'submission.csv'
_DH_BEFORE = _DH_WORK / 'submission_before_dip_rate_hmm.csv'
_DH_PER_WELL = _DH_WORK / 'dip_rate_hmm_per_well.csv'
_DH_REPORT = _DH_WORK / 'dip_rate_hmm_deploy_report.json'


def _dh_sha256(path):
    return _dh_hashlib.sha256(_DhPath(path).read_bytes()).hexdigest()


_dh_data = _DhPath(
    getattr(
        CFG,
        'DATA',
        getattr(
            CFG,
            'dataset_path',
            '/kaggle/input/competitions/rogii-wellbore-geology-prediction',
        ),
    )
)
_dh_test = _dh_data / 'test'
_dh_sample_path = _dh_data / 'sample_submission.csv'
if not _dh_test.exists() or not _dh_sample_path.exists() or not _DH_SUBMISSION.exists():
    raise RuntimeError('dip-rate HMM deployment inputs are missing')

_dh_submission = pd.read_csv(_DH_SUBMISSION, dtype={{'id': 'string'}})
_dh_sample = pd.read_csv(_dh_sample_path, dtype={{'id': 'string'}})
if list(_dh_submission.columns) != ['id', 'tvt']:
    raise RuntimeError('dip-rate HMM expected id,tvt submission schema')
if len(_dh_submission) != len(_dh_sample):
    raise RuntimeError('dip-rate HMM submission row count differs from sample')
if not _dh_submission['id'].astype(str).equals(_dh_sample['id'].astype(str)):
    raise RuntimeError('dip-rate HMM submission ID order differs from sample')
if not _dh_submission['id'].is_unique:
    raise RuntimeError('dip-rate HMM submission IDs are not unique')
_dh_before_values = pd.to_numeric(
    _dh_submission['tvt'], errors='coerce'
).to_numpy(dtype=float)
if not np.isfinite(_dh_before_values).all():
    raise RuntimeError('dip-rate HMM base predictions are non-finite')
_dh_submission.to_csv(_DH_BEFORE, index=False)

_dh_id_to_position = {{
    row_id: position
    for position, row_id in enumerate(_dh_submission['id'].astype(str).tolist())
}}
_dh_delta = np.zeros(len(_dh_submission), dtype=float)
_dh_covered = np.zeros(len(_dh_submission), dtype=bool)
_dh_rows = []
for _dh_horizontal_path in sorted(_dh_test.glob('*__horizontal_well.csv')):
    _dh_well = _dh_horizontal_path.name.removesuffix('__horizontal_well.csv')
    _dh_typewell_path = _dh_test / f'{{_dh_well}}__typewell.csv'
    if not _dh_typewell_path.exists():
        raise RuntimeError(f'missing typewell file for {{_dh_well}}')
    _dh_horizontal = pd.read_csv(_dh_horizontal_path)
    _dh_typewell = pd.read_csv(_dh_typewell_path)
    _dh_prediction, _dh_uncertainty = predict_hidden_hmm(
        _dh_horizontal, _dh_typewell
    )
    _dh_tvt_input = pd.to_numeric(
        _dh_horizontal['TVT_input'], errors='coerce'
    ).to_numpy(dtype=float)
    _dh_hidden = np.flatnonzero(~np.isfinite(_dh_tvt_input))
    if len(_dh_hidden) != len(_dh_prediction):
        raise RuntimeError(f'dip-rate HMM hidden row mismatch for {{_dh_well}}')
    _dh_positions = []
    for _dh_hidden_index in _dh_hidden.tolist():
        _dh_id = f'{{_dh_well}}_{{int(_dh_hidden_index)}}'
        if _dh_id not in _dh_id_to_position:
            raise RuntimeError(f'dip-rate HMM ID missing from submission: {{_dh_id}}')
        _dh_position = _dh_id_to_position[_dh_id]
        if _dh_covered[_dh_position]:
            raise RuntimeError(f'duplicate dip-rate HMM row ID: {{_dh_id}}')
        _dh_positions.append(_dh_position)
    _dh_positions = np.asarray(_dh_positions, dtype=np.int64)
    _dh_base = _dh_before_values[_dh_positions]
    _dh_correction = OVERLAY_SCALE * np.clip(
        _dh_prediction - _dh_base,
        -DISAGREEMENT_CLIP,
        DISAGREEMENT_CLIP,
    )
    if not np.isfinite(_dh_correction).all():
        raise RuntimeError(f'non-finite dip-rate HMM correction for {{_dh_well}}')
    if float(np.max(np.abs(_dh_correction))) > 1.0000001:
        raise RuntimeError(f'dip-rate HMM movement cap exceeded for {{_dh_well}}')
    _dh_delta[_dh_positions] = _dh_correction
    _dh_covered[_dh_positions] = True
    _dh_rows.append(
        {{
            'well': _dh_well,
            'rows': int(len(_dh_positions)),
            'correction_mean': float(np.mean(_dh_correction)),
            'correction_rms': float(np.sqrt(np.mean(_dh_correction**2))),
            'correction_minimum': float(np.min(_dh_correction)),
            'correction_maximum': float(np.max(_dh_correction)),
            'hmm_uncertainty_mean': float(np.mean(_dh_uncertainty)),
        }}
    )

if not _dh_covered.all():
    _dh_missing = _dh_submission.loc[~_dh_covered, 'id'].astype(str).head().tolist()
    raise RuntimeError(f'dip-rate HMM coverage incomplete: {{_dh_missing}}')
_dh_final_values = _dh_before_values + _dh_delta
if not np.isfinite(_dh_final_values).all():
    raise RuntimeError('dip-rate HMM final predictions are non-finite')
_dh_submission['tvt'] = _dh_final_values
_dh_submission.to_csv(_DH_SUBMISSION, index=False)
pd.DataFrame(_dh_rows).to_csv(_DH_PER_WELL, index=False)

_dh_report = {{
    'candidate': 'target-free dip-rate HMM clipped overlay',
    'parent_stack': 'heralces/rogii-pf-public-reproduction',
    'contract_scope': 'dip_rate_hmm_overlay_only',
    'controlled_public_validation_only': True,
    'high_risk_candidate': True,
    'confirmation_decision': 'FAIL_tail_gates',
    'confirmation_failure_reasons': [
        'well_win_fraction_0.6383_below_0.65',
        'worst_well_delta_0.6836_above_0.25',
    ],
    'rank_seeking_rationale': (
        'Untouched confirmation improved pooled RMSE by 0.156756 and all 12 '
        'spatial/Typewell folds; competition ranking uses pooled RMSE.'
    ),
    'target_free_runtime_inputs_only': True,
    'training_or_hidden_truth_used_for_prediction': False,
    'well_id_encoded_as_feature': False,
    'absolute_coordinates_used': False,
    'fixed_test_well_used': False,
    'fixed_visible_rows_used': False,
    'expected_visible_hash_used': False,
    'leaderboard_score_used_for_selection': False,
    'probe_flags_used': False,
    'same_well_or_contact_override_in_overlay': False,
    'meta_learner_in_overlay': False,
    'parameter_grid_searched': False,
    'notebook_numba_cache_disabled': True,
    'parameters': {{
        'grid_step': float(GRID_STEP),
        'rate_states': int(RATE_STATES),
        'rate_span': float(RATE_SPAN),
        'position_band': float(POSITION_BAND),
        'position_sigma': float(POSITION_SIGMA),
        'rate_stay_probability': float(RATE_STAY_PROBABILITY),
        'overlay_scale': float(OVERLAY_SCALE),
        'hmm_disagreement_clip': float(DISAGREEMENT_CLIP),
        'maximum_final_movement': float(OVERLAY_SCALE * DISAGREEMENT_CLIP),
    }},
    'confirmation': {{
        'wells': 741,
        'pooled_delta': -0.15675641408341612,
        'well_win_fraction': 0.6383265856950068,
        'p95_well_delta': -0.12914116104679962,
        'worst_well_delta': 0.6836268403655126,
        'spatial_improved_folds': 7,
        'spatial_fold_count': 7,
        'typewell_improved_folds': 5,
        'typewell_fold_count': 5,
    }},
    'test_wells': int(len(_dh_rows)),
    'rows': int(len(_dh_submission)),
    'changed_rows': int((np.abs(_dh_delta) > 1e-12).sum()),
    'correction_mean': float(np.mean(_dh_delta)),
    'correction_rms': float(np.sqrt(np.mean(_dh_delta**2))),
    'correction_minimum': float(np.min(_dh_delta)),
    'correction_maximum': float(np.max(_dh_delta)),
    'id_order_matches_sample': True,
    'ids_unique': True,
    'finite_predictions': True,
    'before_submission_sha256': _dh_sha256(_DH_BEFORE),
    'submission_sha256': _dh_sha256(_DH_SUBMISSION),
    'elapsed_seconds': float(_dh_time.time() - _DH_START_TIME),
}}
_DH_REPORT.write_text(
    _dh_json.dumps(_dh_report, indent=2, sort_keys=True) + '\\n',
    encoding='utf-8',
)
print('dip-rate HMM deployment report:', _dh_report, flush=True)
'''
    return _code_cell(code, "dip_rate_hmm_overlay_code")


def _contract_cell() -> dict[str, object]:
    return _code_cell(
        r'''
# Final cross-file contract after inherited full-stack audit.
import hashlib as _dh_final_hashlib
import json as _dh_final_json
from pathlib import Path as _DhFinalPath

import numpy as _dh_final_np
import pandas as _dh_final_pd

_DH_FINAL_WORK = (
    _DhFinalPath('/kaggle/working')
    if _DhFinalPath('/kaggle/working').exists()
    else _DhFinalPath('.')
)
_DH_FINAL_SUB = _DH_FINAL_WORK / 'submission.csv'
_DH_FINAL_BEFORE = _DH_FINAL_WORK / 'submission_before_dip_rate_hmm.csv'
_DH_FINAL_REPORT = _DH_FINAL_WORK / 'dip_rate_hmm_deploy_report.json'
_DH_FINAL_PER_WELL = _DH_FINAL_WORK / 'dip_rate_hmm_per_well.csv'
for _dh_required in (_DH_FINAL_SUB, _DH_FINAL_BEFORE, _DH_FINAL_REPORT, _DH_FINAL_PER_WELL):
    if not _dh_required.exists():
        raise RuntimeError(f'missing dip-rate HMM contract output: {_dh_required}')

_dh_final_sub = _dh_final_pd.read_csv(_DH_FINAL_SUB, dtype={'id': 'string'})
_dh_final_before = _dh_final_pd.read_csv(_DH_FINAL_BEFORE, dtype={'id': 'string'})
_dh_final_per_well = _dh_final_pd.read_csv(_DH_FINAL_PER_WELL)
_dh_final_report = _dh_final_json.loads(_DH_FINAL_REPORT.read_text(encoding='utf-8'))
if list(_dh_final_sub.columns) != ['id', 'tvt'] or list(_dh_final_before.columns) != ['id', 'tvt']:
    raise RuntimeError('dip-rate HMM contract schema mismatch')
if not _dh_final_sub['id'].astype(str).equals(_dh_final_before['id'].astype(str)):
    raise RuntimeError('dip-rate HMM before/final ID order mismatch')
if len(_dh_final_sub) != int(_dh_final_report['rows']):
    raise RuntimeError('dip-rate HMM report row count mismatch')
if int(_dh_final_per_well['rows'].sum()) != len(_dh_final_sub):
    raise RuntimeError('dip-rate HMM per-well row total mismatch')
_dh_final_values = _dh_final_pd.to_numeric(
    _dh_final_sub['tvt'], errors='coerce'
).to_numpy(dtype=float)
_dh_final_before_values = _dh_final_pd.to_numeric(
    _dh_final_before['tvt'], errors='coerce'
).to_numpy(dtype=float)
if (
    not _dh_final_np.isfinite(_dh_final_values).all()
    or not _dh_final_np.isfinite(_dh_final_before_values).all()
):
    raise RuntimeError('dip-rate HMM contract found non-finite values')
_dh_final_sha = _dh_final_hashlib.sha256(_DH_FINAL_SUB.read_bytes()).hexdigest()
if _dh_final_sha != str(_dh_final_report['submission_sha256']):
    raise RuntimeError('dip-rate HMM report SHA differs from final submission')
_dh_final_delta = _dh_final_values - _dh_final_before_values
if float(_dh_final_np.max(_dh_final_np.abs(_dh_final_delta))) > 1.0000001:
    raise RuntimeError('dip-rate HMM final movement cap exceeded')
if int((_dh_final_np.abs(_dh_final_delta) > 1e-12).sum()) != int(_dh_final_report['changed_rows']):
    raise RuntimeError('dip-rate HMM changed-row count mismatch')
if _dh_final_report['target_free_runtime_inputs_only'] is not True:
    raise RuntimeError('dip-rate HMM target-free contract is not asserted')
if _dh_final_report['high_risk_candidate'] is not True:
    raise RuntimeError('dip-rate HMM risk disclosure is missing')
print('dip-rate HMM final contract: PASS', _dh_final_sha, flush=True)
''',
        "dip_rate_hmm_final_contract",
    )


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    parent = json.loads(parent_bytes)
    parent_metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))
    if parent_metadata.get("id") != "heralces/rogii-pf-public-reproduction":
        raise RuntimeError("unexpected dip-rate HMM parent kernel")
    if len(parent["cells"]) <= FINAL_AUDIT_CELL:
        raise RuntimeError("dip-rate HMM parent notebook is incomplete")

    retained_indices = list(range(BRANCH_CELL + 1))
    cells = [copy.deepcopy(parent["cells"][index]) for index in retained_indices]
    for cell, parent_index in zip(cells, retained_indices, strict=True):
        cell_source = source(cell)
        if parent_index == CONFIG_CELL:
            cell_source = _replace_once(
                cell_source,
                "RUN_OVERLAP_DRY_RUN_PROBE = True",
                "RUN_OVERLAP_DRY_RUN_PROBE = False",
                "dry-run probe toggle",
            )
        elif parent_index == DEMO_CELL:
            cell_source = _replace_once(
                cell_source,
                (
                    'DEMO = "00bbac68" if '
                    '(CFG.DATA/"train"/"00bbac68__horizontal_well.csv").exists() '
                    'else _demo_well()'
                ),
                "DEMO = _demo_well()",
                "fixed demo well",
            )
        cell["source"] = cell_source
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cell.setdefault("metadata", {})["dip_rate_hmm_parent_index"] = parent_index

    final_audit = copy.deepcopy(parent["cells"][FINAL_AUDIT_CELL])
    final_audit["source"] = source(final_audit)
    final_audit["execution_count"] = None
    final_audit["outputs"] = []
    final_audit.setdefault("metadata", {})["dip_rate_hmm_parent_index"] = FINAL_AUDIT_CELL

    notebook = copy.deepcopy(parent)
    notebook["cells"] = [
        {
            "cell_type": "markdown",
            "metadata": {"dip_rate_hmm_overlay": True},
            "source": (
                "# ROGII target-free dip-rate HMM v1\n\n"
                "High-risk controlled public validation on the exact 6.477 parent. "
                "The frozen HMM improved pooled RMSE by 0.156756 on 741 untouched "
                "wells and all 12 group folds, but failed preregistered well-win and "
                "worst-tail gates. This run is risk-disclosed and does not alter final "
                "selection without a strictly better terminal Kaggle score."
            ),
        },
        _start_cell(),
        *cells,
        _overlay_cell(),
        final_audit,
        _contract_cell(),
    ]
    notebook.setdefault("metadata", {})["dip_rate_hmm_build"] = {
        "source_kernel": parent_metadata["id"],
        "source_notebook_sha256": parent_sha256,
        "source_evaluator_sha256": hashlib.sha256(HMM_EVALUATOR.read_bytes()).hexdigest(),
        "retained_parent_cells": retained_indices + [FINAL_AUDIT_CELL],
        "overlay_inserted_after_parent_cell": BRANCH_CELL,
        "high_risk_candidate": True,
        "confirmation_pooled_delta": -0.15675641408341612,
        "confirmation_all_spatial_typewell_folds_improve": True,
        "confirmation_tail_gate_passed": False,
        "target_free_overlay": True,
        "leaderboard_score_used_for_runtime_selection": False,
    }

    overlay_code = source(_overlay_cell())
    forbidden = ("00e12e8b", "00bbac68", "000d7d20", "14151")
    if any(value in overlay_code for value in forbidden):
        raise RuntimeError("dip-rate HMM overlay contains fixed visible identifiers or rows")
    required = (
        "target_free_runtime_inputs_only",
        "high_risk_candidate",
        "confirmation_decision",
        "submission_before_dip_rate_hmm.csv",
        "dip_rate_hmm_deploy_report.json",
        "leaderboard_score_used_for_selection",
    )
    missing = [value for value in required if value not in overlay_code]
    if missing:
        raise RuntimeError(f"dip-rate HMM overlay contract is incomplete: {missing}")

    metadata = {
        "id": "heralces/rogii-dip-rate-hmm-v1",
        "title": "ROGII Dip Rate HMM v1",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["geology", "hmm", "target-free"],
        "dataset_sources": parent_metadata.get("dataset_sources", []),
        "kernel_sources": parent_metadata.get("kernel_sources", []),
        "competition_sources": parent_metadata.get("competition_sources", []),
        "model_sources": parent_metadata.get("model_sources", []),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
