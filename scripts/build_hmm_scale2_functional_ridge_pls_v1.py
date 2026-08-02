"""Build, but do not push, the gated HMM scale2 plus Ridge/PLS notebook."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HMM_DIR = ROOT / "kaggle_notebooks" / "dip_rate_hmm_scale2_v1"
HMM_NOTEBOOK = HMM_DIR / "rogii-dip-rate-hmm-scale2-v1.ipynb"
HMM_METADATA = HMM_DIR / "kernel-metadata.json"
FUNCTIONAL_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_pls_v1"
FUNCTIONAL_NOTEBOOK = FUNCTIONAL_DIR / "rogii-functional-ridge-pls-v1.ipynb"
FUNCTIONAL_METADATA = FUNCTIONAL_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "hmm_scale2_functional_ridge_pls_v1"
OUTPUT = OUTPUT_DIR / "rogii-hmm-scale2-functional-ridge-pls-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _one(
    notebook: dict[str, Any], predicate: Callable[[dict[str, Any]], bool], label: str
) -> dict[str, Any]:
    cells = [cell for cell in notebook["cells"] if predicate(cell)]
    if len(cells) != 1:
        raise RuntimeError(f"expected one {label} cell, found {len(cells)}")
    return copy.deepcopy(cells[0])


def _marked(notebook: dict[str, Any], marker: str) -> dict[str, Any]:
    return _one(notebook, lambda cell: bool(cell.get("metadata", {}).get(marker)), marker)


def _code_cell(code: str, marker: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {marker: True},
        "outputs": [],
        "source": code.strip(),
    }


def _snapshot_cell() -> dict[str, Any]:
    code = r'''
# Preserve the exact HMM intermediate before the additive Functional overlay.
import shutil as _combo_snapshot_shutil
from pathlib import Path as _ComboSnapshotPath

_COMBO_SNAPSHOT_WORK = (
    _ComboSnapshotPath('/kaggle/working')
    if _ComboSnapshotPath('/kaggle/working').exists()
    else _ComboSnapshotPath('.')
)
_COMBO_HMM_SNAPSHOT = _COMBO_SNAPSHOT_WORK / 'submission_after_dip_rate_hmm_scale2.csv'
_combo_snapshot_shutil.copyfile(_COMBO_SNAPSHOT_WORK / 'submission.csv', _COMBO_HMM_SNAPSHOT)
print('preserved HMM scale2 intermediate:', _COMBO_HMM_SNAPSHOT, flush=True)
'''
    return _code_cell(code, "hmm_scale2_intermediate_snapshot")


def _combined_contract_cell() -> dict[str, Any]:
    code = r'''
# Cross-component contract for the conditionally authorized additive candidate.
import hashlib as _combo_hashlib
import json as _combo_json
from pathlib import Path as _ComboPath

import numpy as _combo_np
import pandas as _combo_pd

_COMBO_WORK = (
    _ComboPath('/kaggle/working')
    if _ComboPath('/kaggle/working').exists()
    else _ComboPath('.')
)
_COMBO_SUB = _COMBO_WORK / 'submission.csv'
_COMBO_HMM_SUB = _COMBO_WORK / 'submission_after_dip_rate_hmm_scale2.csv'
_COMBO_FR_BEFORE = _COMBO_WORK / 'submission_before_functional_ridge_pls_ensemble.csv'
_COMBO_HMM_REPORT = _COMBO_WORK / 'dip_rate_hmm_deploy_report.json'
_COMBO_FR_REPORT = _COMBO_WORK / 'functional_ridge_pls_ensemble_deploy_report.json'
_COMBO_AUDIT = _COMBO_WORK / 'submission_audit.json'
_COMBO_REPORT = _COMBO_WORK / 'hmm_scale2_functional_ridge_pls_combined_report.json'
for _combo_required in (
    _COMBO_SUB,
    _COMBO_HMM_SUB,
    _COMBO_FR_BEFORE,
    _COMBO_HMM_REPORT,
    _COMBO_FR_REPORT,
    _COMBO_AUDIT,
):
    if not _combo_required.exists():
        raise RuntimeError(f'missing additive-candidate input: {_combo_required}')

_combo_hmm = _combo_json.loads(_COMBO_HMM_REPORT.read_text(encoding='utf-8'))
_combo_fr = _combo_json.loads(_COMBO_FR_REPORT.read_text(encoding='utf-8'))
_combo_audit = _combo_json.loads(_COMBO_AUDIT.read_text(encoding='utf-8'))
_combo_hmm_sub = _combo_pd.read_csv(_COMBO_HMM_SUB, dtype={'id': 'string'})
_combo_fr_before = _combo_pd.read_csv(_COMBO_FR_BEFORE, dtype={'id': 'string'})
if list(_combo_hmm_sub.columns) != ['id', 'tvt']:
    raise RuntimeError('HMM intermediate schema mismatch')
if not _combo_hmm_sub['id'].astype(str).equals(_combo_fr_before['id'].astype(str)):
    raise RuntimeError('HMM and Functional intermediate ID order differs')
_combo_hmm_values = _combo_pd.to_numeric(
    _combo_hmm_sub['tvt'], errors='coerce'
).to_numpy(dtype=float)
_combo_fr_before_values = _combo_pd.to_numeric(
    _combo_fr_before['tvt'], errors='coerce'
).to_numpy(dtype=float)
if not _combo_np.array_equal(_combo_hmm_values, _combo_fr_before_values):
    raise RuntimeError('Functional overlay did not start from the HMM intermediate')

_combo_sha = _combo_hashlib.sha256(_COMBO_SUB.read_bytes()).hexdigest()
_combo_hmm_sha = _combo_hashlib.sha256(_COMBO_HMM_SUB.read_bytes()).hexdigest()
if _combo_hmm_sha != str(_combo_hmm['submission_sha256']):
    raise RuntimeError('HMM intermediate SHA differs from its report')
if _combo_sha != str(_combo_fr['submission_sha256']):
    raise RuntimeError('combined final SHA differs from Functional report')
if _combo_sha != str(_combo_audit['sha256_submission_csv']):
    raise RuntimeError('combined final SHA differs from inherited audit')

_combo_hmm_parameters = _combo_hmm.get('parameters', {})
if _combo_hmm_parameters.get('overlay_scale') != 0.4:
    raise RuntimeError('combined HMM scale is not frozen at 0.4')
if _combo_hmm_parameters.get('maximum_final_movement') != 2.0:
    raise RuntimeError('combined HMM movement cap is not frozen at 2 ft')
if _combo_hmm.get('confirmation_test_wells_excluded') is not True:
    raise RuntimeError('HMM confirmation test-well exclusion is missing')
_combo_fr_parameters = _combo_fr.get('parameters', {})
if _combo_fr_parameters.get('ridge_weight') != 0.75:
    raise RuntimeError('combined Ridge weight mismatch')
if _combo_fr_parameters.get('pls_weight') != 0.25:
    raise RuntimeError('combined PLS weight mismatch')
if _combo_fr_parameters.get('overlay_scale') != 0.2:
    raise RuntimeError('combined Functional overlay scale mismatch')
if _combo_fr_parameters.get('weight_grid_searched') is not False:
    raise RuntimeError('combined candidate unexpectedly searched a weight grid')
for _combo_component in (_combo_hmm, _combo_fr):
    for _combo_flag in (
        'fixed_test_well_used',
        'fixed_visible_rows_used',
        'expected_visible_hash_used',
        'leaderboard_score_used_for_selection',
        'probe_flags_used',
    ):
        if _combo_component.get(_combo_flag) is not False:
            raise RuntimeError(f'combined runtime contract flag mismatch: {_combo_flag}')

_combo_report = {
    'candidate': 'HMM scale2 plus fixed 75/25 Functional Ridge/PLS additive overlay',
    'authorization_status': 'PREPARED_NOT_AUTHORIZED',
    'candidate_selection_public_score_informed': False,
    'runtime_reads_leaderboard_score': False,
    'runtime_uses_fixed_test_well': False,
    'runtime_uses_fixed_visible_rows': False,
    'runtime_uses_expected_visible_hash': False,
    'probe_flags_used': False,
    'authorization_required_before_execution': {
        'hmm_scale2_ref_55181026_public_rmse_strictly_below_6_477': True,
        'functional_ridge_pls_ref_55179387_public_rmse_strictly_below_6_477': True,
    },
    'preregistered_additive_oof': {
        'evaluation_wells': 738,
        'spatial_additional_gain_over_hmm_scale2': 0.036776586472816675,
        'spatial_improved_folds_over_hmm_scale2': 6,
        'spatial_fold_count': 7,
        'typewell_additional_gain_over_hmm_scale2': 0.02856978122130016,
        'typewell_improved_folds_over_hmm_scale2': 5,
        'typewell_fold_count': 5,
        'tail_risk_disclosed': True,
        'spatial_true_p95_well_delta_over_hmm_scale2': 0.5743766601900632,
        'typewell_true_p95_well_delta_over_hmm_scale2': 0.5857201872153267,
    },
    'parameters': {
        'hmm_overlay_scale': float(OVERLAY_SCALE),
        'hmm_maximum_movement_ft': float(OVERLAY_SCALE * DISAGREEMENT_CLIP),
        'ridge_weight': float(_FR_RIDGE_WEIGHT),
        'pls_weight': float(_FR_PLS_WEIGHT),
        'functional_overlay_scale': float(_FR_OVERLAY_SCALE),
    },
    'rows': int(_combo_fr['rows']),
    'hmm_changed_rows': int(_combo_hmm['changed_rows']),
    'functional_changed_rows': int(_combo_fr['changed_rows']),
    'hmm_intermediate_sha256': _combo_hmm_sha,
    'submission_sha256': _combo_sha,
    'elapsed_seconds': float(_fr_time.time() - _FR_START_TIME),
}
_COMBO_REPORT.write_text(
    _combo_json.dumps(_combo_report, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print('HMM scale2 + Functional Ridge/PLS combined contract: PASS', _combo_report, flush=True)
'''
    return _code_cell(code, "hmm_scale2_functional_ridge_pls_combined_contract")


def _merge_list(left: list[Any], right: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in [*left, *right]:
        if value not in result:
            result.append(value)
    return result


def main() -> None:
    hmm = json.loads(HMM_NOTEBOOK.read_text(encoding="utf-8"))
    functional = json.loads(FUNCTIONAL_NOTEBOOK.read_text(encoding="utf-8"))
    hmm_metadata = json.loads(HMM_METADATA.read_text(encoding="utf-8"))
    functional_metadata = json.loads(FUNCTIONAL_METADATA.read_text(encoding="utf-8"))
    if hmm_metadata.get("id") != "heralces/rogii-dip-rate-hmm-scale2-v1":
        raise RuntimeError("unexpected HMM scale2 source Notebook")
    if functional_metadata.get("id") != "heralces/rogii-functional-ridge-pls-v1":
        raise RuntimeError("unexpected Functional Ridge/PLS source Notebook")

    hmm_start = _marked(hmm, "dip_rate_hmm_start")
    functional_start = _marked(functional, "functional_ridge_start")
    hmm_overlay = _marked(hmm, "dip_rate_hmm_overlay_code")
    hmm_contract = _marked(hmm, "dip_rate_hmm_final_contract")
    functional_bootstrap = _marked(functional, "functional_ridge_source_bootstrap")
    functional_overlay = _marked(functional, "functional_ridge_pls_overlay_code")
    functional_contract = _marked(functional, "functional_ridge_pls_final_contract")
    parent_audit = _one(
        hmm,
        lambda cell: _source(cell).startswith(
            "# Final submission audit: verify the final file"
        ),
        "inherited final submission audit",
    )
    parent_cells = [
        copy.deepcopy(cell)
        for cell in hmm["cells"]
        if "dip_rate_hmm_parent_index" in cell.get("metadata", {})
        and cell is not parent_audit
        and not _source(cell).startswith("# Final submission audit: verify the final file")
    ]
    parent_cells.sort(key=lambda cell: int(cell["metadata"]["dip_rate_hmm_parent_index"]))

    notebook = copy.deepcopy(hmm)
    notebook["cells"] = [
        {
            "cell_type": "markdown",
            "metadata": {"hmm_scale2_functional_ridge_pls_intro": True},
            "source": (
                "# ROGII HMM scale2 + Functional Ridge/PLS v1\n\n"
                "Prepared, not authorized. This additive candidate may run only if "
                "both independent component submissions first score strictly below "
                "6.477. OOF improves over HMM scale2 in 6/7 spatial and 5/5 Typewell "
                "folds, but positive well-level p95 deltas disclose material tail risk."
            ),
        },
        hmm_start,
        functional_start,
        *parent_cells,
        hmm_overlay,
        _snapshot_cell(),
        hmm_contract,
        functional_bootstrap,
        functional_overlay,
        parent_audit,
        functional_contract,
        _combined_contract_cell(),
    ]
    notebook.setdefault("metadata", {})["hmm_scale2_functional_ridge_pls_build"] = {
        "source_notebooks": [hmm_metadata["id"], functional_metadata["id"]],
        "authorization_status": "PREPARED_NOT_AUTHORIZED",
        "execution_authorized": False,
        "authorization_gate": (
            "refs 55181026 and 55179387 must both have terminal Public RMSE < 6.477"
        ),
        "component_order": ["hmm_scale2", "functional_ridge_pls"],
        "hmm_overlay_scale": 0.4,
        "hmm_maximum_movement_ft": 2.0,
        "ridge_weight": 0.75,
        "pls_weight": 0.25,
        "functional_overlay_scale": 0.2,
        "weight_grid_searched": False,
        "current_test_wells_excluded_from_oof": True,
        "leaderboard_score_used_for_oof_selection": False,
        "spatial_additional_gain_over_hmm_scale2": 0.036776586472816675,
        "spatial_improved_folds_over_hmm_scale2": "6/7",
        "typewell_additional_gain_over_hmm_scale2": 0.02856978122130016,
        "typewell_improved_folds_over_hmm_scale2": "5/5",
        "tail_risk_disclosed": True,
    }

    metadata = copy.deepcopy(hmm_metadata)
    metadata.update(
        {
            "id": "heralces/rogii-hmm-scale2-functional-ridge-pls-v1",
            "title": "ROGII HMM Scale2 Functional Ridge PLS v1",
            "code_file": OUTPUT.name,
            "is_private": True,
            "enable_internet": False,
            "keywords": ["geology", "hmm", "ridge", "pls", "target-free"],
        }
    )
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        metadata[key] = _merge_list(
            list(hmm_metadata.get(key, [])), list(functional_metadata.get(key, []))
        )

    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    for forbidden in ("000d7d20", "00bbac68", "00e12e8b", "14151"):
        if forbidden in code:
            raise RuntimeError(f"fixed visible-test fragment survived: {forbidden}")
    for marker in (
        "dip_rate_hmm_overlay_code",
        "hmm_scale2_intermediate_snapshot",
        "functional_ridge_source_bootstrap",
        "functional_ridge_pls_overlay_code",
        "hmm_scale2_functional_ridge_pls_combined_contract",
    ):
        _marked(notebook, marker)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
