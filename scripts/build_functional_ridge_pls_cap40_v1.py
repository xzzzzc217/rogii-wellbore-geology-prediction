"""Build, but do not push, the gated cap4 plus Ridge/PLS notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import build_functional_ridge_pls_v1 as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_pls_cap40_v1"
OUTPUT = OUTPUT_DIR / "rogii-functional-ridge-pls-cap40-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
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


def _combined_contract_cell() -> dict[str, object]:
    code = r'''
# Cross-component contract for conditionally authorized cap4 + Ridge/PLS.
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
_COMBO_BRANCH = _COMBO_WORK / 'pf_seed_branch_hedge_report.csv'
_COMBO_FUNCTIONAL = (
    _COMBO_WORK / 'functional_ridge_pls_ensemble_deploy_report.json'
)
_COMBO_REPORT = _COMBO_WORK / 'functional_ridge_pls_cap40_combined_report.json'
for _combo_required in (_COMBO_SUB, _COMBO_BRANCH, _COMBO_FUNCTIONAL):
    if not _combo_required.exists():
        raise RuntimeError(f'missing combined-candidate input: {_combo_required}')
if not _combo_np.isclose(float(_BH_STRENGTH), 0.62, atol=0.0, rtol=0.0):
    raise RuntimeError('combined candidate branch strength is not frozen at 0.62')
if not _combo_np.isclose(float(_BH_CAP), 4.0, atol=0.0, rtol=0.0):
    raise RuntimeError('combined candidate branch cap is not frozen at 4.0')

_combo_branch = _combo_pd.read_csv(_COMBO_BRANCH)
_combo_functional = _combo_json.loads(_COMBO_FUNCTIONAL.read_text(encoding='utf-8'))
_combo_required_columns = {'well', 'reason', 'shift', 'moved_rows'}
if not _combo_required_columns.issubset(_combo_branch.columns):
    raise RuntimeError('combined candidate branch report is incomplete')
_combo_applied = _combo_branch[
    _combo_branch['reason'].astype(str).str.lower().eq('applied')
].copy()
_combo_shifts = _combo_pd.to_numeric(
    _combo_applied['shift'], errors='raise'
).to_numpy(dtype=float)
if len(_combo_shifts) and bool((_combo_np.abs(_combo_shifts) > 4.0 + 1e-12).any()):
    raise RuntimeError('combined candidate exceeded the dynamic branch cap')
if _combo_functional.get('target_free_features_only') is not True:
    raise RuntimeError('combined candidate lost the target-free contract')
if _combo_functional.get('leaderboard_score_used_for_selection') is not False:
    raise RuntimeError('Ridge/PLS runtime unexpectedly used a leaderboard score')
_combo_parameters = _combo_functional.get('parameters', {})
if _combo_parameters.get('ridge_weight') != 0.75:
    raise RuntimeError('combined candidate Ridge weight mismatch')
if _combo_parameters.get('pls_weight') != 0.25:
    raise RuntimeError('combined candidate PLS weight mismatch')
if _combo_parameters.get('weight_grid_searched') is not False:
    raise RuntimeError('combined candidate unexpectedly searched a weight grid')

_combo_sha = _combo_hashlib.sha256(_COMBO_SUB.read_bytes()).hexdigest()
if _combo_sha != str(_combo_functional['submission_sha256']):
    raise RuntimeError('combined final SHA differs from the Ridge/PLS report')
_combo_report = {
    'candidate': 'PF branch cap4 plus fixed 75/25 Functional Ridge/PLS overlay',
    'candidate_selection_public_score_informed': True,
    'runtime_reads_leaderboard_score': False,
    'runtime_uses_fixed_test_well': False,
    'runtime_uses_fixed_visible_rows': False,
    'runtime_uses_expected_visible_hash': False,
    'probe_flags_used': False,
    'authorization_required_before_submission': {
        'cap40_public_score_strictly_below_6_477': True,
        'functional_ridge_pls_public_score_strictly_below_6_477': True,
    },
    'parameters': {
        'pf_branch_strength': float(_BH_STRENGTH),
        'pf_branch_shift_cap': float(_BH_CAP),
        'ridge_weight': float(_FR_RIDGE_WEIGHT),
        'pls_weight': float(_FR_PLS_WEIGHT),
        'functional_scale': float(_FR_OVERLAY_SCALE),
    },
    'branch_applied_well_count': int(len(_combo_applied)),
    'branch_applied_wells': sorted(_combo_applied['well'].astype(str).tolist()),
    'branch_moved_rows': int(
        _combo_pd.to_numeric(_combo_applied['moved_rows'], errors='raise').sum()
    ),
    'rows': int(_combo_functional['rows']),
    'functional_changed_rows': int(_combo_functional['changed_rows']),
    'submission_sha256': _combo_sha,
    'elapsed_seconds': float(_fr_time.time() - _FR_START_TIME),
}
_COMBO_REPORT.write_text(
    _combo_json.dumps(_combo_report, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print('Functional Ridge/PLS cap4 combined contract: PASS', _combo_report, flush=True)
'''
    return _code_cell(code, "functional_ridge_pls_cap40_combined_contract")


def main() -> None:
    base.main()
    notebook = json.loads(base.OUTPUT.read_text(encoding="utf-8"))
    metadata = json.loads(base.METADATA.read_text(encoding="utf-8"))

    branch_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_parent_index") == 53
    )
    branch_source = _source(branch_cell)
    branch_source = _replace_once(
        branch_source,
        "_BH_STRENGTH = 0.60",
        "_BH_STRENGTH = 0.62",
        "PF branch strength",
    )
    branch_source = _replace_once(
        branch_source,
        "_BH_CAP = 2.00",
        "_BH_CAP = 4.00",
        "PF branch cap",
    )
    branch_cell["source"] = branch_source

    notebook["cells"].insert(
        1,
        {
            "cell_type": "markdown",
            "metadata": {"functional_ridge_pls_cap40_combined": True},
            "source": (
                "## Conditional cap4 + fixed Ridge/PLS combination\n\n"
                "Prepared but not authorized until both individual candidates "
                "score strictly better than 6.477. Runtime scoring reads no "
                "leaderboard score, fixed test well, visible row count, or "
                "expected output hash."
            ),
        },
    )
    notebook["cells"].append(_combined_contract_cell())
    notebook.setdefault("metadata", {})["functional_ridge_pls_cap40_build"] = {
        "source_builder": "build_functional_ridge_pls_v1.py",
        "pf_branch_strength": 0.62,
        "pf_branch_shift_cap": 4.0,
        "ridge_weight": 0.75,
        "pls_weight": 0.25,
        "weight_grid_searched": False,
        "candidate_selection_public_score_informed": True,
        "runtime_reads_leaderboard_score": False,
        "fixed_test_well_used": False,
        "expected_visible_hash_used": False,
        "authorization_status": "PREPARED_NOT_AUTHORIZED",
        "authorization_gate": (
            "cap40 and Functional Ridge/PLS individual Public RMSE must both be < 6.477"
        ),
    }

    metadata = copy.deepcopy(metadata)
    metadata["id"] = "heralces/rogii-functional-ridge-pls-cap40-v1"
    metadata["title"] = "ROGII Functional Ridge PLS Cap40 v1"
    metadata["code_file"] = OUTPUT.name

    combined_code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    if combined_code.count("_BH_STRENGTH = 0.62") != 1:
        raise RuntimeError("combined branch strength was not installed exactly once")
    if combined_code.count("_BH_CAP = 4.00") != 1:
        raise RuntimeError("combined branch cap was not installed exactly once")
    prohibited = (
        "af43d97ab5b76f7f0bbbc5c72fb6e9a829e939c48582fc4130144340547360a2",
        "d4d7e368c940d84cdc1504fcad18809675f18499b0efd6e089011b21febb7da3",
        "_EX_EXPECTED_WELL",
        "_EX_EXPECTED_ROWS",
        "_EX_EXPECTED_BASE_SHA",
    )
    survived = [fragment for fragment in prohibited if fragment in combined_code]
    if survived:
        raise RuntimeError(f"prohibited combined-candidate fragments survived: {survived}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
