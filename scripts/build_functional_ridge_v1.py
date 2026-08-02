"""Build the private Functional Ridge overlay Kaggle notebook.

The notebook retains the current 6.477 public full-stack parent and adds one
small, preregistered residual overlay after the PF branch layer.  The overlay
is trained from the private sequence-pilot cache and uses only target-free
well-functional features.  It is deliberately audited separately from the
inherited parent stack: the target-free contract applies to this new layer,
not retroactively to every parent component.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT / "kaggle_notebooks" / "pf_public_reproduction"
PARENT = PARENT_DIR / "rogii-pf-public-reproduction.ipynb"
PARENT_METADATA = PARENT_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_v1"
OUTPUT = OUTPUT_DIR / "rogii-functional-ridge-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"
TRAJECTORY_SOURCE = ROOT / "src" / "rogii" / "trajectory.py"
RESIDUAL_SOURCE = ROOT / "src" / "rogii" / "residual_gbdt.py"

CONFIG_CELL = 7
DEMO_CELL = 36
BRANCH_CELL = 53
FINAL_AUDIT_CELL = 54
CACHE_DATASET = "heralces/rogii-private-sequence-pilot-cache-v1"


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _start_cell() -> dict[str, object]:
    return _code_cell(
        """
import time as _fr_time
_FR_START_TIME = _fr_time.time()
""",
        "functional_ridge_start",
    )


def _source_bootstrap_cell() -> dict[str, object]:
    trajectory = TRAJECTORY_SOURCE.read_text(encoding="utf-8")
    residual = RESIDUAL_SOURCE.read_text(encoding="utf-8")
    code = f"""
# Install the repository's audited target-free feature builder into this run.
import hashlib as _fr_source_hashlib
import sys as _fr_source_sys
from pathlib import Path as _FrSourcePath

_FR_SOURCE_ROOT = _FrSourcePath('/kaggle/working/_functional_ridge_src')
_FR_SOURCE_PACKAGE = _FR_SOURCE_ROOT / 'rogii'
_FR_SOURCE_PACKAGE.mkdir(parents=True, exist_ok=True)
(_FR_SOURCE_PACKAGE / '__init__.py').write_text('', encoding='utf-8')
(_FR_SOURCE_PACKAGE / 'trajectory.py').write_text({trajectory!r}, encoding='utf-8')
(_FR_SOURCE_PACKAGE / 'residual_gbdt.py').write_text({residual!r}, encoding='utf-8')
if str(_FR_SOURCE_ROOT) not in _fr_source_sys.path:
    _fr_source_sys.path.insert(0, str(_FR_SOURCE_ROOT))

_FR_SOURCE_SHA256 = {{
    'trajectory.py': _fr_source_hashlib.sha256(
        (_FR_SOURCE_PACKAGE / 'trajectory.py').read_bytes()
    ).hexdigest(),
    'residual_gbdt.py': _fr_source_hashlib.sha256(
        (_FR_SOURCE_PACKAGE / 'residual_gbdt.py').read_bytes()
    ).hexdigest(),
}}
print('Functional Ridge source SHA256:', _FR_SOURCE_SHA256, flush=True)
"""
    return _code_cell(code, "functional_ridge_source_bootstrap")


def _overlay_cell() -> dict[str, object]:
    code = r'''
# Preregistered target-free well-functional residual Ridge overlay.
import hashlib as _fr_hashlib
import json as _fr_json
from pathlib import Path as _FrPath

import numpy as _fr_np
import pandas as _fr_pd
from sklearn.impute import SimpleImputer as _FrSimpleImputer
from sklearn.linear_model import Ridge as _FrRidge
from sklearn.pipeline import make_pipeline as _fr_make_pipeline
from sklearn.preprocessing import StandardScaler as _FrStandardScaler

from rogii.residual_gbdt import (
    RESIDUAL_FEATURE_NAMES as _FR_ALL_FEATURE_NAMES,
    build_residual_features as _fr_build_residual_features,
)

_FR_SIGNATURE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
_FR_CURVE_POINTS = 64
_FR_RIDGE_ALPHA = 100.0
_FR_OVERLAY_SCALE = 0.20
_FR_RESIDUAL_CLIP = 20.0
_FR_WORK = _FrPath('/kaggle/working') if _FrPath('/kaggle/working').exists() else _FrPath('.')
_FR_SUBMISSION = _FR_WORK / 'submission.csv'
_FR_BEFORE = _FR_WORK / 'submission_before_functional_ridge.csv'
_FR_PER_WELL = _FR_WORK / 'functional_ridge_per_well.csv'
_FR_REPORT = _FR_WORK / 'functional_ridge_deploy_report.json'


def _fr_sha256(path):
    return _fr_hashlib.sha256(_FrPath(path).read_bytes()).hexdigest()


def _fr_signature(features):
    features = _fr_np.asarray(features, dtype=_fr_np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise RuntimeError('Functional Ridge requires a non-empty feature matrix')
    indices = _fr_np.rint(
        _fr_np.asarray(_FR_SIGNATURE_QUANTILES) * (len(features) - 1)
    ).astype(_fr_np.int64)
    return features[indices].reshape(-1)


def _fr_curve(residual):
    residual = _fr_np.asarray(residual, dtype=float)
    clipped = _fr_np.clip(residual, -_FR_RESIDUAL_CLIP, _FR_RESIDUAL_CLIP)
    return _fr_np.interp(
        _fr_np.linspace(0.0, 1.0, _FR_CURVE_POINTS),
        _fr_np.linspace(0.0, 1.0, len(clipped)),
        clipped,
    ).astype(_fr_np.float32)


_fr_required_cache_files = {
    'sequence_cache_metadata.json',
    'features.float16.npy',
    'competitive_base.float32.npy',
    'truth.float32.npy',
    'well_offsets.int64.npy',
}
_fr_cache_candidates = []
for _fr_meta_path in _FrPath('/kaggle/input').rglob('sequence_cache_metadata.json'):
    _fr_candidate = _fr_meta_path.parent
    if all((_fr_candidate / name).exists() for name in _fr_required_cache_files):
        _fr_cache_candidates.append(_fr_candidate)
if len(_fr_cache_candidates) != 1:
    raise RuntimeError(
        f'expected exactly one sequence-pilot cache, found {_fr_cache_candidates}'
    )
_FR_CACHE = _fr_cache_candidates[0]

_fr_metadata = _fr_json.loads(
    (_FR_CACHE / 'sequence_cache_metadata.json').read_text(encoding='utf-8')
)
_fr_features = _fr_np.load(_FR_CACHE / 'features.float16.npy', mmap_mode='r')
_fr_base = _fr_np.load(_FR_CACHE / 'competitive_base.float32.npy', mmap_mode='r')
_fr_truth = _fr_np.load(_FR_CACHE / 'truth.float32.npy', mmap_mode='r')
_fr_offsets = _fr_np.load(_FR_CACHE / 'well_offsets.int64.npy')
_fr_well_count = len(_fr_offsets) - 1
if _fr_well_count != len(_fr_metadata['per_well']):
    raise RuntimeError('cache metadata and well offsets disagree')
if int(_fr_offsets[-1]) != len(_fr_features):
    raise RuntimeError('cache feature and offset rows disagree')
if len(_fr_base) != len(_fr_truth) or len(_fr_base) != len(_fr_features):
    raise RuntimeError('cache arrays disagree')
_fr_cache_contract = _fr_metadata.get('contract', {})
if _fr_cache_contract.get('absolute_position_features_removed') is not True:
    raise RuntimeError('cache does not certify removal of absolute-position features')
if _fr_cache_contract.get('training_rows_only') is not True:
    raise RuntimeError('cache is not certified as training-rows-only')
if _fr_cache_contract.get('target_feature_leakage') is not False:
    raise RuntimeError('cache target-feature leakage contract is not false')
if _fr_cache_contract.get('well_id_encoded_as_feature') is not False:
    raise RuntimeError('cache unexpectedly encodes well identity as a feature')

_fr_cache_feature_names = [str(name) for name in _fr_metadata['features']]
_fr_missing_names = [
    name for name in _fr_cache_feature_names if name not in _FR_ALL_FEATURE_NAMES
]
if _fr_missing_names:
    raise RuntimeError(f'cache features missing from runtime builder: {_fr_missing_names}')
_fr_feature_indices = _fr_np.asarray(
    [_FR_ALL_FEATURE_NAMES.index(name) for name in _fr_cache_feature_names],
    dtype=_fr_np.int64,
)
if _fr_features.shape[1] != len(_fr_cache_feature_names):
    raise RuntimeError('cache feature width disagrees with metadata')

_fr_signatures = _fr_np.empty(
    (_fr_well_count, len(_fr_cache_feature_names) * len(_FR_SIGNATURE_QUANTILES)),
    dtype=_fr_np.float32,
)
_fr_curves = _fr_np.empty((_fr_well_count, _FR_CURVE_POINTS), dtype=_fr_np.float32)
for _fr_well_index in range(_fr_well_count):
    _fr_start = int(_fr_offsets[_fr_well_index])
    _fr_end = int(_fr_offsets[_fr_well_index + 1])
    _fr_signatures[_fr_well_index] = _fr_signature(
        _fr_np.asarray(_fr_features[_fr_start:_fr_end], dtype=_fr_np.float32)
    )
    _fr_curves[_fr_well_index] = _fr_curve(
        _fr_np.asarray(_fr_truth[_fr_start:_fr_end], dtype=float)
        - _fr_np.asarray(_fr_base[_fr_start:_fr_end], dtype=float)
    )

_fr_model = _fr_make_pipeline(
    _FrSimpleImputer(strategy='median'),
    _FrStandardScaler(),
    _FrRidge(alpha=_FR_RIDGE_ALPHA),
)
_fr_model.fit(_fr_signatures, _fr_curves)

_fr_data = _FrPath(
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
_fr_test = _fr_data / 'test'
_fr_sample_path = _fr_data / 'sample_submission.csv'
if not _fr_test.exists() or not _fr_sample_path.exists() or not _FR_SUBMISSION.exists():
    raise RuntimeError('Functional Ridge deployment inputs are missing')

_fr_submission = _fr_pd.read_csv(_FR_SUBMISSION, dtype={'id': 'string'})
_fr_sample = _fr_pd.read_csv(_fr_sample_path, dtype={'id': 'string'})
if list(_fr_submission.columns) != ['id', 'tvt']:
    raise RuntimeError('Functional Ridge expected id,tvt submission schema')
if len(_fr_submission) != len(_fr_sample):
    raise RuntimeError('Functional Ridge submission row count differs from sample')
if not _fr_submission['id'].astype(str).equals(_fr_sample['id'].astype(str)):
    raise RuntimeError('Functional Ridge submission ID order differs from sample')
if not _fr_submission['id'].is_unique:
    raise RuntimeError('Functional Ridge submission IDs are not unique')
_fr_before_values = _fr_pd.to_numeric(
    _fr_submission['tvt'], errors='coerce'
).to_numpy(dtype=float)
if not _fr_np.isfinite(_fr_before_values).all():
    raise RuntimeError('Functional Ridge base predictions are non-finite')
_fr_submission.to_csv(_FR_BEFORE, index=False)

_fr_corrections = {}
_fr_rows = []
for _fr_horizontal_path in sorted(_fr_test.glob('*__horizontal_well.csv')):
    _fr_well = _fr_horizontal_path.name.removesuffix('__horizontal_well.csv')
    _fr_typewell_path = _fr_test / f'{_fr_well}__typewell.csv'
    if not _fr_typewell_path.exists():
        raise RuntimeError(f'missing typewell file for {_fr_well}')
    _fr_horizontal = _fr_pd.read_csv(_fr_horizontal_path)
    _fr_typewell = _fr_pd.read_csv(_fr_typewell_path)
    _fr_block = _fr_build_residual_features(_fr_horizontal, _fr_typewell)
    _fr_runtime_features = _fr_np.asarray(
        _fr_block.features[:, _fr_feature_indices], dtype=_fr_np.float32
    )
    _fr_predicted_curve = _fr_np.asarray(
        _fr_model.predict(_fr_signature(_fr_runtime_features)[None, :])[0],
        dtype=float,
    )
    _fr_correction = _FR_OVERLAY_SCALE * _fr_np.clip(
        _fr_np.interp(
            _fr_np.linspace(0.0, 1.0, len(_fr_block.hidden_indices)),
            _fr_np.linspace(0.0, 1.0, _FR_CURVE_POINTS),
            _fr_predicted_curve,
        ),
        -_FR_RESIDUAL_CLIP,
        _FR_RESIDUAL_CLIP,
    )
    if not _fr_np.isfinite(_fr_correction).all():
        raise RuntimeError(f'non-finite Functional Ridge correction for {_fr_well}')
    for _fr_hidden_index, _fr_value in zip(
        _fr_block.hidden_indices.tolist(), _fr_correction.tolist(), strict=True
    ):
        _fr_id = f'{_fr_well}_{int(_fr_hidden_index)}'
        if _fr_id in _fr_corrections:
            raise RuntimeError(f'duplicate Functional Ridge row ID: {_fr_id}')
        _fr_corrections[_fr_id] = float(_fr_value)
    _fr_rows.append(
        {
            'well': _fr_well,
            'rows': int(len(_fr_correction)),
            'mean_correction': float(_fr_np.mean(_fr_correction)),
            'rms_correction': float(_fr_np.sqrt(_fr_np.mean(_fr_correction**2))),
            'minimum_correction': float(_fr_np.min(_fr_correction)),
            'maximum_correction': float(_fr_np.max(_fr_correction)),
        }
    )

_fr_ids = _fr_submission['id'].astype(str).tolist()
if set(_fr_corrections) != set(_fr_ids):
    _fr_missing = sorted(set(_fr_ids) - set(_fr_corrections))[:5]
    _fr_extra = sorted(set(_fr_corrections) - set(_fr_ids))[:5]
    raise RuntimeError(
        f'Functional Ridge coverage mismatch: missing={_fr_missing} '
        f'extra={_fr_extra}'
    )
_fr_delta = _fr_np.asarray([_fr_corrections[row_id] for row_id in _fr_ids], dtype=float)
_fr_final_values = _fr_before_values + _fr_delta
if not _fr_np.isfinite(_fr_final_values).all():
    raise RuntimeError('Functional Ridge final predictions are non-finite')
_fr_submission['tvt'] = _fr_final_values
_fr_submission.to_csv(_FR_SUBMISSION, index=False)
_fr_pd.DataFrame(_fr_rows).to_csv(_FR_PER_WELL, index=False)

_fr_report = {
    'candidate': 'well-functional residual Ridge overlay',
    'parent_stack': 'heralces/rogii-pf-public-reproduction',
    'contract_scope': 'functional_ridge_overlay_only',
    'deployment_base_matches_oof_base_exactly': False,
    'deployment_base_risk': (
        'The inherited 6.477 full stack is related to but not identical to the '
        'competitive grouped-OOF base used to gate the overlay.'
    ),
    'target_free_features_only': True,
    'cache_training_rows_only': True,
    'well_id_encoded_as_feature': False,
    'absolute_coordinates_used': False,
    'fixed_test_well_used': False,
    'fixed_visible_rows_used': False,
    'expected_visible_hash_used': False,
    'leaderboard_score_used_for_selection': False,
    'probe_flags_used': False,
    'parameters': {
        'signature_quantiles': list(_FR_SIGNATURE_QUANTILES),
        'curve_points': int(_FR_CURVE_POINTS),
        'ridge_alpha': float(_FR_RIDGE_ALPHA),
        'overlay_scale': float(_FR_OVERLAY_SCALE),
        'residual_clip': float(_FR_RESIDUAL_CLIP),
        'feature_count': int(len(_fr_cache_feature_names)),
        'signature_width': int(_fr_signatures.shape[1]),
    },
    'preregistered_oof': {
        'spatial_cluster_7_seed_20260727': {
            'pooled_delta': -0.0276029,
            'improved_folds': 5,
            'fold_count': 7,
            'p95_well_delta': -0.07058,
            'worst_well_delta': 0.19252,
        },
        'typewell_fingerprint': {
            'pooled_delta': -0.0187671,
            'improved_folds': 5,
            'fold_count': 5,
            'p95_well_delta': -0.05606,
            'worst_well_delta': 0.01616,
        },
    },
    'cache_path': str(_FR_CACHE),
    'cache_wells': int(_fr_well_count),
    'cache_rows': int(len(_fr_features)),
    'test_wells': int(len(_fr_rows)),
    'rows': int(len(_fr_submission)),
    'changed_rows': int((_fr_np.abs(_fr_delta) > 1e-12).sum()),
    'correction_mean': float(_fr_np.mean(_fr_delta)),
    'correction_rms': float(_fr_np.sqrt(_fr_np.mean(_fr_delta**2))),
    'correction_minimum': float(_fr_np.min(_fr_delta)),
    'correction_maximum': float(_fr_np.max(_fr_delta)),
    'id_order_matches_sample': True,
    'ids_unique': True,
    'finite_predictions': True,
    'source_sha256': _FR_SOURCE_SHA256,
    'before_submission_sha256': _fr_sha256(_FR_BEFORE),
    'submission_sha256': _fr_sha256(_FR_SUBMISSION),
    'elapsed_seconds': float(_fr_time.time() - _FR_START_TIME),
}
_FR_REPORT.write_text(
    _fr_json.dumps(_fr_report, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print('Functional Ridge deployment report:', _fr_report, flush=True)
'''
    return _code_cell(code, "functional_ridge_overlay_code")


def _contract_cell() -> dict[str, object]:
    code = r'''
# Final cross-file contract after the inherited full-stack audit.
import hashlib as _fr_final_hashlib
import json as _fr_final_json
from pathlib import Path as _FrFinalPath

import numpy as _fr_final_np
import pandas as _fr_final_pd

_FR_FINAL_WORK = (
    _FrFinalPath('/kaggle/working')
    if _FrFinalPath('/kaggle/working').exists()
    else _FrFinalPath('.')
)
_FR_FINAL_SUB = _FR_FINAL_WORK / 'submission.csv'
_FR_FINAL_BEFORE = _FR_FINAL_WORK / 'submission_before_functional_ridge.csv'
_FR_FINAL_REPORT = _FR_FINAL_WORK / 'functional_ridge_deploy_report.json'
_FR_FINAL_PER_WELL = _FR_FINAL_WORK / 'functional_ridge_per_well.csv'
for _fr_required in (_FR_FINAL_SUB, _FR_FINAL_BEFORE, _FR_FINAL_REPORT, _FR_FINAL_PER_WELL):
    if not _fr_required.exists():
        raise RuntimeError(f'missing Functional Ridge contract output: {_fr_required}')

_fr_final_sub = _fr_final_pd.read_csv(_FR_FINAL_SUB, dtype={'id': 'string'})
_fr_final_before = _fr_final_pd.read_csv(_FR_FINAL_BEFORE, dtype={'id': 'string'})
_fr_final_per_well = _fr_final_pd.read_csv(_FR_FINAL_PER_WELL)
_fr_final_report = _fr_final_json.loads(_FR_FINAL_REPORT.read_text(encoding='utf-8'))
if list(_fr_final_sub.columns) != ['id', 'tvt'] or list(_fr_final_before.columns) != ['id', 'tvt']:
    raise RuntimeError('Functional Ridge contract schema mismatch')
if not _fr_final_sub['id'].astype(str).equals(_fr_final_before['id'].astype(str)):
    raise RuntimeError('Functional Ridge before/final ID order mismatch')
if len(_fr_final_sub) != int(_fr_final_report['rows']):
    raise RuntimeError('Functional Ridge report row count mismatch')
if int(_fr_final_per_well['rows'].sum()) != len(_fr_final_sub):
    raise RuntimeError('Functional Ridge per-well row total mismatch')
_fr_final_values = _fr_final_pd.to_numeric(
    _fr_final_sub['tvt'], errors='coerce'
).to_numpy(dtype=float)
_fr_final_before_values = _fr_final_pd.to_numeric(
    _fr_final_before['tvt'], errors='coerce'
).to_numpy(dtype=float)
if (
    not _fr_final_np.isfinite(_fr_final_values).all()
    or not _fr_final_np.isfinite(_fr_final_before_values).all()
):
    raise RuntimeError('Functional Ridge contract found non-finite values')
_fr_final_sha = _fr_final_hashlib.sha256(_FR_FINAL_SUB.read_bytes()).hexdigest()
if _fr_final_sha != str(_fr_final_report['submission_sha256']):
    raise RuntimeError('Functional Ridge report SHA differs from final submission')
_fr_final_delta = _fr_final_values - _fr_final_before_values
if int((_fr_final_np.abs(_fr_final_delta) > 1e-12).sum()) != int(_fr_final_report['changed_rows']):
    raise RuntimeError('Functional Ridge changed-row count mismatch')
if _fr_final_report['target_free_features_only'] is not True:
    raise RuntimeError('Functional Ridge target-free contract is not asserted')
print('Functional Ridge final contract: PASS', _fr_final_sha, flush=True)
'''
    return _code_cell(code, "functional_ridge_final_contract")


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    parent_sha256 = sha256_bytes(parent_bytes)
    parent = json.loads(parent_bytes)
    parent_metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))
    if parent_metadata.get("id") != "heralces/rogii-pf-public-reproduction":
        raise RuntimeError("unexpected Functional Ridge parent kernel")
    if len(parent["cells"]) <= FINAL_AUDIT_CELL:
        raise RuntimeError("Functional Ridge parent notebook is incomplete")

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
                    "else _demo_well()"
                ),
                "DEMO = _demo_well()",
                "fixed demo well",
            )
        cell["source"] = cell_source
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        cell.setdefault("metadata", {})["functional_ridge_parent_index"] = parent_index

    final_audit = copy.deepcopy(parent["cells"][FINAL_AUDIT_CELL])
    final_audit["source"] = source(final_audit)
    final_audit["execution_count"] = None
    final_audit["outputs"] = []
    final_audit.setdefault("metadata", {})[
        "functional_ridge_parent_index"
    ] = FINAL_AUDIT_CELL

    notebook = copy.deepcopy(parent)
    notebook["cells"] = [
        {
            "cell_type": "markdown",
            "metadata": {"functional_ridge_overlay": True},
            "source": (
                "# ROGII Functional Ridge overlay v1\n\n"
                "A private controlled verification on the frozen 6.477 full-stack. "
                "The new overlay was preregistered and passed both spatial-cluster "
                "7-fold and Typewell-fingerprint 5-fold OOF gates. Its contract is "
                "target-free and dynamic; that contract applies only to the added "
                "overlay, not to every inherited parent component."
            ),
        },
        _start_cell(),
        *cells,
        _source_bootstrap_cell(),
        _overlay_cell(),
        final_audit,
        _contract_cell(),
    ]
    notebook.setdefault("metadata", {})["functional_ridge_build"] = {
        "source_kernel": parent_metadata["id"],
        "source_notebook_sha256": parent_sha256,
        "retained_parent_cells": retained_indices + [FINAL_AUDIT_CELL],
        "excluded_parent_cells": list(range(FINAL_AUDIT_CELL + 1, len(parent["cells"]))),
        "overlay_inserted_after_parent_cell": BRANCH_CELL,
        "overlay_scale": 0.20,
        "ridge_alpha": 100.0,
        "curve_points": 64,
        "target_free_overlay": True,
        "parent_stack_target_free_claimed": False,
        "leaderboard_score_used_for_overlay_selection": False,
    }

    overlay_code = source(_overlay_cell())
    required_fragments = (
        "target_free_features_only",
        "leaderboard_score_used_for_selection",
        "deployment_base_matches_oof_base_exactly",
        "functional_ridge_deploy_report.json",
        "submission_before_functional_ridge.csv",
        "set(_fr_corrections) != set(_fr_ids)",
    )
    missing = [fragment for fragment in required_fragments if fragment not in overlay_code]
    if missing:
        raise RuntimeError(f"Functional Ridge overlay contract is incomplete: {missing}")

    dataset_sources = list(parent_metadata.get("dataset_sources", []))
    if CACHE_DATASET not in dataset_sources:
        dataset_sources.append(CACHE_DATASET)
    metadata = {
        "id": "heralces/rogii-functional-ridge-v1",
        "title": "ROGII Functional Ridge v1",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["geology", "ridge", "target-free"],
        "dataset_sources": dataset_sources,
        "kernel_sources": parent_metadata.get("kernel_sources", []),
        "competition_sources": parent_metadata.get("competition_sources", []),
        "model_sources": parent_metadata.get("model_sources", []),
    }

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
