"""Build the private fixed 75/25 Functional Ridge/PLS Kaggle notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_v1"
BASE_NOTEBOOK = BASE_DIR / "rogii-functional-ridge-v1.ipynb"
BASE_METADATA = BASE_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "functional_ridge_pls_v1"
OUTPUT = OUTPUT_DIR / "rogii-functional-ridge-pls-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected one {label} fragment, found {value.count(old)}")
    return value.replace(old, new)


def _transform_overlay(code: str) -> str:
    code = _replace_once(
        code,
        "from sklearn.impute import SimpleImputer as _FrSimpleImputer\n",
        (
            "from sklearn.cross_decomposition import PLSRegression as _FrPLSRegression\n"
            "from sklearn.impute import SimpleImputer as _FrSimpleImputer\n"
        ),
        "PLS import",
    )
    code = _replace_once(
        code,
        "_FR_RIDGE_ALPHA = 100.0\n_FR_OVERLAY_SCALE = 0.20",
        (
            "_FR_RIDGE_ALPHA = 100.0\n"
            "_FR_PLS_COMPONENTS = 8\n"
            "_FR_RIDGE_WEIGHT = 0.75\n"
            "_FR_PLS_WEIGHT = 0.25\n"
            "_FR_OVERLAY_SCALE = 0.20"
        ),
        "ensemble constants",
    )
    signature_end = """    return features[indices].reshape(-1)


def _fr_curve(residual):
"""
    augmented = """    return features[indices].reshape(-1)


def _fr_augmented_signature(features):
    values = _fr_np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise RuntimeError('Functional ensemble requires at least two feature rows')
    values = _fr_np.where(_fr_np.isfinite(values), values, _fr_np.nan)
    point_signature = _fr_signature(values)
    means = _fr_np.nanmean(values, axis=0)
    standard_deviations = _fr_np.nanstd(values, axis=0)
    position = _fr_np.linspace(-0.5, 0.5, len(values), dtype=float)[:, None]
    finite = _fr_np.isfinite(values)
    centered = values - means
    slope_numerator = _fr_np.nansum(position * centered, axis=0)
    slope_denominator = _fr_np.sum((position**2) * finite, axis=0)
    slopes = _fr_np.divide(
        slope_numerator,
        slope_denominator,
        out=_fr_np.full(values.shape[1], _fr_np.nan, dtype=float),
        where=slope_denominator > 0.0,
    )
    return _fr_np.concatenate(
        [point_signature, means, standard_deviations, slopes]
    ).astype(_fr_np.float32)


def _fr_curve(residual):
"""
    code = _replace_once(code, signature_end, augmented, "augmented signature")
    code = _replace_once(
        code,
        """_fr_curves = _fr_np.empty((_fr_well_count, _FR_CURVE_POINTS), dtype=_fr_np.float32)
for _fr_well_index in range(_fr_well_count):
""",
        """_fr_augmented_signatures = _fr_np.empty(
    (_fr_well_count, len(_fr_cache_feature_names) * (len(_FR_SIGNATURE_QUANTILES) + 3)),
    dtype=_fr_np.float32,
)
_fr_curves = _fr_np.empty((_fr_well_count, _FR_CURVE_POINTS), dtype=_fr_np.float32)
for _fr_well_index in range(_fr_well_count):
""",
        "augmented signature allocation",
    )
    code = _replace_once(
        code,
        """    _fr_signatures[_fr_well_index] = _fr_signature(
        _fr_np.asarray(_fr_features[_fr_start:_fr_end], dtype=_fr_np.float32)
    )
""",
        """    _fr_cache_well_features = _fr_np.asarray(
        _fr_features[_fr_start:_fr_end], dtype=_fr_np.float32
    )
    _fr_signatures[_fr_well_index] = _fr_signature(_fr_cache_well_features)
    _fr_augmented_signatures[_fr_well_index] = _fr_augmented_signature(
        _fr_cache_well_features
    )
""",
        "cache signatures",
    )
    code = _replace_once(
        code,
        """_fr_model = _fr_make_pipeline(
    _FrSimpleImputer(strategy='median'),
    _FrStandardScaler(),
    _FrRidge(alpha=_FR_RIDGE_ALPHA),
)
_fr_model.fit(_fr_signatures, _fr_curves)
""",
        """_fr_ridge_model = _fr_make_pipeline(
    _FrSimpleImputer(strategy='median'),
    _FrStandardScaler(),
    _FrRidge(alpha=_FR_RIDGE_ALPHA),
)
_fr_pls_model = _fr_make_pipeline(
    _FrSimpleImputer(strategy='median'),
    _FrPLSRegression(
        n_components=_FR_PLS_COMPONENTS,
        scale=True,
        max_iter=500,
        tol=1e-6,
    ),
)
_fr_ridge_model.fit(_fr_signatures, _fr_curves)
_fr_pls_model.fit(_fr_augmented_signatures, _fr_curves)
""",
        "ensemble model fit",
    )
    code = _replace_once(
        code,
        """    _fr_predicted_curve = _fr_np.asarray(
        _fr_model.predict(_fr_signature(_fr_runtime_features)[None, :])[0],
        dtype=float,
    )
""",
        """    _fr_ridge_curve = _fr_np.asarray(
        _fr_ridge_model.predict(_fr_signature(_fr_runtime_features)[None, :])[0],
        dtype=float,
    )
    _fr_pls_curve = _fr_np.asarray(
        _fr_pls_model.predict(
            _fr_augmented_signature(_fr_runtime_features)[None, :]
        )[0],
        dtype=float,
    )
    _fr_predicted_curve = (
        _FR_RIDGE_WEIGHT * _fr_ridge_curve + _FR_PLS_WEIGHT * _fr_pls_curve
    )
""",
        "ensemble prediction",
    )
    code = _replace_once(
        code,
        "'candidate': 'well-functional residual Ridge overlay'",
        "'candidate': 'fixed 75/25 well-functional Ridge/PLS residual overlay'",
        "candidate label",
    )
    code = _replace_once(
        code,
        """        'ridge_alpha': float(_FR_RIDGE_ALPHA),
        'overlay_scale': float(_FR_OVERLAY_SCALE),
""",
        """        'ridge_alpha': float(_FR_RIDGE_ALPHA),
        'pls_components': int(_FR_PLS_COMPONENTS),
        'ridge_weight': float(_FR_RIDGE_WEIGHT),
        'pls_weight': float(_FR_PLS_WEIGHT),
        'weight_grid_searched': False,
        'overlay_scale': float(_FR_OVERLAY_SCALE),
""",
        "report ensemble parameters",
    )
    code = _replace_once(
        code,
        "'signature_width': int(_fr_signatures.shape[1])",
        (
            "'base_signature_width': int(_fr_signatures.shape[1]),\n"
            "        'augmented_signature_width': int(_fr_augmented_signatures.shape[1])"
        ),
        "report signature widths",
    )
    replacements = {
        "'pooled_delta': -0.0276029": "'pooled_delta': -0.0352086425",
        "'p95_well_delta': -0.07058": "'p95_well_delta': -0.06408264",
        "'worst_well_delta': 0.19252": "'worst_well_delta': 0.20036961",
        "'pooled_delta': -0.0187671": "'pooled_delta': -0.0230585034",
        "'p95_well_delta': -0.05606": "'p95_well_delta': -0.05562910",
        "'worst_well_delta': 0.01616": "'worst_well_delta': -0.01281703",
    }
    for old, new in replacements.items():
        code = _replace_once(code, old, new, f"OOF report {old}")
    for old, new in {
        "submission_before_functional_ridge.csv": (
            "submission_before_functional_ridge_pls_ensemble.csv"
        ),
        "functional_ridge_per_well.csv": "functional_ridge_pls_ensemble_per_well.csv",
        "functional_ridge_deploy_report.json": (
            "functional_ridge_pls_ensemble_deploy_report.json"
        ),
    }.items():
        code = code.replace(old, new)
    return code.replace("Functional Ridge", "Functional Ridge PLS ensemble")


def _transform_contract(code: str) -> str:
    for old, new in {
        "submission_before_functional_ridge.csv": (
            "submission_before_functional_ridge_pls_ensemble.csv"
        ),
        "functional_ridge_per_well.csv": "functional_ridge_pls_ensemble_per_well.csv",
        "functional_ridge_deploy_report.json": (
            "functional_ridge_pls_ensemble_deploy_report.json"
        ),
    }.items():
        code = code.replace(old, new)
    code = _replace_once(
        code,
        """if _fr_final_report['target_free_features_only'] is not True:
    raise RuntimeError('Functional Ridge target-free contract is not asserted')
""",
        """if _fr_final_report['target_free_features_only'] is not True:
    raise RuntimeError('Functional Ridge target-free contract is not asserted')
_fr_final_parameters = _fr_final_report.get('parameters', {})
if _fr_final_parameters.get('ridge_weight') != 0.75:
    raise RuntimeError('Functional ensemble Ridge weight contract mismatch')
if _fr_final_parameters.get('pls_weight') != 0.25:
    raise RuntimeError('Functional ensemble PLS weight contract mismatch')
if _fr_final_parameters.get('weight_grid_searched') is not False:
    raise RuntimeError('Functional ensemble unexpectedly searched a weight grid')
""",
        "final ensemble contract",
    )
    return code.replace("Functional Ridge", "Functional Ridge PLS ensemble")


def _normalize_source_bootstrap(code: str) -> str:
    old = ", encoding='utf-8')"
    if code.count(old) != 3:
        raise RuntimeError(
            f"expected three bootstrap text writes, found {code.count(old)}"
        )
    return code.replace(old, ", encoding='utf-8', newline='\\n')")


def main() -> None:
    notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    base_metadata = json.loads(BASE_METADATA.read_text(encoding="utf-8"))
    if base_metadata.get("id") != "heralces/rogii-functional-ridge-v1":
        raise RuntimeError("unexpected Functional Ridge base Notebook")

    notebook = copy.deepcopy(notebook)
    notebook["cells"][0]["source"] = (
        "# ROGII Functional Ridge/PLS ensemble v1\n\n"
        "A private controlled verification of a preregistered fixed 75/25 "
        "curve ensemble. Both component models passed the pooled and tail "
        "checks; the stable Ridge anchor makes all five Typewell folds improve."
    )
    overlay_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_overlay_code")
    ]
    contract_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_final_contract")
    ]
    if len(overlay_cells) != 1 or len(contract_cells) != 1:
        raise RuntimeError("Functional Ridge base markers are incomplete")
    bootstrap_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("functional_ridge_source_bootstrap")
    ]
    if len(bootstrap_cells) != 1:
        raise RuntimeError("Functional Ridge source bootstrap marker is incomplete")
    bootstrap_cells[0]["source"] = _normalize_source_bootstrap(
        _source(bootstrap_cells[0])
    )
    overlay_cells[0]["source"] = _transform_overlay(_source(overlay_cells[0]))
    overlay_cells[0]["metadata"] = {"functional_ridge_pls_overlay_code": True}
    contract_cells[0]["source"] = _transform_contract(_source(contract_cells[0]))
    contract_cells[0]["metadata"] = {"functional_ridge_pls_final_contract": True}

    notebook["metadata"]["functional_ridge_pls_build"] = {
        "source_kernel": base_metadata["id"],
        "ridge_weight": 0.75,
        "pls_weight": 0.25,
        "weight_grid_searched": False,
        "pls_components": 8,
        "overlay_scale": 0.20,
        "target_free_overlay": True,
        "leaderboard_score_used_for_overlay_selection": False,
    }

    metadata = copy.deepcopy(base_metadata)
    metadata.update(
        {
            "id": "heralces/rogii-functional-ridge-pls-v1",
            "title": "ROGII Functional Ridge PLS v1",
            "code_file": OUTPUT.name,
            "is_private": True,
            "enable_internet": False,
            "keywords": ["geology", "ridge", "pls", "target-free"],
        }
    )
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
