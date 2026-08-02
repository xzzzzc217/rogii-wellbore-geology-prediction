from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_hmm_scale2_functional_ridge_pls_v1.py"
VALIDATOR = ROOT / "scripts" / "validate_hmm_scale2_functional_ridge_pls_output.py"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "hmm_scale2_functional_ridge_pls_v1"
NOTEBOOK = OUTPUT_DIR / "rogii-hmm-scale2-functional-ridge-pls-v1.ipynb"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _marker_index(notebook: dict[str, object], marker: str) -> int:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    matches = [
        index
        for index, cell in enumerate(cells)
        if isinstance(cell, dict) and cell.get("metadata", {}).get(marker)
    ]
    assert len(matches) == 1
    return matches[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_builder_prepares_ordered_additive_notebook_without_authorizing_it() -> None:
    subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    order = [
        _marker_index(notebook, "dip_rate_hmm_overlay_code"),
        _marker_index(notebook, "hmm_scale2_intermediate_snapshot"),
        _marker_index(notebook, "dip_rate_hmm_final_contract"),
        _marker_index(notebook, "functional_ridge_source_bootstrap"),
        _marker_index(notebook, "functional_ridge_pls_overlay_code"),
        _marker_index(notebook, "functional_ridge_pls_final_contract"),
        _marker_index(
            notebook, "hmm_scale2_functional_ridge_pls_combined_contract"
        ),
    ]
    assert order == sorted(order)
    source = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )
    assert "OVERLAY_SCALE = 0.40" in source
    assert "_FR_RIDGE_WEIGHT = 0.75" in source
    assert "_FR_PLS_WEIGHT = 0.25" in source
    assert "_FR_OVERLAY_SCALE = 0.20" in source
    assert "submission_after_dip_rate_hmm_scale2.csv" in source
    assert "hmm_scale2_functional_ridge_pls_combined_report.json" in source
    for forbidden in ("000d7d20", "00bbac68", "00e12e8b", "14151"):
        assert forbidden not in source
    build = notebook["metadata"]["hmm_scale2_functional_ridge_pls_build"]
    assert build["authorization_status"] == "PREPARED_NOT_AUTHORIZED"
    assert build["execution_authorized"] is False
    assert build["component_order"] == ["hmm_scale2", "functional_ridge_pls"]
    assert build["spatial_improved_folds_over_hmm_scale2"] == "6/7"
    assert build["typewell_improved_folds_over_hmm_scale2"] == "5/5"
    assert metadata["id"] == "heralces/rogii-hmm-scale2-functional-ridge-pls-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert "heralces/rogii-private-sequence-pilot-cache-v1" in metadata[
        "dataset_sources"
    ]
    for cell in notebook["cells"]:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_validator_accepts_a_composed_audited_fixture(tmp_path: Path) -> None:
    hmm_dir = ROOT / "artifacts" / "kaggle_runs" / "dip_rate_hmm_scale2_v1_v1"
    functional_dir = (
        ROOT / "artifacts" / "kaggle_runs" / "functional_ridge_pls_v1_v1"
    )
    run_dir = tmp_path / "combined"
    run_dir.mkdir()
    shutil.copyfile(
        hmm_dir / "submission_before_dip_rate_hmm.csv",
        run_dir / "submission_before_dip_rate_hmm.csv",
    )
    shutil.copyfile(
        hmm_dir / "submission.csv",
        run_dir / "submission_after_dip_rate_hmm_scale2.csv",
    )
    shutil.copyfile(
        hmm_dir / "submission.csv",
        run_dir / "submission_before_functional_ridge_pls_ensemble.csv",
    )
    shutil.copyfile(
        hmm_dir / "dip_rate_hmm_per_well.csv",
        run_dir / "dip_rate_hmm_per_well.csv",
    )
    shutil.copyfile(
        functional_dir / "functional_ridge_pls_ensemble_per_well.csv",
        run_dir / "functional_ridge_pls_ensemble_per_well.csv",
    )

    hmm = pd.read_csv(hmm_dir / "submission.csv", dtype={"id": "string"})
    functional_before = pd.read_csv(
        functional_dir / "submission_before_functional_ridge_pls_ensemble.csv",
        dtype={"id": "string"},
    )
    functional_final = pd.read_csv(
        functional_dir / "submission.csv", dtype={"id": "string"}
    )
    assert hmm["id"].equals(functional_before["id"])
    final = hmm.copy()
    final["tvt"] = hmm["tvt"] + functional_final["tvt"] - functional_before["tvt"]
    final.to_csv(run_dir / "submission.csv", index=False)

    hmm_report = json.loads(
        (hmm_dir / "dip_rate_hmm_deploy_report.json").read_text(encoding="utf-8")
    )
    _write_json(run_dir / "dip_rate_hmm_deploy_report.json", hmm_report)
    functional_report = json.loads(
        (
            functional_dir / "functional_ridge_pls_ensemble_deploy_report.json"
        ).read_text(encoding="utf-8")
    )
    functional_report["before_submission_sha256"] = _sha256(
        run_dir / "submission_before_functional_ridge_pls_ensemble.csv"
    )
    functional_report["submission_sha256"] = _sha256(run_dir / "submission.csv")
    _write_json(
        run_dir / "functional_ridge_pls_ensemble_deploy_report.json",
        functional_report,
    )
    audit = json.loads(
        (functional_dir / "submission_audit.json").read_text(encoding="utf-8")
    )
    audit["sha256_submission_csv"] = _sha256(run_dir / "submission.csv")
    _write_json(run_dir / "submission_audit.json", audit)
    combined = {
        "authorization_status": "PREPARED_NOT_AUTHORIZED",
        "runtime_reads_leaderboard_score": False,
        "runtime_uses_fixed_test_well": False,
        "runtime_uses_fixed_visible_rows": False,
        "runtime_uses_expected_visible_hash": False,
        "probe_flags_used": False,
        "hmm_intermediate_sha256": _sha256(
            run_dir / "submission_after_dip_rate_hmm_scale2.csv"
        ),
        "submission_sha256": _sha256(run_dir / "submission.csv"),
        "preregistered_additive_oof": {
            "spatial_improved_folds_over_hmm_scale2": 6,
            "typewell_improved_folds_over_hmm_scale2": 5,
            "tail_risk_disclosed": True,
        },
    }
    _write_json(
        run_dir / "hmm_scale2_functional_ridge_pls_combined_report.json", combined
    )

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(run_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "PASS"' in result.stdout
