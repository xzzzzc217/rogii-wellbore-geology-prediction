"""Validate the hidden output of the dynamic V4 T3920 PF continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data" / "competition" / "sample_submission.csv"
EXPECTED_VISIBLE_SHA256 = "29e7c11ae721383007177d01c262592048fe7bd5f914896a9c85e43ff07d7610"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    run_dir: Path,
    sample_path: Path,
    *,
    require_visible_sha: bool = True,
) -> dict[str, object]:
    submission_path = run_dir / "submission.csv"
    before_path = run_dir / "submission_before_v4_t3920.csv"
    report_path = run_dir / "v4_t3920_final_summary.json"
    audit_path = run_dir / "submission_audit.json"
    for path in (submission_path, before_path, report_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    submission = pd.read_csv(submission_path, dtype={"id": "string"})
    before = pd.read_csv(before_path, dtype={"id": "string"})
    if list(submission.columns) != ["id", "tvt"]:
        raise ValueError(f"unexpected submission columns: {list(submission.columns)}")
    if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
        raise ValueError("submission rows or ID order differ from official sample")
    if not submission["id"].is_unique:
        raise ValueError("submission IDs are not unique")
    if not before["id"].equals(submission["id"]):
        raise ValueError("before/final ID order mismatch")

    final_values = pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(dtype=float)
    before_values = pd.to_numeric(before["tvt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(final_values).all() or not np.isfinite(before_values).all():
        raise ValueError("non-finite predictions")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_flags = {
        "leaderboard_bias_applied": True,
        "fixed_test_well_used": False,
        "visible_shape_constants_used": False,
        "expected_visible_hash_used": False,
    }
    for key, expected in required_flags.items():
        if report.get(key) is not expected:
            raise ValueError(f"report contract mismatch: {key}")

    file_sha = _sha256(submission_path)
    if report.get("final_file_sha") != file_sha:
        raise ValueError("final report SHA mismatch")
    if audit.get("sha256_submission_csv") != file_sha:
        raise ValueError("submission audit SHA mismatch")
    if require_visible_sha and file_sha != EXPECTED_VISIBLE_SHA256:
        raise ValueError(f"visible reproduction SHA mismatch: {file_sha}")

    delta = final_values - before_values
    wells = submission["id"].astype(str).str.rsplit("_", n=1).str[0]
    extensions = report.get("runtime_pf_extensions")
    if not isinstance(extensions, list):
        raise ValueError("runtime_pf_extensions is not a list")
    target = np.zeros(len(submission), dtype=bool)
    for item in extensions:
        well = str(item["well"])
        mask = wells.eq(well).to_numpy()
        rows = int(item["rows"])
        extra = float(item["additional_shift_ft"])
        if int(mask.sum()) != rows:
            raise ValueError(f"reported row count mismatch for {well}")
        if not np.allclose(delta[mask], extra, atol=1e-10, rtol=0.0):
            raise ValueError(f"delta mismatch for {well}")
        target |= mask
    if np.any(np.abs(delta[~target]) > 0.0):
        raise ValueError("non-target rows changed")
    if int(report.get("target_rows", -1)) != int(target.sum()):
        raise ValueError("target_rows mismatch")
    if int(report.get("changed_rows", -1)) != int(np.count_nonzero(delta)):
        raise ValueError("changed_rows mismatch")

    return {
        "status": "PASS",
        "rows": int(len(submission)),
        "applied_wells": int(len(extensions)),
        "changed_rows": int(np.count_nonzero(delta)),
        "sha256": file_sha,
        "visible_sha_required": require_visible_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--allow-hidden-sha", action="store_true")
    args = parser.parse_args()
    result = validate(
        args.run_dir.resolve(),
        args.sample.resolve(),
        require_visible_sha=not args.allow_hidden_sha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
