"""Generate a JSON audit of the downloaded ROGII competition data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.data_audit import audit_dataset, write_json_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover train/test well pairs, audit schemas and evaluation zones, "
            "validate sample-submission IDs, and write a JSON report."
        )
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Extracted competition data root (default: data/raw).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "data_audit.json",
        help="JSON report path (default: artifacts/data_audit.json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when the completed report contains error-level issues.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_dataset(args.data_root)
    output = write_json_report(report, args.output)
    summary = report["issue_summary"]
    errors = int(summary["by_severity"].get("error", 0))
    warnings = int(summary["by_severity"].get("warning", 0))
    train_wells = int(report["splits"]["train"]["well_count"])
    test_wells = int(report["splits"]["test"]["well_count"])
    print(f"Audit report: {output}")
    print(f"Wells: train={train_wells:,}, test={test_wells:,}")
    print(f"Issues: errors={errors:,}, warnings={warnings:,}")
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
