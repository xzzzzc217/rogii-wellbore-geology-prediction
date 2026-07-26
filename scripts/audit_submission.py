"""Audit a ROGII submission artifact before upload."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.submission_guard import (  # noqa: E402
    DEFAULT_FALLBACK_FRACTION,
    DEFAULT_MAX_ADJACENT_JUMP,
    DEFAULT_TVT_MAX,
    DEFAULT_TVT_MIN,
    audit_submission,
    write_submission_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate schema, ID alignment, finite predictions, per-well coverage, "
            "ranges, jumps, and fallback signals in a ROGII submission."
        )
    )
    parser.add_argument("submission", type=Path, help="Candidate submission.csv path.")
    parser.add_argument(
        "--sample",
        type=Path,
        default=REPO_ROOT / "data" / "competition" / "sample_submission.csv",
        help="Reference sample_submission.csv path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "submission_audit.json",
        help="JSON report path.",
    )
    parser.add_argument("--tvt-min", type=float, default=DEFAULT_TVT_MIN)
    parser.add_argument("--tvt-max", type=float, default=DEFAULT_TVT_MAX)
    parser.add_argument(
        "--max-adjacent-jump",
        type=float,
        default=DEFAULT_MAX_ADJACENT_JUMP,
    )
    parser.add_argument(
        "--fallback-fraction",
        type=float,
        default=DEFAULT_FALLBACK_FRACTION,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 for structural errors; warnings remain non-fatal.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_submission(
        args.submission,
        args.sample,
        tvt_min=args.tvt_min,
        tvt_max=args.tvt_max,
        max_adjacent_jump=args.max_adjacent_jump,
        fallback_fraction=args.fallback_fraction,
    )
    output = write_submission_report(report, args.output)
    summary = report["issue_summary"]
    errors = int(summary["by_severity"].get("error", 0))
    warnings = int(summary["by_severity"].get("warning", 0))
    print(f"Submission audit: {output}")
    print(f"Rows: {report.get('structure', {}).get('submission_row_count', 0):,}")
    print(f"Issues: errors={errors:,}, warnings={warnings:,}")
    print(f"Structural validity: {report['structural_valid']}")
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
