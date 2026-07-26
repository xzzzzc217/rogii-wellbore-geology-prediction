"""Fail-safe auditing for ROGII ``submission.csv`` artifacts.

The guard separates structural validity from statistical plausibility:

* malformed schemas, IDs, coverage, ordering, or non-finite predictions are
  error-level issues;
* suspicious ranges, jumps, and fallback/constant trajectories are warnings.

This distinction lets a CLI ``--strict`` check block submissions that Kaggle
cannot score correctly without rejecting a structurally valid experiment only
because its predictions look unusual.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.data_audit import ParsedSubmissionId, parse_submission_id

REPORT_VERSION = 1
EXPECTED_COLUMNS = ("id", "tvt")
DEFAULT_TVT_MIN = 0.0
DEFAULT_TVT_MAX = 25_000.0
DEFAULT_MAX_ADJACENT_JUMP = 25.0
DEFAULT_CONSTANT_TOLERANCE = 1e-9
DEFAULT_FALLBACK_TOLERANCE = 1e-9
DEFAULT_FALLBACK_FRACTION = 0.95
EXAMPLE_LIMIT = 10


def _issue(
    severity: str,
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    result = {"severity": severity, "code": code, "message": message}
    result.update(details)
    return result


def _read_csv(path: Path, *, label: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    if not path.is_file():
        return None, _issue(
            "error",
            f"{label}_missing",
            f"{label.replace('_', ' ').title()} does not exist: {path}",
            path=str(path),
        )
    try:
        return pd.read_csv(path, dtype={"id": "string"}, low_memory=False), None
    except Exception as exc:  # pandas exposes several parser/encoding exception classes
        return None, _issue(
            "error",
            f"{label}_read_failed",
            f"{type(exc).__name__}: {exc}",
            path=str(path),
        )


def _id_values(frame: pd.DataFrame) -> list[str | None]:
    if "id" not in frame:
        return []
    return [value if isinstance(value, str) else None for value in frame["id"].tolist()]


def _parse_ids(values: Sequence[str | None]) -> tuple[list[ParsedSubmissionId | None], list[int]]:
    parsed: list[ParsedSubmissionId | None] = []
    invalid_positions: list[int] = []
    for position, value in enumerate(values):
        try:
            parsed.append(parse_submission_id(value))
        except ValueError:
            parsed.append(None)
            invalid_positions.append(position)
    return parsed, invalid_positions


def _is_contiguous(indices: Sequence[int]) -> bool:
    unique = sorted(set(indices))
    return bool(unique) and all(
        right - left == 1 for left, right in zip(unique, unique[1:], strict=False)
    )


def _counter_examples(counter: Counter[str | None]) -> list[str]:
    examples: list[str] = []
    for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
        rendered = "<missing>" if value is None else value
        examples.extend([rendered] * min(count, EXAMPLE_LIMIT - len(examples)))
        if len(examples) >= EXAMPLE_LIMIT:
            break
    return examples


def _finite_statistics(values: np.ndarray) -> dict[str, int | float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "min": None,
            "max": None,
            "span": None,
            "mean": None,
            "std": None,
            "median": None,
            "q01": None,
            "q05": None,
            "q95": None,
            "q99": None,
        }
    quantiles = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "span": float(np.ptp(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(quantiles[2]),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def _range_statistics(
    values: np.ndarray,
    *,
    tvt_min: float,
    tvt_max: float,
) -> dict[str, int | float | None]:
    result = _finite_statistics(values)
    finite = values[np.isfinite(values)]
    below = finite < tvt_min
    above = finite > tvt_max
    result.update(
        {
            "plausible_min": float(tvt_min),
            "plausible_max": float(tvt_max),
            "below_plausible_min_count": int(np.count_nonzero(below)),
            "above_plausible_max_count": int(np.count_nonzero(above)),
            "outside_plausible_range_count": int(np.count_nonzero(below | above)),
        }
    )
    return result


def _jump_statistics(
    rows: Sequence[tuple[int, str, float]],
    *,
    threshold: float,
) -> tuple[dict[str, int | float | None], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: item[0])
    jumps: list[float] = []
    examples: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_index, left_id, left_value = left
        right_index, right_id, right_value = right
        if right_index - left_index != 1:
            continue
        jump = abs(right_value - left_value)
        jumps.append(jump)
        if jump > threshold and len(examples) < EXAMPLE_LIMIT:
            examples.append(
                {
                    "from_id": left_id,
                    "to_id": right_id,
                    "absolute_jump": float(jump),
                }
            )

    jump_values = np.asarray(jumps, dtype=float)
    if jump_values.size == 0:
        summary: dict[str, int | float | None] = {
            "adjacent_pair_count": 0,
            "max_absolute_jump": None,
            "mean_absolute_jump": None,
            "q95_absolute_jump": None,
            "threshold": float(threshold),
            "above_threshold_count": 0,
        }
    else:
        summary = {
            "adjacent_pair_count": int(jump_values.size),
            "max_absolute_jump": float(np.max(jump_values)),
            "mean_absolute_jump": float(np.mean(jump_values)),
            "q95_absolute_jump": float(np.quantile(jump_values, 0.95)),
            "threshold": float(threshold),
            "above_threshold_count": int(np.count_nonzero(jump_values > threshold)),
        }
    return summary, examples


def _fallback_statistics(
    positions: Sequence[int],
    *,
    ids: Sequence[str | None],
    values: np.ndarray,
    reference_values: Mapping[str, float],
    tolerance: float,
) -> dict[str, int | float | bool]:
    comparable = 0
    matches = 0
    for position in positions:
        candidate_id = ids[position]
        candidate_value = values[position]
        if candidate_id not in reference_values or not math.isfinite(candidate_value):
            continue
        comparable += 1
        if math.isclose(
            candidate_value,
            reference_values[candidate_id],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            matches += 1
    fraction = matches / comparable if comparable else 0.0
    return {
        "comparable_count": comparable,
        "matching_count": matches,
        "matching_fraction": float(fraction),
        "all_comparable_rows_match": bool(comparable and matches == comparable),
    }


def _issue_summary(issues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_severity = Counter(str(issue["severity"]) for issue in issues)
    by_code = Counter(str(issue["code"]) for issue in issues)
    return {
        "total": len(issues),
        "by_severity": dict(sorted(by_severity.items())),
        "by_code": dict(sorted(by_code.items())),
    }


def audit_submission(
    submission_path: str | Path,
    sample_submission_path: str | Path,
    *,
    tvt_min: float = DEFAULT_TVT_MIN,
    tvt_max: float = DEFAULT_TVT_MAX,
    max_adjacent_jump: float = DEFAULT_MAX_ADJACENT_JUMP,
    constant_tolerance: float = DEFAULT_CONSTANT_TOLERANCE,
    fallback_tolerance: float = DEFAULT_FALLBACK_TOLERANCE,
    fallback_fraction: float = DEFAULT_FALLBACK_FRACTION,
) -> dict[str, Any]:
    """Audit a prediction artifact against the official sample submission.

    Statistical thresholds are deliberately broad guardrails. Exceeding them
    produces warnings, while errors are reserved for violations that can make
    the artifact invalid or misaligned at scoring time.
    """

    if tvt_min >= tvt_max:
        raise ValueError("tvt_min must be less than tvt_max")
    if max_adjacent_jump <= 0:
        raise ValueError("max_adjacent_jump must be positive")
    if constant_tolerance < 0 or fallback_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    if not 0.0 <= fallback_fraction <= 1.0:
        raise ValueError("fallback_fraction must be between 0 and 1")

    submission = Path(submission_path).expanduser().resolve()
    sample = Path(sample_submission_path).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "submission_path": str(submission),
        "sample_submission_path": str(sample),
        "config": {
            "expected_columns": list(EXPECTED_COLUMNS),
            "tvt_min": float(tvt_min),
            "tvt_max": float(tvt_max),
            "max_adjacent_jump": float(max_adjacent_jump),
            "constant_tolerance": float(constant_tolerance),
            "fallback_tolerance": float(fallback_tolerance),
            "fallback_fraction": float(fallback_fraction),
        },
    }

    candidate_frame, candidate_error = _read_csv(submission, label="submission")
    sample_frame, sample_error = _read_csv(sample, label="sample_submission")
    if candidate_error:
        issues.append(candidate_error)
    if sample_error:
        issues.append(sample_error)
    if candidate_frame is None or sample_frame is None:
        report["issues"] = issues
        report["issue_summary"] = _issue_summary(issues)
        report["structural_valid"] = False
        return report

    candidate_columns = [str(column) for column in candidate_frame.columns]
    sample_columns = [str(column) for column in sample_frame.columns]
    candidate_columns_exact = candidate_columns == list(EXPECTED_COLUMNS)
    sample_columns_exact = sample_columns == list(EXPECTED_COLUMNS)
    if not candidate_columns_exact:
        issues.append(
            _issue(
                "error",
                "submission_columns_mismatch",
                "Submission columns must be exactly ['id', 'tvt'] in that order.",
                actual_columns=candidate_columns,
            )
        )
    if not sample_columns_exact:
        issues.append(
            _issue(
                "error",
                "sample_submission_columns_mismatch",
                "Sample-submission columns must be exactly ['id', 'tvt'] in that order.",
                actual_columns=sample_columns,
            )
        )

    structure: dict[str, Any] = {
        "submission_columns": candidate_columns,
        "sample_submission_columns": sample_columns,
        "columns_exact": candidate_columns_exact,
        "sample_columns_exact": sample_columns_exact,
        "submission_row_count": int(len(candidate_frame)),
        "sample_submission_row_count": int(len(sample_frame)),
        "row_count_exact": len(candidate_frame) == len(sample_frame),
    }
    report["structure"] = structure
    if not structure["row_count_exact"]:
        issues.append(
            _issue(
                "error",
                "submission_row_count_mismatch",
                "Submission row count differs from sample_submission.csv.",
                expected=int(len(sample_frame)),
                actual=int(len(candidate_frame)),
            )
        )

    if "id" not in candidate_frame or "id" not in sample_frame:
        report["issues"] = issues
        report["issue_summary"] = _issue_summary(issues)
        report["structural_valid"] = False
        return report

    candidate_ids = _id_values(candidate_frame)
    expected_ids = _id_values(sample_frame)
    parsed_candidate, malformed_candidate_positions = _parse_ids(candidate_ids)
    parsed_expected, malformed_expected_positions = _parse_ids(expected_ids)
    candidate_counter = Counter(candidate_ids)
    expected_counter = Counter(expected_ids)
    missing_counter = expected_counter - candidate_counter
    extra_counter = candidate_counter - expected_counter
    duplicate_candidate_ids = {
        candidate_id: count
        for candidate_id, count in candidate_counter.items()
        if candidate_id is not None and count > 1
    }
    duplicate_expected_ids = {
        candidate_id: count
        for candidate_id, count in expected_counter.items()
        if candidate_id is not None and count > 1
    }

    structure.update(
        {
            "submission_ids_unique": not duplicate_candidate_ids,
            "sample_ids_unique": not duplicate_expected_ids,
            "duplicate_submission_id_count": int(
                sum(count - 1 for count in duplicate_candidate_ids.values())
            ),
            "duplicate_sample_id_count": int(
                sum(count - 1 for count in duplicate_expected_ids.values())
            ),
            "malformed_submission_id_count": len(malformed_candidate_positions),
            "malformed_sample_id_count": len(malformed_expected_positions),
            "id_order_exact": candidate_ids == expected_ids,
            "id_multiset_exact": candidate_counter == expected_counter,
            "missing_expected_id_count": int(sum(missing_counter.values())),
            "extra_submission_id_count": int(sum(extra_counter.values())),
            "missing_expected_id_examples": _counter_examples(missing_counter),
            "extra_submission_id_examples": _counter_examples(extra_counter),
            "all_submission_ids_belong_to_sample": not extra_counter,
        }
    )

    if duplicate_candidate_ids:
        issues.append(
            _issue(
                "error",
                "submission_ids_not_unique",
                "Submission IDs must be unique.",
                duplicate_count=structure["duplicate_submission_id_count"],
                examples=list(sorted(duplicate_candidate_ids))[:EXAMPLE_LIMIT],
            )
        )
    if duplicate_expected_ids:
        issues.append(
            _issue(
                "error",
                "sample_ids_not_unique",
                "Sample-submission IDs are not unique; the reference is invalid.",
                duplicate_count=structure["duplicate_sample_id_count"],
                examples=list(sorted(duplicate_expected_ids))[:EXAMPLE_LIMIT],
            )
        )
    if malformed_candidate_positions:
        issues.append(
            _issue(
                "error",
                "submission_ids_malformed",
                "Submission IDs must use the form {well_id}_{non-negative row_index}.",
                count=len(malformed_candidate_positions),
                row_positions=malformed_candidate_positions[:EXAMPLE_LIMIT],
            )
        )
    if malformed_expected_positions:
        issues.append(
            _issue(
                "error",
                "sample_ids_malformed",
                "Sample-submission IDs do not follow the required format.",
                count=len(malformed_expected_positions),
                row_positions=malformed_expected_positions[:EXAMPLE_LIMIT],
            )
        )
    if missing_counter or extra_counter:
        issues.append(
            _issue(
                "error",
                "submission_id_coverage_mismatch",
                "Submission IDs do not exactly cover sample_submission.csv.",
                missing_count=structure["missing_expected_id_count"],
                extra_count=structure["extra_submission_id_count"],
                missing_examples=structure["missing_expected_id_examples"],
                extra_examples=structure["extra_submission_id_examples"],
            )
        )
    if candidate_ids != expected_ids:
        issues.append(
            _issue(
                "error",
                "submission_id_order_mismatch",
                "Submission IDs must be in the same order as sample_submission.csv.",
            )
        )

    expected_by_well: dict[str, list[tuple[int, str]]] = defaultdict(list)
    candidate_by_well: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for value in parsed_expected:
        if value is not None:
            expected_by_well[value.well_id].append((value.row_index, value.raw))
    for position, value in enumerate(parsed_candidate):
        if value is not None:
            candidate_by_well[value.well_id].append((value.row_index, value.raw, position))

    non_contiguous_reference_wells = [
        well_id
        for well_id, rows in expected_by_well.items()
        if not _is_contiguous([row_index for row_index, _ in rows])
    ]
    if non_contiguous_reference_wells:
        issues.append(
            _issue(
                "error",
                "sample_row_indices_not_contiguous",
                "Each sample-submission well must contain a contiguous row-index interval.",
                well_count=len(non_contiguous_reference_wells),
                examples=sorted(non_contiguous_reference_wells)[:EXAMPLE_LIMIT],
            )
        )

    if "tvt" not in candidate_frame:
        report["coverage_by_well"] = {}
        report["issues"] = issues
        report["issue_summary"] = _issue_summary(issues)
        report["structural_valid"] = False
        return report

    numeric_tvt = pd.to_numeric(candidate_frame["tvt"], errors="coerce")
    values = numeric_tvt.to_numpy(dtype=float, na_value=np.nan)
    missing_or_non_numeric = np.isnan(values)
    infinite = np.isinf(values)
    non_finite = ~np.isfinite(values)
    structure.update(
        {
            "tvt_missing_or_non_numeric_count": int(np.count_nonzero(missing_or_non_numeric)),
            "tvt_infinite_count": int(np.count_nonzero(infinite)),
            "tvt_non_finite_count": int(np.count_nonzero(non_finite)),
            "all_tvt_values_finite_numeric": not bool(np.any(non_finite)),
        }
    )
    if np.any(non_finite):
        positions = np.flatnonzero(non_finite).tolist()
        issues.append(
            _issue(
                "error",
                "submission_tvt_non_finite",
                "Every tvt prediction must be a finite numeric value.",
                count=len(positions),
                row_positions=positions[:EXAMPLE_LIMIT],
                id_examples=[
                    candidate_ids[position] or "<missing>" for position in positions[:EXAMPLE_LIMIT]
                ],
            )
        )

    reference_values: dict[str, float] = {}
    if "tvt" in sample_frame:
        reference_numeric = pd.to_numeric(sample_frame["tvt"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        for candidate_id, value in zip(expected_ids, reference_numeric, strict=True):
            if candidate_id is not None and math.isfinite(value):
                reference_values[candidate_id] = float(value)

    all_well_ids = sorted(set(expected_by_well) | set(candidate_by_well))
    coverage_by_well: dict[str, Any] = {}
    per_well_statistics: dict[str, Any] = {}
    non_contiguous_candidate_wells: list[str] = []
    coverage_mismatch_wells: list[str] = []
    out_of_range_wells: list[str] = []
    jump_wells: list[str] = []
    constant_wells: list[str] = []
    fallback_wells: list[str] = []
    all_jump_examples: list[dict[str, Any]] = []

    for well_id in all_well_ids:
        expected_rows = expected_by_well.get(well_id, [])
        candidate_rows = candidate_by_well.get(well_id, [])
        expected_well_ids = [raw for _, raw in expected_rows]
        candidate_well_ids = [raw for _, raw, _ in candidate_rows]
        expected_well_counter = Counter(expected_well_ids)
        candidate_well_counter = Counter(candidate_well_ids)
        expected_indices = [row_index for row_index, _ in expected_rows]
        candidate_indices = [row_index for row_index, _, _ in candidate_rows]
        expected_id_set = set(expected_well_ids)
        positions = [position for _, _, position in candidate_rows]
        all_indices_in_sample = all(raw in expected_id_set for raw in candidate_well_ids)
        coverage_exact = candidate_well_counter == expected_well_counter
        indices_contiguous = _is_contiguous(candidate_indices)

        coverage_by_well[well_id] = {
            "expected_row_count": len(expected_rows),
            "submission_row_count": len(candidate_rows),
            "coverage_exact": coverage_exact,
            "expected_first_row_index": min(expected_indices) if expected_indices else None,
            "expected_last_row_index": max(expected_indices) if expected_indices else None,
            "submission_first_row_index": min(candidate_indices) if candidate_indices else None,
            "submission_last_row_index": max(candidate_indices) if candidate_indices else None,
            "expected_indices_contiguous": _is_contiguous(expected_indices),
            "submission_indices_contiguous": indices_contiguous,
            "all_submission_indices_belong_to_sample": all_indices_in_sample,
            "missing_id_count": int(sum((expected_well_counter - candidate_well_counter).values())),
            "extra_id_count": int(sum((candidate_well_counter - expected_well_counter).values())),
        }
        if not coverage_exact:
            coverage_mismatch_wells.append(well_id)
        if candidate_rows and not indices_contiguous:
            non_contiguous_candidate_wells.append(well_id)

        well_values = values[positions] if positions else np.asarray([], dtype=float)
        range_stats = _range_statistics(well_values, tvt_min=tvt_min, tvt_max=tvt_max)
        jump_rows = [
            (row_index, raw, float(values[position]))
            for row_index, raw, position in candidate_rows
            if math.isfinite(values[position])
        ]
        jump_stats, jump_examples = _jump_statistics(
            jump_rows,
            threshold=max_adjacent_jump,
        )
        fallback_stats = _fallback_statistics(
            positions,
            ids=candidate_ids,
            values=values,
            reference_values=reference_values,
            tolerance=fallback_tolerance,
        )
        finite_well_values = well_values[np.isfinite(well_values)]
        is_constant = bool(
            finite_well_values.size > 1 and float(np.ptp(finite_well_values)) <= constant_tolerance
        )
        zero_count = int(
            np.count_nonzero(np.isclose(finite_well_values, 0.0, rtol=0.0, atol=fallback_tolerance))
        )
        per_well_statistics[well_id] = {
            "range": range_stats,
            "adjacent_jumps": jump_stats,
            "fallback_vs_sample": fallback_stats,
            "is_constant_trajectory": is_constant,
            "zero_count": zero_count,
            "zero_fraction": (
                float(zero_count / finite_well_values.size) if finite_well_values.size else 0.0
            ),
        }
        if range_stats["outside_plausible_range_count"]:
            out_of_range_wells.append(well_id)
        if jump_stats["above_threshold_count"]:
            jump_wells.append(well_id)
            all_jump_examples.extend({"well_id": well_id, **example} for example in jump_examples)
        if is_constant:
            constant_wells.append(well_id)
        if (
            fallback_stats["comparable_count"]
            and fallback_stats["matching_fraction"] >= fallback_fraction
        ):
            fallback_wells.append(well_id)

    report["coverage_by_well"] = coverage_by_well
    structure.update(
        {
            "well_coverage_exact": not coverage_mismatch_wells,
            "coverage_mismatch_well_count": len(coverage_mismatch_wells),
            "all_submission_well_indices_contiguous": not non_contiguous_candidate_wells,
            "non_contiguous_submission_well_count": len(non_contiguous_candidate_wells),
        }
    )
    if coverage_mismatch_wells:
        issues.append(
            _issue(
                "error",
                "submission_well_coverage_mismatch",
                "One or more wells do not exactly cover their sample-submission rows.",
                well_count=len(coverage_mismatch_wells),
                examples=coverage_mismatch_wells[:EXAMPLE_LIMIT],
            )
        )
    if non_contiguous_candidate_wells:
        issues.append(
            _issue(
                "error",
                "submission_row_indices_not_contiguous",
                "Submission row indices must be contiguous within each well.",
                well_count=len(non_contiguous_candidate_wells),
                examples=non_contiguous_candidate_wells[:EXAMPLE_LIMIT],
            )
        )

    overall_range = _range_statistics(values, tvt_min=tvt_min, tvt_max=tvt_max)
    overall_fallback = _fallback_statistics(
        list(range(len(candidate_frame))),
        ids=candidate_ids,
        values=values,
        reference_values=reference_values,
        tolerance=fallback_tolerance,
    )
    finite_values = values[np.isfinite(values)]
    overall_constant = bool(
        finite_values.size > 1 and float(np.ptp(finite_values)) <= constant_tolerance
    )
    zero_count = int(
        np.count_nonzero(np.isclose(finite_values, 0.0, rtol=0.0, atol=fallback_tolerance))
    )
    report["statistics"] = {
        "overall": {
            "range": overall_range,
            "fallback_vs_sample": overall_fallback,
            "is_constant_submission": overall_constant,
            "zero_count": zero_count,
            "zero_fraction": float(zero_count / finite_values.size) if finite_values.size else 0.0,
        },
        "per_well": per_well_statistics,
    }

    if overall_range["outside_plausible_range_count"]:
        issues.append(
            _issue(
                "warning",
                "tvt_outside_plausible_range",
                "Some predictions fall outside the configured broad TVT range.",
                count=overall_range["outside_plausible_range_count"],
                bounds=[float(tvt_min), float(tvt_max)],
                well_count=len(out_of_range_wells),
                well_examples=out_of_range_wells[:EXAMPLE_LIMIT],
            )
        )
    if jump_wells:
        issues.append(
            _issue(
                "warning",
                "large_adjacent_tvt_jumps",
                "Some within-well adjacent predictions exceed the configured jump threshold.",
                threshold=float(max_adjacent_jump),
                well_count=len(jump_wells),
                well_examples=jump_wells[:EXAMPLE_LIMIT],
                jump_examples=all_jump_examples[:EXAMPLE_LIMIT],
            )
        )
    if overall_constant:
        issues.append(
            _issue(
                "warning",
                "constant_submission",
                "All finite predictions are effectively constant.",
                tolerance=float(constant_tolerance),
            )
        )
    if constant_wells:
        issues.append(
            _issue(
                "warning",
                "constant_well_trajectories",
                "One or more wells have effectively constant prediction trajectories.",
                tolerance=float(constant_tolerance),
                well_count=len(constant_wells),
                well_examples=constant_wells[:EXAMPLE_LIMIT],
            )
        )
    if (
        overall_fallback["comparable_count"]
        and overall_fallback["matching_fraction"] >= fallback_fraction
    ):
        issues.append(
            _issue(
                "warning",
                "submission_matches_sample_fallback",
                "Most predictions still match the sample-submission fallback values.",
                matching_fraction=overall_fallback["matching_fraction"],
                threshold=float(fallback_fraction),
            )
        )
    if fallback_wells:
        issues.append(
            _issue(
                "warning",
                "well_trajectories_match_sample_fallback",
                "One or more wells mostly match sample-submission fallback values.",
                threshold=float(fallback_fraction),
                well_count=len(fallback_wells),
                well_examples=fallback_wells[:EXAMPLE_LIMIT],
            )
        )

    report["issues"] = issues
    report["issue_summary"] = _issue_summary(issues)
    report["structural_valid"] = not any(issue["severity"] == "error" for issue in issues)
    return report


def write_submission_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write an audit report as strict, human-readable JSON."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
