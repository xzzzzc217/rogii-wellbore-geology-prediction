"""Data discovery and audit helpers for the ROGII competition files.

The official archive has ``train`` and ``test`` directories. Each horizontal
well is accompanied by a typewell:

``{well_id}__horizontal_well.csv``
``{well_id}__typewell.csv``

An older task-description slide used
``{well_id}__typewell__{typewell_id}.csv``; discovery intentionally supports
both forms. The implementation is read-only except for explicit JSON report
writing and processes one CSV at a time so the full archive is never held in
memory.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

REPORT_VERSION = 1
SPLITS = ("train", "test")
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_MARKER = "__typewell"
HASH_WELL_ID = re.compile(r"^[0-9a-f]{8}$", flags=re.IGNORECASE)

HORIZONTAL_REQUIRED = {
    "train": ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"),
    "test": ("MD", "X", "Y", "Z", "GR", "TVT_input"),
}
TYPEWELL_REQUIRED = {
    "train": ("TVT", "GR", "Geology"),
    # The public test typewells intentionally omit the train-only formation
    # label, even though the task-description slide presents the richer schema.
    "test": ("TVT", "GR"),
}
NUMERIC_EXPECTED = {
    "horizontal": ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"),
    "typewell": ("TVT", "GR"),
}


@dataclass(frozen=True)
class ParsedSubmissionId:
    """A submission ID split at its final underscore."""

    raw: str
    well_id: str
    row_index: int


@dataclass(frozen=True)
class WellFiles:
    """All discovered files for one well within one split."""

    split: str
    well_id: str
    horizontal_files: tuple[Path, ...]
    typewell_files: tuple[Path, ...]
    embedded_typewell_ids: tuple[str | None, ...]

    @property
    def horizontal_path(self) -> Path | None:
        """Return the deterministic primary horizontal file, if present."""

        return self.horizontal_files[0] if self.horizontal_files else None

    @property
    def typewell_path(self) -> Path | None:
        """Return the deterministic primary typewell file, if present."""

        return self.typewell_files[0] if self.typewell_files else None

    @property
    def is_complete_pair(self) -> bool:
        """Whether exactly one file of each required role was found."""

        return len(self.horizontal_files) == 1 and len(self.typewell_files) == 1


@dataclass(frozen=True)
class DiscoveryResult:
    """Result of recursively discovering competition well files."""

    root: Path
    wells: tuple[WellFiles, ...]
    ignored_csvs: tuple[Path, ...]
    issues: tuple[dict[str, Any], ...]

    def wells_for_split(self, split: str) -> tuple[WellFiles, ...]:
        """Return well records belonging to ``split``."""

        normalized = split.casefold()
        return tuple(well for well in self.wells if well.split == normalized)


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    split: str | None = None,
    well_id: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if split is not None:
        result["split"] = split
    if well_id is not None:
        result["well_id"] = well_id
    if path is not None:
        result["path"] = path
    return result


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _infer_split(path: Path, root: Path) -> str | None:
    """Infer the closest ``train``/``test`` directory containing ``path``."""

    try:
        relative = path.resolve().relative_to(root.resolve())
        parts = relative.parts[:-1]
    except ValueError:
        parts = path.parts[:-1]

    candidates = [part.casefold() for part in parts if part.casefold() in SPLITS]
    if candidates:
        return candidates[-1]
    if root.name.casefold() in SPLITS:
        return root.name.casefold()
    return None


def _parse_well_filename(path: Path) -> tuple[str, str, str | None] | None:
    """Return ``(well_id, role, embedded_typewell_id)`` for a known filename."""

    name = path.name
    folded = name.casefold()
    horizontal_suffix = HORIZONTAL_SUFFIX.casefold()
    if folded.endswith(horizontal_suffix):
        well_id = name[: -len(HORIZONTAL_SUFFIX)]
        return (well_id, "horizontal", None) if well_id else None

    csv_suffix = ".csv"
    # Use the first marker: the documented legacy name
    # ``well__typewell__Typewell20001.csv`` contains "typewell" twice when
    # compared case-insensitively.
    marker_index = folded.find(TYPEWELL_MARKER)
    if not folded.endswith(csv_suffix) or marker_index <= 0:
        return None

    suffix = name[marker_index + len(TYPEWELL_MARKER) : -len(csv_suffix)]
    if suffix and not suffix.startswith("__"):
        return None
    well_id = name[:marker_index]
    embedded = suffix[2:] if suffix.startswith("__") and len(suffix) > 2 else None
    return (well_id, "typewell", embedded)


def discover_well_files(data_root: str | Path) -> DiscoveryResult:
    """Recursively discover train/test horizontal and typewell CSVs.

    Matching is case-insensitive for directory and suffix names. Well IDs are
    paired case-insensitively, while the spelling from the horizontal filename
    is retained for reporting and submission checks.
    """

    root = Path(data_root).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    if not root.exists():
        issues.append(
            _issue(
                "error", "data_root_missing", f"Data root does not exist: {root}", path=str(root)
            )
        )
        return DiscoveryResult(root, (), (), tuple(issues))
    if not root.is_dir():
        issues.append(
            _issue("error", "data_root_not_directory", f"Data root is not a directory: {root}")
        )
        return DiscoveryResult(root, (), (), tuple(issues))

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    ignored: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.casefold() != ".csv":
            continue
        if path.parent.resolve() == root and path.name.casefold() == "sample_submission.csv":
            continue
        split = _infer_split(path, root)
        parsed = _parse_well_filename(path)
        if split is None or parsed is None:
            ignored.append(path)
            continue

        well_id, role, embedded = parsed
        key = (split, well_id.casefold())
        record = grouped.setdefault(
            key,
            {
                "well_ids": [],
                "horizontal": [],
                "typewell": [],
                "embedded_typewell_ids": [],
            },
        )
        record["well_ids"].append(well_id)
        record[role].append(path.resolve())
        if role == "typewell":
            record["embedded_typewell_ids"].append(embedded)

    wells: list[WellFiles] = []
    for (split, _), record in sorted(grouped.items()):
        horizontal = tuple(sorted(record["horizontal"], key=lambda item: str(item).casefold()))
        typewell = tuple(sorted(record["typewell"], key=lambda item: str(item).casefold()))
        horizontal_ids = [
            parsed[0] for path in horizontal if (parsed := _parse_well_filename(path)) is not None
        ]
        well_id = horizontal_ids[0] if horizontal_ids else record["well_ids"][0]
        entry = WellFiles(
            split=split,
            well_id=well_id,
            horizontal_files=horizontal,
            typewell_files=typewell,
            embedded_typewell_ids=tuple(record["embedded_typewell_ids"]),
        )
        wells.append(entry)

        context = {"split": split, "well_id": well_id}
        if not horizontal:
            issues.append(
                _issue(
                    "error",
                    "horizontal_missing",
                    "Typewell has no matching horizontal-well file.",
                    **context,
                )
            )
        elif len(horizontal) > 1:
            issues.append(
                _issue(
                    "error",
                    "duplicate_horizontal_files",
                    f"Found {len(horizontal)} horizontal files for one well.",
                    **context,
                )
            )
        if not typewell:
            issues.append(
                _issue(
                    "error",
                    "typewell_missing",
                    "Horizontal well has no matching typewell file.",
                    **context,
                )
            )
        elif len(typewell) > 1:
            issues.append(
                _issue(
                    "error",
                    "duplicate_typewell_files",
                    f"Found {len(typewell)} typewell files for one well.",
                    **context,
                )
            )
        if not HASH_WELL_ID.fullmatch(well_id):
            issues.append(
                _issue(
                    "warning",
                    "nonstandard_well_id",
                    "Well ID is not the expected eight-character hexadecimal hash.",
                    **context,
                )
            )

    for split in SPLITS:
        if not any(well.split == split and well.horizontal_files for well in wells):
            issues.append(
                _issue(
                    "error",
                    "split_has_no_horizontal_wells",
                    f"No horizontal-well CSVs were found for split '{split}'.",
                    split=split,
                )
            )

    return DiscoveryResult(root, tuple(wells), tuple(ignored), tuple(issues))


def _column_name(frame: pd.DataFrame, expected: str) -> str | None:
    """Resolve an expected column name, tolerating case-only differences."""

    if expected in frame.columns:
        return expected
    matches = [
        str(column) for column in frame.columns if str(column).casefold() == expected.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


def _json_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer() and isinstance(value, (int, np.integer)):
        return int(number)
    return number


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize schema, rows, missingness, and finite numeric ranges."""

    row_count = int(len(frame))
    missingness: dict[str, dict[str, int | float]] = {}
    numeric_ranges: dict[str, dict[str, int | float | None]] = {}

    for raw_column in frame.columns:
        column = str(raw_column)
        series = frame[raw_column]
        missing = int(series.isna().sum())
        missingness[column] = {
            "missing_count": missing,
            "missing_fraction": float(missing / row_count) if row_count else 0.0,
            "non_missing_count": row_count - missing,
        }

        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
            continue
        values = series.to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        infinite = np.isinf(values)
        finite_values = values[finite]
        numeric_ranges[column] = {
            "finite_count": int(finite.sum()),
            "infinite_count": int(infinite.sum()),
            "min": _json_number(finite_values.min()) if finite_values.size else None,
            "max": _json_number(finite_values.max()) if finite_values.size else None,
            "mean": _json_number(finite_values.mean()) if finite_values.size else None,
        }

    return {
        "row_count": row_count,
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missingness": missingness,
        "numeric_ranges": numeric_ranges,
    }


def _boolean_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(positions) > 1) + 1
    chunks = np.split(positions, split_points)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]


def detect_evaluation_zone(frame: pd.DataFrame) -> dict[str, Any]:
    """Detect the masked prediction interval from ``TVT_input`` or ``TVT``.

    ``TVT_input`` missingness is authoritative. A missing ``TVT`` fallback is
    useful for malformed or legacy data, but cannot identify an evaluation zone
    when both columns are absent.
    """

    row_count = int(len(frame))
    tvt_input = _column_name(frame, "TVT_input")
    tvt = _column_name(frame, "TVT")
    md = _column_name(frame, "MD")

    if tvt_input is not None:
        mask = frame[tvt_input].isna().to_numpy(dtype=bool)
        source = "TVT_input_missing"
    elif tvt is not None:
        mask = frame[tvt].isna().to_numpy(dtype=bool)
        source = "TVT_missing_fallback"
    else:
        mask = np.zeros(row_count, dtype=bool)
        source = "unavailable"

    positions = np.flatnonzero(mask)
    blocks = _boolean_blocks(mask)
    first = int(positions[0]) if positions.size else None
    last = int(positions[-1]) if positions.size else None
    is_suffix = bool(
        positions.size
        and first is not None
        and positions.size == row_count - first
        and np.all(mask[first:])
    )

    start_md: int | float | None = None
    if first is not None and md is not None:
        md_value = pd.to_numeric(frame[md], errors="coerce").iloc[first]
        start_md = _json_number(md_value)

    tvt_available = 0
    tvt_missing = int(positions.size)
    target_range: dict[str, int | float | None] = {
        "finite_count": 0,
        "min": None,
        "max": None,
    }
    if tvt is not None and positions.size:
        target = pd.to_numeric(frame.loc[mask, tvt], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        finite = target[np.isfinite(target)]
        tvt_available = int(finite.size)
        tvt_missing = int(positions.size - finite.size)
        target_range = {
            "finite_count": tvt_available,
            "min": _json_number(finite.min()) if finite.size else None,
            "max": _json_number(finite.max()) if finite.size else None,
        }

    overlap_rows = 0
    mismatch_rows = 0
    max_abs_difference: float | None = None
    if tvt_input is not None and tvt is not None:
        left = pd.to_numeric(frame[tvt_input], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        right = pd.to_numeric(frame[tvt], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        overlap = np.isfinite(left) & np.isfinite(right)
        overlap_rows = int(overlap.sum())
        if overlap_rows:
            differences = np.abs(left[overlap] - right[overlap])
            mismatch_rows = int((differences > 1e-9).sum())
            max_abs_difference = _json_number(differences.max())

    return {
        "source": source,
        "row_count": row_count,
        "evaluation_row_count": int(positions.size),
        "evaluation_fraction": float(positions.size / row_count) if row_count else 0.0,
        "known_row_count": int(row_count - positions.size),
        "first_evaluation_position": first,
        "last_evaluation_position": last,
        "prediction_start_md": start_md,
        "blocks": [{"start": start, "end": end} for start, end in blocks],
        "block_count": len(blocks),
        "is_contiguous": len(blocks) <= 1,
        "is_suffix": is_suffix,
        "tvt_available_in_evaluation_rows": tvt_available,
        "tvt_missing_in_evaluation_rows": tvt_missing,
        "evaluation_target_range": target_range,
        "tvt_and_input_overlap_rows": overlap_rows,
        "tvt_input_mismatch_rows": mismatch_rows,
        "tvt_input_max_abs_difference": max_abs_difference,
    }


class _RoleAggregate:
    """Streaming aggregate of per-file frame summaries."""

    def __init__(self) -> None:
        self.file_count = 0
        self.total_rows = 0
        self.column_presence: Counter[str] = Counter()
        self.present_rows: Counter[str] = Counter()
        self.missing_rows: Counter[str] = Counter()
        self.dtype_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.ranges: dict[str, dict[str, int | float | None]] = {}
        self.schemas: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}

    def add(self, summary: Mapping[str, Any], path: str) -> None:
        self.file_count += 1
        rows = int(summary["row_count"])
        self.total_rows += rows
        dtypes: Mapping[str, str] = summary["dtypes"]
        schema_key = tuple((column, dtypes[column]) for column in summary["columns"])
        schema = self.schemas.setdefault(
            schema_key,
            {
                "columns": list(summary["columns"]),
                "dtypes": dict(dtypes),
                "file_count": 0,
                "examples": [],
            },
        )
        schema["file_count"] += 1
        if len(schema["examples"]) < 3:
            schema["examples"].append(path)

        for column, missing in summary["missingness"].items():
            self.column_presence[column] += 1
            self.present_rows[column] += rows
            self.missing_rows[column] += int(missing["missing_count"])
            self.dtype_counts[column][dtypes[column]] += 1

        for column, range_summary in summary["numeric_ranges"].items():
            target = self.ranges.setdefault(
                column,
                {
                    "finite_count": 0,
                    "infinite_count": 0,
                    "min": None,
                    "max": None,
                },
            )
            target["finite_count"] = int(target["finite_count"]) + int(
                range_summary["finite_count"]
            )
            target["infinite_count"] = int(target["infinite_count"]) + int(
                range_summary["infinite_count"]
            )
            if range_summary["min"] is not None:
                target["min"] = (
                    range_summary["min"]
                    if target["min"] is None
                    else min(float(target["min"]), float(range_summary["min"]))
                )
            if range_summary["max"] is not None:
                target["max"] = (
                    range_summary["max"]
                    if target["max"] is None
                    else max(float(target["max"]), float(range_summary["max"]))
                )

    def finish(self) -> dict[str, Any]:
        missingness: dict[str, Any] = {}
        for column in sorted(self.column_presence):
            present_rows = self.present_rows[column]
            missing_rows = self.missing_rows[column]
            missingness[column] = {
                "present_file_count": self.column_presence[column],
                "absent_file_count": self.file_count - self.column_presence[column],
                "present_row_count": present_rows,
                "missing_row_count": missing_rows,
                "missing_fraction_when_present": (
                    float(missing_rows / present_rows) if present_rows else 0.0
                ),
            }
        return {
            "file_count": self.file_count,
            "row_count": self.total_rows,
            "column_presence": {
                column: {
                    "present_file_count": self.column_presence[column],
                    "absent_file_count": self.file_count - self.column_presence[column],
                    "dtypes": dict(sorted(self.dtype_counts[column].items())),
                }
                for column in sorted(self.column_presence)
            },
            "missingness": missingness,
            "numeric_ranges": {column: self.ranges[column] for column in sorted(self.ranges)},
            "schema_variants": sorted(
                self.schemas.values(),
                key=lambda schema: (-int(schema["file_count"]), tuple(schema["columns"])),
            ),
        }


def _schema_issues(
    frame: pd.DataFrame,
    *,
    role: str,
    split: str,
    well_id: str,
    path: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required = HORIZONTAL_REQUIRED[split] if role == "horizontal" else TYPEWELL_REQUIRED[split]
    for column in required:
        if _column_name(frame, column) is None:
            issues.append(
                _issue(
                    "error",
                    "required_column_missing",
                    f"{role} file is missing required column '{column}'.",
                    split=split,
                    well_id=well_id,
                    path=path,
                )
            )
    for expected in NUMERIC_EXPECTED[role]:
        actual = _column_name(frame, expected)
        if actual is not None and not is_numeric_dtype(frame[actual].dtype):
            issues.append(
                _issue(
                    "warning",
                    "expected_numeric_column_is_not_numeric",
                    f"Column '{actual}' has dtype {frame[actual].dtype}.",
                    split=split,
                    well_id=well_id,
                    path=path,
                )
            )
    return issues


def _horizontal_checks(
    frame: pd.DataFrame,
    *,
    split: str,
    well_id: str,
    path: str,
    evaluation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    well_column = _column_name(frame, "WELLNAME")
    if well_column is not None:
        values = frame[well_column].dropna().astype(str)
        unique_values = sorted(values.unique().tolist())
        mismatch_count = int((values != well_id).sum())
        checks["wellname_column"] = {
            "unique_values": unique_values[:20],
            "unique_count": len(unique_values),
            "missing_count": int(frame[well_column].isna().sum()),
            "filename_mismatch_count": mismatch_count,
        }
        if mismatch_count:
            issues.append(
                _issue(
                    "error",
                    "wellname_filename_mismatch",
                    f"{mismatch_count} WELLNAME values differ from filename ID '{well_id}'.",
                    split=split,
                    well_id=well_id,
                    path=path,
                )
            )

    md_column = _column_name(frame, "MD")
    if md_column is not None:
        md = pd.to_numeric(frame[md_column], errors="coerce")
        finite_md = md.dropna()
        checks["md_order"] = {
            "is_monotonic_increasing": bool(finite_md.is_monotonic_increasing),
            "duplicate_count": int(finite_md.duplicated().sum()),
            "non_numeric_or_missing_count": int(md.isna().sum()),
        }
        if not finite_md.is_monotonic_increasing:
            issues.append(
                _issue(
                    "warning",
                    "md_not_monotonic",
                    "MD is not monotonic increasing in file row order.",
                    split=split,
                    well_id=well_id,
                    path=path,
                )
            )

    if evaluation["source"] != "TVT_input_missing":
        issues.append(
            _issue(
                "error",
                "evaluation_zone_not_from_tvt_input",
                "Evaluation zone could not be identified from TVT_input missingness.",
                split=split,
                well_id=well_id,
                path=path,
            )
        )
    elif int(evaluation["evaluation_row_count"]) == 0:
        issues.append(
            _issue(
                "warning",
                "evaluation_zone_empty",
                "TVT_input has no masked evaluation rows.",
                split=split,
                well_id=well_id,
                path=path,
            )
        )
    elif not evaluation["is_suffix"]:
        issues.append(
            _issue(
                "warning",
                "evaluation_zone_not_suffix",
                "Masked TVT_input rows are not one contiguous suffix.",
                split=split,
                well_id=well_id,
                path=path,
            )
        )

    if int(evaluation["tvt_input_mismatch_rows"]):
        issues.append(
            _issue(
                "warning",
                "tvt_input_disagrees_with_tvt",
                "TVT_input differs from TVT where both values are available.",
                split=split,
                well_id=well_id,
                path=path,
            )
        )
    if split == "train" and int(evaluation["evaluation_row_count"]) > int(
        evaluation["tvt_available_in_evaluation_rows"]
    ):
        issues.append(
            _issue(
                "error",
                "train_evaluation_target_missing",
                "Some training evaluation-zone rows lack a finite TVT target.",
                split=split,
                well_id=well_id,
                path=path,
            )
        )
    return checks, issues


def parse_submission_id(value: Any) -> ParsedSubmissionId:
    """Parse ``{well_id}_{row_index}``, preserving underscores in well IDs."""

    if not isinstance(value, str):
        raise ValueError("submission ID must be a string")
    if value != value.strip() or "_" not in value:
        raise ValueError("submission ID must contain a final underscore and no outer whitespace")
    well_id, raw_index = value.rsplit("_", maxsplit=1)
    if not well_id:
        raise ValueError("submission ID has an empty well ID")
    if not raw_index.isascii() or not raw_index.isdigit():
        raise ValueError("submission row index must be a non-negative base-10 integer")
    return ParsedSubmissionId(raw=value, well_id=well_id, row_index=int(raw_index))


def _audit_sample_submission(
    path: Path,
    *,
    root: Path,
    test_rows: Mapping[str, int],
    test_evaluation_indices: Mapping[str, set[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    relative = _relative_path(path, root)
    result: dict[str, Any] = {"path": relative, "exists": path.is_file()}
    if not path.is_file():
        issues.append(
            _issue(
                "error",
                "sample_submission_missing",
                "sample_submission.csv was not found.",
                path=relative,
            )
        )
        return result, issues
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pandas exposes parser/encoding errors through several classes
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        issues.append(
            _issue(
                "error",
                "sample_submission_read_failed",
                result["read_error"],
                path=relative,
            )
        )
        return result, issues

    result["summary"] = summarize_frame(frame)
    id_column = _column_name(frame, "id")
    target_column = _column_name(frame, "tvt")
    if id_column is None:
        issues.append(
            _issue(
                "error",
                "sample_id_column_missing",
                "sample_submission.csv has no 'id' column.",
                path=relative,
            )
        )
        return result, issues
    if target_column is None:
        issues.append(
            _issue(
                "error",
                "sample_target_column_missing",
                "sample_submission.csv has no 'tvt' column.",
                path=relative,
            )
        )

    duplicate_count = int(frame[id_column].duplicated(keep=False).sum())
    parsed: list[ParsedSubmissionId] = []
    malformed: list[dict[str, Any]] = []
    for position, raw_id in enumerate(frame[id_column].tolist()):
        try:
            parsed.append(parse_submission_id(raw_id))
        except ValueError as exc:
            if len(malformed) < 20:
                malformed.append(
                    {
                        "position": position,
                        "value": None if pd.isna(raw_id) else str(raw_id),
                        "error": str(exc),
                    }
                )

    known_wells = set(test_rows)
    parsed_pairs = {(item.well_id, item.row_index) for item in parsed}
    expected_pairs = {
        (well_id, index)
        for well_id, indices in test_evaluation_indices.items()
        for index in indices
    }
    unknown = [item for item in parsed if item.well_id not in known_wells]
    out_of_bounds = [
        item
        for item in parsed
        if item.well_id in test_rows and item.row_index >= test_rows[item.well_id]
    ]
    outside_evaluation = [
        item
        for item in parsed
        if item.well_id in test_evaluation_indices
        and item.row_index < test_rows[item.well_id]
        and item.row_index not in test_evaluation_indices[item.well_id]
    ]
    missing_expected = expected_pairs - parsed_pairs
    unexpected = parsed_pairs - expected_pairs
    referenced_wells = sorted({item.well_id for item in parsed})

    id_checks = {
        "row_count": int(len(frame)),
        "parsed_count": len(parsed),
        "malformed_count": int(len(frame) - len(parsed)),
        "malformed_examples": malformed,
        "duplicate_row_count": duplicate_count,
        "referenced_well_count": len(referenced_wells),
        "referenced_wells": referenced_wells,
        "unknown_well_count": len(unknown),
        "unknown_wells": sorted({item.well_id for item in unknown}),
        "out_of_bounds_count": len(out_of_bounds),
        "out_of_bounds_examples": [item.raw for item in out_of_bounds[:20]],
        "outside_evaluation_zone_count": len(outside_evaluation),
        "outside_evaluation_zone_examples": [item.raw for item in outside_evaluation[:20]],
        "expected_evaluation_id_count": len(expected_pairs),
        "missing_expected_evaluation_id_count": len(missing_expected),
        "missing_expected_evaluation_id_examples": [
            f"{well_id}_{index}" for well_id, index in sorted(missing_expected)[:20]
        ],
        "unexpected_id_count": len(unexpected),
        "unexpected_id_examples": [
            f"{well_id}_{index}" for well_id, index in sorted(unexpected)[:20]
        ],
        "exact_evaluation_coverage": (
            not malformed
            and duplicate_count == 0
            and parsed_pairs == expected_pairs
            and len(parsed) == len(expected_pairs)
        ),
    }
    result["id_checks"] = id_checks

    check_to_issue = (
        (id_checks["malformed_count"], "sample_ids_malformed", "submission IDs are malformed"),
        (duplicate_count, "sample_ids_duplicated", "submission IDs are duplicated"),
        (len(unknown), "sample_ids_unknown_well", "submission IDs reference unknown test wells"),
        (
            len(out_of_bounds),
            "sample_ids_out_of_bounds",
            "submission row indices are out of bounds",
        ),
        (
            len(outside_evaluation),
            "sample_ids_outside_evaluation_zone",
            "submission IDs reference non-evaluation rows",
        ),
        (
            len(missing_expected),
            "sample_ids_missing_evaluation_rows",
            "expected evaluation-row IDs are absent from sample_submission.csv",
        ),
    )
    for count, code, message in check_to_issue:
        if count:
            issues.append(_issue("error", code, f"{count} {message}.", path=relative))
    return result, issues


def audit_dataset(data_root: str | Path) -> dict[str, Any]:
    """Run a complete, read-only audit and return a JSON-serializable report."""

    discovery = discover_well_files(data_root)
    root = discovery.root
    issues = list(discovery.issues)
    aggregates = {
        split: {"horizontal": _RoleAggregate(), "typewell": _RoleAggregate()} for split in SPLITS
    }
    split_details: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    evaluation_totals: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    test_rows: dict[str, int] = {}
    test_evaluation_indices: dict[str, set[int]] = {}

    for well in discovery.wells:
        detail: dict[str, Any] = {
            "well_id": well.well_id,
            "is_complete_pair": well.is_complete_pair,
            "horizontal_files": [_relative_path(path, root) for path in well.horizontal_files],
            "typewell_files": [_relative_path(path, root) for path in well.typewell_files],
            "embedded_typewell_ids": list(well.embedded_typewell_ids),
        }
        split_details[well.split].append(detail)

        for role, path in (
            ("horizontal", well.horizontal_path),
            ("typewell", well.typewell_path),
        ):
            if path is None:
                continue
            relative = _relative_path(path, root)
            try:
                frame = pd.read_csv(path, low_memory=False)
            except Exception as exc:  # keep auditing other wells after one corrupt CSV
                read_error = f"{type(exc).__name__}: {exc}"
                detail[role] = {"path": relative, "read_error": read_error}
                issues.append(
                    _issue(
                        "error",
                        "csv_read_failed",
                        read_error,
                        split=well.split,
                        well_id=well.well_id,
                        path=relative,
                    )
                )
                continue

            summary = summarize_frame(frame)
            detail[role] = {"path": relative, **summary}
            aggregates[well.split][role].add(summary, relative)
            issues.extend(
                _schema_issues(
                    frame,
                    role=role,
                    split=well.split,
                    well_id=well.well_id,
                    path=relative,
                )
            )

            if role != "horizontal":
                continue
            evaluation = detect_evaluation_zone(frame)
            checks, horizontal_issues = _horizontal_checks(
                frame,
                split=well.split,
                well_id=well.well_id,
                path=relative,
                evaluation=evaluation,
            )
            detail["evaluation_zone"] = evaluation
            detail["horizontal_checks"] = checks
            issues.extend(horizontal_issues)
            totals = evaluation_totals[well.split]
            totals["well_count"] += 1
            totals["row_count"] += int(evaluation["row_count"])
            totals["evaluation_row_count"] += int(evaluation["evaluation_row_count"])
            totals["empty_evaluation_well_count"] += int(
                int(evaluation["evaluation_row_count"]) == 0
            )
            totals["non_suffix_well_count"] += int(
                int(evaluation["evaluation_row_count"]) > 0 and not bool(evaluation["is_suffix"])
            )

            if well.split == "test":
                test_rows[well.well_id] = int(len(frame))
                tvt_input = _column_name(frame, "TVT_input")
                if tvt_input is not None:
                    test_evaluation_indices[well.well_id] = set(
                        np.flatnonzero(frame[tvt_input].isna().to_numpy(dtype=bool)).tolist()
                    )
                else:
                    test_evaluation_indices[well.well_id] = set()

    train_ids = {
        well.well_id for well in discovery.wells if well.split == "train" and well.horizontal_files
    }
    test_ids = {
        well.well_id for well in discovery.wells if well.split == "test" and well.horizontal_files
    }
    overlap = sorted(train_ids & test_ids)
    if overlap:
        issues.append(
            _issue(
                "warning",
                "train_test_well_overlap",
                f"{len(overlap)} well IDs occur in both train and test.",
            )
        )

    sample_path = root / "sample_submission.csv"
    sample, sample_issues = _audit_sample_submission(
        sample_path,
        root=root,
        test_rows=test_rows,
        test_evaluation_indices=test_evaluation_indices,
    )
    issues.extend(sample_issues)

    split_report: dict[str, Any] = {}
    for split in SPLITS:
        totals = evaluation_totals[split]
        row_count = int(totals["row_count"])
        eval_rows = int(totals["evaluation_row_count"])
        split_report[split] = {
            "well_count": len(split_details[split]),
            "complete_pair_count": sum(
                bool(detail["is_complete_pair"]) for detail in split_details[split]
            ),
            "aggregate": {
                role: aggregates[split][role].finish() for role in ("horizontal", "typewell")
            },
            "evaluation_zone": {
                "well_count": int(totals["well_count"]),
                "row_count": row_count,
                "evaluation_row_count": eval_rows,
                "evaluation_fraction": float(eval_rows / row_count) if row_count else 0.0,
                "empty_evaluation_well_count": int(totals["empty_evaluation_well_count"]),
                "non_suffix_well_count": int(totals["non_suffix_well_count"]),
            },
            "wells": split_details[split],
        }

    severity_counts = Counter(issue["severity"] for issue in issues)
    code_counts = Counter(issue["code"] for issue in issues)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_root": str(root),
        "discovery": {
            "well_record_count": len(discovery.wells),
            "ignored_csv_count": len(discovery.ignored_csvs),
            "ignored_csvs": [_relative_path(path, root) for path in discovery.ignored_csvs],
            "train_test_well_overlap_count": len(overlap),
            "train_test_well_overlap": overlap,
        },
        "splits": split_report,
        "sample_submission": sample,
        "issue_summary": {
            "total": len(issues),
            "by_severity": dict(sorted(severity_counts.items())),
            "by_code": dict(sorted(code_counts.items())),
        },
        "issues": issues,
    }
    # Refuse to return a report that Python can serialize only via non-standard
    # NaN/Infinity tokens.
    json.dumps(report, allow_nan=False)
    return report


def write_json_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write an audit report as standards-compliant UTF-8 JSON."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary.write_text(f"{payload}\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def issue_codes(report: Mapping[str, Any], severity: str | None = None) -> Iterable[str]:
    """Yield issue codes, optionally filtered by severity."""

    for issue in report.get("issues", []):
        if severity is None or issue.get("severity") == severity:
            yield str(issue["code"])
