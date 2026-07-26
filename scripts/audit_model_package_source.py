from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VISIBLE_SAME_ID_WELLS = ("000d7d20", "00bbac68", "00e12e8b")
MODEL_ID_COLUMNS = ("id", "well_id", "row_index")
MODEL_TARGET_COLUMNS = ("TVT", "target_tvt", "target_delta", "target_delta_from_last_known")
GEOMETRY_COLUMNS = ("prefix_len", "tail_len", "tail_row_number", "n_rows", "row_frac")
FIXED_ROW_COUNTS = (14151, 4301)


def _line_numbers(text: str, pattern: str, *, flags: int = 0) -> list[int]:
    rx = re.compile(pattern, flags)
    return [i for i, line in enumerate(text.splitlines(), start=1) if rx.search(line)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_core(core_path: Path, competition_root: Path):
    os.environ["ROGII_DATA_DIR"] = str(competition_root.resolve())
    module_name = "rogii_feature_core_source_audit"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one function {name!r}, found {len(matches)}")
    return matches[0]


def _subscript_key(node: ast.Subscript) -> str | None:
    value = node.slice
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _test_mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(part, ast.Name) and part.id == name for part in ast.walk(node))


def _guarded_query_target_accesses(core_text: str) -> list[dict[str, Any]]:
    tree = ast.parse(core_text)
    function = _find_function(tree, "make_tail_features_for_well")
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    accesses: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript) or _subscript_key(node) != "TVT":
            continue
        receiver = _name_of(node.value)
        if receiver not in {"df", "tail", "prefix"}:
            continue
        current: ast.AST | None = node
        guarding_if = None
        while current is not None and current is not function:
            current = parent.get(current)
            if isinstance(current, ast.If) and _test_mentions_name(current.test, "include_target"):
                guarding_if = current
                break
        accesses.append(
            {
                "line": int(node.lineno),
                "receiver": receiver,
                "guarded_by_include_target": guarding_if is not None,
                "guard_line": int(guarding_if.lineno) if guarding_if is not None else None,
            }
        )
    return accesses


def _builder_call_flags(builder_text: str) -> dict[str, Any]:
    tree = ast.parse(builder_text)
    function = _find_function(tree, "build_features")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_tail_feature_frame"
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"Expected one build_tail_feature_frame call in build_features, found {len(calls)}"
        )
    call = calls[0]
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}

    def constant_bool(name: str) -> bool | None:
        node = kwargs.get(name)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        return None

    return {
        "call_line": int(call.lineno),
        "include_target": constant_bool("include_target"),
        "exclude_query_well_from_formation": constant_bool("exclude_query_well_from_formation"),
    }


def _selected_plane_wells(imputer, xy_query: np.ndarray, self_wid: str | None) -> list[str]:
    k = int(imputer.k)
    n_fetch = min(max(k + 5, k), len(imputer.df))
    dist, idx = imputer.tree.query(np.atleast_2d(xy_query) / imputer.scale, k=n_fetch)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    if self_wid is not None and self_wid in imputer.well_to_idx:
        dist = np.where(idx == imputer.well_to_idx[self_wid], np.inf, dist)
    valid = np.isfinite(dist)
    kth = min(max(k - 1, 0), dist.shape[1] - 1)
    order = np.argpartition(np.where(valid, dist, np.inf), kth=kth, axis=1)[:, :k]
    chosen = np.take_along_axis(idx, order, axis=1)[0]
    chosen_dist = np.take_along_axis(dist, order, axis=1)[0]
    return [
        str(imputer.df.iloc[int(i)]["well_id"])
        for i, distance in zip(chosen, chosen_dist, strict=True)
        if np.isfinite(distance)
    ]


def _selected_row_wells(
    imputer,
    xy_query: np.ndarray,
    self_wid: str | None,
    *,
    k: int = 20,
    extra_fetch: int = 450,
) -> list[str]:
    n_fetch = min(max(k + extra_fetch, k), len(imputer.ancc))
    dist, idx = imputer.tree.query(
        np.atleast_2d(xy_query) / imputer.scale,
        k=n_fetch,
        workers=-1,
    )
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    if self_wid is not None:
        dist = np.where(imputer.wids[idx] == self_wid, np.inf, dist)
    valid = np.isfinite(dist)
    kth = min(max(k - 1, 0), dist.shape[1] - 1)
    order = np.argpartition(np.where(valid, dist, np.inf), kth=kth, axis=1)[:, :k]
    chosen = np.take_along_axis(idx, order, axis=1)[0]
    chosen_dist = np.take_along_axis(dist, order, axis=1)[0]
    return [
        str(imputer.wids[int(i)])
        for i, distance in zip(chosen, chosen_dist, strict=True)
        if np.isfinite(distance)
    ]


class _PlaneSpy:
    ready = True

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def impute(self, xy_query, self_wid=None, k=None):
        del k
        self.calls.append(self_wid)
        n = len(np.atleast_2d(xy_query))
        return np.zeros((n, 6), dtype=np.float32), np.ones(n, dtype=np.float32)


class _RowSpy:
    ready = True

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def impute(self, xy_query, self_wid=None, k=20, extra_fetch=450):
        del k, extra_fetch
        self.calls.append(self_wid)
        n = len(np.atleast_2d(xy_query))
        values = np.zeros(n, dtype=np.float32)
        return values, values.copy(), np.ones(n, dtype=np.float32)


def _runtime_exclusion_checks(core, competition_root: Path) -> dict[str, Any]:
    train_wells = np.array(sorted(core.train_h_ids))
    plane = core.FormationPlaneKNN(train_wells, core.TRAIN_DIR, k=10)
    row = core.RowANCCKNN(train_wells, core.TRAIN_DIR, samples_per_well=400, seed=42)
    if not plane.ready or not row.ready:
        raise AssertionError("Formation imputers were not ready on the full training set")

    results: dict[str, Any] = {}
    for well_id in VISIBLE_SAME_ID_WELLS:
        test_path = competition_root / "test" / f"{well_id}__horizontal_well.csv"
        test_df = pd.read_csv(test_path, usecols=["X", "Y", "TVT_input"])
        hidden = np.flatnonzero(test_df["TVT_input"].isna().to_numpy())
        if not len(hidden):
            raise AssertionError(f"{well_id} has no hidden suffix")
        xy_query = test_df.loc[int(hidden[0]), ["X", "Y"]].to_numpy(dtype=float)

        plane_without = _selected_plane_wells(plane, xy_query, None)
        plane_with = _selected_plane_wells(plane, xy_query, well_id)
        row_without = _selected_row_wells(row, xy_query, None)
        row_with = _selected_row_wells(row, xy_query, well_id)
        _, plane_distance_without = plane.impute(xy_query, self_wid=None)
        _, plane_distance_with = plane.impute(xy_query, self_wid=well_id)
        _, _, row_distance_without = row.impute(xy_query, self_wid=None)
        _, _, row_distance_with = row.impute(xy_query, self_wid=well_id)

        plane_spy = _PlaneSpy()
        row_spy = _RowSpy()
        frame = core.make_tail_features_for_well(
            well_id,
            core.TEST_DIR,
            include_target=False,
            use_beam_features=False,
            required_feature_columns=["formation_plane_ancc", "formation_row_ancc"],
            formation_plane_imputer=plane_spy,
            row_ancc_imputer=row_spy,
            exclude_query_well_from_formation=True,
        )

        checks = {
            "present_in_train_imputer": well_id in plane.well_to_idx,
            "plane_no_exclusion_self_count": int(plane_without.count(well_id)),
            "plane_exclusion_self_count": int(plane_with.count(well_id)),
            "row_no_exclusion_self_count": int(row_without.count(well_id)),
            "row_exclusion_self_count": int(row_with.count(well_id)),
            "plane_min_distance_no_exclusion": float(plane_distance_without[0]),
            "plane_min_distance_with_exclusion": float(plane_distance_with[0]),
            "row_min_distance_no_exclusion": float(row_distance_without[0]),
            "row_min_distance_with_exclusion": float(row_distance_with[0]),
            "plane_spy_self_wids": plane_spy.calls,
            "row_spy_self_wids": row_spy.calls,
            "target_columns_in_inference_frame": sorted(
                set(MODEL_TARGET_COLUMNS).intersection(frame.columns)
            ),
            "inference_rows": int(len(frame)),
        }
        if checks["plane_exclusion_self_count"] != 0:
            raise AssertionError(f"Plane imputer retained query well {well_id}")
        if checks["row_exclusion_self_count"] != 0:
            raise AssertionError(f"Row imputer retained query well {well_id}")
        if plane_spy.calls != [well_id, well_id]:
            raise AssertionError(
                f"Plane self_wid propagation failed for {well_id}: {plane_spy.calls}"
            )
        if row_spy.calls != [well_id, well_id]:
            raise AssertionError(f"Row self_wid propagation failed for {well_id}: {row_spy.calls}")
        if checks["target_columns_in_inference_frame"]:
            raise AssertionError(
                f"Inference frame exposed target columns for {well_id}: "
                f"{checks['target_columns_in_inference_frame']}"
            )
        results[well_id] = checks

    # Use a train-schema file that physically contains TVT to prove the False gate,
    # rather than relying only on the target-free test-file schema.
    probe_well = VISIBLE_SAME_ID_WELLS[0]
    probe = core.make_tail_features_for_well(
        probe_well,
        core.TRAIN_DIR,
        include_target=False,
        use_beam_features=False,
        required_feature_columns=[],
        exclude_query_well_from_formation=True,
    )
    train_schema_target_columns = sorted(set(MODEL_TARGET_COLUMNS).intersection(probe.columns))
    if train_schema_target_columns:
        raise AssertionError(
            "include_target=False exposed target columns on a file that contains TVT: "
            f"{train_schema_target_columns}"
        )
    return {
        "train_wells_in_imputers": int(len(train_wells)),
        "visible_same_id_wells": results,
        "train_schema_false_gate_probe": {
            "well_id": probe_well,
            "rows": int(len(probe)),
            "target_columns": train_schema_target_columns,
        },
    }


def audit(source_root: Path, competition_root: Path) -> dict[str, Any]:
    builder_path = source_root / "build_features.py"
    core_path = source_root / "rogii_feature_core.py"
    feature_columns_path = source_root / "full_source" / "feature_columns.json"
    manifest_path = source_root / "model_package_manifest.json"
    blend_path = source_root / "blend_config.json"
    builder_mirror = source_root / "full_source" / "build_features.py"
    core_mirror = source_root / "full_source" / "rogii_feature_core.py"
    required = (builder_path, core_path, feature_columns_path, manifest_path, blend_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing audit inputs: {missing}")

    builder_text = builder_path.read_text(encoding="utf-8")
    core_text = core_path.read_text(encoding="utf-8")
    feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    columns = list(feature_columns["drift_ncc_v1"])
    searchable = "\n".join(
        [
            builder_text,
            core_text,
            feature_columns_path.read_text(encoding="utf-8"),
            manifest_path.read_text(encoding="utf-8"),
            blend_path.read_text(encoding="utf-8"),
        ]
    )

    well_id_pattern = r"(?<![A-Za-z0-9])[0-9a-fA-F]{8}(?![A-Za-z0-9])"
    literal_well_ids = sorted(set(re.findall(well_id_pattern, searchable)))
    fixed_row_hits = {
        str(value): _line_numbers(searchable, rf"(?<!\d){value}(?!\d)")
        for value in FIXED_ROW_COUNTS
    }
    hash_hits = _line_numbers(
        searchable,
        r"sha256|hashlib|md5|prediction_fingerprint|expected_(?:hash|rows?|ids?)|canary",
        flags=re.IGNORECASE,
    )
    builder_flags = _builder_call_flags(builder_text)
    query_target_accesses = _guarded_query_target_accesses(core_text)

    if builder_flags["include_target"] is not False:
        raise AssertionError(f"Builder did not freeze include_target=False: {builder_flags}")
    if builder_flags["exclude_query_well_from_formation"] is not True:
        raise AssertionError(f"Builder did not freeze query exclusion: {builder_flags}")
    if not query_target_accesses or not all(
        item["guarded_by_include_target"] for item in query_target_accesses
    ):
        raise AssertionError(f"Unguarded query-target access found: {query_target_accesses}")
    if literal_well_ids or any(fixed_row_hits.values()) or hash_hits:
        raise AssertionError(
            "Static identifier/count/hash sentinel found: "
            f"ids={literal_well_ids}, rows={fixed_row_hits}, hash_lines={hash_hits}"
        )

    core = _load_core(core_path, competition_root)
    runtime = _runtime_exclusion_checks(core, competition_root)

    return {
        "verdict": {
            "hidden_query_target_free": True,
            "query_well_exclusion_effective": True,
            "static_target_or_identifier_leak_found": False,
        },
        "source": {
            "builder": str(builder_path.resolve()),
            "core": str(core_path.resolve()),
            "feature_columns": str(feature_columns_path.resolve()),
            "builder_sha256": _sha256(builder_path),
            "core_sha256": _sha256(core_path),
            "fresh_download_mirror_matches": {
                "builder": builder_mirror.exists()
                and _sha256(builder_path) == _sha256(builder_mirror),
                "core": core_mirror.exists() and _sha256(core_path) == _sha256(core_mirror),
            },
        },
        "static": {
            "builder_call": builder_flags,
            "query_horizontal_tvt_accesses": query_target_accesses,
            "literal_well_ids": literal_well_ids,
            "fixed_row_count_hits": fixed_row_hits,
            "hash_or_canary_lines": hash_hits,
            "model_feature_count": int(len(columns)),
            "direct_identifier_features": [name for name in MODEL_ID_COLUMNS if name in columns],
            "direct_target_features": [name for name in MODEL_TARGET_COLUMNS if name in columns],
            "geometry_features": [name for name in GEOMETRY_COLUMNS if name in columns],
            "legal_train_target_import_side_effect": {
                "estimate_function_line": 2768,
                "train_tvt_read_line": 2772,
                "module_level_call_line": 2816,
                "scoring_uses_of_result": 0,
            },
        },
        "runtime": runtime,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/external/model_package_audit"),
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=Path("data/competition"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model_package_source_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.source_root, args.competition_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["verdict"], sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
