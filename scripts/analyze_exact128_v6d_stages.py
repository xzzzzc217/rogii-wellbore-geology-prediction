"""Post-hoc diagnostic attribution for the frozen Exact128 v6d v1 run.

All prediction stages and run audits are loaded and fingerprinted before the
three visible same-ID truth trajectories are read.  This script never writes a
submission and its RMSE values are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_candidate_deltas import (
    REPO_ROOT,
    _delta_stats,
    _load_demo_truth,
    _load_prediction,
    _score,
)

DEFAULT_RUN_DIR = REPO_ROOT / "artifacts" / "kaggle_runs" / "pf_exact128_v6d_v1"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "pf_exact128_v6d_v1_stage_analysis.json"

STAGE_FILES = {
    "sp45_projection": "sp45_projection_submission.csv",
    "learned_trajectory": "learned_trajectory_submission.csv",
    "clean_anchor": "submission_sp45_learned_w0.60.csv",
    "after_visible_prefix": "submission_before_model_package.csv",
    "after_selected_model_package": "submission_before_branch_hedge.csv",
    "final_after_branch_hedge": "submission.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    return [
        {column: _jsonable(row[column]) for column in frame.columns} for _, row in frame.iterrows()
    ]


def _read_log_events(path: Path) -> list[dict[str, Any]]:
    # Kaggle logs occasionally contain a non-UTF8 progress glyph. Replacement
    # preserves the JSON structure and all ASCII timing/control messages.
    events = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(events, list):
        raise ValueError("Kaggle log is not a JSON event list")
    return events


def _event_time(
    events: list[dict[str, Any]],
    needle: str,
    *,
    allow_multiple_take_last: bool = False,
) -> float:
    matches = [float(event["time"]) for event in events if needle in str(event.get("data", ""))]
    if allow_multiple_take_last and matches:
        return matches[-1]
    if len(matches) != 1:
        raise ValueError(f"expected one log event containing {needle!r}, got {len(matches)}")
    return matches[0]


def _stage_timings(events: list[dict[str, Any]]) -> dict[str, Any]:
    control = _event_time(events, "submission profile: exact128_v6d")
    selector_start = _event_time(events, "Processing 1/3: 000d7d20")
    projection_done = _event_time(events, "saved sp45_projection_submission.csv")
    learned_start = _event_time(events, "DATA: /kaggle/input/competitions/")
    learned_done = _event_time(events, "submission.csv written (14151 rows) in 71s")
    blend_done = _event_time(events, "wrote final submission.csv from")
    vp_start = _event_time(events, "Visible-prefix calibration: {")
    vp_done = _event_time(events, "Visible-prefix selected submission.csv:")
    package_done = _event_time(events, "model package p95 diff")
    hedge_done = _event_time(events, "PF seed-branch hedge:")
    final_audit = _event_time(events, "Exact128 v6d final audit:")
    nbconvert_done = _event_time(
        events,
        "[NbConvertApp] Writing",
        allow_multiple_take_last=True,
    )

    def duration(start: float, end: float) -> float:
        return float(end - start)

    return {
        "source": "Kaggle JSON log event timestamps",
        "kernel_start_to_nbconvert_complete_sec": float(nbconvert_done),
        "profile_control_to_final_audit_sec": duration(control, final_audit),
        "initialization_and_anchor_setup_before_selector_sec": duration(
            control,
            selector_start,
        ),
        "selector_and_projection_sec": duration(selector_start, projection_done),
        "learned_trajectory_sec": duration(learned_start, learned_done),
        "clean_blend_write_sec": duration(learned_done, blend_done),
        "visible_prefix_sec": duration(vp_start, vp_done),
        "model_package_sec": duration(vp_done, package_done),
        "branch_hedge_sec": duration(package_done, hedge_done),
        "final_contract_and_audit_sec": duration(hedge_done, final_audit),
        "event_times_sec": {
            "profile_control": control,
            "selector_start": selector_start,
            "projection_done": projection_done,
            "learned_start": learned_start,
            "learned_done": learned_done,
            "blend_done": blend_done,
            "visible_prefix_start": vp_start,
            "visible_prefix_done": vp_done,
            "model_package_done": package_done,
            "branch_hedge_done": hedge_done,
            "final_audit": final_audit,
            "nbconvert_done": nbconvert_done,
        },
    }


def _find_exact_package_candidate(
    run_dir: Path,
    package_rows: list[dict[str, Any]],
) -> Path:
    matches = [
        row
        for row in package_rows
        if abs(float(row["gated_max_weight"]) - 0.00425) <= 1e-12
        and abs(float(row["scale"]) - 6.0) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one exact 0.00425/6 package row, got {len(matches)}")
    path = run_dir / str(matches[0]["file"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_report(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()

    # Freeze every artifact and prediction in memory before diagnostic truth is
    # touched.  This ordering is part of the anti-tuning evidence boundary.
    package_rows = _records(run_dir / "model_package_correction_report.csv")
    package_summary = _records(run_dir / "model_package_correction_summary.csv")
    hedge_rows = _records(run_dir / "pf_seed_branch_hedge_report.csv")
    vp_audit = _json(run_dir / "gold_prefix_submission_audit.json")
    scoring_audit = _json(run_dir / "exact128_v6d_scoring_path_audit.json")
    contract_audit = _json(run_dir / "hidden_submission_contract_audit.json")
    final_audit = _json(run_dir / "exact128_v6d_final_audit.json")
    log_events = _read_log_events(run_dir / "rogii-exact128-v6d.log")
    stage_timings = _stage_timings(log_events)

    stage_paths = {name: run_dir / filename for name, filename in STAGE_FILES.items()}
    stage_paths["exact_model_package_candidate"] = _find_exact_package_candidate(
        run_dir,
        package_rows,
    )
    predictions = {name: _load_prediction(path) for name, path in stage_paths.items()}
    reference_ids = predictions["clean_anchor"].index
    for name, prediction in predictions.items():
        if not prediction.index.equals(reference_ids):
            raise ValueError(f"{name}: ID order differs from clean_anchor")

    frozen_paths = sorted(
        (path for path in run_dir.iterdir() if path.is_file()),
        key=str,
    )
    frozen_artifacts = {
        path.name: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in frozen_paths
    }

    # Diagnostic truth is deliberately read only after all predictions and
    # audits above are frozen and fingerprinted.
    truth, wells = _load_demo_truth(
        reference_ids,
        REPO_ROOT / "data" / "competition" / "train",
    )
    stage_scores = {
        name: _score(prediction, truth, wells) for name, prediction in predictions.items()
    }
    layer_pairs = {
        "projection_to_clean_blend": ("sp45_projection", "clean_anchor"),
        "visible_prefix": ("clean_anchor", "after_visible_prefix"),
        "hypothetical_exact_model_package": (
            "after_visible_prefix",
            "exact_model_package_candidate",
        ),
        "selected_model_package": (
            "after_visible_prefix",
            "after_selected_model_package",
        ),
        "branch_hedge": (
            "after_selected_model_package",
            "final_after_branch_hedge",
        ),
        "full_actual_stack": ("clean_anchor", "final_after_branch_hedge"),
    }
    layer_moves = {
        name: {
            "before_stage": before,
            "after_stage": after,
            **_delta_stats(
                predictions[before],
                predictions[after],
                truth,
                wells,
            ),
        }
        for name, (before, after) in layer_pairs.items()
    }

    exact_package_row = next(
        row
        for row in package_rows
        if abs(float(row["gated_max_weight"]) - 0.00425) <= 1e-12
        and abs(float(row["scale"]) - 6.0) <= 1e-12
    )
    report = {
        "scope": {
            "label": "pf_exact128_v6d_v1",
            "run_dir": str(run_dir),
            "diagnostic_only": True,
            "not_hidden_evidence": True,
            "truth_loaded_after_all_predictions_and_audits_were_frozen": True,
            "truth_source": "same-ID train TVT used only for post-hoc scoring",
            "rows": int(len(reference_ids)),
            "wells": sorted(wells.unique()),
            "candidate_was_not_built_or_selected_from_this_truth": True,
            "frozen_local_files_before_truth": int(len(frozen_artifacts)),
            "downloaded_kaggle_outputs": int(
                len(frozen_artifacts) - int("independent_submission_audit.json" in frozen_artifacts)
            ),
        },
        "frozen_artifacts": frozen_artifacts,
        "stage_paths": {name: str(path) for name, path in stage_paths.items()},
        "stage_scores": stage_scores,
        "layer_moves": layer_moves,
        "runtime_decisions": {
            "stage_timings": stage_timings,
            "scoring_audit": scoring_audit,
            "visible_prefix_audit": vp_audit,
            "model_package_exact_row": exact_package_row,
            "model_package_all_rows": package_rows,
            "model_package_summary": package_summary,
            "branch_hedge_rows": hedge_rows,
            "contract_audit": contract_audit,
            "final_audit": final_audit,
        },
        "interpretation_boundary": {
            "valid_uses": [
                "regression diagnostic",
                "stage-attribution sanity check",
                "ordinary-run safety verification",
            ],
            "invalid_uses": [
                "hidden-score estimate",
                "same-three-well parameter tuning",
                "medal or leaderboard claim",
            ],
        },
    }
    return report


def _self_check(report: dict[str, Any]) -> None:
    scope = report["scope"]
    if scope["rows"] != 14_151 or len(scope["wells"]) != 3:
        raise AssertionError("expected the frozen 14,151-row, three-well diagnostic")
    decisions = report["runtime_decisions"]
    scoring = decisions["scoring_audit"]
    if scoring["profile"] != "exact128_v6d":
        raise AssertionError("unexpected scoring profile")
    if scoring["selector_pf_seeds"] != 128 or scoring["learned_pf_seeds"] != 128:
        raise AssertionError("128/128 PF configuration drifted")
    if scoring["visible_prefix"] != {
        "profile": "balanced",
        "cal_seeds": 24,
        "final_seeds": 48,
        "particles": 350,
        "cut_fracs": [0.5, 0.65, 0.75],
    }:
        raise AssertionError("balanced visible-prefix configuration drifted")
    if scoring["model_package"]["max_weight"] != 0.00425:
        raise AssertionError("model-package max weight drifted")
    if scoring["model_package"]["scale"] != 6.0:
        raise AssertionError("model-package scale drifted")
    if scoring["model_package"]["p95_disable"] != 25.0:
        raise AssertionError("model-package p95 guard drifted")
    if not scoring["generalized_branch_hedge"] or scoring["u_continuity"]:
        raise AssertionError("hedge/U composition drifted")
    if not scoring["query_well_self_exclusion"]:
        raise AssertionError("query self-exclusion audit is false")
    if any(scoring["unsafe_controls"].values()):
        raise AssertionError(f"unsafe runtime control active: {scoring['unsafe_controls']}")
    vp = decisions["visible_prefix_audit"]
    if vp["contact_override_enabled"] or vp["selected_profile"] != "balanced":
        raise AssertionError("visible-prefix safety/profile drifted")
    if vp["budget_guard_stopped"]:
        raise AssertionError("visible-prefix budget fallback unexpectedly fired")
    package = decisions["model_package_exact_row"]
    selected_move = report["layer_moves"]["selected_model_package"]
    if bool(package["disabled_by_diff_guard"]) and selected_move["moved_rows"]:
        raise AssertionError("guard-disabled model package changed selected predictions")
    contract = decisions["contract_audit"]
    if (
        not contract["id_order_matches_hidden_sample"]
        or contract["missing_ids"]
        or contract["extra_ids"]
    ):
        raise AssertionError("dynamic sample contract failed")
    if report["layer_moves"]["full_actual_stack"]["moved_rows"] < 0:
        raise AssertionError("invalid moved-row count")
    timings = decisions["stage_timings"]
    if not 800.0 <= timings["profile_control_to_final_audit_sec"] <= 1200.0:
        raise AssertionError("unexpected total scoring-path runtime")
    if not 100.0 <= timings["visible_prefix_sec"] <= 300.0:
        raise AssertionError("unexpected visible-prefix runtime")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only attribution of the frozen Exact128 v6d v1 run."
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN_DIR)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.run_dir)
    if args.self_check:
        _self_check(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exact128 v6d stage report: {args.output}")
    print(
        "Diagnostic pooled RMSE:",
        {
            name: round(details["pooled_rmse"], 6)
            for name, details in report["stage_scores"].items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
