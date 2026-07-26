"""Post-hoc stage attribution for a frozen Guard640 Kaggle run.

The script loads every frozen prediction stage before reading the visible demo
truth. It never creates a submission and is intended only for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from analyze_candidate_deltas import (
    REPO_ROOT,
    _delta_stats,
    _load_demo_truth,
    _load_prediction,
    _score,
)

STAGE_FILES = {
    "clean_anchor": "submission_sp45_learned_w0.60.csv",
    "after_vp": "submission_before_model_package.csv",
    "after_selected_model_package": "submission_before_branch_hedge.csv",
    "after_hedge": "submission_before_continuity.csv",
    "final_after_continuity": "submission.csv",
}


def _find_model_candidate(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else run_dir / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted(run_dir.glob("submission_model_package_gated_*.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"{run_dir}: expected one gated model-package candidate, found {len(matches)}"
        )
    return matches[0]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _read_model_package_report(run_dir: Path) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(run_dir / "model_package_correction_report.csv")
    if len(frame) != 1:
        raise ValueError("expected one model-package correction report row")
    row = frame.iloc[0]
    return {column: _jsonable(row[column]) for column in frame.columns}


def _read_safety_controls(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "submission_audit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("safety_controls", {})


def build_report(
    run_dir: Path,
    *,
    label: str,
    model_candidate: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    paths = {name: run_dir / filename for name, filename in STAGE_FILES.items()}
    paths["unselected_model_package_candidate"] = _find_model_candidate(run_dir, model_candidate)
    predictions = {name: _load_prediction(path) for name, path in paths.items()}

    reference_ids = predictions["clean_anchor"].index
    for name, prediction in predictions.items():
        if not prediction.index.equals(reference_ids):
            raise ValueError(f"{name}: id order differs from clean_anchor")

    truth, wells = _load_demo_truth(
        reference_ids,
        REPO_ROOT / "data" / "competition" / "train",
    )
    scores = {name: _score(prediction, truth, wells) for name, prediction in predictions.items()}
    layer_pairs = {
        "visible_prefix": ("clean_anchor", "after_vp"),
        "hypothetical_model_package_candidate": (
            "after_vp",
            "unselected_model_package_candidate",
        ),
        "selected_model_package": (
            "after_vp",
            "after_selected_model_package",
        ),
        "branch_hedge": ("after_selected_model_package", "after_hedge"),
        "u_continuity": ("after_hedge", "final_after_continuity"),
        "full_actual_stack": ("clean_anchor", "final_after_continuity"),
    }
    layers = {
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

    package_report = _read_model_package_report(run_dir)
    safety_controls = _read_safety_controls(run_dir)
    actual_layers = [
        "visible_prefix",
        "selected_model_package",
        "branch_hedge",
        "u_continuity",
    ]
    ranked_actual_layers = sorted(
        (
            {
                "layer": name,
                "pooled_rmse_delta_after_minus_before": layers[name][
                    "pooled_rmse_delta_after_minus_before"
                ],
            }
            for name in actual_layers
        ),
        key=lambda item: item["pooled_rmse_delta_after_minus_before"],
    )

    vp_delta = layers["visible_prefix"]["pooled_rmse_delta_after_minus_before"]
    hedge_delta = layers["branch_hedge"]["pooled_rmse_delta_after_minus_before"]
    continuity_delta = layers["u_continuity"]["pooled_rmse_delta_after_minus_before"]
    guard_disabled = bool(package_report["disabled_by_diff_guard"])
    report = {
        "scope": {
            "label": label,
            "run_dir": str(run_dir),
            "diagnostic_only": True,
            "truth_loaded_after_frozen_predictions": True,
            "truth_source": "same-ID train TVT used only for post-hoc scoring",
            "rows": int(len(reference_ids)),
            "wells": sorted(wells.unique()),
            "prohibited_for_recommendation": [
                "same-ID lookup",
                "contact reconstruction",
                "overlap recovery",
            ],
        },
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "safety_controls": safety_controls,
        "stage_scores": scores,
        "layer_moves": layers,
        "model_package_report": package_report,
        "actual_layers_ranked_best_to_worst": ranked_actual_layers,
        "go_no_go": {
            "visible_prefix": (
                "no_go_from_demo_evidence"
                if vp_delta >= 0
                else "weak_go_for_broader_validation_only"
            ),
            "model_package_candidate": (
                "no_go_guard_rejected" if guard_disabled else "selected_in_actual_stack"
            ),
            "branch_hedge": (
                "no_go_from_demo_evidence"
                if hedge_delta >= 0
                else "weak_go_for_broader_validation_only"
            ),
            "u_continuity": (
                "go_keep_for_broader_validation"
                if continuity_delta < 0
                else "no_go_from_demo_evidence"
            ),
            "actual_composition": (
                "go_for_independent_validation"
                if scores["final_after_continuity"]["pooled_rmse"]
                < scores["clean_anchor"]["pooled_rmse"]
                else "no_go_from_demo_evidence"
            ),
        },
    }
    return report


def _self_check(report: dict[str, Any]) -> None:
    if report["scope"]["rows"] != 14_151:
        raise AssertionError("expected 14,151 visible demo rows")
    if len(report["scope"]["wells"]) != 3:
        raise AssertionError("expected exactly three visible demo wells")
    controls = report["safety_controls"]
    forbidden_flags = [
        "allow_same_id_train_lookup",
        "run_guarded_overlap_override",
        "run_overlap_dry_run_probe",
        "run_exact_match_recovery",
        "apply_global_lb_bias",
    ]
    active_forbidden = [name for name in forbidden_flags if bool(controls.get(name))]
    if active_forbidden:
        raise AssertionError(f"forbidden controls active: {active_forbidden}")
    package_report = report["model_package_report"]
    selected_layer = report["layer_moves"]["selected_model_package"]
    if bool(package_report["disabled_by_diff_guard"]) and selected_layer["moved_rows"]:
        raise AssertionError("guard-disabled model package changed selected predictions")
    if bool(package_report["selected_for_submission_csv"]) == bool(
        package_report["disabled_by_diff_guard"]
    ):
        raise AssertionError("inconsistent model-package selection/guard state")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc, diagnostic-only attribution of Guard640 run stages."
    )
    parser.add_argument("run_dir", type=Path, help="Downloaded Kaggle run output directory.")
    parser.add_argument("--label", default=None, help="Report label; defaults to directory name.")
    parser.add_argument(
        "--model-candidate",
        type=Path,
        default=None,
        help="Gated model-package candidate path/name; auto-detected by default.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="JSON report path.")
    parser.add_argument("--self-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label = args.label or args.run_dir.name
    output = args.output or REPO_ROOT / "artifacts" / f"{label}_stage_analysis.json"
    report = build_report(
        args.run_dir,
        label=label,
        model_candidate=args.model_candidate,
    )
    if args.self_check:
        _self_check(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Guard640 stage report: {output}")
    print(
        "Stage pooled RMSE:",
        {
            name: round(details["pooled_rmse"], 6)
            for name, details in report["stage_scores"].items()
        },
    )
    print("Go/no-go:", report["go_no_go"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
