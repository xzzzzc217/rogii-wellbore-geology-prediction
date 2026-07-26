"""Post-hoc delta analysis for the three visible ROGII demo wells.

This script deliberately keeps demo truth separate from candidate construction. It
reads truth only after loading frozen Kaggle run artifacts and never writes a
submission. The resulting metrics are diagnostic evidence, not a tuning objective.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "candidate_delta_analysis.json"
RUNS_ROOT = REPO_ROOT / "artifacts" / "kaggle_runs"

CANDIDATE_PATHS = {
    "safe_v2_final": RUNS_ROOT / "pf_safe_v2_v1" / "submission.csv",
    "safe_v3_base": RUNS_ROOT / "pf_safe_v3_v1" / "submission_sp45_learned_w0.60.csv",
    "safe_v3_final": RUNS_ROOT / "pf_safe_v3_v1" / "submission.csv",
    "exact6451_clean_anchor": RUNS_ROOT
    / "pf_public_reproduction_v1"
    / "submission_sp45_learned_w0.60.csv",
    "exact6451_pre_modelpkg": RUNS_ROOT
    / "pf_public_reproduction_v1"
    / "submission_before_model_package.csv",
    "exact6451_pre_hedge": RUNS_ROOT
    / "pf_public_reproduction_v1"
    / "submission_before_branch_hedge.csv",
    "exact6451_raw_final": RUNS_ROOT / "pf_public_reproduction_v1" / "submission.csv",
}

CANDIDATE_RISK = {
    "safe_v2_final": "clean_target_free",
    "safe_v3_base": "clean_target_free",
    "safe_v3_final": "clean_target_free",
    "exact6451_clean_anchor": "clean_target_free_before_forbidden_layers",
    "exact6451_pre_modelpkg": "confounded_by_same_well_contact_overlap",
    "exact6451_pre_hedge": "confounded_by_same_well_contact_overlap",
    "exact6451_raw_final": "confounded_by_same_well_contact_overlap",
}


def _float(value: Any) -> float:
    return float(value)


def _load_prediction(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["id", "tvt"]:
        raise ValueError(f"{path}: expected id,tvt columns, got {frame.columns.tolist()}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids")
    values = pd.to_numeric(frame["tvt"], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite tvt values")
    return pd.Series(values.to_numpy(dtype=float), index=frame["id"].astype(str), name=path.name)


def _load_demo_truth(ids: pd.Index, train_dir: Path) -> tuple[pd.Series, pd.Series]:
    wells = pd.Series([item.rsplit("_", 1)[0] for item in ids], index=ids, name="well")
    rows = pd.Series(
        [int(item.rsplit("_", 1)[1]) for item in ids],
        index=ids,
        name="row_idx",
        dtype=int,
    )
    truth = pd.Series(index=ids, dtype=float, name="truth")
    for well in sorted(wells.unique()):
        frame = pd.read_csv(train_dir / f"{well}__horizontal_well.csv", usecols=["TVT"])
        mask = wells.eq(well)
        row_idx = rows.loc[mask].to_numpy(dtype=int)
        if row_idx.min() < 0 or row_idx.max() >= len(frame):
            raise IndexError(f"{well}: submission row index outside diagnostic truth frame")
        truth.loc[mask] = frame.iloc[row_idx]["TVT"].to_numpy(dtype=float)
    if not np.isfinite(truth).all():
        raise ValueError("diagnostic truth contains non-finite values")
    return truth, wells


def _rmse(error: np.ndarray | pd.Series) -> float:
    array = np.asarray(error, dtype=float)
    return _float(np.sqrt(np.mean(np.square(array))))


def _score(prediction: pd.Series, truth: pd.Series, wells: pd.Series) -> dict[str, Any]:
    error = prediction - truth
    return {
        "rows": int(len(error)),
        "pooled_rmse": _rmse(error),
        "per_well_rmse": {
            well: _rmse(error.loc[wells.eq(well)]) for well in sorted(wells.unique())
        },
    }


def _delta_stats(
    before: pd.Series,
    after: pd.Series,
    truth: pd.Series,
    wells: pd.Series,
) -> dict[str, Any]:
    delta = after - before
    before_score = _score(before, truth, wells)
    after_score = _score(after, truth, wells)
    per_well: dict[str, Any] = {}
    for well in sorted(wells.unique()):
        values = delta.loc[wells.eq(well)]
        per_well[well] = {
            "mean_signed_move": _float(values.mean()),
            "mean_abs_move": _float(values.abs().mean()),
            "p95_abs_move": _float(values.abs().quantile(0.95)),
            "max_abs_move": _float(values.abs().max()),
            "rms_move": _rmse(values),
            "rmse_before": before_score["per_well_rmse"][well],
            "rmse_after": after_score["per_well_rmse"][well],
            "rmse_delta_after_minus_before": (
                after_score["per_well_rmse"][well] - before_score["per_well_rmse"][well]
            ),
        }
    return {
        "mean_signed_move": _float(delta.mean()),
        "mean_abs_move": _float(delta.abs().mean()),
        "p95_abs_move": _float(delta.abs().quantile(0.95)),
        "max_abs_move": _float(delta.abs().max()),
        "rms_move": _rmse(delta),
        "moved_rows": int(np.count_nonzero(np.abs(delta.to_numpy()) > 1e-12)),
        "pooled_rmse_before": before_score["pooled_rmse"],
        "pooled_rmse_after": after_score["pooled_rmse"],
        "pooled_rmse_delta_after_minus_before": (
            after_score["pooled_rmse"] - before_score["pooled_rmse"]
        ),
        "per_well": per_well,
    }


def _analytic_alpha(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is not None:
        first = first[mask]
        second = second[mask]
        truth = truth[mask]
    direction = second - first
    denominator = _float(np.dot(direction, direction))
    if denominator <= 0:
        return 0.0
    alpha = -_float(np.dot(first - truth, direction)) / denominator
    return _float(np.clip(alpha, 0.0, 1.0))


def _blend_analysis(
    first_name: str,
    second_name: str,
    predictions: dict[str, pd.Series],
    truth: pd.Series,
    wells: pd.Series,
) -> dict[str, Any]:
    first = predictions[first_name].to_numpy(dtype=float)
    second = predictions[second_name].to_numpy(dtype=float)
    target = truth.to_numpy(dtype=float)
    well_array = wells.to_numpy(dtype=str)
    first_score = _score(predictions[first_name], truth, wells)
    second_score = _score(predictions[second_name], truth, wells)

    pooled_alpha = _analytic_alpha(first, second, target)
    per_well_alpha = {
        well: _analytic_alpha(first, second, target, well_array == well)
        for well in sorted(wells.unique())
    }
    leave_one_well_out_alpha = {
        held_out: _analytic_alpha(first, second, target, well_array != held_out)
        for held_out in sorted(wells.unique())
    }

    grid_rows: list[dict[str, Any]] = []
    for alpha in np.linspace(0.0, 1.0, 101):
        blend = pd.Series(
            (1.0 - alpha) * first + alpha * second,
            index=truth.index,
        )
        score = _score(blend, truth, wells)
        best_endpoint = min(first_score["pooled_rmse"], second_score["pooled_rmse"])
        within_worst_endpoint_by_well = all(
            score["per_well_rmse"][well]
            <= max(
                first_score["per_well_rmse"][well],
                second_score["per_well_rmse"][well],
            )
            + 1e-12
            for well in sorted(wells.unique())
        )
        grid_rows.append(
            {
                "alpha_second": _float(alpha),
                "pooled_rmse": score["pooled_rmse"],
                "strict_endpoint_dominance": bool(
                    score["pooled_rmse"] <= best_endpoint + 1e-12 and within_worst_endpoint_by_well
                ),
            }
        )

    strict_alphas = [row["alpha_second"] for row in grid_rows if row["strict_endpoint_dominance"]]
    nontrivial_strict_alphas = [alpha for alpha in strict_alphas if 1e-12 < alpha < 1.0 - 1e-12]
    return {
        "first": first_name,
        "second": second_name,
        "alpha_definition": "(1-alpha)*first + alpha*second",
        "pooled_optimal_alpha": pooled_alpha,
        "per_well_optimal_alpha": per_well_alpha,
        "leave_one_well_out_optimal_alpha": leave_one_well_out_alpha,
        "endpoint_pooled_rmse": {
            "first": first_score["pooled_rmse"],
            "second": second_score["pooled_rmse"],
        },
        "strict_endpoint_dominance_definition": (
            "pooled RMSE no worse than the better endpoint and every well no worse "
            "than its worse endpoint"
        ),
        "strict_alpha_min": min(strict_alphas) if strict_alphas else None,
        "strict_alpha_max": max(strict_alphas) if strict_alphas else None,
        "has_nontrivial_strict_blend": bool(nontrivial_strict_alphas),
        "grid_step": 0.01,
    }


def build_report() -> dict[str, Any]:
    predictions = {name: _load_prediction(path) for name, path in CANDIDATE_PATHS.items()}
    reference_ids = predictions["safe_v2_final"].index
    for name, prediction in predictions.items():
        if not prediction.index.equals(reference_ids):
            raise ValueError(f"{name}: id order differs from safe_v2_final")

    truth, wells = _load_demo_truth(
        reference_ids,
        REPO_ROOT / "data" / "competition" / "train",
    )
    candidate_scores = {
        name: {
            "risk_class": CANDIDATE_RISK[name],
            **_score(prediction, truth, wells),
        }
        for name, prediction in predictions.items()
    }

    pairwise_names = [
        ("safe_v2_final", "safe_v3_final"),
        ("safe_v2_final", "exact6451_clean_anchor"),
        ("safe_v3_final", "exact6451_clean_anchor"),
        ("exact6451_clean_anchor", "exact6451_raw_final"),
    ]
    pairwise_deltas = {
        f"{first}__to__{second}": _delta_stats(
            predictions[first],
            predictions[second],
            truth,
            wells,
        )
        for first, second in pairwise_names
    }

    stages = {
        "safe_v3_visible_prefix": {
            "evidence_class": "clean_target_free_demo_diagnostic",
            **_delta_stats(
                predictions["safe_v3_base"],
                predictions["safe_v3_final"],
                truth,
                wells,
            ),
        },
        "exact6451_forbidden_prefix_stack": {
            "evidence_class": "confounded_do_not_use_for_candidate_choice",
            "note": (
                "This combines the public notebook's same-well/contact/overlap layers "
                "and cannot isolate visible-prefix calibration."
            ),
            **_delta_stats(
                predictions["exact6451_clean_anchor"],
                predictions["exact6451_pre_modelpkg"],
                truth,
                wells,
            ),
        },
        "exact6451_selected_model_package": {
            "evidence_class": "actual_selected_stage_upstream_confounded",
            "note": "The p95 disagreement guard disabled the package, so selected move is zero.",
            **_delta_stats(
                predictions["exact6451_pre_modelpkg"],
                predictions["exact6451_pre_hedge"],
                truth,
                wells,
            ),
        },
        "exact6451_branch_hedge": {
            "evidence_class": "target_free_stage_but_upstream_confounded",
            **_delta_stats(
                predictions["exact6451_pre_hedge"],
                predictions["exact6451_raw_final"],
                truth,
                wells,
            ),
        },
        "exact6451_u_continuity": {
            "evidence_class": "not_present",
            "note": "No U-continuity stage exists in the exact 6.451 notebook/run artifacts.",
            "moved_rows": 0,
            "actual_contribution": 0.0,
        },
    }

    correction_report_path = (
        RUNS_ROOT / "pf_public_reproduction_v1" / "model_package_correction_report.csv"
    )
    correction_report = pd.read_csv(correction_report_path).iloc[0]
    model_package_guard = {
        "p95_abs_model_package_diff": _float(correction_report["p95_abs_model_package_diff"]),
        "selected_for_submission_csv": bool(correction_report["selected_for_submission_csv"]),
        "disabled_by_diff_guard": bool(correction_report["disabled_by_diff_guard"]),
    }

    blend_pairs = [
        ("safe_v2_final", "safe_v3_final"),
        ("safe_v2_final", "exact6451_clean_anchor"),
        ("safe_v3_final", "exact6451_clean_anchor"),
    ]
    blends = {
        f"{first}__with__{second}": _blend_analysis(
            first,
            second,
            predictions,
            truth,
            wells,
        )
        for first, second in blend_pairs
    }

    report = {
        "scope": {
            "diagnostic_only": True,
            "truth_source": ("same-ID train TVT, read only after frozen run artifacts are loaded"),
            "rows": int(len(reference_ids)),
            "wells": sorted(wells.unique()),
            "prohibited_for_recommendation": [
                "same-ID lookup",
                "contact reconstruction",
                "overlap recovery",
            ],
        },
        "candidate_scores": candidate_scores,
        "pairwise_deltas": pairwise_deltas,
        "correction_stages": stages,
        "model_package_guard": model_package_guard,
        "blend_analysis": blends,
        "go_no_go": {
            "safe_v3_hidden_submission": "no_go_from_demo_evidence",
            "exact6451_raw_final_as_generalization_evidence": "no_go_confounded",
            "exact6451_clean_anchor_as_target_free_candidate": "go_for_independent_validation",
            "model_package_00425_scale6": "no_go_selected_move_was_zero",
            "branch_hedge_cap2": "no_go_from_demo_evidence",
            "u_continuity": "no_evidence_in_exact6451",
            "fixed_interior_blend": "no_go_no_nontrivial_strict_interval",
        },
    }
    return report


def _self_check(report: dict[str, Any]) -> None:
    if report["scope"]["rows"] != 14_151:
        raise AssertionError("expected 14,151 visible demo rows")
    if len(report["scope"]["wells"]) != 3:
        raise AssertionError("expected exactly three visible demo wells")
    model_stage = report["correction_stages"]["exact6451_selected_model_package"]
    if model_stage["moved_rows"] != 0:
        raise AssertionError("exact 6.451 model package was expected to be guard-disabled")
    continuity = report["correction_stages"]["exact6451_u_continuity"]
    if continuity["evidence_class"] != "not_present":
        raise AssertionError("exact 6.451 should not contain U-continuity")
    if any(item["has_nontrivial_strict_blend"] for item in report["blend_analysis"].values()):
        raise AssertionError("unexpected nontrivial strict blend interval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc, diagnostic-only comparison of frozen ROGII candidates."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Assert the expected frozen-artifact invariants.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report()
    if args.self_check:
        _self_check(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate delta report: {args.output}")
    print(
        "Clean pooled RMSE:",
        {
            name: round(details["pooled_rmse"], 6)
            for name, details in report["candidate_scores"].items()
            if details["risk_class"].startswith("clean")
        },
    )
    print("Go/no-go:", report["go_no_go"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
