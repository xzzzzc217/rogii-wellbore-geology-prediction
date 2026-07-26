"""Evaluate the frozen Guard640 U-continuity layer on all masked train suffixes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
    TrajectoryConfig,
    predict_trajectory,
)
from rogii.u_continuity import apply_u_continuity  # noqa: E402

FROZEN_CAP = 8.0
FROZEN_TAU = 240.0
FROZEN_PF_SEEDS = tuple(range(8))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen cap=8/tau=240 target-free U-continuity."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "target_free_eval_773_ucontinuity.json",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-wells", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _metric_summary(per_well: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rmse = np.asarray([row["rmse"][method] for row in per_well], dtype=float)
    counts = np.asarray([row["scored_rows"] for row in per_well], dtype=float)
    worst = int(np.argmax(rmse))
    return {
        "pooled_rmse": float(np.sqrt(np.sum(counts * rmse**2) / np.sum(counts))),
        "well_macro_rmse": float(np.mean(rmse)),
        "well_median_rmse": float(np.median(rmse)),
        "well_p90_rmse": float(np.quantile(rmse, 0.90)),
        "well_p95_rmse": float(np.quantile(rmse, 0.95)),
        "well_p99_rmse": float(np.quantile(rmse, 0.99)),
        "worst_well_id": per_well[worst]["well_id"],
        "worst_well_rmse": float(rmse[worst]),
        "scored_wells": int(len(per_well)),
        "scored_rows": int(np.sum(counts)),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "signed_mean": float(np.mean(values)),
        "mean_abs": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
    }


def _boundary_summary(per_well: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [row["u_continuity"] for row in per_well]
    before = np.asarray([row["boundary_gap_u_before"] for row in diagnostics], dtype=float)
    after = np.asarray([row["boundary_gap_u_after"] for row in diagnostics], dtype=float)
    first_move = np.asarray([row["first_move"] for row in diagnostics], dtype=float)
    mean_move = np.asarray([row["mean_abs_move"] for row in diagnostics], dtype=float)
    max_move = np.asarray([row["max_abs_move"] for row in diagnostics], dtype=float)
    return {
        "boundary_gap_u_before": _distribution(before),
        "boundary_gap_u_after": _distribution(after),
        "first_move": _distribution(first_move),
        "mean_abs_move": _distribution(mean_move),
        "max_abs_move": _distribution(max_move),
        "cap_saturated_wells": int(np.count_nonzero(np.abs(before) >= FROZEN_CAP)),
        "cap_saturated_fraction": float(np.mean(np.abs(before) >= FROZEN_CAP)),
    }


def evaluate(
    data_root: Path,
    *,
    max_wells: int,
    progress_every: int,
    fail_fast: bool,
) -> dict[str, Any]:
    """Run the frozen target-free continuity evaluation."""

    if max_wells < 0:
        raise ValueError("--max-wells cannot be negative")
    paths = sorted((data_root.expanduser().resolve() / "train").glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError("no training horizontal wells found")
    selected = paths if max_wells == 0 else paths[:max_wells]

    config = TrajectoryConfig(pf_seeds=FROZEN_PF_SEEDS)
    per_well: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for position, horizontal_path in enumerate(selected, start=1):
        well_id = horizontal_path.name.split("__", maxsplit=1)[0]
        typewell_path = horizontal_path.parent / f"{well_id}__typewell.csv"
        well_started = time.perf_counter()
        try:
            scoring = pd.read_csv(
                horizontal_path,
                usecols=[*HORIZONTAL_INPUT_COLUMNS, "TVT"],
            )
            model_horizontal = scoring.loc[:, HORIZONTAL_INPUT_COLUMNS].copy()
            model_typewell = pd.read_csv(
                typewell_path,
                usecols=list(TYPEWELL_INPUT_COLUMNS),
            )

            # TVT is isolated before model inference and continuity.  Neither
            # callable accepts a truth array.
            truth_full = pd.to_numeric(scoring["TVT"], errors="coerce").to_numpy(
                dtype=float,
                na_value=np.nan,
            )
            result = predict_trajectory(model_horizontal, model_typewell, config=config)
            corrected, diagnostics = apply_u_continuity(
                result.prediction,
                pd.to_numeric(model_horizontal["MD"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(model_horizontal["Z"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(model_horizontal["TVT_input"], errors="coerce").to_numpy(dtype=float),
                cap=FROZEN_CAP,
                tau=FROZEN_TAU,
            )

            score_mask = result.evaluation_mask & np.isfinite(truth_full)
            if not np.any(score_mask):
                raise ValueError("well has no finite target rows in its hidden suffix")
            if not np.array_equal(score_mask, ~np.isfinite(model_horizontal["TVT_input"])):
                raise ValueError("scoring mask differs from the official TVT_input suffix")

            truth = truth_full[score_mask]
            baseline = np.asarray(result.prediction, dtype=float)[score_mask]
            continuity = corrected[score_mask]
            if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(continuity)):
                raise ValueError("non-finite scored prediction")

            per_well.append(
                {
                    "well_id": well_id,
                    "scored_rows": int(score_mask.sum()),
                    "runtime_seconds": float(time.perf_counter() - well_started),
                    "rmse": {
                        "final": _rmse(truth, baseline),
                        "final_u_continuity": _rmse(truth, continuity),
                    },
                    "u_continuity": diagnostics.to_dict(),
                }
            )
        except Exception as error:
            failures.append(
                {
                    "well_id": well_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if fail_fast:
                raise

        if progress_every > 0 and (position % progress_every == 0 or position == len(selected)):
            print(
                f"[{position:>3}/{len(selected)}] successful={len(per_well)} "
                f"failures={len(failures)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if not per_well:
        raise RuntimeError("no wells were scored")

    final_rmse = np.asarray([row["rmse"]["final"] for row in per_well], dtype=float)
    corrected_rmse = np.asarray(
        [row["rmse"]["final_u_continuity"] for row in per_well], dtype=float
    )
    tolerance = 1e-12
    metrics = {
        "final": _metric_summary(per_well, "final"),
        "final_u_continuity": _metric_summary(per_well, "final_u_continuity"),
    }
    return {
        "evaluation_contract": {
            "split": "train",
            "mask": "official TVT_input missing suffix",
            "target_role": "TVT is held outside model and correction; scoring only",
            "model_horizontal_columns": list(HORIZONTAL_INPUT_COLUMNS),
            "model_typewell_columns": list(TYPEWELL_INPUT_COLUMNS),
            "same_id_lookup": False,
            "train_target_feature": False,
            "formation_feature": False,
            "leaderboard_bias": False,
            "public_precomputed_prediction": False,
        },
        "frozen_u_continuity": {
            "source": "Guard640 v4 public composition",
            "formula": (
                "-clip(first_hidden_U-last_visible_U,-cap,cap)"
                "*exp(-(hidden_MD-last_visible_MD)/tau)"
            ),
            "cap": FROZEN_CAP,
            "tau": FROZEN_TAU,
            "pf_seeds": list(FROZEN_PF_SEEDS),
            "threshold_tuning_on_this_run": False,
        },
        "selection": {
            "available_train_wells": int(len(paths)),
            "requested_max_wells": int(max_wells),
            "selected_wells": int(len(selected)),
            "successful_wells": int(len(per_well)),
            "failed_wells": int(len(failures)),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "metrics": metrics,
        "comparison": {
            "pooled_rmse_delta": float(
                metrics["final_u_continuity"]["pooled_rmse"] - metrics["final"]["pooled_rmse"]
            ),
            "well_macro_rmse_delta": float(
                metrics["final_u_continuity"]["well_macro_rmse"]
                - metrics["final"]["well_macro_rmse"]
            ),
            "continuity_wins": int(np.count_nonzero(corrected_rmse < final_rmse - tolerance)),
            "continuity_losses": int(np.count_nonzero(corrected_rmse > final_rmse + tolerance)),
            "ties": int(np.count_nonzero(np.abs(corrected_rmse - final_rmse) <= tolerance)),
        },
        "boundary_diagnostics": _boundary_summary(per_well),
        "per_well": per_well,
        "failures": failures,
    }


def _print_report(report: dict[str, Any]) -> None:
    print()
    print("method                    pooled    macro      p90      p95      p99    worst")
    print("-" * 77)
    for method, values in report["metrics"].items():
        print(
            f"{method:<25}"
            f"{values['pooled_rmse']:>8.4f}"
            f"{values['well_macro_rmse']:>9.4f}"
            f"{values['well_p90_rmse']:>9.4f}"
            f"{values['well_p95_rmse']:>9.4f}"
            f"{values['well_p99_rmse']:>9.4f}"
            f"{values['worst_well_rmse']:>9.4f}"
        )
    comparison = report["comparison"]
    print(
        f"wins={comparison['continuity_wins']} "
        f"losses={comparison['continuity_losses']} ties={comparison['ties']} "
        f"runtime={report['runtime_seconds']:.1f}s "
        f"failures={report['selection']['failed_wells']}"
    )


def main() -> int:
    args = _parser().parse_args()
    report = evaluate(
        args.data_root,
        max_wells=args.max_wells,
        progress_every=args.progress_every,
        fail_fast=args.fail_fast,
    )
    _print_report(report)
    output = args.json_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
