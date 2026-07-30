"""Evaluate a frozen target-free GR notch transform on all masked train suffixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rogii.gr_notch import notch_strongest_periodic_component  # noqa: E402
from rogii.trajectory import (  # noqa: E402
    HORIZONTAL_INPUT_COLUMNS,
    TYPEWELL_INPUT_COLUMNS,
    TrajectoryConfig,
    predict_trajectory,
)
from rogii.u_continuity_gate import (  # noqa: E402
    SPATIAL_CLUSTERS,
    SPATIAL_N_INIT,
    SPATIAL_RANDOM_STATE,
    deployment_decision,
    paired_well_bootstrap,
    pooled_rmse,
)

FROZEN_MIN_FINITE = 64
FROZEN_NOTCH_RADIUS = 1
METHOD = "fft_notch_final"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen strongest-periodic-component GR notch."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "target_free_eval_773.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "target_free_eval_773_gr_notch.json",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-wells", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value


def _score_one(arguments: tuple[str, str]) -> dict[str, Any]:
    horizontal_name, typewell_name = arguments
    horizontal_path = Path(horizontal_name)
    typewell_path = Path(typewell_name)
    well_id = horizontal_path.name.split("__", maxsplit=1)[0]
    started = time.perf_counter()

    scoring = pd.read_csv(
        horizontal_path,
        usecols=[*HORIZONTAL_INPUT_COLUMNS, "TVT"],
    )
    model_horizontal = scoring.loc[:, HORIZONTAL_INPUT_COLUMNS].copy()
    model_typewell = pd.read_csv(
        typewell_path,
        usecols=list(TYPEWELL_INPUT_COLUMNS),
    )
    truth_full = pd.to_numeric(scoring["TVT"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    original_gr = pd.to_numeric(model_horizontal["GR"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    filtered_gr, notch = notch_strongest_periodic_component(
        original_gr,
        min_finite=FROZEN_MIN_FINITE,
        notch_radius=FROZEN_NOTCH_RADIUS,
    )
    model_horizontal["GR"] = filtered_gr

    result = predict_trajectory(
        model_horizontal,
        model_typewell,
        config=TrajectoryConfig(),
    )
    score_mask = result.evaluation_mask & np.isfinite(truth_full)
    if not np.any(score_mask):
        raise ValueError(f"{well_id}: no finite target rows in the masked suffix")
    if not np.array_equal(score_mask, ~np.isfinite(model_horizontal["TVT_input"])):
        raise ValueError(f"{well_id}: scoring mask differs from official suffix")
    prediction = np.asarray(result.prediction, dtype=float)[score_mask]
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"{well_id}: non-finite prediction")

    xy = scoring.loc[:, ["X", "Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )
    finite_xy = np.all(np.isfinite(xy), axis=1)
    if not np.any(finite_xy):
        raise ValueError(f"{well_id}: no finite X/Y rows")
    return {
        "well_id": well_id,
        "scored_rows": int(score_mask.sum()),
        "notch_rmse": _rmse(truth_full[score_mask], prediction),
        "centroid_x": float(np.mean(xy[finite_xy, 0])),
        "centroid_y": float(np.mean(xy[finite_xy, 1])),
        "runtime_seconds": float(time.perf_counter() - started),
        "notch": notch.to_dict(),
        "predictor_contract": {
            key: result.diagnostics.get(key)
            for key in (
                "target_free",
                "same_id_lookup",
                "train_target_used",
                "formation_columns_used",
                "leaderboard_bias_used",
                "precomputed_predictions_used",
            )
        },
    }


def _metric_summary(rmse: np.ndarray, rows: np.ndarray, well_ids: list[str]) -> dict[str, Any]:
    worst = int(np.argmax(rmse))
    return {
        "pooled_rmse": pooled_rmse(rmse, rows),
        "well_macro_rmse": float(np.mean(rmse)),
        "well_median_rmse": float(np.median(rmse)),
        "well_p90_rmse": float(np.quantile(rmse, 0.90)),
        "well_p95_rmse": float(np.quantile(rmse, 0.95)),
        "well_p99_rmse": float(np.quantile(rmse, 0.99)),
        "worst_well_id": well_ids[worst],
        "worst_well_rmse": float(rmse[worst]),
        "scored_wells": int(rmse.size),
        "scored_rows": int(np.sum(rows)),
    }


def _spatial_deltas(
    baseline_rmse: np.ndarray,
    notch_rmse: np.ndarray,
    rows: np.ndarray,
    centroids: np.ndarray,
) -> list[dict[str, Any]]:
    scaled = StandardScaler().fit_transform(centroids)
    labels = KMeans(
        n_clusters=SPATIAL_CLUSTERS,
        random_state=SPATIAL_RANDOM_STATE,
        n_init=SPATIAL_N_INIT,
    ).fit_predict(scaled)
    result = []
    for cluster in range(SPATIAL_CLUSTERS):
        mask = labels == cluster
        baseline = pooled_rmse(baseline_rmse[mask], rows[mask])
        corrected = pooled_rmse(notch_rmse[mask], rows[mask])
        result.append(
            {
                "cluster": cluster,
                "wells": int(mask.sum()),
                "rows": int(rows[mask].sum()),
                "baseline_pooled_rmse": baseline,
                "notch_pooled_rmse": corrected,
                "delta": corrected - baseline,
            }
        )
    return result


def evaluate(
    data_root: Path,
    baseline_path: Path,
    *,
    jobs: int,
    max_wells: int,
    progress_every: int,
    fail_fast: bool,
) -> dict[str, Any]:
    """Run the frozen notch evaluation and compare with the frozen baseline."""

    if jobs <= 0:
        raise ValueError("--jobs must be positive")
    if max_wells < 0:
        raise ValueError("--max-wells cannot be negative")

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_config = _normalize(asdict(TrajectoryConfig()))
    if baseline_payload.get("config") != expected_config:
        raise ValueError("baseline config does not match the frozen TrajectoryConfig")
    baseline_rows = {
        str(row["well_id"]): row for row in baseline_payload.get("per_well", [])
    }

    train_dir = data_root.expanduser().resolve() / "train"
    horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if max_wells:
        horizontal_paths = horizontal_paths[:max_wells]
    arguments = [
        (str(path), str(train_dir / f"{path.name.split('__', maxsplit=1)[0]}__typewell.csv"))
        for path in horizontal_paths
    ]
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    def accept(position: int, item: dict[str, Any]) -> None:
        completed.append(item)
        if progress_every > 0 and (
            position % progress_every == 0 or position == len(arguments)
        ):
            print(
                f"[{position:>3}/{len(arguments)}] successful={len(completed)} "
                f"failures={len(failures)} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if jobs == 1:
        for position, argument in enumerate(arguments, start=1):
            try:
                accept(position, _score_one(argument))
            except Exception as error:
                failures.append(
                    {
                        "well_id": Path(argument[0]).name.split("__", maxsplit=1)[0],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if fail_fast:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            pending = {
                executor.submit(_score_one, argument): argument for argument in arguments
            }
            for position, future in enumerate(as_completed(pending), start=1):
                argument = pending[future]
                try:
                    accept(position, future.result())
                except Exception as error:
                    failures.append(
                        {
                            "well_id": Path(argument[0]).name.split("__", maxsplit=1)[0],
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    if fail_fast:
                        raise

    completed.sort(key=lambda row: row["well_id"])
    if not completed:
        raise RuntimeError("no wells were scored")
    if failures:
        raise RuntimeError(f"{len(failures)} wells failed; refusing a partial decision")

    well_ids = [row["well_id"] for row in completed]
    if any(well_id not in baseline_rows for well_id in well_ids):
        raise ValueError("notch evaluation includes a well absent from the baseline")
    rows = np.asarray([row["scored_rows"] for row in completed], dtype=float)
    baseline_count = np.asarray(
        [baseline_rows[well_id]["scored_rows"] for well_id in well_ids],
        dtype=float,
    )
    if not np.array_equal(rows, baseline_count):
        raise ValueError("notch and baseline scored-row counts differ")

    baseline_rmse = np.asarray(
        [baseline_rows[well_id]["rmse"]["final"] for well_id in well_ids],
        dtype=float,
    )
    notch_rmse = np.asarray([row["notch_rmse"] for row in completed], dtype=float)
    centroids = np.asarray(
        [[row["centroid_x"], row["centroid_y"]] for row in completed],
        dtype=float,
    )
    spatial = _spatial_deltas(baseline_rmse, notch_rmse, rows, centroids)
    baseline_metrics = _metric_summary(baseline_rmse, rows, well_ids)
    notch_metrics = _metric_summary(notch_rmse, rows, well_ids)
    pooled_delta = notch_metrics["pooled_rmse"] - baseline_metrics["pooled_rmse"]
    macro_delta = notch_metrics["well_macro_rmse"] - baseline_metrics["well_macro_rmse"]
    bootstrap = paired_well_bootstrap(baseline_rmse, notch_rmse, rows)
    gate = deployment_decision(
        pooled_delta=pooled_delta,
        macro_delta=macro_delta,
        bootstrap_ci_upper=float(bootstrap["ci95_upper"]),
        cluster_deltas=[float(row["delta"]) for row in spatial],
    )
    tolerance = 1e-12

    return {
        "contract": {
            "source": "public full-stack dormant GR FFT notch, isolated transcription",
            "minimum_finite_rows": FROZEN_MIN_FINITE,
            "notch_radius": FROZEN_NOTCH_RADIUS,
            "removed_signal": "strongest non-DC rFFT bin and immediate neighbors",
            "predictor_config": expected_config,
            "uses_truth_for_inference": False,
            "uses_well_id_as_feature": False,
            "same_id_lookup": False,
            "formation_feature": False,
            "leaderboard_bias": False,
            "probe_or_canary": False,
            "threshold_tuning_after_result": False,
        },
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": _sha256(baseline_path),
            "metrics": baseline_metrics,
        },
        "notch": {
            "metrics": notch_metrics,
            "applied_wells": int(sum(row["notch"]["applied"] for row in completed)),
        },
        "comparison": {
            "pooled_rmse_delta": pooled_delta,
            "well_macro_rmse_delta": macro_delta,
            "wins": int(np.count_nonzero(notch_rmse < baseline_rmse - tolerance)),
            "losses": int(np.count_nonzero(notch_rmse > baseline_rmse + tolerance)),
            "ties": int(np.count_nonzero(np.abs(notch_rmse - baseline_rmse) <= tolerance)),
            "paired_well_bootstrap": bootstrap,
            "spatial_clusters": spatial,
        },
        "decision": gate,
        "selection": {
            "available_wells": int(len(list(train_dir.glob("*__horizontal_well.csv")))),
            "selected_wells": int(len(arguments)),
            "successful_wells": int(len(completed)),
            "failed_wells": int(len(failures)),
            "jobs": int(jobs),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "per_well": completed,
        "failures": failures,
    }


def _print_report(report: dict[str, Any]) -> None:
    baseline = report["baseline"]["metrics"]
    notch = report["notch"]["metrics"]
    comparison = report["comparison"]
    print()
    print("method          pooled    macro      p90      p95      p99    worst")
    print("-" * 69)
    for label, values in (("baseline", baseline), ("fft_notch", notch)):
        print(
            f"{label:<14}{values['pooled_rmse']:>8.4f}"
            f"{values['well_macro_rmse']:>9.4f}"
            f"{values['well_p90_rmse']:>9.4f}"
            f"{values['well_p95_rmse']:>9.4f}"
            f"{values['well_p99_rmse']:>9.4f}"
            f"{values['worst_well_rmse']:>9.4f}"
        )
    print(
        f"delta={comparison['pooled_rmse_delta']:+.6f} "
        f"wins/losses/ties={comparison['wins']}/{comparison['losses']}/{comparison['ties']} "
        f"decision={report['decision']['decision']}"
    )


def main() -> int:
    args = _parser().parse_args()
    report = evaluate(
        args.data_root,
        args.baseline.expanduser().resolve(),
        jobs=args.jobs,
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
