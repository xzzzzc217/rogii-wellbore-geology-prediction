"""Apply the pre-frozen statistical deployment gate to a completed full report."""

from __future__ import annotations

import argparse
import json
import sys
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

from rogii.u_continuity_gate import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SPATIAL_CLUSTERS,
    SPATIAL_N_INIT,
    SPATIAL_RANDOM_STATE,
    deployment_decision,
    paired_well_bootstrap,
    pooled_rmse,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the frozen U-continuity run.")
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "artifacts" / "target_free_eval_773_ucontinuity.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _spatial_coordinates(
    per_well: list[dict[str, Any]],
    train_dir: Path,
) -> np.ndarray:
    coordinates = np.empty((len(per_well), 2), dtype=float)
    for index, row in enumerate(per_well):
        frame = pd.read_csv(
            train_dir / f"{row['well_id']}__horizontal_well.csv",
            usecols=["X", "Y"],
        )
        coordinates[index, 0] = float(pd.to_numeric(frame["X"], errors="coerce").median())
        coordinates[index, 1] = float(pd.to_numeric(frame["Y"], errors="coerce").median())
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("non-finite spatial well coordinates")
    return coordinates


def _subgroup(
    mask: np.ndarray,
    baseline: np.ndarray,
    corrected: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "wells": 0,
            "rows": 0,
            "final_pooled_rmse": None,
            "final_u_continuity_pooled_rmse": None,
            "pooled_rmse_delta": None,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    baseline_pooled = pooled_rmse(baseline[mask], rows[mask])
    corrected_pooled = pooled_rmse(corrected[mask], rows[mask])
    return {
        "wells": int(np.count_nonzero(mask)),
        "rows": int(np.sum(rows[mask])),
        "final_pooled_rmse": baseline_pooled,
        "final_u_continuity_pooled_rmse": corrected_pooled,
        "pooled_rmse_delta": float(corrected_pooled - baseline_pooled),
        "wins": int(np.count_nonzero(corrected[mask] < baseline[mask] - 1e-12)),
        "losses": int(np.count_nonzero(corrected[mask] > baseline[mask] + 1e-12)),
        "ties": int(np.count_nonzero(np.abs(corrected[mask] - baseline[mask]) <= 1e-12)),
    }


def analyze(report: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Add the frozen bootstrap, spatial holdouts, subgroup, and gate decision."""

    frozen = report["frozen_u_continuity"]
    if frozen["cap"] != 8.0 or frozen["tau"] != 240.0:
        raise ValueError("expected frozen cap=8.0 and tau=240.0")
    if frozen["pf_seeds"] != list(range(8)):
        raise ValueError("expected frozen PF seeds 0..7")
    if report["selection"]["successful_wells"] != 773 or report["failures"]:
        raise ValueError("deployment gate requires a failure-free 773-well report")

    per_well = report["per_well"]
    baseline = np.asarray([row["rmse"]["final"] for row in per_well], dtype=float)
    corrected = np.asarray(
        [row["rmse"]["final_u_continuity"] for row in per_well],
        dtype=float,
    )
    rows = np.asarray([row["scored_rows"] for row in per_well], dtype=float)
    gaps = np.asarray(
        [row["u_continuity"]["boundary_gap_u_before"] for row in per_well],
        dtype=float,
    )
    mean_abs_move = np.asarray(
        [row["u_continuity"]["mean_abs_move"] for row in per_well],
        dtype=float,
    )

    baseline_metrics = report["metrics"]["final"]
    corrected_metrics = report["metrics"]["final_u_continuity"]
    pooled_delta = float(corrected_metrics["pooled_rmse"] - baseline_metrics["pooled_rmse"])
    macro_delta = float(corrected_metrics["well_macro_rmse"] - baseline_metrics["well_macro_rmse"])
    bootstrap = paired_well_bootstrap(
        baseline,
        corrected,
        rows,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    coordinates = _spatial_coordinates(per_well, data_root / "train")
    standardized = StandardScaler().fit_transform(coordinates)
    labels = KMeans(
        n_clusters=SPATIAL_CLUSTERS,
        random_state=SPATIAL_RANDOM_STATE,
        n_init=SPATIAL_N_INIT,
    ).fit_predict(standardized)
    holdouts: list[dict[str, Any]] = []
    for cluster in range(SPATIAL_CLUSTERS):
        mask = labels == cluster
        summary = _subgroup(mask, baseline, corrected, rows)
        summary["cluster"] = int(cluster)
        summary["median_x"] = float(np.median(coordinates[mask, 0]))
        summary["median_y"] = float(np.median(coordinates[mask, 1]))
        holdouts.append(summary)

    cluster_deltas = [row["pooled_rmse_delta"] for row in holdouts]
    decision = deployment_decision(
        pooled_delta=pooled_delta,
        macro_delta=macro_delta,
        bootstrap_ci_upper=float(bootstrap["ci95_upper"]),
        cluster_deltas=cluster_deltas,
    )
    cap_saturated = np.abs(gaps) >= 8.0
    report["deployment_gate"] = {
        "frozen_before_full_result": True,
        "policy": {
            "pooled_rmse_delta_limit": -0.020,
            "well_macro_delta_must_be_negative": True,
            "paired_well_bootstrap_ci95_upper_must_be_negative": True,
            "spatial_required_nonpositive_clusters": 4,
            "spatial_max_worse_delta": 0.020,
            "all_conditions_required": True,
        },
        "paired_well_bootstrap": bootstrap,
        "spatial_method": {
            "coordinates": "per-well median X/Y over the horizontal trajectory",
            "scaler": "StandardScaler",
            "algorithm": "KMeans",
            "clusters": SPATIAL_CLUSTERS,
            "random_state": SPATIAL_RANDOM_STATE,
            "n_init": SPATIAL_N_INIT,
        },
        "spatial_cluster_holdouts": holdouts,
        "cap_saturated_subgroup": _subgroup(
            cap_saturated,
            baseline,
            corrected,
            rows,
        ),
        "row_weighted_movement": {
            "mean_abs_move": float(np.sum(rows * mean_abs_move) / np.sum(rows)),
            "scored_rows": int(np.sum(rows)),
        },
        "metric_deltas": {
            "pooled_rmse": pooled_delta,
            "well_macro_rmse": macro_delta,
            "well_p90_rmse": float(
                corrected_metrics["well_p90_rmse"] - baseline_metrics["well_p90_rmse"]
            ),
            "well_p95_rmse": float(
                corrected_metrics["well_p95_rmse"] - baseline_metrics["well_p95_rmse"]
            ),
            "well_p99_rmse": float(
                corrected_metrics["well_p99_rmse"] - baseline_metrics["well_p99_rmse"]
            ),
            "worst_well_rmse": float(
                corrected_metrics["worst_well_rmse"] - baseline_metrics["worst_well_rmse"]
            ),
        },
        **decision,
    }
    return report


def main() -> int:
    args = _parser().parse_args()
    report_path = args.report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analyzed = analyze(report, args.data_root.expanduser().resolve())
    output = args.output.expanduser().resolve() if args.output is not None else report_path
    output.write_text(
        json.dumps(analyzed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = analyzed["deployment_gate"]
    print(
        f"decision={gate['decision']} "
        f"pooled_delta={gate['metric_deltas']['pooled_rmse']:+.6f} "
        f"bootstrap_ci95_upper={gate['paired_well_bootstrap']['ci95_upper']:+.6f} "
        f"spatial_nonpositive={gate['spatial_nonpositive_clusters']}/5 "
        f"spatial_max={gate['spatial_maximum_delta']:+.6f}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
