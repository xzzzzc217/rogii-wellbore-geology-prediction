"""Evaluate a preregistered globally augmented Functional Ridge signature.

The passing Functional Ridge samples every target-free feature at five fixed
relative-depth locations.  This candidate appends three whole-well statistics
per feature: mean, standard deviation, and least-squares slope versus relative
row position.  The response model and all overlay parameters remain frozen.

All statistics are target-free.  This script is OOF-only and never writes a
Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_functional_ridge as plain
import numpy as np
import pandas as pd

DEFAULT_OUTPUT = plain.ROOT / "artifacts" / "functional_ridge_augmented_oof"
SUMMARY_STATISTICS = ("mean", "standard_deviation", "relative_position_slope")
DOMINANCE_MARGIN = 0.003
PLAIN_RIDGE_REFERENCE = {
    "spatial_cluster_7_seed_20260727": -0.0276029,
    "typewell_fingerprint": -0.0187671,
}
POINT_SIGNATURE = plain._signature


def _augmented_signature(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("well features must contain at least two rows")
    values = np.where(np.isfinite(values), values, np.nan)
    point_signature = POINT_SIGNATURE(values)
    means = np.nanmean(values, axis=0)
    standard_deviations = np.nanstd(values, axis=0)
    position = np.linspace(-0.5, 0.5, len(values), dtype=float)[:, None]
    finite = np.isfinite(values)
    centered = values - means
    slope_numerator = np.nansum(position * centered, axis=0)
    slope_denominator = np.sum((position**2) * finite, axis=0)
    slopes = np.divide(
        slope_numerator,
        slope_denominator,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=slope_denominator > 0.0,
    )
    return np.concatenate(
        [point_signature, means, standard_deviations, slopes]
    ).astype(np.float32)


def _apply_deployment_gate(report: dict[str, Any]) -> None:
    for protocol, result in report["protocols"].items():
        result["plain_ridge_reference_delta"] = PLAIN_RIDGE_REFERENCE[protocol]
        result["delta_vs_plain_ridge"] = (
            result["pooled_delta"] - PLAIN_RIDGE_REFERENCE[protocol]
        )
        result["gate_checks"]["beats_plain_ridge_by_margin"] = bool(
            result["pooled_delta"]
            <= PLAIN_RIDGE_REFERENCE[protocol] - DOMINANCE_MARGIN
        )
        result["gate_passed"] = bool(all(result["gate_checks"].values()))
    report["decision"] = (
        "PASS"
        if all(result["gate_passed"] for result in report["protocols"].values())
        else "FAIL"
    )


def evaluate(cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    original_signature = plain._signature
    try:
        plain._signature = _augmented_signature
        report, detail = plain.evaluate(cache_dir)
    finally:
        plain._signature = original_signature

    report["candidate"] = "target-free globally augmented Functional Ridge"
    report["contract"].update(
        {
            "base_signature_width": int(report["contract"]["feature_count"] * 5),
            "summary_statistics": list(SUMMARY_STATISTICS),
            "summary_statistics_target_free": True,
            "dominance_margin_vs_plain_ridge": DOMINANCE_MARGIN,
        }
    )
    _apply_deployment_gate(report)
    return report, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=plain.DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, detail = evaluate(args.cache_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "functional_ridge_augmented_oof_report.json"
    detail_path = args.output_dir / "functional_ridge_augmented_oof_per_well.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(detail_path, index=False)
    print(
        json.dumps(
            {"decision": report["decision"], "protocols": report["protocols"]},
            indent=2,
        )
    )
    print(f"wrote {report_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
