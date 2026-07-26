"""Evaluate target-free trajectory baselines on official masked train suffixes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score hold-last, deterministic U-state PF, constrained beam/Viterbi, "
            "and their target-free blend on train TVT_input suffixes."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "competition",
        help="Competition root containing train/ and test/.",
    )
    parser.add_argument(
        "--max-wells",
        type=int,
        default=50,
        help="Deterministic representative sample size; 0 evaluates every train well.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260723,
        help="Seed used only to select wells, never for model randomness.",
    )
    parser.add_argument(
        "--selection-offset",
        type=int,
        default=0,
        help="Offset into the seeded well permutation, useful for resumable chunks.",
    )
    parser.add_argument(
        "--pf-particles",
        type=int,
        default=None,
        help="Override the default particle count.",
    )
    parser.add_argument(
        "--pf-seeds",
        type=int,
        default=None,
        help="Override the number of deterministic seeds, using seeds 0..N-1.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the complete metrics and per-well diagnostics JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress after this many completed wells; 0 disables progress.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first malformed or failed well.",
    )
    return parser


def _select_wells(
    horizontal_paths: list[Path],
    *,
    max_wells: int,
    seed: int,
    offset: int,
) -> list[Path]:
    if max_wells < 0:
        raise ValueError("--max-wells cannot be negative")
    if offset < 0 or offset >= len(horizontal_paths):
        raise ValueError("--selection-offset is outside the available well range")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(horizontal_paths))
    if max_wells == 0:
        selected = permutation[offset:]
    else:
        selected = permutation[offset : offset + max_wells]
    if selected.size == 0:
        raise ValueError("the requested selection chunk is empty")
    return [horizontal_paths[index] for index in sorted(selected)]


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _json_number(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_number(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_number(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _aggregate(
    predictions: dict[str, list[np.ndarray]],
    truths: list[np.ndarray],
    per_well: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    truth = np.concatenate(truths)
    for method, chunks in predictions.items():
        prediction = np.concatenate(chunks)
        well_rows = [
            row
            for row in per_well
            if method in row.get("rmse", {}) and row["rmse"][method] is not None
        ]
        well_rmse = np.asarray(
            [row["rmse"][method] for row in well_rows],
            dtype=float,
        )
        worst_index = int(np.argmax(well_rmse))
        result[method] = {
            "pooled_rmse": _rmse(truth, prediction),
            "well_macro_rmse": float(np.mean(well_rmse)),
            "well_median_rmse": float(np.median(well_rmse)),
            "well_p90_rmse": float(np.quantile(well_rmse, 0.90)),
            "well_p95_rmse": float(np.quantile(well_rmse, 0.95)),
            "well_p99_rmse": float(np.quantile(well_rmse, 0.99)),
            "worst_well_id": well_rows[worst_index]["well_id"],
            "worst_well_rmse": float(well_rmse[worst_index]),
            "scored_wells": int(len(well_rows)),
            "scored_rows": int(len(truth)),
        }
    return result


def evaluate(
    data_root: Path,
    *,
    config: TrajectoryConfig,
    max_wells: int,
    selection_seed: int,
    selection_offset: int,
    progress_every: int,
    fail_fast: bool,
) -> dict[str, Any]:
    """Run a deterministic, pooled suffix evaluation."""

    train_dir = data_root.expanduser().resolve() / "train"
    horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if not horizontal_paths:
        raise FileNotFoundError(f"no train horizontal wells found under {train_dir}")
    selected = _select_wells(
        horizontal_paths,
        max_wells=max_wells,
        seed=selection_seed,
        offset=selection_offset,
    )

    truths: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {}
    per_well: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for position, horizontal_path in enumerate(selected, start=1):
        well_id = horizontal_path.name.split("__", maxsplit=1)[0]
        typewell_path = train_dir / f"{well_id}__typewell.csv"
        well_started = time.perf_counter()
        try:
            # TVT is loaded into a separate scoring array and is never passed
            # through the predictor's explicit input allow-list.
            horizontal_scoring = pd.read_csv(
                horizontal_path,
                usecols=[*HORIZONTAL_INPUT_COLUMNS, "TVT"],
            )
            horizontal_model = horizontal_scoring.loc[:, HORIZONTAL_INPUT_COLUMNS].copy()
            truth_full = pd.to_numeric(
                horizontal_scoring["TVT"],
                errors="coerce",
            ).to_numpy(dtype=float, na_value=np.nan)
            typewell_model = pd.read_csv(
                typewell_path,
                usecols=list(TYPEWELL_INPUT_COLUMNS),
            )
            result = predict_trajectory(
                horizontal_model,
                typewell_model,
                config=config,
            )
            mask = result.evaluation_mask & np.isfinite(truth_full)
            if not np.any(mask):
                raise ValueError("well has no finite target rows in its masked suffix")
            truth = truth_full[mask]
            components = dict(result.components)
            components["final"] = result.prediction

            well_predictions: dict[str, np.ndarray] = {}
            well_rmse: dict[str, float] = {}
            for method, full_prediction in components.items():
                scored_prediction = np.asarray(full_prediction, dtype=float)[mask]
                if not np.all(np.isfinite(scored_prediction)):
                    raise ValueError(f"{method} produced non-finite predictions")
                well_predictions[method] = scored_prediction
                well_rmse[method] = _rmse(truth, scored_prediction)

            if predictions and set(well_predictions) != set(predictions):
                missing = sorted(set(predictions) - set(well_predictions))
                extra = sorted(set(well_predictions) - set(predictions))
                raise ValueError(f"inconsistent component set; missing={missing}, extra={extra}")
            truths.append(truth)
            for method, scored_prediction in well_predictions.items():
                predictions.setdefault(method, []).append(scored_prediction)
            per_well.append(
                {
                    "well_id": well_id,
                    "scored_rows": int(len(truth)),
                    "runtime_seconds": float(time.perf_counter() - well_started),
                    "rmse": well_rmse,
                    "diagnostics": result.diagnostics,
                }
            )
        except Exception as error:
            failure = {
                "well_id": well_id,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            if fail_fast:
                raise

        if progress_every > 0 and (position % progress_every == 0 or position == len(selected)):
            elapsed = time.perf_counter() - started
            print(
                f"[{position:>3}/{len(selected)}] "
                f"successful={len(per_well)} failures={len(failures)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if not truths:
        raise RuntimeError("no wells were scored")
    if failures:
        # Pooled component arrays must remain aligned. A failed well contributes
        # to neither truth nor predictions, and the failure is explicit below.
        print(f"warning: {len(failures)} wells failed and were excluded", file=sys.stderr)

    metrics = _aggregate(predictions, truths, per_well)
    return {
        "evaluation_contract": {
            "split": "train",
            "mask": "official TVT_input missing suffix",
            "target_role": "TVT is held outside the model and used only for scoring",
            "model_horizontal_columns": list(HORIZONTAL_INPUT_COLUMNS),
            "model_typewell_columns": list(TYPEWELL_INPUT_COLUMNS),
            "same_id_lookup": False,
            "train_target_feature": False,
            "formation_feature": False,
            "leaderboard_bias": False,
            "public_precomputed_prediction": False,
        },
        "selection": {
            "available_train_wells": int(len(horizontal_paths)),
            "requested_max_wells": int(max_wells),
            "selection_seed": int(selection_seed),
            "selection_offset": int(selection_offset),
            "selected_wells": [path.name.split("__", maxsplit=1)[0] for path in selected],
            "successful_wells": int(len(per_well)),
            "failed_wells": int(len(failures)),
        },
        "config": _json_number(config.__dict__),
        "runtime_seconds": float(time.perf_counter() - started),
        "metrics": metrics,
        "per_well": per_well,
        "failures": failures,
    }


def _print_metrics(report: dict[str, Any]) -> None:
    print()
    print("method                 pooled    macro      p90      p95      p99    worst  worst_well")
    print("-" * 88)
    for method, values in report["metrics"].items():
        print(
            f"{method:<21}"
            f"{values['pooled_rmse']:>8.4f}"
            f"{values['well_macro_rmse']:>9.4f}"
            f"{values['well_p90_rmse']:>9.4f}"
            f"{values['well_p95_rmse']:>9.4f}"
            f"{values['well_p99_rmse']:>9.4f}"
            f"{values['worst_well_rmse']:>9.4f}"
            f"  {values['worst_well_id']}"
        )
    print()
    print(
        f"runtime={report['runtime_seconds']:.1f}s "
        f"successful={report['selection']['successful_wells']} "
        f"failed={report['selection']['failed_wells']}"
    )
    print(
        "input audit: horizontal=MD/X/Y/Z/GR/TVT_input, typewell=TVT/GR; "
        "no same-ID lookup, formation feature, train-target feature, LB bias, "
        "or public precomputed prediction"
    )


def main() -> int:
    args = _parser().parse_args()
    config = TrajectoryConfig()
    updates: dict[str, Any] = {}
    if args.pf_particles is not None:
        updates["pf_particles"] = args.pf_particles
    if args.pf_seeds is not None:
        if args.pf_seeds <= 0:
            raise ValueError("--pf-seeds must be positive")
        updates["pf_seeds"] = tuple(range(args.pf_seeds))
    if updates:
        config = replace(config, **updates)

    report = evaluate(
        args.data_root,
        config=config,
        max_wells=args.max_wells,
        selection_seed=args.selection_seed,
        selection_offset=args.selection_offset,
        progress_every=args.progress_every,
        fail_fast=args.fail_fast,
    )
    _print_metrics(report)
    if args.json_output is not None:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_json_number(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
