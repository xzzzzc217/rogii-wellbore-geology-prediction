"""Run one preregistered GPU TCN residual pilot against the competitive OOF base.

This is deliberately an experiment, not a submission builder.  The primary
candidate is a fixed 0.20 overlay of a bounded TCN residual prediction.  A
single strict spatial fold is used as the cheap first gate; full multi-fold
training is authorized only if this pilot passes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PRIMARY_SCALE = 0.20
DIAGNOSTIC_SCALES = (0.10, 0.20, 0.30)
TARGET_CLIP = 50.0
OUTPUT_BOUND = 30.0
WINDOW_ROWS = 256
BATCH_SIZE = 32
TRAINING_STEPS = 4_000
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 20_260_727
PROTOCOL = "spatial_cluster_7_seed_20260727"
VALIDATION_FOLD = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("sequence_tcn_pilot_report.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sequence_tcn_pilot.pt"))
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--window-rows", type=int, default=WINDOW_ROWS)
    parser.add_argument(
        "--well-sampling",
        choices=("length", "uniform"),
        default="length",
        help=(
            "Choose training wells proportional to suffix length (original pilot) "
            "or uniformly by well."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--smoke-wells",
        type=int,
        default=0,
        help="Limit both partitions for a local code-path smoke test; zero uses the full fold.",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.network(values))


class SequenceResidualTCN(nn.Module):
    def __init__(self, input_features: int, channels: int = 64) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels, dilation) for dilation in (1, 2, 4, 8, 16, 32))
        )
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.blocks(self.input(values))).squeeze(1)
        return OUTPUT_BOUND * torch.tanh(raw / OUTPUT_BOUND)


def _load_cache(cache_dir: Path) -> dict[str, Any]:
    metadata = json.loads(
        (cache_dir / "sequence_cache_metadata.json").read_text(encoding="utf-8")
    )
    features = np.load(cache_dir / "features.float16.npy", mmap_mode="r", allow_pickle=False)
    base = np.load(
        cache_dir / "competitive_base.float32.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    truth = np.load(cache_dir / "truth.float32.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(
        cache_dir / "well_offsets.int64.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected = int(metadata["rows"])
    if features.shape != (expected, int(metadata["feature_count"])):
        raise ValueError("feature cache shape does not match metadata")
    if base.shape != (expected,) or truth.shape != (expected,):
        raise ValueError("base/truth cache shape does not match metadata")
    if offsets.shape != (int(metadata["wells"]) + 1,) or int(offsets[-1]) != expected:
        raise ValueError("well offsets do not cover the cache")
    return {
        "metadata": metadata,
        "features": features,
        "base": base,
        "truth": truth,
        "offsets": offsets,
    }


def _partition(metadata: dict[str, Any], smoke_wells: int) -> tuple[np.ndarray, np.ndarray]:
    rows = metadata["per_well"]
    validation = np.asarray(
        [index for index, row in enumerate(rows) if row["folds"][PROTOCOL] == VALIDATION_FOLD],
        dtype=np.int64,
    )
    validation_fingerprints = {rows[index]["typewell_fingerprint"] for index in validation}
    training = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["folds"][PROTOCOL] != VALIDATION_FOLD
            and row["typewell_fingerprint"] not in validation_fingerprints
        ],
        dtype=np.int64,
    )
    if smoke_wells:
        training = training[:smoke_wells]
        validation = validation[: max(1, smoke_wells // 4)]
    if not len(training) or not len(validation):
        raise ValueError("training/validation partition is empty")
    return training, validation


def _normalization(
    features: np.ndarray,
    offsets: np.ndarray,
    training_wells: np.ndarray,
    *,
    maximum_rows: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    lengths = offsets[training_wells + 1] - offsets[training_wells]
    total = int(lengths.sum())
    stride = max(1, math.ceil(total / maximum_rows))
    samples: list[np.ndarray] = []
    cursor = 0
    for well_index, length in zip(training_wells, lengths, strict=True):
        start = int(offsets[well_index])
        local_offset = (-cursor) % stride
        samples.append(
            np.asarray(
                features[start + local_offset : start + int(length) : stride],
                dtype=np.float32,
            )
        )
        cursor += int(length)
    sample = np.concatenate(samples, axis=0)
    center = np.nanmedian(sample, axis=0)
    q25 = np.nanquantile(sample, 0.25, axis=0)
    q75 = np.nanquantile(sample, 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0).astype(np.float32)
    return center, scale


def _window_batch(
    rng: np.random.Generator,
    features: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    offsets: np.ndarray,
    wells: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
    window_rows: int,
    well_sampling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = offsets[wells + 1] - offsets[wells]
    if well_sampling == "length":
        probability = lengths.astype(np.float64)
        probability /= probability.sum()
    elif well_sampling == "uniform":
        probability = None
    else:
        raise ValueError(f"unsupported well sampling: {well_sampling}")
    selected = rng.choice(wells, size=batch_size, replace=True, p=probability)
    batch_x = np.zeros((batch_size, len(center), window_rows), dtype=np.float32)
    batch_y = np.zeros((batch_size, window_rows), dtype=np.float32)
    batch_mask = np.zeros((batch_size, window_rows), dtype=np.float32)
    for batch_index, well_index in enumerate(selected):
        well_start = int(offsets[well_index])
        well_stop = int(offsets[well_index + 1])
        length = well_stop - well_start
        local_start = int(rng.integers(0, max(1, length - window_rows + 1)))
        stop = min(well_stop, well_start + local_start + window_rows)
        rows = stop - (well_start + local_start)
        values = np.asarray(
            features[well_start + local_start : stop],
            dtype=np.float32,
        )
        values = np.clip((values - center) / scale, -8.0, 8.0)
        values[~np.isfinite(values)] = 0.0
        residual = np.asarray(
            truth[well_start + local_start : stop] - base[well_start + local_start : stop],
            dtype=np.float32,
        )
        batch_x[batch_index, :, :rows] = values.T
        batch_y[batch_index, :rows] = np.clip(residual, -TARGET_CLIP, TARGET_CLIP)
        batch_mask[batch_index, :rows] = 1.0
    return batch_x, batch_y, batch_mask


@torch.inference_mode()
def _predict_well(
    model: nn.Module,
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    values = np.clip((values - center) / scale, -8.0, 8.0)
    values[~np.isfinite(values)] = 0.0
    tensor = torch.from_numpy(values.T[None]).to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = model(tensor)
    return prediction.squeeze(0).float().cpu().numpy()


def _evaluate(
    model: nn.Module,
    cache: dict[str, Any],
    validation_wells: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    features = cache["features"]
    base_values = cache["base"]
    truth_values = cache["truth"]
    offsets = cache["offsets"]
    metadata = cache["metadata"]
    per_well: list[dict[str, Any]] = []
    all_truth: list[np.ndarray] = []
    all_base: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    model.eval()
    for well_index in validation_wells:
        start = int(offsets[well_index])
        stop = int(offsets[well_index + 1])
        truth = np.asarray(truth_values[start:stop], dtype=np.float32)
        base = np.asarray(base_values[start:stop], dtype=np.float32)
        raw = _predict_well(model, features[start:stop], center, scale, device)
        all_truth.append(truth)
        all_base.append(base)
        all_raw.append(raw)
        per_well.append(
            {
                "well": metadata["per_well"][well_index]["well"],
                "rows": int(len(truth)),
                "base_rmse": _rmse(truth, base),
            }
        )
    truth = np.concatenate(all_truth)
    base = np.concatenate(all_base)
    raw = np.concatenate(all_raw)
    scale_results: dict[str, Any] = {}
    for candidate_scale in DIAGNOSTIC_SCALES:
        prediction = base + candidate_scale * raw
        well_cursor = 0
        well_rmse: list[float] = []
        for row in per_well:
            stop = well_cursor + int(row["rows"])
            well_rmse.append(_rmse(truth[well_cursor:stop], prediction[well_cursor:stop]))
            well_cursor = stop
        base_well_rmse = np.asarray([row["base_rmse"] for row in per_well], dtype=float)
        candidate_well_rmse = np.asarray(well_rmse, dtype=float)
        scale_results[f"{candidate_scale:.2f}"] = {
            "pooled_rmse": _rmse(truth, prediction),
            "pooled_delta": _rmse(truth, prediction) - _rmse(truth, base),
            "macro_well_rmse": float(np.mean(candidate_well_rmse)),
            "p95_well_delta": float(
                np.quantile(candidate_well_rmse, 0.95) - np.quantile(base_well_rmse, 0.95)
            ),
            "worst_well_delta": float(
                np.max(candidate_well_rmse) - np.max(base_well_rmse)
            ),
            "nonpositive_well_fraction": float(
                np.mean(candidate_well_rmse <= base_well_rmse)
            ),
        }
    primary = scale_results[f"{PRIMARY_SCALE:.2f}"]
    checks = {
        "pooled_delta_at_most_minus_0_02": bool(primary["pooled_delta"] <= -0.02),
        "p95_not_worse_than_plus_0_10": bool(primary["p95_well_delta"] <= 0.10),
        "worst_not_worse_than_plus_0_50": bool(primary["worst_well_delta"] <= 0.50),
        "at_least_half_wells_nonpositive": bool(primary["nonpositive_well_fraction"] >= 0.50),
    }
    return {
        "rows": int(len(truth)),
        "wells": int(len(validation_wells)),
        "base_pooled_rmse": _rmse(truth, base),
        "raw_residual": {
            "mean": float(np.mean(raw)),
            "std": float(np.std(raw)),
            "p95_abs": float(np.quantile(np.abs(raw), 0.95)),
            "maximum_abs": float(np.max(np.abs(raw))),
        },
        "scale_results": scale_results,
        "primary_scale": PRIMARY_SCALE,
        "gate": {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.window_rows < 32:
        raise ValueError("training controls are invalid")
    started = time.perf_counter()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"device={device} torch={torch.__version__}", flush=True)

    cache = _load_cache(args.cache_dir.expanduser().resolve())
    training_wells, validation_wells = _partition(cache["metadata"], args.smoke_wells)
    center, feature_scale = _normalization(
        cache["features"],
        cache["offsets"],
        training_wells,
        maximum_rows=50_000 if args.smoke_wells else 500_000,
    )
    model = SequenceResidualTCN(len(center)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.SmoothL1Loss(reduction="none", beta=2.0)
    rng = np.random.default_rng(RANDOM_SEED)
    losses: list[float] = []
    model.train()
    for step in range(args.steps):
        batch_x, batch_y, batch_mask = _window_batch(
            rng,
            cache["features"],
            cache["base"],
            cache["truth"],
            cache["offsets"],
            training_wells,
            center,
            feature_scale,
            batch_size=args.batch_size,
            window_rows=args.window_rows,
            well_sampling=args.well_sampling,
        )
        values = torch.from_numpy(batch_x).to(device)
        target = torch.from_numpy(batch_y).to(device)
        mask = torch.from_numpy(batch_mask).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(values)
            loss = (loss_function(prediction, target) * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, min(250, args.steps // 4)) == 0:
            print(
                f"[train {step + 1:4d}/{args.steps}] "
                f"loss={np.mean(losses[-100:]):.5f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    evaluation = _evaluate(
        model,
        cache,
        validation_wells,
        center,
        feature_scale,
        device,
    )
    report = {
        "contract": {
            "writes_submission": False,
            "protocol": PROTOCOL,
            "validation_fold": VALIDATION_FOLD,
            "validation_typewell_fingerprints_excluded_from_training": True,
            "well_id_encoded_as_feature": False,
            "absolute_position_features_removed": True,
            "training_well_sampling": args.well_sampling,
            "primary_scale_frozen_before_training": PRIMARY_SCALE,
            "diagnostic_scales": list(DIAGNOSTIC_SCALES),
            "primary_gate": {
                "pooled_delta_limit": -0.02,
                "p95_well_delta_limit": 0.10,
                "worst_well_delta_limit": 0.50,
                "minimum_nonpositive_well_fraction": 0.50,
            },
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "seconds": float(time.perf_counter() - started),
        },
        "training": {
            "wells": int(len(training_wells)),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "window_rows": int(args.window_rows),
            "well_sampling": args.well_sampling,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "target_clip": TARGET_CLIP,
            "output_bound": OUTPUT_BOUND,
            "final_loss_mean_100": float(np.mean(losses[-100:])),
            "smoke_wells": int(args.smoke_wells),
        },
        "validation": evaluation,
        "decision": {
            "decision": evaluation["gate"]["decision"],
            "full_multifold_authorized": evaluation["gate"]["decision"] == "PASS"
            and not args.smoke_wells,
            "kaggle_submission_authorized": False,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "center": center,
            "scale": feature_scale,
            "feature_names": cache["metadata"]["features"],
            "contract": report["contract"],
        },
        checkpoint,
    )
    primary = evaluation["scale_results"][f"{PRIMARY_SCALE:.2f}"]
    print(
        f"decision={evaluation['gate']['decision']} "
        f"base={evaluation['base_pooled_rmse']:.6f} "
        f"primary={primary['pooled_rmse']:.6f} "
        f"delta={primary['pooled_delta']:+.6f}",
        flush=True,
    )
    print(f"wrote {output} and {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
