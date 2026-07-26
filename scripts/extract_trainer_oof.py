"""Extract one published koolbox Trainer OOF vector in an isolated process.

Loading all five serialized Trainer objects in one Python process retains a
large allocator heap because each artifact also contains fitted estimators.
Running this helper once per artifact lets the operating system reclaim that
memory before the competitive OOF evaluation starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-rows", type=int, default=3_783_989)
    return parser


def main() -> int:
    args = _parser().parse_args()
    trainer_path = args.trainer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    trainer = joblib.load(trainer_path)
    prediction = np.asarray(trainer.oof_preds, dtype=np.float32)
    if prediction.shape != (args.expected_rows,):
        raise ValueError(
            f"{trainer_path}: OOF shape {prediction.shape}, expected {(args.expected_rows,)}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError(f"{trainer_path}: OOF vector contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prediction, allow_pickle=False)
    metadata = {
        "trainer": str(trainer_path),
        "rows": int(len(prediction)),
        "dtype": str(prediction.dtype),
        "overall_score": float(trainer.overall_score),
        "fold_scores": [float(value) for value in trainer.fold_scores],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"extracted {trainer_path.name}: rows={len(prediction)} "
        f"score={metadata['overall_score']:.6f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
