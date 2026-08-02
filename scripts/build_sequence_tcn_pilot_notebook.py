"""Build the private Kaggle GPU notebook for the sequence TCN pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "kaggle_notebooks" / "sequence_tcn_pilot_v1"
KERNEL_ID = "heralces/rogii-sequence-tcn-pilot-v2"
DATASET_ID = "heralces/rogii-private-sequence-pilot-cache-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_source = (PROJECT_ROOT / "scripts" / "sequence_tcn_pilot.py").read_text(
        encoding="utf-8"
    )

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "rogii_sequence_tcn_pilot": {
                "version": 2,
                "writes_submission": False,
                "protocol": "spatial_cluster_7_seed_20260727",
                "validation_fold": 0,
                "primary_overlay_scale": 0.20,
                "training_steps": 4000,
                "dataset": DATASET_ID,
            },
        },
        "cells": [
            {
                "id": "sequence-tcn-overview",
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    "# ROGII private sequence TCN pilot v1\n\n"
                    "This GPU-only experiment trains a small temporal convolutional residual "
                    "model against the reconstructed competitive Ridge/PF OOF trajectory. "
                    "Validation holds out one complete seven-cluster spatial fold and removes "
                    "all matching typewell fingerprints from training. It writes a report and "
                    "checkpoint only; it cannot create a competition submission."
                ),
            },
            {
                "id": "sequence-tcn-run",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": (
                    "from pathlib import Path\n"
                    "import json\n"
                    "import subprocess\n"
                    "import sys\n\n"
                    "candidates = [\n"
                    "    Path('/kaggle/input/rogii-private-sequence-pilot-cache-v1'),\n"
                    "    Path('/kaggle/input/datasets/heralces/"
                    "rogii-private-sequence-pilot-cache-v1'),\n"
                    "]\n"
                    "cache = next((p for p in candidates if "
                    "(p / 'sequence_cache_metadata.json').exists()), None)\n"
                    "if cache is None:\n"
                    "    raise RuntimeError(f'private sequence cache not found: {candidates}')\n"
                    f"source = {training_source!r}\n"
                    "script = Path('/kaggle/working/sequence_tcn_pilot.py')\n"
                    "script.write_text(source, encoding='utf-8')\n"
                    "command = [\n"
                    "    sys.executable, str(script),\n"
                    "    '--cache-dir', str(cache),\n"
                    "    '--output', '/kaggle/working/sequence_tcn_pilot_report.json',\n"
                    "    '--checkpoint', '/kaggle/working/sequence_tcn_pilot.pt',\n"
                    "    '--device', 'cuda',\n"
                    "]\n"
                    "print('running:', ' '.join(command), flush=True)\n"
                    "subprocess.run(command, check=True)\n"
                    "report_path = Path(command[command.index('--output') + 1])\n"
                    "report = json.loads(report_path.read_text())\n"
                    "print(json.dumps({\n"
                    "    'decision': report['decision'],\n"
                    "    'runtime': report['runtime'],\n"
                    "    'training': report['training'],\n"
                    "    'validation': report['validation'],\n"
                    "}, indent=2, sort_keys=True))\n"
                ),
            },
        ],
    }
    notebook_path = output_dir / "rogii-sequence-tcn-pilot-v1.ipynb"
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    metadata = {
        "id": KERNEL_ID,
        "title": "ROGII Sequence TCN Pilot v2",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
