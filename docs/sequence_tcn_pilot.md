# GPU sequence TCN pilot

## Decision scope

This experiment asks one narrow question: can a small temporal convolutional
network provide a stable residual correction to the reconstructed competitive
Ridge/PF OOF trajectory?

It is not a competition-submission notebook. A passing result authorizes only
a full multi-fold study; it does not authorize packaging or Kaggle submission.

## Data contract

The private cache is built from the 773 official training wells and their
3,783,989 masked suffix rows. For each suffix row it stores:

- the competitive Ridge/PF OOF prediction;
- the official training truth, used only as the residual label;
- 51 spatial-invariant features produced by the existing residual feature
  builder;
- well boundaries and validation assignments in metadata.

The model feature matrix excludes absolute X/Y/Z and anchor X/Y/Z, well ID,
formation labels, same-ID truth lookup, and all hidden competition-test data.
Well IDs remain only in metadata so complete wells can be held out. The float16
feature conversion clips at 60,000; cache v1 recorded zero clipped values.

The generated cache is approximately 397 MB and lives under the ignored
`artifacts/sequence_pilot_cache_v1/` directory. Its Kaggle copy is a private
Dataset owned by the competition account. No cache data or model checkpoint is
committed to Git.

## Validation split

The cheap first gate uses fold 0 of the frozen
`spatial_cluster_7_seed_20260727` protocol. Every validation well is excluded
from training. In addition, any training well whose Typewell fingerprint
matches a validation fingerprint is removed.

Feature centering and scale are estimated from training wells only. Training
windows never cross a well boundary. This prevents random-row leakage and
keeps the spatial extrapolation question explicit.

## Model

The TCN is intentionally small:

- 51 input channels;
- 64 hidden channels;
- six residual Conv1D blocks with dilations `1,2,4,8,16,32`;
- kernel width 5, GroupNorm, GELU, and 0.08 dropout;
- 256-row random windows, batch size 32, 4,000 optimizer steps;
- AdamW at `8e-4`, weight decay `1e-4`;
- Smooth L1 loss on competitive-base residuals clipped to ±50 ft;
- output bounded to ±30 ft with `tanh`.

The preregistered primary candidate adds exactly 0.20 times the TCN output to
the competitive OOF prediction. Scales 0.10 and 0.30 are reported only as
diagnostics and cannot replace the primary result for the v1 gate.

## Gate

The primary 0.20 overlay must satisfy all four checks:

| Check | Limit |
|---|---:|
| Pooled RMSE delta versus competitive base | at most -0.020 |
| Well-level p95 RMSE delta | at most +0.100 |
| Worst-well RMSE delta | at most +0.500 |
| Fraction of validation wells not worsened | at least 0.500 |

A local two-step CPU smoke test verified only the code path and artifact
contract. Its metric is deliberately not treated as model evidence. Kaggle
version 1 was assigned a Tesla P100 (compute capability 6.0), while its
PyTorch 2.10 image supported only compute capability 7.0 and newer. It failed
at the first Conv1D after approximately 25 seconds with `no kernel image`; no
training metric was produced. Version 2, published as private Notebook
`heralces/rogii-sequence-tcn-pilot-v2`, fixes the requested machine shape to
`NvidiaTeslaT4` without changing the model, split, scale, steps, or gate. The
real result must report CUDA usage, the frozen fold, the fixed primary scale,
and the full gate.

## Reproduction

Build the ignored local cache:

```powershell
uv run python scripts/build_sequence_pilot_cache.py
```

Build the private Kaggle Notebook source:

```powershell
uv run python scripts/build_sequence_tcn_pilot_notebook.py
```

The training entry point can also be smoke-tested on a local PyTorch runtime:

```powershell
python scripts/sequence_tcn_pilot.py `
  --cache-dir artifacts/sequence_pilot_cache_v1 `
  --steps 2 --batch-size 2 --window-rows 64 `
  --device cpu --smoke-wells 12
```

The production Notebook writes only:

- `sequence_tcn_pilot_report.json`;
- `sequence_tcn_pilot.pt`.

It contains no code path that creates `submission.csv`.
