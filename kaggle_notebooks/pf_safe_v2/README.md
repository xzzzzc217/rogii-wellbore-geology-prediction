# ROGII PF Safe v2

This private notebook is a safety-focused derivative of Roman Mironov's public
[PF-Config Strategy | High-Scoring 6.491](https://www.kaggle.com/code/helloromiro/pf-config-strategy-high-scoring-6-491),
captured on 2026-07-23. It was created from our attributed v1 reproduction.
The public `6.491` score is not claimed for this modified profile.

## Purpose

Safe v2 keeps the public PF/beam, ridge-artifact, learned-trajectory, projection,
blend, and final schema-audit structure while removing the highest-risk default
paths:

- no same-ID train lookup or contact reconstruction;
- test formation/ANCC priors always exclude the query well;
- no guarded contact override;
- no overlap probe;
- no visible-prefix calibration pass;
- no model-package correction;
- no final PF midpoint hedge;
- selector and learned PF stages use 64 seeds instead of 128.

The expected runtime for about 200 hidden wells is approximately 3.5–5.5 hours,
but this is an engineering estimate. A full Kaggle code run must confirm it
before this version is treated as submission-ready.

## Default safety controls

```python
SUBMISSION_PROFILE = "safe_v2"
ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_VISIBLE_PREFIX_CALIBRATION = False
RUN_MODEL_PACKAGE_CORRECTION = False
RUN_PF_BRANCH_HEDGE = False
SP45_SELECTOR_N_SEEDS = 64
CFG.PF_SEEDS = 64
```

The final audit still requires exactly two columns (`id,tvt`), exact sample row
count and ID order, and finite predictions.

## Mounted inputs

Only inputs used by the default execution path remain in `kernel-metadata.json`:

- `phongnguyn23021656/koolbox-offline`
- `fleongg/rogii-claude-models-pub`
- `ravaghi/wellbore-geology-prediction-artifacts`
- the ROGII competition data

The model-package, static prediction, TabICL, and unrelated artifact datasets
from v1 were removed.

## Residual risks

- The ridge artifact is about 7.39 GB and its trainer files, plus the fleongg
  boosters, are external pickle/joblib assets. Loading time, memory, runtime
  compatibility, and upstream provenance remain risks.
- Some original Numba single-PF kernels still use global random state. Reducing
  seeds does not make every upstream feature bitwise deterministic.
- Disabling visible-prefix calibration and optional correction layers changes
  predictive quality; safe v2 must earn its own leaderboard score.
- The notebook retains disabled helper code for auditability. Static checks must
  verify that its safety gates remain false before every push.

This directory is a local artifact only. It must not be pushed to Kaggle until
JSON validation, static safety checks, and a deliberate review are complete.
