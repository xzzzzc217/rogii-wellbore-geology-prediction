# ROGII Guard640 v4

This directory contains an independent, private, unpushed Kaggle candidate
derived from Fleong Guo's public
[`ROGII Codex Public 6.40 Guard`](https://www.kaggle.com/code/fleongg/rogii-codex-public-6-40-guard),
downloaded on 2026-07-24. The public title score `6.40` is attribution and
prioritization evidence only; it is not claimed for this modified notebook.

The Kaggle kernel ID is `heralces/rogii-guard640-v4`.

## Retained model composition

The candidate preserves the public guard's target-free, hidden-compatible
layers:

- balanced visible-prefix calibration with final multiplier `0.35`;
- model-package disagreement gate with max weight `0.49` and scale `192 ft`;
- generalized PF seed-branch midpoint hedge with a `3 ft` cap;
- boundary continuity in `U = TVT + Z`, capped at `8 ft` with `240 MD` decay.

The main selector PF and learned-trajectory PF use 96 seeds. Visible-prefix
calibration uses 12 seeds, final reconstruction uses 24 seeds, and both use 250
particles. These are deliberate runtime reductions from the public
128 / 24 / 48 / 350 composition.

## Immutable leakage controls

The control cell and final audit fail closed unless:

```python
SUBMISSION_PROFILE = "guard640_v4"
ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_EXACT_MATCH_RECOVERY = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_PUBLIC_OVERLAP_CALIBRATION = False
APPLY_GLOBAL_LB_BIAS = False

RUN_VISIBLE_PREFIX_CALIBRATION = True
VISIBLE_PREFIX_ALPHA_MULTIPLIER = 0.35
RUN_MODEL_PACKAGE_CORRECTION = True
RUN_PF_BRANCH_HEDGE = True
RUN_U_CONTINUITY = True
```

Both learned feature builders pass the query well ID into the formation and
dense imputers, so even the three visible demonstration wells exclude their
own training copies. The public notebook's 5.5% exact-overlap train-TVT cell is
removed, not merely bypassed. Its same-ID physical/contact routes are doubly
gated and cannot execute.

The notebook contains no fixed test-well ID, fixed sample row count, expected
prediction SHA, canary, leaderboard probe, or global score-derived TVT shift.
SHA-256 is used only after inference to audit the dynamically generated file.

## Dynamic final contract

The last writer reads the mounted competition `sample_submission.csv` at
runtime. It derives the required row count, ID set, and order from that file,
rejects missing or duplicate IDs and non-finite values, rewrites
`submission.csv` in sample order, then re-reads and hashes the result.

## External assets

Only the default execution inputs remain mounted:

- `phongnguyn23021656/koolbox-offline`
- `fleongg/rogii-claude-models-pub`
- `ravaghi/wellbore-geology-prediction-artifacts`
- `pilkwang/rogii-model-package`
- the competition data

Automatic model-package discovery is disabled and the package is required when
the runtime guard permits that stage. A 25-ft p95 disagreement guard keeps the
audited visible-prefix submission if the package diverges excessively.

## Runtime estimate

The 6.451/6.40 public family took about 13.3 minutes on the three visible demo
wells. Historical stage timings were approximately:

| Stage | Three-well time |
|---|---:|
| artifact load and base features | 281 s |
| 128-seed selector | 184 s |
| 128-seed learned branch | 66 s |
| 24/48/350 visible prefix | 119 s |
| model package | 43 s |

Scaling to 96 seeds and 12/24/250 visible-prefix settings gives an engineering
estimate of **5.5–7.2 hours** for about 200 hidden wells. This is not a
benchmark. The notebook uses a 7.5-hour hard budget with a final 30-minute
reserve. Visible-prefix processing retains the audited base for unprocessed
wells if the guard fires; model-package inference is skipped unless its
reserved allowance remains before that stage starts.

The generalized hedge and U-continuity are inexpensive. Read-only plotting,
fixed-ID demo, atlas, and visual integrity cells were removed.

## Validation

Run:

```powershell
python kaggle_notebooks/pf_guard640_v4/validate_notebook.py
```

The validator compiles every code cell, executes the isolated control cell,
checks the exact profile values and metadata, verifies query-well exclusion and
dynamic final audit code, rejects same-ID/public-overlap/fixed-contract/LB
patterns, and confirms every cell is unexecuted.

This candidate has not been pushed, run, or submitted.
