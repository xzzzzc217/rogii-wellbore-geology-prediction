# ROGII Guard640 v4b

Local-only sibling of `pf_guard640_v4`. It has not been pushed, run, or
submitted.

## Single behavioral difference

```python
# v4
MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0

# v4b
MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9
```

The v4b value matches the downloaded public 6.40 source. All PF, visible-prefix,
model-package, branch-hedge, U-continuity, runtime-budget, dynamic-sample, and
leakage-control code otherwise remains byte-for-byte identical to v4. Kernel
identity strings and audit filenames differ only to keep future run artifacts
separate.

## Why the 25-ft switch is not a safe fit for the 6.40 gate

The completed exact 6.451 reproduction measured a model-package/base
disagreement p95 of `26.700659 ft`. Its `25 ft` global switch therefore disabled
the entire layer. That was nearly immaterial for the 6.451 gate
(`max_weight=0.00425`, `scale=6`): the inactive candidate's p95 final move was
only `0.012384 ft`.

Guard640 instead uses `max_weight=0.49`, `scale=192`. At the same `26.700659 ft`
disagreement, its row-wise gate would move the trajectory by about `12.84 ft`.
Thus transplanting the 6.451 global switch into Guard640 can turn a principal
composition layer completely off because of a marginal threshold crossing.

The row-wise gate is already self-bounding:

```text
move(d) = 0.49 * d / (1 + (d / 192)^2)
max move = 0.49 * 192 / 2 = 47.04 ft
```

Large disagreements do not create unbounded moves; after `d=192 ft`, the move
shrinks. The final dynamic contract, finite-value checks, stage reports, runtime
fallback, branch cap, and U-continuity cap remain unchanged.

## Leakage and contract invariants

The validator rejects any same-ID/contact lookup, exact-overlap train-TVT
calibration, overlap probe, global leaderboard bias, fixed test ID, fixed row
count, expected prediction hash, or canary. Test feature builders continue to
self-exclude the query well, and the last writer still derives row count, IDs,
and order from the mounted `sample_submission.csv`.

## Decision gate

`v4b` is a **go for an ordinary visible-data Kaggle dry-run**, not an automatic
go for hidden submission. A hidden submission should require a completed run,
runtime below the budget, a valid independent submission audit, and confirmation
that the model-package candidate was selected rather than skipped by the runtime
guard.

Run the local checks with:

```powershell
python scripts/build_pf_guard640_v4b.py
python kaggle_notebooks/pf_guard640_v4b/validate_notebook.py
```
