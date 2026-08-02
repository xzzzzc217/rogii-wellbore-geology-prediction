# Family-balanced public OOF stack — preregistration

Date frozen: 2026-07-30, before any target-scored family-balanced result was
inspected.

## Hypothesis

The published competitive trajectory gives five correlated model predictions
to a positive Ridge: three LightGBM variants and two CatBoost variants. Even
with a supervised stacker, presenting every variant independently may let the
larger LightGBM family receive an unstable effective weight. The single
candidate in this audit first averages models inside each family, then gives
Ridge exactly two features:

1. the arithmetic mean of the three LightGBM grouped-OOF vectors;
2. the arithmetic mean of the two CatBoost grouped-OOF vectors.

The Ridge parameters, `0.91 / 0.09` Ridge/PF blend, and measured-depth ramp
with `tau=85` remain unchanged. There is no alpha, family weight, PF weight,
ramp, or per-well tuning after the result.

## Evidence boundary

All five source vectors are published well-grouped OOF predictions. The
second-level base and candidate are cross-fitted with the same published
five-fold `GroupKFold` by well. Exact typewell-fingerprint and seven-cluster
X/Y partitions are used to describe geographic and typewell robustness.

This is not a fully nested refit of the five expensive base models inside
those two robustness partitions. It is therefore a secondary stack audit,
not standalone deployment evidence. A PASS can authorize a later integration
or stricter reproduction audit, but cannot directly authorize a Kaggle
submission.

Hidden TVT is used only as the stack fitting target and offline score. The
candidate receives no well identifier, fixed visible-test well, fixed row
count/hash, leaderboard score, probe, canary, or same-ID target lookup.

## Frozen gate

The candidate is compared with the exactly reconstructed five-column
Ridge/PF OOF track on all 3,783,989 hidden-suffix rows from all 773 official
training wells. Every condition must pass:

- pooled RMSE delta at most `-0.020`;
- macro-well RMSE delta below zero;
- paired-well bootstrap 95% upper bound below zero;
- at least 4 of 5 typewell partitions non-worse;
- at least 6 of 7 spatial partitions non-worse;
- no typewell or spatial partition worse than `+0.050`;
- p95-well RMSE delta at most `+0.100`;
- worst-well RMSE delta at most `+0.500`.

`NO-GO` permanently stops this family-collapse configuration. No family
weight, Ridge alpha, or downstream blend will be retuned from its result.

Reproduction command:

```powershell
uv run python scripts/evaluate_family_balanced_stack.py
```

## Frozen result

Decision: **NO-GO**.

- Runtime: 51.2 seconds on CPU.
- Rows / wells: 3,783,989 / 773.
- Published five-column Ridge/PF reconstruction: 10.370056 pooled RMSE.
- Family-balanced two-column Ridge/PF: 10.399843 pooled RMSE.
- Pooled delta: **+0.029786**.
- Macro-well delta: **+0.001060**.
- Per-well outcomes: 386 wins and 387 losses.
- Paired-well bootstrap 95% interval:
  `[-0.007816, +0.067396]`.
- Typewell partitions non-worse: 2 / 5.
- Spatial partitions non-worse: 3 / 7.
- Maximum partition delta: **+0.103686**.
- p95-well delta: **+0.148644**.
- Worst-well delta: **+0.433701**.

Only the worst-well guard passed. The pooled, macro, bootstrap, typewell,
spatial, maximum-partition, and p95 checks failed. Equalizing the number of
inputs per model family discards useful within-family specialization; the
positive Ridge was already suppressing weak redundant columns when needed.

No family weight, Ridge alpha, PF weight, or ramp will be tuned from this
result. No Notebook or Kaggle submission is authorized.

Artifact:
`artifacts/family_balanced_stack_oof.json`

SHA-256:
`fa0fc662273d2a81fbdd6df79e4bdf11d7204e5d5abe62968adc7fe9aec0311e`
