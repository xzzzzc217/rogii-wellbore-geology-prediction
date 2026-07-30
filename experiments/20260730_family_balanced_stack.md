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

Pending.
