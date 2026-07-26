# Spatial-invariant residual model: frozen triple-protocol OOF

## Decision

**NO-GO.** Removing absolute X/Y/Z features improved both spatial aggregate
scores, but the one frozen CatBoost model still failed the predeclared
regional tail-risk gate in both spatial protocols. No Kaggle submission is
authorized by this result.

This v2 hypothesis was informed by the v1 spatial failure. Its second spatial
seed is therefore an additional robustness check, not an untouched
confirmation set.

## Frozen change

Only these six features were removed:

- `x_scaled`, `y_scaled`, `z_scaled`
- `anchor_x_scaled`, `anchor_y_scaled`, `anchor_z_scaled`

Relative geometry (`dx/dy/dz`, derivatives, and distance), visible-prefix
signals, typewell features, model parameters, residual target, clipping, and
leakage controls were unchanged.

The model had to pass all three protocols:

1. five typewell-fingerprint grouped folds;
2. five spatial clusters using seed 20260725;
3. six spatial clusters using seed 20260726.

## Results

Hold-last TVT was the selected comparator at 15.909853 pooled RMSE.

| Protocol | Model RMSE | Pooled delta | CI95 upper | Fold result | Gate |
|---|---:|---:|---:|---|---|
| Typewell fingerprint, 5-fold | 14.475568 | -1.434285 | -1.075658 | 5/5 improve; worst -1.224476 | PASS |
| Spatial clusters, 5-fold | 15.589873 | -0.319980 | +0.143313 | 4/5 improve; worst +0.839080 | FAIL |
| Spatial clusters, 6-fold | 15.409073 | -0.500780 | -0.041847 | 5/6 improve; worst +1.677417 | FAIL |

The five-cluster protocol also worsened well-level p95 RMSE by +0.691574.
The six-cluster protocol passed its aggregate, bootstrap, fold-count, p95,
and worst-well checks, but failed the hard maximum-fold-deterioration limit
of +0.20. That gate remains unchanged.

## Evidence

- JSON: `artifacts/residual_gbdt_invariant_oof.json`
- Final artifact SHA-256:
  `D77FC17A267336BAA4A1971689CA184A1F7C3F9C04DD338386006AA3DF965232`
- Runtime: 471.7 seconds
- Standard error log: empty
- Evaluator: `scripts/residual_gbdt_invariant_evaluate.py`
- Feature builder: `src/rogii/residual_gbdt.py`
