# Residual GBDT: frozen dual-protocol OOF

## Decision

**NO-GO for the current feature set.** The one frozen CatBoost residual model
passed typewell-fingerprint grouped OOF but failed spatial-cluster OOF. Both
protocols were required to pass.

The local LightGBM binary raised a native Dataset-construction access
violation before any full OOF score was observed. The same frozen features,
folds, target, and deployment gate were therefore executed with a fixed
CatBoost CPU backend.

## Leakage contract

- Target: hidden `TVT - last_visible_TVT`.
- Features: only mounted horizontal `MD/X/Y/Z/GR/TVT_input` and typewell
  `TVT/GR`.
- Hidden `TVT` was isolated after feature construction and used only as an
  offline label.
- Well ID, formation columns, same-ID target lookup, and leaderboard
  constants were absent.
- At most 96 evenly spaced training rows were sampled per well.
- Spatial folds also removed every validation typewell fingerprint from their
  training wells.

## Results

The best frozen simple comparator was hold-last TVT at 15.909853 pooled RMSE.

| Protocol | CatBoost RMSE | Delta | Bootstrap CI95 | Fold deltas | Gate |
|---|---:|---:|---|---|---|
| Typewell-fingerprint 5-fold | 14.207941 | -1.701911 | [-2.088280, -1.314516] | all five negative | PASS |
| Spatial-cluster 5-fold | 15.858111 | -0.051741 | [-0.566017, +0.473390] | 4/5 negative; worst +1.407903 | FAIL |

Spatial OOF additionally worsened well-level p95 RMSE by +0.641942. The
current model therefore lacks geographic-shift robustness even though its
within-distribution signal is strong.

The result motivates a separate future hypothesis that removes absolute
coordinate/depth features and retains only relative geometry. It does not
authorize weakening this experiment's gate or submitting this model.

## Evidence

- JSON: `artifacts/residual_gbdt_oof.json`
- SHA-256:
  `EFF3ABFBE65AFB5AAFF24A206F25F0BA7416491797787A70A348168DA616060B`
- Runtime: 313.2 seconds
- Feature module: `src/rogii/residual_gbdt.py`
- Evaluator: `scripts/residual_gbdt_evaluate.py`
- Tests: `tests/test_residual_gbdt.py`
