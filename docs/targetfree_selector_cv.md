# Target-free adaptive selector: frozen OOF evaluation

## Decision

**NO-GO.** The deployment decision is controlled only by the predeclared `core + ridge_delta` selector. The ExtraTrees model and hidden-GR feature expansion are sensitivity checks, not a model search.

The GO gate was frozen before evaluation: the same primary selector must improve pooled RMSE by at least 0.10 and improve at least 4/5 folds under both shuffled-well and spatial-cluster holdout.

## Leakage contract

- Features are an explicit allow-list of X/Y well geometry and target-free calibration, PF, beam, masking-horizon, and predicted-trajectory diagnostics.
- The expanded feature set additionally uses observed hidden-well GR versus typewell GR diagnostics. GR is an inference-time input; hidden TVT is not used.
- TVT truth and per-method TVT RMSE/SSE are labels and offline scoring values only. They are never present in the feature matrix.
- Well IDs are joins only and are not encoded as features.
- Every imputer, scaler, and regressor is fitted inside its training fold.

## Fixed evaluation design

- Wells: 773; scored rows: 3,783,989.
- Selector candidates: `final`, `particle_filter`, `tracker_blend`. `beam_viterbi` and `hold_last` are included only in the oracle ceiling because their frozen pooled baselines are much weaker.
- Shuffled protocol: fixed-seed 5-fold well KFold.
- Spatial protocol: fixed 5-cluster KMeans on standardized X/Y centroids, holding out one complete cluster at a time.
- Models: Ridge(alpha=20) and ExtraTrees(256 trees, depth 5, leaf 20), with no tuned threshold or post-hoc abstention margin.

## Baseline and oracle ceiling

| Method | Pooled RMSE | Choice counts |
|---|---:|---|
| Always final | 12.5185 | final=773 |
| Oracle over eligible 3 | 12.3886 | `{'final': 481, 'particle_filter': 226, 'tracker_blend': 66}` |
| Oracle over all 5 | 8.8881 | `{'final': 242, 'particle_filter': 226, 'tracker_blend': 61, 'beam_viterbi': 113, 'hold_last': 131}` |

Oracle rows use held-out TVT truth and show only the theoretical ceiling; they are not deployable selectors.

## OOF summary

| Feature set | Model | Protocol | Selector | Delta vs final | Improved folds | Choices (final / PF / tracker) | Gate |
|---|---|---|---:|---:|---:|---|---|
| core | ridge_delta | shuffled_well | 12.5231 | +0.0046 | 1/5 | 663 / 14 / 96 | fail |
| core | ridge_delta | spatial_cluster | 12.5273 | +0.0089 | 2/5 | 622 / 22 / 129 | fail |
| core | extra_trees_delta | shuffled_well | 12.5185 | -0.0000 | 2/5 | 771 / 0 / 2 | fail |
| core | extra_trees_delta | spatial_cluster | 12.5185 | +0.0000 | 0/5 | 773 / 0 / 0 | fail |
| expanded_hidden_gr | ridge_delta | shuffled_well | 12.5396 | +0.0211 | 1/5 | 653 / 21 / 99 | fail |
| expanded_hidden_gr | ridge_delta | spatial_cluster | 12.5363 | +0.0178 | 0/5 | 625 / 19 / 129 | fail |
| expanded_hidden_gr | extra_trees_delta | shuffled_well | 12.5185 | -0.0000 | 1/5 | 772 / 0 / 1 | fail |
| expanded_hidden_gr | extra_trees_delta | spatial_cluster | 12.5185 | +0.0000 | 0/5 | 773 / 0 / 0 | fail |

## Primary selector per-fold results

### shuffled_well

| Fold | Validation wells | Rows | Final | Selector | Delta | Choices (final / PF / tracker) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 155 | 766,954 | 11.6583 | 11.6863 | +0.0280 | 138 / 0 / 17 |
| 1 | 155 | 757,542 | 12.9512 | 12.9378 | -0.0135 | 137 / 4 / 14 |
| 2 | 155 | 764,814 | 13.4898 | 13.4901 | +0.0003 | 135 / 1 / 19 |
| 3 | 154 | 741,666 | 13.0953 | 13.1005 | +0.0051 | 130 / 3 / 21 |
| 4 | 154 | 753,013 | 11.2514 | 11.2569 | +0.0055 | 123 / 6 / 25 |

### spatial_cluster

| Fold | Validation wells | Rows | Final | Selector | Delta | Choices (final / PF / tracker) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 208 | 1,081,628 | 11.4206 | 11.4508 | +0.0301 | 130 / 15 / 63 |
| 1 | 79 | 340,190 | 12.8491 | 12.8409 | -0.0082 | 67 / 1 / 11 |
| 2 | 107 | 540,934 | 11.5466 | 11.5505 | +0.0039 | 88 / 1 / 18 |
| 3 | 215 | 1,030,819 | 12.9833 | 12.9902 | +0.0070 | 191 / 3 / 21 |
| 4 | 164 | 790,418 | 13.7701 | 13.7669 | -0.0032 | 146 / 2 / 16 |

## Conclusion

The primary selector did not clear the same frozen gate in both protocols. Do not deploy or submit an adaptive selector from this experiment, and do not retune thresholds on these 773 labels.

Reproduce with:

```powershell
uv run python scripts/analyze_targetfree_selector.py
```
