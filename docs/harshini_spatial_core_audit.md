# Harshini spatial-core audit

## Decision

The public `luffyh04/harshini-submission-f` route is a **NO-GO** for a
controlled Kaggle submission.

Its fixed surface/PF core fails the first strict spatial holdout, and the
visible-data runtime is already long enough to create a material hidden-rerun
timeout risk. The expensive boosted ensemble was therefore not repeated.

## Candidate

The public notebook is structurally different from the existing exact
full-stack family. It:

- reconstructs six formation surfaces from the field well cloud;
- estimates several prefix-datum variants per surface;
- combines the surfaces using their prefix reconstruction errors;
- adds three particle-filter likelihood scales;
- trains two LightGBM models and one CatBoost model;
- combines the fixed blend and boosted legs with a positive Ridge model; and
- projects the result onto a robust fourth-degree `TVT + Z` trajectory.

The saved output contains 14,151 finite predictions in exact sample order.
Its SHA-256 is
`9607342F903F67778F6567F2661A85001B571AA558027A29DFFDC5CE4DEA822D`.
Relative to the current 6.477 output, the prediction RMSE distance is 2.990 ft.

The notebook reports the following random-well GroupKFold scores:

| Candidate | Pooled RMSE |
|---|---:|
| Flat | 15.589 |
| Fixed surface/PF blend | 9.048 |
| LightGBM 1 | 9.048 |
| LightGBM 2 | 8.942 |
| CatBoost | 9.035 |
| Positive Ridge stack | 8.741 |

The visible run took approximately 12,464 seconds, or 3.46 hours.

## Leakage audit

The implementation has two distinct evaluation paths:

1. The `EXACT datum` branch reads each training well's true formation columns
   and is used only to estimate an oracle ceiling.
2. The production `P` and `PR` dictionaries use the known `TVT_input` prefix
   and an imputed formation surface.

The final row features and inference path use only `P` and `PR`; the oracle
`BEX` values do not enter the final model. In ordinary leave-one-well-out
surface construction, the target well is also removed correctly.

The remaining problem is regional leakage. A random held-out well can still
receive formation surfaces from adjacent wells that are in the same spatial
region and remain in the field point cloud. GroupKFold around the boosted
model does not remove those validation-region labels from its inputs.

## Strict spatial protocol

`scripts/audit_harshini_spatial_core.py` reproduces the public notebook's
fixed production core under a stricter transaction:

1. Standardize each usable field well's X/Y centroid.
2. Create seven deterministic KMeans clusters with seed `20260727`.
3. Hold out fold 0 in full.
4. Remove every fold-0 well from the formation point cloud.
5. Refit the fallback formation trends using source wells only.
6. Reconstruct all fold-0 suffix surfaces using the nearest remaining wells.
7. Apply the frozen public constants:
   - imputation shrink 0.60;
   - imputation clip 80 ft;
   - surface weight 0.55;
   - PF weight 0.45; and
   - warm-up tau 85 ft.
8. Compare against flat, PF, and the original single-well LOO core.

The public pickle outputs were inspected before loading. Their only pickle
globals are NumPy array reconstruction primitives. The audit script uses a
restricted unpickler that rejects every other global.

Input hashes:

- `imp_cache.pkl`:
  `E4D1F260AC59B3863E122103A5826C0BF1FD29246044875E0E406ECAAF7BF615`
- `pf_train_cache.pkl`:
  `F6B1F52B84F7D2E64028BAD485EA7BDF7CBAEF4E297BD0FB98B5A7AAEDCB49CA`

## Result

Fold 0 contains 173 validation wells and 910,183 scored suffix rows. Removing
the complete region leaves 592 source wells. The average distance to the
nearest remaining formation source is 17,389 ft.

| Candidate | Pooled RMSE | Delta vs flat |
|---|---:|---:|
| Flat | 14.969 | — |
| PF | 8.598 | -6.371 |
| Original single-well LOO fixed core | 8.008 | -6.961 |
| Strict-spatial surface branch | 30.278 | +15.309 |
| Strict-spatial surface/PF core | **17.944** | **+2.975** |

The strict core is 9.937 RMSE worse than the original LOO core and 2.975 worse
than flat. The PF leg remains useful, but the new formation-surface lever does
not survive regional removal.

## Visible-output limitation

The three visible test trajectories have same-ID copies in the local training
directory, so a public-data output can find extremely close same-ID formation
points. That does not establish hidden Code Submission quality. The submitted
Frontier II run demonstrates the distinction: its downloaded visible output
is only 0.108 RMSE away from the 6.477 file, yet the hidden rerun scored 7.507,
1.030 worse. Those two score differences cannot arise on one fixed evaluation
vector by the RMSE triangle inequality. The hidden rerun therefore must be
treated as a different prediction transaction.

For that reason, neither visible same-ID diagnostics nor a public notebook
author's self-reported score can override the strict hidden-safety gate.

## Reproduction

The large caches and generated JSON remain ignored by Git:

```powershell
uv run python scripts/audit_harshini_spatial_core.py `
  --fold 0 `
  --output artifacts/harshini_spatial_core_fold0.json
```

The audit intentionally stops after the fixed core. Full seven-fold boosted
training is not authorized after this failure.
