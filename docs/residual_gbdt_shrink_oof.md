# Conservative residual model v3: frozen four-protocol OOF

## OOF decision

**GO as a robust residual component.** The fixed 0.20 residual scale passed
every predeclared check in all four protocols, including the new seven-cluster
spatial split that had not been scored when the scale was frozen.

## Configuration

- Same spatial-invariant feature set and CatBoost parameters as v2.
- Predicted residual clipped to +/-80 ft, then multiplied by exactly 0.20.
- No coordinate rule, well identifier, formation feature, same-ID lookup, or
  hidden TVT feature.
- At most 96 training rows per well.
- All four protocols had to pass the unchanged -0.10 pooled improvement,
  bootstrap, fold consistency, p95, and worst-well gates.

## Results

Hold-last TVT was the comparator at 15.909853 pooled RMSE.

| Protocol | Model RMSE | Delta | CI95 upper | Worst fold delta | Gate |
|---|---:|---:|---:|---:|---|
| Typewell fingerprint, 5-fold | 15.408506 | -0.501347 | -0.429750 | -0.426051 | PASS |
| Spatial clusters, 5-fold | 15.552337 | -0.357516 | -0.268930 | -0.229121 | PASS |
| Spatial clusters, 6-fold | 15.525020 | -0.384833 | -0.298198 | -0.049794 | PASS |
| Spatial clusters, 7-fold | 15.513411 | -0.396442 | -0.313100 | -0.049794 | PASS |

Every fold improved. The shared 35-well difficult region improved slightly
after shrinkage rather than failing catastrophically.

## Deployment qualification

OOF GO does not imply that this standalone model is leaderboard-competitive.
A frozen leave-three-test-wells-out diagnostic trained on the other 770 wells:

| Candidate | Three-well pooled RMSE |
|---|---:|
| Hold-last | 11.539311 |
| Spatial-invariant v3 | 11.332989 |

The component improved its comparator by 0.206322, but its absolute level is
well behind the current 6.477 full-stack submission. It is therefore
**NO-GO as a standalone Kaggle submission**. It remains eligible only as a
future component whose incremental effect is validated against a competitive
base trajectory.

## Evidence

- JSON: `artifacts/residual_gbdt_shrink_oof.json`
- Final artifact SHA-256:
  `D0EB5D9BD4B5C4CFB37F2ACD56EB09AE390C53D96400A47E7379A730AD772BC4`
- Runtime: 651.6 seconds
- Standard error log: empty
- Evaluator: `scripts/residual_gbdt_shrink_evaluate.py`
