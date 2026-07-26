# Guard640-v4 stage analysis

Date: 2026-07-24

## Decision

On the three visible demo wells, the frozen Guard640-v4 final trajectory has
pooled RMSE `3.7747`, improving its clean anchor `3.8246` by `0.0498`.
U-continuity is the only actual layer with a material positive contribution:

- VP: `+0.0008` pooled RMSE, essentially neutral but slightly negative;
- selected model package: `0.0000`, because the disagreement guard disabled it;
- branch hedge: `+0.0496`, negative;
- U-continuity: `-0.1002`, positive and large enough to recover the hedge loss;
- complete actual stack: `-0.0498` versus the clean anchor.

The unselected `0.49 / scale=192` model-package candidate is a clear **NO-GO**:
it worsens pooled RMSE from `3.8254` to `7.5237`.

These values are post-hoc diagnostics, not candidate-fitting scores. Frozen
prediction artifacts were loaded before same-ID train truth was read. No truth,
same-ID lookup, contact reconstruction, or overlap recovery was used to create
or alter a candidate.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\analyze_guard640_stages.py `
  artifacts\kaggle_runs\guard640_v4_v1 `
  --label guard640_v4_v1 `
  --self-check
```

Outputs:

- `artifacts/guard640_v4_v1_stage_analysis.json`
- `scripts/analyze_guard640_stages.py`

The same script can be applied unchanged to a v4b output directory.

## Safety state

The frozen audit confirms:

- same-ID train lookup: off;
- guarded overlap override and overlap probe: off;
- exact-match recovery: off;
- contact override: off;
- global LB bias: off;
- hidden sample ID order and 14,151-row contract: valid.

VP, branch hedge, and U-continuity ran. The model-package computation ran, but
its candidate was not selected.

## Stage RMSE

| Frozen stage | `000d7d20` | `00bbac68` | `00e12e8b` | Pooled |
|---|---:|---:|---:|---:|
| Clean anchor (`w0.60`) | 3.0368 | 3.5809 | 4.6874 | 3.8246 |
| After VP | 3.0368 | 3.5829 | 4.6874 | 3.8254 |
| After selected modelpkg | 3.0368 | 3.5829 | 4.6874 | 3.8254 |
| After hedge | 3.0368 | 3.5829 | 4.8196 | 3.8750 |
| Final after U-continuity | 3.0447 | 3.5711 | 4.5586 | **3.7747** |
| Unselected modelpkg `0.49/192` | 6.3600 | 9.2083 | 5.6215 | **7.5237** |

The final composition is better than the clean anchor on `00bbac68` and
`00e12e8b`, and worse by only `0.0078` on `000d7d20`.

## Layer moves

| Layer | Mean signed move | Mean absolute move | P95 absolute | Max absolute | Moved rows | Pooled RMSE delta |
|---|---:|---:|---:|---:|---:|---:|
| VP | -0.0174 | 0.0552 | 0.2579 | 0.5529 | 6,013 | +0.0008 |
| Modelpkg `0.49/192`, unselected | -7.9162 | 7.9447 | 13.2441 | 14.3012 | 14,151 | +3.6983 |
| Selected modelpkg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| Branch hedge | +0.1573 | 0.1573 | 0.5175 | 0.5175 | 4,301 | +0.0496 |
| U-continuity | +0.1084 | 0.1402 | 0.7809 | 5.5821 | 14,151 | **-0.1002** |
| Full actual stack | +0.2483 | 0.3466 | 1.0228 | 6.0995 | 14,151 | **-0.0498** |

The notebook log's continuity `mean_abs_move=0.14665` is the unweighted mean
of the three per-well means. The row-weighted value across all 14,151
submission rows is `0.14017`, as shown above.

## Per-well attribution

### VP

Only `00bbac68` moved:

| Well | Mean signed | Mean absolute | Max absolute | RMSE before | RMSE after | Delta |
|---|---:|---:|---:|---:|---:|---:|
| `000d7d20` | 0.0000 | 0.0000 | 0.0000 | 3.0368 | 3.0368 | 0.0000 |
| `00bbac68` | -0.0409 | 0.1300 | 0.5529 | 3.5809 | 3.5829 | +0.0020 |
| `00e12e8b` | 0.0000 | 0.0000 | 0.0000 | 4.6874 | 4.6874 | 0.0000 |

The selected VP move is too small and slightly negative to justify itself from
these wells.

### Unselected model package

| Well | Mean signed | Mean absolute | Max absolute | RMSE before | Candidate RMSE | Delta |
|---|---:|---:|---:|---:|---:|---:|
| `000d7d20` | -8.1797 | 8.1797 | 12.5192 | 3.0368 | 6.3600 | +3.3232 |
| `00bbac68` | -9.3368 | 9.3368 | 14.3012 | 3.5829 | 9.2083 | +5.6254 |
| `00e12e8b` | -5.6948 | 5.7885 | 8.9161 | 4.6874 | 5.6215 | +0.9341 |

The underlying package disagreement has P95 `27.5867 ft`, exceeding the
`25 ft` guard. The guard correctly set
`selected_for_submission_csv=False`; the actual selected-stage move is exactly
zero.

### Branch hedge

Only `00e12e8b` moved. Every one of its 4,301 hidden rows received the same
`+0.51745 ft` shift:

| Well | Mean move | RMSE before | RMSE after | Delta |
|---|---:|---:|---:|---:|
| `000d7d20` | 0.0000 | 3.0368 | 3.0368 | 0.0000 |
| `00bbac68` | 0.0000 | 3.5829 | 3.5829 | 0.0000 |
| `00e12e8b` | +0.5175 | 4.6874 | 4.8196 | +0.1322 |

This is negative demo evidence for the current hedge.

### U-continuity

| Well | Mean signed | Mean absolute | Max absolute | RMSE before | RMSE after | Delta |
|---|---:|---:|---:|---:|---:|---:|
| `000d7d20` | -0.0586 | 0.0586 | 0.9350 | 3.0368 | 3.0447 | +0.0078 |
| `00bbac68` | +0.0692 | 0.0692 | 1.7305 | 3.5829 | 3.5711 | -0.0118 |
| `00e12e8b` | +0.3121 | 0.3121 | 5.5821 | 4.8196 | 4.5586 | -0.2610 |

U-continuity is especially valuable on `00e12e8b`. Its improvement remains
net-positive after accounting for the preceding hedge: final pooled RMSE
`3.7747` is also better than the pre-hedge/VP stage `3.8254`.

## Go/no-go

- **GO for broader validation:** U-continuity with cap `8`, tau `240`.
- **Conditional GO for an independent candidate:** the actual v4 composition,
  because the final is `0.0498` better than its clean anchor and passes the
  structural safety audit.
- **NO-GO from demo evidence:** the current VP and branch hedge.
- **NO-GO:** forcing the `0.49 / scale=192` model-package candidate past the
  P95 guard.

The practical clean ablation to test next is `anchor + U-continuity`, with VP,
model package, and hedge disabled. This recommendation is based only on
target-free layers and still requires broader frozen validation rather than
demo tuning.
