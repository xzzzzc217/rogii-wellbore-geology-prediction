# ROGII candidate delta analysis

Date: 2026-07-24

## Decision

The strongest clean signal is the **exact-6.451 run's pre-correction, 128-seed
anchor**, not its VP/model-package/continuity/hedge layers. On the three visible
demo wells it has pooled RMSE `3.2556`, versus `4.2662` for safe-v2 and `4.5211`
for safe-v3 final.

This is only a post-hoc demo diagnostic. It supports independently validating a
sanitized 128-seed anchor, but it does not support selecting a hidden submission
from these three wells alone.

Executable calls:

- **GO:** independently validate a sanitized exact-6.451 clean anchor with all
  same-ID/contact/overlap logic disabled.
- **NO-GO:** submit safe-v3 solely because its visible-prefix layer moved a demo
  well.
- **NO-GO:** treat the exact-6.451 raw final demo score as generalization evidence.
- **NO-GO:** add the `0.00425 / scale=6` model package or cap-2 branch hedge based
  on this run.
- **NO-GO:** use a fixed interior blend. No nontrivial strict blend interval
  exists.

## Scope and leakage boundary

The analysis loaded frozen normal-run submissions first, then read the same-ID
train `TVT` only to calculate post-hoc metrics. Truth was never used to construct
or modify a candidate. Same-ID lookup, contact reconstruction, and overlap
recovery are explicitly excluded from every recommendation.

The exact public notebook's normal visible run did execute same-well/contact/
overlap layers. Therefore its raw final and all stages after those layers are
reported for forensic accounting only. They are not clean candidate evidence.

Reproduce the report:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_candidate_deltas.py --self-check
```

Machine-readable output:
`artifacts/candidate_delta_analysis.json`.

## Frozen artifact map

| Label | Artifact | Evidence class |
|---|---|---|
| safe-v2 final | `artifacts/kaggle_runs/pf_safe_v2_v1/submission.csv` | clean target-free |
| safe-v3 base | `artifacts/kaggle_runs/pf_safe_v3_v1/submission_sp45_learned_w0.60.csv` | clean target-free |
| safe-v3 final | `artifacts/kaggle_runs/pf_safe_v3_v1/submission.csv` | clean target-free |
| exact clean anchor | `artifacts/kaggle_runs/pf_public_reproduction_v1/submission_sp45_learned_w0.60.csv` | clean, before forbidden layers |
| exact pre-modelpkg | `artifacts/kaggle_runs/pf_public_reproduction_v1/submission_before_model_package.csv` | upstream-confounded |
| exact pre-hedge | `artifacts/kaggle_runs/pf_public_reproduction_v1/submission_before_branch_hedge.csv` | upstream-confounded |
| exact raw final | `artifacts/kaggle_runs/pf_public_reproduction_v1/submission.csv` | upstream-confounded |

The safe-v2/v3 logs show a 64-seed PF ensemble. The exact run shows 128 seeds.
All three chose the same visible selector variants and use the same `0.60`
SP45/learned blend. The observed clean-anchor gain is consistent with the
higher-fidelity exact configuration, but this comparison does not prove that
seed count alone caused it.

## Unified demo RMSE

| Candidate | `000d7d20` | `00bbac68` | `00e12e8b` | Pooled |
|---|---:|---:|---:|---:|
| safe-v2 final | 3.2913 | 3.6614 | 5.6103 | 4.2662 |
| safe-v3 base | 3.3602 | 4.0309 | 5.9773 | 4.5634 |
| safe-v3 final | 3.3602 | 3.9171 | 5.9773 | 4.5211 |
| exact clean anchor | **1.8343** | 3.7973 | **3.4217** | **3.2556** |
| exact raw final | 0.0053 | 0.0052 | 1.9988 | 1.1019 |

The raw final's apparent advantage is invalid as a generalization comparison.
Immediately before the hedge, the upstream-confounded trajectory has pooled
RMSE `0.0053`, which is the unmistakable signature of the visible same-well/
contact/overlap path.

Safe-v3 is worse than safe-v2 on every demo well. The exact clean anchor is much
better on `000d7d20` and `00e12e8b`, but is `0.1360` worse on `00bbac68`.

## Prediction deltas

| Change | Mean signed | Mean absolute | P95 absolute | Max absolute | RMS move |
|---|---:|---:|---:|---:|---:|
| safe-v2 → safe-v3 final | +0.2584 | 0.4347 | 1.2881 | 1.8972 | 0.5921 |
| safe-v2 → exact clean anchor | -0.2062 | 1.5913 | 3.1849 | 4.7331 | 1.8342 |
| safe-v3 final → exact clean anchor | -0.4646 | 1.6286 | 3.6178 | 4.9771 | 1.9498 |
| exact clean anchor → raw final | -0.6957 | 2.2884 | 6.9361 | 10.0006 | 3.1592 |

The clean exact anchor is not a trivial perturbation or blend of safe-v2/v3.
Its mean absolute difference from safe-v2 is `1.5913 ft`, with well-level RMS
moves of `1.6884`, `1.4092`, and `2.3979 ft`.

## Which correction contributed?

### Visible-prefix calibration

The exact run's VP contribution cannot be isolated: its VP input had already
been replaced by the same-well/contact/overlap stack, and the selected contact
candidates kept the contaminated base.

Safe-v3 provides the closest clean VP proxy:

- only `00bbac68` moved;
- mean absolute move on that well: `0.5109 ft`;
- maximum move: `1.9031 ft`;
- well RMSE: `4.0309 → 3.9171`;
- pooled RMSE: `4.5634 → 4.5211`, a small `0.0423` improvement.

This is a real but weak clean contribution. It does not recover safe-v2, which
remains better on all three wells.

### Model package

Actual selected contribution: **zero**.

The package disagreed with its already-confounded input by P95 `26.7007 ft`,
exceeding the `25 ft` guard. The correction report marks
`disabled_by_diff_guard=True` and `selected_for_submission_csv=False`;
pre-modelpkg and pre-hedge files are byte-equivalent in prediction values.

### U-continuity

Actual contribution: **zero / absent**.

The exact-6.451 notebook and run artifacts do not contain a U-continuity final
stage. No conclusion about U-continuity can be drawn from this candidate.

### PF seed-branch hedge

Observed contribution: negative on this demo, with upstream confounding noted.

The hedge moved all `4,301` rows of `00e12e8b` by exactly `+2 ft`, while leaving
the other wells unchanged. It changed:

- `00e12e8b` RMSE: `0.0054 → 1.9988`;
- pooled RMSE: `0.0053 → 1.1019`.

Because the input trajectory is already contaminated, this is not a clean
estimate of hedge generalization. It is nevertheless no positive evidence for
the cap-2 hedge.

### Attribution summary

Among identifiable, allowed corrections:

1. clean VP proxy: small positive (`-0.0423` pooled RMSE);
2. selected model package: zero;
3. U-continuity: absent;
4. branch hedge: negative in the observed run.

The large apparent `3.2503` RMSE gain between the clean exact anchor and
pre-modelpkg stage comes from a confounded same-well/contact/overlap stack and
must be discarded. The largest useful signal is therefore the clean 128-seed
anchor itself.

## Fixed-blend audit

Blend definition:

`prediction = (1 - alpha) * first + alpha * second`.

| Pair | Pooled optimum | Per-well optima | Leave-one-well-out optima | Strict interval |
|---|---:|---|---|---|
| safe-v2 / safe-v3 | 0.0000 | 0, 0, 0 | 0, 0, 0 | `{0}` |
| safe-v2 / exact clean | 1.0000 | 1.0000, 0.2447, 1.0000 | 1.0000, 1.0000, 0.9928 | `{1}` |
| safe-v3 / exact clean | 1.0000 | 1.0000, 0.8113, 1.0000 | 1.0000, 1.0000, 1.0000 | `{1}` |

“Strict” means pooled RMSE is no worse than the better endpoint and every well
is no worse than its worse endpoint, evaluated on an alpha grid of `0.01`.

There is no nontrivial interior interval. The diagnostics choose an endpoint:
safe-v2 over safe-v3, and the exact clean anchor over either safe candidate.
With only three same-ID demo wells, inventing a looser interval would be
post-hoc tuning, so a fixed blend is **NO-GO**.

## Recommended next evidence

Run the exact clean anchor as a separate sanitized profile:

- retain the 128-seed PF/selector/learned anchor;
- disable same-ID, contact, overlap, VP, model package, U-continuity, hedge, and
  global bias for the first attribution run;
- validate on the frozen broader target-free well split;
- add at most one target-free correction per later ablation.

This isolates whether the clean-anchor improvement survives beyond the three
demo wells and prevents a correction layer from taking credit for the public
notebook's forbidden path.
