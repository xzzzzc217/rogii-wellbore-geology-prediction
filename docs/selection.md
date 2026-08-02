# Competition selection — 2026-07-23

## Decision

Target **ROGII - Wellbore Geology Prediction** as the short-horizon medal competition.

## Verified facts

| Criterion | Verified value |
|---|---|
| Competition type | Featured Code Competition |
| Medal status | Awards Points & Medals |
| Metric | RMSE on `tvt` |
| Public/private split | Approximately 26% / 74% |
| Entry deadline | 2026-07-29 23:59 UTC |
| Final deadline | 2026-08-05 23:59 UTC |
| Notebook limit | CPU or GPU at most 9 hours |
| Internet | Disabled during submission run |
| External data | Freely and publicly available external data allowed |
| Participation at selection | 5,520 teams and 114,643 submissions |

These values were read from the live Kaggle competition pages on 2026-07-23.

## Rejected short-horizon candidates

### Predicting Student Health Risk (Playground Series S6E7)

It ends earlier and is much easier to iterate, but Kaggle explicitly says that it does not award
points or medals. It does not satisfy the primary objective.

### Soccer Feature Engineering Hackathon

Only three days remained at selection time. It is a Hackathon rather than a standard
medal-bearing Featured competition, so it is not an acceptable replacement for a Kaggle medal
attempt.

### Autonomous Agent Prediction (Beta)

Its reward is swag, and the competition page does not provide the same explicit medal signal as
ROGII.

## Risk assessment

ROGII is highly competitive and a medal cannot be guaranteed. The strongest controllable
advantages are:

1. leakage-safe validation by well;
2. reconstructing the target's physical geometry rather than treating rows as independent;
3. exploiting permitted public geological data only when provenance is documented;
4. using public-leaderboard probes sparingly because the public set is only 26%;
5. maintaining multiple diverse final notebook candidates.

## Final-submission freeze audit — 2026-07-31

On 2026-07-31, the authenticated Kaggle API reported **zero selected final submissions**. On
2026-08-02, the signed-in Submissions page independently showed **0/2** and stated that up to two
submissions may be selected. If fewer than two are selected, Kaggle will automatically select
from the best-scoring submissions. All nine competition submissions were in a terminal state,
and the three frozen fallback Notebooks were `COMPLETE`. No duplicate submission was needed.

Later on 2026-08-02, after explicit user confirmation, refs `54952634` and `54967104` were selected
manually. The page displayed **2/2**, both target checkboxes were checked, and an independent API
read returned exactly those two refs with `COMPLETE` status. No Notebook or competition
submission was created during selection.

| Priority | Submission ref | Public RMSE | Kaggle Notebook / version id | Output SHA-256 | Decision |
|---:|---:|---:|---|---|---|
| 1 | `54952634` | **6.477** | `heralces/rogii-pf-public-reproduction` / `337405234` | `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a` | Primary |
| 2 | `54967104` | **6.604** | `heralces/rogii-pf-public-no-hedge` / `337779623` | `fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9` | Branch-risk hedge |
| 3 | `54927581` | 7.514 | `heralces/rogii-pf-safe-v2` / `337407706` | Revalidate only if promoted | Structurally safer disaster fallback |

The primary and no-hedge CSVs were revalidated against the official sample: 14,151 rows,
exact `id,tvt` columns, exact ID order, unique IDs, and finite predictions. Their only difference
is 4,301 rows from one runtime-selected PF branch; every primary-minus-no-hedge delta is exactly
+2.0 ft, giving an all-row RMSE distance of 1.102608. The no-hedge candidate is therefore not an
independent model, but it covers one concentrated final-branch decision at a much smaller Public
penalty than safe-v2.

Final state:

1. `54952634` and `54967104` are manually selected under the confirmed two-entry allowance;
2. Kaggle's automatic selection is not relied upon;
3. keep `54927581` runnable as a disaster fallback, but do not displace the second candidate
   without new evidence;
4. treat the 2026-08-02 read-only selection preflight as complete;
5. perform only low-frequency read-only checks through the official 2026-08-05 23:59 UTC deadline.

The latest audited public mirror,
`farhanabidtech786/rogii-best-score-wellbore-geology-prediction`, produced the exact primary SHA
and reused the same public full-stack branch logic. It is not a new candidate and does not justify
another Notebook or submission.

## Post-freeze rank-improvement candidate — 2026-08-02

The freeze was reopened after the user explicitly asked to keep improving rank. A fresh public
scan found `raunakdey07/rogii-physics-informed-stacked-ensemble-v2`, whose author leaderboard
score is **6.432**. Its final output does not use the notebook's new MDN/GBDT prediction: it is
exactly the current 6.477 primary plus 2.0 ft on the same 4,301-row runtime-selected PF branch.

We reproduced that output with a private dynamic candidate rather than copying the public
fixed-test logic. `heralces/rogii-pf-branch-cap40-targetfree-v1` version 2 uses generic
PF branch statistics, strength 0.62, and a 4.0 ft cap. Its scoring path contains no fixed test
well, visible row count, expected artifact hash, probe flag, or leaderboard score. Candidate
selection is nevertheless explicitly recorded as public-score-informed.

Version 1 was rejected before submission because strength 0.60 produced only a 3.908582 ft move.
Version 2 completed in 785.8 seconds and passed the full visible-run contract: 14,151 rows, exact
`id,tvt` schema and sample ID order, unique IDs, finite predictions, 4,301 moved rows, exact
+4.0 ft delta, and report/file SHA agreement. Its visible-run SHA is byte-identical to the public
artifact associated with 6.432:
`af43d97ab5b76f7f0bbbc5c72fb6e9a829e939c48582fc4130144340547360a2`.
This is not a guaranteed score: Kaggle reruns Code Submissions on hidden data, so ref `55177746`
is the only authoritative evaluation.

The unique Code Submission is ref `55177746`, currently `PENDING`. Existing manual selections
remain `54952634 + 54967104` until the new ref has a terminal score. If it is strictly better
than 6.477, the recommended pair becomes `55177746 + 54952634`; do not change selection before
the score is confirmed.

## Target-free Functional Ridge/PLS candidate — 2026-08-02

A fixed 75% plain Functional Ridge plus 25% supervised Functional PLS curve ensemble passed every
preregistered spatial and Typewell gate. Spatial pooled delta was -0.035209 with 5/7 folds
improving; Typewell pooled delta was -0.023059 with all 5/5 folds improving. Both protocols beat
the independently submitted plain Ridge reference by more than the frozen margin.

Private Notebook `heralces/rogii-functional-ridge-pls-v1` version 1 completed and passed the full
hidden artifact contract. The final SHA is
`ffeb394ecc60ef4a95215b665baad9397119652e93505484806dc3c4bfdf482c`, based on the exact current
primary parent SHA `b192d3f...ded4a`. It uses no fixed test well, visible row count, expected hash,
leaderboard score, probe flag, or encoded well identity. The unique Code Submission is ref
`55179387`, currently `PENDING`.

Do not change the manually selected pair before ref `55179387` has a terminal score. If it is
strictly below 6.477, it should displace the 6.604 no-hedge selection at minimum. If cap4 and
Ridge/PLS both improve, first evaluate the already prepared conditional combination; otherwise
select the two strongest terminal candidates with the 6.477 primary retained as the robustness
anchor unless stronger evidence justifies replacing it.

## High-risk dip-rate HMM controlled validation — 2026-08-02

The target-free dip-rate HMM was the first new trajectory-state model to produce a large signal.
On 741 confirmation wells it improved pooled RMSE by **0.156756**, all 7/7 spatial folds, and
all 5/5 Typewell folds. A later audit found that the reported `p95_well_delta=-0.12914` was the
5th percentile mislabeled as p95; the true 95th percentile is adverse (about +0.628 after the
three current test-well IDs are excluded). It also failed its auxiliary confirmation gate:
63.83% of wells improved versus a 65% minimum, and the worst-well aggregate tail worsened by
0.68363 versus a 0.25 cap. The confirmation remains recorded as FAIL; the quantile correction
does not change that decision or the pooled/all-fold evidence.

The user's active objective is rank improvement, and the official metric is pooled row-level
RMSE rather than worst-well RMSE. Because the pooled gain is large, every independent group fold
improved, movement is hard-capped at +/-1 ft, and two daily submission slots remained, one
explicitly risk-disclosed controlled public validation is authorized. This is an exception to the
auxiliary tail stop rule, not a claim that the confirmation passed.

The exact 6.477 parent remains protected. The HMM Notebook may be pushed once and submitted once
only after its hidden artifact passes dynamic schema, ID, finite-value, movement-cap, report, and
SHA checks. No final selection changes automatically; only a terminal score strictly below 6.477
can promote the HMM candidate.

Version 1 completed in about 783 seconds. Its hidden artifact passed the full validator with
14,151 rows, exact ID order, three unique wells, finite predictions, all 14,151 rows changed,
maximum movement 1 ft, and matching before/final/report/audit hashes. The unique controlled
submission is ref `55180816`; it is pending and must not be duplicated.

## Rank-seeking dip-rate HMM scale2 — 2026-08-02

The fixed HMM correction has an exact quadratic SSE curve as its multiplier changes. After
excluding every current test-well ID from label-based analysis, the aggregate optimum was about
2.76 times the frozen scale1 correction and every spatial/Typewell fold had an optimum above
1.0. A round, deliberately sub-optimal multiplier of 2.0 was preregistered before confirmation;
it corresponds to overlay scale 0.40 and a hard maximum movement of +/-2 ft.

On the 738 non-test wells, scale2 improved pooled RMSE by **0.244037**, versus **0.156071** for
scale1. It improved all 7/7 spatial folds and all 5/5 Typewell folds; relative to scale1 it gained
another **0.087966** RMSE. This is materially stronger primary-metric evidence, but tail risk is
larger: true p95 well delta is +1.3359, worst-well delta +1.8142, and only 57.86% of wells improve.
Therefore it is a transparent rank-seeking candidate, not a robust default. It may receive one
independent private Notebook run and at most one controlled submission after artifact validation;
it cannot replace a final selection unless its terminal Public score is strictly below 6.477.

Private Notebook `heralces/rogii-dip-rate-hmm-scale2-v1` version 1 completed in about 766 seconds.
Its hidden output passed `scripts/validate_dip_rate_hmm_scale2_output.py`: 14,151 dynamic rows,
exact ID order, unique and finite predictions, all rows changed, 2 ft maximum movement, and
matching parent/final/report/audit hashes. Its sole controlled submission is ref `55181026` and
is pending. The daily submission allowance is now exhausted; this candidate must not be repeated.

## Conditional scale2.3 continuation — 2026-08-02

Before either HMM Public score was available, multiplier 2.3 was preregistered from the weakest
non-test fold's approximate zero-gain boundary of 2.354. On the same 738 non-test confirmation
wells it improved pooled RMSE by **0.256847**, another **0.012810** over scale2, while retaining
7/7 spatial and 5/5 Typewell fold improvements. The weakest Typewell fold improves by only
0.003804, so 2.3 is treated as the endpoint; no larger multiplier is authorized.

Tail risk rises again (true p95 +1.5538, worst well +2.0891, 56.23% well win rate). A local-only
Notebook and validator are prepared, with `execution_authorized=false`. It may be pushed once only
if ref `55181026` first receives a terminal Public score strictly below 6.477. A non-improving or
missing scale2 score closes scale2.3 without using GPU or a submission slot.

## Conditional HMM scale2 + Functional Ridge/PLS additive candidate — 2026-08-02

Before any component Public score was visible, the two frozen target-free corrections were added
in their deployment order: HMM scale2 first, then the fixed 75/25 Functional Ridge/PLS overlay.
Evaluation excludes the 32 HMM discovery wells and every current test-well ID. Functional models
remain outer-fold isolated, and no weight, scale, clip, HMM parameter, or component order was
searched after validation labels were observed.

Across 738 wells, spatial pooled RMSE improves from 10.215544 for HMM scale2 to **10.178767**,
an additional gain of **0.036777**; 6/7 spatial folds improve. Typewell pooled RMSE improves from
10.215544 to **10.186974**, an additional gain of **0.028570**; all 5/5 Typewell folds improve.
The total deltas versus the competitive base are **-0.280814** and **-0.272607**. This is the
strongest independent additive OOF evidence currently available.

The candidate remains high risk: true p95 well deltas versus HMM scale2 are +0.5744 and +0.5857,
and only about 51% of wells improve. It is therefore prepared locally as
`heralces/rogii-hmm-scale2-functional-ridge-pls-v1`, with a dedicated intermediate/final SHA
validator, but `execution_authorized=false`. It may be pushed, run, and submitted once only if
both component refs `55181026` and `55179387` first receive terminal Public RMSE strictly below
6.477. If either component fails, errors, or has no score, the combination closes without using
GPU or a submission slot. It never changes the manually selected final pair automatically.
