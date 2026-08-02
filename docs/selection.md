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
