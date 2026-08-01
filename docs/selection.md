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

The authenticated Kaggle API reports **zero currently selected final submissions**. On
2026-08-02, the signed-in Submissions page independently showed **0/2** and stated that up to two
submissions may be selected. If fewer than two are selected, Kaggle will automatically select
from the best-scoring submissions. All nine competition submissions are in a terminal state,
and the three frozen fallback Notebooks are `COMPLETE`. No duplicate submission is needed.

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

Current recommendation:

1. manually select `54952634` and `54967104`, the confirmed two-entry allowance;
2. do not rely on Kaggle's automatic selection when the intended pair is already frozen;
3. keep `54927581` runnable as a disaster fallback, but do not displace the second candidate
   without new evidence;
4. treat the 2026-08-02 read-only selection preflight as complete;
5. complete the live selection by **2026-08-05 20:00 UTC**, approximately four hours before the
   official 2026-08-05 23:59 UTC deadline.

The latest audited public mirror,
`farhanabidtech786/rogii-best-score-wellbore-geology-prediction`, produced the exact primary SHA
and reused the same public full-stack branch logic. It is not a new candidate and does not justify
another Notebook or submission.
