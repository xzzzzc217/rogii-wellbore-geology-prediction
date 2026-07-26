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
