# Medal strategy

## Success criterion

Finish in a Kaggle medal position on the private leaderboard. Public leaderboard position is only
an intermediate diagnostic. A reproducible final notebook and a valid final submission are
minimum completion conditions even if the medal target is missed.

## Phase 0 — access and reproducibility (July 23-24)

1. Accept the competition rules before the July 29 entry deadline.
2. Configure Kaggle CLI authentication without committing credentials.
3. Download metadata, competition files, public leaderboard, and the highest-signal public
   notebooks and discussions.
4. Build one command that reproduces local validation and one notebook that reproduces a Kaggle
   submission.

Exit condition: a scored baseline submission and a recorded leakage-safe validation score.

## Phase 1 — data geometry and validation (July 24-26)

Audit:

- file inventory, row counts, dtypes, missingness, and image/log modalities;
- well identifiers encoded in `id` or explicit columns;
- train/test well overlap and contiguous row structure;
- spacing and ordering along measured depth;
- target smoothness, discontinuities, range, and per-well distributions;
- duplicates, near-duplicates, and repeated geological patterns;
- whether public auxiliary data has defensible provenance and direct overlap.

Validation candidates:

1. leave-one-well-out or grouped folds;
2. grouped temporal/depth blocks within wells;
3. adversarial validation for train/test shift;
4. a deliberately pessimistic holdout that resembles hidden wells.

The chosen validation must rank several materially different models consistently. Random row CV
is retained only as a leakage diagnostic.

## Phase 2 — model families (July 26-30)

Run distinct model families rather than shallow hyperparameter sweeps:

1. physical/analytic reconstruction from geometry and incidence angle;
2. gradient-boosted trees on raw, relative-depth, rolling, derivative, and well-level features;
3. sequence model over ordered well rows;
4. nearest-pattern or template matching across wells;
5. post-processing with physically plausible smoothing or piecewise constraints.

For each family, record:

- hypothesis;
- validation protocol and score;
- per-well error;
- error by target range and local curvature;
- public score for the few submissions used as probes;
- failure mode and next decision.

## Phase 3 — ensemble and robustness (July 30-August 2)

- Generate out-of-fold predictions for every surviving family.
- Select blends on grouped out-of-fold RMSE, with fold-level stability constraints.
- Reject components that improve only random-row CV or only the public leaderboard.
- Add uncertainty estimates from fold/model disagreement.
- Stress-test missing inputs, well-length extremes, and distribution shift.

## Phase 4 — final code competition packaging (August 2-3)

- Freeze dependencies and input dataset versions.
- Run the notebook with internet disabled.
- Verify `submission.csv` row count, IDs, column order, finite predictions, and target range.
- Keep runtime below eight hours to preserve one hour of safety margin.
- Prepare at least two diverse final notebook versions.

## Phase 5 — final selection (August 4-5)

- Reserve daily submission quota for final notebook reruns and recovery.
- Select final submissions using validation robustness, not the single best public score.
- Do not make last-hour structural changes.
- Finalize submissions before **2026-08-05 20:00 UTC**, four hours before the official cutoff.
