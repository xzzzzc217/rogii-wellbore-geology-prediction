# ROGII PF Safe v3

This directory contains an independent, unpushed Kaggle notebook candidate. It
is a safety-focused derivative of Roman Mironov's public
[PF-Config Strategy | High-Scoring 6.491](https://www.kaggle.com/code/helloromiro/pf-config-strategy-high-scoring-6-491),
captured on 2026-07-23 through our attributed reproduction and Safe v2 audit.
The public `6.491` score is not claimed for this modified notebook.

## Why v3 exists

Safe v2 completed the hidden rerun with a public score of `7.514`. V3 keeps its
leakage controls and restores the most plausible omitted improvement:
per-well visible-prefix calibration. The calibration evaluates candidate
trajectories only against the competition-provided known prefix, then applies a
bounded, profile-gated move to the unknown suffix.

The default profile is:

```python
SUBMISSION_PROFILE = "safe_v3_prefix"
VISIBLE_PREFIX_CAL_SEEDS = 12
VISIBLE_PREFIX_FINAL_SEEDS = 24
VISIBLE_PREFIX_PARTICLES = 250
RUN_MODEL_PACKAGE_CORRECTION = False
```

This is deliberately cheaper than the public 24/48-seed, 350-particle path.
It retains all three prefix cuts `(0.50, 0.65, 0.75)` and the balanced
application gate.

## Profiles

| Profile | Visible prefix | Model package | Intended use |
|---|---:|---:|---|
| `safe_v3_prefix` | 12/24 seeds, 250 particles | Off | Default candidate |
| `safe_v3_prefix_modelpkg` | Same | 49% max, 192-ft disagreement gate | Optional experiment only |

The external `pilkwang/rogii-model-package` interface was statically inspected:
its test builder requests `include_target=False` and query-well exclusion. That
does not independently verify the full training provenance, claimed score, or
hidden-scale runtime. It therefore remains off by default. The optional
`0.49`/`192` gate is taken from the newer public 6.40 guard's stated
300-well calibration plus 300 disjoint-well holdout selection. The downloaded
notebook contains no embedded execution outputs that reproduce that audit, so
this is promising public evidence rather than independent verification.
Auto-search remains disabled, and the inherited 25-ft p95 disagreement guard
can reject the package and retain the prefix-only submission.

## Review of the newer public 6.451/6.40 notebooks

The two downloaded 6.40 guard notebooks are source-identical. Their only clean,
potentially transferable late layer is a target-free boundary-continuity fade
in `U = TVT + Z` with an 8-ft cap and 240-MD decay. It uses no fixed well IDs,
row counts, submission hashes, or labels. However, those constants were
reported as validated after a materially different full composition: 128-seed
PF, visible-prefix alpha 0.35, model-package 0.49/192, and an active final PF
hedge. There are no embedded outputs proving transfer to our reduced,
package-free Safe v3 path. Because the task forbids an unvalidated final hedge,
the continuity fade is documented but not copied into this candidate.

Other deltas were explicitly rejected:

- the 6.451 A31 toe tilt relies on a final branch hedge and fingerprint guards;
- the 6.40 guard enables a stronger unconditional final PF midpoint hedge;
- its public-overlap cell reads same-ID train `TVT` and is prohibited even
  though it would normally be inactive for unseen hidden IDs;
- no fixed-ID/row/SHA/canary contract was imported.

## Immutable safety controls

Both profiles fail closed unless all of the following remain true:

```python
ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_EXACT_MATCH_RECOVERY = False
RUN_PF_BRANCH_HEDGE = False
APPLY_GLOBAL_LB_BIAS = False
MODEL_PACKAGE_ALLOW_AUTO_SEARCH = False
MODEL_PACKAGE_REQUIRE = False
```

Test feature construction passes the query well ID to both formation and dense
imputers for self-exclusion. The visible-prefix contact candidate and contact
reapplication are doubly gated by the disabled same-ID/contact controls.

## Runtime estimate for about 200 hidden wells

The observed Safe v2 visible-data run took about 525 seconds for three demo
wells. Its prior hidden-scale engineering estimate was 3.5–5.5 hours. The
public notebook's 24/48-seed, 350-particle prefix stage took about 108 seconds
for three wells. Scaling the dominant PF work by
`(12+24)/(24+48) * 250/350` gives about 39 seconds for three wells, or about
0.7 hours for 200 wells before I/O and scheduling variance.

Expected end-to-end ranges:

| Profile | Estimated 200-well runtime | Assessment |
|---|---:|---|
| `safe_v3_prefix` | 4.2–6.8 h | Designed to stay under the 8 h working budget |
| `safe_v3_prefix_modelpkg` | 5.2–8.8+ h | Not submission-ready without a full dry run |

The notebook records a 7.5-hour hard engineering budget and reserves the final
30 minutes for output and audit. Visible-prefix processing stops cleanly at the
guard, retaining the already-audited base trajectory for unprocessed wells.
The optional model package is skipped unless at least two hours plus the final
reserve remain before it starts. This cannot interrupt an already-running
external model package, so only the default profile is considered budget-safe.

## Validation

Run the local static validator before any deliberate push:

```powershell
python kaggle_notebooks/pf_safe_v3/validate_notebook.py
```

It parses the notebook and metadata, compiles every code cell independently,
checks default and optional profile definitions, confirms all prohibited gates,
checks query-well self-exclusion markers, rejects public-overlap, continuity,
forced-hedge, and fixed-contract code, and confirms that every cell is
unexecuted.

This notebook has not been pushed to Kaggle and must not be submitted until a
fresh review of the validator output and a hidden-scale dry run.
