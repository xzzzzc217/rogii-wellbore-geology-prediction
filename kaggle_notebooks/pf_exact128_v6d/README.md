# ROGII Exact128 v6d

Exact128 v6d is a deterministic and sanitized derivative of the audited
public-6.451 source notebook. After independent root review, version 1 was
pushed for one ordinary visible-data run and completed normally. It has never
been submitted to the competition, and no version 2 has been created.

## Retained scoring stack

- selector PF: 128 seeds, 500 particles, ridge/selector weights 0.30/0.70;
- learned trajectory PF: 128 seeds;
- robust projection: degree 3, weight 0.75;
- clean blend: 0.60 projected SP45 / 0.40 learned;
- visible-prefix: original `balanced` profile, `24/48` PF seeds, 350 particles,
  cut fractions `(0.50, 0.65, 0.75)`, and `profile` final selection;
- model package: exact public maximum gate `0.00425`, scale `6.0`, and global
  disagreement-p95 disable threshold `25.0`;
- generalized PF branch hedge: strength `0.60`, minimum branch mass `0.25`,
  separation range `4..40`, and cap `2.0`;
- U-continuity: off.

The public title score is provenance only and is not a score claim for this
sanitized candidate.

## Safety and determinism

Same-ID/contact candidates and reconstruction helpers, the same-well overlap
probe and override, train-TVT overlap routes, global LB bias, fixed IDs, fixed
row counts, expected hashes, fingerprints, and canaries are absent from the
scoring path. Both train-derived test feature builders pass the query well as
`self_wid`; the visible-prefix surface and dense imputers do the same.

The proven Clean128 v5d repair is reused with a new namespace:

```text
SHA256("rogii-exact128-v6d\0" + well_id + "\0" + branch)
```

Every formerly unseeded `_pf_ancc` and `_pf_z` Numba kernel seeds before its
first draw. `_resamp` is deterministic and consumes draws owned by the seeded
parent. The already seeded selector and 128-seed likelihood PF remain intact.

The selector and learned stages have fail-closed fallbacks. Visible-prefix and
model-package stages have start/per-well budget guards under a 27,000-second
hard envelope with a 1,800-second final reserve. The final CSV contract derives
its row count, ID set, and order only from the mounted sample submission.

## Local validation

```powershell
uv run python scripts/build_pf_exact128_v6d.py
uv run python kaggle_notebooks/pf_exact128_v6d/validate_notebook.py
```

The validator compiles every code cell; reruns the builder to prove byte
idempotence; compares every retained parent cell; enumerates all random Numba
kernels and every seed call; audits 6,184 well/branch seed pairs; verifies
query-well exclusion and exact retained parameters; and proves the dynamic
sample contract is the last CSV writer.

## Authorization state

Version 1 ordinary execution passed its dynamic contract and an independent
strict audit. It remains unauthorized for a hidden rerun or competition
submission. Frozen evidence is in
`artifacts/kaggle_runs/pf_exact128_v6d_v1` and
`artifacts/pf_exact128_v6d_v1_stage_analysis.json`.
