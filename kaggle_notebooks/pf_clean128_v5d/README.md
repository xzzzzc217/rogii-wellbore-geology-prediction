# ROGII PF Clean128 v5d

V5d is a deterministic, local-only child of `rogii-pf-clean128-v5`. It keeps
the complete 128-seed clean configuration and changes only stochastic state
initialization in the two formerly unseeded single-PF feature branches.

## Deterministic repair

For each well, stage, split, and PF branch, the notebook derives a positive
31-bit seed from:

```text
SHA256("rogii-pf-clean128-v5d\0" + well_id + "\0" + branch)
```

The `_pf_ancc` and `_pf_z` Numba kernels call `np.random.seed(seed)` before
their first random draw. Their `_resamp` helper no longer owns an implicit RNG:
the seeded parent draws `u0` and interleaved position/auxiliary noise in the
same order as the original helper, then passes those arrays into a deterministic
resampling function. This makes results independent of joblib scheduling while
preserving the original draw order after selecting a seed.

The already deterministic selector `run_particle_filter` retains its
`default_rng(seed)` path. The learned 128-seed `_pf_lik_allseeds` kernel retains
its existing `np.random.seed(seed_base + s)` loop. Random JIT warm-up inputs
were replaced by deterministic linspace arrays.

## Unchanged clean contract

- selector PF seeds: 128;
- learned PF seeds: 128;
- projection: degree 3, blend weight 0.75;
- final blend: 0.60 projected SP45 / 0.40 learned;
- same query-well exclusion and 7.5-hour fallback;
- zero VP/model-package/hedge/U-continuity/overlap correction cells;
- same dynamic runtime-sample contract as the final CSV writer.

## Validation

```powershell
.\.venv\Scripts\python.exe scripts\build_pf_clean128_v5d.py
.\.venv\Scripts\python.exe kaggle_notebooks\pf_clean128_v5d\validate_notebook.py
```

The validator enumerates every Numba function that calls `np.random`, verifies
the seed dominates its random draws, proves `_resamp` contains no random call,
checks every scoring wrapper and `build_well` call carries a stable well/branch
seed, and audits collision-free seeds over all 773 locally mounted wells and
eight scoring branch labels.

This candidate is not authorized for push or submission until root review.
