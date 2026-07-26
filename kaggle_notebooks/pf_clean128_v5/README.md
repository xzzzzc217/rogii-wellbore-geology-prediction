# ROGII PF Clean128 v5

This is a local-only, sanitized candidate derived from the exact public
`6.451` clean anchor. The public title score is attribution only; it is not a
score claim for this modified notebook.

## Frozen scoring path

The candidate keeps exactly four mathematical layers:

1. selector/ridge SP45 anchor with selector PF at 128 seeds;
2. the existing degree-3, weight-0.75 robust projection;
3. learned trajectory inference with PF at 128 seeds and the mounted audited
   boosters;
4. a fixed `0.60 * projected SP45 + 0.40 * learned` blend.

All train-derived spatial priors receive the current query well ID and remove
that well before neighbor aggregation. A precomputed ID-matched learned
submission is not accepted; hidden predictions must be dynamically constructed
from the active test files.

## Removed correction paths

The notebook contains no executable cells for same-ID/contact reconstruction,
overlap probing, exact-overlap recovery, visible-prefix calibration, model
package correction, PF branch hedge, U-continuity, global leaderboard bias, or
fixed test fingerprints. Their control flags are immutable false and the
validator rejects any implementation marker or later prediction writer.

## Runtime guard

The engineering hard budget is 7.5 hours with a 30-minute final reserve and a
two-hour learned-stage reserve. The selector loop uses the last visible TVT for
remaining wells if that reserve boundary is reached. If the learned stage
cannot start or fails, the already-written projected SP45 trajectory is
restored before the fixed blend and dynamic sample contract.

The exact public visible run reached the clean blend in about 627 seconds for
three wells. Scaling the measured selector and learned PF work to the hidden
well count, while allowing Kaggle scheduling variance, gives an estimated
hidden runtime of about 5.0–6.8 hours. The fallback is designed to finish the
contract rather than start an unbounded late stage.

## Local validation

```powershell
.\.venv\Scripts\python.exe scripts\build_pf_clean128_v5.py
.\.venv\Scripts\python.exe kaggle_notebooks\pf_clean128_v5\validate_notebook.py
```

This directory is not authorized for push or submission until the root task
reviews the validator result and makes a separate go/no-go decision.
