# GR strongest-periodic-component notch — preregistration

Date frozen: 2026-07-30, before any target-scored notch result was inspected.

## Hypothesis

The public full-stack source contains a dormant, target-free GR preprocessing
lever that removes the strongest non-DC Fourier component and its immediate
neighbors.  Repeated mechanical or acquisition-period noise may otherwise
dominate the horizontal/typewell GR likelihood used by both the deterministic
particle filter and beam tracker.

This experiment isolates that transform from the public stack.  It does not
enable same-ID contact recovery, formation columns, leaderboard bias, probes,
canaries, fixed test wells, or public prediction artifacts.

## Frozen transform

- Apply only when at least 64 GR rows are finite.
- Fill missing rows by linear interpolation for spectrum calculation only.
- Median-center the full inference-time horizontal GR sequence.
- Compute `rfft`, identify the largest non-DC magnitude bin, and zero that bin
  plus one adjacent bin on each side.
- Invert the transform and replace only originally finite GR rows.
- Run the unchanged deterministic `TrajectoryConfig` with eight seeds.
- No frequency, radius, blending weight, or per-well activation threshold may
  be tuned after seeing the result.

## Frozen evaluation and gate

The candidate is evaluated on all 773 official train wells, scoring only each
well's official `TVT_input`-missing suffix.  The existing
`artifacts/target_free_eval_773.json` is the frozen no-notch comparator and
must match the current default predictor configuration and scored row counts.

The decision uses the existing U-continuity deployment gate without changes:

- pooled RMSE delta at most `-0.020`;
- macro-well RMSE delta below zero;
- paired-well bootstrap 95% upper bound below zero;
- at least four of five fixed spatial clusters non-worse;
- no spatial cluster worse than `+0.020`.

`PASS` authorizes only a later competitive-base integration audit.  It does
not authorize a Kaggle submission.  `NO-GO` stops the route; the notch radius,
minimum rows, and spectral selection rule will not be retuned.

Reproduction command:

```powershell
uv run python scripts/evaluate_gr_notch.py --jobs 6 --fail-fast
```

## Frozen result

Decision: **NO-GO**.

- Runtime: 716.7 seconds with six local worker processes.
- Successful wells: 773 / 773; failures: 0.
- The notch activated on all 773 wells.
- Baseline pooled RMSE: 12.518489.
- Notch pooled RMSE: 16.811012; delta **+4.292523**.
- Baseline macro-well RMSE: 9.104812.
- Notch macro-well RMSE: 13.448156; delta **+4.343344**.
- Per-well outcomes: 225 wins, 548 losses, 0 ties.
- Paired-well bootstrap 95% interval: `[+3.069503, +5.542658]`.
- Every spatial cluster worsened; deltas were `+2.953472`, `+3.324595`,
  `+4.767833`, `+5.376331`, and `+4.487587`.
- All five preregistered gate checks failed.

Artifact:
`artifacts/target_free_eval_773_gr_notch.json`

SHA-256:
`51bd98350310ef2c79225413f2525acca0af55e71cbf774c12b5d65692628798`

The strongest periodic component is carrying useful stratigraphic signal, not
a safely removable acquisition artifact.  The route stops without tuning the
frequency choice, radius, minimum rows, or any downstream blend.
