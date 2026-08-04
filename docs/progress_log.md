# Progress log

## 2026-07-23

- Joined `ROGII - Wellbore Geology Prediction` and accepted the official rules.
- Authenticated Kaggle CLI as `heralces`.
- Downloaded the 2,327-file competition archive, verified every ZIP member by CRC, and
  extracted the clean copy to `data/competition/`.
- Audited all 773 training wells and the 3 visible demonstration test wells:
  0 errors, 1 expected train/test well-overlap warning, and exact sample-submission coverage.
- Captured a public leaderboard snapshot: 5,520 teams; about 6.644 RMSE at the top-10%
  boundary and 6.515 at the top-5% boundary.
- Established a leakage-safe validation policy: well-grouped prefix backtests, pooled and
  per-well tail metrics, and no same-ID target recovery.
- Launched private Kaggle notebook version 1,
  `heralces/rogii-pf-public-reproduction`, as an attributed reproduction of the strongest
  currently identified public PF/trajectory baseline.
- Static audit found the exact public-reproduction source unsafe as a primary submission:
  same-ID demo paths, unverified late layers, nondeterministic PF kernels, and a projected
  7.5-10.5+ hour hidden runtime.
- Created and launched `heralces/rogii-pf-safe-v2` version 1. It disables same-ID/contact,
  visible-prefix, model-package, overlap-probe, and final-hedge layers; enforces query-well
  self-exclusion; and reduces both principal PF stages to 64 seeds. Estimated hidden runtime is
  3.5-5.5 hours.
- The attributed public-reproduction run completed in about 804 seconds and generated a valid
  14,151-row artifact. All three visible demo wells were overridden by same-ID contact paths,
  confirming that its apparent demo accuracy is contaminated; it was rejected for hidden
  submission. Safe-v2 remains the submission candidate.
- Safe-v2 completed its visible-data Kaggle run in about 525 seconds. The logs confirm that all
  guarded high-risk layers were skipped, and both native and independent submission audits
  passed with no errors or warnings. It scored 4.266 RMSE against the three overlapping demo
  truths without using same-ID lookup; this is only a sanity check.
- Submitted safe-v2 version 1 to the hidden code-competition rerun as submission reference
  `54927581`. It completed with public RMSE **7.514**, rank **2,004 / 5,604** in the
  2026-07-24 snapshot.

The same snapshot places the top-10% boundary near 6.518 and top-5% near 6.464. Safe-v2 is a
clean anchor rather than a medal candidate. Next gate: restore only audited high-signal
alignment/calibration layers under an eight-hour runtime budget, then submit a materially
different v3 candidate.

On 2026-07-24, a newly published notebook titled `ROGII Public 6.451 Base` was downloaded and
compared against our attributed public-reproduction notebook. All 45 code-cell sources are
byte-for-byte identical. Based on this new reproducibility evidence, the exact reproduction was
submitted as hidden candidate `54952634`; it is pending. The same-ID demo and runtime cautions
remain documented, and the independently modified safe-v3 line continues in parallel.

A later page-level audit corrected the attribution strength: the notebook's actual Kaggle
Current/Best score is **6.504**, not the `6.451` written in its title. The code identity remains
valid, but `6.504` is the only authoritative score evidence for that page. The same audit found
that the upstream notebook titled `ROGII Codex Public 6.40 Guard` actually scored **11.719** and
activated same-ID/exact-geometry public-demo paths on all 14,151 rows. It is therefore not a
6.40 result, is a strict raw-submission NO-GO, and supplies no score evidence for v4/v4b. No
additional public notebook passed the full-source and hidden-generalization audit.

Safe-v3 version 1 was then pushed as `heralces/rogii-pf-safe-v3`. It restores only bounded
visible-prefix calibration at 12/24 seeds and 250 particles, while keeping same-ID/contact,
overlap, global bias, model-package correction, and the late hedge disabled by default. Its
visible-data Kaggle run completed in about 623 seconds and passed the strict 14,151-row audit.
The target-free three-well diagnostic was 4.521 RMSE versus safe-v2's 4.266, so its hidden
submission was deferred; it remains a clean fallback rather than a medal candidate.

Guard640-v4 version 1 was also pushed as `heralces/rogii-guard640-v4`. It is the primary
target-free medal candidate: a sanitized, runtime-reduced derivative of the public 6.40 guard,
with 96-seed core PF, 12/24/250 visible-prefix calibration, 0.49/192 model-package gating, the
generalized branch hedge, and U-continuity. Same-ID and exact-overlap train-target paths are
removed. Its visible-data run completed and passed every submission/contract audit, but the
27.587-ft model-package disagreement p95 exceeded its borrowed 25-ft switch and disabled that
entire layer. It is therefore retained as a conservative control and will not be submitted as
the claimed public-6.40 composition.

An independent model-package audit then found that v4's borrowed 25-ft global disagreement
switch can disable the whole 0.49/192 correction at a normal observed p95 of 26.701 ft. A v4b
control restores only the public 6.40 source's `1e9` switch while leaving its bounded row-wise
gate, leakage controls, runtime guards, and dynamic sample contract unchanged. Version 1 of
`heralces/rogii-guard640-v4b` was pushed for an ordinary visible-data dry-run; it will not be
submitted to the competition until its outputs pass an independent audit. The run passed every
structural audit and selected the intended model package, but its post-hoc three-well RMSE jumped
from 3.552 for the clean anchor to 7.724 after that layer and ended at 7.566. V4b was therefore
rejected for hidden submission.

Stage-by-stage visible-data analysis also identified a separate clean signal: the exact public
reproduction's pre-contact, pre-correction 128-seed anchor scored 3.256 pooled RMSE, versus
safe-v2's 4.266 and safe-v3's 4.521. This is only a three-well diagnostic, but it supports
building a fully sanitized 128-seed anchor as a diverse candidate rather than inventing a fixed
blend or crediting contaminated downstream stages. That candidate passed static review with no
executable VP/model-package/hedge/U/contact/overlap cells and was pushed as private version 1 of
`heralces/rogii-pf-clean128-v5` for an ordinary visible-data run. It passed all structural and
runtime audits, but its pooled three-well diagnostic was 4.587, not the earlier exact-source
draw's 3.256. Its hidden submission was deferred pending a deterministic rebuild.

The otherwise scoring-identical v4/v4b normal runs also differed by 0.734 ft RMS at their clean
anchor stage. Static tracing found unseeded Numba particle-filter helpers in both the ridge and
learned feature paths. A separate deterministic clean128 build is being prepared with stable
per-well/per-branch seeds; the original v5 remains frozen so that this reproducibility fix is an
explicit ablation rather than an in-place change. The repaired notebook passed static checks for
all five random Numba kernels and 6,184 mounted well/branch seed pairs, then was pushed as
`heralces/rogii-pf-clean128-v5d` version 1 for the first of two ordinary reproducibility runs.

Clean128-v5d version 1 completed normally in about 692 seconds and passed the independent
14,151-row strict audit with no warnings or errors. Its three-well self-excluded diagnostic was
4.725 RMSE; as before, this small same-ID demo is recorded only as a regression diagnostic and
not used as hidden-test evidence. The byte-identical source was pushed as version 2 for the
decisive cross-run artifact comparison. Version 2 also passed its strict audit, and all four
stage/final CSVs were byte-identical across the two runs, with zero numeric RMS and maximum
absolute difference. The deterministic candidate therefore passed its predeclared gate and was
submitted as hidden reference `54954100`; three daily slots remained.

The separate frozen disaster-fallback study completed across all 773 training wells and
3,783,989 scored suffix rows with zero failures. Its predeclared trigger fired on 179 wells, but
both candidate actions made pooled RMSE worse (hold 12.671 and beam 12.757 versus baseline
12.519). That rule is a full-sample **NO-GO** and will not be threshold-tuned. A separate,
predeclared full-773 evaluation of the exact public U-continuity formula (`cap=8`, `tau=240`) was
launched without changing either parameter.

A frozen adaptive branch-selector study used only target-free diagnostics from the same 773-well
artifact and evaluated both shuffled-well and spatial-cluster OOF folds. The main core+Ridge
selector slightly worsened pooled RMSE to 12.523 and 12.527, respectively, with poor fold
consistency; hidden-GR features and ExtraTrees did not rescue it. This is another **NO-GO**, and
the selector will not be tuned further on the same held-out labels.

The deterministic sanitized Exact128-v6d full stack also completed one ordinary Kaggle run in
916 seconds and passed its independent 14,151-row audit. Balanced VP moved zero demo wells, and
the conservative model-package layer was correctly disabled because its 28.050-ft p95
disagreement exceeded the frozen 25-ft guard. The remaining branch hedge shifted one full well
by +2 ft and worsened the frozen post-hoc diagnostic from 4.702 to 5.168 RMSE. That exact final
composition is therefore a **hidden NO-GO**; it was neither repeated nor submitted.

The frozen U-continuity study then completed all 773 wells with zero failures. It improved
pooled RMSE only from 12.518489 to 12.518109 (delta -0.000380): statistically directional but
far below the predeclared -0.020 minimum. Its macro delta, paired-well bootstrap, and all five
spatial clusters passed, so magnitude was the sole failed gate. The final decision is still
**NO-GO**; `cap=8` and `tau=240` will not be retuned on these labels.

On 2026-07-25 both long code reruns completed. The exact public-family full
stack (`54952634`) scored **6.477**, while deterministic clean128
(`54954100`) scored **7.515**, essentially matching safe-v2's 7.514. The
5,653-team leaderboard snapshot places 6.477 at rank **437** (top 7.73%),
inside the public top-10% boundary of 6.498 but outside the top-5% boundary of
6.456. Exact full stack is therefore the primary candidate and a provisional
public bronze position; clean128 is a reproducible modeling NO-GO. No medal is
claimed before the private 74% leaderboard is finalized.

Two additional target-free OOF studies then finished without using leaderboard
feedback for selection. Four visible-prefix residual corrections all slightly
worsened the 478-well discovery set (+0.002 to +0.007 pooled RMSE), so the
reserved 295-well confirmation set remained untouched and the entire family
was rejected. A fixed residual CatBoost improved typewell-fingerprint grouped
OOF from 15.910 to 14.208, but spatial-cluster OOF improved only to 15.858,
with one cluster worsening by 1.408 and a bootstrap interval crossing zero.
That absolute-coordinate feature set is also a **NO-GO**; no Kaggle artifact
was built or submitted.

A separate spatial-invariant v2 then removed only the six absolute X/Y/Z and
anchor X/Y/Z features while retaining relative geometry. Typewell-fingerprint
OOF remained strong (15.910 to 14.476, all five folds improved). Aggregate
spatial performance improved more than v1: to 15.590 in the original
five-cluster split and 15.409 in a second-seed six-cluster split. However,
one regional fold still worsened by +0.839 and +1.677, respectively; the
five-cluster bootstrap interval also crossed zero and its well-level p95
worsened. Both frozen spatial gates therefore failed. Spatial-invariant v2 is
a **NO-GO** and was not submitted.

The next frozen v3 retained the spatial-invariant model but shrank its
predicted residual by exactly 0.20 and added an unscored seven-cluster
protocol. It passed all four OOF protocols: typewell delta -0.501 and spatial
deltas -0.358, -0.385, and -0.396; every fold improved and every bootstrap
interval was negative. A separate deployment qualification trained on 770
wells while excluding all three test IDs. On those three wells it improved
hold-last only from 11.539 to 11.333 RMSE, far behind the current 6.477
full-stack submission. V3 is therefore a validated residual component but a
**standalone submission NO-GO**.

An unchanged version-2 rerun of the exact public-family full stack then
completed in Kaggle. Its final 14,151-row `submission.csv` was byte-identical
to version 1 (SHA-256
`B192D3F348AE00680DC4DF942B95CEF5FD708C636A741F77DFB6B6E89B9DED4A`);
an independent audit reported zero errors and zero warnings. The duplicate was
not submitted.

A separate `rogii-pf-public-no-hedge` notebook kept all 58 parent cells
byte-identical and appended one final cell that promoted
`submission_before_branch_hedge.csv` to `submission.csv`. Its ordinary run
passed independent audit with zero errors and warnings. The output SHA-256 was
`FDF4A8175B6EC6A70C9B78FD6916AC3C317E43F7E9C08BBCA87CD02314801CA9`;
relative to the 6.477 file, only the 4,301 rows of `00e12e8b` changed, each by
-2.0 ft. The code-competition submission was accepted at
2026-07-25 04:37:05 UTC and completed with public RMSE **6.604**. Removing
the final +2 ft hedge therefore worsened the score by **0.127** versus the
current 6.477 best, falling outside the 6.498 public top-10% boundary. The
no-hedge ablation is a **NO-GO**; the exact full-stack 6.477 submission
remains the primary candidate.

## 2026-07-26

- Downloaded a fresh 5,716-team public leaderboard snapshot. The 6.477 exact
  full-stack candidate is rank **557 / 5,716** (top 9.75%). The current top-10%
  boundary is 6.478 at rank 571 and the top-5% boundary is 6.443 at rank 285.
  The public bronze margin is therefore only 0.001 and remains provisional.
- Reconstructed the public competitive Ridge/PF OOF baseline from five trainer
  artifacts and approximately 3.78 million suffix rows. A spatial-invariant
  residual-v3 overlay improved typewell-fingerprint OOF by only 0.0115, below
  the frozen 0.020 minimum, and worsened the seven-cluster spatial protocol by
  0.0464. The competitive overlay is a **NO-GO** and was not submitted.
- Audited the public `ROGII Ultra || Sub-6 RMSE` notebook. Its model-package
  correction was disabled by its own disagreement guard, and its final
  `submission.csv` SHA-256 was exactly the existing 6.477 artifact. It provides
  no new candidate despite the notebook title.
- Audited the public A28/Q0522 frontier run. Its run-local A28 candidate uses
  `0.62 * SP45 + 0.38 * learned PF-1.3`; Q0522 extends the direction of a
  dynamically detected PF branch by 0.522 ft, and the final public notebook
  averages the two candidates 50/50.
- Submitted the completed public A28/Q0522 pair Notebook output for controlled
  verification. It remains pending.
- Built private Notebook `heralces/rogii-frontier-a28-v1` version 1 by retaining
  the public source byte-for-byte through the A28 final audit and excluding the
  later Q0522/pair cells. Its 14,151-row output passed schema, ID-order,
  uniqueness, finiteness, file-hash, and prediction-hash checks. The A28-only
  code submission was accepted and remains pending.
- The public Dip-aware HMM + GBM notebook remains in a normal run. Its markdown
  claim of 5.14 is not backed by a printed or saved OOF result in the inspected
  source, so it is not eligible for submission unless the completed artifacts
  provide independently verifiable evidence.
- Both frontier code submissions reached a terminal Kaggle status without a
  score. Submission `54992035` (A28/Q0522 pair) and submission `54992736`
  (A28-only) each report that the hidden submission notebook exceeded the
  allowed runtime. Their ordinary visible-data runs were valid, but the hidden
  scale invalidates the current packaging. Both are runtime **NO-GO** results;
  neither is retried, and the 6.477 exact full stack remains the primary
  candidate.
- The public Dip-aware HMM + GBM run ended with
  `CANCEL_ACKNOWLEDGED` and exposes no output files. Its markdown-only 5.14
  claim remains unverifiable and supplies no submission candidate.

## 2026-07-27

- Downloaded a fresh 5,766-team public leaderboard snapshot. The 6.477 exact
  full-stack candidate is rank **653 / 5,766** (top 11.33%). The top-10%
  boundary is now 6.469 at rank 576 and the top-5% boundary is 6.436 at rank
  288, so the current candidate is outside the provisional public bronze zone.
- Audited public Notebook `prvsiyan/rogii-public-score-frontier-ii-visuals`.
  Of its 48 scoring code cells, 44 are byte-identical to the existing exact
  full stack. Its substantive changes disable two public-byte-inert runtime
  layers and add a signed 0.196-ft correction selected from the run-local PF
  branch report. The scoring transaction contains no fixed visible well ID or
  row count. Its pre-correction submission SHA-256 is exactly the current
  6.477 artifact; the ordinary public run completed in approximately 643
  seconds. The manifest's self-reported 6.39 is treated as unverified.
- Submitted Frontier II version 3 exactly once as a Code Submission with
  description `Frontier II runtime-fast DYNQ0196 controlled verification`.
  It entered `PENDING`; a four-hour monitor will not duplicate the submission.
- Built a private sequence-pilot cache from 3,783,989 official training suffix
  rows across all 773 wells. The cache has 51 spatial-invariant, target-free
  features, the competitive Ridge/PF OOF trajectory, truth labels, well
  boundaries, and frozen fold metadata. Absolute position and well ID are not
  model features. Float16 conversion recorded zero clipped values.
- Created private Kaggle Dataset
  `heralces/rogii-private-sequence-pilot-cache-v1`; downloaded metadata confirms
  `isPrivate: true` and status `ready`.
- Pushed private GPU Notebook `heralces/rogii-sequence-tcn-pilot-v1` version 1.
  It trains a six-block dilated TCN on spatial-fold training wells and evaluates
  the fixed 0.20 residual overlay on held-out spatial fold 0 after excluding
  matching Typewell fingerprints. The preregistered gate requires pooled delta
  at most -0.020 plus p95, worst-well, and well-win-rate safeguards. It writes
  only an OOF report and checkpoint and cannot create a competition submission.
  Version 1 was assigned a Tesla P100 (compute capability 6.0), but the current
  PyTorch 2.10 image supports only capability 7.0 and newer. It failed at the
  first Conv1D after approximately 25 seconds with `no kernel image`, before
  producing any training metric. The model and gate were left unchanged;
  version 2 explicitly requests `NvidiaTeslaT4` and is the sole retry. Kaggle
  normalized the active private Notebook ID to
  `heralces/rogii-sequence-tcn-pilot-v2`; its initial status is `RUNNING`.
- TCN v2 completed successfully on a Tesla T4 in 81.33 seconds. On 899,519
  rows from 170 held-out spatial-fold wells, the frozen 0.20 overlay improved
  pooled RMSE from 9.639581 to 9.584363 (delta -0.055218). Well-level p95 and
  worst-well RMSE also improved by 0.075164 and 1.150265, respectively.
  However, only 44.71% of validation wells did not worsen, below the
  preregistered 50% minimum. The report therefore returns `FAIL`,
  `full_multifold_authorized: false`, and
  `kaggle_submission_authorized: false`. No more folds or submission package
  are created from this version.
- Frontier II completed with public RMSE **7.507**. It is 1.030 worse than the
  6.477 exact full-stack best, 1.038 worse than the 6.469 public top-10%
  boundary, and only 0.007 better than Safe PF v2's 7.514. The public
  Notebook manifest's self-reported 6.39 is therefore not validated by the
  competition result. DYNQ0196 is a **NO-GO** and will not be retuned or
  allowed to replace the current best.
- Downloaded a later 5,785-team leaderboard snapshot. The 6.477 exact
  full-stack candidate is rank **701 / 5,785** (top 12.12%). The top-10%
  boundary is 6.467 at rank 579 and the top-5% boundary is 6.432 at rank 290.
- Scanned the latest-run public notebooks. The all-valid GBDT gate is another
  copy of the already rejected full-stack frontier, while two other candidates
  have OOF or visible-test errors far outside the competitive range. The only
  structurally new candidate was `luffyh04/harshini-submission-f`.
- Audited Harshini's 496.7 MB surface cache and 60.1 MB PF cache before
  restricted loading. The only pickle globals are NumPy array reconstruction
  primitives. The final model does not consume the notebook's oracle
  `EXACT datum`; its production surfaces do exclude the target well under
  ordinary LOO.
- Added a strict spatial-core audit that removes the full validation cluster
  from the formation point cloud. On fold 0, 173 wells and 910,183 suffix rows,
  flat scored 14.969, PF 8.598, the original single-well LOO core 8.008, the
  strict-spatial surface branch 30.278, and the strict surface/PF core 17.944.
  The new core is 2.975 worse than flat and 9.937 worse than its LOO estimate.
  The public notebook also required about 12,464 seconds on visible data.
  Harshini is therefore a spatial and runtime **NO-GO**; no boosted multi-fold
  rerun or Kaggle submission was authorized.
- Audited public Notebook `youill0317/rogii-mha180sep3-public-repro`. Its
  published transaction combines a run-local PF-bimodal midpoint adjustment
  with a global -0.40 correction decoded from a leaderboard canary and
  flag-gated probe cells. The canary correction, probes, and later leaderboard
  diagnostics were excluded from the private target-free build.
- Built and ran private GPU Notebook
  `heralces/rogii-mha180-target-free-v1` version 1. The retained midpoint rule
  uses alpha 1.8, minor mass 0.22, separation 3--40 ft, and a 4-ft cap. The
  run completed in approximately 889 seconds. Its final contract verified
  14,151 rows, exact sample ID order, unique IDs, finite predictions, and a
  report-bound submission SHA-256. It records no leaderboard canary or
  derived-bias use and no retained probe cells.
- All three wells with runtime PF statistics were already covered by the
  guarded overlap transaction, so the midpoint rule qualified **zero wells**
  and moved **zero rows**. The resulting submission SHA-256 is
  `fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9`,
  byte-identical to the existing no-hedge ablation that scored **6.604**.
  A duplicate Code Submission was therefore not created. MHA180 target-free
  is a **NO-GO**, and the 6.477 exact full stack remains the primary candidate.

## 2026-07-28

- Downloaded a 5,804-team public leaderboard snapshot. The 6.477 exact
  full-stack candidate is now rank **757 / 5,804** (top 13.04%). The public
  top-10% boundary is 6.461 at rank 580 and the top-5% boundary is 6.425 at
  rank 290.
- Merged the latest-run public Notebook list with leaderboard authors. Most
  new entries were attached to scores far behind 6.477. The strongest new
  lead was `mtoshidesu/rogii-test-another-approch-2nd`; its author currently
  has a 6.407 public score.
- The public A23 scoring tail was rejected because it hard-codes one visible
  test well, 4,301 rows, a historical output hash, and a reported public
  score. The upstream generic PF seed-branch selector is structurally usable:
  it selects branches only from run-local bimodal centers and masses. A clean
  derivative changes only its symmetric shift cap from 2.0 ft to 2.5 ft while
  preserving strength 0.60, minimum minor mass 0.25, and separation 4--40 ft.
- Added a provenance-locked builder and contract test. The generated scoring
  notebook contains no fixed target well, visible row count, expected output
  hash, or leaderboard score. It also replaces the fixed visualization demo
  well and disables the unrelated overlap dry-run probe. All 68 repository
  tests and Ruff pass.
- Pushed private GPU Notebook
  `heralces/rogii-pf-branch-cap25-targetfree-v1` version 1. It is running as
  an ordinary Kaggle job. It completed in 779.6 seconds. The dynamic contract
  passed: one run-local PF branch qualified, 4,301 rows moved by +2.5 ft, and
  the final SHA-256 exactly matched the independently downloaded public A23
  artifact. No fixed well, expected visible hash, leaderboard bias, or probe
  flag was present in the scoring code.
- Submitted Notebook v1 exactly once with description
  `Target-free PF seed-branch cap2.5 controlled verification`. It completed
  with Public RMSE **6.654**, which is 0.177 worse than the 6.477 primary
  candidate, 0.193 worse than the 6.461 public top-10% boundary, and 0.050
  worse than the 6.604 no-hedge ablation. The author's 6.407 score is not
  reproduced by this isolated target-free change. Cap 2.5 is a **NO-GO**;
  the branch cap will not be tuned further against the public leaderboard.

## 2026-07-29

- Downloaded a 5,878-team public leaderboard snapshot. The 6.477 exact
  full-stack candidate is rank **855 / 5,878** (top 14.55%). The public
  top-10% boundary is 6.452 at rank 587 and the top-5% boundary is 6.414 at
  rank 293. The primary candidate is 0.025 outside the current public
  top-10% line.
- Merged the latest 200 public Notebook runs with leaderboard authors and
  audited the strongest apparent leads. Outputs from `mitomeat823`,
  `odyssey189`, and `hityth` were byte-identical to the existing 6.477
  artifact. Several other high-ranked notebooks ended in the known Q0522
  fixed-visible-well family. GeoAnchor used a public-score quadratic inversion,
  a fixed visible well, and a fixed 4,301-row shift. None supplied a new,
  hidden-safe candidate.
- Added a frozen target-free well gate for the completed TCN fold-0 checkpoint.
  Its 110 well-level inputs summarize only the 51 target-free sequence
  features, the competitive base prediction, and the frozen TCN output.
  Absolute coordinates define five outer spatial subclusters but are not
  model features; duplicate Typewell fingerprints stay in the same fold.
  Every gate probability is out-of-spatial-fold.
- The gate selected 82 / 170 wells (48.24%), but selected-well win rate was
  only 51.22%, gated pooled delta was only **-0.006155**, and just two of five
  spatial folds improved. It failed the frozen -0.020 pooled, 55% selected-win,
  and four-of-five-fold gates. No multi-fold TCN training or submission was
  authorized.
- Audited the hidden runtime failure of the earlier A28 build. It used
  `1,000 particles x 256 seeds` instead of the hidden-proven exact core's
  `500 x 128`, then executed visible-prefix, model-package, and branch-hedge
  stages whose output A28 later overwrote. Built private Notebook
  `heralces/rogii-a28-runtime-fast-v1` by retaining the successful exact core
  only through SP45 and learned-trajectory generation, applying the frozen
  0.62 / 0.38 blend, and pruning those redundant post-blend stages.
- Ordinary A28 runtime-fast v1 completed in **590.3 seconds**. The dynamic
  contract passed: 14,151 rows, exact sample ID order, unique IDs, finite
  predictions, and matching hashes across the selection report, generic
  submission audit, final contract, named output, and `submission.csv`.
  Runtime scoring used no source score, leaderboard bias, probe, fixed test
  well, visible row constant, or expected output hash. Final SHA-256 was
  `9307dc0e30eda8bca81c4d03f5828da340d8d98e5585dda3981f312b12696033`.
- Submitted Notebook v1 exactly once with description
  `A28 runtime-fast exact-core PF1.3 w0.62 controlled verification`. It
  completed with Public RMSE **7.678**, which is 1.201 worse than the 6.477
  primary candidate, 1.226 worse than the 6.452 top-10% boundary, and 1.264
  worse than the 6.414 top-5% boundary. The public source's 6.433 was not
  reproduced. Runtime-fast A28 is a definitive **NO-GO** and cannot replace
  the current best.
- Rechecked the latest public outputs from `my0705` and `tamerlan`. Both are
  the same fixed-Q0522 artifact with SHA-256
  `8328487738fbb7e1e89c733b80367ced2281e6dd19f4746804adfd77a923071d`:
  the pre-Q0522 b192 input matches the current 6.477 candidate, then fixed
  visible well `00e12e8b` moves 4,301 of 14,151 rows. Neither output is a new
  hidden-safe candidate.
- Tested the low-cost global OOF scalar-bias hypothesis against every strict
  protocol. It worsened pooled RMSE by +0.013658 on `spatial_cluster`
  (1/5 folds improved), +0.017874 on `spatial_cluster_6` (2/6),
  +0.022172 on `spatial_cluster_7` (3/7), and +0.019451 on the
  Typewell-fingerprint protocol (1/5). The direction is rejected.
- Ran private Notebook `heralces/rogii-sequence-tcn-well-balanced-v1`
  version 1 exactly once as a no-submission pilot. Its report and checkpoint
  exist, no `submission.csv` was written, and the frozen contract passed:
  uniform well sampling, `spatial_cluster_7_seed_20260727`, fold 0, primary
  scale 0.20, 4,000 steps, and `writes_submission=false`. Tesla T4 runtime
  was 83.58 seconds.
- The uniform-well TCN changed pooled RMSE from 9.639581 to 9.689167,
  delta **+0.049586**, versus **-0.055218** for the original length-weighted
  pilot. At scale 0.20 its p95 well delta was +0.104819, worst-well delta
  was -0.774178, and only 44.71% of wells were non-worse—the same inadequate
  coverage as the original pilot. The pooled, p95, and at-least-half-wells
  checks failed; only the worst-well check passed. The report decision is
  **FAIL**, with `full_multifold_authorized=false` and
  `kaggle_submission_authorized=false`. No scale/step changes, multi-fold
  training, or competition submission are authorized.

## 2026-07-30

- Isolated the dormant strongest-periodic-component GR notch from the public
  full-stack source. Before reading any target-scored notch output, froze
  minimum 64 finite rows, one-bin radius, the unchanged eight-seed target-free
  predictor, and the existing pooled/macro/bootstrap/five-spatial-cluster
  deployment gate. Commit `841a7eb` records the implementation and contract
  before the full result.
- The six-process run completed all 773 wells with zero failures in 716.7
  seconds. The notch activated for every well and changed pooled RMSE from
  12.518489 to 16.811012, delta **+4.292523**. Macro-well RMSE worsened by
  +4.343344; 225 wells improved and 548 worsened.
- The paired-well bootstrap 95% interval was entirely adverse at
  `[+3.069503, +5.542658]`. All five spatial clusters worsened, by +2.9535
  through +5.3763. Every preregistered gate check failed. The GR notch is a
  definitive **NO-GO** and does not authorize GPU work, a Notebook, or a
  competition submission. Its frequency, radius, and activation rules will
  not be retuned.
- Inspected the public model-package outputs before opening another route.
  Their learned trajectory already includes multiple `pred_delta_drift_ncc_*`
  tree-model tracks, so direct multi-scale GR-NCC is not an independent
  candidate. No duplicate NCC evaluation, Notebook, or submission was made.
- Preregistered one structural audit of the five published grouped-OOF model
  vectors in commit `1e7e360`. The candidate collapses the three LightGBM
  columns to one family mean and the two CatBoost columns to another, then
  keeps the existing positive Ridge, 0.91 / 0.09 Ridge/PF blend, and 85-ft
  depth ramp unchanged. Typewell and seven-cluster spatial slices are
  explicitly secondary robustness evidence, not nested base-model refits.
- The CPU audit covered 3,783,989 rows from all 773 wells in 51.2 seconds.
  Family balancing changed pooled RMSE from 10.370056 to 10.399843, delta
  **+0.029786**; macro-well delta was +0.001060. Outcomes were nearly split
  at 386 winning and 387 losing wells, but the bootstrap 95% interval
  `[-0.007816, +0.067396]` did not exclude harm.
- Only 2 / 5 Typewell partitions and 3 / 7 spatial partitions were
  non-worse. Maximum partition delta was +0.103686 and p95-well delta was
  +0.148644; only the +0.433701 worst-well guard passed. The other seven
  frozen checks failed. Family balancing is a definitive **NO-GO**: no
  weight/alpha/ramp retuning, GPU work, Notebook, or competition submission
  is authorized. The 6.477 exact full stack remains the primary candidate.

## 2026-07-31

- Downloaded and audited the latest close public candidate,
  `farhanabidtech786/rogii-best-score-wellbore-geology-prediction`. Its final
  `submission.csv` SHA-256 is
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`,
  byte-identical to the current 6.477 primary in 45 existing artifact copies.
  Its source is the same public full-stack and final branch hedge, not a new
  independent algorithm. No mirror Notebook or duplicate submission was made.
- Refreshed the 5,998-team leaderboard snapshot: 6.477 is rank 1,060
  (17.67%), while the top-10% and top-5% boundaries are approximately 6.429
  and 6.386. The Public board remains only the 26% slice, so this is not a
  claim about the 74% Private result.
- Used the authenticated Kaggle API/CLI to freeze exact submission identity.
  The 6.477 primary is ref `54952634`, Notebook
  `heralces/rogii-pf-public-reproduction` version id `337405234`; the 6.604
  no-hedge backup is ref `54967104`, Notebook
  `heralces/rogii-pf-public-no-hedge` version id `337779623`. Both Notebooks
  and all nine existing competition submissions are terminal/complete, so no
  new submission or Notebook push is required.
- Revalidated both outputs against the official sample: 14,151 rows, exact
  `id,tvt` columns and ID order, unique IDs, and finite predictions. Primary
  SHA is `b192d3...ded4a`; no-hedge SHA is `fdf4a817...01ca9`. Exactly 4,301
  rows differ, each by primary-minus-no-hedge = +2.0 ft, for an all-row RMSE
  distance of 1.102608.
- The API currently reports **zero selected final submissions**. The frozen
  recommendation is primary `54952634` plus branch-risk hedge `54967104` if
  the live UI permits two, or primary `54952634` alone if it permits one.
  Safe-v2 ref `54927581` (7.514) remains a runnable structural disaster
  fallback but is not recommended over the closer no-hedge candidate.
  Selection preflight is due by August 2 and live selection by August 5
  20:00 UTC, leaving about four hours before the official deadline.

## 2026-08-02

- Completed the scheduled read-only final-selection preflight. The
  authenticated API still reports zero selected submissions, and the signed-in
  Kaggle Submissions page independently displays **0/2**.
- The page explicitly states that up to two submissions may count toward the
  final leaderboard and that Kaggle will automatically choose from the
  best-scoring submissions if fewer than two are selected. The final allowance
  is therefore confirmed rather than inferred from other competitions.
- Reconfirmed the official deadline `2026-08-05 23:59 UTC`, five daily Code
  Submissions, and `COMPLETE` status for the primary, no-hedge backup, and
  safe-v2 disaster fallback Notebooks. The repository worktree is clean.
- The manual selection recommendation is now unconditional: ref `54952634`
  (6.477 primary) plus ref `54967104` (6.604 branch-risk hedge). No checkbox
  was changed during this read-only preflight; live selection remains due by
  August 5 20:00 UTC.
- After explicit user confirmation later the same day, manually selected those
  two submissions. The Kaggle page displayed **2/2**, and both target rows were
  checked. An independent authenticated API read returned exactly refs
  `54952634` and `54967104`, both `COMPLETE`; no third ref was present.
- Final selection is therefore complete more than three days before the
  official deadline. No new Notebook or competition submission was created.
  Remaining work is limited to low-frequency read-only integrity checks unless
  genuinely new evidence requires a user decision.
- Reopened the rank-improvement sprint after explicit user direction. A fresh
  6,055-team leaderboard snapshot places 6.477 at rank 1,159 (top 19.14%),
  with top-10% and top-5% boundaries of 6.419 and 6.374.
- Audited `raunakdey07/rogii-physics-informed-stacked-ensemble-v2`. Although
  the notebook contains a long MDN/GBDT experiment, its final submission is
  exactly the current primary plus 2.0 ft on the same 4,301-row PF branch and
  has known Public RMSE 6.432. The file SHA is
  `af43d97ab5b76f7f0bbbc5c72fb6e9a829e939c48582fc4130144340547360a2`.
- Built a private dynamic target-free reproduction. Version 1 correctly failed
  the pre-submission equivalence gate because strength 0.60 produced a
  3.908582 ft shift. Version 2 changed only generic branch strength to 0.62,
  reached the 4.0 ft cap, completed in 785.8 seconds, and passed all sample,
  ID, finite-value, report, delta, and SHA checks. Its output is byte-identical
  to the known 6.432 file. Candidate selection is recorded as Public-informed;
  the runtime path does not read fixed test constants, probes, or scores.
- Created exactly one Code Submission, ref `55177746`, description
  `Target-free PF seed-branch cap4.0 controlled verification`. It is currently
  `PENDING`; the existing `54952634 + 54967104` final selection remains intact.
  If the new ref scores strictly below 6.477, recommend replacing only
  `54967104`, subject to a final user-confirmed selection change.
- Corrected the cap-4 evidence boundary: its visible-run CSV is byte-identical
  to a public 6.432 artifact, but Code Competition scoring reruns the Notebook
  on hidden data. Identical visible SHA therefore does not guarantee identical
  Public scoring behavior; ref `55177746` remains the only authoritative test.
- Implemented a structurally independent well-functional residual family on
  the 773-well, 3,783,989-row target-free sequence cache. Each well uses five
  relative-depth samples of 51 invariant features (255-dimensional signature)
  to predict a 64-point residual curve over the competitive grouped-OOF base.
- The preregistered 16-neighbor curve transfer failed both protocols: spatial
  7-fold pooled delta was +0.008253 and Typewell-fingerprint 5-fold delta was
  +0.028481. It is frozen as NO-GO without neighbor or distance tuning.
- The one allowed structurally different follow-up, multi-output Ridge with
  alpha 100, scale 0.20, and residual clip 20, passed both frozen protocols.
  Spatial 7-fold delta was **-0.027603** with 5/7 folds improving, p95 delta
  -0.07058, and worst-well delta +0.19252. Typewell 5-fold delta was
  **-0.018767** with 5/5 folds improving, p95 delta -0.05606, and worst-well
  delta +0.01616.
- Added the reproducible evaluator, tests, and private deployment builder.
  Pushed `heralces/rogii-functional-ridge-v1` version 1 exactly once. The new
  target-free layer runs after the parent PF branch and before the final audit;
  it records that the inherited 6.477 deployment base is related to but not
  exactly the OOF gate base. The Notebook is `RUNNING`; no competition
  submission exists until its dynamic output and cross-file SHA contract pass.
- Preregistered one low-rank response ablation before reading its output. Each
  outer fold fit an 8-component PCA only on training-well residual curves, then
  predicted those coefficients from the same target-free signatures. It had
  to beat the plain Ridge pooled delta by at least 0.002 in both protocols.
- The eight components explained roughly 96.9%--97.1% of training-curve
  variance, but spatial pooled delta was -0.027280 and Typewell delta was
  -0.018631. These were respectively 0.000323 and 0.000136 worse than the
  plain Functional Ridge. Both dominance checks failed, so the low-rank route
  is a definitive NO-GO without PCA-dimension, alpha, scale, or clip tuning.
- Prepared, but did not push, a conditional cap4 plus Functional Ridge builder.
  Its frozen authorization requires both individual Kaggle candidates to score
  strictly below 6.477 and pass their dynamic contracts. The combined runtime
  records Public-informed candidate selection but reads no score, fixed test
  well, visible row count, or expected hash. Until both gates resolve true it
  remains `PREPARED_NOT_AUTHORIZED` and consumes no Kaggle runtime or submission.
- Functional Ridge Notebook version 1 failed after 766.0 seconds at the first
  overlay contract check. Kaggle logs showed that the full parent stack,
  private cache mount, source bootstrap, PF branch report, and parent
  `submission.csv` all completed; no Functional Ridge model was fitted and no
  competition submission was created.
- Root cause was a metadata-path bug: the cache certifies
  `absolute_position_features_removed` inside `metadata["contract"]`, while v1
  checked the top level. The fix reads the nested contract and additionally
  requires `training_rows_only=true`, `target_feature_leakage=false`, and
  `well_id_encoded_as_feature=false`. Parameters, OOF evidence, and prediction
  logic are unchanged. Tests cover the exact nested lookup. Version 2 was then
  pushed once to the same private Notebook and is `RUNNING`; no competition
  submission has been created.
- An equivalent local dry run used the failed v1 parent's generated
  `submission.csv`, the official test files, and the same private cache. The
  corrected nested contract passed; all 14,151 IDs were covered exactly once,
  all rows changed with finite corrections, RMS was 0.409577 ft, and the range
  was [-1.155607, +0.669601] ft. This removes the known static deployment risk
  while leaving the Kaggle v2 artifact audit as the authoritative next gate.
- Refreshed the 6,056-team leaderboard and latest 200 public Notebooks. Georgy
  Mamarin's 6.320-team Notebook is an analysis-only noise-floor study with no
  submission. Evgendvorkin's 6.405-team output is byte-identical to the cap4
  visible artifact. Yaroslav's 6.419-team Notebook adds a different final file,
  but its transaction fixes visible well `00e12e8b`, 4,301 rows, and a desired
  1.75-ft shift before scaling the other wells.
- Isolated only Yaroslav's target-free GR/Typewell raw datum mechanism. Before
  evaluation, froze a no-intercept, hidden-row-weighted scale fitted on outer
  training wells only, scale guard +/-0.25, and correction cap +/-2.5 ft. No
  fixed test well, visible row count, expected hash, or score entered runtime.
- Raw datum construction succeeded for all 773 wells, but learned scales
  repeatedly changed sign. Spatial 7-fold pooled delta was **+0.008490** with
  0/7 folds improving. Typewell 5-fold delta was **+0.033743** with 0/5 folds
  improving, p95 delta +0.058724, and worst-well delta +0.389703. The clean
  transfer is a definitive NO-GO; no grid, smoothing, scale, or cap tuning and
  no Notebook/submission are authorized.

### 2026-08-02 - Functional Ridge v2 artifact gate passed and submitted once

- Private Notebook `heralces/rogii-functional-ridge-v1` version 2 completed in
  846.0 seconds after the nested cache-contract fix.
- Downloaded and audited the complete output. `submission.csv` has 14,151 rows,
  exact `id,tvt` columns, official ID order, unique IDs, and finite predictions.
  Its SHA-256 is `a30f6384b4d6f06c2d6f6330bb0b7cfe7997bb0267c129d2b0f081828d5180de`,
  matching both `functional_ridge_deploy_report.json` and
  `submission_audit.json`; the parent file SHA and all three per-well delta
  summaries also match the final file.
- Confirmed the target-free runtime flags: no absolute coordinates, fixed test
  well, visible row count, expected visible hash, leaderboard score, probe
  flag, or encoded well ID was used. The cache is training-row-only.
- Created exactly one Code Submission from Notebook version 2, ref `55178775`,
  description `Functional Ridge OOF overlay s0.20 controlled verification`.
  It is currently `PENDING`. The cap4 ref `55177746` is also still `PENDING`;
  neither candidate changes the current best or selected final pair until a
  strictly better terminal score exists.

### 2026-08-02 - Nonlinear Functional Kernel Ridge failed both protocols

- Before evaluation, preregistered one RBF Kernel Ridge follow-up using the
  exact Functional Ridge signatures and residual curves. Every outer fold fit
  imputation, scaling, the inverse-median-distance bandwidth, and kernel model
  on training wells only. Alpha 10, scale 0.20, clip 20, and a required 0.005
  dominance margin over plain Ridge were frozen in advance.
- Spatial 7-fold pooled delta was **+0.000323**, only 4/7 folds improved, and
  p95-well delta was +0.03008. Typewell 5-fold pooled delta was **+0.005069**,
  only 1/5 folds improved, and p95-well delta was +0.08637.
- Both protocols worsened and missed the plain Ridge reference by more than
  0.0238 RMSE. This is a definitive NO-GO: no kernel alpha, bandwidth, scale,
  signature, or clip tuning; no Notebook or competition submission.

### 2026-08-02 - Globally augmented Functional Ridge was not Typewell-stable

- Preregistered one linear follow-up that appended per-feature whole-well mean,
  standard deviation, and least-squares relative-position slope to the five
  fixed depth samples. Ridge alpha 100, scale 0.20, clip 20, and a required
  0.003 dominance margin over plain Ridge were frozen before evaluation.
- Spatial 7-fold improved strongly: pooled delta **-0.037769**, 5/7 folds,
  p95-well delta -0.15216, and 0.010166 better than plain Ridge.
- Typewell 5-fold pooled delta was only **-0.014021** with 4/5 folds improving.
  It was 0.004746 worse than plain Ridge and missed the all-fold requirement,
  despite favorable p95 and worst-well deltas.
- The candidate is a preregistered FAIL. Whole-well statistics contain useful
  spatial signal, but this exact representation is not stable across Typewell
  families. No statistic deletion, alpha/scale adjustment, Notebook, or
  submission is allowed.

### 2026-08-02 - Supervised Functional PLS improved pooled OOF but missed one fold

- Preregistered eight-component PLSRegression on the exact augmented signature,
  with components, scaling, and response fit isolated to each outer-training
  fold. Scale 0.20, clip 20, and a 0.003 dominance margin over plain Ridge were
  fixed before evaluation.
- Spatial 7-fold passed strongly: pooled delta **-0.050250**, 6/7 folds,
  p95-well delta -0.04365, worst-well delta +0.22428, and 0.022647 better than
  plain Ridge.
- Typewell pooled delta was **-0.028257**, p95-well delta -0.06047, worst-well
  delta -0.09963, and 0.009490 better than plain Ridge. However, fold 0 worsened
  by +0.028927, leaving only 4/5 folds improved.
- Standalone PLS is therefore a FAIL under the mandatory all-Typewell-fold gate.
  It will not be submitted or tuned. Plain Ridge improved all five Typewell
  folds, so one separate conservative fixed-weight Ridge/PLS ensemble may be
  preregistered; no weight grid or PLS parameter change is allowed.

### 2026-08-02 - Fixed 75/25 Functional Ridge/PLS ensemble passed every gate

- Before evaluation, froze a single conservative curve ensemble: 75% of the
  plain Ridge prediction and 25% of the supervised PLS prediction. No weight
  grid was searched. Deployment requires at least 0.002 pooled improvement over
  plain Ridge in both protocols plus every existing stability gate.
- Spatial 7-fold pooled delta was **-0.035209**, 5/7 folds improved, p95-well
  delta was -0.06408, worst-well delta +0.20037, and the ensemble beat plain
  Ridge by 0.007606.
- Typewell 5-fold pooled delta was **-0.023059** and all 5/5 folds improved.
  P95-well delta was -0.05563, worst-well delta -0.01282, and the ensemble beat
  plain Ridge by 0.004291. Every preregistered gate passed.
- Built the private Notebook `heralces/rogii-functional-ridge-pls-v1` from the
  already audited Functional Ridge v2 parent. The new layer remains target-free,
  uses no fixed test well/row count/hash, score, probe, or well identity, and
  records both component weights and that no grid was searched.
- A full local deployment dry run on the 773-well cache and official three-well
  test contract passed: 14,151 exact ordered unique IDs, finite predictions,
  all rows changed, correction RMS 0.36753 ft, range [-1.04282, +0.54504] ft,
  and final SHA-256
  `d4d7e368c940d84cdc1504fcad18809675f18499b0efd6e089011b21febb7da3`.
  Exactly one private Notebook run is authorized; a competition submission is
  allowed only after the hidden-run output independently passes the same audit.
- Pushed `heralces/rogii-functional-ridge-pls-v1` version 1 exactly once. It is
  currently `RUNNING`; no competition submission exists yet. The monitor now
  owns the hidden artifact audit and may create only the single preregistered
  Code Submission after every dynamic contract check passes.
- Prepared and tested, but did not push, a conditional cap4 plus fixed 75/25
  Ridge/PLS builder. It inherits both exact components and adds cross-report
  SHA, cap, weight, and no-grid checks. Authorization requires cap4 ref
  `55177746` and the future standalone Ridge/PLS submission to each score
  strictly below 6.477 and pass their dynamic contracts. It is explicitly
  `PREPARED_NOT_AUTHORIZED` and consumes no Kaggle runtime or submission now.

### 2026-08-02 - Live leaderboard and Raunak incremental public audit

- Refreshed the full 6,057-team leaderboard: current rank is 1,161 at 6.477;
  top-10% remains 6.419 at rank 606 and top-5% remains 6.374 at rank 303.
- Audited the newly rerun public Notebook
  `raunakdey07/rogii-physics-informed-stacked-ensemble-v2`, whose author team
  is at 6.432. Its final `submission.csv` SHA-256 is exactly the existing cap4
  visible artifact `af43d97a...60a2`; the branch report again moves only
  `00e12e8b`, 4,301 rows, by +4.0 ft.
- The added visualization appendix contains fixed visible row/well presentation
  constants but does not modify the final file. This is not a structurally new
  candidate and will not be wrapped, rerun, or submitted again.

### 2026-08-02 - Target-free distance-gated Ridge/PLS missed one Typewell fold

- Preregistered one training-support gate: PLS weight 0.50 inside the outer
  training distribution and 0.25 outside it. OOD was defined by the 95th
  percentile of training-well leave-one-out nearest-neighbor distance after
  training-only imputation and scaling. No weight, distance, or quantile grid
  was searched; validation truth was not available to the gate.
- Spatial pooled delta was **-0.039708**, 5/7 folds, and 0.004499 better than
  the fixed 25% PLS ensemble. Typewell pooled delta was **-0.026327**, also
  0.003268 better, with favorable p95 and worst-well deltas.
- Typewell fold 0 nevertheless worsened by +0.004770, leaving only 4/5 folds
  improved. The candidate is a preregistered FAIL and will not be tuned or
  deployed. The fixed 75/25 ensemble remains the only authorized model.

### 2026-08-02 - Ridge/PLS hidden artifact passed and was submitted once

- Private Notebook `heralces/rogii-functional-ridge-pls-v1` version 1 completed
  in 771.9 seconds. Downloaded the full output and validated 14,151 rows, exact
  `id,tvt` schema and official ID order, unique IDs, finite predictions, all-row
  coverage, both source SHA values, the fixed 0.75/0.25 weights, no weight grid,
  target-free/cache flags, per-well deltas, and report/audit/final SHA agreement.
- The authoritative hidden output SHA-256 is
  `ffeb394ecc60ef4a95215b665baad9397119652e93505484806dc3c4bfdf482c`;
  its parent SHA is the exact 6.477 primary
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`.
- The earlier local dry-run SHA differed because its input was already a
  once-serialized parent CSV and the overlay performed a second pandas CSV
  round trip. Hidden versus local correction statistics differed by at most
  0.00000251 ft, while source, model, weights, IDs, and dynamic contracts were
  identical. This is serialization-level evidence, not prediction drift.
- Created exactly one Code Submission, ref `55179387`, description
  `Functional Ridge PLS fixed75-25 OOF controlled verification`. It is
  `PENDING`; two daily submission slots remain. Current best and the manually
  selected final pair remain unchanged until a strictly better score exists.

### 2026-08-02 - Nonlinear Functional ExtraTrees was informative but failed

- Before evaluation, froze one multi-output ExtraTrees model with 384 trees,
  `max_features=0.50`, `min_samples_leaf=8`, no bootstrap, no hyperparameter
  search, the augmented target-free well signature, and the existing 0.20
  residual-curve overlay. The preregistration was pushed as commit `394cc2b`.
- Spatial 7-fold pooled delta was **-0.027884**, with 6/7 folds improved and
  favorable p95/worst-well tails, but it was 0.007324 worse than the fixed
  Ridge/PLS reference and therefore failed the dominance gate.
- Typewell 5-fold pooled delta was **-0.026643** and all 5/5 folds improved,
  beating Ridge/PLS by 0.003585. Its p95-well delta nevertheless worsened by
  +0.013422, failing the mandatory tail gate.
- The candidate is a preregistered FAIL. Its tree parameters will not be tuned,
  it will not be blended after observing these results, and no Notebook or
  competition submission is authorized.

### 2026-08-02 - Late public Notebook audit found no clean new candidate

- Audited three newly run public Notebooks. `hellbore-v-6` and
  `rogii-geologia-v92-geographic-restoration` produced no submission artifact
  or executable source result, so neither provides testable candidate evidence.
- `akiirolabs-tvt-prediction-model` produced a distinct final SHA, but its final
  stage found exact coordinate/prefix matches for all three visible test wells
  and replaced all 14,151 predictions using their corresponding training-well
  TVT trajectories. The audit itself records zero prefix RMSE, three exact
  overlap wells, and 14,151 overridden rows.
- That output is same-well target leakage, not hidden-generalization evidence.
  The earlier SP45/Fleongg branches are already represented by audited public
  model-package routes. Nothing from this scan is eligible for a private
  Notebook or submission.

### 2026-08-02 - Sequential Ridge residual-PLS did not beat the fixed ensemble

- Preregistered a distinct sequential model before evaluation: plain Ridge
  predicts the residual curve first, then an eight-component PLS model learns
  only the remaining outer-training curve residual from the augmented
  target-free signature. The second-stage weight was fixed at 0.25 with no
  parameter or weight grid; preregistration was pushed as commit `73a9039`.
- Spatial 7-fold pooled delta was **-0.029618**, with 5/7 folds improved and
  favorable p95 tail, but it was 0.005590 worse than the fixed Ridge/PLS
  ensemble.
- Typewell 5-fold pooled delta was **-0.019420**, 0.003639 worse than the fixed
  ensemble, and fold 0 worsened by +0.003062, leaving only 4/5 folds improved.
- This sequential candidate is a preregistered FAIL. Its stage weight and PLS
  configuration will not be tuned, and no Notebook, blend, or competition
  submission is authorized.

### 2026-08-02 - Target-free dip-rate HMM discovery passed decisively

- Isolated the only structurally new idea from the unaudited public Dip-Aware
  HMM claim: a joint hidden state over typewell TVT position and local
  `d(TVT+Z)/dMD` rate. Reimplemented it as a target-free forward filter and
  removed the public same-well/contact overrides, pretrained submission lookup,
  meta learner, score claims, and submission path.
- Before evaluation, froze a 32-well discovery set from spatial fold 0 by
  SHA-256 well ordering, a 0.5-ft position grid, 21 rate states, fixed
  transitions, and a 0.20 overlay of HMM/base disagreement clipped to +/-5 ft.
  Thus the final movement cannot exceed +/-1 ft. Preregistration was pushed as
  commit `efb3d68`.
- Discovery pooled RMSE improved from **8.590338 to 8.343511**, a delta of
  **-0.246827**. 28/32 wells improved (87.5%); p95-well delta was -0.34618 and
  the worst-well aggregate tail improved by -0.41994. Mean runtime was 10.9
  seconds per well. Every frozen quality, tail, win-rate, and runtime gate
  passed by a wide margin.
- The discovery set does not authorize a Notebook or submission. It authorizes
  one unchanged confirmation run over all 741 untouched training wells, scored
  under both spatial7 and Typewell5 protocols. Confirmation parameters and
  gates are frozen before those labels are inspected.

### 2026-08-02 - Dip-rate HMM confirmation gained strongly but failed tail gates

- Ran the unchanged discovery model on every one of the 741 untouched official
  training wells. Eight-thread parallelism changed runtime only; the model,
  state grid, transitions, overlay, and clip were imported unchanged.
- Confirmation pooled RMSE improved from **10.449281 to 10.292524**, a large
  **-0.156756** delta. Every spatial fold improved (7/7), from -0.06032 to
  -0.28267, and every Typewell fold improved (5/5), from -0.04541 to -0.24128.
  A later audit found that the recorded `p95_well_delta=-0.12914` was actually
  the 5th percentile mislabeled as p95. The true 95th percentile is adverse
  (about +0.628 after excluding the three current test-well IDs). This does not
  change the already-FAIL decision or the strong pooled/all-fold evidence.
- Nevertheless, only 63.83% of wells improved versus the frozen 65% minimum,
  and the worst-well aggregate tail worsened by +0.68363 versus the +0.25 cap.
  Those are explicit confirmation failures even though the pooled signal is
  much stronger than every earlier residual experiment.
- The confirmation decision is therefore **FAIL**. Because all 773 wells have
  now contributed either discovery or confirmation evidence, building a new
  uncertainty/well gate after seeing these labels would be post-hoc. The HMM
  will not be tuned, gated, blended, packaged, or submitted. This preserves the
  untouched-confirmation contract rather than converting a promising average
  into an unaudited leaderboard risk.

### 2026-08-02 - Rank objective authorized one disclosed HMM public validation

- Re-audited the HMM evidence against the actual competition objective. The
  official metric is pooled row-level RMSE; the untouched pooled gain was
  -0.156756, every one of 12 group folds improved, and the final correction is
  hard-capped at +/-1 ft. Those primary-metric facts are materially stronger
  than the failed auxiliary 65% well-win and worst-well gates.
- Recorded an explicit high-risk exception rather than rewriting the earlier
  confirmation result. The confirmation remains FAIL, the HMM parameters are
  unchanged, and no post-hoc uncertainty/well gate is introduced. Exactly one
  private Notebook run and, after hidden artifact audit, one public controlled
  submission are authorized. Current final selections remain protected.
- Built `heralces/rogii-dip-rate-hmm-v1` from the exact 6.477 parent. Notebook
  runtime code is extracted from the confirmed evaluator; only Numba disk
  caching is disabled because dynamic Notebook cells have no cache locator.
- The full local three-well deployment dry run passed: dynamic 14,151-row
  coverage, exact sample ID order, unique IDs, finite values, all rows changed,
  correction range exactly [-1,+1] ft, report/final SHA agreement, and 41.8 s
  runtime. The local final SHA is
  `bc2be8533d138706217d7ef3ba6882ff5a9a085cc5298af094fdbab228a82c80`;
  it is an offline reproduction check, never a runtime selector.

### 2026-08-02 - Scale-gate audit and rank-seeking HMM scale2 confirmation

- Tested a preregistered target-free per-well scale gate using only row count,
  correction RMS, and HMM uncertainty. Every current test-well ID was excluded
  from label-based fitting/evaluation. The spatial protocol still selected the
  un-gated scale1 correction, and the gate failed true-p95, worst-well, and
  well-win checks. This route is closed and will not be submitted.
- Audited the fixed correction's exact quadratic SSE curve. Across the 738
  non-test wells, the pooled optimum was about 2.76 times the current scale1
  correction, and every spatial/Typewell fold optimum exceeded 1.0. Frozen
  multiplier 2.0 (overlay scale 0.40, maximum movement +/-2 ft) was then
  preregistered rather than selecting the pooled optimum.
- Scale2 passed its primary rank gates: pooled delta **-0.244037**, spatial
  **7/7** folds improved, Typewell **5/5** folds improved, and it beat scale1
  by another **0.087966** RMSE. Tail risk increased (true p95 +1.3359,
  worst-well +1.8142, 57.86% wells improved), so the candidate is explicitly
  high risk and cannot alter final selection without a Public score strictly
  below 6.477.
- Built an independent private Notebook contract
  `heralces/rogii-dip-rate-hmm-scale2-v1`. It reuses the identical target-free
  HMM state path, changes only overlay scale 0.20 -> 0.40 and movement cap
  1 -> 2 ft, contains no fixed visible IDs/row counts, and has a dedicated
  hidden-output validator.
- The original scale1 Notebook completed in about 783 seconds. Downloaded
  output passed the strict validator: 14,151 dynamic rows, exact official ID
  order, unique IDs, finite values, all rows changed, maximum movement 1 ft,
  and matching before/final/report/audit SHA. Created its sole Code Submission,
  ref `55180816`, which is pending.
- Pushed scale2 private Notebook version 1 exactly once. It is running; the
  final daily submission slot is reserved for it only after hidden artifact
  validation passes. Neither HMM candidate changes the manually selected final
  pair automatically.
- Scale2 completed in about 766 seconds and passed its dedicated hidden-output
  validator: 14,151 rows, exact official ID order, unique/finite predictions,
  all rows changed, maximum movement exactly 2 ft, parent SHA
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`,
  and final/report/audit SHA
  `4846486289f4a105dde3ba12e662640c6066ba2097aa1cf74475bdebf776bd75`.
  Created its only Code Submission, ref `55181026`; it is pending. The daily
  quota is now 0, so no further candidate can be submitted today.
- Before either HMM Public score was visible, preregistered multiplier 2.3 from
  the weakest non-test fold's zero-gain boundary (~2.354). It passed the frozen
  primary gates: pooled delta **-0.256847**, spatial 7/7 and Typewell 5/5 folds
  improved, and another 0.012810 gain over scale2. Tail risk rose further
  (true p95 +1.5538, worst +2.0891, well win 56.23%), and the weakest Typewell
  fold improved by only 0.003804; therefore no multiplier above 2.3 is allowed.
- Built scale2.3 as a local-only Notebook/validator with
  `execution_authorized=false`. It may be pushed only if scale2 ref `55181026`
  first scores strictly below 6.477. This preserves a pre-Public continuation
  without spending GPU or a future submission on a failed direction.

### 2026-08-02 - HMM scale2 plus fixed Ridge/PLS additive OOF passed

- Before any of the five pending Public scores was available, preregistered one
  additive combination of the two unchanged target-free components: HMM scale2
  first, then the fixed 75/25 Functional Ridge/PLS overlay. There was no scale,
  weight, clip, state, or order grid.
- Excluded all 32 HMM discovery wells and every current test-well ID from the
  738-well evaluation. The Functional model stayed isolated to each outer
  training fold under both spatial7 and Typewell5 protocols.
- Spatial pooled RMSE improved from HMM2's 10.215544 to **10.178767**, an
  additional **0.036777** gain; 6/7 folds improved. Typewell improved from
  10.215544 to **10.186974**, an additional **0.028570** gain; all 5/5 folds
  improved. Total deltas versus the competitive base were **-0.280814** and
  **-0.272607**.
- Tail risk remains material: true p95 well deltas versus HMM2 were +0.5744 and
  +0.5857, worst-well deltas +1.5234 and +1.4834, and only about 51% of wells
  improved. The candidate is rank-seeking rather than a robustness default.
- Built a local-only private Notebook contract and validator. The Notebook
  preserves and hashes the HMM intermediate, validates the Functional overlay
  against that exact intermediate, then binds final/report/audit SHA values.
  `execution_authorized=false`; it may be pushed and submitted once only if
  refs `55181026` and `55179387` both first score strictly below 6.477. Today’s
  Kaggle submission allowance remains exhausted, and no runtime was consumed.

### 2026-08-02 - Bidirectional HMM smoother failed its frozen tail/runtime gates

- Preregistered a fixed forward-backward smoother on 24 SHA-selected wells
  excluded from the original HMM discovery and current test IDs. The smoother
  uses the complete hidden-suffix GR sequence, which is available at inference,
  but no targets, absolute coordinates, well IDs, leaderboard values, or probes.
- Relative to forward HMM scale2 it gained **0.063336** pooled RMSE and improved
  62.5% of wells. However true-p95 well delta was **+0.404437**, above the
  +0.10 gate, and mean runtime was **46.69 s/well**, above the 45-second gate.
  Both forward and smoother variants also remained worse than the competitive
  base on this untouched discovery population.
- The preregistered decision is `FAIL_STOP`: no unchanged confirmation,
  parameter adjustment, Notebook, or submission is authorized.

### 2026-08-02 - Dynamic cross-well duplicate/template audits both failed

- Rechecked the official rules page before the audit. It allows Competition
  Data use and forbids hand labeling or human prediction of test records, but
  contains no explicit same-ID or automated-overlap clause. This is a rules
  interpretation, not organizer confirmation, so the project kept all matching
  dynamic and target-free and did not create a submission.
- Preregistered a strict translated-twin fingerprint over the complete relative
  MD/XYZ trajectory, horizontal GR and missingness, visible TVT shape, and
  Typewell shape. Across all 773 training wells there were **zero** different-ID
  duplicate groups and therefore zero transferable wells. The route stopped
  without relaxing any tolerance or building a Notebook.
- Preregistered one structurally different exact-Typewell template audit. For
  each query in the 13 duplicate fingerprint groups, the nearest distinct well
  supplied a normalized suffix TVT curve; a fixed 0.25 overlay and 5-ft
  pre-scale clip were frozen before scoring. All unmatched wells retained the
  competitive base.
- The 34 eligible wells worsened from pooled **10.094492 to 10.428320**;
  only 29.41% improved, p95-well delta was +1.0852, and worst-well delta was
  +1.1958. Across the full 3,783,989-row OOF population, RMSE worsened by
  **+0.013356**. The exact-Typewell transfer is permanently stopped without
  neighbor, scale, cap, or interpolation tuning.

### 2026-08-02 - Global Viterbi HMM had large pooled gain but failed tail gates

- Preregistered a single global max-product state path on 24 new SHA-selected
  wells, excluding all original and smoother discovery wells plus current test
  IDs. It kept the frozen HMM grid, transitions, scale2 overlay, and 5-ft
  disagreement clip, and used no labels, scores, probes, or fitted thresholds.
- The first execution was invalid because the evaluator inherited the original
  scale1 constant 0.20 instead of the preregistered scale2 value 0.40. Only that
  contract bug was corrected, a regression assertion was added, and the same
  wells/algorithm/gates were rerun; no invalid-run metric informed a parameter.
- At correct scale2, Viterbi improved pooled RMSE over forward HMM2 by a large
  **0.357791** and improved 62.5% of wells at 13.87 s/well. Its tail was
  unacceptable: true-p95 well delta **+1.421315** and worst-well delta
  **+2.700369**. Direct Viterbi is therefore `FAIL_STOP`, with no confirmation,
  Notebook, or submission.

### 2026-08-02 - Parameter-free Viterbi confidence fusion did not transfer

- Derived one threshold-free follow-up and preregistered it on 32 further
  untouched wells. Per-row Viterbi weight was fixed analytically as
  `u^2 / (u^2 + (v-f)^2 + 1e-12)`, where `u` is forward posterior uncertainty
  and `v-f` is Viterbi/forward path disagreement. No labels or learned
  parameters enter the fusion.
- On 150,159 hidden suffix rows, forward HMM2 RMSE was 11.910267 and the fused
  candidate was 11.913663: pooled gain **-0.003397** (an actual worsening).
  Only 50% of wells improved and true-p95 delta was +0.110259; the worst delta
  stayed bounded at +0.227661 and runtime averaged 13.12 s/well.
- Three frozen gates failed. The entire Viterbi/fusion family is closed without
  changing the formula, adding a threshold, running confirmation, or building
  a Kaggle candidate.

### 2026-08-02 - Final public-frontier deduplication found no deployable model

- Refreshed the full 6,060-team leaderboard. Public RMSE 6.477 is rank 1,164
  (top 19.21%); the rank-606 top-10% boundary is 6.418 and the rank-303
  top-5% boundary is 6.373.
- Re-read the score-ascending public Notebook frontier with UTF-8 output and
  audited the previously unnamed Roman, Nikita, Raunak Ultra, Boristown,
  i-aztec, and Blacklions branches. The first five are exact full-stack/cap4
  clones, visualization-only extensions, or fixed-Q0522 continuations. Nikita
  again changes exactly 4,301 rows of visible well `00e12e8b` by +0.522 ft;
  Roman applies the same fixed move on top of cap4.
- Blacklions' two-coefficient well-level GBDT is the only structurally new
  scoring layer. Its published execution report says its own random-by-well
  five-fold OOF worsened from 10.372253 to 10.415607 (gain -0.043355), with
  P90 and median well RMSE degrading by +0.301558 and +0.045423. Its internal
  gate failed and the output reverted to the fixed Q0522 anchor.
- Because this model already fails the author's less stringent OOF, a new
  spatial/Typewell rerun would not be a rational use of deadline compute. No
  private Notebook was built, no GPU was used, and no competition submission
  was created. The five existing refs remain pending and today's quota remains
  exhausted.

### 2026-08-02 - Depth-affine HMM scaling collapsed to global scale2.3

- Before scoring, froze one two-parameter HMM extension. The unchanged scale1
  correction receives a multiplier linearly interpolated from a hidden-suffix
  start coefficient to an end coefficient. Both coefficients are fit by exact
  bounded least squares on outer-training rows only and limited to [0, 2.3].
  No breakpoint, coefficient grid, ridge, intercept, well ID, absolute
  coordinate, leaderboard value, probe, or visible-test target is used.
- The 738-well run excluded all 32 HMM discovery wells and the three current
  visible test wells from both fitting and validation. It completed without
  failed wells in 1,044.0 seconds.
- Spatial7 improved fixed scale2 by only **0.007625** (10.215544 to 10.207919),
  below the frozen 0.010 threshold, and only 5/7 folds improved versus the
  required 6/7. Typewell5 gained **0.006569**, improved 4/5 folds, and passed
  its individual gate. Both protocols still improved the competitive base in
  every fold.
- The decisive structural result is that the all-training start and end
  multipliers both hit the 2.3 upper bound. The model therefore learns no
  depth dependence and collapses to the already preregistered global scale2.3.
  The overall decision is `FAIL_STOP`; no depth breakpoint, basis, or upper
  bound will be tuned, and no Notebook or submission is authorized.

### 2026-08-02 - Target-free cap4 scored 6.661 and closed both cap4 stacks

- The sole dynamic cap4 submission, ref `55177746`, completed with Public RMSE
  **6.661**. It is 0.184 worse than the 6.477 primary and 0.007 worse than the
  earlier cap2.5 result at 6.654. The byte-identical visible cap4 artifact's
  association with a public 6.432 source did not transfer to Kaggle's hidden
  Code Submission rerun.
- The candidate cannot replace either manually selected final submission. The
  current pair remains `54952634 + 54967104`.
- Both prepared cap4 combinations had an explicit prerequisite that cap4 score
  strictly below 6.477. That prerequisite is now false, so cap4 plus plain
  Functional Ridge and cap4 plus fixed Ridge/PLS are permanently marked
  `CLOSED_PUBLIC_GATE_FAIL`. No combined Notebook will be pushed and no extra
  submission will be created, regardless of the still-pending Functional
  component results.

### 2026-08-02 - Functional public scores missed the protected best

- Plain Functional Ridge ref `55178775` completed at Public RMSE **6.509**,
  0.032 worse than the protected 6.477 primary.
- Fixed 75/25 Functional Ridge/PLS ref `55179387` completed at **6.482**. The
  frozen ensemble improved plain Ridge by 0.027 but remained 0.005 worse than
  6.477, so neither Functional candidate can replace a final selection.
- The conditional HMM scale2 plus Ridge/PLS additive route required both
  components to score strictly below 6.477. Ridge/PLS failed that frozen gate,
  so the route is now `CLOSED_PUBLIC_GATE_FAIL`; its Notebook will not be
  pushed and no submission will be created.
- HMM scale1 ref `55180816` and scale2 ref `55181026` remain pending. If scale2
  later scores strictly below 6.477, the only authorized continuation is the
  already preregistered scale2.3 candidate. Final selections remain
  `54952634 + 54967104` and will not change automatically.

### 2026-08-03 - Both HMM submissions failed the Public gate

- Dip-rate HMM scale1 ref `55180816` completed at Public RMSE **6.573**, 0.096
  worse than the protected 6.477 primary. It cannot replace a final selection.
- Rank-seeking HMM scale2 ref `55181026` completed at **6.664**, 0.187 worse
  than 6.477 and 0.091 worse than scale1. Increasing the correction despite its
  stronger spatial/Typewell OOF signal moved Public performance in the wrong
  direction.
- The preregistered scale2.3 route required scale2 to score strictly below
  6.477. That frozen gate is false, so scale2.3 is permanently
  `CLOSED_PUBLIC_GATE_FAIL`: no Notebook push, run, submission, larger scale,
  or follow-up tuning is authorized.
- The current best remains 6.477 and the manually selected final pair remains
  `54952634 + 54967104`. No selection was changed automatically.

### 2026-08-03 - New public frontier audit authorized one dynamic second-slot hedge

- Downloaded and inspected five post-audit public Notebooks and their scoring
  outputs. Raunak Robust v3 and Physics-Informed v2 are byte-identical fixed
  Q0522 outputs (SHA `832848...071d`). Blacklions Final Hierarchy writes the
  exact current 6.477 primary (SHA `b192d3...ed4a`); its new TTA GBDT is inactive
  because its own OOF tail gate failed. Yaroslav Contact/U Restore fits replay
  coefficients directly from multiple Public scores and is rejected as
  leaderboard score-space reverse engineering.
- Daniil's public team score is genuinely **6.390**, rank 433/6,083 (top 7.12%),
  versus current top-10% boundary 6.412 and top-5% boundary 6.365. Its original
  last layer changes fixed well `00e12e8b` and suppresses failed source checks,
  so the public Notebook was not copied as-is.
- Rewrote only the final transaction to use runtime PF report rows: every
  applied branch saturating the generic 3.5-ft cap receives a signed 0.42-ft
  continuation. Official sample rows/IDs remain dynamic; fixed well, fixed row
  count, expected visible hash, and disabled mismatch guards were removed.
  Offline replay selected one 4,301-row branch and matched the public final
  values within 1.9e-12.
- Pushed exactly one private version of
  `heralces/rogii-v4-t3920-dynamic-pf-continuation-v1`; version 1 is running.
  No competition submission exists yet. Hidden output must pass
  `scripts/validate_v4_t3920_dynamic_output.py` before one controlled submission
  is allowed. The protected 6.477 primary and current final selections remain
  unchanged.
- Version 1 (version id `339924257`) completed and its downloaded artifact
  passed the validator: 14,151 exact-order rows, one runtime PF extension,
  4,301 changed rows, zero non-target movement, and exact final SHA
  `29e7c11...7610`. All report/audit flags and SHAs agree.
- After confirming the exact description did not already exist, created the
  one allowed Code Submission, ref `55215554`, description
  `Dynamic V4 T3920 PF-report continuation controlled verification`. It is
  pending and must not be duplicated. No final selection changed.

### 2026-08-04 - Dynamic V4 scored 6.498 and closed without selection change

- Ref `55215554` completed with Public RMSE **6.498**. It is 0.106 better than
  the selected 6.604 no-hedge but 0.021 worse than the protected 6.477 primary.
- The preregistered adoption rule required a score strictly below 6.477 before
  this leaderboard-informed candidate could be considered for the second final
  slot. That condition is false, so the result is `PUBLIC_NO_GO`.
- No 0.42-ft continuation tuning, retry, duplicate Notebook, duplicate
  submission, or automatic final-selection change is authorized. The final
  pair remains `54952634 + 54967104`.

### 2026-08-04 - Frozen quadratic PF minimax candidate submitted once

- Verified that refs `54967104`, `54952634`, and `55215554` are exactly
  collinear on one 4,301-row runtime PF branch, with total shifts 0, 2.00, and
  3.92 ft and zero non-target differences. Their Public scores 6.604, 6.477,
  and 6.498 determine the squared-RMSE response curve along this direction.
- The central optimum is 2.6739999 ft. Across all three-score rounding corners,
  the frozen minimax total is **2.672 ft** and the predicted Public interval is
  **6.46768–6.46890**. No weight or shift will be changed after this freeze.
- Audited the only newly relevant public source (`navazshfathi/best-score-roggi`,
  team score 6.411). Its current artifact is the already rejected fixed-Q0522
  SHA `832848...071d`, with fixed visible well and row-count assumptions; it
  was not adopted.
- Pushed exactly one private Kernel,
  `heralces/rogii-pf-quadratic-minimax-total-2-672-v1`, version 1 / version id
  `340024171`. The hidden output passed the dynamic validator: 14,151 ordered
  unique IDs, finite values, one runtime branch, 4,301 moved rows, signed total
  2.672 ft, zero non-target change, and final SHA `3135e369...3aa9`.
- Created exactly one Code Submission, ref `55225517`, description
  `Quadratic PF total2.672 minimax controlled verification`. It entered
  `PENDING`.
  The manual final pair remains `54952634 + 54967104`; no automatic change was
  made.

### 2026-08-04 - Quadratic PF candidate scored 6.482 and closed

- Ref `55225517` completed with Public RMSE **6.482**, 0.005 worse than the
  protected 6.477 primary. It failed the strict replacement gate and does not
  replace either manually selected submission.
- The score is also outside the preregistered 6.46768–6.46890 interval. Although
  the downloaded control and candidate artifacts are collinear to machine
  precision, that saved-artifact geometry did not transfer exactly to the
  scored Code Submission. The precise hidden execution difference is not
  observable, so the analytic curve is rejected as a further tuning signal.
- The route is `PUBLIC_NO_GO`: no version 2, shift refit, retry, duplicate
  submission, or automatic selection change. The retained pair remains
  `54952634 + 54967104`.
