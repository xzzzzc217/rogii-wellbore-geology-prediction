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
