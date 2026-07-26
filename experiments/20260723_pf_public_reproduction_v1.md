# 20260723_pf_public_reproduction_v1

- Timestamp: 2026-07-23 Asia/Shanghai
- Kaggle notebook: `heralces/rogii-pf-public-reproduction`
- Kaggle version: 1
- Visibility: private
- Hardware: Kaggle GPU
- Internet: disabled
- Parent: Roman Mironov,
  `helloromiro/pf-config-strategy-high-scoring-6-491`
- Hypothesis: the attributed public PF/beam, residual-model, stratigraphic-projection, and
  visible-prefix stack provides a competitive first hidden-rerun anchor.
- Material change from parent: provenance note and private owner metadata only; algorithm and
  mounted public datasets are intentionally unchanged for the first reproducibility check.
- Validation status: public author's score claim only; independent grouped OOF audit pending.
- Leakage status: static audit pending. Same-well guards, score-specific constants, and mounted
  model packages must not be interpreted as independent private-leaderboard evidence.
- Runtime status: complete in about 804 seconds on the 3 visible demo wells.
- Output status: structurally valid, 14,151 rows, exact ID order, all finite.
- Diagnostic result: the same-ID/contact route overrode all three visible wells. The final
  unvalidated PF midpoint hedge then moved one well by about 2 ft. Visible-demo pooled RMSE
  against the overlapping training truth is 1.102, while two wells are near-exact and the moved
  well is about 1.999 RMSE. This is leakage-path/runtime evidence, not hidden-test validation.
- Hidden code-rerun result: submission `54952634` completed with public RMSE
  **6.477**.
- New evidence on 2026-07-24: the newly published
  `arnavsalkade/rogii-public-6-451-base` has all 45 code-cell sources byte-for-byte identical
  to this reproduction. Its title says 6.451, but a later page-level audit found that the
  authoritative Current/Best score for the only version is **6.504**. The source identity and
  6.504 page score materially strengthen the runtime/score evidence; the title number itself is
  not treated as evidence.
- Hidden submission: reference `54952634`, submitted 2026-07-24 13:53:17 UTC.
- 2026-07-25 leaderboard snapshot: rank **437 / 5,653** (top 7.73%).
  The top-10% boundary was rank 566 / 6.498 and the top-5% boundary was rank
  283 / 6.456.
- Decision: current primary candidate and provisional public bronze position.
  Keep the earlier leakage/runtime audit attached and do not treat the public
  score as private-leaderboard validation; the 74% private split can still
  reshuffle the final result.
