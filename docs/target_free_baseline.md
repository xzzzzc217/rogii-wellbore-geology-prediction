# Deterministic target-free trajectory baseline

This baseline recovers the masked TVT suffix as a sequence. It is intentionally
small, CPU-only, deterministic, and auditable. Its purpose is to provide a
leakage-safe trajectory candidate and diagnostics for later OOF blending, not
to claim medal-level performance by itself.

## Inference contract

The model has an explicit input allow-list:

| Source | Columns read by the model |
|---|---|
| horizontal well | `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input` |
| typewell | `TVT`, `GR` |

`X` and `Y` are accepted so the interface exactly matches the hidden-test
schema, but this first baseline does not yet use them. Extra columns are
ignored. In particular:

- horizontal-well `TVT` is never passed to the predictor;
- train-only contact/formation columns are never read;
- typewell `Geology` is never read;
- there is no train/test same-ID lookup;
- there is no leaderboard-derived bias or probe;
- there is no mounted model package or public precomputed prediction.

The evaluator loads train `TVT` into a separate scoring array. It constructs a
new frame from the six allowed horizontal columns before calling the model.
Unit tests also add sentinel `TVT`, contact, and `Geology` columns and verify
that changing them cannot change a prediction.

## Trajectory construction

### 1. Hold-last safety path

Let \(i_0-1\) be the final visible row. The safety prediction is

\[
\widehat T_i^{hold}=T^{input}_{i_0-1},\qquad i\ge i_0.
\]

It is returned directly when the typewell or visible heel is insufficient, and
it remains a small part of the normal final blend.

### 2. Typewell GR interpolation and heel calibration

Finite typewell rows are sorted by TVT; duplicate TVTs are averaged. Candidate
GR is obtained by linear interpolation without extrapolation. On at most the
last 900 visible comparisons, iteratively reweighted least squares fits

\[
G_i^{horizontal}\approx a\,G^{typewell}(T_i^{input})+c.
\]

Huber weights limit isolated spikes. The gain is constrained to
\([0.25,4]\), and the robust residual scale has a 10-GR floor. The fitted gain,
offset, scale, correlation, row count, and status are returned in every
well's diagnostics.

### 3. Deterministic multi-seed particle filter

The physical state is

\[
U_i=T_i+Z_i.
\]

Particles start around the exact visible anchor with a 4.5-ft standard
deviation. A local visible-heel estimate initializes the \(U\) rate. The
transition is a slowly varying rate model:

\[
v_i=0.998v_{i-1}+\epsilon_i^v,\qquad
U_i=U_{i-1}+v_i\Delta MD_i+\epsilon_i^U.
\]

For a particle \(U_i^{(p)}\), the candidate TVT is \(U_i^{(p)}-Z_i\). Its
calibrated typewell GR supplies a clipped Gaussian observation likelihood.
The robust calibration scale is multiplied by three in the PF likelihood.
This deliberately weakens a pointwise GR match because repeated motifs can
otherwise collapse every particle onto the wrong bed. Systematic resampling
adds small, seeded state/rate jitter.

Eight independent streams use fixed seeds 0 through 7. If \(\ell_s\) is a
stream's accumulated predictive log likelihood, the deterministic ensemble
weight is

\[
w_s=
\frac{\exp((\ell_s-\max_r\ell_r)/5)}
{\sum_r\exp((\ell_r-\max_q\ell_q)/5)}.
\]

Diagnostics expose every seed, weight, total and average log evidence,
resampling count, and posterior-spread summary.

### 4. Constrained beam/Viterbi tracker

The second tracker is independent of PF randomness. Its state is an index on
the sorted typewell TVT grid. It starts at the typewell row nearest the final
visible TVT and permits index moves \(\{-2,-1,0,1,2\}\) per horizontal row.
After light deterministic GR smoothing, a transition has cost

\[
C_i(j\rightarrow k)=
C_{i-1}(j)
\left[
\frac{G_i^{horizontal}-(aG_k^{typewell}+c)}{s}
\right]^2
20|k-j|.
\]

Only the ten lowest-cost distinct states survive each step. Parent pointers
are retained, so the returned path is the best terminal path after Viterbi
backtracking rather than a row-wise greedy sequence. The resulting TVT-grid
path is represented internally as \(U=T+Z\).

### 5. Conservative final blend

The tracker candidate is

\[
\widehat T^{tracker}
=0.95\widehat T^{PF}+0.05\widehat T^{beam}.
\]

A heel-quality trust score uses only calibration row count and visible-prefix
GR correlation. It maps to a tracker weight between 0.85 and 0.95; the
remainder is hold-last. Finally, the path is constrained to typewell support
and to at most 90 ft from the visible anchor. There is no constant TVT bias.

## API and diagnostics

The main entry point is:

```python
from rogii.trajectory import predict_trajectory

result = predict_trajectory(horizontal_frame, typewell_frame)
hidden_prediction = result.prediction[result.evaluation_mask]
audit = result.diagnostics
```

`result.components` contains `hold_last`, `particle_filter`,
`beam_viterbi`, `tracker_blend`, and `final`. Diagnostics include:

- the exact model column allow-list and all prohibited-dependency flags;
- row counts, suffix start, anchor TVT and anchor \(U\);
- heel gain/offset/scale/correlation;
- PF seed likelihoods, weights, resampling counts, and posterior spread;
- beam average cost, terminal margin, and state count;
- hidden GR residuals for every component;
- trust/blend weights, output range, status, and fallback reason.

## Reproducing suffix evaluation

The default command evaluates a seeded representative sample of 50 official
train suffixes:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_target_free.py `
  --data-root data\competition `
  --max-wells 50 `
  --pf-seeds 8 `
  --fail-fast
```

`--max-wells 0` requests all 773 wells. `--selection-offset` supports
non-overlapping resumable chunks from the same seeded permutation.
`--json-output` writes the complete metrics and per-well diagnostics.

Pooled RMSE is computed after concatenating every scored suffix row. Macro and
tail statistics are computed from one RMSE per well.

## Bounded validation snapshot

Run on 2026-07-23 with seed 20260723, 50 of 773 train wells, 64 particles,
eight PF seeds, and no failed wells:

| component | pooled RMSE | well-macro | P90 | P95 | P99 | worst well |
|---|---:|---:|---:|---:|---:|---|
| hold-last | 16.2052 | 12.2838 | 21.5151 | 31.1720 | 49.0272 | `ba48188d` (50.9372) |
| particle filter | 14.0900 | 10.5302 | 23.9467 | 27.1651 | 35.9723 | `df71d1ff` (41.3115) |
| beam/Viterbi | 16.0664 | 12.3752 | 20.9295 | 30.5257 | 47.5650 | `ba48188d` (50.1219) |
| tracker blend | 13.5599 | 10.1527 | 23.0954 | 25.5605 | 34.5269 | `df71d1ff` (39.8780) |
| final | **12.8415** | **9.6905** | **19.9515** | **23.5661** | **32.8864** | `df71d1ff` (38.1285) |

Runtime was 158.0 seconds on the local Windows CPU environment. This snapshot
is a bounded engineering check, not a frozen full-CV estimate. It shows a
clear pooled improvement over hold-last but also shows that rare branch errors
still dominate the tail. A later candidate should use well-level
pseudo-holdouts or an independently trained residual model to gate those
branches; hidden GR fit alone was not a reliable disaster detector in this
sample.

## Public inspiration and provenance

The implementation is original project code. It re-expresses general public
ideas rather than importing notebook code or predictions:

- likelihood-weighted multi-seed PF, conservative beam blends, and the
  \(U=TVT+Z\) framing:
  [PF-Config Strategy — High-Scoring 6.491](https://www.kaggle.com/code/helloromiro/pf-config-strategy-high-scoring-6-491);
- target-free alignment, \(U\)-space projection, and auditable fallback
  thinking:
  [Target-Free TVT Geosteering](https://www.kaggle.com/code/pilkwang/rogii-target-free-tvt-geosteering);
- visible-heel gain/offset calibration and dual-track formulation:
  [Dual-Track Prefix-Calibrated Geosteering](https://www.kaggle.com/code/pilkwang/rogii-dual-track-prefix-calibrated-geosteering);
- constrained beam search as an independent sequence signal:
  [Triple-Signal Beam Search / Dual PF / LightGBM](https://www.kaggle.com/code/shinyanagai123/triple-signal-beam-search-dual-pf-lightgbm).

None of those notebooks, their model packages, or their generated submissions
is a runtime dependency of this baseline.
