# Visible-prefix residual calibration: frozen evaluation

## Decision

**NO-GO.** None of the four candidates passed the discovery gate, so the
295-well confirmation set was not built, scored, or inspected.

## Contract

- 773 wells were split into six deterministic X/Y spatial clusters.
- Clusters ranked 1 and 4 were reserved as untouched confirmation blocks.
- Discovery contained 478 wells; confirmation contained 295 wells.
- Each cache was created without reading hidden `TVT`.
- The final 256 visible `TVT_input` rows were pseudo-masked in a frame
  truncated at the official boundary.
- Four bounded candidates were frozen before discovery:
  `level64_cap4_tau240`, `level128_cap4_tau480`,
  `level128_cap8_tau480`, and `level_slope128_cap4_tau480`.
- A candidate needed at least -0.020 pooled RMSE, negative macro and
  bootstrap deltas, at least 3/4 nonpositive discovery clusters, and the
  predeclared tail-risk limits.

## Discovery result

Baseline pooled RMSE was 12.273985.

| Candidate | Pooled delta | Macro delta | CI95 upper | Nonpositive clusters | Gate |
|---|---:|---:|---:|---:|---|
| `level64_cap4_tau240` | +0.001990 | +0.004530 | +0.004036 | 2/4 | FAIL |
| `level128_cap4_tau480` | +0.002350 | +0.006572 | +0.007727 | 2/4 | FAIL |
| `level128_cap8_tau480` | +0.005876 | +0.014784 | +0.012780 | 2/4 | FAIL |
| `level_slope128_cap4_tau480` | +0.007083 | +0.018276 | +0.013489 | 2/4 | FAIL |

The result is consistently too small and directionally adverse. Do not tune
the windows, caps, slopes, or decay constants on this discovery set.

## Evidence

- JSON: `artifacts/prefix_residual_evaluation.json`
- SHA-256:
  `03E81AE85BF6E4A6AB3FA7A20798CA0439026047C4DF97CAC0D8B4995ECB6AAF`
- Runtime: 880.1 seconds with four workers
- Tests: `tests/test_prefix_residual.py`
