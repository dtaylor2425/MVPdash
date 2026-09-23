# Macro Options Flow -- 60-Day Predictive Research Study

Dataset: `reports/options-flow-research-dataset.parquet` (immutable, unmodified) joined to `reports/options-flow-forward-returns.parquet`. 360 rows, 2026-06-26 to 2026-09-21, tickers: SPY, QQQ, IWM, SMH, TLT, GLD.

This is a **signal-discovery study**: no weights were optimized, no thresholds were brute-forced, no ML model was trained, and no significance claim here should be read as a production trading signal. All quintile/sign/regime cutoffs are plain quantiles of the data or existing product constants (e.g. the +-0.08 NEUTRAL sentiment band), never chosen to maximize a result.

**Leakage check:** max feature date `2026-09-21` <= last available price date `2026-09-23` -> **True**. Forward returns for dates near the end of the window are real, honest missing values (`None`, never `0.0`) where the future session hasn't happened yet -- see Q10.

## Q1. Does net_trade_sentiment predict forward returns? (H1)

Pooled quintile spread (Q5 = most bullish sentiment, Q1 = most bearish), all 6 ETFs, all dates:

| Horizon | Q1 N | Q1 mean | Q1 HAC-t | Q5 N | Q5 mean | Q5 HAC-t | Q5-Q1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D | 72 | -0.18% | -0.998 | 72 | -0.08% | -0.401 | 0.10% |
| 3D | 71 | -0.13% | -0.303 | 71 | -0.22% | -0.546 | -0.09% |
| 5D | 67 | -0.10% | -0.220 | 69 | -0.11% | -0.152 | -0.01% |
| 10D | 57 | -0.76% | -0.888 | 64 | -0.36% | -0.327 | 0.40% |
| 20D | 48 | -1.05% | -0.959 | 46 | 0.88% | 0.577 | 1.93% |

Per-ETF Q5-Q1 spread at 5D/10D, used only to check directional (sign) consistency across names for the verdict rule -- **not** for magnitude claims. Per-ETF quintile buckets have n as low as 9-14 at 10D, and the HAC lag (horizon-1=9) is then close to n, which is a known small-sample pathology that makes the HAC t-stat unstable (e.g. a large-looking |t| can appear from a handful of observations, not a real strong effect):

| Ticker | 5D Q5-Q1 | 5D Q5 HAC-t | 10D Q5-Q1 | 10D Q5 HAC-t |
|---|---:|---:|---:|---:|
| SPY | 0.83% | 0.956 | 0.67% | 1.489 |
| QQQ | -0.98% | -0.452 | 0.72% | 1.102 |
| IWM | 0.30% | -0.943 | 0.28% | -1.861 |
| SMH | -3.30% | -1.338 | 0.26% | -0.771 |
| TLT | 0.07% | -0.750 | 0.15% | -6.400 |
| GLD | -0.21% | 0.404 | -0.74% | 0.912 |

## Q2. Does delta_imbalance_ratio predict forward returns? (H2)

| Horizon | Q1 N | Q1 mean | Q1 HAC-t | Q5 N | Q5 mean | Q5 HAC-t | Q5-Q1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D | 72 | -0.21% | -1.115 | 72 | 0.12% | 0.537 | 0.32% |
| 3D | 71 | -0.37% | -0.863 | 71 | 0.08% | 0.188 | 0.45% |
| 5D | 67 | -0.29% | -0.546 | 68 | 0.07% | 0.099 | 0.36% |
| 10D | 60 | -0.58% | -0.762 | 60 | -0.35% | -0.303 | 0.23% |
| 20D | 50 | -0.71% | -0.794 | 48 | 0.39% | 0.230 | 1.10% |

| Ticker | 5D Q5-Q1 | 5D Q5 HAC-t | 10D Q5-Q1 | 10D Q5 HAC-t |
|---|---:|---:|---:|---:|
| SPY | 1.00% | 1.950 | 0.47% | 1.363 |
| QQQ | 0.32% | -0.061 | 1.29% | 0.701 |
| IWM | -0.25% | -1.724 | -0.82% | -1.954 |
| SMH | -2.27% | -0.888 | 0.45% | -0.578 |
| TLT | -0.02% | -0.597 | -0.36% | -4.036 |
| GLD | 1.92% | 1.771 | 0.07% | 2.081 |

## Q3. Does sentiment/delta-imbalance agreement matter more than either alone?

Category counts (neutral band = +-0.08, the product's existing threshold): agree_bullish=26, agree_bearish=40, disagreement=0, approximately_neutral=294

| Category | 5D mean | 5D N | 10D mean | 10D N | 20D mean | 20D N |
|---|---:|---:|---:|---:|---:|---:|
| agree_bullish | 0.19% | 24 | -0.17% | 22 | 1.61% | 14 |
| agree_bearish | 0.38% | 37 | -0.82% | 31 | -1.16% | 27 |
| disagreement | n/a | 0 | n/a | 0 | n/a | 0 |
| approximately_neutral | -0.07% | 281 | -0.09% | 259 | 0.32% | 211 |

Continuous `sentiment x delta_imbalance_ratio` interaction, Spearman rank corr vs 10D return: rho=-0.068, p=0.2331, n=312.

## Q4. Is the relationship monotonic across quintiles (not just Q1 vs Q5)?

**net_trade_sentiment**, 10D forward return by quintile:
| Q1 | Q2 | Q3 | Q4 | Q5 |
|---:|---:|---:|---:|---:|
| -0.76% | 0.03% | 0.04% | 0.17% | -0.36% |
Spearman(quintile, 10D return): rho=0.022, p=0.7035, n=312.

**delta_imbalance_ratio**, 10D forward return by quintile:
| Q1 | Q2 | Q3 | Q4 | Q5 |
|---:|---:|---:|---:|---:|
| -0.58% | -0.73% | 0.27% | 0.44% | -0.35% |
Spearman(quintile, 10D return): rho=0.062, p=0.2765, n=312.

## Q5. Does the signal cross-sectionally discriminate across the 6 ETFs?

Per date, rank the 6 ETFs by the feature; highest-ranked minus lowest-ranked forward-return spread (never raw net_dollar_delta for this test -- liquidity differs too much across ETFs):

| Feature | Horizon | Dates used | Spread mean | Spread HAC-t |
|---|---|---:|---:|---:|
| net_trade_sentiment | 1D | 60/60 | 0.36% | 1.383 |
| net_trade_sentiment | 3D | 59/60 | 0.25% | 0.563 |
| net_trade_sentiment | 5D | 57/60 | -0.19% | -0.433 |
| net_trade_sentiment | 10D | 52/60 | -0.13% | -0.217 |
| net_trade_sentiment | 20D | 42/60 | 0.37% | 0.277 |
| delta_imbalance_ratio | 1D | 60/60 | 0.50% | 1.742 |
| delta_imbalance_ratio | 3D | 59/60 | 0.88% | 1.895 |
| delta_imbalance_ratio | 5D | 57/60 | 0.39% | 0.813 |
| delta_imbalance_ratio | 10D | 52/60 | 0.48% | 0.773 |
| delta_imbalance_ratio | 20D | 42/60 | 1.38% | 1.525 |

## Q6. Do IV percentile / 0DTE-share / put-skew regimes modulate the signal?

Predetermined median splits (not optimized). Q5-Q1 spread of the primary feature, within each regime:

**Regime: iv_percentile_60d** (median=30.000, n high=178, n low=182)

| Signal | Regime | 5D Q5-Q1 | 10D Q5-Q1 |
|---|---|---:|---:|
| net_trade_sentiment | high | -0.87% | -1.36% |
| net_trade_sentiment | low | 0.89% | 2.11% |
| delta_imbalance_ratio | high | 0.72% | -0.59% |
| delta_imbalance_ratio | low | 0.53% | 1.57% |

**Regime: zero_dte_share** (median=0.166, n high=180, n low=180)

| Signal | Regime | 5D Q5-Q1 | 10D Q5-Q1 |
|---|---|---:|---:|
| net_trade_sentiment | high | 0.89% | 1.67% |
| net_trade_sentiment | low | -1.16% | -0.59% |
| delta_imbalance_ratio | high | 1.14% | 1.33% |
| delta_imbalance_ratio | low | 0.26% | 0.04% |

**Regime: put_skew_25d** (median=0.038, n high=173, n low=174)

| Signal | Regime | 5D Q5-Q1 | 10D Q5-Q1 |
|---|---|---:|---:|
| net_trade_sentiment | high | 0.01% | 0.08% |
| net_trade_sentiment | low | -0.35% | 0.17% |
| delta_imbalance_ratio | high | -0.10% | -0.14% |
| delta_imbalance_ratio | low | 0.98% | 0.59% |

## Q7. Are the results consistent per-ETF, or only in the pooled sample?

See Q1/Q2 per-ETF tables above. Directional consistency across at least 4 of 6 ETFs at 5D or 10D is required for the PROCEED verdict; pooled significance alone is not sufficient. Per-ETF HAC t-stats are read for **sign only** here, never for magnitude -- per-ETF n (9-14 at 10D/20D) is too small relative to the HAC lag for the t-stat itself to be trustworthy (see the caveat under Q1/Q2). The pooled tests (n=42-72) are the ones load-bearing for the verdict.

## Q8. Does the flagged GLD extreme day (GLD, 2026-09-02) drive any conclusion?

That single day is GLD's max observed delta_imbalance_ratio, net_dollar_delta, and net_trade_sentiment in Phase 1 (flagged, not excluded, in the QA report). Kept in the primary dataset per instruction; this re-run only checks whether removing it flips a conclusion, and removes no other observation.

**delta_imbalance_ratio** (n 360 including -> 359 excluding):
| Horizon | Q5-Q1 incl. | Q5-Q1 excl. | Sign flip? |
|---|---:|---:|---|
| 1D | 0.32% | 0.30% | no |
| 3D | 0.45% | 0.47% | no |
| 5D | 0.36% | 0.38% | no |
| 10D | 0.23% | 0.23% | no |
| 20D | 1.10% | 1.10% | no |

**net_dollar_delta** (n 360 including -> 359 excluding):
| Horizon | Q5-Q1 incl. | Q5-Q1 excl. | Sign flip? |
|---|---:|---:|---|
| 1D | 0.16% | 0.15% | no |
| 3D | 0.24% | 0.33% | no |
| 5D | 0.02% | 0.16% | no |
| 10D | 0.59% | 0.75% | no |
| 20D | 0.78% | 0.81% | no |

## Q9. Do supporting features show standalone predictive value?

Pooled Q5-Q1 spread only (net_dollar_delta, zero_dte_share, put_skew_25d, atm_iv, iv_percentile_20d/60d/126d/252d):

| Feature | 5D Q5-Q1 | 5D HAC-t | 10D Q5-Q1 | 10D HAC-t | 20D Q5-Q1 | 20D HAC-t |
|---|---:|---:|---:|---:|---:|---:|
| net_dollar_delta | 0.02% | 0.182 | 0.59% | 0.290 | 0.78% | 0.831 |
| zero_dte_share | 0.74% | 0.532 | 0.76% | 0.437 | 1.95% | 1.564 |
| put_skew_25d | -0.32% | -0.441 | -1.61% | -0.707 | -2.76% | -0.601 |
| atm_iv | 0.40% | -0.124 | 0.32% | -0.312 | -0.04% | -0.736 |
| iv_percentile_20d | 0.57% | 1.104 | 1.17% | 0.499 | -0.05% | 0.102 |
| iv_percentile_60d | -0.29% | -0.834 | 0.29% | -0.586 | -0.60% | -0.687 |
| iv_percentile_126d | -0.23% | -0.600 | -0.15% | -0.442 | -0.71% | -0.612 |
| iv_percentile_252d | 0.22% | -0.447 | 0.31% | -0.472 | 0.81% | -0.616 |

## Q10. What missing-data / sample-size caveats limit confidence?

Forward-return coverage by ticker/horizon (identical pattern across all 6 ETFs -- driven purely by how close each feature date is to "today", 2026-09-23, not by any per-ETF issue):

| Ticker | 1D | 3D | 5D | 10D | 20D |
|---|---:|---:|---:|---:|---:|
| SPY | 60 | 59 | 57 | 52 | 42 | (of 60)
| QQQ | 60 | 59 | 57 | 52 | 42 | (of 60)
| IWM | 60 | 59 | 57 | 52 | 42 | (of 60)
| SMH | 60 | 59 | 57 | 52 | 42 | (of 60)
| TLT | 60 | 59 | 57 | 52 | 42 | (of 60)
| GLD | 60 | 59 | 57 | 52 | 42 | (of 60)

Feature NaN counts by ticker (only `put_skew_25d` has missing values, all attributable to thin 25-delta put quotes on specific days -- see the QA report; no feature was zero-filled anywhere in this study, every N reported above is the true count of non-missing pairs):

| Feature | SPY | QQQ | IWM | SMH | TLT | GLD |
|---|---:|---:|---:|---:|---:|---:|
| put_skew_25d | 1 | 12 | 0 | 0 | 0 | 0 |

## Q11. Verdict

**NO EVIDENCE YET**

Verdict rule (robustness/consistency-based, not the best single bucket): PROCEED requires (a) at least one of H1/H2 shows |HAC t| >= 1.96 on the pooled Q5-Q1 spread at 2 of 3 decision-relevant horizons (5D/10D/20D) with a consistent sign, (b) at least 4 of 6 ETFs individually show the same-signed spread at 5D or 10D, and (c) the GLD-extreme-day exclusion does not flip the pooled sign. Meeting (a) but not (b)/(c) -> PROMISING BUT MODIFY RESEARCH DESIGN FIRST. Meeting none -> NO EVIDENCE YET.

No production model was fit. No weights or thresholds were optimized. This report does not start the 252-day full-flow backfill; that decision is left to the reader of this report.