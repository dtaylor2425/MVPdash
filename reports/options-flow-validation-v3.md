# Macro Options Flow -- Signal Validation v3

Primary feature under validation: **`z_gross_premium_20d`** (frozen in `reports/options-flow-v3-validation-manifest.json`, discovered in commit `19dc02c`, expected sign: **POSITIVE**, recorded BEFORE any new data was pulled).

## Sample composition

| Sample | Sessions/ETF | Rows | Role |
|---|---:|---:|---|
| DISCOVERY (original, frozen) | 60 | 360 | Displayed separately -- NOT part of the primary significance test |
| VALIDATION (new, unseen) | 192 | 1152 | **Primary test sample** |
| warmup-only (new, unseen) | 20 | 120 | Seeds the trailing-20D baseline for the earliest validation dates only -- never counted as an observation |
| COMBINED (descriptive) | 252 | 1512 | Descriptive view only -- **not** an independent significance test (60 of these 252 sessions are the original discovery data, not out-of-sample) |

## Q1/Q2: Does the primary feature retain its direction and clear a bar in the new sample?

Pooled Q5-Q1 spread, validation sample only (n shown per horizon):

| Horizon | N | Q1 mean | Q5 mean | Q5-Q1 | Panel beta (cluster by date) | Cluster t | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D | 1152 | 0.09% | 0.33% | 0.24% | n/a | n/a | n/a |
| 3D | 1152 | 0.12% | 0.61% | 0.49% | n/a | n/a | n/a |
| 5D | 1152 | 0.31% | 0.97% | 0.66% | 0.00161 | 1.609 | 0.1093 |
| 10D | 1152 | 0.65% | 1.62% | 0.97% | 0.00181 | 1.482 | 0.1401 |
| 20D | 1152 | 2.17% | 2.35% | 0.18% | 0.00035 | 0.246 | 0.8059 |

**Discovery-sample result (original, frozen, for comparison only):** 10D Q5-Q1 = 3.68% (n=300).

## Q3: Does it survive date-clustered inference (the primary weakness of v2)?

Panel regression with ticker fixed effects, standard errors clustered by market_date (6 same-date ETF observations are not 6 independent draws):

- **5D**: beta=0.00161, cluster SE=0.00100, t=1.609, p=0.1093, n=1152, clusters(dates)=192
- **10D**: beta=0.00181, cluster SE=0.00122, t=1.482, p=0.1401, n=1152, clusters(dates)=192
- **20D**: beta=0.00035, cluster SE=0.00143, t=0.246, p=0.8059, n=1152, clusters(dates)=192

## Q4: Does it survive a date/block bootstrap?

10,000 resamples of unique DATES (not ETF-days); iid (block=1) and a 5-date moving block (preserves short-run serial correlation):

| Statistic | Bootstrap | Mean | 95% CI low | 95% CI high | Share same sign as full sample |
|---|---|---:|---:|---:|---:|
| Panel beta (10D) | n=10000 | 0.00176 | -0.00071 | 0.00412 | 0.92 |
| Panel beta (10D, block=5) | n=10000 | 0.00179 | -0.00111 | 0.00456 | 0.89 |
| Q5-Q1 spread (10D) | n=10000 | 0.01040 | -0.00012 | 0.02086 | 0.97 |
| Q5-Q1 spread (10D, block=5) | n=10000 | 0.01049 | -0.00286 | 0.02378 | 0.94 |

## Q5: Is the quintile relationship monotonic?

10D forward return by quintile (Q1=lowest z_gross_premium_20d, Q5=highest):
| Q1 | Q2 | Q3 | Q4 | Q5 |
|---:|---:|---:|---:|---:|
| 0.65% | 1.09% | 1.11% | 1.34% | 1.62% |
Spearman(quintile, 10D return): rho=0.052, p=0.0800, n=1152.

## Q6: Is the effect present across ETFs?

| Ticker | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 |
|---|---:|---:|---:|
| SPY | 0.30% | 0.23% | 0.25% |
| QQQ | 0.55% | 1.08% | -0.45% |
| IWM | -0.16% | -1.52% | -2.01% |
| SMH | 1.19% | 2.39% | -1.51% |
| TLT | 0.16% | 0.52% | -0.38% |
| GLD | 1.36% | 2.26% | 3.33% |

## Q7: Is the effect driven by a handful of macro-event dates?

Leave-one-date-out (192 reruns, dropping one validation date at a time), 10D Q5-Q1 spread: full-sample=0.97%, range=[0.80%, 1.17%], share of reruns preserving the expected sign=1.00.

Most influential dates (largest shift in the spread when removed):
- 2026-04-06: leave-one-out spread 1.17% (shift 0.20%)
- 2026-03-30: leave-one-out spread 0.80% (shift -0.17%)
- 2025-11-06: leave-one-out spread 1.10% (shift 0.12%)
- 2026-03-10: leave-one-out spread 0.86% (shift -0.11%)
- 2026-01-29: leave-one-out spread 1.08% (shift 0.10%)

Dropping the **5 most influential dates simultaneously**: 10D Q5-Q1 spread = 1.20% (full sample: 0.97%).

## Q8: Is it robust across time and regimes?

Predetermined splits only (chronological half, SPY 200DMA, SPY realized-vol median, SPY 20D momentum sign, this ETF's own IV-percentile-60D median) -- 10D Q5-Q1 spread within each level:

| Regime | Level | N | 5D Q5-Q1 | 10D Q5-Q1 |
|---|---|---:|---:|---:|
| chronological_half | first_half | 576 | 0.21% | 0.20% |
| chronological_half | second_half | 576 | 0.97% | 1.67% |
| _spy_200dma | above | 1080 | 0.75% | 1.06% |
| _spy_200dma | below | 72 | -1.50% | -0.68% |
| _spy_vol | high | 576 | 0.14% | 0.65% |
| _spy_vol | low | 576 | 1.17% | 1.54% |
| _spy_momentum | positive | 798 | 0.35% | 0.72% |
| _spy_momentum | negative | 354 | 1.34% | 1.71% |
| _iv_pctile | high | 576 | 0.57% | 0.78% |
| _iv_pctile | low | 576 | 0.30% | 0.10% |

## Q9 (part 1): How much of this is market-wide event-day activity vs. ETF-specific?

Two-regressor panel regression: `ret_10d ~ ticker_specific_z + market_wide_median_z` (ticker fixed effects, clustered by date):
- ETF-specific `z_gross_premium_20d`: beta=0.00047, t=0.458
- Market-wide median `z_gross_premium_20d` (cross-ETF, same date): beta=0.00356, t=1.580
- n=1152, clusters(dates)=192

## Q10: Did the original 60-day discovery result replicate?

| Horizon | Discovery Q5-Q1 (n=60x6, frozen) | Validation Q5-Q1 (n=192x6, new) | Same sign? |
|---|---:|---:|---|
| 5D | 2.04% | 0.66% | yes |
| 10D | 3.68% | 0.97% | yes |
| 20D | 3.67% | 0.18% | yes |

## Combined (descriptive) sample -- NOT an independent significance test

Pooling all 252 sessions (60 discovery + 192 validation) x 6 ETFs = 1512 rows purely for descriptive reference. 10D Q5-Q1 spread: 1.52% (n=1404). **60 of these 252 sessions are the original discovery data that produced the hypothesis in the first place -- this number is NOT out-of-sample and must never be quoted as such.**

## Secondary candidates (put_sentiment, top10_concentration)

Evaluated only after the primary result above; Benjamini-Hochberg FDR-corrected across the panel-regression p-values at all 3 primary horizons for both candidates (6 tests):

| Feature | Horizon | Panel beta | t | raw p | BH q | Reject at 0.05? |
|---|---|---:|---:|---:|---:|---|
| put_sentiment | 5D | -0.00100 | -0.164 | 0.8698 | 0.8895 | no |
| put_sentiment | 10D | -0.00365 | -0.461 | 0.6457 | 0.8895 | no |
| put_sentiment | 20D | 0.00767 | 0.586 | 0.5582 | 0.8895 | no |
| top10_concentration | 5D | -0.00127 | -0.139 | 0.8895 | 0.8895 | no |
| top10_concentration | 10D | -0.02202 | -1.882 | 0.0614 | 0.3681 | no |
| top10_concentration | 20D | -0.02184 | -1.287 | 0.1995 | 0.5985 | no |

Secondary results do not redefine the primary verdict below, regardless of outcome.

## Verdict checklist (item 14 -- majority of criteria, not one t-stat)

| Criterion | Met? |
|---|---|
| Same direction as v2 (positive) | YES |
| Meaningful effect in the new 192-session sample (|cluster t|>=1.96, 10D) | no |
| Date-clustered inference supportive | no |
| Date-block bootstrap CI supportive (excludes 0, expected sign) | no |
| Q5-Q1 spread in expected direction (>=2 of 3 primary horizons) | YES |
| Reasonable monotonicity across quintiles (10D) | YES |
| Effect not driven by 1-5 dates (>=85% LOO sign-preserved, survives dropping top-5) | YES |
| Majority of ETFs directionally consistent (>=4/6, 10D) | YES |
| Reasonable stability across time/regimes (>=2.5/5) | YES |
| Market-wide activity control does not eliminate the effect | YES |

**7 of 10 criteria met.**

## Verdict

**PARTIALLY VALIDATED -- MORE RESEARCH REQUIRED**

No production weights were fit. No ETF universe expansion. No new signal search was started. This report does not modify the live Options Flow page.