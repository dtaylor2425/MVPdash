# Macro Options Flow -- Signal Research v4: Market-Factor Decomposition

v3 found that z_gross_premium_20d's validation-sample effect leans more on the same-day cross-ETF median than the ETF-specific residual. This asks what that common factor actually predicts, using the frozen 252-session combined dataset (60 discovery + 192 validation) -- z_gross_premium_20d itself, its 20D window, methodology_version, and the discovery/validation split are all unmodified from v3.

`market_options_activity(date) = median(z_gross_premium_20d across SPY, QQQ, IWM, SMH, TLT, GLD)`, computed from 242 of 252 unique calendar dates (needs >=3 of 6 tickers non-missing that day).

## Q1/Q2/Q3: Does the market factor predict direction, volatility, or magnitude?

### SPY

| Outcome | N | Q1 mean | Q5 mean | Q5-Q1 | Spearman rho | p |
|---|---:|---:|---:|---:|---:|---:|
| 1D return | 242 | -0.05% | 0.30% | 0.35% | 0.137 | 0.0327 |
| 3D return | 241 | -0.01% | 0.79% | 0.79% | 0.189 | 0.0032 |
| 5D return | 239 | 0.00% | 1.04% | 1.03% | 0.224 | 0.0005 |
| 10D return | 234 | 0.49% | 1.85% | 1.36% | 0.185 | 0.0046 |
| 20D return | 224 | 1.41% | 2.50% | 1.09% | 0.098 | 0.1424 |
| 5D fwd realized vol | 241 | 0.0057 | 0.0086 | 0.0029 | 0.312 | 0.0000 |
| 10D fwd realized vol | 236 | 0.0068 | 0.0084 | 0.0016 | 0.207 | 0.0014 |
| 20D fwd realized vol | 227 | 0.0069 | 0.0084 | 0.0015 | 0.301 | 0.0000 |
| 5D fwd max drawdown | 231 | -1.20% | -1.39% | -0.18% | -0.040 | 0.5408 |
| 10D fwd max drawdown | 216 | -2.09% | -1.98% | 0.11% | 0.039 | 0.5693 |
| 20D fwd max drawdown | 192 | -2.74% | -2.76% | -0.03% | 0.021 | 0.7674 |
| 5D VIX change | 241 | 0.5394 | -1.7321 | -2.2715 | -0.301 | 0.0000 |
| 10D VIX change | 236 | 0.2577 | -2.8636 | -3.1213 | -0.267 | 0.0000 |
| 20D VIX change | 227 | -0.0198 | -2.8128 | -2.7930 | -0.157 | 0.0179 |

### QQQ

| Outcome | N | Q1 mean | Q5 mean | Q5-Q1 | Spearman rho | p |
|---|---:|---:|---:|---:|---:|---:|
| 1D return | 242 | -0.11% | 0.39% | 0.51% | 0.123 | 0.0565 |
| 3D return | 241 | -0.09% | 0.95% | 1.05% | 0.181 | 0.0049 |
| 5D return | 239 | -0.05% | 1.45% | 1.50% | 0.209 | 0.0012 |
| 10D return | 234 | 0.82% | 2.50% | 1.68% | 0.177 | 0.0067 |
| 20D return | 224 | 2.67% | 3.05% | 0.38% | 0.048 | 0.4703 |
| 5D fwd realized vol | 241 | 0.0084 | 0.0127 | 0.0043 | 0.305 | 0.0000 |
| 10D fwd realized vol | 236 | 0.0105 | 0.0127 | 0.0022 | 0.190 | 0.0034 |
| 20D fwd realized vol | 227 | 0.0106 | 0.0131 | 0.0025 | 0.335 | 0.0000 |
| 5D fwd max drawdown | 231 | -1.85% | -2.08% | -0.24% | -0.046 | 0.4875 |
| 10D fwd max drawdown | 216 | -3.30% | -3.08% | 0.22% | 0.030 | 0.6616 |
| 20D fwd max drawdown | 192 | -4.05% | -4.29% | -0.24% | -0.033 | 0.6496 |
| 5D VIX change | 241 | 0.5394 | -1.7321 | -2.2715 | -0.301 | 0.0000 |
| 10D VIX change | 236 | 0.2577 | -2.8636 | -3.1213 | -0.267 | 0.0000 |
| 20D VIX change | 227 | -0.0198 | -2.8128 | -2.7930 | -0.157 | 0.0179 |

### IWM

| Outcome | N | Q1 mean | Q5 mean | Q5-Q1 | Spearman rho | p |
|---|---:|---:|---:|---:|---:|---:|
| 1D return | 242 | -0.10% | 0.39% | 0.49% | 0.123 | 0.0564 |
| 3D return | 241 | -0.05% | 1.03% | 1.08% | 0.171 | 0.0079 |
| 5D return | 239 | -0.01% | 1.41% | 1.41% | 0.168 | 0.0092 |
| 10D return | 234 | 0.63% | 2.85% | 2.22% | 0.163 | 0.0124 |
| 20D return | 224 | 1.61% | 3.42% | 1.82% | 0.034 | 0.6082 |
| 5D fwd realized vol | 241 | 0.0090 | 0.0129 | 0.0039 | 0.287 | 0.0000 |
| 10D fwd realized vol | 236 | 0.0101 | 0.0123 | 0.0022 | 0.231 | 0.0003 |
| 20D fwd realized vol | 227 | 0.0102 | 0.0120 | 0.0019 | 0.221 | 0.0008 |
| 5D fwd max drawdown | 231 | -1.82% | -2.05% | -0.22% | -0.065 | 0.3251 |
| 10D fwd max drawdown | 216 | -2.78% | -2.61% | 0.17% | 0.048 | 0.4786 |
| 20D fwd max drawdown | 192 | -3.65% | -3.70% | -0.05% | -0.011 | 0.8804 |
| 5D VIX change | 241 | 0.5394 | -1.7321 | -2.2715 | -0.301 | 0.0000 |
| 10D VIX change | 236 | 0.2577 | -2.8636 | -3.1213 | -0.267 | 0.0000 |
| 20D VIX change | 227 | -0.0198 | -2.8128 | -2.7930 | -0.157 | 0.0179 |

## Q4: Is the effect concentrated around scheduled macro events?

**Not testable from existing Macro Engine data.** No FOMC/CPI/NFP/Treasury-refunding calendar module exists in this codebase (checked `src/`, `api/`, `jobs/`, `scripts/` for an event-date source; only trading-session calendars exist). Per the explicit instruction not to introduce an external event dataset for this purpose, event-day tagging and the exclude-event-days rerun were **not performed**. This caps the verdict below -- see item 6 in the report spec.

## Q5 (regimes): Does it behave differently in bullish/bearish or high/low-vol regimes?

SPY-based regimes (predetermined: same-day sign, 200DMA, 20D momentum, SPY IV-percentile-60D median split), 10D outcomes:

| Regime | Level | N | ret_10d Q5-Q1 | fwd_vol_10d Q5-Q1 |
|---|---|---:|---:|---:|
| same_day_direction | up | 136 | 0.97% | 0.0010 |
| same_day_direction | down | 116 | 1.35% | 0.0020 |
| spy_200dma | above | 158 | 0.41% | 0.0012 |
| spy_200dma | below | 12 | -1.11% | 0.0007 |
| spy_momentum | positive | 169 | 0.68% | 0.0009 |
| spy_momentum | negative | 83 | 0.90% | 0.0014 |
| spy_iv_percentile | high | 124 | 1.18% | -0.0004 |
| spy_iv_percentile | low | 128 | 0.33% | 0.0000 |

## Q6/Q7: Is there ETF-specific information left after removing the market factor?

Panel regression (ticker fixed effects, clustered by date), quintile spread, and a date-level cross-sectional high-minus-low portfolio, all on `residual_activity` (ticker's own z_gross_premium_20d minus that date's market factor):

| Horizon | Panel beta | Cluster t | p | Pooled Q5-Q1 | Cross-sectional H-L (date-level) |
|---|---:|---:|---:|---:|---:|
| 5D | -0.00005 | -0.077 | 0.9388 | 0.28% | 0.09% (n dates=239) |
| 10D | 0.00121 | 1.311 | 0.1912 | 0.82% | 0.12% (n dates=234) |
| 20D | 0.00182 | 1.437 | 0.1521 | 1.33% | 0.19% (n dates=224) |

Per-ETF residual Q5-Q1 spread (10D):
| SPY | QQQ | IWM | SMH | TLT | GLD |
|---:|---:|---:|---:|---:|---:|
| -0.09% | 1.72% | -2.70% | 3.77% | 0.02% | -0.13% |

## Q8/9 (part 2): Persistence and forward paths

Autocorrelation of the market factor with itself N sessions later:
- lag1: corr=0.411 (n=241)
- lag5: corr=0.094 (n=237)
- lag10: corr=-0.003 (n=232)

Average cumulative SPY return path, top-quintile vs bottom-quintile market-activity days (k = sessions after the signal date):

| k | Top-quintile cum. return (n) | Bottom-quintile cum. return (n) |
|---:|---:|---:|
| 1 | 0.30% (n=49) | -0.05% (n=49) |
| 3 | 0.75% (n=49) | -0.01% (n=49) |
| 5 | 1.02% (n=47) | -0.01% (n=48) |
| 10 | 1.89% (n=46) | 0.38% (n=47) |
| 15 | 2.41% (n=45) | 1.01% (n=47) |
| 20 | 2.54% (n=45) | 1.40% (n=45) |

Average rolling 5-session realized vol along the same path:
| k | Top-quintile vol | Bottom-quintile vol |
|---:|---:|---:|
| 1 | n/a | n/a |
| 3 | n/a | n/a |
| 5 | 0.0086 | 0.0057 |
| 10 | 0.0067 | 0.0066 |
| 15 | 0.0067 | 0.0060 |
| 20 | 0.0069 | 0.0059 |

## Bootstrap (date-clustering-robust check on the SPY 10D volatility result)

10,000-draw date-block bootstrap (block=5) of the SPY market-activity vs fwd_vol_10d Spearman-quintile spread: mean=0.0015, 95% CI=[0.0001, 0.0028], share same sign as full sample=0.98, n dates=252.

## Answers

1. **Does market-wide abnormal options activity predict direction?** Yes (SPY 10D return: Spearman rho=0.185, p=0.0046).
2. **Does it predict future volatility better than future returns?** Both show a relationship (10D fwd vol: rho=0.207, p=0.0014; 10D return: rho=0.185, p=0.0046).
3. **Does it predict absolute market movement?** 10D max-drawdown Q5-Q1 spread = 0.11% (rho=0.039, p=0.5693).
4. **Is the effect concentrated around scheduled macro events?** Not testable -- no macro-event calendar exists in this codebase; no external dataset was introduced per instruction.
5. **Does it survive excluding event days?** Not testable for the same reason -- no exclusion rerun was performed.
6. **Does it behave differently in bullish/bearish regimes?** See the regime table above -- report the sign/magnitude split by same-day direction, 200DMA, momentum, and IV-percentile regime rather than a single number here.
7. **Does ETF-specific residual activity predict cross-sectional ETF returns?** No (pooled panel regression, 10D: beta=0.00121, cluster t=1.311, p=0.1912).
8. **Is the original result mostly common-factor or ETF-specific?** Common-factor-dominant, consistent with v3's market-wide-control finding (residual panel t=1.311 vs market-factor SPY 10D-vol Spearman p=0.0014).
9. **What is the economic magnitude?** SPY 10D fwd-vol Q5-Q1 spread = 0.0016 (daily-return-std units); 10D max-drawdown Q5-Q1 spread = 0.11%.
10. **Is there enough evidence to build a production market-activity indicator?** Not yet -- event-day concentration cannot be ruled out from existing data, see the verdict below.

## Verdict

**MARKET ACTIVITY SIGNAL PROMISING BUT NOT YET VALIDATED**

No production dashboard was modified. No ETF universe expansion. No new feature search was performed -- this explains the v2/v3 signal, it does not replace it.