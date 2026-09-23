# Macro Options Flow -- Signal Design v2 (Segmented Flow Features)

Built entirely from Phase 1's already-collected 60-day / 6-ETF dataset (360 ETF-days, 2026-06-26 to 2026-09-21) -- **no new ThetaData call, no new backfill**. Adds DTE-bucket-level sentiment, call/put-separate flow, 15-minute time-of-day buckets, and the day's 25 largest trades (all already stored in each Phase 1 snapshot's `payload` JSONB but not exported to the v1 daily-aggregate parquet), plus derived ratios, 1D/3D changes, and trailing-20D (prior-observations-only) z-scores.

**Baseline (Signal Research v1, frozen, not recomputed here):** daily-aggregate net_trade_sentiment and delta_imbalance_ratio both showed NO EVIDENCE (pooled Q5-Q1 spreads at 5D/10D/20D all |HAC t| < 1). v2 asks whether averaging the whole day together washed out real information that exists in a more specific slice of the same flow.

Every threshold below is a plain quintile or median split of the data -- none were tuned to maximize a result. No ML model was fit, no weights were optimized.

## What could NOT be tested from existing data

- **Moneyness (deep ITM / near ATM / moderately OTM / far OTM):** No moneyness classification is computed or stored anywhere in the Phase 1 pipeline -- api/services/options_flow_metrics.py never buckets by strike-to-spot distance, only by DTE. Cannot be tested from existing data.
- **Full trade-size percentile segmentation (top 1% / 5% / 10% of ALL trades):** Only the day's 25 largest trades are persisted per ticker (payload['largeTrades']); the full trade-size distribution is discarded after each snapshot (by design -- raw ticks are never stored). large_trade_sentiment/top25_premium_hhi below are a top-25-only proxy, not a true percentile-of-all-trades measure.
- **Large trade size relative to each ticker's own normal trade size:** Would need the full per-trade size distribution (or at least its median/percentiles) persisted per ticker-day; only top-25 trades and daily aggregate trade_count are stored.

## Full feature battery: pooled Q5-Q1 spread by horizon

`**` marks |HAC t| >= 1.96 on the Q5 or Q1 bucket itself (not the spread's own t-stat). "ETF agree" = how many of the 6 ETFs individually show the same-signed Q5-Q1 spread at both 5D and 10D (exploratory -- per-ETF n is 9-14 at these horizons).

| Feature | Family | Hyp. | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 | Spearman(q,10D) | ETF agree | Score |
|---|---|---|---|---|---|---:|---:|---:|
| large_trade_sentiment | Large/unusual trades | H1 | -0.58% (n=360) | -0.50% (n=360) | 0.61% (n=360) | -0.084 | 5/6 | 5 |
| large_trade_0dte_share | Large/unusual trades | H1 | -0.25% (n=360) | 1.02% (n=360) | 1.66% ** (n=360) | 0.118 | 4/6 | 14 |
| dte_0dte_sentiment | DTE segmentation | H2 | 0.92% (n=307) | 1.27% ** (n=307) | 1.82% (n=307) | 0.132 | 5/6 | 20 |
| dte_1_7d_sentiment | DTE segmentation | H2 | 0.39% (n=359) | 1.33% ** (n=359) | 1.41% (n=359) | 0.085 | 2/6 | 17 |
| dte_8_30d_sentiment | DTE segmentation | H2 | -0.37% (n=360) | 0.43% (n=360) | 0.30% (n=360) | 0.035 | 4/6 | 4 |
| dte_31_60d_sentiment | DTE segmentation | H2 | -0.38% (n=360) | -0.53% (n=360) | 0.49% (n=360) | -0.056 | 5/6 | 5 |
| call_sentiment | Call/put-separate flow | other | -0.27% (n=360) | 0.17% (n=360) | -1.69% (n=360) | -0.040 | 5/6 | 5 |
| put_sentiment | Call/put-separate flow | other | -0.37% (n=360) | -1.83% ** (n=360) | -3.61% ** (n=360) | -0.127 | 4/6 | 29 |
| opening_hour_sentiment | Time of day | H3 | -0.87% (n=360) | -1.44% (n=360) | -2.31% ** (n=360) | -0.119 | 4/6 | 19 |
| midday_sentiment | Time of day | H3 | 0.72% (n=360) | 0.93% ** (n=360) | 1.44% (n=360) | 0.057 | 3/6 | 18 |
| closing_hour_sentiment | Time of day | H3 | -0.43% (n=360) | 0.22% (n=360) | 1.30% (n=360) | 0.037 | 4/6 | 4 |
| sentiment_change_1d | Flow acceleration | H4 | 0.32% (n=354) | -0.06% (n=354) | -0.21% (n=354) | -0.008 | 5/6 | 5 |
| sentiment_change_3d | Flow acceleration | H4 | -0.58% (n=342) | 0.10% (n=342) | 0.08% (n=342) | 0.029 | 5/6 | 5 |
| delta_imbalance_change_1d | Flow acceleration | H4 | 0.43% (n=354) | -0.33% (n=354) | 0.66% (n=354) | -0.025 | 4/6 | 4 |
| delta_imbalance_change_3d | Flow acceleration | H4 | -0.11% (n=342) | 0.29% (n=342) | 0.43% (n=342) | 0.096 | 5/6 | 5 |
| gross_premium_pct_change_1d | Flow acceleration | H4 | 0.62% (n=354) | 0.09% (n=354) | 0.73% (n=354) | 0.028 | 4/6 | 9 |
| zero_dte_share_change_1d | Flow acceleration | H4 | 0.21% (n=354) | 0.12% (n=354) | -0.01% (n=354) | 0.019 | 4/6 | 4 |
| put_skew_change_1d | Flow acceleration | H4 | -0.14% (n=335) | -0.42% (n=335) | -0.51% (n=335) | -0.008 | 4/6 | 9 |
| atm_iv_change_1d | Flow acceleration | H4 | 1.48% ** (n=354) | 1.45% (n=354) | 1.63% (n=354) | 0.130 | 4/6 | 19 |
| top10_concentration | Concentration | H5 | -1.25% ** (n=360) | -1.77% (n=360) | -3.36% ** (n=360) | -0.172 | 1/6 | 26 |
| top25_premium_hhi | Concentration | H5 | -0.16% (n=360) | -0.27% (n=360) | -0.60% (n=360) | -0.026 | 3/6 | 8 |
| z_sentiment_20d | Baseline-relative (z-score) | H6 | -0.28% (n=300) | -0.08% (n=300) | 1.47% (n=300) | 0.034 | 5/6 | 5 |
| z_delta_imbalance_20d | Baseline-relative (z-score) | H6 | 0.34% (n=300) | 0.56% (n=300) | 1.73% (n=300) | 0.049 | 5/6 | 10 |
| z_gross_premium_20d | Baseline-relative (z-score) | H6 | 2.04% ** (n=300) | 3.68% ** (n=300) | 3.67% ** (n=300) | 0.332 | 6/6 | 41 |
| z_large_trade_premium_20d | Baseline-relative (z-score) | H6 | 0.91% (n=300) | 1.77% (n=300) | 1.59% ** (n=300) | 0.213 | 5/6 | 20 |
| z_0dte_share_20d | Baseline-relative (z-score) | H6 | -0.01% (n=300) | 0.22% (n=300) | -0.44% ** (n=300) | -0.029 | 4/6 | 14 |

## H1: Do large/unusual trades outperform the full-day aggregate?

`large_trade_sentiment` (25 largest trades/day only) Q5-Q1: 5D -0.58%, 10D -0.50%, 20D 0.61% vs. the full-day `net_trade_sentiment` baseline: 5D -0.01%, 10D 0.40%, 20D 1.93%.

## H2: Is longer-dated flow more predictive than 0DTE flow?

| DTE bucket | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 | N (10D) |
|---|---:|---:|---:|---:|
| 0DTE | 0.92% | 1.27% | 1.82% | 307 |
| 1-7D | 0.39% | 1.33% | 1.41% | 359 |
| 8-30D | -0.37% | 0.43% | 0.30% | 360 |
| 31-60D | -0.38% | -0.53% | 0.49% | 360 |

(60D+ bucket: 0 of 360 ETF-days had any 60D+ trade in this universe/window -- not testable.)

## H3: Is closing-hour flow more predictive than full-day flow?

| Window | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 |
|---|---:|---:|---:|
| Opening hour | -0.87% | -1.44% | -2.31% |
| Midday | 0.72% | 0.93% | 1.44% |
| Closing hour | -0.43% | 0.22% | 1.30% |

## H4: Is flow acceleration more predictive than flow level?

| Feature | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 |
|---|---:|---:|---:|
| sentiment_change_1d | 0.32% | -0.06% | -0.21% |
| sentiment_change_3d | -0.58% | 0.10% | 0.08% |
| delta_imbalance_change_1d | 0.43% | -0.33% | 0.66% |
| delta_imbalance_change_3d | -0.11% | 0.29% | 0.43% |

Compare to the level baselines: net_trade_sentiment 10D 0.40%, delta_imbalance_ratio 10D 0.23%.

## H5: Is concentrated flow more informative than diffuse flow?

Median split on `top10_concentration` (top-10-trade premium / full-day gross premium) crossed with the sign of the day's net_trade_sentiment:

| Horizon | Concentrated bullish | Diffuse bullish | Concentrated bearish | Diffuse bearish |
|---|---:|---:|---:|---:|
| 5D | -0.27% (n=77) | 0.33% (n=73) | 0.03% (n=95) | -0.06% (n=97) |
| 10D | -0.68% (n=72) | 0.61% (n=69) | -0.41% (n=86) | -0.11% (n=85) |
| 20D | -0.25% (n=53) | 1.39% (n=61) | -0.66% (n=73) | 0.56% (n=65) |

## H6: Is flow relative to an ETF's own baseline more informative than raw flow?

| Feature | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 | N (10D) |
|---|---:|---:|---:|---:|
| z_sentiment_20d | -0.28% | -0.08% | 1.47% | 300 |
| z_delta_imbalance_20d | 0.34% | 0.56% | 1.73% | 300 |
| z_gross_premium_20d | 2.04% | 3.68% | 3.67% | 300 |
| z_large_trade_premium_20d | 0.91% | 1.77% | 1.59% | 300 |
| z_0dte_share_20d | -0.01% | 0.22% | -0.44% | 300 |

z-scores need a 20D trailing window (min 10 prior observations) computed from ONLY prior dates for that ETF -- with 60 total dates per ETF, this leaves at most 50 usable dates per ETF (n≈300 pooled at best), noticeably smaller than the other tests. Treat as exploratory.

## Ranking (robustness / consistency / monotonicity / sample size / plausibility)

Ranked by the composite robustness score (10 x horizons clearing |HAC t|>=1.96 with a consistent sign, +5 if sign is consistent across 5D/10D/20D, +1 per ETF that individually agrees in sign) -- NOT by best single bucket:

| Rank | Feature | Family | Strong horizons | Sign consistent | ETF agree | Min N | Score |
|---|---|---|---:|---|---:|---:|---:|
| 1 | z_gross_premium_20d | Baseline-relative (z-score) | 3/3 | yes | 6/6 | 300 | 41 |
| 2 | put_sentiment | Call/put-separate flow | 2/3 | yes | 4/6 | 360 | 29 |
| 3 | top10_concentration | Concentration | 2/3 | yes | 1/6 | 360 | 26 |
| 4 | dte_0dte_sentiment | DTE segmentation | 1/3 | yes | 5/6 | 307 | 20 |
| 5 | z_large_trade_premium_20d | Baseline-relative (z-score) | 1/3 | yes | 5/6 | 300 | 20 |
| 6 | opening_hour_sentiment | Time of day | 1/3 | yes | 4/6 | 360 | 19 |
| 7 | atm_iv_change_1d | Flow acceleration | 1/3 | yes | 4/6 | 354 | 19 |
| 8 | midday_sentiment | Time of day | 1/3 | yes | 3/6 | 360 | 18 |
| 9 | dte_1_7d_sentiment | DTE segmentation | 1/3 | yes | 2/6 | 359 | 17 |
| 10 | large_trade_0dte_share | Large/unusual trades | 1/3 | no | 4/6 | 360 | 14 |
| 11 | z_0dte_share_20d | Baseline-relative (z-score) | 1/3 | no | 4/6 | 300 | 14 |
| 12 | z_delta_imbalance_20d | Baseline-relative (z-score) | 0/3 | yes | 5/6 | 300 | 10 |
| 13 | gross_premium_pct_change_1d | Flow acceleration | 0/3 | yes | 4/6 | 354 | 9 |
| 14 | put_skew_change_1d | Flow acceleration | 0/3 | yes | 4/6 | 335 | 9 |
| 15 | top25_premium_hhi | Concentration | 0/3 | yes | 3/6 | 360 | 8 |

**Cross-ETF date-clustering check on the top-ranked feature (z_gross_premium_20d):** its quintile-5 bucket has 60 ETF-day rows spread across only 34 unique calendar dates (1.76 rows/date on average) -- some "ETF agreement" reflects the same macro-calendar days (FOMC/CPI/OPEX) shared across tickers, not fully independent confirmations. This is exactly the kind of ambiguity a longer, multi-regime history would resolve.

**Two results deserve more scrutiny than their rank alone suggests:**

- `top10_concentration` ranks #3 by score, but only **1 of 6 ETFs** individually shows the same-signed relationship (vs. 4-6/6 for every other feature in the top 10). A pooled result this concentrated in one or two names is not the broad, cross-ETF pattern the score implies -- the composite score under-penalizes low ETF agreement here. Read it as single-name, not a market-wide concentration effect.
- `put_sentiment` ranks #2 and has an economically legible direction: (putBought-putSold)/(putBought+putSold) is POSITIVE when puts are being aggressively bought, and its Q5-Q1 spread is NEGATIVE at 10D/20D -- i.e. aggressive put-buying days are followed by weaker forward returns, the intuitive sign for put positioning as a bearish/hedging signal. It isn't one of the pre-registered H1-H6 hypotheses, so treat it as exploratory, but it is worth carrying into a 252-day test on that basis alone.

## Verdict

**SOME PROMISING SIGNALS, NEED MORE HISTORY**

**Why this is capped at PROMISING rather than STRONG:** z_gross_premium_20d clears the STRONG bar on its own (|HAC t|>=1.96 at 3/3 horizons, consistent sign, 6/6 ETFs agree), but this study spans only 60 trading days (~1 quarter) -- one regime, not several. Its quintile-5 bucket also has 60 rows across only 34 unique calendar dates (~1.8x per date), so "6/6 ETFs agree" partly reflects the same macro-calendar days shared across tickers, not 6 independent confirmations. Capped at PROMISING until it is re-tested on the 252-day history.

### Fields to make sure the 252-day backfill preserves

If any of the above families are pursued further, the 252-day historical job must keep (not just the daily aggregate) at minimum:
- Per-DTE-bucket `sentiment`/`deltaRatio`/`signedPremium` (already computed in the live payload's `dte.buckets`, but only `grossPremium` per bucket reached the v1 export -- confirm the 252-day export keeps the full bucket dict).
- `aggression` (callBought/callSold/putBought/putSold) at daily granularity, already computed -- confirm it's exported, not just used to derive the combined sentiment.
- The 15-minute `intraday` series, or at least pre-aggregated opening/midday/closing sentiment, per ticker-day.
- Either (a) the full per-trade size distribution's percentiles (p50/p90/p95/p99) computed at snapshot time, or (b) more than 25 large trades persisted -- top-25-only concentration/large-trade metrics are a proxy, not the real thing.
- A moneyness classification (e.g. |delta| bucket: >0.7 ITM, 0.4-0.6 near-ATM, 0.15-0.4 OTM, <0.15 far OTM) per trade, aggregated to daily buckets -- **not computed at all today**, and item 2B of this study could not be tested for exactly this reason.

No production model was fit. No weights or thresholds were optimized. This report does not start the 252-day full-flow backfill; that decision is left to the reader.