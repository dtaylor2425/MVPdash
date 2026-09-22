# Macro Options Flow — backend contract

Private feature. The **worker calculates, Postgres stores, the API reads, the page displays.**
The browser never talks to ThetaData or to these endpoints.

```
theta-options-worker (Py 3.12, Railway cron)  ->  Railway Postgres  <-  FastAPI (Py 3.11)  <-  Next.js server route  <-  browser
 jobs/options_flow_refresh.py                     options_flow_*         /api/private/options-flow/*   (checks PRIVATE_OPTIONS_EMAILS,
 THETADATA_API_KEY                                                       X-Internal-Options-Token       adds the token server-side)
```

## Environment variables

| Where | Var | Notes |
|---|---|---|
| worker | `THETADATA_API_KEY` | required; server env only; never logged (scrubbed from errors) |
| worker + API | `DATABASE_URL` | existing Railway Postgres |
| API + Next server | `INTERNAL_OPTIONS_API_SECRET` | same value on both; **never `NEXT_PUBLIC_*`** |
| Next server only | `PRIVATE_OPTIONS_EMAILS` | comma-separated allow-list, checked against the logged-in user |
| worker + API | `OPTIONS_FLOW_UNIVERSE` | optional. `INDEX:SPY,QQQ;TECH:XLK` or JSON `{"INDEX":["SPY"]}` or flat `SPY,QQQ` |
| worker | `OPTIONS_FLOW_MAX_DTE` (60), `OPTIONS_FLOW_STRIKE_RANGE` (25), `OPTIONS_FLOW_GREEK_INTERVAL` (1m), `OPTIONS_FLOW_BUCKET_MIN` (15), `OPTIONS_FLOW_THETA_CONCURRENCY` (max 4, hard cap) | see `api/services/options_flow_config.py` |
| worker + API | `OPTIONS_FLOW_LIVE_MAX_AGE_MIN` (20) / `OPTIONS_FLOW_STALE_AFTER_MIN` (60) | LIVE → DELAYED → STALE thresholds |

## Deploy

1. `python scripts/run_options_flow_migration.py` (or let the worker's first run apply `sql/005_options_flow.sql`; it is idempotent).
2. New Railway service **theta-options-worker** from this repo: builder Dockerfile (`Dockerfile.theta-worker`, Python 3.12) — or set its config-as-code path to `railway.theta-worker.json`. Set `THETADATA_API_KEY`, `DATABASE_URL`. Cron `*/15 13-21 * * 1-5` (UTC). The job picks the right trading date itself and exits 0 without work once the post-close (>= 16:15 ET) run is published.
3. Existing API service: set `INTERNAL_OPTIONS_API_SECRET`. Do **not** add `thetadata` to it.
4. First run: `python jobs/options_flow_refresh.py --dry-run --tickers SPY` and read the output.

Worker exit codes: `0` ok / nothing to do · `1` failed (nothing published, previous snapshot untouched) · `2` partial (published, ≥1 ticker failed).

## API

All routes need header `X-Internal-Options-Token: $INTERNAL_OPTIONS_API_SECRET`.
Wrong/missing token → `404`; server secret unset → `503`; unknown ticker → `404`; no data yet → `404`; DB down → `503`.
Responses carry `Cache-Control: no-store`, `X-Robots-Tag: noindex`. Routes are excluded from `/docs`.

| Route | Returns |
|---|---|
| `GET /api/private/options-flow/latest` | board: `{run, groups, tickers[], missing[]}`. Tickers are the **summary** payload (no `intraday`, no `largeTrades`). |
| `GET /api/private/options-flow/ticker/{T}?include_trades=true` | `{run, ticker}` full payload. **Pass `include_trades=false` for capture mode** so the large-trade table never reaches the page. |
| `GET /api/private/options-flow/history?ticker=SPY&days=20` | `{ticker, days, series[]}` one point per market date (last snapshot of the day), oldest first. `days` 1–260. |
| `GET /api/private/options-flow/status` | `{latestPublished, lastRun, lastRunDiagnostics, universe}` |

`run`: `{id, marketDate, asOf, publishedAt, status: success|partial, dataStatus: LIVE|DELAYED|CLOSED|STALE, dataStatusReason, ageMinutes}`.
`asOf` = newest trade in the run (the "Last update" timestamp; convert to ET for display). `publishedAt` drives freshness.
A failed Theta job never removes anything: the API only serves `success`/`partial` runs, so the last good board keeps rendering and `dataStatus` flips to `STALE`.
A ticker missing from the newest run keeps its previous snapshot with `carriedForward: true`.

### Ticker payload (all camelCase; NaN never appears — missing = `null`)

```
ticker, group, marketDate, asOf, spot, publishedAt, carriedForward
sentiment   { value [-1,1]|null, label, lowConfidence, reasons[] }
              label: BULLISH | LEAN BULLISH | NEUTRAL | LEAN BEARISH | BEARISH | LOW CONFIDENCE
premium     { gross, bullish, bearish, netDirectional, totalAllTrades, top10Concentration }
delta       { netContracts, netDollar, grossDollar, ratio, matchedPctPremium, matchedTrades }
iv          { atm, atmChange1d, atmChange5d, iv7d, iv30d, iv60d, spread7v30, spread30v60,
              skew25d30d, percentile (0-100), percentileObs, termStructure[{expiration,dte,atmIv}] }
dte         { zeroDteShare, buckets[{bucket: 0DTE|1-7D|8-30D|31-60D|60D+, trades, grossPremium, sharePct,
              signedPremium, sentiment, deltaImbalance, deltaRatio}] }
aggression  { callBought, callSold, putBought, putSold }            (USD premium)
openInterest{ matchedPctPremium, contractsToOi, premiumPerOiContract, tradedContracts, oiContracts }
quality     { trades, eligibleTrades, classifiedPctTrades, classifiedPctPremium (0-1), deltaMatchedPctPremium,
              oiMatchedPctPremium, lastTradeTime, thetaFetchedAt, expirations, thetaRequests, fetchSeconds, warnings[] }
intraday[]  { t (ET ISO), netPremium, grossPremium, cumNetPremium, cumDollarDelta, rollingSentiment, trades }
largeTrades[] (private) { time, ticker, expiration, strike, right C|P, size, price, premium, bid, ask,
              aggressor, aggressorLabel (AT ASK|ABOVE MID|MID|BELOW MID|AT BID), direction, delta, iv, dte, openInterest }
```

## Definitions (decisions worth knowing)

- **Units:** IV is a decimal (`0.152` = 15.2%); IV changes/spreads/skew are decimal differences; premium and delta in USD; ratios in [-1, 1]; `classifiedPct*` in 0–1.
- **Sentiment** = `netDirectional / gross`, where `gross` is premium of *eligible* (classified) trades only; `quality.classifiedPct*` show how much was dropped. Inside-spread prints count fractionally (aggressor score), so `bullish`/`bearish` are score-weighted.
- **LOW CONFIDENCE** (value still returned) when eligible premium < `$250K`, eligible trades < 25, or classified premium < 50% (env-tunable: `OPTIONS_FLOW_MIN_*`).
- **ATM IV = 30-day constant-maturity** (variance-interpolated across bracketing expirations, 0DTE excluded); `iv7d/iv30d/iv60d` use the same method. `skew25d30d` = 25Δ put IV − 25Δ call IV interpolated to 30 DTE.
- **IV 1D / 5D change** compare against the *final snapshot of the previous 1st / 5th stored market date* — intraday-vs-prior-close, not same-time-of-day. `percentile` needs ≥ 20 stored days (`percentileObs`), else `null`; it fills in as the worker accumulates history (no backfill from ThetaData).
- **Delta imbalance** uses each trade's nearest *prior* first-order Greek (same expiration/strike/right, ≤ 15 min; falls back to nearest either side within the same tolerance). Put deltas stay negative.
- **Open interest** is context only (`contractsToOi`, `premiumPerOiContract`); it never feeds sentiment.
- **Retention:** every intraday run for 5 days, then only the last snapshot per ticker per market date. No raw ticks are stored.
