# Options Flow — historical backfill

`jobs/options_flow_backfill.py` rebuilds one derived DAILY snapshot per (ticker, trading day) into the
same tables the live worker uses, tagged `source = 'historical_backfill'`. It runs in the same
Python 3.12 worker image (`Dockerfile.theta-worker`); run it as a one-off job, not on the cron.

```
# phase 1 (default universe = SPY,QQQ,IWM,SMH,TLT,GLD; default window = last 60 sessions)
python jobs/options_flow_backfill.py --tickers SPY,QQQ,SMH,IWM,TLT,GLD --start 2026-06-01 --end 2026-09-20
python jobs/options_flow_backfill.py --tickers SPY --days 60
python jobs/options_flow_backfill.py --ticker SPY --days 5 --dry-run          # do this first
python jobs/options_flow_backfill.py --days 60 --resume                       # continue after a kill
python jobs/options_flow_backfill.py --tickers SPY --days 60 --overwrite      # rebuild backfill rows only
python jobs/options_flow_backfill.py --days 60 --resume --retry-partial       # also rebuild 'partial' days
# phase 2, only after phase 1 looks right
python jobs/options_flow_backfill.py --tickers all --days 60
```

Flags: `--tickers`/`--ticker`, `--start`/`--end` or `--days N`, `--resume`, `--overwrite`
(mutually exclusive), `--retry-partial`, `--dry-run`, `--max-dte`, `--allow-large`, `--mode`.
Without `--resume` or `--overwrite`, a range that already has backfill rows is refused rather than guessed at.
`--allow-large` is required for > 6 tickers with > 126 sessions, or > 1,500 ticker-days.
Exit codes: `0` all published/present · `1` nothing published or fatal · `2` some published, some failed · `130` interrupted.
A fully-satisfied `--resume` (nothing left to fetch) never constructs a ThetaData client at all.

## IV-warmup mode (`--mode iv-warmup`)

For populating IV percentile history cheaply, without touching the trade tape, ahead of a research period:

```
# warmup: the 252 sessions before the research window
python jobs/options_flow_backfill.py --tickers SPY,QQQ,IWM,SMH,TLT,GLD --days 252 --end 2026-06-19 --mode iv-warmup
# research: the following 60 (or 252) sessions, full trade tape
python jobs/options_flow_backfill.py --tickers SPY,QQQ,IWM,SMH,TLT,GLD --start 2026-06-22 --end 2026-09-18
```

`iv-warmup` fetches **only first-order Greeks** (no `trade_quote`, no `open_interest`) and stores ATM/7D/30D/60D IV,
the 7D-30D and 30D-60D term spreads, 25-delta put skew, spot and the term structure -- built with
`options_flow_metrics.build_iv_observation`, the *same* `iv_block()` full-flow days use, so a warmup day and a
full-flow day agree given the same Greek observations. It writes into the *same* tables and the *same*
(ticker, market_date) backfill slot as full-flow, tagged `mode = 'iv_warmup'` (full-flow rows are `mode = 'full_flow'`,
independent of `source`). `sentiment`, `premium`, `delta`, `dte`, `aggression`, `openInterest`, `intraday` and
`largeTrades` are explicitly `null` for a warmup day -- never a manufactured value. A later full-flow run for a
warmed-up date needs `--overwrite` to upgrade it, same as rebuilding any backfill day.

Because `load_atm_series` / `build_iv_history` read the ATM-IV column regardless of mode, **warmup observations
feed the research period's percentiles transparently** -- the first evaluation day already sees up to 252 prior
sessions once warmup has run. Per the point-in-time rule (below), a **warmup-only date's own row never itself
gets a populated percentile** unless further ITS OWN 20/60/126/252-session history exists; that is expected and
is why warmup and research periods should not overlap for the analysis you actually care about -- exclude
`mode = 'iv_warmup'` rows from a predictive-backtest sample for this reason.

Cost: 1 dated contract list + 1 Greeks request per expiration (no trades, no OI) --
`estimate_requests(ticker, max_dte, n_dates, mode="iv_warmup")` prints roughly half of full-flow's count, and each
request is far cheaper (a Greeks response is tiny next to a full day's trade tape).

## Guarantees

- **Same math as live:** each ticker-day goes through `process_ticker` -> `options_flow_metrics` (the backfill
  file has no analytics). Differences are provenance only (`payload.source`, `quality.dataStatus/missingReasons`).
- **Point in time:** date *t* uses only *t*'s own Theta data — contract universe from the **dated** contract list for
  *t* (never today's expirations; `max_dte` is applied client-side relative to *t*) — plus prior daily IV observations
  from the sessions strictly before *t* (`store.build_iv_history`). Sessions still trading (incl. the 16:00–16:15
  options tail) are never backfilled.
- **Rolling IV:** `atmChange1d/5d/20d` compare against the 1st/5th/20th prior *trading session* (a missing session
  is `null`, never bridged). `percentiles.{20d,60d,126d,252d}` rank **today's own observation within the trailing
  window that ends at and includes today** -- i.e. the (n-1) prior sessions plus today -- and are `null` unless
  the window has at least `{20d: 15, 60d: 45, 126d: 95, 252d: 190}` total observations (including today's own).
  `percentile`/`percentileObs` (all prior observations, min 20, unbounded window) are unchanged. After each
  ticker the job re-states these fields on that ticker's backfill rows from the stored IV series, so filling an
  earlier gap (e.g. via iv-warmup) corrects later days -- this is also how a research period picks up warmup history.
- **Nothing manufactured:** no trade data -> the day is **not published** (a `failed` audit run records the reason,
  e.g. `no_trade_data`, `no_expirations_listed`, `error: ...`). Trades but missing Greeks / OI / low Greek match ->
  published as run status `partial` with `quality.missingReasons` and null delta/IV fields. Zeros are measured zeros only.
- **Safe to interrupt:** one transaction (run + snapshot) per ticker-day, ticker-major and chronological.
- **Live untouchable:** `--overwrite` deletes only `source = 'historical_backfill'` rows; a failed rebuild leaves the
  previous snapshot in place. Latest-snapshot selection prefers `live` over backfill for the same market date, and the
  status endpoint reports only the live worker's runs.

## Schema (`sql/006_options_flow_backfill.sql` + `sql/007_options_flow_iv_warmup.sql`, both idempotent)

`options_flow_runs.source` and `options_flow_symbol_snapshots.source` (`'live'` | `'historical_backfill'`, default
`'live'` so existing rows are unchanged) and a **partial unique index** `(ticker, market_date) WHERE source =
'historical_backfill'` — one backfill row per ticker-day, enforced by the database, regardless of mode.
`options_flow_runs.mode` / `options_flow_symbol_snapshots.mode` (`'full_flow'` | `'iv_warmup'`, default
`'full_flow'`) is independent of `source`. Backfill runs are one per ticker-day (`config.ticker`); `diagnostics`
holds the per-ticker-day audit data -- full-flow: `tradesDownloaded, eligibleTrades, classifiedPctTrades/Premium,
greekMatchedPctPremium, oiMatchedPctPremium, contractsTraded, grossPremium, expirations, thetaRequests, runtimeSec,
stageSeconds, missingReasons, warnings, error`; iv-warmup: `expirations, contracts, atmIv, skew, thetaRequests,
runtimeSec, stageSeconds, missingReasons, warnings, error`.

## Runtime profiling and the QA report

Every run times, per ticker-day: `contract_lookup, trade_quote_fetch, greeks_fetch, open_interest_fetch, normalize,
classify, greek_asof_join, iv_snapshot, aggregate, postgres_write` (pure observation via
`options_flow_metrics.StageTimer` -- it never changes a result; `_NullTimer` is a no-op default elsewhere). At the
end of any non-trivial run (`jobs/options_flow_backfill.py`, not the live worker) a **QA report** prints: successful
/ partial / failed ticker-day counts, wall-clock runtime, ThetaData requests per ticker-day (median/mean/total),
Postgres size added (`pg_database_size` before/after), median seconds per stage, and (full-flow only) median/min
classification, Greek-match and OI-match coverage, plus missing-IV / missing-skew / missing-all-four-percentiles
counts (any mode). Contract lookup is already requested exactly once per ticker-day and reused for OI's
per-expiration requests (see the current-day OI fix below); no further obvious duplicated ThetaData calls were
found to remove.

## Cost (measured on real ThetaData Standard, SPY, max_dte=60, strike_range=25)

Per ticker-day: 1 dated contract list + 1 open-interest + 2 x (expirations traded within `max_dte`) requests
(trade_quote + first-order Greeks per expiration). Measured: **~16 expirations, 34 requests per SPY day**
(49 for the current day, which needs per-expiration open interest), i.e. **~2,000 requests for 60 days** of SPY.
Volume, not request count, is the cost: ~1.1-1.5M SPY option trades/day, **~2-3 minutes per SPY day** (~2.5 h for 60 days).
The printed plan shows an estimate (accurate for SPY-like tickers, a guess for others) before anything runs.

## Verified against real ThetaData

- Column names for historical trade_quote / first-order Greeks / open interest match the live path; no normalizer errors.
- The dated contract list (`option_list_contracts("trade", date, ...)`) works and returns `expiration`.
- Real full-flow output, SPY 2026-09-21: 1,467,080 trades, 99.99% premium classified, 100% Greek-matched, 94% OI-matched,
  30D ATM IV 12.3%, 25-delta put skew +2.7 vol pts, 34% of premium in 0DTE.
- ThetaData rejects wildcard-expiration open interest for the CURRENT day; the fetcher now requests it per expiration.
  (This also affected the live worker, which never had OI on the day it ran.)
- One real full-flow write to Railway Postgres (SPY 2026-09-18): one `historical_backfill` run, `status='success'`,
  ATM IV / skew / sentiment / delta imbalance / OI coverage all present, and `--resume` on the same range correctly
  skipped the day without constructing a ThetaData client at all (a real bug this caught and fixed -- see below).
- One real iv-warmup write to Railway Postgres (SPY 2026-07-15): `mode='iv_warmup'`, ATM 13.0%, skew +4.3%, and
  `sentiment`/`premium`/`delta`/`dte`/`aggression`/`openInterest` all explicitly `null` (not fetched, not manufactured).
- **Real stage-timing profile** (SPY 2026-09-17, full-flow, wall clock 139s): `trade_quote_fetch` 211.8s and
  `greeks_fetch` 188.9s of cumulative thread time dominate (parallel across up to 4 expirations, so they exceed wall
  clock); local processing (`normalize` 9.8s + `classify` 5.2s + `greek_asof_join` 19.1s + `iv_snapshot` 13.5s +
  `aggregate` 10.1s) totals ~58s, an order of magnitude less. **Network I/O is the cost, not local computation** --
  no further duplicated-work optimisation was found beyond the current-day OI fix (already applied). The same day
  in iv-warmup mode (Greeks only) ran in 43s wall clock, ~3x faster.
- **Bug this validation found and fixed:** `--resume` unconditionally constructed a `ThetaFetcher` (and therefore
  authenticated to ThetaData) even when every requested ticker-day was already published and there was nothing to
  fetch. Fixed: the fetcher is now constructed only when at least one ticker-day actually needs fetching.

## `strike_range` and zero-trade-expiration findings (SPY, 2026-07-15, an older date)

- **`strike_range` is centered on that date's own spot**, not today's: with `strike_range=25`, `option_history_trade_quote`
  for the nearest expiration returned strikes $730–779 around a probe-measured spot of $754.47 -- $24.47 below,
  $24.53 above, 50 one-dollar strikes. Confirmed working as point-in-time intended; nothing to change.
- **Zero-trade expirations:** of the 8 nearest dated-contract expirations checked, every one had both `trade_quote`
  rows and Greek observations that day (`listedNoTrades: []`, `listedNoGreeks: []`, `greeksWithoutTrades: []`).
  On this sample the dated contract list, trades and Greeks agree, so a trades-vs-Greeks build choice made no
  observed difference here -- but this was checked on a liquid, near-dated SPY expiration, not proven for
  far-dated or thin expirations. `build_iv_observation` (iv-warmup mode) already builds the term structure from
  Greeks alone with no trade requirement, which is the safer choice regardless.

## Still unverified

1. ThetaData's historical Greeks use the rate/dividend inputs of that date (ETF dividend assumptions are
   ThetaData's and cannot be audited here) -- documented as a provider-model assumption, not a blocker.
2. Open interest for date *t* is the start-of-day figure (published ~06:30 ET), i.e. known before *t*'s trades.
3. Whether a far-dated or thin expiration can be Greeks-only or trades-only (checked only on 8 near-dated,
   liquid SPY expirations -- see finding above).
