# FX currency-strength project — status

Last updated: 2026-09-11. Written after the first cron run surfaced a batch of
issues, a full backend fix pass, and a follow-up frontend fix pass. See
`docs/FX-BACKEND-IMPLEMENTATION.md` for the detailed rationale behind every
backend deviation from `docs/FX-BACKEND-SPEC.md` — this file is the current
snapshot of "where things stand," not the design record.

## Where the code lives

| Repo | Path | Deploy target |
|---|---|---|
| Backend (FastAPI + Postgres) | `macro_engine` (this repo) | Railway project **practical-vision** |
| Frontend (Next.js) | sibling repo `macro-engine-web` (`https://github.com/dtaylor2425/macro-engine-web`) | `www.macro-engine.com` — confirmed auto-deploys from `main` (live-checked, current commit's matrix/breadth fixes are rendering) |

Railway resources (project `practical-vision`, environment `production`):
- **macro-engine** — the API service, `https://api.macro-engine.com`
- **forex job** — cron `0 6 * * 1-5`, runs `jobs/fx_snapshot_job.py`
- **Postgres** — snapshot storage (`fx_snapshots` table, one row per `as_of`)
- other cron jobs (`vol job`, `portfolio-admin-job`, `stock-intelligence-job`) are unrelated to FX

**Important:** the API service and the cron job are separate Railway
resources with no shared filesystem. Anything that depends on a local disk
cache (Frankfurter parquet, FRED parquet) is only ever populated inside the
cron job's container — the API process never has it. See "Known
architecture constraint" below.

## Model shape, as currently live

Five scored components (not six — see below), equal-ish weights:

| Component | Weight | Source |
|---|---|---|
| carry | 31.25% | BIS policy rate (or FRED money-market proxy) + govt yield + real rate |
| policyMomentum | 31.25% | Δ govt yield, 1m/3m |
| termsOfTrade | 12.5% | export-weighted commodity basket, 3m return |
| trend | 12.5% | static trade-weighted FX index momentum |
| valuation | 12.5% | BIS REER vs 10y average, sign-inverted |

`macroVsMandate` (CPI YoY vs. central-bank target) was **removed from the
model entirely**, not just dropped-per-run: the OECD MEI CPI feed on FRED is
dead for every scored currency except USD/EUR (500+ days stale), so it could
never score above the 6-currency minimum. CPI YoY / CB target / mandate gap
are still shown as unscored context (`mandate` field) for USD and EUR only.

10 scored currencies (USD EUR JPY GBP CHF CAD AUD NZD SEK NOK) + CNY
display-only. Universe, weights, and thresholds all live in
`src/fx/series_map.py` / `src/fx/scoring.py`.

## Data sources (four, as of today)

1. **FRED** — policy rates (money-market proxy, fallback only now), govt
   yields, CPI, commodities. `src/fx/fred_client.py`, `src/fx/series_map.py`.
2. **Frankfurter** — daily ECB reference fixings, spot/cross rates, trend
   index, breadth module. `src/fx/frankfurter.py`.
3. **BIS REER** — real effective exchange rate for `valuation`.
   `src/fx/reer.py`.
4. **BIS central bank policy rates** (added today) — the actual announced
   rate per currency, replacing FRED's money-market proxy
   (`IRSTCI01*`/`IR3TIB01*`) as the primary source for the displayed policy
   rate and the carry input. `src/fx/policy_rates.py`. Every currency entry
   carries `policyRateIsProxy` / `policyRateInstrument` so it's always clear
   which source served that value.

## Coverage, as of the last live run (2026-09-11)

```
carry          10/10   policyMomentum 10/10   termsOfTrade 10/10
trend          10/10   valuation      10/10
incompleteCurrencies: []      droppedComponents: []
```

Series validation: 31 ok / 10 stale / 0 failed (404s: none). The 10 stale are
all CPI feeds for the 8 non-USD/EUR currencies plus CNY — the systemic OECD
CPI gap described above, expected to stay stale until OECD resumes the feed
or a paid replacement is found.

Validated against real production data, not just synthetic tests:
- `pairs`: 45 entries, antisymmetric, self-pair zero (test + live-checked)
- `breadth`: embedded as the full **g10** universe (all 10 currencies, not
  the 7-currency "majors" subset) — see below for why
- exact-identity invariant (`strength(a) - strength(b) == log(a/b)`, <1e-10):
  passes both in tests and against live data

Test suite: 31/31 passing (`tests/test_fx.py`, `tests/test_fx_breadth.py`).

## What was actually wrong vs. what was suspected

The original fix-list assumed the backend wasn't producing pairs/breadth
data at all. It was — the real defects were narrower and, in most cases, in
the *other* repo:

| Symptom | Actual cause | Layer |
|---|---|---|
| Validator reported 33/41 series stale | Validator used a flat 60-day rule instead of series_map's per-frequency thresholds | backend |
| EUR policyMomentum always unavailable | EUR's y10 (`IRLTLT01EZM156N`) is a genuinely dead FRED feed, not just normal lag | backend |
| CHF/CAD "policy rate" showed money-market noise, not the announced rate | No real-policy-rate source was wired in at all | backend |
| SEK/NOK/NZD breadth always empty | Snapshot only embedded the 7-currency "majors" breadth universe, and the API's live-recompute fallback path never works in this Railway topology (see below) | backend |
| Pair matrix always blank | Frontend called `/api/fx/pairs` with no `base` (422 — that param is required); the full triangle was already sitting unused on the snapshot it had already fetched | **frontend** |
| Matrix populated but lower-left triangle still blank | `pairMap()` only stored the single direction the backend sends (one triangle) — never built the antisymmetric mirror for the reverse orientation every other cell needs | **frontend** |
| Currency board's Momentum column always "—" | Read component key `'momentum'` — the real key is `policyMomentum`; there has never been a `momentum` key anywhere in the backend | **frontend** |
| Detail page component list missing termsOfTrade, showing a dead macroVsMandate row, never marking anything "dropped" | `COMPONENT_ORDER` was stale (pre-dated today's 5-component model) and read `raw.dropped`, a field the backend never sends (it sends `available`) | **frontend** |
| All four breadth sub-panels blank/wrong | Frontend's `normaliseBreadth()` read field names (`beatCount`, `payload.heatmap`, `payload.breadthSeries`, etc.) that don't exist anywhere in the backend's actual response shape | **frontend** |
| "Not enough data" shown even when data existed | Same cause — `beatCount`/`totalCount` were always `undefined` | **frontend** |

## Known architecture constraint (not fixed, by design choice)

`GET /api/fx/breadth` tries to recompute on demand from a local Frankfurter
disk cache (`frankfurter.load_cached()`) so arbitrary `base`/`horizon`/
`universe` combinations stay cheap. In this deployment, that cache is never
present in the API process (separate Railway resource from the cron job), so
this path silently falls back to whatever's embedded in the last published
snapshot. That's why the embedded snapshot was switched to the full g10
universe today — it's the *only* breadth data that reliably reaches any
consumer. Custom on-demand queries (e.g. a horizon the embedded snapshot
didn't compute) will keep silently falling back too. Options if this needs
to be fully solved: attach a shared Railway volume between the two services,
or move the Frankfurter fixing history into Postgres so the API can recompute
without touching local disk.

## Open items (identified, not yet done)

1. **Breadth history is missing rms, trendShare, and cross-level detail.**
   `compute_breadth_snapshot()`'s history loop (`src/fx/breadth.py:242-248`)
   computes all nine `base_metrics()` fields at every historical date but
   only retains three (`share`, `breadth`, `cumulativeAD`) — `rms`,
   `trendShare`, and `crosses` are computed and discarded every iteration.
   This is why the frontend's "Cross heatmap" and the rms/trendShare lines
   on the other two charts are honestly empty rather than wired up: the data
   doesn't exist yet, full stop. Fix is a few more `.append()` calls in that
   loop, not a redesign — flagged, not yet done pending a go-ahead.
2. ~~Frontend hosting unconfirmed~~ **Resolved.** `www.macro-engine.com` auto-deploys
   from `macro-engine-web`'s `main` on push; live-checked in a real browser
   after each frontend fix and confirmed rendering (matrix cells populated,
   network calls hitting the correct API endpoints).
3. ~~Cache-invalidation quirk on series remap~~ **Resolved.** `build_fx_fred_frame()`
   now keys the FRED disk cache (and the live fetch) by `"{logical}::{seriesId}"`
   instead of the bare logical name, then renames back to the logical name
   before returning. Remapping a logical name to a different series ID (as
   done for EUR) now lands in a brand-new column automatically — the old
   column is simply abandoned in the parquet (harmless), and there's no more
   need to remember `--no-cache` after a remap.
4. **BIS bulk URLs are unversioned and have moved before** (both
   `WS_CBPOL_csv_row.zip` and `WS_EER_csv_row.zip`, and the FRED series IDs
   in `series_map.py` generally) — re-run `scripts/validate_fx_series.py`
   periodically; nothing here is guaranteed to stay resolvable.
