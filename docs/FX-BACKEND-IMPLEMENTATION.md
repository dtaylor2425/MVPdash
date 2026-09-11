# FX backend — implementation notes

Implements `docs/FX-BACKEND-SPEC (1).md` (composite score + section 4A breadth
module) against this repo's actual stack: **Python / FastAPI / Postgres**, not
the Next.js layout the spec document assumes. File mapping:

| Spec | This repo |
|---|---|
| `src/lib/fx/seriesMap.js` | `src/fx/series_map.py` |
| `scripts/validateFxSeries.js` | `scripts/validate_fx_series.py` |
| `src/app/api/fx/*` | `api/routers/fx.py` |
| snapshot writer + cron | `jobs/fx_snapshot_job.py` |
| — | `sql/002_fx_snapshots.sql`, `src/fx/*`, `tests/test_fx*.py` |

## Layout

```
src/fx/
  series_map.py    FRED series IDs, CB targets, commodity/trend baskets, staleness rules
  frankfurter.py    Frankfurter client (fixings, cached to disk)
  fred_client.py    FRED fetch wrapper + validator (scripts/validate_fx_series.py)
  reer.py           BIS REER bulk CSV loader (optional component)
  components.py     the 6 composite-score component calculators
  scoring.py        cross-sectional z, winsorise, reweight, composite, pairs, flags
  reconciliation.py FX USD score vs the existing macro `dollar` signal
  breadth.py         section 4A: independent strength/breadth/attribution model
  snapshot.py       orchestrator -> the full /api/fx/snapshot payload

jobs/fx_snapshot_job.py     ingest + score + publish (Postgres + disk fallback)
api/routers/fx.py           GET /api/fx/{snapshot,currency/{code},pairs,breadth,status}
sql/002_fx_snapshots.sql    fx_snapshots table (payload JSONB, one row per as_of, retained)
scripts/validate_fx_series.py
tests/test_fx.py            spec section 7 checks, synthetic data, no network
tests/test_fx_breadth.py    spec section 4A.7 checks, synthetic data, no network
```

Run the job: `python jobs/fx_snapshot_job.py` (add `--no-db` to skip Postgres,
`--dry-run` to build without publishing). Recommended Railway cron: weekly,
after the ECB fixing — `0 6 * * 1`.

Run tests: `python tests/test_fx.py && python tests/test_fx_breadth.py`
(also work under `pytest tests/`, if installed — it isn't in this repo yet).

## Deliberate departures from the spec's literal text

**1. Staleness must be frequency-aware, or the fallback that makes the
system honest becomes the thing that breaks it.**

The spec's `get_fred_cached` (existing repo code) forward-fills every series
so downstream consumers see a contiguous daily frame. If `components.py` read
values off that ffilled frame the way the original composite spec implied, a
FRED series that died eight months ago would keep silently reporting its last
real value forever — exactly the anti-pattern spec section 3.3 warns about
("a silently dead series ID... looks plausible while being wrong"), just
introduced by the *fix* for it rather than by a typo.

So `fred_client.build_fx_fred_frame()` returns the **raw, un-forward-filled**
observations (reading back the pre-ffill parquet `get_fred_cached` already
writes to disk — no extra network call), and every component lookup in
`components.py` checks how old the most recent real observation is via
`series_map.max_age_for()` before using it. A daily series older than 10 days,
or a monthly one older than 120, is treated as unavailable, not stale-but-used.

The spec's literal "no observation in the last 60 days for a daily or monthly
series" is kept as the **validator's** rule (`fred_client.py`,
`scripts/validate_fx_series.py`) — a single fixed number makes for an honest
diagnostic report. It is deliberately *not* the same number the scoring
engine uses, because 60 days flags routine OECD publication lag (verified:
most `IRLTLT01*`/`IRSTCI01*` series run ~100 days behind on a normal month)
as if it were a dead feed. Using 60 days for both would have dropped
`carry`/`policyMomentum` for 8 of 10 currencies on every run.

**2. Several series IDs in the spec's own candidate table are dead or wrong;
replacements were found and verified live against FRED on 2026-09-10** (see
`src/fx/series_map.py` comments on each entry):

- `EUR` policy rate: switched from `IRSTCI01EZM156N` (frozen since 2026-01) to
  `ECBDFR` (ECB deposit facility rate — live, daily).
- `CHF`/`NZD`/`SEK` policy rate: `IRSTCI01[CC]M156N` is dead for these three
  (frozen since 2020–2024); switched to `IR3TIB01[CC]M156N` (3m interbank,
  live).
- `CNY` 10y yield: the spec's own candidate (`INTGSTCNM193N`) is a 404.
  Set to `None` — CNY is display-only, never scored, so this only affects
  what's shown on its context card, not any score.
- `EUR` CPI: switched to `CP0000EZ19M086NEST` (HICP index, live, ~2mo lag)
  from the OECD MEI family.

**A systemic gap, not a bug:** OECD's MEI CPI feed on FRED for every scored
currency except USD and EUR appears to have stopped updating around
February 2025 (every candidate — `CPALTT01[CC]M659N`, the `*CPIALLMINMEI`
family, and their quarterly counterparts for AUD/NZD — reads 500+ days stale
for JPY/GBP/CHF/CAD/AUD/NZD/SEK/NOK/CNY). No free replacement was found.
Given the 6-of-10 minimum, `macroVsMandate` and the real-carry sub-input
correctly get dropped and reweighted on essentially every current run — the
model is not guessing values for the missing 80%, per spec's own instruction
never to silently fill. `meta.droppedComponents` and `meta.seriesHealth`
report this plainly; re-run `scripts/validate_fx_series.py` periodically in
case OECD resumes the feed or a replacement source appears.

**3. BIS REER bulk file:** the spec's assumed URL pattern
(`www.bis.org/statistics/full_eer_d_csv_row.zip`) 404s. The live path is
`https://data.bis.org/static/bulk/WS_EER_csv_row.zip`, and its actual layout
is a wide CSV with metadata carried in the first several *rows* (Frequency /
Type / Basket / Reference area / ... / Title) rather than columns —
`src/fx/reer.py` parses that literally. Verified live: all 10 scored
currencies resolve, monthly, back to 1994. If BIS reorganises this again,
`load_reer()` degrades to `{}` (never raises) and `valuation` drops and
reweights, per spec 3.4.

**4. Frankfurter's actual response shape.** `api.frankfurter.dev/v2/rates`
returns a flat list of `{date, base, quote, rate}` records for both the
"latest" and history forms — not the nested `{"rates": {...}}` shape the
spec's own example implies (that's the *v1* Frankfurter API shape).
`frankfurter._parse_rates_payload()` handles the real (list) shape, with the
nested shape kept as a defensive fallback in case a future deploy reverts it.

**5. Breadth route reads a disk cache, not a live Frankfurter call.**
`GET /api/fx/breadth` needs to answer arbitrary `base`/`horizon`/`universe`
combinations cheaply, and section 5's "never call FRED or Frankfurter on
request" has to mean something concrete here. The resolution:
`jobs/fx_snapshot_job.py` is the only thing that ever calls Frankfurter; the
route reads back the same parquet cache the job already wrote
(`frankfurter.load_cached()`, no network) and recomputes the requested
combination from data already in memory — cheap, and genuinely zero calls to
either upstream source on any request path.

## What "publishable" means here

`snapshot_is_publishable()` implements spec section 5's abort rule (>3 of 10
currencies incomplete) and is checked by the job before every write. A build
that throws entirely (both sources unreachable) does not touch the previous
snapshot — the API's staleness check (`age > 8 days`) is what marks it
`stale: true` / `staleSince` for the UI, per spec.

## Known gaps for a follow-up pass

- `growth/labour surprise` for `macroVsMandate` (spec 4.1: "if available") is
  not implemented — no free, timely surprise-index source was identified.
- `change1w` is computed by re-scoring at `asOf − 7d`, not from a stored
  prior week's snapshot — fine once the job has run for a few weeks, but the
  first several runs will show a `change1w` computed against a
  freshly-backfilled point rather than an actually-published one.
- `trend`'s "trade-weighted" partner weights (`series_map.py`,
  `TREND_PARTNER_WEIGHTS`) are hand-set approximations, not sourced from an
  official trade-share table — flagged `proxy: true` throughout, as the spec
  requires for any non-literal input.
