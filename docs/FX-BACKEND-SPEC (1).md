# Macro Engine — FX Currency Score: Backend Specification

Build the data ingestion, scoring model, and API routes for a new Currency Strength
page. This document covers **backend only**. A separate document covers the UI.

---

## 1. What we are building and why

A per-currency composite score for G10 currencies, plus a derived score for all
currency pairs.

The critical design point: **a currency has no absolute value.** Growth and
inflation have levels; a currency only exists as a ratio against another
currency. So we compute a standalone composite per currency, then derive every
pair as the *difference* between two composites. Ten currency scores produce 45
pair signals from one model.

Do not build a model that scores pairs directly. It does not generalise and the
matrix will not be internally consistent.

---

## 2. Universe

**Scored (10):** USD, EUR, JPY, GBP, CHF, CAD, AUD, NZD, SEK, NOK

**Display-only (1):** CNY — managed float. Show the data, do not score it, and
label it "managed — not scored" in the payload with `scored: false`. Scoring a
managed currency with a free-float model produces confident nonsense.

---

## 3. Data sources

Two sources only. Both free, no paid tier.

### 3.1 FRED (already integrated in this codebase)

Covers policy rates, government yields, CPI, commodity prices, and the broad
dollar index. Reuse the existing FRED client and API key handling.

### 3.2 Frankfurter — `https://api.frankfurter.dev/v2/rates`

FX spot and history. No API key, no quota, ECB and other central-bank sources,
history back to 1999.

```
# Latest, USD base, selected quotes
GET https://api.frankfurter.dev/v2/rates?base=usd&quotes=eur,jpy,gbp,chf,cad,aud,nzd,sek,nok

# Time series from a date
GET https://api.frankfurter.dev/v2/rates?base=usd&from=2024-01-01&quotes=eur,jpy

# Monthly downsample for long series
GET https://api.frankfurter.dev/v2/rates?base=usd&from=2020-01-01&group=month
```

**Important characteristic:** Frankfurter returns **daily central-bank reference
fixings, not live quotes.** For a weekly-refreshed score this is correct and
preferable — fixings are consistent and do not jitter. But every payload must
carry the observation date, and the UI must label it as a fixing. Never present a
fixing as a live rate.

Pegged currencies: Frankfurter derives those rows from the peg rather than
provider data. Not relevant to G10, but do not silently treat derived rows as
market data if the universe is ever extended.

### 3.3 Series ID handling — read this carefully

**Do not hardcode FRED series IDs from memory or from this document without
verifying them.** FRED's international series, particularly OECD-sourced policy
and long-term rate series, are periodically discontinued or renamed. A silently
dead series ID produces a currency that scores on four components instead of six
and looks plausible while being wrong.

Required implementation:

1. Put every series ID in a single config file: `src/lib/fx/seriesMap.js`
2. Write `scripts/validateFxSeries.js` that requests each ID from FRED and
   reports: OK, 404, or stale (no observation in the last 60 days for a daily
   or monthly series).
3. Run it as part of the ingest job. If a series fails, the affected component
   must be marked unavailable for that currency and the composite reweighted
   across the surviving components — never silently zero-filled.
4. Log every failure with the currency, component, and series ID.

Starting candidates to verify, not to trust:

| Input | Candidate | Confidence |
|---|---|---|
| US 2y yield | `DGS2` | High |
| US 10y yield | `DGS10` | High |
| Fed funds | `DFF` / `FEDFUNDS` | High |
| US CPI | `CPIAUCSL` | High |
| Broad dollar index | `DTWEXBGS` | High |
| Brent | `DCOILBRENTEU` | High |
| Copper | `PCOPPUSDM` | Medium |
| Foreign policy rates | OECD/BIS series per country | **Low — verify each** |
| Foreign 2y / long yields | `IRLTLT01[CC]M156N` family | **Low — verify each** |
| Foreign CPI | `CPALTT01[CC]M659N` family | **Low — verify each** |

Where a foreign 2y yield genuinely has no free source, fall back to the
long-term government yield and record `proxy: true` on that component so the UI
can disclose it.

### 3.4 REER (optional for v1)

BIS publishes real effective exchange rate indices as a free bulk CSV, updated
monthly. There is no API. If included, fetch monthly, cache to disk, and treat
absence as an unavailable component rather than a hard failure.

---

## 4. The scoring model

### 4.1 Components and weights

Each component is computed per currency, converted to a z-score across the
10-currency cross-section, then combined.

| Component | Weight | Inputs |
|---|---|---|
| `carry` | 25% | Policy rate; 2y yield; real policy rate (policy rate − YoY CPI) |
| `policyMomentum` | 25% | Change in 2y yield over 1m and 3m |
| `macroVsMandate` | 20% | CPI YoY vs that central bank's own target; growth/labour surprise if available |
| `termsOfTrade` | 10% | Export-weighted commodity basket return, 3m |
| `trend` | 10% | Trade-weighted index momentum, 1m/3m/6m blend |
| `valuation` | 10% | REER deviation from 10y average, **sign inverted** |

Notes on intent:

- `policyMomentum` is the most predictive component and the hardest to source.
  What moves FX is the *repricing of the expected rate path*, not the level.
  Proper OIS curves are not free; change in 2y government yield is the proxy.
  Record `proxy: true` and disclose it in the methodology text.
- `macroVsMandate` must compare each country's inflation against **its own
  central bank's target** (ECB 2%, Fed 2% PCE, BoJ 2%, RBA 2–3% midpoint, SNB
  0–2%, etc.). Put the targets in config. Do not compare every country to 2%.
- `valuation` is inverted deliberately: an expensive currency scores *negative*.
  It is a mean-reversion brake on the carry signal, which otherwise keeps you
  long a currency 20% above fair value right up until it unwinds.
- `termsOfTrade` matters disproportionately for NOK, CAD, AUD, NZD. Without it
  the commodity currencies mis-score badly.

### 4.2 Z-score methodology

- Cross-sectional z-score across the 10 scored currencies for each component,
  per observation date.
- **Winsorise at ±2.5 sigma** before combining. One currency in a policy crisis
  will otherwise dominate the entire cross-section.
- Require a minimum of 6 currencies with valid data for a component; below that,
  drop the component and reweight.
- Final composite: weighted sum of component z-scores, then rescaled to a
  **0–100 display score** where 50 is neutral, using a fixed sigma mapping
  (`score = 50 + 12.5 * composite_z`, clamped 0–100). A fixed mapping keeps
  scores comparable week to week; a percentile rank does not.

### 4.3 Pair derivation

For every ordered pair (BASE, QUOTE):

```
pairScore = baseComposite - quoteComposite     // in z-space, not display space
pairDisplay = 50 + 12.5 * pairScore            // clamped 0-100
```

Matrix is antisymmetric: `pair(EUR,USD) = -pair(USD,EUR)` in z-space. Compute one
triangle and mirror it. Never compute both independently — they will drift.

### 4.4 Flags

Attach these to each currency and pair. They are what separate a useful model
from a number.

- `carryTrendConflict` — true when `carry` z > +0.5 and `trend` z < −0.5. This is
  the classic pre-unwind setup. Surface it as a distinct state rather than
  letting a blended score hide it.
- `interventionRisk` — hard-code a watch flag for JPY and CHF, raised when the
  currency's 1m trend z is beyond ±1.5. A model that goes maximum-long a currency
  the authorities are actively defending looks naive.
- `dataIncomplete` — true when any component was dropped for that currency.
- `proxyInputs` — array of component names computed from a proxy series.

### 4.5 Reconciliation with the existing dollar signal

The macro pillars already score the dollar (`dollar` signal, used on the rates
and growth pages). **The FX page's USD score and the existing dollar signal must
be visibly consistent.** Expose both in the snapshot:

```json
"reconciliation": {
  "fxUsdScore": 63,
  "macroDollarSignal": { "value": 118.9, "zscore": 0.62, "direction": 1 },
  "agreement": "aligned"      // "aligned" | "divergent"
}
```

Mark `divergent` when the FX USD score is above 55 while the macro dollar z-score
is negative, or vice versa. The UI will show it. Two models that contradict each
other in public is worse than one model.

---

## 4A. Breadth and attribution module

This is a **second, independent model** that runs alongside the composite score.
Do not merge them. The composite in section 4 is a forecast: what *should* happen
based on carry, policy and valuation. This module is realised attribution: what
*did* happen over the last N bars, and which currency caused it.

Both are needed. A pair trader's first question about any move is "was that the
base or the quote?" and the composite score cannot answer it.

Ported from a Pine Script implementation (CFR FX Breadth). The maths below is
exact, not approximate — implement it literally.

### 4A.1 Currency strength — the exact decomposition

For a universe of N currencies, using **log prices**:

```
strength(c) = (1/N) * SUM over all x in universe of log(c/x)      // log(c/c) = 0
```

Then for any pair:

```
log(a/b) = strength(a) - strength(b)        // holds exactly, zero residual
```

**Critical implementation detail:** divide by **N**, the count *including* the
currency itself, not N−1. The self term contributes 0 to the sum but must be
counted in the divisor, or the identity breaks. This looks like an off-by-one bug
and will get "corrected" into being wrong. Guard it with the test in 4A.6.

Proof, for the reviewer's benefit:
`strength(a) − strength(b) = (1/N) Σ_x [log(a/x) − log(b/x)] = (1/N) Σ_x log(a/b) = log(a/b)`

### 4A.2 Sourcing this from Frankfurter — cheap

The Pine version pulls 21 separate cross symbols because TradingView requires
market-convention tickers. We do not have that constraint. Fetch **one base**
and derive every cross:

```
log(a/b) = log(a/EUR) - log(b/EUR)
```

So a single request gives the entire matrix:

```
GET https://api.frankfurter.dev/v2/rates?base=eur&from={start}&quotes=usd,jpy,gbp,chf,cad,aud,nzd,sek,nok
```

One call, nine series, all 45 crosses derivable. Use `group=month` for long
history, daily for the trailing window. No FRED dependency for this module at all.

### 4A.3 Metrics, per base currency

Let `r_i` be the horizon log return of the base against each of the other
`M = N − 1` currencies (`r_i = x_i − x_i[horizon]`, where `x_i = log(base/other_i)`).

```
upN     = count of r_i >  threshold
downN   = count of r_i < -threshold
breadth = upN - downN                           // -M .. +M

meanR   = mean(r_i)
msq     = mean(r_i^2)
rms     = sqrt(msq)                             // "size of the moves"
share   = meanR^2 / msq                         // 0..1, fraction that is the base itself
```

`share` comes from the identity `mean(r²) = mean(r)² + var(r)`. The common
component is the base's own move; the variance is idiosyncratic moves in the
others. A share of 0.8 means four-fifths of all the movement across those crosses
*was* the base. A share of 0.15 means the base barely moved and the others were
doing their own thing.

`threshold` is a noise filter (default 0, configurable as a percentage) below
which a cross counts as unchanged for breadth purposes.

**Trend share** — the same ratio computed over a trailing window of 1-bar moves,
answering "is today's reading normal for this regime?":

```
d_i     = 1-bar log return of each cross
num     = M * mean(d_i)^2      per bar
den     = sum(d_i^2)           per bar
trendShare = rolling_sum(num, window) / rolling_sum(den, window)
```

**Cumulative advance/decline** — one step per bar, the sum of
`sign(d_i)` across the crosses, accumulated. Read the slope, not the level.

### 4A.4 The verdict taxonomy

Four states, from breadth extremity and share. Keep these labels — they are the
plain-English payoff of the whole module.

| Verdict | Condition | Meaning |
|---|---|---|
| `BASE_DRIVEN` | abs(breadth) ≥ M−1 **and** share ≥ 0.5 | The base moved against nearly everything, and it was the base doing it. Trade the base. |
| `BASE_PLUS_STRESS` | abs(breadth) ≥ M−1 **and** share < 0.5 | Base beat nearly everything, but one or two crosses did most of the work. Something idiosyncratic is happening in a counterpart. |
| `OTHERS_MOVING` | abs(breadth) < M−1 **and** share < 0.5 | The base is roughly still; the others are moving independently. Do not read a base signal into it. |
| `QUIET_MIXED` | otherwise | Nothing dominant. |

`BASE_PLUS_STRESS` is the most useful state and the one most models miss — it is
the signature of a single-currency event (intervention, a shock, a central-bank
surprise) hiding inside what looks like a clean base trend.

### 4A.5 Per-pair attribution

For every pair, alongside the forecast score from section 4.3:

```
move        = strength(base) - strength(quote)        // = log(base/quote), exact
fromBase    = strength(base)
fromQuote   = -strength(quote)
baseShare   = abs(strength(base)) / (abs(strength(base)) + abs(strength(quote)))
```

`baseShare` above 0.5 means the pair's move was mostly the base. This is a
magnitude attribution, deliberately distinct from the variance-based `share` in
4A.3 — do not conflate the two field names.

### 4A.6 Universe note

The Pine original uses 7 majors (USD, EUR, GBP, JPY, AUD, CHF, CAD). Our
composite universe is 10. The maths generalises for any N, but be aware that
adding SEK and NOK tilts the strength index toward the EUR bloc, since both track
EUR closely — a "strong EUR" week will mechanically look broader than it is.

Compute and expose **both**:

- `majors` — the 7-currency set, matching the original and cleaner for reading
- `g10` — all 10, consistent with the composite score

Default the UI to `majors`, offer `g10` as a toggle.

### 4A.7 Validation — non-negotiable

1. **Exact identity:** for every pair, `abs((strength(a) − strength(b)) − log(a/b)) < 1e-10`. If this fails, the divisor is wrong (see 4A.1).
2. **Strength sums to zero:** `SUM over all c of strength(c) === 0` within floating-point tolerance.
3. **Share is bounded:** `0 <= share <= 1` always. A value outside that means `msq` was computed before the mean, or a NaN leaked.
4. **Breadth bounds:** `-M <= breadth <= M`.
5. **Base-invariance:** computing strengths from a USD base and an EUR base must produce identical values.

### 4A.8 What not to port

The Pine file contains substantial TradingView-specific workarounds — stacked
plot bands at arbitrary y-offsets, invisible bound plots used as `fill()` anchors,
21 hardcoded symbol lookups, label arrays redrawn on `barstate.islast`. None of
that carries over. In React these are just separate components with real layout.
Port the **maths and the verdict taxonomy**, not the rendering strategy.

### 4A.9 Additional API surface

Extend the snapshot with a `breadth` block, and add one route.

```json
"breadth": {
  "universe": "majors",
  "horizon": 21,
  "trendWindow": 63,
  "threshold": 0.0,
  "strengths": { "USD": 0.0142, "EUR": -0.0031, "...": 0 },
  "byBase": {
    "USD": {
      "breadth": 4,
      "upN": 5, "downN": 1,
      "meanMove": 0.0118,
      "rms": 0.0164,
      "share": 0.52,
      "trendShare": 0.41,
      "cumulativeAD": 38,
      "verdict": "BASE_DRIVEN",
      "crosses": [
        { "quote": "JPY", "move": 0.0231, "fromBase": 0.0142, "fromQuote": 0.0089, "baseShare": 0.61 }
      ]
    }
  },
  "history": {
    "USD": { "share": [{ "date": "2026-08-01", "value": 0.44 }],
             "breadth": [{ "date": "2026-08-01", "value": 2 }],
             "cumulativeAD": [{ "date": "2026-08-01", "value": 12 }] }
  }
}
```

`GET /api/fx/breadth?base=USD&horizon=21&universe=majors` — returns the `byBase`
entry plus history, for the interactive panel. `horizon` and `universe` are
query-configurable because the UI exposes them as controls; everything else comes
from the nightly snapshot.

Compute `byBase` for all currencies in the snapshot job — it is 10 cheap
calculations over data already in memory.

---

## 5. Ingestion and storage

Follow the existing snapshot pattern used by the portfolio pages: **compute on a
schedule, publish a snapshot, never recompute on page load.**

- Nightly cron (or Vercel scheduled function) runs the ingest + score job.
- Output written as a single JSON snapshot, versioned by date.
- The API routes read the latest snapshot. They must not call FRED or
  Frankfurter on request.
- Roughly 30–60 series total. That is a trivial daily load against both sources.
- Retain history: every snapshot is kept so the published scores are auditable
  later. This matters — a timestamped record of calls is the point.

Failure policy:

- Partial data → publish with `dataIncomplete` flags set. Do not fail the whole
  snapshot for one dead series.
- Total source failure → keep serving the previous snapshot and set
  `stale: true` with `staleSince`. The UI has a state for this.
- Never publish a snapshot where more than 3 of 10 currencies are incomplete.
  Abort and alert instead.

---

## 6. API routes

All under `src/app/api/fx/`. Response shapes are contracts — the frontend spec
depends on these exact field names.

### `GET /api/fx/snapshot`

```json
{
  "asOf": "2026-09-09",
  "observationDate": "2026-09-08",
  "source": "ECB reference fixing via Frankfurter; rates and macro via FRED",
  "stale": false,
  "currencies": [
    {
      "code": "USD",
      "name": "US Dollar",
      "scored": true,
      "score": 63,
      "rank": 1,
      "change1w": 4,
      "components": {
        "carry":          { "z": 0.82, "display": 70, "proxy": false, "available": true },
        "policyMomentum": { "z": 1.10, "display": 74, "proxy": true,  "available": true },
        "macroVsMandate": { "z": 0.40, "display": 55, "proxy": false, "available": true },
        "termsOfTrade":   { "z": -0.10, "display": 49, "proxy": false, "available": true },
        "trend":          { "z": 0.55, "display": 57, "proxy": false, "available": true },
        "valuation":      { "z": -0.60, "display": 43, "proxy": false, "available": true }
      },
      "flags": {
        "carryTrendConflict": false,
        "interventionRisk": false,
        "dataIncomplete": false,
        "proxyInputs": ["policyMomentum"]
      },
      "policyRate": 3.625,
      "policyRateLabel": "3.50-3.75%",
      "cpiYoY": 3.7,
      "cbTarget": 2.0,
      "yield2y": 4.24,
      "read": "Carry and policy momentum both supportive; valuation is the drag."
    }
  ],
  "pairs": [
    { "base": "USD", "quote": "JPY", "z": 1.42, "display": 68, "flags": { "interventionRisk": true } }
  ],
  "reconciliation": { "...": "see 4.5" },
  "meta": {
    "componentWeights": { "carry": 0.25, "policyMomentum": 0.25, "macroVsMandate": 0.20,
                          "termsOfTrade": 0.10, "trend": 0.10, "valuation": 0.10 },
    "universe": ["USD","EUR","JPY","GBP","CHF","CAD","AUD","NZD","SEK","NOK"],
    "displayOnly": ["CNY"]
  }
}
```

`pairs` contains one triangle (45 entries) — the UI mirrors for display.

### `GET /api/fx/currency/[code]`

Everything in the snapshot entry, plus history for the detail page:

```json
{
  "code": "EUR",
  "...": "all snapshot currency fields",
  "history": {
    "score":  [{ "date": "2026-08-01", "value": 54 }],
    "spotUsd":[{ "date": "2026-08-01", "value": 1.0812 }],
    "yield2y":[{ "date": "2026-08-01", "value": 2.31 }]
  }
}
```

Return **at least 2 years** of history where available. Charts with one or two
points render as an empty plot.

### `GET /api/fx/pairs?base=USD`

Filtered pair list for a single base currency.

---

## 7. Validation before shipping

Write these as tests. They catch the failure modes that make a scoring model
embarrassing rather than merely wrong.

1. **Antisymmetry:** `pair(A,B).z === -pair(B,A).z` for all pairs.
2. **Self-pair is zero:** `pair(A,A).z === 0`.
3. **Rank consistency:** the highest-scored currency must win every pair it
   appears in as base.
4. **Weights sum to 1.0**, including after reweighting for dropped components.
5. **No NaN reaches the payload.** Any NaN must become `available: false`.
6. **Sanity anchor:** at any date, the currency with the highest real policy
   rate and positive momentum should not rank in the bottom three. If it does,
   the sign convention is inverted somewhere.
7. **Series validation script passes** with zero 404s.

---

## 8. Build order

1. Frankfurter client + the **breadth module (section 4A)**. It depends on
   nothing but spot data, has an exact identity to test against, and is a
   shippable page on its own.
2. Series map + validation script. Do not proceed past this until every ID
   resolves.
3. FRED client wrapper with caching.
4. Component calculators, each independently unit-tested.
5. Cross-sectional z-scoring, winsorisation, composite.
6. Pair derivation + flags.
7. Snapshot writer + cron, covering both models.
8. API routes.
9. Reconciliation against the existing dollar signal.

Ship steps 1–8 before touching the UI. The frontend spec assumes
`/api/fx/snapshot` returns real data.

Note the ordering: the breadth module comes first. It is the cheaper and more
robust half of this project, it needs no FRED series at all, and it answers the
question subscribers actually ask about a pair — so it should not sit blocked
behind the composite model's data dependencies.
