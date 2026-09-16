# Macro Engine — Macro Thesis Engine
## Backend specification

Build the engine behind a new page at `/dashboard/macro` — a synthesis layer
that turns the five pillar pages (growth, inflation, rates, credit, volatility)
plus FX into a single house view.

This is **not** another dashboard. It is an opinion with evidence, scenarios,
and pre-committed invalidation.

---

## 1. The frameworks, and why these three

Three layers, each answering a different question a macro PM asks daily.

**Layer 1 — Growth/Inflation quadrant. "What is being repriced?"**
Assets are priced off growth and inflation surprising relative to what's
discounted. Four regimes, each with reliable relative winners. The critical
detail: the quadrant is determined by **rate of change, not level**. An economy
with 3% inflation decelerating behaves like disinflation; the same 3%
accelerating behaves like an inflation scare. Level sets context, the second
derivative sets the trade.

**Layer 2 — Debt cycle position. "Where are we in the bigger arc?"**
Dalio's template: a short-term debt cycle of roughly 5–8 years, driven by credit
availability and the central bank's reaction function, oscillating around a
long-term debt cycle measured in decades and driven by debt service capacity and
whether real rates exceed real growth. The short-term cycle tells you what the
next two quarters look like. The long-term cycle tells you which policy
responses are still available.

**Layer 3 — Cross-asset confirmation. "Does the tape agree?"**
The model's read is a hypothesis. Whether credit spreads, the curve, the dollar,
and equity breadth confirm it is the test. Divergence is the highest-value
output on the page — it is either an early signal or a wrong model, and naming
it is what an honest process does.

---

## 2. Data — nearly all of it already exists

From existing pillar computations: claims, continuing claims, breadth, NFCI,
curve (3m10y and 2s10s), copper/gold, breakevens 5y/10y and 5y5y, CPI, real fed
funds, real yields, policy rate, net liquidity, dollar, IG and HY OAS, VIX term
structure, FX strengths.

**New series required** (FRED unless noted):

| Purpose | Series |
|---|---|
| Debt burden | Total credit to private non-financial sector, % GDP (BIS) |
| Debt service | Debt service ratio, private non-financial (BIS) |
| Fiscal | Federal debt held by public % GDP; deficit % GDP |
| r vs g | 10y nominal yield minus nominal GDP growth |
| CB balance sheet | Fed total assets % GDP |
| Term premium | ACM 10y term premium |
| Output gap proxy | Unemployment rate minus CBO NAIRU, or U-3 vs its 36m low |

BIS bulk CSVs are already integrated for REER and policy rates — reuse that
client. Run `scripts/validate_fx_series.py` (or its equivalent) over every new
FRED ID before wiring it; assume nothing resolves until proven.

**History requirement: 25+ years.** The transition probabilities and
quadrant-conditioned returns in §5 and §6 are the differentiating output, and
they need enough cycles to mean anything. Monthly frequency is sufficient.

---

## 3. The quadrant engine

### 3.1 Two axes

Each axis combines a **level** z-score and a **momentum** z-score. Momentum is
the 3-month change in the level composite, itself z-scored across history.

```
GROWTH axis inputs (equal weight after z-scoring, sign-aligned so + = stronger):
  initial claims (inverted), continuing claims (inverted), breadth (RSP/SPY),
  NFCI (inverted), copper/gold, 3m10y curve, consumer sentiment

INFLATION axis inputs (+ = hotter):
  10y breakeven, 5y breakeven, 5y5y forward, CPI YoY, commodity basket,
  dollar (inverted), real fed funds (inverted)
```

Winsorise at ±2.5σ before combining. Require at least 60% of inputs present or
the axis is marked unavailable.

### 3.2 Quadrant assignment

Quadrant is assigned on **momentum sign**, with level as context:

```
growthMomentum >= 0 and inflationMomentum <  0  ->  GOLDILOCKS
growthMomentum >= 0 and inflationMomentum >= 0  ->  REFLATION
growthMomentum <  0 and inflationMomentum >= 0  ->  STAGFLATION
growthMomentum <  0 and inflationMomentum <  0  ->  DEFLATION
```

Also emit `quadrantStrength` = Euclidean distance from origin in
(growthMomentum, inflationMomentum) space. A reading near the origin is a weak
assignment and the UI must say so rather than asserting a regime.

Emit a `transitioning` flag when either momentum has crossed zero within the
last 6 weeks — regime changes are where the money is made and lost, and a
freshly-crossed quadrant deserves different language than an entrenched one.

### 3.3 Historical series

Compute the quadrant monthly back to the earliest common start date. Store as
`macro_quadrant_history(as_of, growth_level, growth_momentum, inflation_level,
inflation_momentum, quadrant, strength)`.

**Revision caveat, and it must be disclosed:** this uses current-vintage data,
so historical quadrants benefit from revisions unavailable in real time. The
methodology text must say so plainly. If you want to remove the bias later, pull
vintages from ALFRED — but ship with the disclosure first rather than delaying.

---

## 4. Debt cycle position

### 4.1 Short-term cycle phase

Six phases. Classify with a rules table, not a black box, so the page can show
its work:

| Phase | Conditions |
|---|---|
| `EARLY_EXPANSION` | growth momentum +, policy easy (real policy rate < 0), spreads narrowing from wides, curve steep |
| `MID_EXPANSION` | growth level and momentum +, inflation contained, spreads tight and stable |
| `LATE_EXPANSION` | growth level + but momentum flattening, inflation momentum +, policy tightening, curve flattening |
| `TIGHTENING` | real policy rate > 0 and rising, curve flat or inverted, spreads beginning to widen |
| `CONTRACTION` | growth momentum negative and accelerating down, spreads widening, claims rising |
| `REFLATION` | growth still weak but policy easing, spreads narrowing, curve re-steepening |

Emit the phase, the conditions met and unmet, and `phaseConfidence` =
met / total. Showing "4 of 6 conditions met" is more useful and more honest than
a bare label.

Also emit `monthsInPhase` and the historical median duration of that phase, so
the page can say "month 14, historical median 11."

### 4.2 Long-term debt cycle gauge

A 0–100 index of how constrained policy is by accumulated debt. Higher = later
in the long cycle = fewer conventional tools available.

```
Components (each z-scored across the longest available history, then averaged,
then mapped to 0-100):

  debt burden        total credit to private NFC % GDP + federal debt % GDP
  debt service       debt service ratio
  r minus g          10y nominal yield - nominal GDP growth   (+ = constraining)
  policy room        policy rate distance above zero          (inverted)
  monetisation       Fed assets % GDP
  term premium       ACM 10y
```

This moves glacially. **Do not recompute the narrative daily** — compute
quarterly, display the level and its 5-year trend, and treat it as context that
frames the short cycle rather than a signal.

Emit alongside it the two or three components currently most extended, so the
page can say *what* is late rather than only that something is.

### 4.3 The map coordinates

For the visual in the frontend spec, emit normalised positions:

```json
"cycleMap": {
  "longTerm": { "position": 0.78, "label": "Late", "trend": "rising" },
  "shortTerm": { "phase": "LATE_EXPANSION", "position": 0.62,
                 "monthsInPhase": 14, "medianPhaseMonths": 11 },
  "productivityTrend": { "realGdpTrend10y": 2.1 }
}
```

`position` is 0–1 along the respective cycle for marker placement.

---

## 5. Transition probabilities

The output that makes this page defensible rather than opinionated.

From `macro_quadrant_history`, compute an empirical transition matrix:

```
P(quadrant at t+3m | quadrant at t)          -- 3 month horizon
P(quadrant at t+6m | quadrant at t)          -- 6 month horizon
```

Condition on the current state where sample size allows: same quadrant **and**
same short-cycle phase. Report `n` alongside every probability, and **suppress
any cell with n < 12** rather than showing a percentage derived from four
observations.

Output shape:

```json
"transitions": {
  "horizonMonths": 3,
  "from": "STAGFLATION",
  "conditionedOn": ["quadrant", "phase"],
  "sampleSize": 47,
  "probabilities": { "STAGFLATION": 0.61, "DEFLATION": 0.21,
                     "REFLATION": 0.12, "GOLDILOCKS": 0.06 }
}
```

These are historical frequencies, not forecasts. The UI must label them that way.

---

## 6. Asset behaviour by quadrant

Backtest simple total returns by quadrant across the same history. Universe:
SPY, QQQ, IWM, XLE, XLF, XLU, XLP, TLT, IEF, HYG, LQD, GLD, DBC, UUP, EEM.

For each quadrant emit: mean 3-month forward return, hit rate (% positive), and
`n`. Rank within quadrant.

Requirements:

- Forward returns only — the quadrant at time `t` maps to returns from `t` to
  `t+3m`. Any other alignment is lookahead.
- Suppress series with fewer than 24 observations in a quadrant.
- Store as a static table refreshed monthly. This is history, not a live signal.

This produces the page's most actionable content: "in this quadrant, these five
things have historically worked, with this hit rate, over this many episodes."

---

## 7. Cross-asset confirmation

For each of five markets, compare what the model implies against what price is
doing over the last 20 sessions:

| Market | Model implication from quadrant | Actual |
|---|---|---|
| Equities (SPY) | risk-on in Goldilocks/Reflation | 20d return |
| Credit (HY OAS) | spreads tighten in risk-on quadrants | 20d change |
| Duration (10y yield) | falls in Deflation, rises in Reflation | 20d change |
| Dollar | rises in Deflation/Stagflation | 20d change |
| Commodities | rise in Reflation/Stagflation | 20d change |

Emit `confirms | diverges | neutral` per market plus a
`confirmationScore` (share confirming). When three or more diverge, set
`modelUnderReview: true` — a signal that the framework and the tape disagree
enough that the page should lead with the conflict rather than the thesis.

---

## 8. Thesis generation

**Deterministic templates, not a runtime LLM call.** The page must produce the
same words from the same state, be auditable after the fact, and never
hallucinate a figure. Compose from slots:

```
BASE CASE
  {quadrant_sentence} {phase_sentence} {confirmation_sentence}
  What would change it: {top_two_triggers}

ALTERNATE 1  (second most likely transition)
  Trigger: {threshold crossing that would produce it}
  Expression: {top ranked assets in that quadrant}

ALTERNATE 2  (third most likely)
  ...

INVALIDATION
  {explicit, pre-committed: named level, named series, named date}
```

Triggers must be **specific and checkable** — "10y breakeven above 2.60",
"initial claims 4-week average above 260k", "HY OAS above 400bp" — never
"if inflation worsens". Derive them from the actual thresholds that would flip a
component's sign.

### House view override

Add an admin-editable layer. The model produces a mechanical read; you may
disagree. Schema:

```sql
house_view (
  as_of date primary key,
  agrees_with_model boolean not null,
  headline text,
  body text,
  conviction text,          -- 'high' | 'medium' | 'low'
  author_note text,
  created_at timestamptz
)
```

The page shows both, clearly labelled: **Model read** and **House view**. When
they disagree, that is displayed rather than resolved. A macro process where the
human and the model are visibly separate is more trustworthy than one where a
person quietly edits the model's output — and over time the disagreement log is
itself evidence about which to trust.

---

## 9. API

```
GET /api/macro/thesis          full payload: quadrant, cycle, transitions,
                               assets, confirmation, thesis, houseView
GET /api/macro/quadrant/history?months=300
GET /api/macro/assets-by-quadrant?quadrant=STAGFLATION
POST /api/admin/house-view     admin only
```

Compute in the nightly job, publish a snapshot, serve from storage. Never
compute on request. Quadrant history and asset tables refresh monthly; the
long-term gauge quarterly.

---

## 10. Validation

1. Quadrant assignment matches known historical episodes: 2008 H2 → Deflation,
   2021 H1 → Reflation, 2022 H1 → Stagflation, 2017 → Goldilocks. If these
   don't come out right, the axes are miscalibrated — fix before shipping.
2. Transition matrix rows sum to 1.0 within tolerance.
3. No asset-return cell computed from fewer than 24 observations is exposed.
4. Forward returns contain no lookahead: assert the return window starts strictly
   after the quadrant observation date.
5. Every emitted trigger references a series that exists and a threshold within
   its historical range.

## 11. Build order

1. Growth and inflation axes + current quadrant
2. Historical quadrant series (this unlocks 5 and 6)
3. Short-cycle phase rules table
4. Transition probabilities
5. Asset returns by quadrant
6. Cross-asset confirmation
7. Thesis templates
8. Long-term gauge and cycle map coordinates
9. House view admin

Steps 1–2 alone justify the page. Ship them before building the rest.
