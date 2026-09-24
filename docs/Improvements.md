# Macro Engine improvement research

**22 September 2026 · Product, interface, data, login, content and commercial strategy**

## Recommendation

Macro Engine has the substance of a useful research product: macro interpretation, evidence, asset comparisons, company workbooks and published portfolios. Its strongest opportunity is to help an investor answer **what changed, why it matters, and what would change the view next**.

The immediate constraint is credibility and coherence. Several analytical outputs are materially unreliable, different pages describe different models as though they were one, and the distinction between newsletter subscriber, website account and paid member is unclear. Fix these before increasing distribution or selling analytical precision. A visual redesign alone would leave the most consequential problems intact.

Keep Substack as the audience and initial billing channel. Make the website the place readers inspect evidence and follow a thesis between issues. Build one reliable weekly research loop before expanding the feature catalogue.

## What was reviewed

Three independent reviews covered frontend source, backend source, and audience/market strategy. The lead review inspected the live interface, all 25 page templates, the ten G10 currency details, 17 asset-signal details, and 99 linked stock workbooks—148 distinct page URLs in total, including the anonymous private-route boundary. See the separate coverage record for specific URLs and load limitations. The internal options tool returned an intentional-looking 404 to an anonymous visitor; its authorization was reviewed in source.

Frontend: `Desktop/macro-engine-web`, a Next.js application. Backend: `Desktop/macro_engine`, a FastAPI/Python project, with scheduled jobs, Postgres publication tables, and an older Streamlit application. The backend directory uses an underscore. Data sources include FRED, Yahoo Finance, FX fixings and a separate private options pipeline.

The audit was read-only. No product files, accounts, prices, subscriptions, publications or production data were changed. No account was supplied, so successful OAuth completion, registered/paid sessions and private-tool functionality were **not** verified live. Those paths were inspected in source. No production exploit was attempted. Desktop screenshots and accessibility trees were reviewed; narrow-width behavior was partially observed, but a controlled 390px test did not take effect reliably, so mobile recommendations need device acceptance testing.

Evidence labels below: **Live** = observed on the website; **Source** = confirmed in local code, which may differ from deployment; **Recommendation** = proposed change, not evidence of customer demand. Market claims inside the product were reviewed for sourcing and consistency, not comprehensively fact-checked.

## The first five priorities

| Priority | Finding | Why it matters | Next action |
|---|---|---|---|
| P0 | CPI YoY uses 12 daily-aligned rows instead of 12 months | Corrupts an input to charts and macro interpretation | Correct native-frequency calculation; rebuild affected outputs |
| P0 | Foreign-company financial values lack currency/share-basis normalization | Produces implausible yields and scenario prices | Suppress incompatible calculations; normalize currencies and ADR basis |
| P0 | Stock portfolio history uses hindsight inputs and same-day timing | Displayed performance cannot be treated as executable historical results | Label simulations; repair timing; establish a forward ledger |
| P1 | Models, timestamps and portfolio exposures do not reconcile clearly | Users cannot tell which view to trust | One data contract; explicit model names and actual-versus-target exposure |
| P1 | Free, registered and paid promises conflict | Weakens conversion and risks disappointing paying readers | One entitlement matrix shared by Substack, website and account |

P0 means correct or quarantine before relying on the output in marketing. P1 means address in the next product cycle. These are research priorities, not claims that every affected page is unusable.

## Data and analytical trust

### 1. Correct CPI at its native frequency

**Live:** Inflation showed headline CPI YoY **0.00%**. Real fed funds and nominal fed funds both showed **3.63%**.

**Source:** The common macro frame is forward-filled, then several functions use `pct_change(12)` on CPI. Twelve rows in that frame are not twelve monthly releases. The defect appears in charts, derived inputs, the regime model and macro thesis calculations. Monthly resampling in the thesis occurs after the erroneous transformation.

**Fix:** Calculate the annual change on monthly CPI first, then align the derived series to daily displays. Preserve release/availability dates separately from observation periods. Recompute affected scores, snapshots and history; publish a dated methodology correction rather than silently replacing history.

**Acceptance:** A mixed-frequency fixture retains the correct annual change between releases; multiple named CPI releases reconcile to the source; real policy rates equal the intended nominal rate minus the correct inflation measure. References: backend `api/deps.py:55`, `api/routers/charts.py:84`, `src/derived.py:77`, `src/regime.py:335`, `src/macro_thesis/engine.py:126`.

### 2. Validate currency, units and ADR basis before valuation

**Live:** [TSM's workbook](https://www.macro-engine.com/stocks/TSM) displayed TTM revenue **$4.44T**, FCF yield **48.3%**, earnings yield **95.3%**, and a base scenario of **$48,402.55 / +10,691.9%**, alongside a share price near **$448.51**.

**Source:** Native financial statements are combined with listing market capitalization, shares and price without currency normalization. The frontend labels the financial model USD. This is strongly consistent with foreign reporting-currency values being treated as quote-currency values; the exact vendor currency and ADR conversion factor were not independently verified.

**Fix:** Carry reporting currency, quote currency, financial period, scale, FX source/date and ordinary-share/ADR basis through the API. Suppress derived values when these are incompatible. Changing the dollar label alone will not fix the arithmetic. Check other foreign issuers, not just TSM.

Five-year scenario outputs also need to say **terminal value in year five**, with cumulative and annualized return, or be discounted before being described as current fair value. Show assumptions as editable research inputs later; first establish their correctness.

**Acceptance:** Reconcile a US issuer and at least two foreign issuers against their financial reports. No valuation should divide values in different currencies or inconsistent share units. References: backend `api/routers/stock_intelligence.py:994`, `:1825`, `:1878`, `:1948`, `:2012`; frontend `src/app/stocks/[ticker]/page.js:588`.

### 3. Separate simulation from published performance

**Live:** Stock Alpha showed **+95.8%** over the last 12 months; SMID showed **+133.7%**, both against SPY **+16.7%**. Volatility and drawdown are visible, which is useful, but the curves need much clearer provenance.

**Source:** Historical selection reuses current fundamental/ranking inputs. New weights incorporate a day's close and are applied to that same day's close-to-close return. These are hindsight and timing biases. Their total effect was not quantified in this audit.

**Fix:** Label existing curves as retrospective simulations with material limitations. Do not use them as the primary purchase argument. Persist the universe, inputs, model version, signal timestamp and executable next-period weights. Begin an immutable forward paper portfolio, report costs and benchmark methodology, and distinguish model history from a real account.

**Acceptance:** A signal cannot earn a return from before it was knowable. Each return can be reproduced from a previously recorded holding and price series. References: backend `api/routers/stock_portfolio.py:718`, `:958`, `:1003`; `api/routers/smid_growth_portfolio.py:959`, `:1020`.

### 4. Make every important number explainable

The overview said **48/100 Neutral**, while The View said **Goldilocks, low conviction** and its house view was unpublished. These can be different valid models, but the interface does not explain that distinction well enough. The four pillar screens also compute local scores that are not the components of the backend regime score. Their displayed weights suggest a relationship that needs reconciliation.

Stock Alpha showed **95% stock exposure and 42% cash** while holdings and sector exposures summed to about **58%**. If 95% is a risk budget and 58% is invested exposure, name both. Otherwise fix the inconsistent snapshot. Require actual invested weights plus cash to reconcile to 100%, with explicit treatment of any leverage.

Continuing claims differed between the macro thesis and growth page. Source review suggests the thesis uses the latest completed month and growth uses the latest weekly observation. Display these dates; do not force unlike frequencies to agree artificially.

Replace generic “Live market data” with the actual observation/publication time and cadence. The regime endpoint currently stamps today's date even when it may be serving an older disk cache. Prefer “Last published Sep 21 · daily update · current” or “Update delayed · showing Sep 19.” Keep last-good data, but disclose its age.

Additional concrete inconsistencies:

- The report landing page labels a hardcoded **52/100** preview “Current regime.” Use a dated real example or mark it illustrative.
- Signals count scores ≥52 as tailwinds, but stance labels require ≥55; XLC at 53 appeared Neutral inside the tailwind population.
- FX summaries are calculated from the five preview currencies. The overview said no conflict and quiet intervention risk, while JPY detail flagged intervention risk and NZD detail flagged a carry/trend conflict. Missing currencies cannot support a whole-universe all-clear.
- Asset details expose `hy_oas`, `real10` and similar internal keys instead of plain-language evidence. Show the current value, change window and reason it supports/opposes this asset.
- BTC's signal price chart was unavailable during review; the other signal pages rendered. Its ticker-to-price-series mapping needs investigation.

Use a common envelope: `observationDate`, `publishedAt`, `source`, `frequency`, `units`, `methodologyVersion`, `missingInputs`, `freshnessStatus`. Label input coverage as coverage, not forecast confidence.

## Interface, flow and presentation

The dark macro pages have a recognizable visual identity, good section headings, clear numerical hierarchy, and useful “what confirms / what breaks” structures. Preserve these. The workbook's light purple treatment, white signal cards and dark dashboards nevertheless feel like separate products. Standardize typography, spacing, chart controls, tables, status badges and loading states before reworking decorative styling.

### A clearer first five minutes

**Proposed journey:** Substack issue → dated weekly brief → inspect one supporting chart → open a related asset/company → save or follow the thesis → return for the next update.

The homepage currently presents many destinations before demonstrating one useful decision process. “Macro Engine Intelligence Platform” identifies the product but gives little reason to choose it. Test a headline such as **“Understand this week's macro changes—and the evidence behind them.”** Put three dated changes and one complete sample below it. Use one primary action, “Read this week's brief,” and a secondary “Explore the evidence.”

Give each research page a small shared header: **Conclusion · What changed · Evidence · What would change the view · Updated**. Put technical depth below it. At the end, offer one relevant next step rather than another general tour of the platform.

Recommended navigation: **This week, Macro evidence, Research, Portfolios**, plus an Advanced tools area for FX/scanners. Test the language with readers before replacing the existing taxonomy. Keep the homepage for new visitors and a concise overview for returning users.

### Specific UI improvements

- Reduce oversized introductory sections that push the first evidence below the fold. The initial narrow homepage view was dominated by the product title.
- Add a section navigator to long stock workbooks and briefing pages. Prioritize diagnosis, changes and caveats; collapse advanced tables.
- Offer essential mobile columns with expandable detail. Some source tables have minimum widths above 1,000px; SMID's is about 1,740px.
- Increase secondary-text legibility on dark cards and test actual contrast. This was a visual concern, not a measured WCAG failure.
- Make dropdown state explicit with `aria-expanded`, keyboard dismissal and reliable touch behavior. Source includes keyboard-focus opening; it is **not** purely hover-only. Hidden menus and clipped access-gated content need focus/accessibility-tree testing.
- Normalize `%`, percentage-point and basis-point changes. Put the time window beside each delta and the unit beside each series.
- Replace “Postgres snapshot,” database IDs and instructions to run Python jobs with useful customer-facing status. Preserve technical diagnostics in internal tools.
- Load independent panels independently. The homepage waits for a group of requests; one slow request can delay unrelated content. Distinguish loading, no data, access required and failed refresh.
- Fix homepage editorial deep links: `#intelligence-item-{rank}` does not match the actual `#brief-{id}` targets.
- Give public pages specific titles, descriptions and share images. Much initial content waits for client mounting; server-render meaningful public summaries and add a sitemap/canonical strategy. This is a discoverability weakness, not proof search engines cannot index the site.

## Login, membership and access

The live login is Google-only, with no password to manage. That is simple for compatible users. However, a Substack reader may use another address or a different Google identity. The promise that the same email automatically links the subscription needs a qualification: source shows a CSV-based newsletter import, not a paid-subscription lifecycle.

The account contract exposes registered/admin/newsletter status, not a working paid entitlement. A schema tier field exists, but it is not enough to grant, expire or revoke paid access. Current registration gates should not be marketed as a paid membership system.

Recommended account experience:

1. Separate **website account**, **newsletter subscription** and **paid membership** status.
2. Preserve the intended destination through login and return users there.
3. Explain email matching and provide a mismatch/recovery path. Consider an email sign-in alternative; the backend already has a magic-link path, but it needs delivery hardening before exposure.
4. Show the last membership sync and a way to resolve incorrect status. Initially, five paid members can be managed through a small auditable process with review/expiry dates while Substack handles billing.
5. Add appropriate privacy, terms, contact and methodology pages. This is a trust/onboarding recommendation, not a jurisdiction-specific legal assessment.

The visible blur/clipping gates are conversion prompts, not secure content protection: full static editorial content is shipped to the browser. Some registered endpoints do restrict data, but access rules differ across modules. Decide public/free/paid capabilities first, then enforce the intended boundaries on the server and keep hidden content out of keyboard/screen-reader traversal.

Security follow-up from source: bind OAuth state to the initiating browser and consume it once; consume magic links atomically; record email-delivery failures without logging bearer links. The current OAuth state is signed and expiring but lacks browser binding. No successful exploit was tested. Existing HttpOnly/Secure cookies, hashed tokens and fail-closed private options access are useful foundations.

Both `/admin` content studios are publicly accessible by design and currently appear to be client-side drafting/copy tools. This is not evidence of public production-publishing access. If owner-only, protect them; if a customer feature, rename and position them as tools. “Generate content” currently competes with the investor's main action on a stock workbook.

## Content and publishing

The themes/geopolitics format is a strength: it connects a development to beneficiaries, catalysts and invalidation. Improve accountability rather than adding more topics.

Every brief should identify its author/reviewer, as-of date, primary sources, what changed since the last version, uncertainty and next review trigger. Separate factual observations from model inference and editorial judgment. Detailed claims currently appear without linked evidence. The headings promise ten items while the actual lists contain 11 themes and 12 events; generate counts from the content.

Use one repeatable weekly issue:

1. One-sentence conclusion and three changes since last week.
2. One annotated chart with source and date.
3. The strongest evidence for and against the view.
4. One focused thesis or company comparison.
5. The next event/threshold that would change the view.
6. One deep link into the matching website evidence.

Keep an archive of calls and subsequent updates, including mistakes and invalidations. This is a more defensible reason to pay than another dashboard score. Establish a correction log for methodology changes, especially the CPI and valuation issues.

Substack's [About page](https://macroengine.substack.com/about) is generic and publication metadata still uses “Macro Dashboard.” Align it with Macro Engine, state the intended reader and explain the research method. The [public RSS feed](https://macroengine.substack.com/feed) showed an approximately 18-day gap between August 23 and September 11; that does not prove private emails were absent, but public evidence did not establish the website's weekly promise.

The [July portfolio introduction](https://macroengine.substack.com/p/macro-engine-stock-portfolio-free) leads with a 168.7% return without clear live-versus-simulation context in the reviewed text. The [Altis article](https://macroengine.substack.com/p/macro-engines-altisfinance-portfolio) introduces another portfolio and says cash allocation is omitted. Clearly distinguish personal/partner portfolios from systematic models, with complete weights and applicable methodology.

## Monetization strategy

### What the current numbers say

You reported **220 subscribers, five paid and $700/year**. If paid readers are included in 220, paid share is **2.27%**. Revenue divided by five is **$140 per payer/year**, but that is not necessarily list price, ARR or net profit. Billing mix, founding subscriptions, churn, fees and the reporting period remain unknown.

The public [Substack subscribe configuration](https://macroengine.substack.com/subscribe) listed **$20/month, $90/year and $240/year founding** during research. Checkout was not purchased or completed. Annual is 62.5% below twelve monthly payments. Its free benefit says occasional public posts, while the website promises a free weekly report; paid benefits mention dashboard access while the website promotes free accounts. Resolve the offer before changing price.

### One membership, clearly defined

| Level | Proposed promise | Purpose |
|---|---|---|
| Public/free newsletter | Dependable weekly brief, headline regime, one complete sample journey, methodology | Demonstrate quality and create a habit |
| Free website account | Consistent access to the explicitly free research tools; saved preferences if implemented | Make repeat use convenient |
| Paid research membership | Clearly scheduled deeper research, thesis-update archive, maintained comparisons and model-change explanations | Sell continuity and interpretation |

Saved monitoring and alerts could strengthen paid value, but do not promise them before they work. Preserve current member terms while testing. Keep one checkout destination on Substack initially; a billing migration is unnecessary at five paying readers.

Interview all five paid readers and five engaged free readers before narrowing the offer: what did they actually use, when did they last return, what would they miss, and why would they cancel? No outreach was sent as part of this audit.

Test the value proposition first. A possible later test for new members is **$15/month or $120/year**, with existing subscribers grandfathered. That is a hypothesis, not an optimal price. Retaining $90/year while improving the offer is also reasonable. Use sequential tests and raw counts; this audience is too small for elaborate multi-arm pricing experiments.

### Competitive context

[Koyfin](https://www.koyfin.com/pricing/) offers free macro/market dashboards and advertises Plus at $39/month with annual billing. Macro Engine cannot differentiate simply by having charts. [Lyn Alden](https://www.lynalden.com/premium/) advertises $29/month or $249/year and a specific twice-monthly research deliverable. The useful lesson is clarity and continuity, not that your current audience will pay the same price.

Avoid advertisements, multiple tool tiers and enterprise sales for now. Focus on one audience and a repeatable editorial/research workflow. Track hosting, data and support costs; $700 annual revenue leaves little room for expensive infrastructure without a deliberate investment budget.

### Illustrative economics—not forecasts

| Scenario | Paying members | Annual revenue/member | Gross annual revenue |
|---|---:|---:|---:|
| Current reported | 5 | $140 implied | $700 reported |
| 5% of current 220 readers, $90 annual plan | 11 | $90 | $990 |
| 5% of current audience, hypothetical $120 plan | 11 | $120 | $1,320 |
| 1,000 readers at hypothetical 5% paid share | 50 | $120 | $6,000 |
| 2,000 readers at hypothetical 5% paid share | 100 | $120 | $12,000 |

These assume a full year paid and exclude churn, discounts, fees, tax, data, hosting and labor. At $120, $10,000 gross requires 84 full-year-equivalent members; at an assumed 5% paid share, about 1,680 readers. Conversion work alone on today's list has a limited ceiling. Reliable publishing, retained readers and audience acquisition all matter.

## A practical 90-day sequence

| Window | Work | Evidence required before moving on |
|---|---|---|
| Days 1–14 | Quarantine/fix CPI and currency defects; label simulations; reconcile portfolio exposure; inventory every freshness field | Source-reconciled calculations and truthful labels; affected snapshots rebuilt or marked unavailable |
| Days 15–30 | Clarify free/paid offer, account status, login errors; repair links; implement one dated weekly-brief journey | Successful anonymous and authorized-account journeys; consistent offer across site/Substack |
| Days 31–60 | Deliver four consecutive issues with matching web updates; add attribution and first-use tracking; improve highest-used UI | Confirmed signup counts and repeat weekly use, not merely pageviews |
| Days 61–90 | Test one paid offer with engaged readers; improve retention; decide which advanced tools deserve investment | Paid conversions, actual use, cancellation reasons and contribution after costs |

Instrument newsletter-source → report view → evidence view → login start/success → first useful action → return in another week → paid checkout/confirmed payment. Preserve source tags; current newsletter CTA source parameters are not consumed. External hosting analytics might exist, but no conversion instrumentation was found in source. An iframe click is not a confirmed subscription.

Suggested early experiment targets, explicitly not industry benchmarks: interview ten readers; publish four scheduled issues; identify 20 readers returning in two distinct weeks; seek five incremental paying members after validating the offer. Report counts and objections alongside rates.

**Recommended investment decision:** spend the next cycle making a smaller research journey demonstrably correct, coherent and repeatable. Retain the broad analytical work underneath it, but let evidence of reader use determine which parts become the paid product.

## Supporting material

- `Macro-Engine-Page-Coverage.md`: route checklist and detail-page review status.
- `Macro-Engine-Technical-Evidence.md`: frontend/backend source findings and commercial research notes, with file/line references and source links.

The recommendations are an audit backlog. They have not been implemented, and the product should be retested after each material correction.

