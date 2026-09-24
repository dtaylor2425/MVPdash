# Macro Engine — portfolio implementation and days 1–14 progress

Updated September 23, 2026. Changes are implemented locally in `Desktop/macro-engine-web` and `Desktop/macro_engine`. They have not been deployed, and no production portfolio, trade, return, account, or database record was changed.

## Portfolio changes

Both portfolio pages now show recorded entry price/date, latest valuation price, price gain since entry, current weight, target weight, volume signal, and relative volume. Rebalance history starts collapsed and sorts newest first. Curves are described as quantitative model history, with a clear retrospective-input disclosure for older history. The portfolio badge says “Published valuations,” not “Live market data.”

Entry references use the first published model record in the current uninterrupted holding period. Adds and trims preserve the entry. A removal followed by a new entry starts a new holding period. The original entry price remains available after a split; the comparison basis adjusts for subsequent splits. Missing prices stay unavailable. These are model references, not personal trade fills or money-weighted position returns.

The accompanying entry register documents **9 Stock Alpha positions and 15 SMID positions**, using **26 public published snapshots per strategy**, through the September 22 model publication. It includes publication date/time, model date, recorded price, weight, and source run. Public history excludes some superseded same-day records; the database implementation also checks those records when resolving entries.

## History and accounting

- Same-day reruns return the existing published run before rebuilding the model. Backdated new publications are rejected.
- A per-strategy database lock serializes the publishing job. Published parent records and position, performance, and rebalance rows are immutable after the SQL migration.
- A database publication trigger rejects older publisher versions, rewritten historical curve prefixes, rewritten trade prefixes, and non-advancing publication dates.
- The new publisher does not recalculate the legacy performance curve. It preserves the latest published curve and appends prospective valuations from stored quantities and cash.
- Migration establishes quantities at the latest completed-session close. Any gap after legacy history is explicitly unmeasured and contributes no fabricated return. This is an accounting opening, not a claim that old trades were executed at that close.
- New allocation decisions are published first and use the next eligible session's opening model reference after the publication day. They cannot earn returns from the bar that produced the decision. The next daily publication reports their recorded execution.
- Split events and dividends affect quantities/cash prospectively. Provider split-adjusted OHLC is normalized to historical-share prices before events are booked. A changed previously recorded provider price stops publication for reconciliation.
- Price changes move weights and entry-price gains at completed-session valuations. They are not intraday streaming quotes. Market drift is not logged as a trade.

Historical methodology limitations remain visible: retained older curves used retrospective inputs, and returns exclude transaction costs and taxes. Retaining published numbers does not validate the original historical methodology.

## Macro allocation and volume

The **long target is 65–95%**, leaving **5–35% target cash**. It is 65% at macro scores of 25 or below, rises linearly to 95% at scores of 75 or above, and is 80% at a score of 50. Decisions update weekly, after a macro target movement of at least five percentage points, or when actual long exposure is below 65% and no decision is already pending.

Individual-stock, sector, and theme limits remain in place. A clearly named **SPY broad-market sleeve fills capacity left by those limits** so concentration caps do not force the target below 65%. This introduces broad-market exposure into both strategies, including SMID, rather than relaxing stock caps.

The 65% floor is a rebalance target, not an intraday guarantee: prices can move actual exposure below it between executions. A legacy portfolio below 65% keeps its truthful existing allocation until the published transition executes; it is not silently rewritten to appear compliant.

Relative volume compares the latest completed session with the preceding 20 sessions. Strong volume is at least 1.5× normal and weak volume is below 0.8×. Up/down price direction separates buying/selling-pressure proxies; this does not claim observed order flow. Strong buying, strong selling, weak buying, and weak selling apply allocation multipliers of 1.15, 0.65, 0.85, and 0.95 respectively before normalization and caps. Missing volume remains unavailable.

## Days 1–14 started

| Area | Implemented locally | Remaining release work |
|---|---|---|
| CPI | Calendar-month year-over-year helper; sparse native observations; no backward fill; charts, real rates, regime, thesis, and legacy pages updated; versioned cache leaves old files intact | Reconcile refreshed output against source releases; recheck historical calibration/disclosures before refreshing dependent published reports |
| Stock currency | Reporting and quote currencies separated; unsafe mixed-currency, unknown-unit, and unreconciled ADR/share-basis valuations suppressed; provider ratios labeled; workbook shows unavailable values and reasons | Refresh or invalidate affected stock snapshots; verify representative live USD, foreign-reporting, and ADR workbooks |
| Portfolio truthfulness | Entry references, exposure reconciliation, dated valuations, quantitative disclosure, protected publisher and database migration | Apply migration in staging/production, exercise the deployed scheduler, publish the first dated transition, and verify both live pages |
| Freshness | Portfolio valuation date distinguished from run/publication timestamps; misleading portfolio live badge removed | Complete the cross-product source-observation/release/retrieval/publication timestamp inventory and stale-state policy |
| Login, offer, monetization | Prior audit and proposed sequencing retained | Implementation is later-phase work; no authentication or payment deployment in this pass |

## Validation

- **57 backend tests passed** covering accounting, first entries/re-entry, reruns, frozen history, price drift, splits/dividends, provider normalization, macro limits, concentration reserve, directional volume, legacy API enrichment, CPI, and currency safeguards. One external-data historical episode test was intentionally excluded.
- **24 frontend tests passed**, including portfolio display and currency behavior; the production build generated all 25 static pages successfully.
- **12 isolated PostgreSQL checks passed** using PGlite. They executed both schema migrations and verified immutable parent/child records, rejection of old publishers and changed curves, valid append publication, and rejection of a same-day replacement. No production database was connected.
- Browser checks used a local fixture built from downloaded public snapshots: both pages displayed entry references and gains; rebalance disclosures started collapsed and expanded successfully. Anonymous gating and a local signed-in fixture were checked. This is not a production login test.
- Git whitespace checks passed. Existing unrelated local changes were preserved. Pre-existing hook/navigation lint findings and a datetime deprecation warning remain outside this change.

## Deployment sequence

1. Export/backup published runs and child rows, including inactive same-day published records. Preserve a copy of each latest curve, entry register, and publication ID. Pause the old portfolio scheduler during migration.
2. Apply `sql/002_portfolio_publication_immutability.sql` after the existing schema, first in staging. Use `python scripts/run_portfolio_immutability_migration.py --backup <new-backup-path.json>` with the publisher's database environment. The runner backs up all four portfolio tables, holds publisher/table locks, and verifies every existing row is unchanged before committing. Choose a persistent backup location; the script refuses to overwrite an existing backup. The older `run_portfolio_snapshot_migration.py` only installs schema 001 and does not install these guards. The new publisher refuses a new publication if the required triggers are absent.
3. Deploy the updated backend and frontend together. The read-only snapshot API can enrich legacy entries and reconcile displayed legacy exposure without rewriting stored historical payloads. Legacy suggestions are not presented as recorded executions.
4. Refresh the new native-CPI cache, verify source observations and macro confidence/freshness, and revalidate affected calibration statements. Refresh affected currency-aware stock snapshots. Do not republish portfolio history to incorporate these corrections.
5. Run `python jobs/nightly_portfolio_refresh.py --all --dry-run` using the current publication date and normal configured environment. Review entry references, completed bars, macro target, SPY reserve, guards, and preserved curve prefix. If today's run already exists, the job correctly skips it; first migration publication must be on a later date.
6. Run the scheduled publisher for that later date. Verify the first ledger publication, next eligible execution, actual cash/weights, dated entry references, and same-day rerun identity. Confirm prior payloads and curves still match the backup.
7. Resume the updated scheduler. On missing prices, revised anchors, stale macro market data, or failed guards, preserve the last publication and investigate. Do not disable immutability guards or revert to the old curve-rebuilding publisher as a rollback.

## Production recovery — September 24, 2026

The immutability migration was applied after backing up 64 runs, 719 positions, 14,616 performance rows, and 58 rebalance rows. Every original row was verified unchanged after migration and after publication. Real-provider dry-runs passed for both strategies. The market parser now ignores entirely empty rows introduced by multi-ticker date alignment, while retaining invalid-action and required-session checks.

Both first ledger publications succeeded: Stock Alpha `7e535291-873e-4c80-8af0-899bb7ca1700` and SMID `dd0fd2e6-73a1-4b9c-9e2f-5b1ef848af68`. They preserve prior curves, mark positions using September 23 completed prices, and publish a 78.2% long target eligible from September 25. The transition does not invent historical trades or immediately replace existing allocations.

The Railway portfolio job uses cron `0 19,20 * * 1-5` with start command `python jobs/nightly_portfolio_refresh.py --all --scheduled-hour 15`. The New York timezone guard permits one daily run at 3 p.m. local time across daylight saving; the other UTC trigger exits before accessing the database. Market holidays are skipped. For a manual catch-up outside that hour, omit `--scheduled-hour`; do not backdate or bypass publication guards. New decisions retain the next-session opening reference convention; daily publication uses completed-session valuations.

Source-data calibration verification beyond the portfolio dry-run remains separate follow-up work.
