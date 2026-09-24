# Macro Engine — portfolio exit results and days 15–30

September 23, 2026. Implemented locally; not deployed. No production trades, returns, accounts, newsletter settings, prices, or subscriptions were changed.

## Portfolio additions

Both Stock Alpha and SMID show **% change since bought** in their position cards and holdings tables, using the first published model entry reference established in the previous pass.

The rebalance log now displays **exit P/L percentages** for new recorded sales and trims. Each sale stores the entry date/reference price, exit session/reference price, whether it was a full exit or trim, and the percentage at that moment. Later quotes do not recalculate those results. Split-adjusted entry references prevent a stock split from creating a false gain or loss.

The percentage is `(exit reference price / split-adjusted entry reference price − 1) × 100`. It is price P/L since the recorded model entry, excluding dividends, fees and taxes; it is not average-cost realized P/L after intervening additions. That basis is disclosed beside the results. Historical exits without recorded references say “not recorded”; no current quote or reconstructed exit is substituted.

Historical published records remain unchanged. New exit details travel in the published payload and appear through the official rebalance-log API. The public history endpoint no longer exposes internal rebalance diagnostics, and SMID now gives signed-out visitors the same sign-in gate for history as Stock Alpha.

## Days 15–30 implementation

| Priority | Local result |
|---|---|
| Free versus paid offer | A shared comparison separates the free website account, free Substack subscription, and paid Substack subscription. Paid pricing and benefits are reviewed on Substack; the website does not claim automatic paid access or infer payment from a newsletter record. |
| Account status | Website registration, imported newsletter records, and unverified paid membership are described separately. Missing newsletter records are not described as cancelled subscriptions. Membership management links lead to Substack. |
| Login recovery | Specific messages for cancelled, expired, unverified, unavailable, and failed Google sign-ins. Intended local destinations survive retry; external/unsafe redirects are rejected. The initial login page has readable content before client hydration. |
| Service and logout failures | An unavailable account service is distinguished from being signed out. Retry updates both the account page and navigation. Failed logout stays on the page with an error rather than claiming success. |
| Dated research journey | Homepage → `/report` → dated model interpretation → supporting growth/inflation charts → asset/company research → editorial newsletter archive or subscription. The model review is clearly distinguished from an editorial newsletter issue. |
| Freshness and uncertainty | Actual snapshot date, seven-day stale warning, low conviction, and explicit model-under-review state. Loading, missing, invalid, future-dated, and failed responses do not produce an invented current reading. Three-month transformed-input changes are not presented as weekly market moves. |
| Links and source handoff | Homepage theme/geopolitical links use real brief anchors. Report evidence links use existing routes. Bounded `source` values are passed into Substack UTM links; this is attribution plumbing, not confirmed subscriber or payment measurement. |
| Newsletter copy | Removed unsupported guarantees of free weekly delivery and automatic account/subscription linking. Existing prices and member terms remain untouched. |

## Validation

- **63 backend tests passed**, including exit profits/losses/zero returns, full exits versus trims, missing entry prices, split-adjusted exits, frozen trade history, and public-history redaction. One external-data historical calibration test was excluded.
- **36 frontend tests passed**, including portfolio percentages, login redirects and error mapping, account-service failures, paid-status separation, dated/stale/invalid research, editorial links, and bounded attribution.
- Production build passed. Focused lint and whitespace checks were run; pre-existing findings are separated from the changed functionality.
- Local browser fixtures verified both portfolio exit logs with positive and negative results; research failure and recovery; stale and under-review notices; preserved login destination; account-service error and retry; shared navigation recovery; and failed logout.
- Browser checks used local fixtures and a downloaded public thesis response. They did not create a real account, send an email, complete Google OAuth, subscribe, or change billing.

The public thesis response read during this pass was dated **September 16, 2026** and explicitly marked the model under review. The new UI shows both facts. It does not represent that response as a fresh September 23 house view.

## Remaining launch acceptance

1. Complete the earlier portfolio database migration and source-data/calibration checks before enabling the new publisher. Keep all published history protected.
2. Deploy backend and frontend together; verify new exit metadata on a prospective publication without backfilling invented historical exits.
3. Complete a real Google sign-in with an authorized test account and confirm return-to-research, account status, logout, and anonymous gates on the deployed domain. This has not been verified by fixture testing.
4. Align the Substack About/offer copy with the website using the accompanying draft, while preserving existing subscriber terms. No Substack settings or published copy were edited in this pass.
5. Publish a genuinely dated editorial issue using the normal editorial process and link it from the research journey. The new model review does not substitute for that editorial publication.
6. Begin the days 31–60 measurement work only with confirmed signup/return/payment events; UTM links and iframe loads are not conversions.

Days 15–30 product changes are implemented locally. Deployment, real authentication acceptance, external offer alignment, and editorial publication remain necessary before calling the live milestone complete.
