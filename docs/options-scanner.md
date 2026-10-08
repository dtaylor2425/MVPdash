# Bullish flow scanner

The private options desk scans every symbol in the latest published stock-ranking
payload. It ranks observed quote-inferred options premium targeting future
expirations, not predicted orders or returns. No portfolio history is changed.

## Collection

`jobs/options_scanner_refresh.py` uses the existing Theta subscription and shared
advisory lock. It collects regular-session trade/quote records for future expiries
through 30 calendar days, including exchange holidays and early closes. The scan
becomes eligible at 17:30 America/New_York. Earlier scheduled slots and non-trading
days exit before collection. Explicit historical runs use ranking membership dated
on or before the requested session. Same-day expired contracts are excluded.

The default strike range is 25 strikes on either side of spot plus an ATM strike
when available; it is a bounded sample, not the full chain. Greek arrays and open
interest are not requested in this first version. The production API only reads
stored snapshots and never triggers vendor collection.

`sql/013_options_scanner.sql` creates independent tables; the worker applies this
idempotent schema. Successful symbols persist immediately. A retry skips them;
pending/failed symbols resume for the same date, universe and methodology profile.
Changing the policy creates a distinct profile. Benchmark subsets never appear in
the default board or historical baseline. Failure never replaces a completed row.

## Railway configuration

Use the existing `theta-options-worker`, start command `python jobs/market_data_refresh.py`.
Set `OPTIONS_SCANNER_ENABLED=true`, `OPTIONS_SCANNER_MAX_SECONDS=900` and
`OPTIONS_SCANNER_MAX_REQUESTS=400`. Keep the existing DST-safe cron. The wrapper runs
the scanner after existing options and stock collectors, serially. Within a ticker,
at most two expiration requests run concurrently. Completed reruns
make no Theta requests. Runtime/request limits are checked between vendor calls;
in-flight calls and vendor retries can exceed the soft budgets. No new always-on service is needed.

Disable by setting `OPTIONS_SCANNER_ENABLED=false`; stored history remains recoverable.
Manual benchmark: `python jobs/options_scanner_refresh.py --tickers NVDA,ANET,AEHR --dry-run`.
Manual resume: `python jobs/options_scanner_refresh.py --date YYYY-MM-DD`.
`--force` explicitly recollects; do not use routinely. The UI reports incomplete
coverage rather than treating missing symbols as zero activity.

## Interpretation and safeguards

Verified standard condition codes are 0 (regular), 18 (electronic options
execution), and 95 (intermarket sweep). Auctions, crosses, complex/multileg orders,
unknown conditions and unsuitable quotes remain unclassified. The inference weights
premium by trade position within the bid/ask spread. Bullish equals call buying plus
put selling; bearish offsets equal call selling plus put buying. Opening/closing
positions and strategy intent remain unknown. Per-condition diagnostics make
excluded activity inspectable.

Unusualness is descriptive gross premium versus 20 prior consecutive completed
exchange sessions with the same profile and window. Until then it is unavailable.
It is not a forecast probability. The strict shortlist requires at least 50%
directional attribution coverage. Manual editorial observations may have lower
coverage, but require completed collection, positive consistent totals and at least
25 classifiable trades. Drafts quantify the classified portion and excluded premium;
this minimum is not statistical confidence. Copying does not publish to Substack.
Contract detail responses show the top 20 by net bullish premium and disclose the
full contract count. All aggregate totals still use the complete collected sample.

The private endpoint `/api/private/options-flow/scanner` retains internal-token
authentication; the frontend proxy also checks the existing user allowlist. No
credentials reach the browser. Same-session stock observations are optional context.

Vendor references: [trade/quote](https://thetadata.net/docs/operations/option_history_trade_quote.html),
[trade conditions](https://thetadata.net/docs/Articles/Errors-Exchanges-Conditions/Trade-Conditions.html).
