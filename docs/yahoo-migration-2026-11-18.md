# Yahoo / FRED migration — November 18, 2026

Stock workbook prices, financial statements and research use Yahoo Finance through yfinance. Macro series continue to use FRED. The old FMP search/PE dependency has been removed. Internal options routes return 404 and options collection/scanner/backfill CLI entrypoints exit without contacting ThetaData. Historical database records are retained.

The existing Railway worker is repurposed to stock-research-worker. Its legacy Dockerfile.theta-worker path remains for compatibility, but installs requirements-stock-worker.txt without ThetaData. Start command: python jobs/market_data_refresh.py (Yahoo stock research only). Recommended cron: 45 22 * * 1-5, after the 17:30 Eastern completed-session gate in both DST and standard time.

The new worker replaces full daily input histories from Yahoo (never splices vendors or split bases), prioritizes published holdings within its existing 60-symbol bound, and requires complete five-minute sessions before intraday analytics. Daily OHLC supports gap and closing-strength scores when intraday data is unavailable. VWAP, turnover and session volume shares remain estimates, never institutional flow. Last-good snapshots survive fetch failures. Legacy Theta research is not served as current; workbook price history remains the fallback until Yahoo publishes.

50/200 moving averages, price charts, financial graphs, fundamentals, valuation, relative strength, realized volatility, volume and descriptive participation scores remain available where Yahoo supplies the inputs. Historical P/E uses complete four-quarter EPS windows and is retrospective, not an investable point-in-time signal. Detailed options trade/quote aggression, sweeps and premium scanner are retired, not imitated with Yahoo option chains.

Portfolio entry records, execution history, returns and allocation logic are untouched.

Macro thesis stays paused with its existing no-op start command, no cron and NEVER restart policy. The staged permanent deletion was canceled. Original code and SQL remain recoverable; re-enable only deliberately.

Suggested workspace schedules: portfolio 0 19,20 * * 1-5 with its --scheduled-hour 15 gate unchanged; stock intelligence daily before portfolio; volatility hourly rather than every 15 minutes; FX weekly Monday unchanged. Retain detached database volume until its backup role has been established; do not delete production data for cosmetic cleanup.

Verification: 24 targeted tests passed; frontend production build passed; live Yahoo read-only AAPL smoke test returned 501 daily observations and 156 intraday bars over two sessions. Deployment and live-source verification are recorded separately after rollout.
