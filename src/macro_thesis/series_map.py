"""
src/macro_thesis/series_map.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 2)

Extra FRED series beyond api/deps.py's FRED_SERIES / src/config.FRED_SERIES,
needed only by the macro thesis engine. Every ID here was checked live
against FRED's own /series endpoint on 2026-09-16 before being trusted --
re-verify with the same call if this module starts returning empty frames
(the FX build hit several dead IDs from its own spec's candidate table; do
not assume a plausible-looking FRED mnemonic actually resolves).

    QUSPAM770A   Total Credit to Private Non-Financial Sector, % of GDP, US
                 (BIS series, mirrored on FRED). Quarterly, starts 1947-10-01.
    FYGFGDQ188S  Federal Debt Held by the Public, % of GDP. Quarterly, 1970-01-01.
    FYFSGDA188S  Federal Surplus/Deficit, % of GDP. ANNUAL ONLY, 1929-01-01 --
                 there is no quarterly vintage; forward-fill within the year
                 when resampling to monthly.
    GDP          Nominal GDP, $bn SAAR. Quarterly, 1947-01-01.
    NROU         CBO Noncyclical Rate of Unemployment (NAIRU proxy). Quarterly,
                 1949-01-01.
    UNRATE       Unemployment rate. Monthly, 1948-01-01.
    THREEFYTP10  ACM 10-year zero-coupon term premium. Daily, 1990-01-02 --
                 resolves directly on FRED, no need to hit the NY Fed's own
                 CSV endpoint.
    T5YIFR       5-Year, 5-Year Forward Inflation Expectation Rate. Daily,
                 2003-01-02.

Everything else the spec's axis/phase tables need (claims, continuing claims,
NFCI, curve, breakevens via nominal-minus-real, CPI YoY, real fed funds,
policy rate, dollar, OAS spreads, VIX term structure) is already in
src.config.FRED_SERIES or api.deps.PRICE_TICKERS -- reused as-is, not
duplicated here.
"""

from __future__ import annotations

EXTRA_FRED_SERIES = {
    "credit_to_gdp": "QUSPAM770A",
    "fed_debt_pub_pct_gdp": "FYGFGDQ188S",
    "deficit_pct_gdp": "FYFSGDA188S",
    "nominal_gdp": "GDP",
    "nairu": "NROU",
    "unrate": "UNRATE",
    "term_premium_10y": "THREEFYTP10",
    "breakeven_5y5y": "T5YIFR",
}

# TIPS-implied real yields (DFII10) only exist from 2003 -- the inflation
# axis (10y breakeven = y10 - real10) and 5y5y forward can't be computed
# before this date. This caps how far back the quadrant history can honestly
# go, short of the spec's "25+ years" target (see engine module docstring).
EARLIEST_RELIABLE_QUADRANT_DATE = "2003-01-01"

# Asset universe for the by-quadrant backtest (spec section 6). Some of these
# have far less than 25y of history (GLD since 2004, DBC since 2006, UUP
# since 2007) -- the spec's own <24-observation suppression rule handles
# that per-quadrant rather than needing special-casing here.
ASSET_UNIVERSE = [
    "SPY", "QQQ", "IWM", "XLE", "XLF", "XLU", "XLP",
    "TLT", "IEF", "HYG", "LQD", "GLD", "DBC", "UUP", "EEM",
]

# RSP (equal-weight S&P, for breadth = RSP/SPY) and CPER (copper, for the
# copper/gold growth input) aren't in ASSET_UNIVERSE but are needed as raw
# axis inputs. DBC (already in ASSET_UNIVERSE) doubles as the inflation
# axis's "commodity basket" input -- one broad commodity ETF rather than an
# ad-hoc blend of oil/copper/gold proxies.
AXIS_ONLY_TICKERS = ["RSP", "CPER"]
ALL_PRICE_TICKERS = sorted(set(ASSET_UNIVERSE + AXIS_ONLY_TICKERS))
