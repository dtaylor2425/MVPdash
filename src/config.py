import os

CACHE_DIR            = "data/cache"
FRED_API_KEY         = os.getenv("FRED_API_KEY", "")
PRICE_LOOKBACK_YEARS = 5
PRICE_PERIOD         = "5y"

ROTATION_ETFS = [
    "SPY", "QQQ", "IWM",
    "XLE", "XLF", "XLK", "XLI",
    "XLV", "XLP",
    "GLD", "UUP",
    "IGV", "SMH",          # software (IGV) and semiconductors (SMH)
]

FRED_SERIES = {
    # Rates
    "y3m":          "DGS3MO",        # 3m Treasury
    "y2":           "DGS2",          # 2y Treasury
    "y5":           "DGS5",          # 5y Treasury  — needed for 5y5y forward
    "y10":          "DGS10",         # 10y Treasury
    # Credit
    "hy_oas":       "BAMLH0A0HYM2",  # High yield OAS
    # Real yields / inflation
    "real10":       "DFII10",        # 10y TIPS real yield
    "real5":        "DFII5",         # 5y TIPS real yield — needed for real 5y5y forward
    "cpi":          "CPIAUCSL",      # CPI all urban, NSA -- converted to YoY in page
    "fed_funds":    "FEDFUNDS",      # Effective fed funds rate
    # Liquidity / dollar
    "fed_assets":   "WALCL",         # Fed total assets
    "dollar_broad": "DTWEXBGS",      # Broad dollar index
    # Growth momentum (Pillar 1 — v5 regime)
    "init_claims":  "ICSA",          # Initial jobless claims (weekly)
    "cont_claims":  "CCSA",          # Continuing claims
    # Investment grade credit (Pillar 4 — v5 regime)
    "ig_oas":       "BAMLC0A0CM",    # IG OAS (complement to HY)
}

# yfinance proxies
YF_PROXIES = {
    "oil":    "USO",
    "copper": "CPER",
    "gold":   "GLD",
    "iwm":    "IWM",
    "spy":    "SPY",
    "rsp":    "RSP",
    # VIX term structure
    "vix":    "^VIX",
    "vix3m":  "^VIX3M",
    "vix6m":  "^VIX6M",
}