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
    "y10":          "DGS10",         # 10y Treasury
    # Credit
    "hy_oas":       "BAMLH0A0HYM2",  # High yield OAS
    # Real yields / inflation
    "real10":       "DFII10",        # 10y TIPS real yield
    "cpi":          "CPIAUCSL",      # CPI all urban, NSA -- converted to YoY in page
    "fed_funds":    "FEDFUNDS",      # Effective fed funds rate
    # Liquidity / dollar
    "fed_assets":   "WALCL",         # Fed total assets
    "dollar_broad": "DTWEXBGS",      # Broad dollar index
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