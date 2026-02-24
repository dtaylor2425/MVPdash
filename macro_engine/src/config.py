PRICE_LOOKBACK_YEARS = 5

import os

CACHE_DIR = "data/cache"

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

import os

CACHE_DIR = "data/cache"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

PRICE_PERIOD = "5y"

ROTATION_ETFS = [
    "SPY", "QQQ", "IWM",
    "XLE", "XLF", "XLK", "XLI",
    "XLV", "XLP",
    "GLD",
    "UUP"
]

FRED_SERIES = {
    "y2": "DGS2",               # 2y Treasury
    "y10": "DGS10",             # 10y Treasury
    "y3m": "DGS3MO",            # 3m Treasury
    "hy_oas": "BAMLH0A0HYM2",   # High yield OAS
    "real10": "DFII10",         # 10y TIPS real yield
    "fed_assets": "WALCL",      # Fed total assets
    "dollar_broad": "DTWEXBGS"  # Broad dollar index
}

# Proxies from yfinance for inflation impulse and breadth
YF_PROXIES = {
    "oil": "USO",   # crude proxy
    "copper": "CPER",
    "gold": "GLD",
    "iwm": "IWM",
    "spy": "SPY",
    "rsp": "RSP"    # optional breadth proxy
}
