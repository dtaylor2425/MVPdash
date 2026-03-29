import os

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
CACHE_DIR = "data/cache"

ROTATION_ETFS = [
    "SPY","QQQ","IWM","XLE","XLF","XLK","XLI","XLV","XLP","GLD"
]

PRICE_LOOKBACK_YEARS = 5


