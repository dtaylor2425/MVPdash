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
    "IGV", "SMH",
]

FRED_SERIES = {
    "y3m": "DGS3MO", "y2": "DGS2", "y5": "DGS5",
    "y10": "DGS10", "y30": "DGS30",
    "hy_oas": "BAMLH0A0HYM2", "ig_oas": "BAMLC0A0CM",
    "real10": "DFII10", "real5": "DFII5",
    "cpi": "CPIAUCSL", "cpi_core": "CPILFESL",
    "pce": "PCEPI", "fed_funds": "FEDFUNDS",
    "fed_assets": "WALCL", "dollar_broad": "DTWEXBGS",
    "rrp": "RRPONTSYD", "tga": "WTREGEN",
    "init_claims": "ICSA", "cont_claims": "CCSA",
    "nfci": "NFCI", "umich": "UMCSENT",
}

YF_PROXIES = {
    # Core equity
    "spy": "SPY", "qqq": "QQQ", "iwm": "IWM", "rsp": "RSP",
    # Sectors
    "xle": "XLE", "xlf": "XLF", "xlk": "XLK", "smh": "SMH",
    "xli": "XLI", "xlv": "XLV", "xlp": "XLP", "xlu": "XLU", "xlc": "XLC",
    # Commodities
    "oil": "USO", "copper": "CPER", "gold": "GLD", "slv": "SLV",
    # VIX
    "vix": "^VIX", "vix3m": "^VIX3M", "vix6m": "^VIX6M",
    # Bonds & vol
    "move": "^MOVE", "tlt": "TLT", "hyg": "HYG",
    # Crypto
    "btc": "BTC-USD",
}