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
    # Rates — full curve
    "y3m":          "DGS3MO",        # 3m Treasury
    "y2":           "DGS2",          # 2y Treasury
    "y5":           "DGS5",          # 5y Treasury
    "y10":          "DGS10",         # 10y Treasury
    "y30":          "DGS30",         # 30y Treasury

    # Credit
    "hy_oas":       "BAMLH0A0HYM2",  # High yield OAS
    "ig_oas":       "BAMLC0A0CM",    # IG OAS

    # Real yields / inflation
    "real10":       "DFII10",        # 10y TIPS real yield
    "real5":        "DFII5",         # 5y TIPS real yield
    "cpi":          "CPIAUCSL",      # CPI all urban, NSA
    "cpi_core":     "CPILFESL",      # Core CPI (ex food & energy)
    "pce":          "PCEPI",         # PCE price index (Fed's preferred)
    "fed_funds":    "FEDFUNDS",      # Effective fed funds rate

    # Liquidity / dollar
    "fed_assets":   "WALCL",         # Fed total assets
    "dollar_broad": "DTWEXBGS",      # Broad dollar index
    "rrp":          "RRPONTSYD",     # Reverse repo (for net liquidity)
    "tga":          "WTREGEN",       # Treasury General Account

    # Growth momentum
    "init_claims":  "ICSA",          # Initial jobless claims (weekly)
    "cont_claims":  "CCSA",          # Continuing claims

    # Financial conditions & sentiment
    "nfci":         "NFCI",          # Chicago Fed National Financial Conditions Index
    "umich":        "UMCSENT",       # U Michigan Consumer Sentiment
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
    # Additional proxies
    "move":   "^MOVE",      # Bond volatility index
    "tlt":    "TLT",        # Long treasury ETF
    "hyg":    "HYG",        # HY bond ETF (price proxy)
    "btc":    "BTC-USD",    # Bitcoin
}