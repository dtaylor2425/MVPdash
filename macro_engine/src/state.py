# src/state.py
import pandas as pd
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_current_regime():
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set")

    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro").sort_index()
    px = fetch_prices(ROTATION_TICKERS, period="5y")
    if px is None or px.empty:
        px = pd.DataFrame()
    else:
        px = px.sort_index()

    regime = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63, momentum_lookback_days=21)
    return regime, macro, px