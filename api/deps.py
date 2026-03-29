"""
api/deps.py
Shared data loading with in-process caching.
Uses separate locks per resource to avoid deadlocks.
"""

import time
import threading
import concurrent.futures
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.regime import compute_regime_v3

_cache: dict = {}

# Separate lock per resource — prevents deadlock when get_regime
# calls get_macro which would re-enter the same lock
_macro_lock  = threading.Lock()
_prices_lock = threading.Lock()
_regime_lock = threading.Lock()

_TTL = 30 * 60  # 30 minutes

PRICE_TICKERS = [
    "SPY", "QQQ", "IWM", "RSP", "TLT", "HYG", "GLD",
    "XLU", "XLC", "CPER", "^VIX", "^VIX3M",
]


def _is_fresh(key: str) -> bool:
    if key not in _cache:
        return False
    return (time.time() - _cache[key]["ts"]) < _TTL


def _store(key: str, value):
    _cache[key] = {"data": value, "ts": time.time()}


def _parquet_path() -> Path:
    return Path(CACHE_DIR) / "fred_macro.parquet"


def _load_from_disk() -> pd.DataFrame:
    p = _parquet_path()
    if p.exists():
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            return df.sort_index().ffill()
        except Exception as e:
            print(f"disk load error: {e}")
    return pd.DataFrame()


def _fetch_fred_background():
    def _do():
        try:
            from src.data_sources import get_fred_cached
            fresh = get_fred_cached(
                FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro"
            ).sort_index()
            if not fresh.empty:
                _store("macro", fresh)
                print("background FRED refresh complete")
        except Exception as e:
            print(f"background FRED refresh failed: {e}")
    threading.Thread(target=_do, daemon=True).start()


def _fetch_prices_with_timeout(tickers: list, period: str = "2y",
                                timeout: int = 30) -> pd.DataFrame:
    def _download():
        try:
            df = yf.download(
                tickers=tickers, period=period,
                auto_adjust=True, progress=False, threads=True,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            out = {}
            if isinstance(df.columns, pd.MultiIndex):
                level0 = list(df.columns.get_level_values(0))
                level1 = list(df.columns.get_level_values(1))
                field_names = {"Close","Open","High","Low","Volume","Adj Close"}
                if level0[0] in field_names:
                    for t in tickers:
                        if t in level1:
                            try:
                                s = df.xs(t, axis=1, level=1)
                                if "Close" in s.columns:
                                    out[t] = s["Close"].dropna()
                            except Exception:
                                pass
                else:
                    for t in tickers:
                        if t in level0:
                            try:
                                sub = df[t]
                                for col in ["Close","close","Adj Close"]:
                                    if isinstance(sub, pd.DataFrame) and col in sub.columns:
                                        out[t] = sub[col].dropna()
                                        break
                                    elif isinstance(sub, pd.Series):
                                        out[t] = sub.dropna()
                                        break
                            except Exception:
                                pass
            else:
                for col in ["Close","close","Adj Close"]:
                    if col in df.columns:
                        if len(tickers) == 1:
                            out[tickers[0]] = df[col].dropna()
                        break
            if not out:
                return pd.DataFrame()
            return pd.DataFrame(out).dropna(how="all")
        except Exception as e:
            print(f"price fetch error: {e}")
            return pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_download)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("price fetch timed out")
            return pd.DataFrame()


def get_macro() -> pd.DataFrame:
    with _macro_lock:
        if _is_fresh("macro"):
            return _cache["macro"]["data"]
        print("loading FRED from disk cache...")
        macro = _load_from_disk()
        print(f"disk cache loaded: {macro.shape}")
        if macro.empty:
            print("no disk cache, fetching from FRED (slow)...")
            try:
                from src.data_sources import get_fred_cached
                macro = get_fred_cached(
                    FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro"
                ).sort_index()
            except Exception as e:
                print(f"FRED fetch failed: {e}")
                macro = pd.DataFrame()
        _store("macro", macro)
        _fetch_fred_background()
        return macro


def get_prices() -> pd.DataFrame:
    with _prices_lock:
        if _is_fresh("prices"):
            return _cache["prices"]["data"]
        print("fetching prices...")
        px = _fetch_prices_with_timeout(PRICE_TICKERS, period="2y", timeout=30)
        if px is None:
            px = pd.DataFrame()
        print(f"prices fetched: {px.shape}")
        _store("prices", px.sort_index() if not px.empty else px)
        return _cache["prices"]["data"]


def get_regime():
    with _regime_lock:
        if _is_fresh("regime"):
            return _cache["regime"]["data"]
        print("computing regime...")
        macro = get_macro()
        px    = get_prices()
        result = compute_regime_v3(
            macro=macro, proxies=px,
            lookback_trend=63, momentum_lookback_days=21,
        )
        print(f"regime computed: {result.score} {result.label}")
        _store("regime", result)
        return result


def invalidate():
    _cache.clear()