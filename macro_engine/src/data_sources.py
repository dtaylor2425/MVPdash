import os
import time
import pandas as pd
import yfinance as yf
from fredapi import Fred


def fetch_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)

    # Case A: single ticker — flat column index
    if len(tickers) == 1:
        t = tickers[0]
        if "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": t}).dropna(how="all")
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                s = df.xs("Close", axis=1, level=0)
                if isinstance(s, pd.DataFrame) and s.shape[1] >= 1:
                    col = s.columns[0]
                    return s[[col]].rename(columns={col: t}).dropna(how="all")
        return pd.DataFrame()

    # Case B: multiple tickers — MultiIndex with ticker at level 0
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        top_level = df.columns.get_level_values(0)
        for t in tickers:
            if t in top_level:
                sub = df[t]
                if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                    out[t] = sub["Close"]
        return pd.DataFrame(out).dropna(how="all")

    # Case C: fallback
    if "Close" in df.columns:
        return df[["Close"]].dropna(how="all")

    return pd.DataFrame()


from .storage import read_parquet, write_parquet, _parquet_path


def fetch_fred(series_map: dict, api_key: str) -> pd.DataFrame:
    """
    Fetch all series in series_map from FRED.
    If a series is unavailable, keep a NaN column rather than crashing.
    """
    fred     = Fred(api_key=api_key)
    out      = {}
    all_index = None

    for name, sid in series_map.items():
        try:
            s = fred.get_series(sid)
            s = pd.Series(s)
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            s.name = name
        except Exception:
            s = pd.Series(dtype="float64", name=name)

        out[name] = s
        if not s.empty:
            all_index = s.index if all_index is None else all_index.union(s.index)

    if all_index is None:
        return pd.DataFrame(out)

    all_index = pd.DatetimeIndex(sorted(all_index))
    aligned   = {
        name: (s.reindex(all_index) if not s.empty
               else pd.Series(index=all_index, dtype="float64", name=name))
        for name, s in out.items()
    }
    return pd.DataFrame(aligned)


# How old the parquet file can be before we force a fresh FRED fetch even if
# Streamlit's in-memory cache still has the old result.
_DISK_CACHE_MAX_AGE_SECONDS = 6 * 3600  # 6 hours


def get_fred_cached(
    series_map: dict[str, str],
    api_key: str,
    cache_dir: str,
    cache_name: str = "fred_macro",
) -> pd.DataFrame:
    """
    Fetch FRED data with two-layer caching:

    1. Disk cache (parquet) — busted if the file is older than
       _DISK_CACHE_MAX_AGE_SECONDS, so stale data never survives an app
       restart longer than 6 hours regardless of Streamlit's TTL.

    2. Streamlit's @st.cache_data TTL (set in the calling page) — prevents
       redundant fetches within the same running session.

    The returned DataFrame contains *only real FRED observations* before
    ffill(), so macro.index.max() correctly reflects the latest date FRED
    actually published data for any series.
    """
    # ── Check disk cache age ──────────────────────────────────────────────────
    cached     = None
    cache_file = _parquet_path(cache_dir, cache_name)
    disk_fresh = False

    try:
        if cache_file.exists():
            age = time.time() - os.path.getmtime(cache_file)
            if age < _DISK_CACHE_MAX_AGE_SECONDS:
                cached     = read_parquet(cache_dir, cache_name)
                disk_fresh = True   # file is young enough to trust
    except Exception:
        pass

    if not disk_fresh:
        # Disk cache is stale or missing — ignore it entirely and fetch fresh
        cached = None

    # ── Always attempt a live FRED fetch ─────────────────────────────────────
    try:
        fresh = fetch_fred(series_map, api_key)
        if cached is not None and not cached.empty:
            # Merge: prefer fresh observations over cached ones for the same date
            df = pd.concat([cached, fresh]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = fresh
    except Exception:
        # Network / API failure — fall back to whatever is on disk
        df = cached if cached is not None else pd.DataFrame()

    if df.empty:
        return df

    # ── Write updated cache to disk before ffill ──────────────────────────────
    # We save the un-ffilled data so that index.max() always reflects the real
    # last FRED publication date, not the last forward-filled synthetic row.
    write_parquet(df.sort_index(), cache_dir, cache_name)

    # ffill for downstream consumers that need contiguous daily series
    return df.sort_index().ffill()