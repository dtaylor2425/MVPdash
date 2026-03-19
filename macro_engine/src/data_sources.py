import os
import time
import pandas as pd
import yfinance as yf
from fredapi import Fred


def fetch_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """
    Robust yfinance downloader that handles both old and new column formats.

    yfinance >= 0.2.31 changed the MultiIndex structure:
      Old: columns = MultiIndex[(field, ticker), ...]  — ticker at level 1
      New: columns = MultiIndex[(ticker, field), ...]  — ticker at level 0
           OR flat columns when group_by is removed

    We detect which format we got and handle all cases.
    """
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    try:
        # New yfinance API — no group_by, returns (ticker, field) MultiIndex
        df = yf.download(
            tickers=tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    out = {}

    # ── MultiIndex columns ────────────────────────────────────────────────────
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))

        # Detect orientation: if level0 contains field names like "Close"
        # we have (field, ticker); if level0 contains tickers we have (ticker, field)
        field_names = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
        if level0[0] in field_names:
            # Old format: (field, ticker) — ticker at level 1
            for t in tickers:
                if t in level1:
                    try:
                        s = df.xs(t, axis=1, level=1)
                        if "Close" in s.columns:
                            out[t] = s["Close"].dropna()
                    except Exception:
                        pass
        else:
            # New format: (ticker, field) — ticker at level 0
            for t in tickers:
                if t in level0:
                    try:
                        sub = df[t]
                        if isinstance(sub, pd.DataFrame):
                            # field name varies: "Close" or "close"
                            for col in ["Close", "close", "Adj Close"]:
                                if col in sub.columns:
                                    out[t] = sub[col].dropna()
                                    break
                        elif isinstance(sub, pd.Series):
                            out[t] = sub.dropna()
                    except Exception:
                        pass

    # ── Flat columns (single ticker or flattened) ─────────────────────────────
    else:
        for col in ["Close", "close", "Adj Close"]:
            if col in df.columns:
                if len(tickers) == 1:
                    out[tickers[0]] = df[col].dropna()
                else:
                    # Each column might be a ticker
                    for t in tickers:
                        if t in df.columns:
                            out[t] = df[t].dropna()
                break

    if not out:
        return pd.DataFrame()

    result = pd.DataFrame(out)
    return result.dropna(how="all")


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