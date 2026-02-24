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

    # Case A: single ticker often returns a flat column index
    if len(tickers) == 1:
        t = tickers[0]

        # flat columns like Open, High, Low, Close, Volume
        if "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": t}).dropna(how="all")

        # multiindex columns with Close somewhere
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                s = df.xs("Close", axis=1, level=0)
                if isinstance(s, pd.DataFrame) and s.shape[1] >= 1:
                    # sometimes the remaining column name is the ticker
                    col = s.columns[0]
                    return s[[col]].rename(columns={col: t}).dropna(how="all")

        return pd.DataFrame()

    # Case B: multiple tickers, likely multiindex with ticker at level 0
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        top_level = df.columns.get_level_values(0)
        for t in tickers:
            if t in top_level:
                sub = df[t]
                if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                    out[t] = sub["Close"]
        return pd.DataFrame(out).dropna(how="all")

    # Case C: fallback, try to use Close if present
    if "Close" in df.columns:
        return df[["Close"]].dropna(how="all")

    return pd.DataFrame()


import pandas as pd
from fredapi import Fred
from .storage import read_parquet, write_parquet

def fetch_fred(series_map: dict, api_key: str):
    """
    Robust FRED fetch:
    - series_map must be {friendly_name: fred_series_id}
    - if a series ID is invalid or unavailable, we keep a column of NaNs instead of crashing
    """
    import pandas as pd
    from fredapi import Fred

    fred = Fred(api_key=api_key)

    out = {}
    all_index = None

    for name, sid in series_map.items():
        try:
            s = fred.get_series(sid)
            s = pd.Series(s)
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            s.name = name
        except Exception:
            # Keep an empty series for this name, later we align index and fill with NaNs
            s = pd.Series(dtype="float64", name=name)

        out[name] = s

        if not s.empty:
            all_index = s.index if all_index is None else all_index.union(s.index)

    if all_index is None:
        # Everything failed
        return pd.DataFrame(out)

    all_index = pd.DatetimeIndex(sorted(all_index))

    # Align everything to a common index
    aligned = {}
    for name, s in out.items():
        if s.empty:
            aligned[name] = pd.Series(index=all_index, dtype="float64", name=name)
        else:
            aligned[name] = s.reindex(all_index)

    df = pd.DataFrame(aligned)
    return df

def get_fred_cached(series_map: dict[str, str], api_key: str, cache_dir: str, cache_name: str = "fred_macro") -> pd.DataFrame:
    cached = read_parquet(cache_dir, cache_name)

    # If we have cache, use it and try to refresh by appending the latest
    if cached is not None and not cached.empty:
        try:
            fresh = fetch_fred(series_map, api_key)
            df = pd.concat([cached, fresh]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        except Exception:
            df = cached
    else:
        df = fetch_fred(series_map, api_key)

    df = df.sort_index().ffill()
    write_parquet(df, cache_dir, cache_name)
    return df

