import pandas as pd
import yfinance as yf
from fredapi import Fred

def fetch_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker"
    )
    if len(tickers) == 1:
        out = df["Close"].to_frame(tickers[0])
    else:
        out = pd.DataFrame({t: df[t]["Close"] for t in tickers})
    out.index = pd.to_datetime(out.index)
    return out.dropna(how="all")

def fetch_fred_series(series_map: dict[str, str], api_key: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    data = {}
    for name, sid in series_map.items():
        s = fred.get_series(sid)
        s.index = pd.to_datetime(s.index)
        data[name] = s
    df = pd.DataFrame(data).sort_index()
    return df
