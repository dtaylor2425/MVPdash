# src/ranges.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class RangeSpec:
    label: str
    days: int
    ma_fast: int
    ma_slow: int
    trend_lookback: int


RANGES = {
    "1w": RangeSpec("1w", 7, 5, 20, 5),
    "1m": RangeSpec("1m", 35, 10, 50, 21),
    "3m": RangeSpec("3m", 110, 20, 100, 63),
    "1y": RangeSpec("1y", 400, 50, 200, 252),
    "5y": RangeSpec("5y", 2000, 50, 200, 252),
}


def slice_df(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    spec = RANGES.get(key, RANGES["1y"])
    end = df.index.max()
    start = end - pd.Timedelta(days=spec.days)
    out = df.loc[df.index >= start]
    return out


def slice_series(s: pd.Series, key: str) -> pd.Series:
    if s is None or s.empty:
        return s
    spec = RANGES.get(key, RANGES["1y"])
    end = s.index.max()
    start = end - pd.Timedelta(days=spec.days)
    out = s.loc[s.index >= start]
    return out