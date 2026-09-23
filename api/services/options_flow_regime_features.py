"""
api/services/options_flow_regime_features.py

Pure, predetermined (never optimized) regime classifiers for Signal Validation v3, item 10.
Every split here is a plain median/sign/fixed-window rule on already-public market data (SPY
price history) or an existing options-flow feature (iv_percentile_60d) -- none are tuned to
produce a favorable result. No I/O: callers pass in a price Series (date-indexed) or a DataFrame
already carrying the feature to split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def chronological_half_split(dates: Sequence[Any]) -> List[str]:
    """'first_half' / 'second_half' by chronological RANK within the given date list (the
    median date itself falls in first_half), independent of calendar gaps."""
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    half = len(order) // 2
    label = [None] * len(dates)
    for rank, idx in enumerate(order):
        label[idx] = "first_half" if rank < half else "second_half"
    return label


def _as_date_indexed_series(prices: pd.Series) -> pd.Series:
    s = prices.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def sma_regime(prices: pd.Series, dates: Sequence[Any], window: int = 200) -> Dict[str, Any]:
    """
    'above' / 'below' the trailing `window`-session SMA (inclusive of the date itself -- the
    standard technical-indicator convention; this is a market-observable REGIME label, not a
    forward-looking options-flow feature, so including the current close is not a leakage
    concern the way it would be for a predictive feature). A date with fewer than `window` prior
    sessions of price history gets None (never a fabricated regime).

    Returns {"byDate": {date_str: 'above'|'below'|None}, "smaByDate": {date_str: float|None}}.
    """
    s = _as_date_indexed_series(prices)
    sma = s.rolling(window, min_periods=window).mean()
    by_date, sma_by_date = {}, {}
    for d in dates:
        ts = pd.Timestamp(d)
        key = str(d)
        if ts not in s.index or ts not in sma.index or pd.isna(sma.loc[ts]):
            by_date[key], sma_by_date[key] = None, None
            continue
        by_date[key] = "above" if s.loc[ts] > sma.loc[ts] else "below"
        sma_by_date[key] = float(sma.loc[ts])
    return {"byDate": by_date, "smaByDate": sma_by_date, "window": window}


def realized_vol_regime(prices: pd.Series, dates: Sequence[Any], window: int = 20) -> Dict[str, Any]:
    """
    'high' / 'low' trailing `window`-session realized volatility (std of daily simple returns,
    inclusive of the date itself), split on the MEDIAN of the values actually observed at the
    given `dates` (a predetermined split of the sample, not an externally chosen number).
    """
    s = _as_date_indexed_series(prices)
    rets = s.pct_change()
    vol = rets.rolling(window, min_periods=window).std()
    raw: Dict[str, Optional[float]] = {}
    for d in dates:
        ts = pd.Timestamp(d)
        raw[str(d)] = float(vol.loc[ts]) if (ts in vol.index and pd.notna(vol.loc[ts])) else None
    valid_vals = [v for v in raw.values() if v is not None]
    if len(valid_vals) < 2:
        return {"byDate": {k: None for k in raw}, "volByDate": raw, "medianSplit": None, "window": window}
    median = float(np.median(valid_vals))
    by_date = {k: (None if v is None else ("high" if v > median else "low")) for k, v in raw.items()}
    return {"byDate": by_date, "volByDate": raw, "medianSplit": median, "window": window}


def momentum_regime(prices: pd.Series, dates: Sequence[Any], window: int = 20) -> Dict[str, Any]:
    """'positive' / 'negative' trailing `window`-session price momentum (simple return from the
    close `window` sessions ago to today's close, inclusive)."""
    s = _as_date_indexed_series(prices)
    mom = s.pct_change(periods=window)
    by_date, mom_by_date = {}, {}
    for d in dates:
        ts = pd.Timestamp(d)
        key = str(d)
        if ts not in mom.index or pd.isna(mom.loc[ts]):
            by_date[key], mom_by_date[key] = None, None
            continue
        by_date[key] = "positive" if mom.loc[ts] > 0 else "negative"
        mom_by_date[key] = float(mom.loc[ts])
    return {"byDate": by_date, "momentumByDate": mom_by_date, "window": window}


def iv_percentile_regime(df: pd.DataFrame, feature_col: str = "iv_percentile_60d") -> pd.Series:
    """'high'/'low' median split of an EXISTING options-flow feature (per-row, i.e. ETF-specific,
    unlike the SPY-based market-wide regimes above)."""
    valid = df[feature_col].dropna()
    if len(valid) < 2:
        return pd.Series([None] * len(df), index=df.index)
    median = float(valid.median())
    return df[feature_col].apply(lambda v: None if pd.isna(v) else ("high" if v > median else "low"))


def market_wide_activity_factor(
    df: pd.DataFrame, feature_col: str, date_col: str = "date", ticker_col: str = "ticker",
) -> pd.Series:
    """Per-date CROSS-SECTIONAL MEDIAN of `feature_col` across all tickers observed that date --
    "is today an unusually active day for the whole options market," as distinct from "is THIS
    ETF unusually active." Broadcast back to every row sharing that date. A date's median is
    computed from whatever tickers have a non-missing value that day (never imputed)."""
    medians = df.groupby(date_col)[feature_col].median()
    return df[date_col].map(medians)
