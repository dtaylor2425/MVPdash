"""
api/services/options_flow_market_factor.py

Signal Research v4: pure functions for decomposing the frozen z_gross_premium_20d signal
(commit 19dc02c, validated -- partially -- in Signal Validation v3) into a market-wide
common factor and an ETF-specific residual, plus the forward market-outcome series (realized
vol, max drawdown, VIX change) needed to test what the common factor actually predicts.

No I/O here -- scripts/run_options_flow_market_factor_v4.py does the real data loading (reusing
v3's EXACT, frozen z_gross_premium_20d loaders verbatim) and calls into this module. Nothing in
this module recomputes or reinterprets z_gross_premium_20d itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def market_activity_factor(df: pd.DataFrame, feature_col: str, date_col: str = "date",
                           min_tickers: int = 3) -> pd.DataFrame:
    """
    Per date: the MEDIAN (primary, per the spec) and the equal-weight MEAN (secondary, for
    robustness) of `feature_col` across whatever tickers have a non-missing value that date.
    A date needs at least `min_tickers` non-missing values to get a factor value at all (never
    computed from 1-2 stragglers); dates with fewer are left as NaN, not imputed or dropped from
    the frame (the caller decides whether to drop them).

    Returns one row per date: date, market_activity_median, market_activity_mean, n_tickers.
    """
    g = df.groupby(date_col)[feature_col]
    n = g.apply(lambda s: int(s.notna().sum()))
    median = g.median()
    mean = g.mean()
    out = pd.DataFrame({"n_tickers": n, "market_activity_median": median, "market_activity_mean": mean})
    out.loc[out["n_tickers"] < min_tickers, ["market_activity_median", "market_activity_mean"]] = np.nan
    return out.reset_index().rename(columns={date_col: "date"})


def residual_activity_from_frame(df: pd.DataFrame, feature_col: str, market_by_date: pd.Series,
                                 date_col: str = "date") -> pd.Series:
    """residual_activity = ticker's own feature_col - market_activity(same date), where
    `market_by_date` is a Series indexed by date (e.g. market_activity_factor()'s median
    column). NaN propagates rather than being filled."""
    return df[feature_col] - df[date_col].map(market_by_date)


# --------------------------------------------------------------------------------------------
# Forward market-outcome series (realized vol, max drawdown, absolute return, VIX change) --
# same "honest missing data" discipline as api/services/options_flow_forward_returns.py: a
# horizon that hasn't happened yet in the price series is None, never 0.0 or imputed.
# --------------------------------------------------------------------------------------------

HORIZONS = (1, 3, 5, 10, 20)


def _session_return_series(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def forward_realized_vol(prices: pd.Series, dates: Sequence[Any], horizon: int) -> Dict[str, Optional[float]]:
    """Std of daily simple returns over the `horizon` sessions STRICTLY AFTER date T (T+1..T+h
    inclusive) -- never includes T's own return. Population std (ddof=0) since this describes a
    fixed forward window, not a sample used to infer a wider population."""
    idx = prices.index
    rets = _session_return_series(prices)
    out: Dict[str, Optional[float]] = {}
    for d in dates:
        ts = pd.Timestamp(d)
        key = str(d)
        if ts not in idx:
            out[key] = None
            continue
        pos = idx.get_loc(ts)
        window = rets.iloc[pos + 1: pos + 1 + horizon]
        out[key] = float(window.std(ddof=0)) if len(window) == horizon and window.notna().all() else None
    return out


def forward_max_drawdown(prices: pd.Series, dates: Sequence[Any], horizon: int) -> Dict[str, Optional[float]]:
    """Max peak-to-trough decline (negative number, e.g. -0.05 = -5%) within the `horizon`
    sessions strictly after T, using T's own close as the initial reference peak."""
    idx = prices.index
    out: Dict[str, Optional[float]] = {}
    for d in dates:
        ts = pd.Timestamp(d)
        key = str(d)
        if ts not in idx:
            out[key] = None
            continue
        pos = idx.get_loc(ts)
        window = prices.iloc[pos: pos + 1 + horizon]  # include T itself as the starting peak reference
        if len(window) != horizon + 1 or window.isna().any():
            out[key] = None
            continue
        running_max = window.cummax()
        drawdown = (window - running_max) / running_max
        out[key] = float(drawdown.iloc[1:].min())  # exclude T itself (drawdown there is always 0)
    return out


def vix_change(vix: pd.Series, dates: Sequence[Any], horizon: int) -> Dict[str, Optional[float]]:
    """VIX(T+horizon) - VIX(T), in VIX points (e.g. +3.2 = VIX rose 3.2 points)."""
    idx = vix.index
    out: Dict[str, Optional[float]] = {}
    for d in dates:
        ts = pd.Timestamp(d)
        key = str(d)
        if ts not in idx:
            out[key] = None
            continue
        pos = idx.get_loc(ts)
        fpos = pos + horizon
        if fpos >= len(idx) or pd.isna(vix.iloc[pos]):
            out[key] = None
            continue
        v0, vN = vix.iloc[pos], vix.iloc[fpos]
        out[key] = float(vN - v0) if pd.notna(vN) else None
    return out


def build_outcome_frame(prices: pd.Series, dates: Sequence[Any], vix: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per date with forward realized vol / max drawdown / VIX change at every horizon.
    Absolute return is NOT duplicated here -- callers derive it as abs(existing ret_Nd)."""
    rows: Dict[str, Dict[str, Any]] = {str(d): {"date": str(d)} for d in dates}
    for h in HORIZONS:
        vol = forward_realized_vol(prices, dates, h)
        dd = forward_max_drawdown(prices, dates, h)
        vc = vix_change(vix, dates, h) if vix is not None else {str(d): None for d in dates}
        for d in dates:
            key = str(d)
            rows[key][f"fwd_vol_{h}d"] = vol[key]
            rows[key][f"fwd_max_drawdown_{h}d"] = dd[key]
            rows[key][f"vix_change_{h}d"] = vc[key]
    return pd.DataFrame(list(rows.values()))


# --------------------------------------------------------------------------------------------
# Persistence / path analysis (item 8)
# --------------------------------------------------------------------------------------------

def autocorrelation(series: pd.Series, lag: int) -> Dict[str, Any]:
    """Pearson correlation of a (date-sorted) series against its own `lag`-session-ahead value.
    Drops any pair with a missing value on either side."""
    s = series.dropna()
    if len(s) <= lag + 2:
        return {"n": len(s), "lag": lag, "corr": None}
    a = s.iloc[:-lag].to_numpy()
    b = s.iloc[lag:].to_numpy()
    n = len(a)
    if n < 3:
        return {"n": n, "lag": lag, "corr": None}
    corr = float(np.corrcoef(a, b)[0, 1])
    return {"n": n, "lag": lag, "corr": corr}


def average_forward_path(prices: pd.Series, dates: Sequence[Any], max_k: int = 20) -> Dict[str, Any]:
    """
    For each date in `dates` (typically a quintile group's dates), the CUMULATIVE return path
    close(T+k)/close(T) - 1 for k=1..max_k, then averaged across all input dates at each k
    (an event-time / cumulative-response alignment, standard event-study construction). Also
    returns the rolling 5-session realized vol as of each k, averaged the same way. A given
    date contributes to a given k only if that many forward sessions actually exist for it
    (never padded/imputed) -- the returned "nAtK" says how many dates contributed at each k.
    """
    idx = prices.index
    rets = _session_return_series(prices)
    cum_paths: List[List[Optional[float]]] = []
    vol_paths: List[List[Optional[float]]] = []
    for d in dates:
        ts = pd.Timestamp(d)
        if ts not in idx:
            continue
        pos = idx.get_loc(ts)
        c0 = prices.iloc[pos]
        cum_row: List[Optional[float]] = []
        vol_row: List[Optional[float]] = []
        for k in range(1, max_k + 1):
            fpos = pos + k
            if fpos < len(idx) and pd.notna(c0) and pd.notna(prices.iloc[fpos]):
                cum_row.append(float(prices.iloc[fpos]) / float(c0) - 1.0)
            else:
                cum_row.append(None)
            vol_window = rets.iloc[max(pos + 1, fpos - 4): fpos + 1]
            vol_row.append(float(vol_window.std(ddof=0)) if len(vol_window) == 5 and vol_window.notna().all() else None)
        cum_paths.append(cum_row)
        vol_paths.append(vol_row)

    def _avg_by_k(paths: List[List[Optional[float]]]) -> Dict[str, Any]:
        avg, n_at_k = [], []
        for k in range(max_k):
            vals = [p[k] for p in paths if p[k] is not None]
            avg.append(float(np.mean(vals)) if vals else None)
            n_at_k.append(len(vals))
        return {"avgByK": avg, "nByK": n_at_k}

    return {"nDates": len(cum_paths), "cumulativeReturn": _avg_by_k(cum_paths), "rolling5dVol": _avg_by_k(vol_paths)}
