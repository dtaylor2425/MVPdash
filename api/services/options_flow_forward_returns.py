"""
api/services/options_flow_forward_returns.py

Pure forward-return computation for the Macro Options Flow predictive study. No network,
no I/O -- scripts/build_options_flow_forward_returns.py does the real price fetch (via
src.data_sources.fetch_prices, the existing Macro Engine historical ETF price source) and
calls into this module.

Forward returns are close-to-close, counted in TRADING sessions (not calendar days):
    ret_Nd(T) = close(the N-th trading session after T) / close(T) - 1

A horizon is left as None (never 0.0) whenever the close N sessions ahead does not exist yet
in the price series -- real, honest missing data, not a manufactured value. This is why the
newest dates in a research window typically lack the longer horizons: those future closes
have not happened yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

HORIZONS = (1, 3, 5, 10, 20)


def compute_forward_returns(prices: pd.DataFrame, dates: List[Any], tickers: List[str]) -> pd.DataFrame:
    """
    prices: DataFrame indexed by trading-session date (ascending, no gaps introduced), one
    column per ticker, close prices (as returned by src.data_sources.fetch_prices).
    dates: the feature dates to compute outcomes for (need not be every date in `prices`).

    Returns one row per (date, ticker) with columns date, ticker, close,
    ret_1d..ret_20d (float or None), and closeTplusN for each horizon (for auditability).
    """
    idx = prices.index
    rows: List[Dict[str, Any]] = []
    for t in tickers:
        if t not in prices.columns:
            for d in dates:
                rows.append({"date": d, "ticker": t, "close": None,
                            **{f"ret_{h}d": None for h in HORIZONS}})
            continue
        series = prices[t]
        for d in dates:
            ts = pd.Timestamp(d)
            if ts not in idx:
                rows.append({"date": d, "ticker": t, "close": None,
                            **{f"ret_{h}d": None for h in HORIZONS}})
                continue
            pos = idx.get_loc(ts)
            c0 = series.iloc[pos]
            row: Dict[str, Any] = {"date": d, "ticker": t, "close": _f(c0)}
            for h in HORIZONS:
                fpos = pos + h
                if fpos < len(idx) and pd.notna(c0):
                    cN = series.iloc[fpos]
                    row[f"ret_{h}d"] = (float(cN) / float(c0) - 1.0) if pd.notna(cN) and c0 else None
                else:
                    row[f"ret_{h}d"] = None  # future session hasn't happened yet -- never 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _f(x: Any) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


def verify_no_future_prices_in_features(feature_dates: List[Any], last_available_price_date: Any) -> Dict[str, Any]:
    """
    Sanity check invoked by the study script: every FEATURE date must be <= the last date we
    have ANY price for (a trivial necessary condition -- the real guarantee that features
    never see future option/Greek data comes from the backfill's point-in-time architecture,
    exercised by tests/test_options_flow_backfill.py's future-IV-leakage tests). Returns a
    dict the report can quote directly.
    """
    max_feature_date = max(feature_dates)
    # Dates may arrive as datetime.date or ISO strings (e.g. after a round trip through a
    # str-typed DataFrame column) -- compare as ISO strings, which sort identically to the
    # underlying dates, rather than assume a matching type on both sides.
    max_str = str(max_feature_date)
    last_str = str(last_available_price_date)
    return {
        "maxFeatureDate": max_str,
        "lastAvailablePriceDate": last_str,
        "featuresPrecedeOrEqualPriceHistory": max_str <= last_str,
    }
