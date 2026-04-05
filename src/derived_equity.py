"""
src/derived_equity.py
═══════════════════════════════════════════════════════════════════════════════
Derived series specifically for the equity/commodity overlay charts
that trend on fintwit. Import these into derived.py's DERIVED_SERIES dict.

These require yfinance proxy data (SPY, QQQ, GLD, etc.) and optionally
FRED macro data for overlay context.
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable


# ─────────────────────────────────────────────────────────────────────────────
# SPY vs Real Yield overlay
# Shows how equity prices move inversely with real yields
# ─────────────────────────────────────────────────────────────────────────────

def spy_vs_real_yield(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """SPY normalized (rebased to 100) for overlay with real yield."""
    if "SPY" not in proxies.columns:
        return pd.Series(dtype=float)
    spy = proxies["SPY"].dropna()
    if spy.empty:
        return pd.Series(dtype=float)
    return (spy / spy.iloc[0] * 100).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Gold vs Real Yield (inverse relationship)
# Classic: when real yields fall, gold rises
# ─────────────────────────────────────────────────────────────────────────────

def gold_vs_real_yield(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """GLD normalized for overlay with real yield (inverted)."""
    if "GLD" not in proxies.columns:
        return pd.Series(dtype=float)
    gld = proxies["GLD"].dropna()
    if gld.empty:
        return pd.Series(dtype=float)
    return (gld / gld.iloc[0] * 100).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# SPY drawdown from all-time high
# Shows current distance from peak — viral when drawdowns deepen
# ─────────────────────────────────────────────────────────────────────────────

def spy_drawdown(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """SPY drawdown from rolling ATH in percentage terms."""
    if "SPY" not in proxies.columns:
        return pd.Series(dtype=float)
    spy = proxies["SPY"].dropna()
    if spy.empty:
        return pd.Series(dtype=float)
    peak = spy.expanding().max()
    dd = ((spy - peak) / peak * 100)
    return dd.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# QQQ/SPY ratio — tech relative strength
# Rising = tech outperforming. Falling = rotation out of tech
# ─────────────────────────────────────────────────────────────────────────────

def qqq_spy_ratio(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    if "QQQ" not in proxies.columns or "SPY" not in proxies.columns:
        return pd.Series(dtype=float)
    q = proxies["QQQ"].dropna()
    s = proxies["SPY"].dropna()
    idx = q.index.intersection(s.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return (q.reindex(idx) / s.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Gold/Silver ratio — fear gauge
# Rising = fear (gold outperforming). Falling = risk appetite
# ─────────────────────────────────────────────────────────────────────────────

def gold_silver_ratio(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    if "GLD" not in proxies.columns or "SLV" not in proxies.columns:
        return pd.Series(dtype=float)
    g = proxies["GLD"].dropna()
    s = proxies["SLV"].dropna()
    idx = g.index.intersection(s.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return (g.reindex(idx) / s.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 earnings yield vs 10Y Treasury
# When earnings yield > 10Y, stocks are "cheap" vs bonds
# This is the equity risk premium proxy
# ─────────────────────────────────────────────────────────────────────────────

def equity_risk_premium(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """Rough ERP = inverse of trailing PE proxy minus 10Y yield.
    Uses SPY price / earnings estimate. Very rough but directionally correct.
    For a proper version, pipe in actual S&P 500 EPS data.
    """
    # This is a placeholder — proper implementation needs S&P 500 EPS data
    # For now, return empty and we'll build it when we add EPS data
    return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# TLT vs SPY — bonds vs stocks rotation
# ─────────────────────────────────────────────────────────────────────────────

def tlt_spy_ratio(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    if "TLT" not in proxies.columns or "SPY" not in proxies.columns:
        return pd.Series(dtype=float)
    t = proxies["TLT"].dropna()
    s = proxies["SPY"].dropna()
    idx = t.index.intersection(s.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return (t.reindex(idx) / s.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Oil vs Dollar (inverse relationship)
# ─────────────────────────────────────────────────────────────────────────────

def oil_normalized(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    if "USO" not in proxies.columns:
        return pd.Series(dtype=float)
    oil = proxies["USO"].dropna()
    if oil.empty:
        return pd.Series(dtype=float)
    return (oil / oil.iloc[0] * 100).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Registry — add to DERIVED_SERIES in derived.py
# ─────────────────────────────────────────────────────────────────────────────

EQUITY_DERIVED: Dict[str, Callable] = {
    "spy_drawdown":     spy_drawdown,
    "qqq_spy":          qqq_spy_ratio,
    "gold_silver":      gold_silver_ratio,
    "tlt_spy":          tlt_spy_ratio,
    "spy_normalized":   spy_vs_real_yield,
    "gld_normalized":   gold_vs_real_yield,
    "oil_normalized":   oil_normalized,
}