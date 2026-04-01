"""
src/derived.py
═══════════════════════════════════════════════════════════════════════════════
Computed / derived macro series that don't exist in FRED or yfinance directly.

Each function takes raw DataFrames (macro, proxies) and returns a pd.Series
with a DatetimeIndex. The charts router calls these when it encounters a
series key that matches a derived name.

All derived series are registered in DERIVED_SERIES at the bottom of this file.
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 5y5y Forward Rates
# Formula: forward = (long_rate × long_maturity - short_rate × short_maturity)
#                    / (long_maturity - short_maturity)
# ─────────────────────────────────────────────────────────────────────────────

def fwd_5y5y_real(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """Real 5y5y forward = (real10 × 10 - real5 × 5) / 5"""
    if "real10" not in macro.columns or "real5" not in macro.columns:
        return pd.Series(dtype=float)
    r10 = macro["real10"].dropna()
    r5  = macro["real5"].dropna()
    idx = r10.index.intersection(r5.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    fwd = (r10.reindex(idx) * 10 - r5.reindex(idx) * 5) / 5.0
    return fwd.dropna()


def fwd_5y5y_inflation(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """Inflation 5y5y forward from breakevens.
    BE_10 = y10 - real10,  BE_5 = y5 - real5
    Inflation 5y5y = (BE_10 × 10 - BE_5 × 5) / 5
    """
    for col in ["y10", "real10", "y5", "real5"]:
        if col not in macro.columns:
            return pd.Series(dtype=float)
    be10 = (macro["y10"] - macro["real10"]).dropna()
    be5  = (macro["y5"]  - macro["real5"]).dropna()
    idx = be10.index.intersection(be5.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    fwd = (be10.reindex(idx) * 10 - be5.reindex(idx) * 5) / 5.0
    return fwd.dropna()


def breakeven_10y(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """10Y breakeven inflation = nominal 10Y - real 10Y TIPS"""
    if "y10" not in macro.columns or "real10" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y10"] - macro["real10"]).dropna()


def breakeven_5y(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """5Y breakeven inflation = nominal 5Y - real 5Y TIPS"""
    if "y5" not in macro.columns or "real5" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y5"] - macro["real5"]).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Curve spreads
# ─────────────────────────────────────────────────────────────────────────────

def curve_2s10s(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    if "y10" not in macro.columns or "y2" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y10"] - macro["y2"]).dropna()


def curve_3m10(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    if "y10" not in macro.columns or "y3m" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y10"] - macro["y3m"]).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity
# ─────────────────────────────────────────────────────────────────────────────

def net_liquidity(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """Net Liquidity = Fed Assets - Reverse Repo - Treasury General Account.
    This is the real liquidity measure that drives asset prices.
    """
    if "fed_assets" not in macro.columns:
        return pd.Series(dtype=float)
    fa  = macro["fed_assets"].dropna()
    rrp = macro.get("rrp", pd.Series(dtype=float)).dropna()
    tga = macro.get("tga", pd.Series(dtype=float)).dropna()

    idx = fa.index
    if not rrp.empty:
        idx = idx.intersection(rrp.index)
    if not tga.empty:
        idx = idx.intersection(tga.index)
    if len(idx) == 0:
        return fa  # fallback to raw fed assets

    result = fa.reindex(idx)
    if not rrp.empty:
        result = result - rrp.reindex(idx).fillna(0)
    if not tga.empty:
        result = result - tga.reindex(idx).fillna(0)
    return result.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Credit derived
# ─────────────────────────────────────────────────────────────────────────────

def hy_ig_diff(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """HY-IG spread differential. Widening = risk discrimination increasing."""
    if "hy_oas" not in macro.columns or "ig_oas" not in macro.columns:
        return pd.Series(dtype=float)
    hy = macro["hy_oas"].dropna()
    ig = macro["ig_oas"].dropna()
    idx = hy.index.intersection(ig.index)
    return (hy.reindex(idx) - ig.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-asset ratios from proxies
# ─────────────────────────────────────────────────────────────────────────────

def copper_gold(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """Copper/Gold ratio — Dr. Copper thesis. Rising = growth > fear."""
    if "CPER" not in proxies.columns or "GLD" not in proxies.columns:
        return pd.Series(dtype=float)
    cu = proxies["CPER"].dropna()
    au = proxies["GLD"].dropna()
    idx = cu.index.intersection(au.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    ratio = cu.reindex(idx) / au.reindex(idx)
    return ratio.dropna()


def rsp_spy(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """RSP/SPY breadth ratio — equal weight vs cap weight."""
    if "RSP" not in proxies.columns or "SPY" not in proxies.columns:
        return pd.Series(dtype=float)
    return (proxies["RSP"] / proxies["SPY"]).dropna()


def vratio(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """V-Ratio = VIX / VIX3M. Above 1.0 = backwardation = stress."""
    vix_col = "^VIX"; v3m_col = "^VIX3M"
    if vix_col not in proxies.columns or v3m_col not in proxies.columns:
        return pd.Series(dtype=float)
    v = proxies[vix_col].dropna()
    v3 = proxies[v3m_col].dropna()
    idx = v.index.intersection(v3.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return (v.reindex(idx) / v3.reindex(idx)).dropna()


def real_fed_funds(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
    """Real Fed Funds = Fed Funds - CPI YoY%. Shows actual monetary tightness."""
    if "fed_funds" not in macro.columns or "cpi" not in macro.columns:
        return pd.Series(dtype=float)
    ff = macro["fed_funds"].dropna()
    cpi = macro["cpi"].dropna()
    if len(cpi) < 13:
        return pd.Series(dtype=float)
    cpi_yoy = cpi.pct_change(12).dropna() * 100.0
    idx = ff.index.intersection(cpi_yoy.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return (ff.reindex(idx) - cpi_yoy.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Proxy passthrough — yfinance series exposed as chart series
# ─────────────────────────────────────────────────────────────────────────────

def _proxy(col: str):
    """Factory for simple proxy passthrough functions."""
    def fn(macro: pd.DataFrame, proxies: pd.DataFrame) -> pd.Series:
        if col not in proxies.columns:
            return pd.Series(dtype=float)
        return proxies[col].dropna()
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps chart series key → compute function
# The charts router checks this dict for any series not found in raw FRED data.
# ─────────────────────────────────────────────────────────────────────────────

DERIVED_SERIES: Dict[str, Callable] = {
    # Computed rates
    "fwd_5y5y_real":      fwd_5y5y_real,
    "fwd_5y5y_inflation": fwd_5y5y_inflation,
    "breakeven":          breakeven_10y,
    "breakeven_10y":      breakeven_10y,
    "breakeven_5y":       breakeven_5y,

    # Curves
    "curve_2s10s":        curve_2s10s,
    "curve_3m10":         curve_3m10,

    # Liquidity
    "net_liquidity":      net_liquidity,

    # Credit derived
    "hy_ig_diff":         hy_ig_diff,

    # Cross-asset
    "copper_gold":        copper_gold,
    "rsp_spy":            rsp_spy,
    "vratio":             vratio,
    "real_fed_funds":     real_fed_funds,

    # Proxy passthroughs (yfinance → chart)
    "vix":                _proxy("^VIX"),
    "vix3m":              _proxy("^VIX3M"),
    "move":               _proxy("^MOVE"),
    "oil":                _proxy("USO"),
    "copper":             _proxy("CPER"),
    "gold":               _proxy("GLD"),
    "tlt":                _proxy("TLT"),
    "hyg":                _proxy("HYG"),
    "btc":                _proxy("BTC-USD"),
}