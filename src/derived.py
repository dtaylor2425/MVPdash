"""
src/derived.py
═══════════════════════════════════════════════════════════════════════════════
Computed / derived macro series. The charts router falls back to
DERIVED_SERIES when a series key isn't found in raw FRED data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 5y5y Forward Rates
# ─────────────────────────────────────────────────────────────────────────────

def fwd_5y5y_real(macro, proxies):
    if "real10" not in macro.columns or "real5" not in macro.columns:
        return pd.Series(dtype=float)
    r10 = macro["real10"].dropna(); r5 = macro["real5"].dropna()
    idx = r10.index.intersection(r5.index)
    if len(idx) == 0: return pd.Series(dtype=float)
    return ((r10.reindex(idx) * 10 - r5.reindex(idx) * 5) / 5.0).dropna()


def fwd_5y5y_inflation(macro, proxies):
    for col in ["y10", "real10", "y5", "real5"]:
        if col not in macro.columns: return pd.Series(dtype=float)
    be10 = (macro["y10"] - macro["real10"]).dropna()
    be5  = (macro["y5"]  - macro["real5"]).dropna()
    idx = be10.index.intersection(be5.index)
    if len(idx) == 0: return pd.Series(dtype=float)
    return ((be10.reindex(idx) * 10 - be5.reindex(idx) * 5) / 5.0).dropna()


def breakeven_10y(macro, proxies):
    if "y10" not in macro.columns or "real10" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y10"] - macro["real10"]).dropna()


def breakeven_5y(macro, proxies):
    if "y5" not in macro.columns or "real5" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y5"] - macro["real5"]).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity
# ─────────────────────────────────────────────────────────────────────────────

def net_liquidity(macro, proxies):
    if "fed_assets" not in macro.columns: return pd.Series(dtype=float)
    fa  = macro["fed_assets"].dropna()
    rrp = macro.get("rrp", pd.Series(dtype=float)).dropna()
    tga = macro.get("tga", pd.Series(dtype=float)).dropna()
    idx = fa.index
    if not rrp.empty: idx = idx.intersection(rrp.index)
    if not tga.empty: idx = idx.intersection(tga.index)
    if len(idx) == 0: return fa
    result = fa.reindex(idx)
    if not rrp.empty: result = result - rrp.reindex(idx).fillna(0)
    if not tga.empty: result = result - tga.reindex(idx).fillna(0)
    return result.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Credit derived
# ─────────────────────────────────────────────────────────────────────────────

def hy_ig_diff(macro, proxies):
    if "hy_oas" not in macro.columns or "ig_oas" not in macro.columns:
        return pd.Series(dtype=float)
    hy = macro["hy_oas"].dropna(); ig = macro["ig_oas"].dropna()
    idx = hy.index.intersection(ig.index)
    return (hy.reindex(idx) - ig.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Monetary policy derived
# ─────────────────────────────────────────────────────────────────────────────

def real_fed_funds(macro, proxies):
    if "fed_funds" not in macro.columns or "cpi" not in macro.columns:
        return pd.Series(dtype=float)
    ff = macro["fed_funds"].dropna(); cpi = macro["cpi"].dropna()
    if len(cpi) < 13: return pd.Series(dtype=float)
    cpi_yoy = cpi.pct_change(12).dropna() * 100.0
    idx = ff.index.intersection(cpi_yoy.index)
    if len(idx) == 0: return pd.Series(dtype=float)
    return (ff.reindex(idx) - cpi_yoy.reindex(idx)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-asset ratios (proxy-based)
# ─────────────────────────────────────────────────────────────────────────────

def _ratio(col_a, col_b):
    """Factory for simple ratio series from proxies."""
    def fn(macro, proxies):
        if col_a not in proxies.columns or col_b not in proxies.columns:
            return pd.Series(dtype=float)
        a = proxies[col_a].dropna(); b = proxies[col_b].dropna()
        idx = a.index.intersection(b.index)
        if len(idx) == 0: return pd.Series(dtype=float)
        return (a.reindex(idx) / b.reindex(idx)).dropna()
    return fn


def spy_drawdown(macro, proxies):
    if "SPY" not in proxies.columns: return pd.Series(dtype=float)
    spy = proxies["SPY"].dropna()
    if spy.empty: return pd.Series(dtype=float)
    peak = spy.expanding().max()
    return ((spy - peak) / peak * 100).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Proxy passthroughs
# ─────────────────────────────────────────────────────────────────────────────

def _proxy(col):
    def fn(macro, proxies):
        if col not in proxies.columns: return pd.Series(dtype=float)
        return proxies[col].dropna()
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Master registry — charts router looks up series keys here
# ─────────────────────────────────────────────────────────────────────────────

DERIVED_SERIES: Dict[str, Callable] = {
    # Forward rates
    "fwd_5y5y_real":      fwd_5y5y_real,
    "fwd_5y5y_inflation": fwd_5y5y_inflation,
    "breakeven":          breakeven_10y,
    "breakeven_10y":      breakeven_10y,
    "breakeven_5y":       breakeven_5y,

    # Liquidity
    "net_liquidity":      net_liquidity,

    # Credit
    "hy_ig_diff":         hy_ig_diff,

    # Monetary
    "real_fed_funds":     real_fed_funds,

    # Cross-asset ratios
    "copper_gold":        _ratio("CPER", "GLD"),
    "rsp_spy":            _ratio("RSP", "SPY"),
    "qqq_spy":            _ratio("QQQ", "SPY"),
    "gold_silver":        _ratio("GLD", "SLV"),
    "tlt_spy":            _ratio("TLT", "SPY"),

    # Equity
    "spy_drawdown":       spy_drawdown,

    # VIX
    "vratio":             _ratio("^VIX", "^VIX3M"),
    "vix":                _proxy("^VIX"),
    "vix3m":              _proxy("^VIX3M"),
    "move":               _proxy("^MOVE"),

    # Commodities & ETFs
    "oil":                _proxy("USO"),
    "copper":             _proxy("CPER"),
    "gold":               _proxy("GLD"),
    "slv":                _proxy("SLV"),
    "tlt":                _proxy("TLT"),
    "hyg":                _proxy("HYG"),
    "btc":                _proxy("BTC-USD"),
    "spy":                _proxy("SPY"),
    "qqq":                _proxy("QQQ"),
}