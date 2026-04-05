"""
src/derived.py
Computed / derived macro series. The charts router falls back to
DERIVED_SERIES when a series key isn't found in raw FRED/inline data.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict


def _find_col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n].dropna()
    return pd.Series(dtype=float)


def fwd_5y5y_real(macro, px):
    if "real10" not in macro.columns or "real5" not in macro.columns:
        return pd.Series(dtype=float)
    r10 = macro["real10"].dropna(); r5 = macro["real5"].dropna()
    idx = r10.index.intersection(r5.index)
    if len(idx) == 0: return pd.Series(dtype=float)
    return ((r10.reindex(idx) * 10 - r5.reindex(idx) * 5) / 5.0).dropna()


def fwd_5y5y_inflation(macro, px):
    for col in ["y10", "real10", "y5", "real5"]:
        if col not in macro.columns: return pd.Series(dtype=float)
    be10 = (macro["y10"] - macro["real10"]).dropna()
    be5  = (macro["y5"]  - macro["real5"]).dropna()
    idx = be10.index.intersection(be5.index)
    if len(idx) == 0: return pd.Series(dtype=float)
    return ((be10.reindex(idx) * 10 - be5.reindex(idx) * 5) / 5.0).dropna()


def breakeven_10y(macro, px):
    if "y10" not in macro.columns or "real10" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y10"] - macro["real10"]).dropna()


def breakeven_5y(macro, px):
    if "y5" not in macro.columns or "real5" not in macro.columns:
        return pd.Series(dtype=float)
    return (macro["y5"] - macro["real5"]).dropna()


def net_liquidity(macro, px):
    if "fed_assets" not in macro.columns: return pd.Series(dtype=float)
    fa = macro["fed_assets"].dropna()
    rrp = macro["rrp"].dropna() if "rrp" in macro.columns else pd.Series(dtype=float)
    tga = macro["tga"].dropna() if "tga" in macro.columns else pd.Series(dtype=float)
    idx = fa.index
    if not rrp.empty: idx = idx.intersection(rrp.index)
    if not tga.empty: idx = idx.intersection(tga.index)
    if len(idx) == 0: return fa
    result = fa.reindex(idx)
    if not rrp.empty: result = result - rrp.reindex(idx).fillna(0)
    if not tga.empty: result = result - tga.reindex(idx).fillna(0)
    return result.dropna()


def hy_ig_diff(macro, px):
    if "hy_oas" not in macro.columns or "ig_oas" not in macro.columns:
        return pd.Series(dtype=float)
    hy = macro["hy_oas"].dropna(); ig = macro["ig_oas"].dropna()
    idx = hy.index.intersection(ig.index)
    return (hy.reindex(idx) - ig.reindex(idx)).dropna()


def real_fed_funds(macro, px):
    if "fed_funds" not in macro.columns or "cpi" not in macro.columns:
        return pd.Series(dtype=float)
    ff = macro["fed_funds"].dropna(); cpi = macro["cpi"].dropna()
    if len(cpi) < 13: return pd.Series(dtype=float)
    cpi_yoy = cpi.pct_change(12).dropna() * 100.0
    idx = ff.index.intersection(cpi_yoy.index)
    if len(idx) == 0: return pd.Series(dtype=float)
    return (ff.reindex(idx) - cpi_yoy.reindex(idx)).dropna()


def cpi_yoy(macro, px):
    if "cpi" not in macro.columns: return pd.Series(dtype=float)
    cpi = macro["cpi"].dropna()
    if len(cpi) < 13: return pd.Series(dtype=float)
    return (cpi.pct_change(12) * 100).dropna()


def _ratio(names_a, names_b):
    if isinstance(names_a, str): names_a = (names_a,)
    if isinstance(names_b, str): names_b = (names_b,)
    def fn(macro, px):
        a = _find_col(px, *names_a)
        b = _find_col(px, *names_b)
        if a.empty or b.empty: return pd.Series(dtype=float)
        idx = a.index.intersection(b.index)
        if len(idx) == 0: return pd.Series(dtype=float)
        return (a.reindex(idx) / b.reindex(idx)).dropna()
    return fn


def spy_drawdown(macro, px):
    spy = _find_col(px, "SPY")
    if spy.empty: return pd.Series(dtype=float)
    peak = spy.expanding().max()
    return ((spy - peak) / peak * 100).dropna()


def _proxy(*names):
    def fn(macro, px):
        return _find_col(px, *names)
    return fn


DERIVED_SERIES: Dict[str, Callable] = {
    "fwd_5y5y_real":      fwd_5y5y_real,
    "fwd_5y5y_inflation": fwd_5y5y_inflation,
    "breakeven":          breakeven_10y,
    "breakeven_10y":      breakeven_10y,
    "breakeven_5y":       breakeven_5y,
    "net_liquidity":      net_liquidity,
    "hy_ig_diff":         hy_ig_diff,
    "real_fed_funds":     real_fed_funds,
    "cpi_yoy":            cpi_yoy,
    "copper_gold":        _ratio(("CPER",), ("GLD",)),
    "rsp_spy":            _ratio(("RSP",), ("SPY",)),
    "qqq_spy":            _ratio(("QQQ",), ("SPY",)),
    "gold_silver":        _ratio(("GLD",), ("SLV",)),
    "tlt_spy":            _ratio(("TLT",), ("SPY",)),
    "spy_drawdown":       spy_drawdown,
    "vratio":             _ratio(("^VIX",), ("^VIX3M",)),
    "vix":                _proxy("^VIX"),
    "vix3m":              _proxy("^VIX3M"),
    "move":               _proxy("^MOVE"),
    "oil":                _proxy("USO"),
    "copper":             _proxy("CPER"),
    "gold":               _proxy("GLD"),
    "slv":                _proxy("SLV"),
    "tlt":                _proxy("TLT"),
    "hyg":                _proxy("HYG"),
    "btc":                _proxy("BTC-USD", "BTC"),
    "spy":                _proxy("SPY"),
    "qqq":                _proxy("QQQ"),
}