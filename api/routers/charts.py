"""
api/routers/charts.py
GET /api/charts/{series}  — time series data for a named signal
GET /api/charts/price/{ticker} — OHLC/close for an ETF

Supports raw FRED series, derived/computed series (from src/derived.py),
and yfinance proxy passthroughs.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd
import numpy as np

from api.deps import get_macro, get_prices

router = APIRouter(tags=["Charts"])

RANGES = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "2y": 504, "5y": 1260}


def _slice(s: pd.Series, rng: str) -> pd.Series:
    days = RANGES.get(rng, 252)
    return s.dropna().iloc[-days:] if len(s.dropna()) > days else s.dropna()


def _to_points(s: pd.Series) -> list:
    return [
        {"date": str(idx.date()), "value": round(float(v), 5)}
        for idx, v in s.items()
        if not np.isnan(v)
    ]


# ── Inline series (fast, no import overhead) ─────────────────────────────────
def _build_inline_map(macro, px):
    """Series that are simple column lookups or one-line computations."""
    def _col(name):
        return macro[name].dropna() if name in macro.columns else pd.Series(dtype=float)

    y10 = _col("y10"); y2 = _col("y2"); y3m = _col("y3m")
    y5 = _col("y5"); y30 = _col("y30")
    r10 = _col("real10"); r5 = _col("real5")
    vix_t = "^VIX"; vix3m_t = "^VIX3M"

    return {
        # Raw FRED columns
        "hy_oas":       lambda: _col("hy_oas"),
        "ig_oas":       lambda: _col("ig_oas"),
        "real10":       lambda: r10,
        "real5":        lambda: r5,
        "y10":          lambda: y10,
        "y2":           lambda: y2,
        "y5":           lambda: y5,
        "y30":          lambda: y30,
        "dollar":       lambda: _col("dollar_broad"),
        "fed_assets":   lambda: _col("fed_assets"),
        "init_claims":  lambda: _col("init_claims"),
        "cont_claims":  lambda: _col("cont_claims"),
        "fed_funds":    lambda: _col("fed_funds"),
        "nfci":         lambda: _col("nfci"),
        "umich":        lambda: _col("umich"),
        "rrp":          lambda: _col("rrp"),
        "tga":          lambda: _col("tga"),

        # Simple derived (kept inline for speed)
        "curve_2s10s":  lambda: (y10 - y2).dropna(),
        "curve_3m10":   lambda: (y10 - y3m).dropna(),
        "cpi_yoy":      lambda: (_col("cpi").pct_change(12) * 100).dropna() if "cpi" in macro.columns else pd.Series(dtype=float),

        # Proxy passthroughs
        "spy":          lambda: px["SPY"].dropna() if "SPY" in px.columns else pd.Series(dtype=float),
        "gld":          lambda: px["GLD"].dropna() if "GLD" in px.columns else pd.Series(dtype=float),
    }


@router.get("/charts/{series}")
def chart_series(
    series: str,
    range: str = Query("1y", description="1m | 3m | 6m | 1y | 2y | 5y"),
):
    try:
        macro = get_macro()
        px    = get_prices()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    inline_map = _build_inline_map(macro, px)
    s = None

    # 1. Check inline map first (fastest)
    if series in inline_map:
        try:
            s = inline_map[series]()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 2. Fall back to derived.py registry (computed series)
    if s is None or (isinstance(s, pd.Series) and s.empty):
        try:
            from src.derived import DERIVED_SERIES
            if series in DERIVED_SERIES:
                s = DERIVED_SERIES[series](macro, px)
        except ImportError:
            pass
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Derived series error: {e}")

    # 3. Not found anywhere
    if s is None or (isinstance(s, pd.Series) and s.empty):
        # Build available list for error message
        available = sorted(set(inline_map.keys()))
        try:
            from src.derived import DERIVED_SERIES
            available = sorted(set(available) | set(DERIVED_SERIES.keys()))
        except ImportError:
            pass
        if s is not None and isinstance(s, pd.Series) and s.empty:
            return {"series": series, "range": range, "points": [], "meta": {}}
        raise HTTPException(
            status_code=404,
            detail=f"Unknown series '{series}'. Available: {available}"
        )

    sliced = _slice(s, range)

    last_val  = float(sliced.iloc[-1]) if not sliced.empty else None
    first_val = float(sliced.iloc[0])  if not sliced.empty else None
    change    = round(last_val - first_val, 4) if last_val is not None and first_val is not None else None

    return {
        "series":  series,
        "range":   range,
        "points":  _to_points(sliced),
        "meta": {
            "last":   round(last_val, 5) if last_val is not None else None,
            "change": change,
            "min":    round(float(sliced.min()), 5) if not sliced.empty else None,
            "max":    round(float(sliced.max()), 5) if not sliced.empty else None,
            "count":  len(sliced),
        }
    }


@router.get("/charts/price/{ticker}")
def chart_price(
    ticker: str,
    range: str = Query("1y", description="1m | 3m | 6m | 1y | 2y | 5y"),
):
    try:
        px = get_prices()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    t = ticker.upper()
    if t not in px.columns:
        raise HTTPException(status_code=404, detail=f"Ticker {t} not in price data")

    s      = px[t].dropna()
    sliced = _slice(s, range)
    last   = float(sliced.iloc[-1]) if not sliced.empty else None
    first  = float(sliced.iloc[0])  if not sliced.empty else None

    return {
        "ticker": t,
        "range":  range,
        "points": _to_points(sliced),
        "meta": {
            "last":       round(last, 4) if last else None,
            "change_pct": round((last / first - 1) * 100, 3) if last and first else None,
            "min":        round(float(sliced.min()), 4) if not sliced.empty else None,
            "max":        round(float(sliced.max()), 4) if not sliced.empty else None,
        }
    }