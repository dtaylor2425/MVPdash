"""
api/routers/charts.py
GET /api/charts/{series}  — time series data for a named signal
GET /api/charts/price/{ticker} — OHLC/close for an ETF

Supported series names:
  hy_oas, ig_oas, real10, curve_2s10s, curve_3m10,
  breakeven, dollar, fed_assets, init_claims, cont_claims,
  cpi_yoy, vratio, rsp_spy
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd
import numpy as np

from api.deps import get_macro, get_prices

router = APIRouter(tags=["Charts"])

# Lookback options in trading days
RANGES = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "2y": 504, "5y": 1260}


def _slice(s: pd.Series, rng: str) -> pd.Series:
    days = RANGES.get(rng, 252)
    return s.dropna().iloc[-days:] if len(s.dropna()) > days else s.dropna()


def _to_points(s: pd.Series) -> list:
    """Convert series to [{date, value}] list, dropping NaN."""
    return [
        {"date": str(idx.date()), "value": round(float(v), 5)}
        for idx, v in s.items()
        if not np.isnan(v)
    ]


@router.get("/charts/{series}")
def chart_series(
    series: str,
    range: str = Query("1y", description="1m | 3m | 6m | 1y | 2y | 5y"),
):
    """
    Returns time-series data for a named macro signal.
    Used by all chart pages on the frontend.
    """
    try:
        macro = get_macro()
        px    = get_prices()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Derived series ────────────────────────────────────────────────────────
    def _col(name): return macro[name].dropna() if name in macro.columns else pd.Series(dtype=float)

    y10  = _col("y10"); y2  = _col("y2"); y3m = _col("y3m")
    r10  = _col("real10")
    vix_t = "^VIX"; vix3m_t = "^VIX3M"

    series_map = {
        "hy_oas":       lambda: _col("hy_oas"),
        "ig_oas":       lambda: _col("ig_oas"),
        "real10":       lambda: r10,
        "curve_2s10s":  lambda: (y10 - y2).dropna(),
        "curve_3m10":   lambda: (y10 - y3m).dropna(),
        "breakeven":    lambda: (y10 - r10).dropna(),
        "dollar":       lambda: _col("dollar_broad"),
        "fed_assets":   lambda: _col("fed_assets"),
        "init_claims":  lambda: _col("init_claims"),
        "cont_claims":  lambda: _col("cont_claims"),
        "cpi_yoy":      lambda: (_col("cpi").pct_change(12) * 100).dropna() if "cpi" in macro.columns else pd.Series(dtype=float),
        "rsp_spy":      lambda: (px["RSP"] / px["SPY"].reindex(px["RSP"].index, method="ffill")).dropna()
                                if "RSP" in px.columns and "SPY" in px.columns else pd.Series(dtype=float),
        "vratio":       lambda: (px[vix_t] / px[vix3m_t].reindex(px[vix_t].index, method="ffill")).dropna()
                                if vix_t in px.columns and vix3m_t in px.columns else pd.Series(dtype=float),
        "vix":          lambda: px[vix_t].dropna() if vix_t in px.columns else pd.Series(dtype=float),
        "spy":          lambda: px["SPY"].dropna() if "SPY" in px.columns else pd.Series(dtype=float),
        "gld":          lambda: px["GLD"].dropna() if "GLD" in px.columns else pd.Series(dtype=float),
        "tlt":          lambda: px["TLT"].dropna() if "TLT" in px.columns else pd.Series(dtype=float),
        "hyg":          lambda: px["HYG"].dropna() if "HYG" in px.columns else pd.Series(dtype=float),
        "y10":          lambda: y10,
        "y2":           lambda: y2,
    }

    if series not in series_map:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown series '{series}'. Available: {sorted(series_map.keys())}"
        )

    try:
        s = series_map[series]()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if s.empty:
        return {"series": series, "range": range, "points": [], "meta": {}}

    sliced = _slice(s, range)

    # Compute basic stats for the response
    last_val  = float(sliced.iloc[-1]) if not sliced.empty else None
    first_val = float(sliced.iloc[0])  if not sliced.empty else None
    change    = round(last_val - first_val, 4) if last_val and first_val else None

    return {
        "series":  series,
        "range":   range,
        "points":  _to_points(sliced),
        "meta": {
            "last":   round(last_val, 5) if last_val else None,
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
    """Returns close price series for a given ETF ticker."""
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