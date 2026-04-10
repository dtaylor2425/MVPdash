"""
api/routers/regime.py
GET /api/regime         -- current regime score + pillars
GET /api/regime/history -- weekly regime score time series
"""

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from api.deps import get_regime, get_macro, get_prices
from src.regime import compute_regime_timeseries

router = APIRouter(tags=["Regime"])


def _pillar_color(score):
    if score >= 75:
        return "#1f7a4f"
    if score >= 60:
        return "#16a34a"
    if score >= 40:
        return "#6b7280"
    if score >= 25:
        return "#d97706"
    return "#ef4444"


def _regime_color(label):
    colors = {
        "Risk On": "#1f7a4f",
        "Bullish": "#16a34a",
        "Neutral": "#6b7280",
        "Bearish": "#d97706",
        "Risk Off": "#ef4444",
    }
    return colors.get(label, "#6b7280")


@router.get("/regime")
def regime_current():
    try:
        r = get_regime()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    pillars = []
    pillar_keys = [
        "growth_momentum",
        "inflation_price",
        "monetary_policy",
        "market_internals",
        "fiscal_external",
        "sentiment",
    ]
    for key in pillar_keys:
        comp = r.components.get(key, {})
        weight = float(comp.get("weight", 0.1))
        if weight == 0:
            weight = 0.1
        contrib = float(comp.get("contribution", 0))
        score = int(round(max(0, min(100, (contrib / weight + 1) * 50))))
        pillars.append({
            "key": key,
            "name": comp.get("name", key.replace("_", " ").title()),
            "score": score,
            "color": _pillar_color(score),
            "zscore": comp.get("zscore"),
            "weight": comp.get("weight", 0),
        })

    return {
        "score": r.score,
        "score_delta": r.score_delta,
        "label": r.label,
        "color": _regime_color(r.label),
        "confidence": r.confidence,
        "momentum": r.momentum_label,
        "summary": r.summary,
        "updated": str(pd.Timestamp.now().date()),
        "pillars": pillars,
        "favored_groups": r.favored_groups,
        "allocation": r.allocation,
    }


@router.get("/regime/history")
def regime_history(
    freq: str = Query("W-FRI", description="Resampling frequency"),
):
    """Weekly regime score history for the regime history chart."""
    try:
        macro = get_macro()
        px = get_prices()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if macro.empty or px.empty:
        return {"points": [], "meta": {}}

    try:
        ts = compute_regime_timeseries(
            macro=macro,
            proxies=px,
            lookback_trend=63,
            freq=freq,
            min_points=60,
            z_window=252,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Regime history error: {}".format(str(e)),
        )

    if ts.empty:
        return {"points": [], "meta": {}}

    points = []
    for idx, row in ts.iterrows():
        points.append({
            "date": str(idx.date()),
            "score": int(row["score"]),
            "label": row.get("label", ""),
        })

    return {
        "points": points,
        "meta": {
            "count": len(points),
            "latest": points[-1] if points else None,
            "min": min(p["score"] for p in points) if points else None,
            "max": max(p["score"] for p in points) if points else None,
        },
    }