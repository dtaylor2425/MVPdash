"""
api/routers/regime.py
GET /api/regime  — current regime score, label, components, delta
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np

from api.deps import get_regime, get_macro, get_prices

router = APIRouter(tags=["Regime"])

# Pillar keys to expose (v5 real pillars only, not backward-compat aliases)
_V5_PILLARS = {
    "growth_momentum", "inflation_price", "monetary_policy",
    "market_internals", "fiscal_external", "sentiment",
}

# Human-readable display names
_PILLAR_NAMES = {
    "growth_momentum":  "Growth Momentum",
    "inflation_price":  "Inflation & Price",
    "monetary_policy":  "Monetary Policy",
    "market_internals": "Market Internals",
    "fiscal_external":  "Fiscal & External",
    "sentiment":        "Sentiment",
}

# Score → colour
def _score_color(score: int) -> str:
    if score >= 75: return "#1f7a4f"
    if score >= 60: return "#16a34a"
    if score >= 40: return "#6b7280"
    if score >= 25: return "#d97706"
    return "#b42318"

def _safe(v):
    """Convert numpy types and NaN to plain Python."""
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if hasattr(v, "item"): return v.item()
    return v


@router.get("/regime")
def regime_endpoint():
    """
    Returns the current macro regime score and all pillar scores.

    Response shape:
    {
      score:       int,          // 0-100
      label:       str,          // "Bearish" etc.
      color:       str,          // hex
      score_raw:   float,
      score_prev:  int | null,
      score_delta: int | null,
      momentum:    str,          // "Improving" | "Stable" | "Deteriorating"
      confidence:  str,          // "High" | "Medium" | "Low"
      summary:     str,
      pillars: [
        { key, name, score, contribution, weight, level, zscore, color }
      ],
      updated:     str           // ISO date string of last FRED observation
    }
    """
    try:
        regime = get_regime()
        macro  = get_macro()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data unavailable: {e}")

    components = getattr(regime, "components", {}) or {}

    # Build pillar list
    pillars = []
    for key in ["growth_momentum", "inflation_price", "monetary_policy",
                "market_internals", "fiscal_external", "sentiment"]:
        c = components.get(key, {})
        if not isinstance(c, dict):
            continue
        raw_signal = c.get("contribution", 0.0) or 0.0
        weight     = c.get("weight", 0.0) or 0.0
        # Derive 0-100 pillar score from contribution/weight
        if weight > 0:
            pillar_score = int(round(
                np.clip((_safe(raw_signal) / weight + 1.0) * 50.0, 0, 100)
            ))
        else:
            pillar_score = 50
        pillars.append({
            "key":          key,
            "name":         _PILLAR_NAMES.get(key, key),
            "score":        pillar_score,
            "contribution": round(_safe(c.get("contribution")) or 0.0, 4),
            "weight":       round(_safe(c.get("weight")) or 0.0, 3),
            "level":        _safe(c.get("level")),
            "zscore":       round(_safe(c.get("zscore")) or 0.0, 3),
            "color":        _score_color(pillar_score),
        })

    # Last data date
    try:
        updated = macro.index.max().date().isoformat()
    except Exception:
        updated = None

    return {
        "score":       regime.score,
        "label":       regime.label,
        "color":       _score_color(regime.score),
        "score_raw":   round(float(regime.score_raw), 2),
        "score_prev":  regime.score_prev,
        "score_delta": regime.score_delta,
        "momentum":    regime.momentum_label,
        "confidence":  regime.confidence,
        "summary":     regime.summary,
        "pillars":     pillars,
        "updated":     updated,
        "favored_groups": getattr(regime, "favored_groups", []),
    }