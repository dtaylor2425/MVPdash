"""
api/routers/assets.py
GET /api/assets        — all asset alignment scores
GET /api/assets/{ticker} — single asset detail
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from api.deps import get_macro, get_prices, get_regime

router = APIRouter(tags=["Assets"])

WATCH = {
    "SPY":  {"name": "S&P 500",      "color": "#1d4ed8", "drivers": ["regime", "credit", "real_yields"]},
    "QQQ":  {"name": "Nasdaq 100",   "color": "#7c3aed", "drivers": ["real_yields", "growth"]},
    "GLD":  {"name": "Gold",         "color": "#d97706", "drivers": ["real_yields", "dollar"]},
    "BTC":  {"name": "Bitcoin",      "color": "#f59e0b", "drivers": ["dollar", "liquidity", "risk"]},
    "SLV":  {"name": "Silver",       "color": "#94a3b8", "drivers": ["real_yields", "dollar", "growth"]},
    "TLT":  {"name": "Long Bond",    "color": "#0ea5e9", "drivers": ["real_yields", "curve", "regime"]},
    "HYG":  {"name": "High Yield",   "color": "#a855f7", "drivers": ["credit", "regime"]},
    "XLU":  {"name": "Utilities",    "color": "#0d9488", "drivers": ["real_yields", "curve"]},
    "XLC":  {"name": "Comms/Media",  "color": "#c026d3", "drivers": ["regime", "real_yields"]},
}

SAFE_HAVENS = {"TLT", "GLD", "SLV", "XLU"}


def _last(s): return float(s.dropna().iloc[-1]) if not s.dropna().empty else None
def _zscore(s, w=252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w, len(s)):]
    sd = float(tail.std())
    return round(float((tail.iloc[-1] - tail.mean()) / sd), 3) if sd else 0.0
def _ret(s, days):
    s = s.dropna()
    if len(s) < days + 1: return None
    return round(float(s.iloc[-1] / s.iloc[-days-1] - 1), 5)
def _ma(s, w): return s.dropna().rolling(w, min_periods=w//2).mean()
def _safe(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    return v


def _score_color(score: int) -> str:
    if score >= 65: return "#16a34a"
    if score >= 55: return "#22c55e"
    if score >= 45: return "#6b7280"
    if score >= 33: return "#d97706"
    return "#ef4444"


def _score_label(score: int) -> str:
    if score >= 65: return "BULLISH"
    if score >= 55: return "MILD TAILWIND"
    if score >= 45: return "NEUTRAL"
    if score >= 33: return "MILD HEADWIND"
    return "BEARISH"


def _compute_alignment(ticker: str, regime, macro: pd.DataFrame, px: pd.DataFrame) -> dict:
    """Compute macro alignment score 0-100 for a single asset."""
    score = 50
    signals = []

    r10 = _last(macro["real10"].dropna()) if "real10" in macro.columns else None
    r10z = _zscore(macro["real10"]) if "real10" in macro.columns else None
    hyz  = _zscore(macro["hy_oas"])  if "hy_oas"  in macro.columns else None
    dz   = _zscore(macro["dollar_broad"]) if "dollar_broad" in macro.columns else None
    cur_score = regime.score
    is_safe = ticker in SAFE_HAVENS
    regime_dir = -1 if is_safe else +1
    regime_sig = regime_dir * (1 if cur_score > 55 else -1 if cur_score < 45 else 0)

    if regime_sig != 0:
        lbl = ("bullish" if cur_score > 55 else "bearish") + " macro backdrop"
        signals.append({"name": "Regime", "direction": regime_sig, "text": f"Score {cur_score} — {lbl}"})

    if hyz is not None:
        credit_dir = +1 if is_safe else -1
        if abs(hyz) > 0.5:
            d = credit_dir * (1 if hyz > 0 else -1)
            signals.append({"name": "Credit", "direction": d,
                            "text": f"HY OAS z {hyz:+.2f} — {'widening headwind' if hyz > 0 else 'tightening tailwind'}"})

    if r10 is not None:
        real_dir = +1 if is_safe else -1
        if r10 > 1.5:
            signals.append({"name": "Real yields", "direction": real_dir,
                            "text": f"{r10:.2f}% — restrictive"})
        elif r10 < 0.5:
            signals.append({"name": "Real yields", "direction": -real_dir,
                            "text": f"{r10:.2f}% — accommodative"})

    if ticker in px.columns:
        s = px[ticker].dropna()
        ma200 = _ma(s, 200).dropna()
        if not ma200.empty:
            above = float(s.iloc[-1]) > float(ma200.iloc[-1])
            signals.append({"name": "200d MA", "direction": +1 if above else -1,
                            "text": ("Above" if above else "Below") + " 200d MA"})

    n = len(signals)
    pos = sum(1 for s in signals if s["direction"] > 0)
    neg = sum(1 for s in signals if s["direction"] < 0)
    raw = (pos - neg) / n if n > 0 else 0
    align = int(max(15, min(85, 50 + raw * 35)))

    return {
        "score":     align,
        "color":     _score_color(align),
        "label":     _score_label(align),
        "signals":   signals,
        "n_confirm": pos,
        "n_against": neg,
    }


@router.get("/assets")
def assets_all():
    """Returns alignment scores for all tracked assets."""
    try:
        macro  = get_macro()
        px     = get_prices()
        regime = get_regime()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = {}
    for ticker, meta in WATCH.items():
        price = _last(px[ticker].dropna()) if ticker in px.columns else None
        ret1w = _ret(px[ticker], 7)  if ticker in px.columns else None
        ret1m = _ret(px[ticker], 30) if ticker in px.columns else None
        align = _compute_alignment(ticker, regime, macro, px)

        result[ticker] = {
            "ticker":    ticker,
            "name":      meta["name"],
            "color":     meta["color"],
            "price":     _safe(price),
            "ret_1w":    _safe(ret1w),
            "ret_1m":    _safe(ret1m),
            "alignment": align,
        }

    return {"assets": result}


@router.get("/assets/{ticker}")
def asset_detail(ticker: str):
    """Returns detailed alignment data for a single asset."""
    ticker = ticker.upper()
    if ticker not in WATCH:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not tracked")
    try:
        macro  = get_macro()
        px     = get_prices()
        regime = get_regime()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    meta  = WATCH[ticker]
    s     = px[ticker].dropna() if ticker in px.columns else pd.Series(dtype=float)
    price = _last(s)
    ma200 = _ma(s, 200).dropna()
    ma200_pct = ((float(s.iloc[-1]) / float(ma200.iloc[-1])) - 1) * 100 if not ma200.empty and not s.empty else None

    align = _compute_alignment(ticker, regime, macro, px)

    return {
        "ticker":    ticker,
        "name":      meta["name"],
        "color":     meta["color"],
        "price":     _safe(price),
        "ret_1w":    _safe(_ret(s, 7)),
        "ret_1m":    _safe(_ret(s, 30)),
        "ret_3m":    _safe(_ret(s, 63)),
        "ma200_pct": _safe(round(ma200_pct, 2)) if ma200_pct is not None else None,
        "alignment": align,
        "regime": {
            "score": regime.score,
            "label": regime.label,
            "delta": regime.score_delta,
        },
    }