"""
api/routers/signals.py
GET /api/signals  — live FRED + price signal values for the ticker tape
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from api.deps import get_macro, get_prices

router = APIRouter(tags=["Signals"])


def _last(s: pd.Series):
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _zscore(s: pd.Series, w: int = 252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w, len(s)):]
    sd = float(tail.std())
    if sd == 0: return 0.0
    return round(float((tail.iloc[-1] - tail.mean()) / sd), 3)

def _delta(s: pd.Series, days: int):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    if not len(prev): return None
    return round(float(s.iloc[-1] - s.loc[prev[-1]]), 4)

def _pct_rank(s: pd.Series, w: int = 252):
    s = s.dropna()
    if len(s) < w: return None
    return round(float((s.iloc[-w:] < s.iloc[-1]).mean() * 100), 1)

def _col(macro, name):
    return macro[name].dropna() if name in macro.columns else pd.Series(dtype=float)

def _safe(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    return v


@router.get("/signals")
def signals_endpoint():
    """
    Returns all live macro signal values.
    Used by: ticker tape, signal cards, live badges.

    Response shape — flat dict of named signals, each with:
    { value, formatted, zscore, delta_7d, color, label, direction }
    """
    try:
        macro = get_macro()
        px    = get_prices()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data unavailable: {e}")

    def sig(value, formatted, zscore=None, delta=None,
            color="#94a3b8", label="", direction=0):
        return {
            "value":     _safe(value),
            "formatted": formatted,
            "zscore":    _safe(zscore),
            "delta":     _safe(delta),
            "color":     color,
            "label":     label,
            "direction": direction,   # +1 bullish, -1 bearish, 0 neutral
        }

    out = {}

    # ── HY OAS ───────────────────────────────────────────────────────────────
    hy = _col(macro, "hy_oas")
    hy_now = _last(hy); hy_z = _zscore(hy); hy_d7 = _delta(hy, 7)
    hy_color = "#ef4444" if (hy_z or 0) > 0.5 else "#22c55e" if (hy_z or 0) < -0.5 else "#94a3b8"
    out["hy_oas"] = sig(hy_now, f"{hy_now:.2f}%" if hy_now else "—",
                        zscore=hy_z, delta=hy_d7, color=hy_color,
                        label="HY OAS", direction=-1 if (hy_z or 0) > 0.5 else 1 if (hy_z or 0) < -0.5 else 0)

    # ── IG OAS ───────────────────────────────────────────────────────────────
    ig = _col(macro, "ig_oas")
    ig_now = _last(ig); ig_z = _zscore(ig)
    ig_color = "#ef4444" if (ig_z or 0) > 0.5 else "#22c55e" if (ig_z or 0) < -0.5 else "#94a3b8"
    out["ig_oas"] = sig(ig_now, f"{ig_now:.2f}%" if ig_now else "—",
                        zscore=ig_z, color=ig_color, label="IG OAS",
                        direction=-1 if (ig_z or 0) > 0.5 else 1 if (ig_z or 0) < -0.5 else 0)

    # ── Real yield 10y ───────────────────────────────────────────────────────
    r10 = _col(macro, "real10")
    r10_now = _last(r10); r10_z = _zscore(r10)
    r10_color = "#ef4444" if (r10_now or 0) > 1.5 else "#22c55e" if (r10_now or 0) < 0.5 else "#f59e0b"
    out["real_yield_10y"] = sig(r10_now, f"{r10_now:.2f}%" if r10_now else "—",
                                zscore=r10_z, color=r10_color, label="Real yield 10y",
                                direction=-1 if (r10_now or 0) > 1.5 else 1 if (r10_now or 0) < 0.5 else 0)

    # ── Curve 2s10s ──────────────────────────────────────────────────────────
    y10 = _col(macro, "y10"); y2 = _col(macro, "y2")
    if not y10.empty and not y2.empty:
        curve = (y10 - y2).dropna()
        c_now = _last(curve); c_d7 = _delta(curve, 7)
        c_color = "#22c55e" if (c_now or 0) > 0.5 else "#ef4444" if (c_now or 0) < 0 else "#f59e0b"
        out["curve_2s10s"] = sig(c_now, f"{c_now:+.2f}pp" if c_now is not None else "—",
                                  delta=c_d7, color=c_color, label="Curve 2s10s",
                                  direction=1 if (c_now or 0) > 0.5 else -1 if (c_now or 0) < 0 else 0)

    # ── Breakeven 10y ────────────────────────────────────────────────────────
    r10_s = _col(macro, "real10")
    if not y10.empty and not r10_s.empty:
        be = (y10 - r10_s).dropna()
        be_now = _last(be)
        be_color = "#f59e0b" if (be_now or 2) > 2.5 else "#22c55e" if (be_now or 2) < 2.0 else "#94a3b8"
        out["breakeven_10y"] = sig(be_now, f"{be_now:.2f}%" if be_now else "—",
                                    color=be_color, label="Breakeven 10y", direction=0)

    # ── Initial claims ───────────────────────────────────────────────────────
    cl = _col(macro, "init_claims")
    cl_now = _last(cl); cl_z = _zscore(cl, w=min(252, len(cl))) if not cl.empty else None
    cl_4w = _delta(cl, 28)
    cl_color = "#ef4444" if (cl_z or 0) > 0.5 else "#22c55e" if (cl_z or 0) < -0.5 else "#94a3b8"
    out["init_claims"] = sig(cl_now,
                              f"{cl_now/1e3:.0f}k" if cl_now else "—",
                              zscore=cl_z, delta=cl_4w, color=cl_color,
                              label="Initial claims",
                              direction=-1 if (cl_z or 0) > 0.5 else 1 if (cl_z or 0) < -0.5 else 0)

    # ── Dollar ───────────────────────────────────────────────────────────────
    dol = _col(macro, "dollar_broad")
    dol_now = _last(dol); dol_z = _zscore(dol)
    dol_color = "#ef4444" if (dol_z or 0) > 0.5 else "#22c55e" if (dol_z or 0) < -0.5 else "#94a3b8"
    out["dollar"] = sig(dol_now, f"{dol_now:.1f}" if dol_now else "—",
                        zscore=dol_z, color=dol_color, label="Dollar (broad)",
                        direction=-1 if (dol_z or 0) > 0.5 else 1 if (dol_z or 0) < -0.5 else 0)

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix_t = "^VIX"
    if vix_t in px.columns:
        vix_s = px[vix_t].dropna()
        vix_now = _last(vix_s)
        vix_color = "#ef4444" if (vix_now or 0) > 25 else "#22c55e" if (vix_now or 0) < 15 else "#94a3b8"
        out["vix"] = sig(vix_now, f"{vix_now:.1f}" if vix_now else "—",
                         color=vix_color, label="VIX",
                         direction=-1 if (vix_now or 0) > 25 else 1 if (vix_now or 0) < 15 else 0)

        # V-Ratio VIX/VIX3M
        vix3m_t = "^VIX3M"
        if vix3m_t in px.columns:
            v3m = px[vix3m_t].dropna()
            idx = vix_s.index.intersection(v3m.index)
            if len(idx):
                vr = float(vix_s.loc[idx[-1]] / v3m.loc[idx[-1]])
                vr_color = "#ef4444" if vr > 1.0 else "#22c55e"
                out["vratio"] = sig(vr, f"{vr:.3f}", color=vr_color,
                                    label="V-Ratio", direction=-1 if vr > 1.0 else 1)

    # ── RSP/SPY breadth ──────────────────────────────────────────────────────
    if "RSP" in px.columns and "SPY" in px.columns:
        rsp_spy = (px["RSP"] / px["SPY"].reindex(px["RSP"].index, method="ffill")).dropna()
        bz = _zscore(rsp_spy)
        b_color = "#22c55e" if (bz or 0) > 0.3 else "#ef4444" if (bz or 0) < -0.3 else "#94a3b8"
        b_label = "IMPROVING" if (bz or 0) > 0.3 else "FADING" if (bz or 0) < -0.3 else "NEUTRAL"
        out["breadth_rsp_spy"] = sig(bz, b_label, zscore=bz, color=b_color,
                                      label="RSP/SPY breadth",
                                      direction=1 if (bz or 0) > 0.3 else -1 if (bz or 0) < -0.3 else 0)

    # ── HY pct rank ──────────────────────────────────────────────────────────
    hy_pr = _pct_rank(hy)
    if hy_pr is not None:
        pr_color = "#ef4444" if hy_pr > 80 else "#22c55e" if hy_pr < 20 else "#94a3b8"
        out["hy_pct_rank"] = sig(hy_pr, f"{hy_pr:.0f}th pct",
                                  color=pr_color, label="HY pct rank",
                                  direction=-1 if hy_pr > 80 else 1 if hy_pr < 20 else 0)

    return {"signals": out, "count": len(out)}