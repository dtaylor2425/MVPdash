# pages/2_Macro_Charts.py
"""
Credit & Macro
══════════════════════════════════════════════════════════════════════════════
Four tabs, each with a KPI strip + interpretation banner + Plotly charts
with dynamic y-axes, regime shading, and "now" annotations.

Tabs:
  1. Credit          — HY OAS percentile bands, HY vs equity, oil/gold, copper/gold
  2. Risk appetite   — IWM/SPY breadth, RSP/SPY equal weight, HY vs equity
  (Curve context → Curve View page)
  (Rates → Fed & Liquidity page)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.ranges import RANGES, slice_series, slice_df
from src.ui import inject_css, sidebar_nav, safe_switch_page

st.set_page_config(page_title="Credit & Macro", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Credit & Macro")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

RKEYS = list(RANGES.keys())

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_data():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    extra = ["IWM","SPY","RSP","GLD","USO","CPER","HYG","TLT","QQQ"]
    vix_t = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
    px    = fetch_prices(list(dict.fromkeys(extra + vix_t)), period="5y")
    return macro, (px if px is not None and not px.empty else pd.DataFrame())

macro, px = load_data()

_fred_daily   = macro["y10"].dropna() if "y10" in macro.columns else macro.dropna(how="all").index.to_series()
last_date     = _fred_daily.index.max() if not _fred_daily.empty else macro.index.max()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(name):
    return macro[name].dropna() if name in macro.columns \
           else pd.Series(dtype=float, name=name)

def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _delta(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] - s.loc[prev[-1]]) if len(prev) else None

def _zscore(s, w=252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w, len(s)):]
    sd   = float(tail.std())
    return float((tail.iloc[-1] - tail.mean()) / sd) if sd > 0 else 0.0

def _pct_rank(s, w=252):
    s = s.dropna()
    if len(s) < w: return None
    return float((s.iloc[-w:] < s.iloc[-1]).mean() * 100)

def cpi_yoy():
    c = _col("cpi")
    return (c.pct_change(12) * 100).dropna() if len(c) >= 13 else None

def fmt(x, nd=2, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return (f"{float(x):+.{nd}f}{suffix}" if plus else f"{float(x):.{nd}f}{suffix}")

def _dyn_range(vals_list, pad=0.12):
    vals = [v for arr in vals_list for v in (arr if hasattr(arr,"__iter__") else [arr])
            if v is not None and not np.isnan(v)]
    if not vals: return None, None
    lo = min(vals); hi = max(vals)
    p  = max((hi - lo) * pad, abs(hi) * 0.02, 0.01)
    return lo - p, hi + p

def _fig(height=280):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=10,r=20,t=28,b=16),
                      plot_bgcolor="white", paper_bgcolor="white",
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0, font_size=11))
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", gridwidth=1)
    return fig

def _kpi(col, label, value, sub, color="#0f172a", bg="#f8fafc"):
    col.markdown(
        f"<div style='padding:11px 13px;border-radius:12px;background:{bg};"
        f"border:1px solid rgba(0,0,0,0.07);'>"
        f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;'>{label}</div>"
        f"<div style='font-size:18px;font-weight:900;color:{color};line-height:1.1;'>{value}</div>"
        f"<div style='font-size:10px;color:rgba(0,0,0,0.48);margin-top:2px;'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)

def _banner(text, color, bg):
    st.markdown(
        f"<div style='padding:11px 16px;border-radius:11px;background:{bg};"
        f"border-left:4px solid {color};font-size:12px;color:rgba(0,0,0,0.80);"
        f"line-height:1.6;margin-bottom:14px;'>{text}</div>",
        unsafe_allow_html=True)

def _regime_shade(fig, zones, row=None, col_n=1):
    """Add horizontal regime bands. zones = [(y0,y1,color), ...]"""
    kw = dict(row=row, col=col_n) if row else {}
    for y0, y1, fc in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fc, line_width=0, **kw)

def _now_line(fig, val, color, label, row=None):
    kw = dict(row=row, col=1) if row else {}
    if val is None: return
    fig.add_hline(y=val, line_color=color, line_width=1.5, line_dash="dot",
                  annotation_text=label, annotation_position="right",
                  annotation_font_color=color, annotation_font_size=9, **kw)

# ── Derived series ────────────────────────────────────────────────────────────

y10      = _col("y10");   y2  = _col("y2");  y3m = _col("y3m")
real10   = _col("real10"); ff  = _col("fed_funds")
hy_oas   = _col("hy_oas"); dollar = _col("dollar_broad")
fed_assets = _col("fed_assets")
cpi      = cpi_yoy()

curve_210  = (y10 - y2).dropna()  if len(y10) > 1 and len(y2)  > 1 else pd.Series(dtype=float)
curve_3m10 = (y10 - y3m).dropna() if len(y10) > 1 and len(y3m) > 1 else pd.Series(dtype=float)
breakeven  = (y10 - real10).dropna() if len(y10)>1 and len(real10)>1 else pd.Series(dtype=float)

iwm = px["IWM"].dropna() if "IWM" in px.columns else pd.Series(dtype=float)
spy = px["SPY"].dropna() if "SPY" in px.columns else pd.Series(dtype=float)
rsp = px["RSP"].dropna() if "RSP" in px.columns else pd.Series(dtype=float)
hyg = px["HYG"].dropna() if "HYG" in px.columns else pd.Series(dtype=float)
gld = px["GLD"].dropna() if "GLD" in px.columns else pd.Series(dtype=float)
uso = px["USO"].dropna() if "USO" in px.columns else pd.Series(dtype=float)
cper= px["CPER"].dropna() if "CPER" in px.columns else pd.Series(dtype=float)

# Ratios
iwm_spy = (iwm / spy.reindex(iwm.index, method="ffill")).dropna() \
          if not iwm.empty and not spy.empty else pd.Series(dtype=float)
rsp_spy = (rsp / spy.reindex(rsp.index, method="ffill")).dropna() \
          if not rsp.empty and not spy.empty else pd.Series(dtype=float)
oil_gld = (uso / gld.reindex(uso.index, method="ffill")).dropna() \
          if not uso.empty and not gld.empty else pd.Series(dtype=float)
cop_gld = (cper/ gld.reindex(cper.index,method="ffill")).dropna() \
          if not cper.empty and not gld.empty else pd.Series(dtype=float)

# ── Live reads ────────────────────────────────────────────────────────────────

c210_now    = _last(curve_210)
c3m10_now   = _last(curve_3m10)
real_now    = _last(real10)
hy_now      = _last(hy_oas)
hy_7d       = _delta(hy_oas, 7)
hy_z        = _zscore(hy_oas)
hy_pct      = _pct_rank(hy_oas)
dollar_now  = _last(dollar)
dollar_1m   = _delta(dollar, 30)
ff_now      = _last(ff)
cpi_now     = _last(cpi) if cpi is not None else None
be_now      = _last(breakeven)
iwm_spy_z   = _zscore(iwm_spy)
rsp_spy_z   = _zscore(rsp_spy)
c210_1m     = _delta(curve_210, 30)

# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Credit &amp; Macro</div>
              <div class="me-subtle">
                Curve context &nbsp;·&nbsp; rates &nbsp;·&nbsp;
                risk appetite &nbsp;·&nbsp; credit
                &nbsp;·&nbsp; FRED through {last_date.date() if pd.notna(last_date) else 'unknown'}
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["💳 Credit", "🔥 Risk appetite"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — CURVE CONTEXT
# ══════════════════════════════════════════════════════════════════════════════


# Curve context → dedicated page

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)