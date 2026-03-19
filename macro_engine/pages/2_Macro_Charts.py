# pages/2_Macro_Charts.py
"""
Credit & Macro
══════════════════════════════════════════════════════════════════════════════
Four tabs, each with a KPI strip + interpretation banner + Plotly charts
with dynamic y-axes, regime shading, and "now" annotations.

Tabs:
  1. Curve context   — curve vs Fed/CPI, curve vs credit, regime classification
  2. Rates           — Treasury stack, real yields, dollar, decomposition
  3. Risk appetite   — IWM/SPY breadth, RSP/SPY equal weight, HY vs equity
  4. Credit          — HY OAS, IG spreads proxy, oil/gold, copper/gold
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
    if st.button("← Home", use_container_width=True):
        safe_switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["💳 Credit", "📐 Curve context", "🔥 Risk appetite", "📈 Rates"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — CURVE CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:

    # KPI strip
    k1,k2,k3,k4,k5,k6 = st.columns(6, gap="small")
    c_col = "#1f7a4f" if (c210_now and c210_now >= 0.75) else \
            ("#d97706" if (c210_now and c210_now >= 0) else \
            ("#b42318" if c210_now is not None else "#6b7280"))
    c_bg  = "#dcfce7" if (c210_now and c210_now >= 0.75) else \
            ("#fef9c3" if (c210_now and c210_now >= 0) else \
            ("#fee2e2" if c210_now is not None else "#f3f4f6"))
    _kpi(k1, "2s10s spread",  fmt(c210_now,  suffix="pp"), "Curve level", c_col, c_bg)
    _kpi(k2, "3m10y spread",  fmt(c3m10_now, suffix="pp"), "Fed preferred", "#1d4ed8","#eff6ff")
    _kpi(k3, "10y yield",     fmt(_last(y10), suffix="%"), "Nominal", "#0f172a","#f8fafc")
    _kpi(k4, "Real yield",    fmt(real_now,   suffix="%"), "Restrictive >1.5%",
         "#b42318" if (real_now and real_now>1.5) else "#d97706" if (real_now and real_now>0.5) else "#1f7a4f",
         "#fee2e2" if (real_now and real_now>1.5) else "#fef9c3" if (real_now and real_now>0.5) else "#dcfce7")
    _kpi(k5, "10y breakeven", fmt(be_now, suffix="%"), "Market inflation exp", "#7c3aed","#f5f3ff")
    _kpi(k6, "1m curve move", fmt(c210_1m, suffix="pp", plus=True),
         "Bear flat <0 · Bear steep >0",
         "#b42318" if (c210_1m and c210_1m < -0.05) else "#d97706" if (c210_1m and c210_1m > 0.05) else "#6b7280")

    st.markdown("")

    # Interpretation banner
    if c210_now is not None:
        if c210_now >= 0.75:
            _banner("Steep curve at +{:.2f}pp. Classic early-expansion signal. Credit historically tightens, cyclicals outperform, real assets bid.".format(c210_now), "#1f7a4f","#dcfce7")
        elif c210_now >= 0:
            regime_txt = "Flat to mildly positive at {:+.2f}pp.".format(c210_now)
            if c210_1m and c210_1m < -0.05:
                regime_txt += " Bear flattening in progress ({:+.2f}pp 1m). Short rates rising faster than long. Late-cycle tightening pattern.".format(c210_1m)
            elif c210_1m and c210_1m > 0.05:
                regime_txt += " Steepening in progress ({:+.2f}pp 1m). Watch whether this is bull or bear steepener.".format(c210_1m)
            _banner(regime_txt, "#6b7280","#f3f4f6")
        elif c210_now >= -0.25:
            _banner("Shallow inversion at {:+.2f}pp. Late-cycle signal. Historically precedes recession by 12-18 months on average. Watch 3m10y — the Fed's preferred measure at {}.".format(c210_now, fmt(c3m10_now,"pp")), "#d97706","#fef9c3")
        else:
            _banner("Deep inversion at {:+.2f}pp. Recession risk elevated. The uninversion (when it comes) historically marks the onset of the recession, not the all-clear.".format(c210_now), "#b42318","#fee2e2")

    rng = st.selectbox("Range", RKEYS, index=RKEYS.index("2y") if "2y" in RKEYS else RKEYS.index("1y"),
                       key="curve_ctx_range")

    ca, cb = st.columns(2, gap="large")

    # Chart 1: Curve vs Fed Funds vs CPI
    with ca:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Curve vs Fed funds vs CPI</div>", unsafe_allow_html=True)
            st.caption("When the curve inverts while fed funds is rising and CPI is hot, that is a full late-cycle tightening signal. Steepening while Fed holds = early recovery.")
            c_sl = slice_series(curve_210, rng)
            if not c_sl.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=c_sl.index, y=c_sl.values,
                    mode="lines", name="2s10s curve", line=dict(color="#1d4ed8",width=2.2),
                    fill="tozeroy", fillcolor="rgba(29,78,216,0.05)"), secondary_y=False)
                ff_sl = slice_series(ff, rng)
                if not ff_sl.empty:
                    fig.add_trace(go.Scatter(x=ff_sl.index, y=ff_sl.values,
                        mode="lines", name="Fed funds", line=dict(color="#dc2626",width=1.6,dash="dot")),
                        secondary_y=True)
                if cpi is not None:
                    cpi_sl = slice_series(cpi, rng)
                    if not cpi_sl.empty:
                        fig.add_trace(go.Scatter(x=cpi_sl.index, y=cpi_sl.values,
                            mode="lines", name="CPI YoY", line=dict(color="#059669",width=1.4,dash="dash")),
                            secondary_y=True)
                fig.add_hline(y=0, line_color="#94a3b8", line_width=1.5, secondary_y=False)
                lo_c, hi_c = _dyn_range([c_sl.values])
                # Build dynamic right-axis range from actual fed funds + CPI values
                right_vals = []
                if not ff_sl.empty:  right_vals += list(ff_sl.dropna().values)
                if cpi is not None and not cpi_sl.empty: right_vals += list(cpi_sl.dropna().values)
                lo_r2, hi_r2 = _dyn_range([right_vals]) if right_vals else (None, None)
                fig.update_layout(height=300, margin=dict(l=10,r=20,t=10,b=10),
                                  plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                                  legend=dict(orientation="h",y=1.04,x=0,font_size=10))
                fig.update_yaxes(range=[lo_c,hi_c], title_text="Spread (pp)", secondary_y=False,
                                 showgrid=True, gridcolor="#f1f5f9")
                if lo_r2 is not None:
                    fig.update_yaxes(range=[lo_r2,hi_r2], title_text="Rate / CPI %",
                                     secondary_y=True, showgrid=False)
                else:
                    fig.update_yaxes(title_text="Rate / CPI %", secondary_y=True, showgrid=False)
                fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Curve vs HY OAS
    with cb:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Curve vs credit spreads (HY OAS)</div>", unsafe_allow_html=True)
            st.caption("Inversion + tight spreads = late-cycle complacency. Inversion + wide spreads = active stress. Curve uninverting while spreads widen = the danger zone.")
            c_sl2 = slice_series(curve_210, rng)
            hy_sl = slice_series(hy_oas, rng)
            if not c_sl2.empty and not hy_sl.empty:
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=c_sl2.index, y=c_sl2.values,
                    mode="lines", name="2s10s curve", line=dict(color="#1d4ed8",width=2.2),
                    fill="tozeroy", fillcolor="rgba(29,78,216,0.05)"), secondary_y=False)
                fig2.add_trace(go.Scatter(x=hy_sl.index, y=hy_sl.values,
                    mode="lines", name="HY OAS", line=dict(color="#dc2626",width=1.6,dash="dot")),
                    secondary_y=True)
                fig2.add_hline(y=0, line_color="#94a3b8", line_width=1.5, secondary_y=False)
                lo_c2, hi_c2 = _dyn_range([c_sl2.values])
                lo_h, hi_h   = _dyn_range([hy_sl.values])
                fig2.update_layout(height=280, margin=dict(l=10,r=20,t=10,b=10),
                                   plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                                   legend=dict(orientation="h",y=1.04,x=0,font_size=10))
                fig2.update_yaxes(range=[lo_c2,hi_c2], title_text="Spread (pp)", secondary_y=False,
                                  showgrid=True, gridcolor="#f1f5f9")
                fig2.update_yaxes(range=[lo_h,hi_h], title_text="HY OAS (%)", secondary_y=True, showgrid=False)
                fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                if hy_now:
                    hy_c = "#b42318" if hy_z and hy_z > 1 else "#d97706" if hy_z and hy_z > 0 else "#1f7a4f"
                    fig2.add_hline(y=hy_now, line_color=hy_c, line_width=1.2, line_dash="dot",
                                   annotation_text=f"HY now: {hy_now:.2f}%",
                                   annotation_position="right", annotation_font_size=9,
                                   secondary_y=True)
                st.plotly_chart(fig2, use_container_width=True)

    # Term structure snapshot table
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Term structure snapshot</div>", unsafe_allow_html=True)
        snap_cols = st.columns(7, gap="small")
        labels = [("3m yield","y3m"),("2y yield","y2"),("10y yield","y10"),
                  ("10y real","real10"),("Breakeven","be"),("Fed funds","fed_funds"),("2s10s","c210")]
        vals   = {
            "y3m": _last(y3m), "y2": _last(y2), "y10": _last(y10),
            "real10": real_now, "be": be_now, "fed_funds": ff_now, "c210": c210_now,
        }
        colors = {
            "y3m":"#0f172a","y2":"#0f172a","y10":"#1d4ed8",
            "real10": "#b42318" if (real_now and real_now>1.5) else "#d97706" if (real_now and real_now>0.5) else "#1f7a4f",
            "be":"#7c3aed","fed_funds":"#dc2626",
            "c210": c_col,
        }
        for i,(lbl,k) in enumerate(labels):
            v = vals[k]
            snap_cols[i].markdown(
                f"<div style='text-align:center;padding:8px 4px;border-radius:8px;"
                f"background:#f8fafc;border:1px solid rgba(0,0,0,0.06);'>"
                f"<div style='font-size:9px;color:rgba(0,0,0,0.40);text-transform:uppercase;"
                f"letter-spacing:0.4px;margin-bottom:3px;'>{lbl}</div>"
                f"<div style='font-size:16px;font-weight:900;color:{colors[k]};'>"
                f"{'—' if v is None else f'{v:.2f}%' if k != 'c210' else f'{v:+.2f}pp'}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("")
    if st.button("Full curve analysis →", use_container_width=False, key="btn_curve_full"):
        safe_switch_page("pages/9_Curve_View.py")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RATES
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:

    y10_now  = _last(y10); y2_now = _last(y2); y3m_now = _last(y3m)
    r10_now  = real_now; spread_10_2 = c210_now

    # KPI strip
    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    _kpi(k1, "10y yield",   fmt(y10_now, suffix="%"),  "Nominal", "#1d4ed8","#eff6ff")
    _kpi(k2, "2y yield",    fmt(y2_now,  suffix="%"),  "Policy-sensitive", "#0f172a","#f8fafc")
    _kpi(k3, "3m yield",    fmt(y3m_now, suffix="%"),  "T-bill rate", "#0f172a","#f8fafc")
    _kpi(k4, "Real 10y",    fmt(r10_now, suffix="%"),  "Restrictive >1.5%",
         "#b42318" if (r10_now and r10_now>1.5) else "#d97706" if (r10_now and r10_now>0.5) else "#1f7a4f",
         "#fee2e2" if (r10_now and r10_now>1.5) else "#fef9c3" if (r10_now and r10_now>0.5) else "#dcfce7")
    _kpi(k5, "Dollar broad", fmt(dollar_now, nd=1),
         f"1m {fmt(dollar_1m, nd=1, plus=True)}", "#1d4ed8","#eff6ff")

    st.markdown("")

    rng_r = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="rates_range")

    ra, rb = st.columns(2, gap="large")

    # Treasury yield stack
    with ra:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Treasury yield stack</div>", unsafe_allow_html=True)
            st.caption("3m, 2y, and 10y together. Parallel shifts = Fed expectations. Spread compression = flattening / inversion.")
            fig_y = _fig(280)
            colors_y = {"y3m":"#94a3b8","y2":"#3b82f6","y10":"#1d4ed8"}
            names_y  = {"y3m":"3m","y2":"2y","y10":"10y"}
            all_y_vals = []
            for col_k, color in colors_y.items():
                s = slice_series(_col(col_k), rng_r)
                if not s.empty:
                    fig_y.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                        name=names_y[col_k], line=dict(color=color,width=2)))
                    all_y_vals += list(s.values)
            lo_y, hi_y = _dyn_range([all_y_vals])
            fig_y.update_yaxes(range=[lo_y,hi_y], title_text="Yield (%)")
            st.plotly_chart(fig_y, use_container_width=True)

    # 10y decomposition: nominal = real + breakeven
    with rb:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>10y decomposition: nominal = real + breakeven</div>", unsafe_allow_html=True)
            st.caption("Rising 10y driven by real yields = tighter financial conditions. Rising 10y driven by breakeven = inflation expectations. Different implications for equities.")
            y10_sl = slice_series(y10, rng_r)
            r10_sl = slice_series(real10, rng_r)
            be_sl  = slice_series(breakeven, rng_r)
            if not y10_sl.empty:
                fig_d = _fig(280)
                fig_d.add_trace(go.Scatter(x=y10_sl.index, y=y10_sl.values,
                    mode="lines", name="10y nominal", line=dict(color="#1d4ed8",width=2.2)))
                if not r10_sl.empty:
                    fig_d.add_trace(go.Scatter(x=r10_sl.index, y=r10_sl.values,
                        mode="lines", name="10y real (TIPS)", line=dict(color="#dc2626",width=1.8,dash="dot")))
                if not be_sl.empty:
                    fig_d.add_trace(go.Scatter(x=be_sl.index, y=be_sl.values,
                        mode="lines", name="Breakeven", line=dict(color="#d97706",width=1.6,dash="dash")))
                # Regime shading for real yield
                fig_d.add_hrect(y0=1.5, y1=6,   fillcolor="rgba(180,35,24,0.06)", line_width=0)
                fig_d.add_hrect(y0=0,   y1=1.5,  fillcolor="rgba(217,119,6,0.04)",  line_width=0)
                fig_d.add_hrect(y0=-5,  y1=0,    fillcolor="rgba(31,122,79,0.05)",   line_width=0)
                all_d = []
                for s in [y10_sl, r10_sl, be_sl]:
                    if not s.empty: all_d += list(s.values)
                lo_d, hi_d = _dyn_range([all_d])
                fig_d.update_yaxes(range=[lo_d, hi_d], title_text="Yield (%)")
                st.plotly_chart(fig_d, use_container_width=True)

    rc, rd = st.columns(2, gap="large")

    # Real yield path
    with rc:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Real rate path (10y TIPS)</div>", unsafe_allow_html=True)
            st.caption("The single most important price for asset valuation. Above 1.5% = restrictive. Below 0% = historically bullish for GLD and equities.")
            r_sl = slice_series(real10, rng_r)
            if not r_sl.empty:
                fig_r = _fig(240)
                for y0,y1,fc in [(-5,0,"rgba(31,122,79,0.07)"),(0,0.5,"rgba(107,114,128,0.04)"),
                                   (0.5,1.5,"rgba(217,119,6,0.05)"),(1.5,6,"rgba(180,35,24,0.07)")]:
                    fig_r.add_hrect(y0=y0, y1=y1, fillcolor=fc, line_width=0)
                fig_r.add_trace(go.Scatter(x=r_sl.index, y=r_sl.values,
                    mode="lines", name="10y real", line=dict(color="#dc2626",width=2.2),
                    fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
                lo_r, hi_r = _dyn_range([r_sl.values])
                for lvl, lbl in [(0,"Zero"),(0.5,"Neutral"),(1.5,"Restrictive")]:
                    if lo_r <= lvl <= hi_r:
                        fig_r.add_hline(y=lvl, line_dash="dash", line_color="#cbd5e1",
                                        line_width=1, annotation_text=lbl,
                                        annotation_position="right", annotation_font_size=9)
                if real_now:
                    rc2 = "#b42318" if real_now>1.5 else "#d97706" if real_now>0.5 else "#1f7a4f"
                    fig_r.add_hline(y=real_now, line_color=rc2, line_width=1.5, line_dash="dot",
                                    annotation_text=f"Now: {real_now:.2f}%",
                                    annotation_position="right", annotation_font_color=rc2,
                                    annotation_font_size=9)
                fig_r.update_yaxes(range=[lo_r, hi_r], title_text="Real yield (%)")
                st.plotly_chart(fig_r, use_container_width=True)

    # Dollar cycle
    with rd:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Dollar cycle</div>", unsafe_allow_html=True)
            st.caption("Strong dollar tightens global financial conditions. Dollar weakening is a tailwind for EM, commodities, and GLD. 63d MA shows trend.")
            d_sl = slice_series(dollar, rng_r)
            if not d_sl.empty:
                ma63  = d_sl.rolling(63, min_periods=10).mean()
                fig_dl = _fig(240)
                fig_dl.add_trace(go.Scatter(x=d_sl.index, y=d_sl.values,
                    mode="lines", name="Dollar broad", line=dict(color="#1d4ed8",width=2.2)))
                if not ma63.dropna().empty:
                    fig_dl.add_trace(go.Scatter(x=ma63.index, y=ma63.values,
                        mode="lines", name="63d MA", line=dict(color="#94a3b8",width=1.4,dash="dash")))
                lo_dl, hi_dl = _dyn_range([d_sl.values])
                fig_dl.update_yaxes(range=[lo_dl, hi_dl], title_text="Index")
                st.plotly_chart(fig_dl, use_container_width=True)

    st.markdown("")
    if st.button("Fed & Liquidity deep dive →", use_container_width=False, key="btn_fed"):
        safe_switch_page("pages/10_Fed_Liquidity.py")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK APPETITE
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:

    # KPI strip
    k1,k2,k3,k4 = st.columns(4, gap="small")
    iwm_spy_pct = _pct_rank(iwm_spy)
    rsp_spy_pct = _pct_rank(rsp_spy)
    _kpi(k1, "IWM/SPY z-score", fmt(iwm_spy_z),
         "Small cap vs large cap breadth",
         "#1f7a4f" if (iwm_spy_z and iwm_spy_z>0.5) else "#b42318" if (iwm_spy_z and iwm_spy_z<-0.5) else "#6b7280")
    _kpi(k2, "IWM/SPY percentile", fmt(iwm_spy_pct, nd=0, suffix="th"),
         "vs 1y history",
         "#1f7a4f" if (iwm_spy_pct and iwm_spy_pct>60) else "#b42318" if (iwm_spy_pct and iwm_spy_pct<40) else "#6b7280")
    _kpi(k3, "RSP/SPY z-score", fmt(rsp_spy_z),
         "Equal weight breadth",
         "#1f7a4f" if (rsp_spy_z and rsp_spy_z>0.5) else "#b42318" if (rsp_spy_z and rsp_spy_z<-0.5) else "#6b7280")
    _kpi(k4, "HY OAS", fmt(hy_now, suffix="%"),
         f"7d {fmt(hy_7d, suffix='pp', plus=True)} · z {fmt(hy_z)}",
         "#b42318" if (hy_z and hy_z>1) else "#d97706" if (hy_z and hy_z>0) else "#1f7a4f",
         "#fee2e2" if (hy_z and hy_z>1) else "#fef9c3" if (hy_z and hy_z>0) else "#dcfce7")

    st.markdown("")

    # Interpretation
    if iwm_spy_z is not None:
        if iwm_spy_z > 0.5 and (hy_z is None or hy_z < 0):
            _banner("Small caps leading large caps + credit tight. Classic healthy risk-on breadth. Both legs of the risk trade are working.", "#1f7a4f","#dcfce7")
        elif iwm_spy_z < -0.5 and hy_z and hy_z > 0.5:
            _banner("Small caps lagging + credit spreads elevated. Narrow breadth with stress signals. Markets being led by a shrinking group of large caps. High-risk environment.", "#b42318","#fee2e2")
        elif iwm_spy_z < -0.5:
            _banner("Small caps underperforming. Breadth narrowing. Not yet a credit stress signal but watch HY OAS for confirmation.", "#d97706","#fef9c3")
        else:
            _banner("Breadth signals mixed. No strong directional read from IWM/SPY or RSP/SPY. Monitor for breakout in either direction.", "#6b7280","#f3f4f6")

    rng_ra = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="risk_range")

    ra1, ra2 = st.columns(2, gap="large")

    def _ratio_fig(ratio_s, name, color, height=260, add_zscore=True):
        fig = _fig(height)
        sl  = slice_series(ratio_s, rng_ra)
        if sl.empty: return fig, False
        ma63 = sl.rolling(63, min_periods=10).mean()
        fig.add_trace(go.Scatter(x=sl.index, y=sl.values, mode="lines",
            name=name, line=dict(color=color,width=2.2),
            fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)"))
        if not ma63.dropna().empty:
            fig.add_trace(go.Scatter(x=ma63.index, y=ma63.values, mode="lines",
                name="63d MA", line=dict(color="#94a3b8",width=1.4,dash="dash")))
        lo, hi = _dyn_range([sl.values])
        fig.update_yaxes(range=[lo,hi], title_text="Ratio")
        return fig, True

    with ra1:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>IWM / SPY — small cap breadth</div>", unsafe_allow_html=True)
            st.caption("Rising = small caps leading = broad risk-on. Falling = large-cap defensive rotation or liquidity stress.")
            fig_is, ok = _ratio_fig(iwm_spy, "IWM/SPY", "#ec4899")
            if ok:
                # Add z-score annotation
                if iwm_spy_z is not None:
                    iz_c = "#1f7a4f" if iwm_spy_z>0.5 else "#b42318" if iwm_spy_z<-0.5 else "#6b7280"
                    st.markdown(f"<div style='font-size:11px;font-weight:700;color:{iz_c};"
                                f"margin-bottom:4px;'>Z-score: {iwm_spy_z:+.2f} · "
                                f"{'Above average' if iwm_spy_z>0 else 'Below average'}</div>",
                                unsafe_allow_html=True)
                st.plotly_chart(fig_is, use_container_width=True)

    with ra2:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>RSP / SPY — equal weight breadth</div>", unsafe_allow_html=True)
            st.caption("Equal-weight outperforming cap-weight = broad participation. Underperforming = market concentration in a handful of mega-caps.")
            fig_rs, ok2 = _ratio_fig(rsp_spy, "RSP/SPY", "#8b5cf6")
            if ok2:
                if rsp_spy_z is not None:
                    rz_c = "#1f7a4f" if rsp_spy_z>0.5 else "#b42318" if rsp_spy_z<-0.5 else "#6b7280"
                    st.markdown(f"<div style='font-size:11px;font-weight:700;color:{rz_c};"
                                f"margin-bottom:4px;'>Z-score: {rsp_spy_z:+.2f} · "
                                f"{'Broad participation' if rsp_spy_z>0 else 'Concentrated market'}</div>",
                                unsafe_allow_html=True)
                st.plotly_chart(fig_rs, use_container_width=True)

    # HY OAS as the credit confirmation of risk appetite
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>HY OAS vs SPY — credit confirming equity</div>", unsafe_allow_html=True)
        st.caption("Credit spreads lead equities by 4-8 weeks. Spreads widening while equities hold = warning. Spreads tightening while equities sell off = buy signal.")
        hy_sl  = slice_series(hy_oas, rng_ra)
        spy_sl = slice_series(spy, rng_ra) if not spy.empty else pd.Series(dtype=float)
        if not hy_sl.empty and not spy_sl.empty:
            fig_hs = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hs.add_trace(go.Scatter(x=spy_sl.index, y=spy_sl.values,
                mode="lines", name="SPY", line=dict(color="#1d4ed8",width=2)),
                secondary_y=False)
            fig_hs.add_trace(go.Scatter(x=hy_sl.index, y=hy_sl.values,
                mode="lines", name="HY OAS", line=dict(color="#dc2626",width=1.6,dash="dot")),
                secondary_y=True)
            lo_s, hi_s = _dyn_range([spy_sl.values])
            lo_h, hi_h = _dyn_range([hy_sl.values])
            fig_hs.update_layout(height=240, margin=dict(l=10,r=20,t=10,b=10),
                                 plot_bgcolor="white", paper_bgcolor="white",
                                 hovermode="x unified",
                                 legend=dict(orientation="h",y=1.04,x=0,font_size=10))
            fig_hs.update_yaxes(range=[lo_s,hi_s], title_text="SPY price", secondary_y=False,
                                showgrid=True, gridcolor="#f1f5f9")
            fig_hs.update_yaxes(range=[lo_h,hi_h], title_text="HY OAS (%)", secondary_y=True, showgrid=False)
            fig_hs.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_hs, use_container_width=True)

    st.markdown("")
    c_btn1, c_btn2 = st.columns(2, gap="medium")
    with c_btn1:
        if st.button("Volatility view →", use_container_width=True, key="btn_vol"):
            safe_switch_page("pages/6_Volatility_View.py")
    with c_btn2:
        if st.button("Regime Playbook →", use_container_width=True, key="btn_play"):
            safe_switch_page("pages/7_Regime_Playbook.py")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CREDIT
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:

    # KPI strip
    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    hy_c = "#b42318" if (hy_z and hy_z>1) else "#d97706" if (hy_z and hy_z>0.3) else "#1f7a4f"
    hy_b = "#fee2e2" if (hy_z and hy_z>1) else "#fef9c3" if (hy_z and hy_z>0.3) else "#dcfce7"
    _kpi(k1, "HY OAS", fmt(hy_now, suffix="%"), f"7d {fmt(hy_7d,suffix='pp',plus=True)}", hy_c, hy_b)
    _kpi(k2, "HY z-score", fmt(hy_z), "vs 1y hist", hy_c, hy_b)
    _kpi(k3, "HY percentile", fmt(hy_pct, nd=0, suffix="th"), "vs 1y",
         "#b42318" if (hy_pct and hy_pct>70) else "#1f7a4f" if (hy_pct and hy_pct<30) else "#6b7280")
    oil_gld_z = _zscore(oil_gld) if not oil_gld.empty else None
    cop_gld_z = _zscore(cop_gld) if not cop_gld.empty else None
    _kpi(k4, "Oil/Gold z", fmt(oil_gld_z), "Inflation impulse proxy",
         "#d97706" if (oil_gld_z and oil_gld_z>0.5) else "#6b7280")
    _kpi(k5, "Copper/Gold z", fmt(cop_gld_z), "Growth vs safety proxy",
         "#1f7a4f" if (cop_gld_z and cop_gld_z>0.5) else "#b42318" if (cop_gld_z and cop_gld_z<-0.5) else "#6b7280")

    st.markdown("")

    # Interpretation
    if hy_z is not None:
        if hy_z > 1.5:
            _banner(f"HY OAS at {hy_now:.2f}% ({hy_z:+.2f}z). Stress conditions. Credit spreads at this level historically precede equity drawdowns by 4-8 weeks. Reduce high-beta exposure.", "#b42318","#fee2e2")
        elif hy_z > 0.3:
            _banner(f"HY OAS widening ({hy_now:.2f}%, {hy_z:+.2f}z). Early-warning zone. Not crisis level but directionally bearish for risk assets.", "#d97706","#fef9c3")
        else:
            _banner(f"HY OAS contained at {hy_now:.2f}% ({hy_z:+.2f}z). Credit market not pricing stress. Supportive backdrop for equities and risk assets.", "#1f7a4f","#dcfce7")

    rng_cr = st.selectbox("Range", RKEYS, index=RKEYS.index("2y") if "2y" in RKEYS else RKEYS.index("1y"), key="credit_range")

    cr1, cr2 = st.columns(2, gap="large")

    # HY OAS with percentile bands
    with cr1:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>HY OAS with historical bands</div>", unsafe_allow_html=True)
            st.caption("Green band = historically tight (bottom 25%). Red band = historically wide (top 25%). Current level vs history shows how stressed credit markets are.")
            hy_sl2 = slice_series(hy_oas, rng_cr)
            if not hy_sl2.empty:
                # Historical bands from full series
                p25 = float(hy_oas.quantile(0.25)) if not hy_oas.empty else None
                p75 = float(hy_oas.quantile(0.75)) if not hy_oas.empty else None
                med = float(hy_oas.median())        if not hy_oas.empty else None
                fig_hy = _fig(260)
                if p25 and p75:
                    fig_hy.add_hrect(y0=0,   y1=p25, fillcolor="rgba(31,122,79,0.07)", line_width=0)
                    fig_hy.add_hrect(y0=p75, y1=15,  fillcolor="rgba(180,35,24,0.07)", line_width=0)
                fig_hy.add_trace(go.Scatter(x=hy_sl2.index, y=hy_sl2.values,
                    mode="lines", name="HY OAS", line=dict(color="#dc2626",width=2.2),
                    fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
                lo_hy, hi_hy = _dyn_range([hy_sl2.values])
                for lvl, lbl, c in [(p25,"25th pct","#1f7a4f"),(med,"Median","#94a3b8"),(p75,"75th pct","#b42318")]:
                    if lvl and lo_hy <= lvl <= hi_hy:
                        fig_hy.add_hline(y=lvl, line_dash="dash", line_color=c,
                                         line_width=1, annotation_text=lbl,
                                         annotation_position="right", annotation_font_size=9)
                if hy_now:
                    fig_hy.add_hline(y=hy_now, line_color=hy_c, line_width=1.8, line_dash="dot",
                                     annotation_text=f"Now: {hy_now:.2f}%",
                                     annotation_position="right", annotation_font_color=hy_c,
                                     annotation_font_size=9)
                fig_hy.update_yaxes(range=[lo_hy, hi_hy], title_text="OAS (%)")
                st.plotly_chart(fig_hy, use_container_width=True)

    # Oil/Gold + Copper/Gold side by side
    with cr2:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Inflation impulse proxies</div>", unsafe_allow_html=True)
            st.caption("Oil/Gold = energy-driven inflation expectations. Copper/Gold = growth vs safety trade. Both rising = reflation. Both falling = deflationary risk.")
            og_sl  = slice_series(oil_gld, rng_cr)
            cg_sl  = slice_series(cop_gld, rng_cr)
            if not og_sl.empty or not cg_sl.empty:
                fig_inf = _fig(260)
                if not og_sl.empty:
                    og_norm = og_sl / og_sl.iloc[0] * 100
                    fig_inf.add_trace(go.Scatter(x=og_norm.index, y=og_norm.values,
                        mode="lines", name="Oil/Gold (indexed)", line=dict(color="#f97316",width=2)))
                if not cg_sl.empty:
                    cg_norm = cg_sl / cg_sl.iloc[0] * 100
                    fig_inf.add_trace(go.Scatter(x=cg_norm.index, y=cg_norm.values,
                        mode="lines", name="Copper/Gold (indexed)", line=dict(color="#8b5cf6",width=2)))
                fig_inf.add_hline(y=100, line_color="#94a3b8", line_width=1, line_dash="dash",
                                   annotation_text="Base", annotation_position="right", annotation_font_size=9)
                all_inf = []
                for s in [og_sl, cg_sl]:
                    if not s.empty: all_inf += list(s / s.iloc[0] * 100)
                lo_inf, hi_inf = _dyn_range([all_inf])
                fig_inf.update_yaxes(range=[lo_inf, hi_inf], title_text="Indexed (base=100)")
                st.plotly_chart(fig_inf, use_container_width=True)

    # HY OAS vs curve: the combined stress matrix
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Credit + curve combined stress read</div>", unsafe_allow_html=True)
        st.caption("Four quadrants: Steep curve + tight spreads = early expansion. Flat/inverted + wide spreads = recession. The transition between quadrants is where the risk is.")
        hy_sl3   = slice_series(hy_oas,   rng_cr)
        c210_sl3 = slice_series(curve_210, rng_cr)
        if not hy_sl3.empty and not c210_sl3.empty:
            fig_cc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_cc.add_trace(go.Scatter(x=c210_sl3.index, y=c210_sl3.values,
                mode="lines", name="2s10s curve", line=dict(color="#1d4ed8",width=2)),
                secondary_y=False)
            fig_cc.add_trace(go.Scatter(x=hy_sl3.index, y=hy_sl3.values,
                mode="lines", name="HY OAS", line=dict(color="#dc2626",width=1.8,dash="dot")),
                secondary_y=True)
            fig_cc.add_hline(y=0, line_color="#94a3b8", line_width=1.5, secondary_y=False)
            lo_cc, hi_cc = _dyn_range([c210_sl3.values])
            lo_hh, hi_hh = _dyn_range([hy_sl3.values])
            fig_cc.update_layout(height=220, margin=dict(l=10,r=20,t=10,b=10),
                                 plot_bgcolor="white", paper_bgcolor="white",
                                 hovermode="x unified",
                                 legend=dict(orientation="h",y=1.04,x=0,font_size=10))
            fig_cc.update_yaxes(range=[lo_cc,hi_cc], title_text="Curve (pp)", secondary_y=False,
                                showgrid=True, gridcolor="#f1f5f9")
            fig_cc.update_yaxes(range=[lo_hh,hi_hh], title_text="HY OAS (%)", secondary_y=True, showgrid=False)
            fig_cc.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_cc, use_container_width=True)

    st.markdown("")
    c_btn3, c_btn4 = st.columns(2, gap="medium")
    with c_btn3:
        if st.button("Curve View →", use_container_width=True, key="btn_cv"):
            safe_switch_page("pages/9_Curve_View.py")
    with c_btn4:
        if st.button("Transition Watch →", use_container_width=True, key="btn_tw"):
            safe_switch_page("pages/8_Transition_Watch.py")

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)