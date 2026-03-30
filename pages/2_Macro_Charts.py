# pages/2_Macro_Charts.py
"""
Credit & Macro
Three tabs:
  1. Growth Momentum  — jobless claims, RSP/SPY breadth, 10y-3m curve
  2. Credit           — HY OAS, IG OAS, HY vs SPY divergence
  3. Risk Appetite    — VIX term structure, oil/gold, copper/gold, breadth
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.ranges import RANGES, slice_series
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

_fred_daily = macro["y10"].dropna() if "y10" in macro.columns else macro.dropna(how="all").index.to_series()
last_date   = _fred_daily.index.max() if not _fred_daily.empty else macro.index.max()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(name):
    return macro[name].dropna() if name in macro.columns else pd.Series(dtype=float)

def _last(s):
    s = s.dropna(); return float(s.iloc[-1]) if not s.empty else None

def _delta(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] - s.loc[prev[-1]]) if len(prev) else None

def _zscore(s, w=252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w, len(s)):]
    sd = float(tail.std())
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

def _now_line(fig, val, color, label):
    if val is None: return
    fig.add_hline(y=val, line_color=color, line_width=1.5, line_dash="dot",
                  annotation_text=label, annotation_position="right",
                  annotation_font_color=color, annotation_font_size=9)

# ── Derived series ────────────────────────────────────────────────────────────

y10     = _col("y10"); y2 = _col("y2"); y3m = _col("y3m")
real10  = _col("real10")
hy_oas  = _col("hy_oas"); ig_oas = _col("ig_oas")
dollar  = _col("dollar_broad"); fed_assets = _col("fed_assets")
claims  = _col("init_claims"); cont_claims = _col("cont_claims")
cpi     = cpi_yoy()

curve_210  = (y10 - y2).dropna()  if len(y10)>1 and len(y2)>1  else pd.Series(dtype=float)
curve_3m10 = (y10 - y3m).dropna() if len(y10)>1 and len(y3m)>1 else pd.Series(dtype=float)
breakeven  = (y10 - real10).dropna() if len(y10)>1 and len(real10)>1 else pd.Series(dtype=float)

spy  = px["SPY"].dropna()  if "SPY"  in px.columns else pd.Series(dtype=float)
rsp  = px["RSP"].dropna()  if "RSP"  in px.columns else pd.Series(dtype=float)
iwm  = px["IWM"].dropna()  if "IWM"  in px.columns else pd.Series(dtype=float)
gld  = px["GLD"].dropna()  if "GLD"  in px.columns else pd.Series(dtype=float)
uso  = px["USO"].dropna()  if "USO"  in px.columns else pd.Series(dtype=float)
cper = px["CPER"].dropna() if "CPER" in px.columns else pd.Series(dtype=float)
hyg  = px["HYG"].dropna()  if "HYG"  in px.columns else pd.Series(dtype=float)
vix_t   = YF_PROXIES.get("vix",  "^VIX")
vix3m_t = YF_PROXIES.get("vix3m","^VIX3M")
vix_s  = px[vix_t].dropna()   if vix_t   in px.columns else pd.Series(dtype=float)
vix3m  = px[vix3m_t].dropna() if vix3m_t in px.columns else pd.Series(dtype=float)

rsp_spy = (rsp / spy.reindex(rsp.index, method="ffill")).dropna() \
          if not rsp.empty and not spy.empty else pd.Series(dtype=float)
iwm_spy = (iwm / spy.reindex(iwm.index, method="ffill")).dropna() \
          if not iwm.empty and not spy.empty else pd.Series(dtype=float)
oil_gld = (uso / gld.reindex(uso.index, method="ffill")).dropna() \
          if not uso.empty and not gld.empty else pd.Series(dtype=float)
cop_gld = (cper/ gld.reindex(cper.index,method="ffill")).dropna() \
          if not cper.empty and not gld.empty else pd.Series(dtype=float)

# V-Ratio
vratio_s = pd.Series(dtype=float)
if not vix_s.empty and not vix3m.empty:
    idx_v = vix_s.index.intersection(vix3m.index)
    if len(idx_v) > 0:
        vratio_s = (vix_s.loc[idx_v] / vix3m.loc[idx_v]).dropna()

# ── Live reads ────────────────────────────────────────────────────────────────

claims_now = _last(claims); claims_4w = _delta(claims, 28)
claims_z   = _zscore(claims, w=min(252, max(30, len(claims))))
cont_now   = _last(cont_claims)
rsp_spy_z  = _zscore(rsp_spy); iwm_spy_z = _zscore(iwm_spy)
c3m10_now  = _last(curve_3m10); c210_now = _last(curve_210)
hy_now     = _last(hy_oas); hy_7d = _delta(hy_oas, 7); hy_z = _zscore(hy_oas)
ig_now     = _last(ig_oas); ig_z = _zscore(ig_oas)
vix_now    = _last(vix_s)
vratio_now = _last(vratio_s)

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
                Growth momentum &nbsp;·&nbsp; credit spreads &nbsp;·&nbsp;
                risk appetite &nbsp;·&nbsp;
                FRED {last_date.date() if pd.notna(last_date) else ''}
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

tab_growth, tab_labour, tab_credit, tab_risk = st.tabs([
    "📈 Growth Momentum", "👷 Labour Market", "💳 Credit", "🔥 Risk Appetite"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GROWTH MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

with tab_growth:

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    cl_c  = "#b42318" if (claims_4w or 0) > 5000 else "#1f7a4f" if (claims_4w or 0) < -5000 else "#6b7280"
    cl_bg = "#fee2e2" if (claims_4w or 0) > 5000 else "#dcfce7" if (claims_4w or 0) < -5000 else "#f3f4f6"
    _kpi(k1, "Initial claims",
         f"{claims_now/1e3:.0f}k" if claims_now else "—",
         f"4w chg {claims_4w/1e3:+.0f}k" if claims_4w else "weekly ICSA",
         cl_c, cl_bg)

    cz_c = "#b42318" if (claims_z or 0) > 0.5 else "#1f7a4f" if (claims_z or 0) < -0.5 else "#6b7280"
    _kpi(k2, "Claims z-score",
         fmt(claims_z, plus=True) if claims_z is not None else "—",
         "rising z = labour weakening",
         cz_c, "#fee2e2" if (claims_z or 0) > 0.5 else "#f3f4f6")

    _kpi(k3, "Continuing claims",
         f"{cont_now/1e6:.2f}M" if cont_now else "—",
         "lags initial by ~4 weeks", "#6b7280")

    rs_c  = "#1f7a4f" if (rsp_spy_z or 0) > 0.3 else "#b42318" if (rsp_spy_z or 0) < -0.3 else "#6b7280"
    rs_bg = "#dcfce7" if (rsp_spy_z or 0) > 0.3 else "#fee2e2" if (rsp_spy_z or 0) < -0.3 else "#f3f4f6"
    _kpi(k4, "Breadth RSP/SPY",
         fmt(rsp_spy_z, plus=True) if rsp_spy_z is not None else "—",
         "equal-wt vs cap-wt z", rs_c, rs_bg)

    cg_c  = "#1f7a4f" if (c3m10_now or 0) > 0.5 else "#b42318" if (c3m10_now or 0) < 0 else "#d97706"
    cg_bg = "#fee2e2" if (c3m10_now or 0) < 0 else "#dcfce7" if (c3m10_now or 0) > 0.5 else "#fef9c3"
    _kpi(k5, "Curve 10y-3m",
         fmt(c3m10_now, nd=2, suffix="pp") if c3m10_now is not None else "—",
         "negative = recession signal",
         cg_c, cg_bg)

    st.markdown("")

    # Interpretation banner
    bad = sum([
        (claims_z or 0) > 0.5,
        (claims_4w or 0) > 10000,
        (rsp_spy_z or 0) < -0.5,
        (c3m10_now or 0) < 0,
    ])
    if bad >= 3:
        _banner("⚠️ Multiple growth signals deteriorating simultaneously: rising claims, "
                "fading breadth, and/or inverted 10y-3m curve. Historically these "
                "co-occur 6-12 months before NBER recession dates. These are leading — "
                "hard data will confirm later.", "#b42318", "#fee2e2")
    elif bad >= 2:
        _banner("Growth momentum is softening. Claims trending higher or breadth fading. "
                "Not yet recession-level but watch the 4-week claims trend — "
                "a sustained move above 250k is the key threshold.",
                "#d97706", "#fef9c3")
    else:
        _banner("Growth momentum is intact. Claims contained, breadth healthy (RSP/SPY), "
                "yield curve not signalling recession. Labour market supporting risk appetite.",
                "#1f7a4f", "#dcfce7")

    gr = st.selectbox("Range", RKEYS, index=RKEYS.index("2y") if "2y" in RKEYS else 2,
                      key="gr_range")
    gc1, gc2 = st.columns(2, gap="large")

    # Chart 1: Initial claims
    with gc1:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Initial jobless claims (ICSA)</div>",
                        unsafe_allow_html=True)
            st.caption("Weekly new filings. 250k = watch level · 300k = recession territory. "
                       "Fastest real-economy signal — updates weekly.")
            if not claims.empty:
                cl_sl = slice_series(claims, gr)
                fig_cl = _fig(260)
                fig_cl.add_hrect(y0=300000, y1=900000,
                                 fillcolor="rgba(180,35,24,0.05)", line_width=0)
                fig_cl.add_hrect(y0=250000, y1=300000,
                                 fillcolor="rgba(217,119,6,0.05)", line_width=0)
                fig_cl.add_trace(go.Scatter(
                    x=cl_sl.index, y=cl_sl.values, mode="lines",
                    name="Initial claims", fill="tozeroy",
                    fillcolor="rgba(29,78,216,0.07)",
                    line=dict(color="#1d4ed8", width=2)))
                # 4-week MA
                ma4_cl = cl_sl.rolling(4, min_periods=1).mean()
                fig_cl.add_trace(go.Scatter(
                    x=ma4_cl.index, y=ma4_cl.values, mode="lines",
                    name="4w MA", line=dict(color="#7c3aed", width=2, dash="dash")))
                for lvl, lbl, lc in [(250000,"250k watch","#d97706"),
                                     (300000,"300k recession","#b42318")]:
                    fig_cl.add_hline(y=lvl, line_color=lc, line_width=1.2,
                                     line_dash="dash",
                                     annotation_text=lbl, annotation_position="right",
                                     annotation_font_color=lc, annotation_font_size=9)
                if claims_now:
                    _now_line(fig_cl, claims_now, "#1d4ed8",
                              f"Now: {claims_now/1e3:.0f}k")
                lo = float(cl_sl.min())*0.94; hi = float(cl_sl.max())*1.08
                fig_cl.update_yaxes(range=[lo, hi], title_text="Claims")
                st.plotly_chart(fig_cl, width='stretch')
            else:
                st.info("Initial claims (ICSA) not yet in FRED cache. "
                        "Add 'init_claims': 'ICSA' to FRED_SERIES in config.py "
                        "and clear the cache to load.")

    # Chart 2: Continuing claims
    with gc2:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Continuing claims (CCSA) — 4w MA</div>",
                        unsafe_allow_html=True)
            st.caption("People still collecting unemployment. Lags initial claims by ~4-6 weeks. "
                       "Confirms whether layoffs are being absorbed by new hiring.")
            if not cont_claims.empty:
                cc_sl = slice_series(cont_claims, gr)
                ma4c  = cc_sl.rolling(4, min_periods=1).mean()
                fig_cc = _fig(260)
                fig_cc.add_trace(go.Scatter(
                    x=cc_sl.index, y=cc_sl.values, mode="lines",
                    name="Continuing", line=dict(color="#94a3b8", width=1.2)))
                fig_cc.add_trace(go.Scatter(
                    x=ma4c.index, y=ma4c.values, mode="lines",
                    name="4w MA", line=dict(color="#7c3aed", width=2.2)))
                if cont_now:
                    _now_line(fig_cc, cont_now, "#7c3aed",
                              f"Now: {cont_now/1e6:.2f}M")
                lo2 = float(cc_sl.min())*0.96; hi2 = float(cc_sl.max())*1.06
                fig_cc.update_yaxes(range=[lo2, hi2])
                st.plotly_chart(fig_cc, width='stretch')
            else:
                st.info("Continuing claims (CCSA) not in FRED cache yet. "
                        "Add 'cont_claims': 'CCSA' to FRED_SERIES in config.py.")

    gc3, gc4 = st.columns(2, gap="large")

    # Chart 3: RSP/SPY breadth
    with gc3:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Market breadth — RSP/SPY ratio</div>",
                        unsafe_allow_html=True)
            st.caption("Equal-weight ÷ cap-weight. Rising = broad participation. "
                       "Falling = rally narrowing to mega-caps. "
                       "More reliable growth signal than IWM/SPY (which conflates size with breadth).")
            if not rsp_spy.empty:
                rs_sl = slice_series(rsp_spy, gr)
                ma63  = rs_sl.rolling(63, min_periods=10).mean()
                fig_rs = _fig(260)
                fig_rs.add_trace(go.Scatter(
                    x=rs_sl.index, y=rs_sl.values, mode="lines",
                    name="RSP/SPY", line=dict(color="#1d4ed8", width=2.2)))
                if not ma63.dropna().empty:
                    fig_rs.add_trace(go.Scatter(
                        x=ma63.index, y=ma63.values, mode="lines",
                        name="63d MA", line=dict(color="#94a3b8", width=1.4, dash="dash")))
                lo3 = float(rs_sl.min())*0.98; hi3 = float(rs_sl.max())*1.02
                fig_rs.update_yaxes(range=[lo3, hi3])
                st.plotly_chart(fig_rs, width='stretch')
            else:
                st.caption("RSP or SPY data unavailable.")

    # Chart 4: 10y-3m curve as growth signal
    with gc4:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Yield curve 10y − 3m (growth signal)</div>",
                        unsafe_allow_html=True)
            st.caption("The Fed's preferred recession predictor (Estrella & Mishkin 1998). "
                       "Sustained inversion for 6+ months has preceded every NBER recession since 1968.")
            if not curve_3m10.empty:
                cg_sl = slice_series(curve_3m10, gr)
                fig_cg = _fig(260)
                fig_cg.add_hrect(y0=-5, y1=0,
                                 fillcolor="rgba(180,35,24,0.07)", line_width=0)
                fig_cg.add_hline(y=0, line_color="#b42318", line_width=1.2,
                                 line_dash="dash",
                                 annotation_text="Inversion threshold",
                                 annotation_position="right",
                                 annotation_font_color="#b42318", annotation_font_size=9)
                bar_colors = ["#1f7a4f" if v >= 0 else "#b42318"
                              for v in cg_sl.values]
                fig_cg.add_trace(go.Bar(
                    x=cg_sl.index, y=cg_sl.values,
                    marker_color=bar_colors, name="10y-3m", opacity=0.85))
                lo4 = float(cg_sl.min())*1.15
                hi4 = max(float(cg_sl.max())*1.1, 0.5)
                fig_cg.update_yaxes(range=[lo4, hi4], title_text="pp")
                st.plotly_chart(fig_cg, width='stretch')
            else:
                st.caption("Curve data unavailable.")

    st.caption("ICSA weekly initial claims · CCSA continuing claims · "
               "RSP/SPY equal-weight breadth · 10y-3m yield curve · via FRED + yfinance")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CREDIT
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LABOUR MARKET DASHBOARD
# Unique visual: stress gauge, claims heatmap, leading/lagging relationship
# ══════════════════════════════════════════════════════════════════════════════

with tab_labour:

    # ── Compute labour stress composite ──────────────────────────────────────
    # A 0-100 score built from: claims level, claims trend, continuing claims
    # trend, and claims z-score. Higher = more labour market stress.
    def _labour_stress():
        signals = []
        # Claims z-score (0.40 weight)
        if claims_z is not None:
            signals.append(("Claims z-score", float(np.clip((claims_z + 2.5) / 5.0, 0, 1)), 0.40))
        # Claims 4-week change direction (0.25 weight)
        if claims_4w is not None and claims_now:
            pct_chg = claims_4w / max(claims_now, 1)
            sig = float(np.clip(0.5 + pct_chg * 25, 0, 1))
            signals.append(("Claims 4w trend", sig, 0.25))
        # Continuing claims level vs 1y pct rank (0.20 weight)
        if not cont_claims.empty:
            pr = _pct_rank(cont_claims, w=min(252, len(cont_claims)))
            if pr is not None:
                signals.append(("Continuing claims rank", pr / 100.0, 0.20))
        # Claims absolute level vs thresholds (0.15 weight)
        if claims_now:
            lvl_sig = float(np.clip((claims_now - 180000) / (350000 - 180000), 0, 1))
            signals.append(("Claims level", lvl_sig, 0.15))
        if not signals:
            return 50.0, []
        w_sum = sum(w for _, _, w in signals)
        score = sum(v * w for _, v, w in signals) / w_sum * 100.0
        return float(np.clip(score, 0, 100)), signals

    stress_score, stress_signals = _labour_stress()
    stress_color = ("#b42318" if stress_score > 65 else
                    "#d97706" if stress_score > 45 else "#1f7a4f")
    stress_bg    = ("#fee2e2" if stress_score > 65 else
                    "#fef9c3" if stress_score > 45 else "#dcfce7")
    stress_label = ("HIGH STRESS" if stress_score > 65 else
                    "ELEVATED"   if stress_score > 45 else "CONTAINED")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    lk1, lk2, lk3, lk4, lk5 = st.columns(5, gap="small")
    _kpi(lk1, "Labour stress score",
         f"{stress_score:.0f}/100", stress_label, stress_color, stress_bg)
    cl_c2 = "#b42318" if (claims_z or 0) > 0.5 else "#1f7a4f" if (claims_z or 0) < -0.5 else "#6b7280"
    _kpi(lk2, "Initial claims",
         f"{claims_now/1e3:.0f}k" if claims_now else "—",
         f"z {fmt(claims_z, plus=True)}" if claims_z is not None else "weekly ICSA",
         cl_c2, "#fee2e2" if (claims_z or 0) > 0.5 else "#f3f4f6")
    _kpi(lk3, "4-week change",
         f"{claims_4w/1e3:+.0f}k" if claims_4w else "—",
         "rising = labour weakening",
         "#b42318" if (claims_4w or 0) > 5000 else "#1f7a4f" if (claims_4w or 0) < -5000 else "#6b7280")
    _kpi(lk4, "Continuing claims",
         f"{cont_now/1e6:.2f}M" if cont_now else "—",
         "lags initial ~4-6 weeks", "#6b7280")
    cont_pr = _pct_rank(cont_claims, w=min(252, len(cont_claims))) if not cont_claims.empty else None
    _kpi(lk5, "Cont. claims rank",
         f"{cont_pr:.0f}th pct" if cont_pr is not None else "—",
         "vs 1y history", "#b42318" if (cont_pr or 50) > 70 else "#6b7280")

    st.markdown("")

    # ── Banner ────────────────────────────────────────────────────────────────
    if stress_score > 65:
        _banner(f"⚠️ Labour stress score {stress_score:.0f}/100 — multiple signals deteriorating. "
                f"Initial claims at {claims_now/1e3:.0f}k (z {fmt(claims_z, plus=True)}), "
                "continuing claims elevated. Historically precedes NBER recession by 6-12 months. "
                "This is a leading indicator — hard data will confirm later.",
                "#b42318", "#fee2e2")
    elif stress_score > 45:
        _banner(f"Labour market softening — stress score {stress_score:.0f}/100. "
                "Claims trending higher but not yet at recessionary levels. "
                "Watch the 4-week trend: sustained move above 250k is the key threshold.",
                "#d97706", "#fef9c3")
    else:
        _banner(f"Labour market healthy — stress score {stress_score:.0f}/100. "
                f"Claims contained at {claims_now/1e3:.0f}k, continuing claims stable. "
                "No leading recession signal from labour data.",
                "#1f7a4f", "#dcfce7")

    lr = st.selectbox("Range", RKEYS, index=RKEYS.index("2y") if "2y" in RKEYS else 2,
                      key="lr_range")

    # ── Row 1: Stress gauge + Claims with recession shading ──────────────────
    lg1, lg2 = st.columns([1, 2], gap="large")

    with lg1:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Labour stress gauge</div>",
                        unsafe_allow_html=True)
            st.caption("Composite of claims level, trend, z-score, and continuing claims rank.")

            # SVG arc gauge
            s = stress_score
            # Map 0-100 to arc angle: 0 = -135deg (left), 100 = +135deg (right)
            angle = -135 + (s / 100.0) * 270.0
            rad   = np.radians(angle)
            cx, cy, r = 110, 105, 75
            nx = cx + r * np.cos(np.radians(angle - 90))
            ny = cy + r * np.sin(np.radians(angle - 90))

            def _arc_path(start_deg, end_deg, rx, cx2, cy2):
                pts = []
                for d in np.linspace(start_deg, end_deg, 40):
                    rad2 = np.radians(d - 90)
                    pts.append(f"{cx2 + rx*np.cos(rad2):.1f},{cy2 + rx*np.sin(rad2):.1f}")
                return "M " + " L ".join(pts)

            green_path  = _arc_path(-135, -45,  75, cx, cy)
            yellow_path = _arc_path(-45,   45,  75, cx, cy)
            red_path    = _arc_path(45,   135,  75, cx, cy)
            needle_path = _arc_path(-135, angle, 75, cx, cy)

            gauge_html = (
                "<div style='display:flex;flex-direction:column;align-items:center;"
                "padding:8px 0;'>"
                "<svg viewBox='0 0 220 130' width='100%' style='max-width:240px;'>"
                # Background arc zones
                f"<path d='{green_path}' fill='none' stroke='#dcfce7' stroke-width='16' stroke-linecap='round'/>"
                f"<path d='{yellow_path}' fill='none' stroke='#fef9c3' stroke-width='16' stroke-linecap='round'/>"
                f"<path d='{red_path}' fill='none' stroke='#fee2e2' stroke-width='16' stroke-linecap='round'/>"
                # Filled progress arc
                f"<path d='{needle_path}' fill='none' stroke='{stress_color}' stroke-width='14' stroke-linecap='round'/>"
                # Needle
                f"<line x1='{cx}' y1='{cy}' x2='{nx:.1f}' y2='{ny:.1f}' "
                f"stroke='{stress_color}' stroke-width='3' stroke-linecap='round'/>"
                f"<circle cx='{cx}' cy='{cy}' r='5' fill='{stress_color}'/>"
                # Score text
                f"<text x='{cx}' y='{cy+28}' text-anchor='middle' "
                f"font-size='22' font-weight='900' fill='{stress_color}' font-family='system-ui'>"
                f"{s:.0f}</text>"
                f"<text x='{cx}' y='{cy+42}' text-anchor='middle' "
                f"font-size='8' fill='#888' font-family='system-ui'>{stress_label}</text>"
                # Zone labels
                "<text x='30' y='118' text-anchor='middle' font-size='7' fill='#1f7a4f' font-family='system-ui'>LOW</text>"
                "<text x='110' y='128' text-anchor='middle' font-size='7' fill='#d97706' font-family='system-ui'>MID</text>"
                "<text x='190' y='118' text-anchor='middle' font-size='7' fill='#b42318' font-family='system-ui'>HIGH</text>"
                "</svg>"
                # Signal breakdown below gauge
                "<div style='width:100%;margin-top:6px;'>"
            )
            for sig_name, sig_val, sig_w in stress_signals:
                bar_w = int(sig_val * 100)
                sc2 = "#b42318" if sig_val > 0.65 else "#d97706" if sig_val > 0.45 else "#1f7a4f"
                gauge_html += (
                    f"<div style='margin-bottom:5px;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:9px;color:rgba(0,0,0,0.55);margin-bottom:2px;'>"
                    f"<span>{sig_name}</span><span style='color:{sc2};font-weight:700;'>"
                    f"{sig_val*100:.0f}</span></div>"
                    f"<div style='background:#f1f5f9;border-radius:3px;height:4px;'>"
                    f"<div style='width:{bar_w}%;height:100%;background:{sc2};"
                    f"border-radius:3px;'></div></div></div>"
                )
            gauge_html += "</div></div>"
            st.markdown(gauge_html, unsafe_allow_html=True)

    with lg2:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Initial claims — with recession context</div>",
                        unsafe_allow_html=True)
            st.caption("250k = watch · 300k = recessionary. "
                       "Shaded zones show historical claim levels during NBER recessions.")
            if not claims.empty:
                cl_sl = slice_series(claims, lr)
                ma4_cl = cl_sl.rolling(4, min_periods=1).mean()
                fig_lc = _fig(280)
                # Recession threshold zones
                fig_lc.add_hrect(y0=300000, y1=max(float(cl_sl.max())*1.1, 400000),
                                 fillcolor="rgba(180,35,24,0.06)", line_width=0,
                                 annotation_text="Recession zone",
                                 annotation_position="top left",
                                 annotation_font_color="#b42318", annotation_font_size=8)
                fig_lc.add_hrect(y0=250000, y1=300000,
                                 fillcolor="rgba(217,119,6,0.05)", line_width=0,
                                 annotation_text="Watch zone",
                                 annotation_position="top left",
                                 annotation_font_color="#d97706", annotation_font_size=8)
                # Claims area
                fig_lc.add_trace(go.Scatter(
                    x=cl_sl.index, y=cl_sl.values, mode="lines",
                    name="Initial claims", fill="tozeroy",
                    fillcolor="rgba(29,78,216,0.08)",
                    line=dict(color="#1d4ed8", width=2)))
                # 4w MA
                fig_lc.add_trace(go.Scatter(
                    x=ma4_cl.index, y=ma4_cl.values, mode="lines",
                    name="4w MA", line=dict(color="#7c3aed", width=2.2, dash="dash")))
                for lvl, lbl, lc in [(250000, "250k", "#d97706"),
                                     (300000, "300k", "#b42318")]:
                    fig_lc.add_hline(y=lvl, line_color=lc, line_width=1.2, line_dash="dot",
                                     annotation_text=lbl, annotation_position="right",
                                     annotation_font_color=lc, annotation_font_size=9)
                lo_lc = float(cl_sl.min()) * 0.93
                hi_lc = max(float(cl_sl.max()) * 1.10, 320000)
                fig_lc.update_yaxes(range=[lo_lc, hi_lc], title_text="Claims (weekly)")
                st.plotly_chart(fig_lc, width='stretch')
            else:
                st.info("Initial claims not in FRED cache yet.")

    # ── Row 2: Leading vs lagging relationship + claims heatmap ──────────────
    lg3, lg4 = st.columns(2, gap="large")

    with lg3:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Initial vs continuing claims — lead/lag</div>",
                        unsafe_allow_html=True)
            st.caption("Initial claims lead continuing by ~4-6 weeks. "
                       "When both rise together, layoffs are not being re-absorbed — "
                       "the most reliable recession confirmation.")
            if not claims.empty and not cont_claims.empty:
                cl_sl2  = slice_series(claims, lr)
                cc_sl2  = slice_series(cont_claims, lr)
                ma4_cl2 = cl_sl2.rolling(4, min_periods=1).mean()
                ma4_cc2 = cc_sl2.rolling(4, min_periods=1).mean()

                fig_ll = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.06,
                    subplot_titles=("Initial claims (4w MA)", "Continuing claims (4w MA)"))

                fig_ll.add_trace(go.Scatter(
                    x=ma4_cl2.index, y=ma4_cl2.values, mode="lines",
                    name="Initial (4w MA)", line=dict(color="#1d4ed8", width=2.2)),
                    row=1, col=1)
                fig_ll.add_trace(go.Scatter(
                    x=ma4_cc2.index, y=ma4_cc2.values, mode="lines",
                    name="Continuing (4w MA)", line=dict(color="#7c3aed", width=2.2)),
                    row=2, col=1)

                fig_ll.update_layout(
                    height=290, margin=dict(l=10, r=20, t=28, b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    hovermode="x unified", showlegend=False)
                fig_ll.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                fig_ll.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig_ll, width='stretch')
            else:
                st.info("Claims data unavailable.")

    with lg4:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Claims heatmap — 4-week average by month</div>",
                        unsafe_allow_html=True)
            st.caption("Each cell = average initial claims that month. "
                       "Darker red = higher stress. Reveals seasonal patterns "
                       "and multi-year deterioration trends at a glance.")
            if not claims.empty and len(claims) >= 52:
                # Build month × year pivot
                cl_df = claims.copy()
                if hasattr(cl_df, 'to_frame'):
                    cl_df = cl_df.to_frame(name="claims")
                else:
                    cl_df = pd.DataFrame({"claims": cl_df})
                cl_df.index = pd.to_datetime(cl_df.index)
                cl_df["year"]  = cl_df.index.year
                cl_df["month"] = cl_df.index.month
                pivot = cl_df.groupby(["year","month"])["claims"].mean().unstack(level=1)
                pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
                # Limit to last 5 years for readability
                pivot = pivot.iloc[-5:] if len(pivot) > 5 else pivot
                pivot_k = pivot / 1000.0  # convert to thousands

                fig_hm = go.Figure(go.Heatmap(
                    z=pivot_k.values,
                    x=list(pivot_k.columns),
                    y=[str(y) for y in pivot_k.index],
                    colorscale=[[0,"#dcfce7"],[0.4,"#fef9c3"],
                                [0.7,"#fee2e2"],[1.0,"#b42318"]],
                    text=[[f"{v:.0f}k" if not np.isnan(v) else ""
                           for v in row] for row in pivot_k.values],
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    hovertemplate="<b>%{y} %{x}</b><br>Claims: %{text}<extra></extra>",
                    showscale=True,
                    colorbar=dict(title="k", thickness=10, len=0.8),
                ))
                fig_hm.update_layout(
                    height=290, margin=dict(l=10, r=60, t=10, b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig_hm, width='stretch')
            else:
                st.info("Insufficient claims history for heatmap (need 1yr+).")

# ══════════════════════════════════════════════════════════════════════════════

with tab_credit:

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    hy_c  = "#b42318" if (hy_z or 0) > 0.5 else "#1f7a4f" if (hy_z or 0) < -0.5 else "#6b7280"
    hy_bg = "#fee2e2" if (hy_z or 0) > 0.5 else "#dcfce7" if (hy_z or 0) < -0.5 else "#f3f4f6"
    _kpi(k1, "HY OAS", fmt(hy_now, suffix="%") if hy_now else "—",
         f"7d {fmt(hy_7d, nd=2, suffix='pp', plus=True)}" if hy_7d else "",
         hy_c, hy_bg)
    _kpi(k2, "HY z-score", fmt(hy_z, plus=True) if hy_z is not None else "—",
         "+ve = spreads wide vs history", hy_c)
    ig_c = "#b42318" if (ig_z or 0) > 0.5 else "#1f7a4f" if (ig_z or 0) < -0.5 else "#6b7280"
    _kpi(k3, "IG OAS", fmt(ig_now, suffix="%") if ig_now else "—",
         "investment grade spreads",
         ig_c, "#fee2e2" if (ig_z or 0) > 0.5 else "#f3f4f6")
    _kpi(k4, "HY pct rank",
         f"{_pct_rank(hy_oas):.0f}th" if _pct_rank(hy_oas) is not None else "—",
         "vs 252d · 0=tightest",
         "#b42318" if (_pct_rank(hy_oas) or 50) > 70 else "#6b7280")
    _kpi(k5, "Breakeven 10y", fmt(_last(breakeven), suffix="%"),
         "market inflation expectation", "#d97706")

    st.markdown("")

    if hy_z is not None:
        if hy_z > 1.0:
            _banner(f"⚠️ HY OAS {hy_now:.2f}% — credit stress elevated (z {hy_z:+.2f}). "
                    "Spreads lead equity drawdowns by 4-8 weeks historically. "
                    "This is the most reliable macro warning signal in the engine. Reduce beta.",
                    "#b42318", "#fee2e2")
        elif hy_z > 0.3:
            _banner(f"HY OAS {hy_now:.2f}% — spreads elevated but not in stress territory (z {hy_z:+.2f}). "
                    "Watch for continuation. Credit is the leading signal.",
                    "#d97706", "#fef9c3")
        else:
            _banner(f"Credit supportive — HY OAS {hy_now:.2f}% (z {hy_z:+.2f}). "
                    "Tight spreads support equity multiples and risk appetite. No stress signal.",
                    "#1f7a4f", "#dcfce7")

    cr = st.selectbox("Range", RKEYS, index=RKEYS.index("2y") if "2y" in RKEYS else 2,
                      key="cr_range")
    cc1, cc2 = st.columns(2, gap="large")

    with cc1:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>HY OAS &amp; IG OAS</div>",
                        unsafe_allow_html=True)
            st.caption("IG moves before HY — watch for IG widening as the early warning. "
                       "Both widening together = meaningful credit stress.")
            hy_sl = slice_series(hy_oas, cr)
            if not hy_sl.empty:
                fig_hy = _fig(280)
                fig_hy.add_trace(go.Scatter(
                    x=hy_sl.index, y=hy_sl.values, mode="lines",
                    name="HY OAS", line=dict(color="#b42318", width=2.2)))
                if not ig_oas.empty:
                    ig_sl = slice_series(ig_oas, cr)
                    if not ig_sl.empty:
                        fig_hy.add_trace(go.Scatter(
                            x=ig_sl.index, y=ig_sl.values, mode="lines",
                            name="IG OAS", line=dict(color="#d97706", width=1.8, dash="dot"),
                            yaxis="y2"))
                _now_line(fig_hy, hy_now, "#b42318",
                          f"HY: {hy_now:.2f}%" if hy_now else "")
                lo5 = float(hy_sl.min())*0.95; hi5 = float(hy_sl.max())*1.08
                fig_hy.update_layout(
                    yaxis=dict(range=[lo5, hi5], title="HY OAS %",
                               showgrid=True, gridcolor="#f1f5f9"),
                    yaxis2=dict(overlaying="y", side="right",
                                title="IG OAS %", showgrid=False))
                st.plotly_chart(fig_hy, width='stretch')
            else:
                st.caption("HY OAS data unavailable.")

    with cc2:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>HY OAS vs SPY — credit leads equity</div>",
                        unsafe_allow_html=True)
            st.caption("Divergence (HY widening + SPY flat) is the key warning. "
                       "Credit historically leads by 4-8 weeks.")
            if not hy_oas.empty and not spy.empty:
                hy_sl2  = slice_series(hy_oas, cr)
                spy_r   = spy.pct_change(21).dropna() * 100
                spy_sl  = slice_series(spy_r, cr)
                idx_div = hy_sl2.index.intersection(spy_sl.index)
                if len(idx_div) > 20:
                    fig_div = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            row_heights=[0.55, 0.45],
                                            vertical_spacing=0.05)
                    fig_div.add_trace(go.Scatter(
                        x=hy_sl2.loc[idx_div].index,
                        y=hy_sl2.loc[idx_div].values,
                        mode="lines", name="HY OAS",
                        line=dict(color="#b42318", width=2)), row=1, col=1)
                    bar_c2 = ["#1f7a4f" if v >= 0 else "#b42318"
                              for v in spy_sl.loc[idx_div].values]
                    fig_div.add_trace(go.Bar(
                        x=spy_sl.loc[idx_div].index,
                        y=spy_sl.loc[idx_div].values,
                        marker_color=bar_c2, name="SPY 21d rtn %"), row=2, col=1)
                    fig_div.add_hline(y=0, line_color="#94a3b8",
                                      line_width=1, row=2, col=1)
                    fig_div.update_layout(
                        height=280, margin=dict(l=10,r=20,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.04, x=0, font_size=10))
                    fig_div.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                    fig_div.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                    st.plotly_chart(fig_div, width='stretch')
                else:
                    st.caption("Insufficient overlapping data.")
            else:
                st.caption("HY OAS or SPY data unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK APPETITE
# ══════════════════════════════════════════════════════════════════════════════

with tab_risk:

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    rs_c2  = "#1f7a4f" if (rsp_spy_z or 0) > 0.3 else "#b42318" if (rsp_spy_z or 0) < -0.3 else "#6b7280"
    rs_bg2 = "#dcfce7" if (rsp_spy_z or 0) > 0.3 else "#fee2e2" if (rsp_spy_z or 0) < -0.3 else "#f3f4f6"
    _kpi(k1, "Breadth RSP/SPY",
         fmt(rsp_spy_z, plus=True) if rsp_spy_z is not None else "—",
         "equal-wt z-score", rs_c2, rs_bg2)
    vix_c2 = "#b42318" if (vix_now or 0) > 25 else "#1f7a4f" if (vix_now or 0) < 15 else "#6b7280"
    _kpi(k2, "VIX",
         fmt(vix_now, nd=1) if vix_now else "—",
         "<15 calm · >25 stress", vix_c2)
    vr_c = "#b42318" if (vratio_now or 0) > 1.0 else "#1f7a4f"
    vr_bg = "#fee2e2" if (vratio_now or 0) > 1.0 else "#f3f4f6"
    _kpi(k3, "V-Ratio VIX/VIX3M",
         fmt(vratio_now, nd=3) if vratio_now else "—",
         ">1 = backwardation = stress", vr_c, vr_bg)
    _kpi(k4, "Oil/Gold",
         fmt(_last(oil_gld), nd=3) if _last(oil_gld) else "—",
         "rising = risk-on / growth", "#6b7280")
    _kpi(k5, "Copper/Gold",
         fmt(_last(cop_gld), nd=3) if _last(cop_gld) else "—",
         "industrial demand signal", "#6b7280")

    st.markdown("")

    bull_ra = sum([(rsp_spy_z or 0) > 0.3, (vix_now or 0) < 18,
                   (vratio_now or 0) < 0.92])
    bear_ra = sum([(rsp_spy_z or 0) < -0.3, (vix_now or 0) > 25,
                   (vratio_now or 0) > 1.0])
    if bear_ra >= 2:
        _banner("Risk-off signals active: breadth fading, VIX elevated and/or "
                "VIX backwardation. This combination coincides with equity drawdowns. "
                "V-Ratio above 1.0 is the most acute — it reflects actual hedging demand, "
                "not just media sentiment.", "#b42318", "#fee2e2")
    elif bull_ra >= 2:
        _banner("Risk appetite healthy: breadth broad, VIX calm, term structure in contango. "
                "Conditions support continued risk-on positioning.",
                "#1f7a4f", "#dcfce7")
    else:
        _banner("Risk appetite signals mixed. Monitor V-Ratio and RSP/SPY "
                "for the next directional signal.", "#6b7280", "#f3f4f6")

    rr = st.selectbox("Range", RKEYS, index=RKEYS.index("1y") if "1y" in RKEYS else 1,
                      key="rr_range")
    rc1, rc2 = st.columns(2, gap="large")

    with rc1:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>RSP/SPY equal-weight breadth</div>",
                        unsafe_allow_html=True)
            st.caption("Rising = broad participation · Falling = narrow mega-cap rally.")
            if not rsp_spy.empty:
                rs2 = slice_series(rsp_spy, rr)
                ma63_2 = rs2.rolling(63, min_periods=10).mean()
                fig_rs2 = _fig(260)
                fig_rs2.add_trace(go.Scatter(
                    x=rs2.index, y=rs2.values, mode="lines",
                    name="RSP/SPY", line=dict(color="#1d4ed8", width=2.2)))
                if not ma63_2.dropna().empty:
                    fig_rs2.add_trace(go.Scatter(
                        x=ma63_2.index, y=ma63_2.values, mode="lines",
                        name="63d MA", line=dict(color="#94a3b8", width=1.4, dash="dash")))
                lo6 = float(rs2.min())*0.98; hi6 = float(rs2.max())*1.02
                fig_rs2.update_yaxes(range=[lo6, hi6])
                st.plotly_chart(fig_rs2, width='stretch')

    with rc2:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>V-Ratio — VIX/VIX3M term structure</div>",
                        unsafe_allow_html=True)
            st.caption(">1.0 = backwardation = acute stress · <0.90 = calm contango. "
                       "More sensitive than raw VIX — reflects actual positioning.")
            if not vratio_s.empty:
                vr_sl = slice_series(vratio_s, rr)
                if not vr_sl.empty:
                    fig_vr = _fig(260)
                    fig_vr.add_hrect(y0=1.0, y1=3.0,
                                     fillcolor="rgba(180,35,24,0.06)", line_width=0)
                    bar_vr = ["#b42318" if v >= 1.0 else "#1f7a4f" for v in vr_sl.values]
                    fig_vr.add_trace(go.Bar(
                        x=vr_sl.index, y=vr_sl.values,
                        marker_color=bar_vr, name="V-Ratio"))
                    fig_vr.add_hline(y=1.0, line_color="#b42318", line_width=1.5,
                                     line_dash="dash",
                                     annotation_text="Backwardation",
                                     annotation_position="right",
                                     annotation_font_color="#b42318", annotation_font_size=9)
                    lo7 = float(vr_sl.min())*0.97; hi7 = float(vr_sl.max())*1.05
                    fig_vr.update_yaxes(range=[lo7, hi7], title_text="VIX/VIX3M")
                    st.plotly_chart(fig_vr, width='stretch')
            else:
                st.caption("VIX3M data unavailable (^VIX3M).")

    rc3, rc4 = st.columns(2, gap="large")

    with rc3:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Oil/Gold — risk vs safety</div>",
                        unsafe_allow_html=True)
            st.caption("Rising = markets pricing growth. Falling = flight to gold / recession.")
            if not oil_gld.empty:
                og = slice_series(oil_gld, rr)
                fig_og = _fig(240)
                fig_og.add_trace(go.Scatter(
                    x=og.index, y=og.values, mode="lines",
                    name="Oil/Gold", line=dict(color="#f97316", width=2)))
                fig_og.add_trace(go.Scatter(
                    x=og.index, y=og.rolling(63, min_periods=10).mean().values,
                    mode="lines", name="63d MA",
                    line=dict(color="#94a3b8", width=1.2, dash="dash")))
                lo8 = float(og.min())*0.96; hi8 = float(og.max())*1.04
                fig_og.update_yaxes(range=[lo8, hi8])
                st.plotly_chart(fig_og, width='stretch')

    with rc4:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Copper/Gold — industrial demand</div>",
                        unsafe_allow_html=True)
            st.caption("Copper/gold ratio tracks global growth expectations. "
                       "Falling = China slowdown or global demand worry.")
            if not cop_gld.empty:
                cg2 = slice_series(cop_gld, rr)
                fig_cg2 = _fig(240)
                fig_cg2.add_trace(go.Scatter(
                    x=cg2.index, y=cg2.values, mode="lines",
                    name="Copper/Gold", line=dict(color="#a855f7", width=2)))
                fig_cg2.add_trace(go.Scatter(
                    x=cg2.index, y=cg2.rolling(63, min_periods=10).mean().values,
                    mode="lines", name="63d MA",
                    line=dict(color="#94a3b8", width=1.2, dash="dash")))
                lo9 = float(cg2.min())*0.96; hi9 = float(cg2.max())*1.04
                fig_cg2.update_yaxes(range=[lo9, hi9])
                st.plotly_chart(fig_cg2, width='stretch')

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)