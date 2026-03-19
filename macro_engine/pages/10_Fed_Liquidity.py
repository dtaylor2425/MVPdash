# pages/10_Fed_Liquidity.py
"""
Fed & Liquidity View  — v2
═══════════════════════════════════════════════════════════════════════════════
Redesigned opening: lead with the policy tension dashboard above the fold.
Three pillars → 4-chart deep-dive → playbook.
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

st.set_page_config(page_title="Fed & Liquidity", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Fed & Liquidity")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

RKEYS = list(RANGES.keys())

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_data():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    px = fetch_prices(["IWM", "SPY", "HYG", "TLT", "GLD"], period="5y")
    return macro, (px if px is not None and not px.empty else pd.DataFrame())

macro, px = load_data()

def _col(name):
    return macro[name].dropna() if name in macro.columns \
           else pd.Series(dtype=float, name=name)

fed_assets = _col("fed_assets")
real10     = _col("real10")
dollar     = _col("dollar_broad")
y10        = _col("y10")
y2         = _col("y2")
ff         = _col("fed_funds")
cpi_raw    = _col("cpi")
hy_oas     = _col("hy_oas")

cpi_yoy   = (cpi_raw.pct_change(12) * 100).dropna() if len(cpi_raw) >= 13 else None
breakeven = (y10 - real10).dropna() if len(y10) > 10 and len(real10) > 10 \
            else pd.Series(dtype=float)
fed_roc13 = fed_assets.pct_change(63).dropna() * 100  if len(fed_assets) >= 70  else None
fed_roc26 = fed_assets.pct_change(126).dropna() * 100 if len(fed_assets) >= 130 else None
fed_z     = None
if len(fed_assets) >= 60:
    w = fed_assets.rolling(252, min_periods=60).mean()
    s = fed_assets.rolling(252, min_periods=60).std()
    fed_z = ((fed_assets - w) / s.replace(0, np.nan)).dropna()

iwm_spy = None
if not px.empty and "IWM" in px.columns and "SPY" in px.columns:
    c = px["IWM"].dropna().index.intersection(px["SPY"].dropna().index)
    if len(c) > 60:
        iwm_spy = (px["IWM"].loc[c] / px["SPY"].loc[c]).dropna()

# ── Live reads ────────────────────────────────────────────────────────────────

def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _delta(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] - s.loc[prev[-1]]) if len(prev) else None

def _pct_delta(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    b = float(s.loc[prev[-1]]) if len(prev) else None
    return float((s.iloc[-1] / b - 1) * 100) if b and b != 0 else None

def _pct_rank(s, window=252*3):
    s = s.dropna()
    if len(s) < 60: return None
    w = s.iloc[-min(window, len(s)):]
    return float((w < s.iloc[-1]).mean() * 100)

fed_now       = _last(fed_assets)
fed_13w_pct   = _pct_delta(fed_assets, 91)
fed_roc13_now = _last(fed_roc13) if fed_roc13 is not None else None
real_now      = _last(real10)
real_1m       = _delta(real10, 30)
real_pctrank  = _pct_rank(real10)
dollar_now    = _last(dollar)
dollar_1m     = _delta(dollar, 30)
dollar_3m     = _delta(dollar, 91)
cpi_now       = _last(cpi_yoy) if cpi_yoy is not None else None
cpi_3m        = _delta(cpi_yoy, 91) if cpi_yoy is not None else None
ff_now        = _last(ff)
be_now        = _last(breakeven)
be_3m         = _delta(breakeven, 91)

# ── Policy stance ─────────────────────────────────────────────────────────────

def classify_stance(real_now, fed_roc13_now, cpi_now):
    if real_now is None:
        return "Unknown", "#6b7280", "#f3f4f6"
    if real_now > 1.5 and (fed_roc13_now is None or fed_roc13_now < 0):
        return "Restrictive", "#b42318", "#fee2e2"
    if real_now < 0.5 or (fed_roc13_now is not None and fed_roc13_now > 3):
        return "Accommodative", "#1f7a4f", "#dcfce7"
    if real_now > 0.8 and (fed_roc13_now is None or fed_roc13_now < 0):
        return "Tightening", "#d97706", "#fef9c3"
    return "Neutral", "#6b7280", "#f3f4f6"

stance_label, stance_color, stance_bg = classify_stance(real_now, fed_roc13_now, cpi_now)

# ── Pillar signal helpers ─────────────────────────────────────────────────────

def _signal_row(label, value_html, sub, signal, sig_color):
    """One row in the pillar card."""
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:8px 0;border-bottom:1px solid rgba(0,0,0,0.05);'>"
        f"<div>"
        f"  <div style='font-size:12px;font-weight:700;color:rgba(0,0,0,0.75);'>{label}</div>"
        f"  <div style='font-size:10px;color:rgba(0,0,0,0.42);margin-top:1px;'>{sub}</div>"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"  <div style='font-size:16px;font-weight:900;color:{sig_color};'>{value_html}</div>"
        f"  <div style='font-size:10px;font-weight:700;color:{sig_color};'>{signal}</div>"
        f"</div>"
        f"</div>"
    )

def _gauge_bar(value_pct, color):
    """Simple horizontal fill bar 0–100%."""
    pct = max(0, min(100, value_pct or 0))
    return (
        f"<div style='background:rgba(0,0,0,0.07);border-radius:4px;height:6px;"
        f"margin:8px 0 4px;overflow:hidden;'>"
        f"<div style='width:{pct:.0f}%;background:{color};height:100%;border-radius:4px;'></div>"
        f"</div>"
        f"<div style='font-size:9px;color:rgba(0,0,0,0.38);'>{pct:.0f}th percentile vs 3y</div>"
    )

def fmt(x, nd=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x):.{nd}f}{suffix}"
def fmt_s(x, nd=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x):+.{nd}f}{suffix}"

def _fig_base(height=260):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=20, b=20),
                      plot_bgcolor="white", paper_bgcolor="white",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig

def _dyn_y(series_list, pad=0.12):
    vals = []
    for s in series_list:
        if s is not None and hasattr(s, '__iter__'):
            vals += [v for v in s if v is not None and not np.isnan(v)]
    if not vals: return None, None
    lo = min(vals); hi = max(vals)
    p = max((hi-lo)*pad, abs(hi)*0.02, 0.01)
    return lo-p, hi+p

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
              <div class="me-title">Fed &amp; Liquidity</div>
              <div class="me-subtle">
                Real rate regime &nbsp;·&nbsp; balance sheet impulse &nbsp;·&nbsp;
                dollar cycle &nbsp;·&nbsp; inflation path
              </div>
            </div>
            <div style="padding:8px 18px;border-radius:20px;background:{stance_bg};">
              <span style="font-weight:800;color:{stance_color};font-size:14px;">
                {stance_label}
              </span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — THREE PILLARS  (first thing you see, all above the fold)
# ══════════════════════════════════════════════════════════════════════════════
# Each pillar = one key question + current reading + regime context + gauge

st.markdown("<div class='me-rowtitle'>Policy conditions at a glance</div>",
            unsafe_allow_html=True)

p1, p2, p3 = st.columns(3, gap="medium")

# ── Pillar 1: Real rates ───────────────────────────────────────────────────────
with p1:
    r_col = stance_color if stance_label in ("Restrictive","Tightening") else \
            ("#1f7a4f" if (real_now and real_now < 0.5) else "#6b7280")
    r_level_lbl = "Restrictive" if real_now and real_now > 1.5 else \
                  ("Tightening" if real_now and real_now > 0.8 else \
                  ("Neutral" if real_now and real_now > 0.3 else "Accommodative"))
    r_col2 = "#b42318" if r_level_lbl in ("Restrictive",) else \
             ("#d97706" if r_level_lbl == "Tightening" else \
             ("#1f7a4f" if r_level_lbl == "Accommodative" else "#6b7280"))
    r_bg2  = "#fee2e2" if r_level_lbl == "Restrictive" else \
             ("#fef9c3" if r_level_lbl == "Tightening" else \
             ("#dcfce7" if r_level_lbl == "Accommodative" else "#f3f4f6"))

    rows_html = (
        _signal_row("10y real yield (TIPS)", fmt(real_now, suffix="%"),
                    "Restrictive >1.5% · Neutral ~0.5%",
                    r_level_lbl, r_col2) +
        _signal_row("1m change", fmt_s(real_1m, suffix="pp"),
                    "Rising = conditions tightening",
                    "↑ tightening" if (real_1m and real_1m > 0.05) else
                    ("↓ easing" if (real_1m and real_1m < -0.05) else "stable"),
                    "#b42318" if (real_1m and real_1m > 0.05) else
                    ("#1f7a4f" if (real_1m and real_1m < -0.05) else "#6b7280")) +
        _signal_row("Breakeven inflation", fmt(be_now, suffix="%"),
                    "Market's inflation expectation",
                    fmt_s(be_3m, suffix="pp 3m") if be_3m is not None else "—",
                    "#d97706" if (be_3m and be_3m > 0.1) else "#6b7280")
    )
    gauge_html = _gauge_bar(real_pctrank, r_col2) if real_pctrank is not None else ""

    st.markdown(
        f"<div style='padding:16px;border-radius:14px;background:{r_bg2};"
        f"border:1.5px solid {r_col2}44;'>"
        f"<div style='font-size:10px;font-weight:700;color:{r_col2};"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>"
        f"Pillar 1 — Real rates</div>"
        f"<div style='font-size:30px;font-weight:900;color:{r_col2};line-height:1;'>"
        f"{fmt(real_now, suffix='%')}</div>"
        f"{gauge_html}"
        f"<div style='margin-top:8px;'>{rows_html}</div>"
        f"<div style='margin-top:10px;font-size:11px;color:rgba(0,0,0,0.60);"
        f"line-height:1.5;padding:8px 10px;border-radius:8px;background:rgba(0,0,0,0.04);'>"
        f"{'Real yields above 1.5% — financial conditions are genuinely restrictive. Every 100bp of real yield compresses equity multiples by ~1–2×.' if (real_now and real_now > 1.5) else 'Real yields in the neutral-to-tightening zone. Not yet restrictive enough to cause credit stress, but not accommodative either.'}"
        f"</div></div>",
        unsafe_allow_html=True)

# ── Pillar 2: Fed balance sheet impulse ───────────────────────────────────────
with p2:
    bs_signal = "Injecting" if (fed_roc13_now and fed_roc13_now > 1) else \
                ("Draining" if (fed_roc13_now and fed_roc13_now < -1) else "Flat")
    bs_col  = "#1f7a4f" if bs_signal == "Injecting" else \
              ("#b42318" if bs_signal == "Draining" else "#6b7280")
    bs_bg   = "#dcfce7" if bs_signal == "Injecting" else \
              ("#fee2e2" if bs_signal == "Draining" else "#f3f4f6")

    # Fed assets 5y percentile rank
    fed_pctrank = _pct_rank(fed_assets)

    bs_rows = (
        _signal_row("Fed assets", f"${fed_now/1e6:.2f}T" if fed_now else "—",
                    "Total balance sheet size",
                    f"{fmt_s(fed_13w_pct, 1, '%')} 13w",
                    "#1f7a4f" if (fed_13w_pct and fed_13w_pct > 0) else "#b42318") +
        _signal_row("13w impulse", fmt_s(fed_roc13_now, 1, "%"),
                    "Rate of change vs 13w ago",
                    bs_signal, bs_col) +
        _signal_row("Fed funds rate", fmt(ff_now, suffix="%") if ff_now else "—",
                    "Policy rate",
                    "Above neutral" if (ff_now and ff_now > 3.5) else
                    ("Near neutral" if (ff_now and ff_now > 2.0) else "Accommodative"),
                    "#b42318" if (ff_now and ff_now > 3.5) else "#6b7280")
    )
    bs_gauge = _gauge_bar(100 - (fed_pctrank or 50), bs_col)  # invert: lower BS = more drained

    st.markdown(
        f"<div style='padding:16px;border-radius:14px;background:{bs_bg};"
        f"border:1.5px solid {bs_col}44;'>"
        f"<div style='font-size:10px;font-weight:700;color:{bs_col};"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>"
        f"Pillar 2 — Liquidity impulse</div>"
        f"<div style='font-size:30px;font-weight:900;color:{bs_col};line-height:1;'>"
        f"{bs_signal}</div>"
        f"{bs_gauge}"
        f"<div style='margin-top:8px;'>{bs_rows}</div>"
        f"<div style='margin-top:10px;font-size:11px;color:rgba(0,0,0,0.60);"
        f"line-height:1.5;padding:8px 10px;border-radius:8px;background:rgba(0,0,0,0.04);'>"
        f"{'Balance sheet is contracting — active QT. Historically this leads to tighter financial conditions with a 3–6 month lag. Watch credit spreads for the signal.' if bs_signal == 'Draining' else ('Balance sheet expanding — liquidity injection is a tailwind for risk assets, particularly small caps and credit.' if bs_signal == 'Injecting' else 'Balance sheet effectively flat. Passive runoff may continue. The impulse, not the level, is what matters for markets.')}"
        f"</div></div>",
        unsafe_allow_html=True)

# ── Pillar 3: Dollar cycle ─────────────────────────────────────────────────────
with p3:
    dl_signal = "Strengthening" if (dollar_3m and dollar_3m > 1.5) else \
                ("Weakening" if (dollar_3m and dollar_3m < -1.5) else "Stable")
    dl_col  = "#b42318" if dl_signal == "Strengthening" else \
              ("#1f7a4f" if dl_signal == "Weakening" else "#6b7280")
    dl_bg   = "#fee2e2" if dl_signal == "Strengthening" else \
              ("#dcfce7" if dl_signal == "Weakening" else "#f3f4f6")
    dl_pctrank = _pct_rank(dollar)

    dl_rows = (
        _signal_row("Broad dollar index", fmt(dollar_now, nd=1),
                    "Broad trade-weighted USD",
                    dl_signal, dl_col) +
        _signal_row("1m change", fmt_s(dollar_1m, nd=1),
                    "Rising = tighter global conditions",
                    "↑ strong" if (dollar_1m and dollar_1m > 0.5) else
                    ("↓ weak" if (dollar_1m and dollar_1m < -0.5) else "flat"),
                    "#b42318" if (dollar_1m and dollar_1m > 0.5) else
                    ("#1f7a4f" if (dollar_1m and dollar_1m < -0.5) else "#6b7280")) +
        _signal_row("CPI YoY", fmt(cpi_now, suffix="%") if cpi_now else "—",
                    "Realised inflation",
                    fmt_s(cpi_3m, suffix="pp 3m") if cpi_3m is not None else "—",
                    "#b42318" if (cpi_3m and cpi_3m > 0.2) else
                    ("#1f7a4f" if (cpi_3m and cpi_3m < -0.2) else "#6b7280"))
    )
    dl_gauge = _gauge_bar(dl_pctrank, dl_col) if dl_pctrank is not None else ""

    st.markdown(
        f"<div style='padding:16px;border-radius:14px;background:{dl_bg};"
        f"border:1.5px solid {dl_col}44;'>"
        f"<div style='font-size:10px;font-weight:700;color:{dl_col};"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>"
        f"Pillar 3 — Dollar &amp; inflation</div>"
        f"<div style='font-size:30px;font-weight:900;color:{dl_col};line-height:1;'>"
        f"{fmt(dollar_now, nd=1)}</div>"
        f"{dl_gauge}"
        f"<div style='margin-top:8px;'>{dl_rows}</div>"
        f"<div style='margin-top:10px;font-size:11px;color:rgba(0,0,0,0.60);"
        f"line-height:1.5;padding:8px 10px;border-radius:8px;background:rgba(0,0,0,0.04);'>"
        f"{'Strong dollar tightens global financial conditions — EM USD debt becomes more expensive and capital flows reverse. Watch for GLD and commodity weakness.' if dl_signal == 'Strengthening' else ('Dollar weakening is a meaningful global easing signal. Historically bullish for EM, commodities, and GLD. Often co-incides with Fed pivot phases.' if dl_signal == 'Weakening' else 'Dollar stable. Not adding to or subtracting from global financial conditions. CPI trajectory is the dominant signal to watch.')}"
        f"</div></div>",
        unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — POLICY TENSION READ  (the "so what" before the charts)
# ══════════════════════════════════════════════════════════════════════════════

# Compute cross-pillar tensions
tensions = []
if real_now and real_now > 1.5 and (fed_roc13_now is None or fed_roc13_now < 0):
    tensions.append(("⚠️ Maximum tightening", "#b42318",
                     f"Real yields {real_now:.2f}% + balance sheet contracting — "
                     "both pillars pointing in the same restrictive direction. "
                     "Credit stress and equity multiple compression are the historical outcome."))
if real_now and real_now > 1.5 and fed_roc13_now and fed_roc13_now > 1:
    tensions.append(("⚡ Tension: restrictive rates + expanding BS", "#d97706",
                     f"Real yields elevated ({real_now:.2f}%) but balance sheet injecting liquidity ({fed_roc13_now:+.1f}% 13w). "
                     "Markets are fighting the Fed's rate signal with the liquidity signal. "
                     "Watch which one breaks first — typically rates win."))
if cpi_now and cpi_now > 3 and real_now and real_now > 1.5:
    tensions.append(("🔴 Stagflation signal", "#b42318",
                     f"CPI at {cpi_now:.1f}% with real yields already at {real_now:.2f}%. "
                     "The Fed cannot ease without re-igniting inflation. "
                     "Historically the worst combination for both equities and bonds."))
if cpi_now and cpi_now < 2 and be_now and be_now > 2.3:
    tensions.append(("📊 Inflation re-acceleration risk", "#d97706",
                     f"CPI has fallen to {cpi_now:.1f}% but breakeven at {be_now:.2f}% "
                     "suggests the bond market expects inflation to re-accelerate. "
                     "Watch for this gap to close — either CPI rises or breakevens correct."))
if not tensions:
    tensions.append((f"● {stance_label}", stance_color,
                     "No significant cross-pillar tensions. Conditions consistent "
                     "with current regime label. Monitor for inflection points."))

for icon_lbl, t_color, t_text in tensions:
    t_bg = "#fee2e2" if "#b42318" in t_color else \
           ("#fef9c3" if "#d97706" in t_color else "#f3f4f6")
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:12px;background:{t_bg};"
        f"border-left:4px solid {t_color};margin-bottom:8px;font-size:13px;"
        f"color:rgba(0,0,0,0.82);line-height:1.6;'>"
        f"<span style='font-weight:800;color:{t_color};'>{icon_lbl}</span>"
        f"&nbsp;&nbsp;{t_text}</div>",
        unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — REAL RATE PATH  |  BALANCE SHEET + IMPULSE  (side by side, 1y default)
# ══════════════════════════════════════════════════════════════════════════════

rr_col, bs_col_w = st.columns(2, gap="large")

with rr_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Real rate path (10y TIPS)</div>",
                    unsafe_allow_html=True)
        st.caption("Above 1.5% = restrictive. Below 0% = accommodative / bullish for GLD + equities.")
        rr_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="rr_range")
        rr_sl    = slice_series(real10, rr_range)
        if not rr_sl.empty:
            fig_rr = _fig_base(260)
            for y0, y1, bg in [(-5, 0, "rgba(31,122,79,0.07)"),
                                (0, 0.5, "rgba(107,114,128,0.04)"),
                                (0.5, 1.5, "rgba(217,119,6,0.05)"),
                                (1.5, 6, "rgba(180,35,24,0.07)")]:
                fig_rr.add_hrect(y0=y0, y1=y1, fillcolor=bg, line_width=0)
            lo_r = float(rr_sl.min()); hi_r = float(rr_sl.max())
            pad_r = max((hi_r - lo_r) * 0.15, 0.05)
            for lvl, lbl in [(0,"Zero"), (0.5,"Neutral"), (1.5,"Restrictive")]:
                if lo_r - pad_r <= lvl <= hi_r + pad_r:
                    fig_rr.add_hline(y=lvl, line_dash="dash", line_color="#cbd5e1",
                                     line_width=1, annotation_text=lbl,
                                     annotation_position="right", annotation_font_size=9)
            fig_rr.add_trace(go.Scatter(x=rr_sl.index, y=rr_sl.values,
                                         mode="lines", name="10y real",
                                         line=dict(color="#dc2626", width=2.2),
                                         fill="tozeroy",
                                         fillcolor="rgba(220,38,38,0.06)"))
            if real_now is not None:
                rc = "#b42318" if real_now > 1.5 else ("#d97706" if real_now > 0.5 else "#1f7a4f")
                fig_rr.add_hline(y=real_now, line_color=rc, line_width=1.5,
                                 annotation_text=f"Now: {real_now:.2f}%",
                                 annotation_position="right", annotation_font_color=rc)
            fig_rr.update_yaxes(range=[lo_r-pad_r, hi_r+pad_r], title_text="Real yield (%)")
            st.plotly_chart(fig_rr, width='stretch')
        else:
            st.caption("Real yield data unavailable.")

with bs_col_w:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Balance sheet &amp; 13w impulse</div>",
                    unsafe_allow_html=True)
        st.caption("Impulse inflection (green→red) leads risk asset drawdowns by ~3–6 months.")
        bs_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="bs_range")
        fa_sl    = slice_series(fed_assets, bs_range)

        if not fa_sl.empty and fed_roc13 is not None:
            roc_sl = slice_series(fed_roc13, bs_range)
            fig_bs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.58, 0.42], vertical_spacing=0.05)
            fa_t = fa_sl / 1e6
            fig_bs.add_trace(go.Scatter(x=fa_t.index, y=fa_t.values,
                                        mode="lines", name="Fed assets ($T)",
                                        line=dict(color="#1d4ed8", width=2),
                                        fill="tozeroy",
                                        fillcolor="rgba(29,78,216,0.06)"),
                             row=1, col=1)
            fa_lo = float(fa_t.min()) * 0.97; fa_hi = float(fa_t.max()) * 1.02
            fig_bs.update_yaxes(range=[fa_lo, fa_hi], title_text="$T", row=1, col=1)

            if not roc_sl.empty:
                bar_colors = ["#1f7a4f" if v >= 0 else "#b42318" for v in roc_sl.values]
                fig_bs.add_trace(go.Bar(x=roc_sl.index, y=roc_sl.values,
                                        marker_color=bar_colors, name="13w impulse %",
                                        showlegend=True), row=2, col=1)
                fig_bs.add_hline(y=0, line_color="#94a3b8", line_width=1, row=2, col=1)
                rv = roc_sl.values
                rlo = min(rv); rhi = max(rv)
                rpad = max(abs(rlo), abs(rhi)) * 0.15
                fig_bs.update_yaxes(range=[rlo-rpad, rhi+rpad], title_text="13w Δ%",
                                    row=2, col=1)

            fig_bs.update_layout(height=260, margin=dict(l=10, r=20, t=10, b=10),
                                 plot_bgcolor="white", paper_bgcolor="white",
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                                 hovermode="x unified", barmode="relative")
            fig_bs.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig_bs.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_bs, width='stretch')
        else:
            st.caption("Fed balance sheet data unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — CPI + BREAKEVEN  |  DOLLAR CYCLE + LIQUIDITY PROXY
# ══════════════════════════════════════════════════════════════════════════════

inf_col, liq_col = st.columns(2, gap="large")

with inf_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>CPI vs breakeven — inflation path</div>",
                    unsafe_allow_html=True)
        st.caption("Breakeven above CPI = market expecting re-acceleration. "
                   "Gap narrowing = disinflation confirmed by market.")
        inf_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="inf_range")
        fig_inf = _fig_base(240)
        all_inf = []
        plotted = False
        if cpi_yoy is not None and not cpi_yoy.empty:
            ci_sl = slice_series(cpi_yoy, inf_range)
            if not ci_sl.empty:
                fig_inf.add_trace(go.Scatter(x=ci_sl.index, y=ci_sl.values,
                                             mode="lines", name="CPI YoY",
                                             line=dict(color="#dc2626", width=2)))
                all_inf += list(ci_sl.values); plotted = True
        if not breakeven.empty:
            be_sl = slice_series(breakeven, inf_range)
            if not be_sl.empty:
                fig_inf.add_trace(go.Scatter(x=be_sl.index, y=be_sl.values,
                                             mode="lines", name="10y breakeven",
                                             line=dict(color="#d97706", width=2, dash="dot")))
                all_inf += list(be_sl.values); plotted = True
        if plotted and all_inf:
            lo_i = min(all_inf); hi_i = max(all_inf)
            pad_i = max((hi_i - lo_i) * 0.15, 0.1)
            if lo_i - pad_i <= 2.0 <= hi_i + pad_i:
                fig_inf.add_hline(y=2.0, line_dash="dash", line_color="#94a3b8",
                                  line_width=1, annotation_text="2% target",
                                  annotation_position="right", annotation_font_size=9)
            fig_inf.update_yaxes(range=[lo_i-pad_i, hi_i+pad_i], title_text="% YoY")
            st.plotly_chart(fig_inf, width='stretch')
        else:
            st.caption("Inflation data unavailable.")

with liq_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Dollar cycle &amp; liquidity proxy</div>",
                    unsafe_allow_html=True)
        st.caption("Dollar (left) + Fed z-score vs IWM/SPY breadth (right). "
                   "All three rising = genuine risk-on. Divergences are the story.")
        lq_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="lq_range")

        dl_sl = slice_series(dollar, lq_range)
        if not dl_sl.empty:
            fig_dl = _fig_base(240)
            ma63 = dl_sl.rolling(63, min_periods=10).mean()
            fig_dl.add_trace(go.Scatter(x=dl_sl.index, y=dl_sl.values,
                                        mode="lines", name="Dollar broad",
                                        line=dict(color="#1d4ed8", width=2)))
            if not ma63.dropna().empty:
                fig_dl.add_trace(go.Scatter(x=ma63.index, y=ma63.values,
                                            mode="lines", name="63d MA",
                                            line=dict(color="#94a3b8", width=1.4, dash="dash")))
            if iwm_spy is not None and fed_z is not None:
                iw_sl = slice_series(iwm_spy, lq_range)
                fz_sl = slice_series(fed_z, lq_range)
                if not iw_sl.empty:
                    fig_dl.add_trace(go.Scatter(x=iw_sl.index, y=iw_sl.values,
                                                mode="lines", name="IWM/SPY",
                                                line=dict(color="#f97316", width=1.6, dash="dot"),
                                                yaxis="y2"))
            vals_dl = list(dl_sl.values)
            lo_d = min(vals_dl); hi_d = max(vals_dl)
            pad_d = max((hi_d-lo_d)*0.08, 0.5)
            fig_dl.update_layout(
                yaxis=dict(range=[lo_d-pad_d, hi_d+pad_d], title="Dollar",
                           showgrid=True, gridcolor="#f1f5f9"),
                yaxis2=dict(overlaying="y", side="right", title="IWM/SPY",
                            showgrid=False, showticklabels=True),
            )
            st.plotly_chart(fig_dl, width='stretch')
        else:
            st.caption("Dollar data unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — POLICY STANCE PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Policy stance playbook</div>",
                unsafe_allow_html=True)
    p1c, p2c, p3c, p4c = st.columns(4, gap="medium")

    def playbook_card(col, regime, trigger, impl, watch, report, color, bg):
        active = stance_label.lower() in regime.lower() or regime.lower() in stance_label.lower()
        border = f"2px solid {color}" if active else "1px solid rgba(0,0,0,0.08)"
        badge  = (f"<span style='font-size:10px;font-weight:700;color:{color};"
                  f"background:{bg};padding:2px 7px;border-radius:8px;margin-left:6px;'>"
                  f"NOW</span>") if active else ""
        col.markdown(
            f"<div style='padding:12px 14px;border-radius:12px;background:{bg};"
            f"border:{border};height:100%;'>"
            f"<div style='font-weight:800;font-size:13px;color:{color};margin-bottom:10px;'>"
            f"{regime}{badge}</div>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>Trigger</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.70);margin-bottom:8px;'>{trigger}</div>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>Implication</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.70);margin-bottom:8px;'>{impl}</div>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>Watch for</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.70);margin-bottom:8px;'>{watch}</div>"
            f"<div style='font-size:9px;font-weight:700;color:{color};"
            f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>Report angle</div>"
            f"<div style='font-size:11px;color:{color};font-weight:700;'>{report}</div>"
            f"</div>", unsafe_allow_html=True)

    playbook_card(p1c, "Accommodative",
        "Real yield < 0.5% OR BS expanding >2%/13w",
        "Maximum tailwind. Strongest equity and credit environment.",
        "Breakeven surging — could force Fed to slow",
        "Overweight equities, credit, GLD. Underweight USD. Add EM.",
        "#1f7a4f", "#dcfce7")
    playbook_card(p2c, "Neutral",
        "Real yield 0.5–0.8%, BS flat",
        "Risk assets can perform but need earnings support.",
        "Real yield or BS inflection breaking the range",
        "Balanced. Quality over beta. Watch Fed language closely.",
        "#6b7280", "#f3f4f6")
    playbook_card(p3c, "Tightening",
        "Real yield 0.8–1.5%, BS flat/declining",
        "Late cycle. Breadth narrows. USD supported.",
        "Real yield >1.5% or credit spreads +50bp",
        "Reduce beta. Add quality & selective duration. Watch credit.",
        "#d97706", "#fef9c3")
    playbook_card(p4c, "Restrictive",
        "Real yield >1.5% AND BS contracting",
        "Maximum headwind. Credit stress follows in 3–9m.",
        "BS inflection to expansion = first pivot signal",
        "Defensive. GLD, TLT, cash. Avoid HY and small caps.",
        "#b42318", "#fee2e2")

st.markdown("")
st.caption("WALCL · DFII10 · DTWEXBGS · CPIAUCSL · FEDFUNDS — data via FRED.")
st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)