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
real5      = _col("real5")
dollar     = _col("dollar_broad")
y10        = _col("y10")
y5         = _col("y5")
y2         = _col("y2")
ff         = _col("fed_funds")
cpi_raw    = _col("cpi")
hy_oas     = _col("hy_oas")

cpi_yoy   = (cpi_raw.pct_change(12) * 100).dropna() if len(cpi_raw) >= 13 else None
breakeven = (y10 - real10).dropna() if len(y10) > 10 and len(real10) > 10 \
            else pd.Series(dtype=float)

# ── Three forward rate series ────────────────────────────────────────────────
# Nominal 5y5y  = 2×10y − 5y          → where Treasury yields expected 5-10y out
# Real 5y5y     = 2×real10 − real5     → real growth/rate expectations 5-10y out
# Inflation 5y5y = nominal − real      → implied inflation expectations 5-10y out
#                                         THIS is what the TIPS desk talks about
#                                         Fed target ~2.0%, concern >2.5%

fwd_5y5y_nom  = pd.Series(dtype=float)  # nominal
fwd_5y5y_real = pd.Series(dtype=float)  # real (inflation-adjusted)
fwd_5y5y_inf  = pd.Series(dtype=float)  # implied inflation forward

if len(y10) > 10 and len(y5) > 10:
    _idx_n = y10.index.intersection(y5.index)
    if len(_idx_n) > 10:
        fwd_5y5y_nom = (2 * y10.loc[_idx_n] - y5.loc[_idx_n]).dropna()

if len(real10) > 10 and len(real5) > 10:
    _idx_r = real10.index.intersection(real5.index)
    if len(_idx_r) > 10:
        fwd_5y5y_real = (2 * real10.loc[_idx_r] - real5.loc[_idx_r]).dropna()

if not fwd_5y5y_nom.empty and not fwd_5y5y_real.empty:
    _idx_i = fwd_5y5y_nom.index.intersection(fwd_5y5y_real.index)
    if len(_idx_i) > 10:
        fwd_5y5y_inf = (fwd_5y5y_nom.loc[_idx_i] - fwd_5y5y_real.loc[_idx_i]).dropna()

# Keep fwd_5y5y as alias for inflation forward (the most important one)
fwd_5y5y = fwd_5y5y_inf if not fwd_5y5y_inf.empty else fwd_5y5y_nom
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
fwd5y5y_now   = _last(fwd_5y5y_inf) if not fwd_5y5y_inf.empty else _last(fwd_5y5y_nom)
fwd5y5y_nom_now = _last(fwd_5y5y_nom) if not fwd_5y5y_nom.empty else None
fwd5y5y_real_now = _last(fwd_5y5y_real) if not fwd_5y5y_real.empty else None
fwd5y5y_1w    = _delta(fwd_5y5y, 7)  if not fwd_5y5y.empty else None
fwd5y5y_1m    = _delta(fwd_5y5y, 30) if not fwd_5y5y.empty else None

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

# ── Bloomberg-style chart colours ────────────────────────────────────────────
BB_BG       = "#0d0d0d"   # near-black background
BB_PAPER    = "#0d0d0d"
BB_GRID     = "#1e1e1e"   # very subtle gridlines
BB_GRID2    = "#2a2a2a"   # slightly more visible
BB_TEXT     = "#c8c8c8"   # axis labels
BB_SUBTEXT  = "#666666"
BB_ORANGE   = "#f79400"   # Bloomberg amber — primary line colour
BB_GREEN    = "#4aba6e"   # green series
BB_RED      = "#e84040"   # red series
BB_PURPLE   = "#a78bfa"   # purple series (5y5y)
BB_YELLOW   = "#ffd700"   # highlight / now line
BB_BLUE     = "#4a8fe8"   # secondary blue

def _fig_base(height=260):
    """Light theme — used for simple supporting charts."""
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=20, b=20),
                      plot_bgcolor="white", paper_bgcolor="white",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig

def _fig_bb(height=280, title=""):
    """Bloomberg dark theme — for primary charts."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=48, t=32, b=12),
        plot_bgcolor=BB_BG,
        paper_bgcolor=BB_PAPER,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e1e1e", font_color=BB_TEXT, font_size=11,
                        bordercolor="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
                    font=dict(size=10, color=BB_TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(family="'Courier New', monospace", color=BB_TEXT, size=10),
        title=dict(text=title, font=dict(size=11, color=BB_TEXT), x=0, xanchor="left",
                   y=0.98) if title else dict(text=""),
        xaxis=dict(
            showgrid=True, gridcolor=BB_GRID, gridwidth=1,
            zeroline=False, showline=True, linecolor=BB_GRID2,
            tickfont=dict(size=9, color=BB_TEXT, family="'Courier New', monospace"),
            tickformat="%b '%y",
        ),
        yaxis=dict(
            showgrid=True, gridcolor=BB_GRID, gridwidth=1,
            zeroline=False, showline=False,
            tickfont=dict(size=9, color=BB_TEXT, family="'Courier New', monospace"),
            side="right",
        ),
    )
    return fig

def _bb_now_line(fig, val, label, color=None):
    """Dotted 'NOW' annotation line — Bloomberg style."""
    if val is None: return
    c = color or BB_YELLOW
    fig.add_hline(
        y=val, line_color=c, line_width=1.2, line_dash="dot",
        annotation_text=f"  {label}: {val:.2f}",
        annotation_position="right",
        annotation_font_color=c, annotation_font_size=9,
        annotation_font_family="'Courier New', monospace",
    )

def _bb_threshold(fig, val, label, color=BB_GRID2):
    """Subtle threshold reference line."""
    if val is None: return
    fig.add_hline(
        y=val, line_color=color, line_width=0.8, line_dash="dash",
        annotation_text=f"  {label}",
        annotation_position="right",
        annotation_font_color=BB_SUBTEXT, annotation_font_size=8,
        annotation_font_family="'Courier New', monospace",
    )

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

    # 5y5y colour + signal
    _fwd_c   = ("#b42318" if (fwd5y5y_now and fwd5y5y_now > 2.5) else
                ("#1f7a4f" if (fwd5y5y_now and fwd5y5y_now < 2.0) else "#6b7280"))
    _fwd_sig = ("Unanchored ↑" if (fwd5y5y_now and fwd5y5y_now > 2.5) else
                ("Anchored ↓"  if (fwd5y5y_now and fwd5y5y_now < 2.0) else "In range"))

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
                    ("#1f7a4f" if (cpi_3m and cpi_3m < -0.2) else "#6b7280")) +
        (_signal_row("5y5y forward",
                     fmt(fwd5y5y_now, nd=2, suffix="%"),
                     "Long-run inflation exp · Fed preferred gauge",
                     fmt_s(fwd5y5y_1w, nd=2, suffix="pp 1w") if fwd5y5y_1w else _fwd_sig,
                     _fwd_c) if fwd5y5y_now is not None else "")
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
if fwd5y5y_now and fwd5y5y_now > 2.5:
    tensions.append(("📊 Long-run inflation expectations elevated", "#d97706",
                     f"5y5y forward at {fwd5y5y_now:.2f}% — above the threshold where "
                     "the Fed historically becomes concerned. Elevated 5y5y constrains "
                     "the Fed's ability to cut even if near-term CPI falls."))
if fwd5y5y_now and fwd5y5y_1w and abs(fwd5y5y_1w) > 0.05:
    _fwd_dir = "rising" if fwd5y5y_1w > 0 else "falling"
    _fwd_txt = ("Rapid moves higher in 5y5y precede Fed hawkish surprises."
                if fwd5y5y_1w > 0 else
                "Falling 5y5y signals de-anchoring of inflation fears — bullish for duration.")
    tensions.append((f"{'⚠️' if fwd5y5y_1w > 0 else '📉'} 5y5y moving fast",
                     "#d97706" if fwd5y5y_1w > 0 else "#1f7a4f",
                     f"5y5y forward {_fwd_dir} {abs(fwd5y5y_1w):.2f}pp this week to "
                     f"{fwd5y5y_now:.2f}%. {_fwd_txt}"))
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
# ROW 3 — REAL RATE PATH  |  BALANCE SHEET + IMPULSE
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
            fig_rr = _fig_bb(280)
            # Bloomberg-style regime bands — subtle on dark bg
            for y0, y1, fc in [(-5, 0,   "rgba(74,186,110,0.10)"),
                                (0,  0.5, "rgba(200,200,200,0.03)"),
                                (0.5,1.5, "rgba(247,148,0,0.08)"),
                                (1.5, 6,  "rgba(232,64,64,0.10)")]:
                fig_rr.add_hrect(y0=y0, y1=y1, fillcolor=fc, line_width=0)
            lo_r = float(rr_sl.min()); hi_r = float(rr_sl.max())
            pad_r = max((hi_r - lo_r) * 0.15, 0.05)
            for lvl, lbl in [(0,"ZERO"), (0.5,"NEUTRAL"), (1.5,"RESTRICTIVE")]:
                if lo_r - pad_r <= lvl <= hi_r + pad_r:
                    _bb_threshold(fig_rr, lvl, lbl, BB_GRID2)
            fig_rr.add_trace(go.Scatter(
                x=rr_sl.index, y=rr_sl.values, mode="lines", name="10Y REAL",
                line=dict(color=BB_ORANGE, width=2.0),
                fill="tozeroy", fillcolor="rgba(247,148,0,0.08)"))
            _bb_now_line(fig_rr, real_now, "NOW", BB_YELLOW)
            fig_rr.update_yaxes(range=[lo_r-pad_r, hi_r+pad_r],
                                 title_text="REAL YIELD %", title_font=dict(size=9, color=BB_SUBTEXT))
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
                                   row_heights=[0.58, 0.42], vertical_spacing=0.04)
            fa_t = fa_sl / 1e6
            fig_bs.add_trace(go.Scatter(
                x=fa_t.index, y=fa_t.values, mode="lines", name="FED ASSETS $T",
                line=dict(color=BB_BLUE, width=2),
                fill="tozeroy", fillcolor="rgba(74,143,232,0.08)"),
                row=1, col=1)
            fa_lo = float(fa_t.min()) * 0.97; fa_hi = float(fa_t.max()) * 1.02
            fig_bs.update_yaxes(range=[fa_lo, fa_hi], title_text="$T",
                                 title_font=dict(size=9, color=BB_SUBTEXT),
                                 tickfont=dict(size=9, color=BB_TEXT,
                                               family="'Courier New', monospace"),
                                 showgrid=True, gridcolor=BB_GRID, row=1, col=1)
            if not roc_sl.empty:
                bar_colors = [BB_GREEN if v >= 0 else BB_RED for v in roc_sl.values]
                fig_bs.add_trace(go.Bar(
                    x=roc_sl.index, y=roc_sl.values,
                    marker_color=bar_colors, name="13W IMPULSE %",
                    showlegend=True), row=2, col=1)
                fig_bs.add_hline(y=0, line_color=BB_GRID2, line_width=1, row=2, col=1)
                rv = roc_sl.values
                rlo = min(rv); rhi = max(rv)
                rpad = max(abs(rlo), abs(rhi)) * 0.15
                fig_bs.update_yaxes(range=[rlo-rpad, rhi+rpad], title_text="13W Δ%",
                                     title_font=dict(size=9, color=BB_SUBTEXT),
                                     tickfont=dict(size=9, color=BB_TEXT,
                                                   family="'Courier New', monospace"),
                                     showgrid=True, gridcolor=BB_GRID, row=2, col=1)
            fig_bs.update_layout(
                height=280, margin=dict(l=12, r=48, t=12, b=12),
                plot_bgcolor=BB_BG, paper_bgcolor=BB_PAPER,
                font=dict(family="'Courier New', monospace", color=BB_TEXT, size=9),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                            font=dict(size=9, color=BB_TEXT), bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1e1e1e", font_color=BB_TEXT, font_size=10),
                barmode="relative")
            fig_bs.update_xaxes(showgrid=True, gridcolor=BB_GRID,
                                 tickfont=dict(size=9, color=BB_TEXT,
                                               family="'Courier New', monospace"),
                                 tickformat="%b '%y")
            st.plotly_chart(fig_bs, width='stretch')
        else:
            st.caption("Fed balance sheet data unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — 5y5y FORWARD: three clean panels, each on its own axis
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown(
        "<div class='me-rowtitle'>5y5y forward rates — nominal · real · implied inflation</div>",
        unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    _inf_c = ("#e84040" if (fwd5y5y_now and fwd5y5y_now > 2.5) else
              ("#4aba6e" if (fwd5y5y_now and fwd5y5y_now < 2.0) else BB_ORANGE))
    _inf_lbl = ("UNANCHORED" if (fwd5y5y_now and fwd5y5y_now > 2.5) else
                ("ANCHORED"  if (fwd5y5y_now and fwd5y5y_now < 2.0) else "IN RANGE"))

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    for _col, _lbl, _val, _sub, _vc in [
        (k1, "INFL 5Y5Y",  fwd5y5y_now,       _inf_lbl,        _inf_c),
        (k2, "NOM 5Y5Y",   fwd5y5y_nom_now,    "nominal fwd",   BB_ORANGE),
        (k3, "REAL 5Y5Y",  fwd5y5y_real_now,   "real / r*",     BB_BLUE),
        (k4, "1W INFL Δ",  fwd5y5y_1w,         "infl fwd",
             "#e84040" if (fwd5y5y_1w and fwd5y5y_1w > 0) else "#4aba6e"),
        (k5, "10Y BE",     be_now,              "near-term exp", BB_SUBTEXT),
    ]:
        _v = (f"{_val:+.3f}pp" if "Δ" in _lbl and _val is not None
              else f"{_val:.3f}%" if _val is not None else "—")
        _col.markdown(
            f"<div style='padding:10px 12px;border-radius:8px;"
            f"background:#111111;border:1px solid #1e1e1e;'>"
            f"<div style='font-size:9px;font-weight:700;color:{BB_SUBTEXT};"
            f"font-family:Courier New,monospace;letter-spacing:0.8px;margin-bottom:4px;'>{_lbl}</div>"
            f"<div style='font-size:18px;font-weight:900;color:{_vc};"
            f"font-family:Courier New,monospace;'>{_v}</div>"
            f"<div style='font-size:9px;color:{BB_SUBTEXT};"
            f"font-family:Courier New,monospace;margin-top:2px;'>{_sub}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("")
    fwd_range = st.selectbox("Range", RKEYS,
                              index=RKEYS.index("2y") if "2y" in RKEYS else RKEYS.index("1y"),
                              key="fwd_range")

    has_nom  = not fwd_5y5y_nom.empty
    has_real = not fwd_5y5y_real.empty
    has_inf  = not fwd_5y5y_inf.empty

    # ── Helper: single-series Bloomberg panel ────────────────────────────────
    def _bb_panel(series, color, name, now_val, thresholds=None, fill=True):
        """Return a clean single-series Bloomberg-style figure. No annotation clutter."""
        sl = slice_series(series, fwd_range)
        if sl.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sl.index, y=sl.values,
            mode="lines", name=name,
            line=dict(color=color, width=2.2),
            fill="tozeroy" if fill else "none",
            fillcolor=f"rgba({int(color[1:3],16)},"
                      f"{int(color[3:5],16)},"
                      f"{int(color[5:7],16)},0.08)" if fill else None,
        ))
        lo = float(sl.min()); hi = float(sl.max())
        pad = max((hi - lo) * 0.25, 0.08)
        # Threshold reference lines — simple, no overlapping annotations
        if thresholds:
            for lvl, lbl, lc in thresholds:
                if lo - pad <= lvl <= hi + pad:
                    fig.add_hline(y=lvl, line_color=lc, line_width=0.7, line_dash="dash")
                    # Put label as a shape annotation positioned at chart edge
                    fig.add_annotation(x=sl.index[-1], y=lvl,
                                       text=f" {lbl}", xanchor="left",
                                       font=dict(size=8, color=lc,
                                                 family="'Courier New', monospace"),
                                       showarrow=False, xref="x", yref="y",
                                       bgcolor=BB_BG, borderpad=2)
        # Single "now" marker on the last point — cleaner than a hline annotation
        if now_val is not None:
            fig.add_trace(go.Scatter(
                x=[sl.index[-1]], y=[now_val],
                mode="markers+text",
                marker=dict(color=BB_YELLOW, size=6, symbol="circle"),
                text=[f" {now_val:.3f}%"],
                textposition="middle right",
                textfont=dict(size=9, color=BB_YELLOW,
                              family="'Courier New', monospace"),
                showlegend=False,
            ))
        fig.update_layout(
            height=190,
            margin=dict(l=10, r=70, t=6, b=6),
            plot_bgcolor=BB_BG, paper_bgcolor=BB_PAPER,
            showlegend=False, hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e1e1e", font_color=BB_TEXT, font_size=10),
            font=dict(family="'Courier New', monospace", color=BB_TEXT, size=9),
        )
        fig.update_xaxes(showgrid=True, gridcolor=BB_GRID, gridwidth=1, zeroline=False,
                         tickfont=dict(size=8, color=BB_TEXT,
                                       family="'Courier New', monospace"),
                         tickformat="%b '%y", showline=False)
        fig.update_yaxes(showgrid=True, gridcolor=BB_GRID, zeroline=False,
                         tickfont=dict(size=8, color=BB_TEXT,
                                       family="'Courier New', monospace"),
                         range=[lo - pad, hi + pad], side="right")
        return fig

    # ── Panel labels ─────────────────────────────────────────────────────────
    def _panel_label(text, color):
        st.markdown(
            f"<div style='font-size:9px;font-weight:700;color:{color};"
            f"font-family:Courier New,monospace;letter-spacing:1px;"
            f"padding:4px 0 2px 2px;'>{text}</div>",
            unsafe_allow_html=True)

    # ── Chart 1: Nominal 5y5y ────────────────────────────────────────────────
    if has_nom:
        _panel_label("NOMINAL 5Y5Y FORWARD  —  where 10y Treasury yields expected 5-10y from now",
                     BB_SUBTEXT)
        fig_nom = _bb_panel(fwd_5y5y_nom, BB_ORANGE, "NOM 5Y5Y", fwd5y5y_nom_now)
        if fig_nom:
            st.plotly_chart(fig_nom, width='stretch')

    # ── Chart 2: Implied inflation 5y5y ──────────────────────────────────────
    if has_inf:
        _panel_label("IMPLIED INFLATION 5Y5Y  —  nominal minus real · the Fed's preferred long-run gauge",
                     BB_SUBTEXT)
        fig_inf2 = _bb_panel(
            fwd_5y5y_inf, BB_ORANGE, "INFL 5Y5Y", fwd5y5y_now,
            thresholds=[(2.5, "2.5% UNANCHORED", BB_RED),
                        (2.0, "2.0% TARGET",     BB_GREEN)],
            fill=True)
        if fig_inf2:
            st.plotly_chart(fig_inf2, width='stretch')

    # ── Chart 3: Real 5y5y ───────────────────────────────────────────────────
    if has_real:
        _panel_label("REAL 5Y5Y FORWARD  —  r* expectations · real borrowing cost 5-10y out",
                     BB_SUBTEXT)
        fig_real = _bb_panel(fwd_5y5y_real, BB_BLUE, "REAL 5Y5Y", fwd5y5y_real_now,
                             fill=True)
        if fig_real:
            st.plotly_chart(fig_real, width='stretch')

    st.caption(
        "Nominal 5y5y = 2×DGS10 − DGS5 · "
        "Real 5y5y = 2×DFII10 − DFII5 · "
        "Inflation 5y5y = nominal minus real. "
        "Inflation 5y5y above 2.5% = long-run expectations unanchored — "
        "the Fed cannot ease without reigniting inflation."
    )

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — CPI + BREAKEVEN  |  DOLLAR CYCLE
# ══════════════════════════════════════════════════════════════════════════════

inf_col, liq_col = st.columns(2, gap="large")

with inf_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>CPI vs breakeven — inflation path</div>",
                    unsafe_allow_html=True)
        st.caption("Breakeven above CPI = market expecting re-acceleration. "
                   "Gap narrowing = disinflation confirmed by market.")
        inf_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="inf_range")
        fig_inf   = _fig_bb(260)
        all_inf   = []; plotted = False
        if cpi_yoy is not None and not cpi_yoy.empty:
            ci_sl = slice_series(cpi_yoy, inf_range)
            if not ci_sl.empty:
                fig_inf.add_trace(go.Scatter(x=ci_sl.index, y=ci_sl.values,
                    mode="lines", name="CPI YoY",
                    line=dict(color=BB_RED, width=2)))
                all_inf += list(ci_sl.values); plotted = True
        if not breakeven.empty:
            be_sl = slice_series(breakeven, inf_range)
            if not be_sl.empty:
                fig_inf.add_trace(go.Scatter(x=be_sl.index, y=be_sl.values,
                    mode="lines", name="10Y BREAKEVEN",
                    line=dict(color=BB_ORANGE, width=2, dash="dot")))
                all_inf += list(be_sl.values); plotted = True
        if not fwd_5y5y.empty:
            fwd_sl2 = slice_series(fwd_5y5y, inf_range)
            if not fwd_sl2.empty:
                fig_inf.add_trace(go.Scatter(x=fwd_sl2.index, y=fwd_sl2.values,
                    mode="lines", name="5Y5Y FWD",
                    line=dict(color=BB_PURPLE, width=1.6, dash="dashdot")))
                all_inf += list(fwd_sl2.values); plotted = True
        if plotted and all_inf:
            lo_i = min(all_inf); hi_i = max(all_inf)
            pad_i = max((hi_i - lo_i) * 0.15, 0.1)
            _bb_threshold(fig_inf, 2.0, "2% TARGET")
            fig_inf.update_yaxes(range=[lo_i-pad_i, hi_i+pad_i], title_text="% YoY",
                                  title_font=dict(size=9, color=BB_SUBTEXT))
            st.plotly_chart(fig_inf, width='stretch')
        else:
            st.caption("Inflation data unavailable.")

with liq_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Dollar cycle &amp; liquidity proxy</div>",
                    unsafe_allow_html=True)
        st.caption("Dollar (left) + IWM/SPY breadth (right). "
                   "All rising = genuine risk-on. Divergences are the story.")
        lq_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="lq_range")
        dl_sl = slice_series(dollar, lq_range)
        if not dl_sl.empty:
            fig_dl = _fig_bb(260)
            ma63 = dl_sl.rolling(63, min_periods=10).mean()
            fig_dl.add_trace(go.Scatter(x=dl_sl.index, y=dl_sl.values,
                mode="lines", name="DOLLAR BROAD",
                line=dict(color=BB_ORANGE, width=2)))
            if not ma63.dropna().empty:
                fig_dl.add_trace(go.Scatter(x=ma63.index, y=ma63.values,
                    mode="lines", name="63D MA",
                    line=dict(color=BB_SUBTEXT, width=1.2, dash="dash")))
            if iwm_spy is not None:
                iw_sl = slice_series(iwm_spy, lq_range)
                if not iw_sl.empty:
                    fig_dl.add_trace(go.Scatter(x=iw_sl.index, y=iw_sl.values,
                        mode="lines", name="IWM/SPY",
                        line=dict(color=BB_GREEN, width=1.4, dash="dot"),
                        yaxis="y2"))
            vals_dl = list(dl_sl.values)
            lo_d = min(vals_dl); hi_d = max(vals_dl)
            pad_d = max((hi_d-lo_d)*0.08, 0.5)
            fig_dl.update_layout(
                yaxis=dict(range=[lo_d-pad_d, hi_d+pad_d], title="DOLLAR",
                           showgrid=True, gridcolor=BB_GRID,
                           tickfont=dict(size=9, color=BB_TEXT,
                                         family="'Courier New', monospace"),
                           title_font=dict(size=9, color=BB_SUBTEXT)),
                yaxis2=dict(overlaying="y", side="left", title="IWM/SPY",
                            showgrid=False, showticklabels=True,
                            tickfont=dict(size=9, color=BB_TEXT,
                                          family="'Courier New', monospace"),
                            title_font=dict(size=9, color=BB_SUBTEXT)),
            )
            _bb_now_line(fig_dl, dollar_now, "NOW", BB_YELLOW)
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
