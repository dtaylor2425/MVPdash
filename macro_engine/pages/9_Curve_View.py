# pages/9_Curve_View.py
"""
Curve View
──────────────────────────────────────────────────────────────────────────────
The yield curve is the single most powerful macro leading indicator. This page
turns it into a daily briefing — KPIs, regime interpretation, historical
context, cross-asset confirmations, and a report-ready playbook.

Data: FRED  y2 (DGS2)  y10 (DGS10)  y3m (DGS3MO)  real10 (DFII10)
      fed_funds (FEDFUNDS)  fed_assets (WALCL)  dollar_broad (DTWEXBGS)
      cpi (CPIAUCSL)  hy_oas (BAMLH0A0HYM2)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.ranges import RANGES, slice_df, slice_series
from src.ui import inject_css, sidebar_nav, safe_switch_page

st.set_page_config(
    page_title="Curve View",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
sidebar_nav(active="Curve View")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_data():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    eq_tickers = ["SPY", "XLF", "IWM", "TLT", "HYG"]
    px = fetch_prices(eq_tickers, period="5y")
    return macro, (px if px is not None and not px.empty else pd.DataFrame())

macro, px = load_data()

RKEYS = list(RANGES.keys())

# ── Derived series ────────────────────────────────────────────────────────────

def _col(name):
    if name in macro.columns:
        return macro[name].dropna()
    return pd.Series(dtype=float, name=name)

y2_s    = _col("y2")
y10_s   = _col("y10")
y3m_s   = _col("y3m")
real10  = _col("real10")
ff_s    = _col("fed_funds")
cpi_s   = _col("cpi")
hy_s    = _col("hy_oas")
dollar  = _col("dollar_broad")

# Main spreads
curve_2_10 = (y10_s - y2_s).dropna()        # primary regime signal
curve_3m_10 = (y10_s - y3m_s).dropna()      # recession predictor (Cleveland Fed style)
breakeven   = (y10_s - real10).dropna()      # 10y inflation expectation

# Nom 10y change over recent windows
def _last(s):
    return float(s.iloc[-1]) if not s.empty else None

def _delta(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    end = s.index.max()
    prev_idx = s.index[s.index <= end - pd.Timedelta(days=days)]
    if len(prev_idx) == 0: return None
    return float(s.iloc[-1]) - float(s.loc[prev_idx[-1]])

c210_now     = _last(curve_2_10)
c3m10_now    = _last(curve_3m_10)
y10_now      = _last(y10_s)
y2_now       = _last(y2_s)
real10_now   = _last(real10)
be_now       = _last(breakeven)
ff_now       = _last(ff_s)
c210_1m      = _delta(curve_2_10, 30)
c210_3m      = _delta(curve_2_10, 90)

# CPI YoY
cpi_yoy = None
if len(cpi_s) >= 13:
    cpi_yoy_s = cpi_s.pct_change(12).dropna() * 100
    cpi_yoy = _last(cpi_yoy_s)

# Percentile rank of 2-10 spread vs 5y history
def _pct_rank(s, window=252*5):
    s = s.dropna()
    if len(s) < 60: return None
    w = s.iloc[-min(window, len(s)):]
    return float((w < s.iloc[-1]).mean() * 100)

c210_pct = _pct_rank(curve_2_10)

# ── Regime classification ─────────────────────────────────────────────────────

def curve_regime(c210, c3m10):
    """Returns (label, color, bg, description)"""
    if c210 is None:
        return "Unknown", "#6b7280", "#f3f4f6", ""

    if c210 >= 0.75:
        return ("Steep", "#1f7a4f", "#dcfce7",
                "Healthy expansion signal. Curve pricing growth and normalised conditions.")
    if c210 >= 0.0:
        return ("Flat / Mildly positive", "#6b7280", "#f3f4f6",
                "Ambiguous. Watch direction of change more than level.")
    if c210 >= -0.25:
        return ("Shallow inversion", "#d97706", "#fef9c3",
                "Early warning. Historically precedes slowdown by 6–18 months.")
    return ("Deep inversion", "#b42318", "#fee2e2",
            "Historically the most reliable US recession predictor. Lead time 12–24 months.")

crv_label, crv_color, crv_bg, crv_desc = curve_regime(c210_now, c3m10_now)

# Bear vs Bull steepener/flattener classification
def steepener_type(c210_delta, y10_delta, y2_delta):
    """Classify the *direction* of curve move over last month."""
    if c210_delta is None or y10_delta is None or y2_delta is None:
        return None, "#6b7280"
    if c210_delta > 0.05:
        if y10_delta > 0 and y2_delta < y10_delta:
            return "Bear steepener", "#d97706"   # long end selling off faster
        return "Bull steepener", "#1f7a4f"       # short end rallying faster
    if c210_delta < -0.05:
        if y2_delta > 0:
            return "Bear flattener", "#b42318"   # short end selling off faster
        return "Bull flattener", "#6b7280"       # long end rallying faster
    return "Stable", "#6b7280"

y10_1m = _delta(y10_s, 30)
y2_1m  = _delta(y2_s, 30)
move_type, move_color = steepener_type(c210_1m, y10_1m, y2_1m)

# ── Helpers ───────────────────────────────────────────────────────────────────

def kpi_card(col, label, value, sub, color="#0f172a", bg="#f8fafc"):
    col.markdown(
        f"<div style='padding:12px 14px;border-radius:12px;background:{bg};"
        f"border:1px solid rgba(0,0,0,0.07);'>"
        f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.4);"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-size:24px;font-weight:900;color:{color};line-height:1.1;'>{value}</div>"
        f"<div style='font-size:11px;color:rgba(0,0,0,0.5);margin-top:3px;'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def fmt(x, nd=2, prefix="", suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{prefix}{float(x):.{nd}f}{suffix}"

def fmt_signed(x, nd=2, suffix="pp"):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x):+.{nd}f}{suffix}"

# ── Topbar ────────────────────────────────────────────────────────────────────

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Curve View</div>
              <div class="me-subtle">
                Yield curve regime &nbsp;·&nbsp; term structure &nbsp;·&nbsp;
                steepener/flattener &nbsp;·&nbsp; cross-asset &nbsp;·&nbsp; report playbook
              </div>
            </div>
            <div style="padding:8px 16px;border-radius:20px;background:{crv_bg};">
              <span style="font-weight:800;color:{crv_color};font-size:14px;">
                {crv_label}
              </span>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5, k6 = st.columns(6, gap="medium")

# 2-10 spread
if c210_now is not None:
    k_col = crv_color; k_bg = crv_bg
else:
    k_col = "#6b7280"; k_bg = "#f3f4f6"
kpi_card(k1, "2s10s spread", fmt(c210_now, suffix="pp"), crv_label, color=k_col, bg=k_bg)

# 3m-10 spread
if c3m10_now is not None:
    k3m_col = "#b42318" if c3m10_now < -0.25 else ("#d97706" if c3m10_now < 0 else "#1f7a4f")
    k3m_bg  = "#fee2e2" if c3m10_now < -0.25 else ("#fef9c3" if c3m10_now < 0 else "#dcfce7")
else:
    k3m_col = "#6b7280"; k3m_bg = "#f3f4f6"
kpi_card(k2, "3m–10y spread", fmt(c3m10_now, suffix="pp"),
         "Recession predictor", color=k3m_col, bg=k3m_bg)

# 10y nominal
kpi_card(k3, "10y yield", fmt(y10_now, suffix="%"),
         f"1m Δ {fmt_signed(y10_1m)}", color="#1d4ed8", bg="#eff6ff")

# 10y real
if real10_now is not None:
    r_col = "#b42318" if real10_now > 1.5 else ("#d97706" if real10_now > 0.5 else "#1f7a4f")
    r_bg  = "#fee2e2" if real10_now > 1.5 else ("#fef9c3" if real10_now > 0.5 else "#dcfce7")
else:
    r_col = "#6b7280"; r_bg = "#f3f4f6"
kpi_card(k4, "10y real yield (TIPS)", fmt(real10_now, suffix="%"),
         "High = headwind for risk", color=r_col, bg=r_bg)

# Breakeven inflation
kpi_card(k5, "10y breakeven", fmt(be_now, suffix="%"),
         "10y nom − 10y real", color="#7c3aed", bg="#f5f3ff")

# 1m move type
kpi_card(k6, "1m curve move", move_type or "—",
         f"2s10s {fmt_signed(c210_1m)}", color=move_color, bg="#f8fafc")

st.markdown("")

# ── Interpretation banner ─────────────────────────────────────────────────────

def _banner(c210, c3m10, c210_1m, move_type, cpi_yoy, real10_now):
    if c210 is None:
        return "Curve data unavailable.", "#f3f4f6", "#e2e8f0"

    lines = []

    # Level read
    if c210 >= 0.75:
        lines.append(
            f"**Curve is steep at {c210:+.2f}pp.** The 2s10s spread is in expansion territory. "
            "Term premium is positive and the market is pricing in future growth above current short rates. "
            "This environment historically supports cyclicals, financials (XLF), and credit."
        )
    elif c210 >= 0:
        lines.append(
            f"**Curve is flat-to-positive at {c210:+.2f}pp.** Ambiguous signal — watch the direction "
            "of change more than the level. A flat curve after a hiking cycle can precede either "
            "re-steepening (soft landing) or inversion (hard landing)."
        )
    elif c210 >= -0.25:
        lines.append(
            f"**Shallow inversion at {c210:+.2f}pp.** The 2-year yield exceeds the 10-year, "
            "signalling the market believes the Fed will need to cut rates. This regime has "
            "preceded US recessions by 6–18 months on average. Not a timing signal — but a risk flag."
        )
    else:
        lines.append(
            f"**Deep inversion at {c210:+.2f}pp.** The most historically reliable US recession signal, "
            "with a lead time of 12–24 months. The 3m–10y spread ({c3m10:+.2f}pp) "
            "is the version the Fed watches most closely."
            if c3m10 is not None else
            f"**Deep inversion at {c210:+.2f}pp.** Historically the strongest recession precursor."
        )

    # Momentum
    if move_type and c210_1m is not None:
        if "bear steepener" in (move_type or "").lower():
            lines.append(
                f"**Bear steepening in progress ({c210_1m:+.2f}pp over 1m).** "
                "The long end is selling off faster than the short end — typically driven by "
                "inflation fears or rising term premium. Negative for long-duration bonds (TLT). "
                "Can be positive for banks (XLF) if driven by growth expectations."
            )
        elif "bull steepener" in (move_type or "").lower():
            lines.append(
                f"**Bull steepening in progress ({c210_1m:+.2f}pp over 1m).** "
                "Short rates are falling (Fed easing expectations) while the long end holds. "
                "Historically bullish for risk assets — the classic 'green light' for equities."
            )
        elif "bear flattener" in (move_type or "").lower():
            lines.append(
                f"**Bear flattening in progress ({c210_1m:+.2f}pp over 1m).** "
                "Short rates are rising faster than long rates — the market is pricing Fed tightening "
                "but not yet growth. This is the regime that historically precedes inversion."
            )
        elif "bull flattener" in (move_type or "").lower():
            lines.append(
                f"**Bull flattening in progress ({c210_1m:+.2f}pp over 1m).** "
                "Long end rallying (growth fears) while short rates hold. Typically risk-off."
            )

    # CPI context
    if cpi_yoy is not None and real10_now is not None:
        if cpi_yoy > 3.5 and c210 < 0:
            lines.append(
                f"⚠️ **Stagflation risk.** CPI YoY at {cpi_yoy:.1f}% while curve inverted — "
                "the Fed faces a policy dilemma. Historically poor for both stocks and bonds."
            )
        elif cpi_yoy < 2.5 and real10_now > 1.5:
            lines.append(
                f"Real yields ({real10_now:.2f}%) are restrictive relative to inflation ({cpi_yoy:.1f}% YoY). "
                "Financial conditions are tight. Watch for credit stress to follow."
            )

    banner_col = crv_bg
    banner_border = crv_color + "55"
    return " ".join(lines), banner_col, banner_border

banner_text, banner_bg, banner_border = _banner(
    c210_now, c3m10_now, c210_1m, move_type, cpi_yoy, real10_now
)
st.markdown(
    f"<div style='padding:14px 18px;border-radius:14px;background:{banner_bg};"
    f"border:1px solid {banner_border};margin-bottom:20px;font-size:13px;"
    f"color:rgba(0,0,0,0.82);line-height:1.7;'>"
    f"{banner_text}</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — CURVE CHART + TERM STRUCTURE SNAPSHOT TABLE
# ══════════════════════════════════════════════════════════════════════════════

chart_col, snap_col = st.columns([1.7, 1.0], gap="medium")

with chart_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>2s10s & 3m10y spreads</div>",
                    unsafe_allow_html=True)
        st.caption("2s10s is the market signal. 3m10y is the Fed's preferred recession predictor.")

        crv_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="crv_main_range")
        c210_sl   = slice_series(curve_2_10, crv_range)
        c3m10_sl  = slice_series(curve_3m_10, crv_range)

        if not c210_sl.empty:
            fig = go.Figure()

            # Dynamic y-axis: pad 15% around the actual data range
            all_vals = list(c210_sl.values)
            if not c3m10_sl.empty:
                all_vals += list(c3m10_sl.values)
            _lo = min(all_vals); _hi = max(all_vals)
            _pad = max((_hi - _lo) * 0.15, 0.05)
            y_lo = _lo - _pad; y_hi = _hi + _pad

            # Regime bands — clipped to the visible range
            BANDS = [
                (min(y_lo, -5), -0.25, "rgba(180,35,24,0.06)"),
                (-0.25,          0,    "rgba(217,119,6,0.05)"),
                (0,              0.75, "rgba(107,114,128,0.03)"),
                (0.75,  max(y_hi, 5),  "rgba(31,122,79,0.06)"),
            ]
            for b0, b1, bg in BANDS:
                if b1 > y_lo and b0 < y_hi:  # only draw if band is in view
                    fig.add_hrect(y0=max(b0, y_lo), y1=min(b1, y_hi),
                                  fillcolor=bg, line_width=0)

            # Threshold lines — only draw if they fall within the visible axis
            for y_val, lbl in [(0, "Inversion"), (-0.25, "Shallow inv."), (0.75, "Steep")]:
                if y_lo <= y_val <= y_hi:
                    fig.add_hline(y=y_val, line_dash="dash", line_color="#cbd5e1",
                                  line_width=1, annotation_text=lbl,
                                  annotation_position="right",
                                  annotation_font_size=10)

            fig.add_trace(go.Scatter(
                x=c210_sl.index, y=c210_sl.values,
                mode="lines", name="2s10s",
                line=dict(color="#1d4ed8", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(29,78,216,0.07)",
            ))
            if not c3m10_sl.empty:
                c3m10_common = c3m10_sl.loc[c3m10_sl.index.isin(c210_sl.index)]
                fig.add_trace(go.Scatter(
                    x=c3m10_common.index, y=c3m10_common.values,
                    mode="lines", name="3m10y",
                    line=dict(color="#dc2626", width=1.5, dash="dot"),
                ))

            fig.add_hline(y=c210_now, line_color=crv_color, line_width=1.5,
                          annotation_text=f"Now: {c210_now:+.2f}pp",
                          annotation_position="right",
                          annotation_font_color=crv_color)

            fig.update_layout(
                height=320, margin=dict(l=10, r=100, t=20, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                hovermode="x unified", yaxis_title="Spread (pp)",
                yaxis=dict(range=[y_lo, y_hi]),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("Curve data unavailable.")

with snap_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Term structure snapshot</div>",
                    unsafe_allow_html=True)

        snap_items = [
            ("3m yield",   _last(y3m_s),  "%"),
            ("2y yield",   y2_now,        "%"),
            ("10y yield",  y10_now,       "%"),
            ("10y real",   real10_now,    "%"),
            ("Breakeven",  be_now,        "%"),
            ("Fed funds",  ff_now,        "%"),
            ("2s10s",      c210_now,      "pp"),
            ("3m10y",      c3m10_now,     "pp"),
            ("1m Δ 2s10s", c210_1m,       "pp"),
            ("3m Δ 2s10s", c210_3m,       "pp"),
        ]
        for label, val, unit in snap_items:
            if val is None: continue
            is_delta = "Δ" in label
            val_str  = fmt_signed(val, suffix=unit) if is_delta else fmt(val, suffix=unit)
            if is_delta:
                v_color = "#1f7a4f" if val > 0 else "#b42318"
            elif label in ("2s10s", "3m10y"):
                v_color = crv_color if label == "2s10s" else (
                    "#b42318" if (val < -0.25) else ("#d97706" if val < 0 else "#1f7a4f"))
            elif label in ("10y real",):
                v_color = "#b42318" if val > 1.5 else ("#d97706" if val > 0.5 else "#1f7a4f")
            else:
                v_color = "rgba(0,0,0,0.80)"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid rgba(0,0,0,0.05);font-size:13px;'>"
                f"<span style='color:rgba(0,0,0,0.55);'>{label}</span>"
                f"<span style='font-weight:800;color:{v_color};'>{val_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.caption(
            "Steep (≥0.75pp) · Flat (0–0.75pp) · "
            "Shallow inv. (–0.25 to 0pp) · Deep inv. (<–0.25pp)"
        )

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — NOMINAL VS REAL + BREAKEVEN (components of the long end)
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Decomposing the 10y — nominal, real, breakeven</div>",
                unsafe_allow_html=True)
    st.caption(
        "10y nominal = 10y real yield + breakeven inflation. "
        "A rising 10y driven by real yields is tighter financial conditions. "
        "A rising 10y driven by breakeven is an inflation-expectation story — different implications."
    )

    decomp_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="decomp_range")
    y10_sl  = slice_series(y10_s,   decomp_range)
    r10_sl  = slice_series(real10,  decomp_range)
    be_sl   = slice_series(breakeven, decomp_range)

    if not y10_sl.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y10_sl.index, y=y10_sl.values,
            mode="lines", name="10y nominal",
            line=dict(color="#1d4ed8", width=2),
        ))
        if not r10_sl.empty:
            r10_common = r10_sl.reindex(y10_sl.index, method="ffill")
            fig2.add_trace(go.Scatter(
                x=r10_common.index, y=r10_common.values,
                mode="lines", name="10y real (TIPS)",
                line=dict(color="#dc2626", width=2, dash="dot"),
            ))
        if not be_sl.empty:
            be_common = be_sl.reindex(y10_sl.index, method="ffill")
            fig2.add_trace(go.Scatter(
                x=be_common.index, y=be_common.values,
                mode="lines", name="Breakeven inflation",
                line=dict(color="#d97706", width=1.8, dash="dash"),
            ))
        fig2.add_hline(y=0, line_color="#94a3b8", line_width=1)
        # Dynamic y-axis
        _vals2 = list(y10_sl.values)
        if not r10_sl.empty: _vals2 += list(r10_sl.dropna().values)
        if not be_sl.empty:  _vals2 += list(be_sl.dropna().values)
        _lo2 = min(_vals2); _hi2 = max(_vals2)
        _pad2 = max((_hi2 - _lo2) * 0.12, 0.1)
        fig2.update_layout(
            height=300, margin=dict(l=10, r=20, t=20, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Yield (%)",
            yaxis=dict(range=[_lo2 - _pad2, _hi2 + _pad2]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig2, width='stretch')
    else:
        st.caption("Yield data unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — STEEPENER / FLATTENER HISTORY  +  CURVE vs CREDIT DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

steep_col, div_col = st.columns(2, gap="medium")

with steep_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Steepening / flattening momentum</div>",
                    unsafe_allow_html=True)
        st.caption("63-day rate of change of the 2s10s spread. Positive = steepening. "
                   "The *sign* tells you direction; the *level* tells you how fast.")

        steep_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="steep_range")
        c210_full   = slice_series(curve_2_10, steep_range)

        if len(c210_full) >= 64:
            roc63 = c210_full.diff(63).dropna()
            fig3  = go.Figure()

            # Color bars by sign
            bar_colors = ["#1f7a4f" if v >= 0 else "#b42318" for v in roc63.values]
            fig3.add_trace(go.Bar(
                x=roc63.index, y=roc63.values,
                marker_color=bar_colors,
                name="63d Δ 2s10s",
            ))
            fig3.add_hline(y=0, line_color="#94a3b8", line_width=1.5)
            _lo3 = min(roc63.values); _hi3 = max(roc63.values)
            _pad3 = max(abs(_lo3), abs(_hi3)) * 0.15
            fig3.update_layout(
                height=280, margin=dict(l=10, r=20, t=20, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False, hovermode="x unified",
                yaxis_title="63d change (pp)",
                yaxis=dict(range=[_lo3 - _pad3, _hi3 + _pad3]),
            )
            fig3.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig3, width='stretch')

            # Current regime annotation
            latest_roc = float(roc63.iloc[-1])
            roc_dir    = "steepening" if latest_roc > 0 else "flattening"
            roc_color  = "#1f7a4f" if latest_roc > 0 else "#b42318"
            st.markdown(
                f"<div style='font-size:12px;color:{roc_color};font-weight:700;margin-top:4px;'>"
                f"Currently {roc_dir} at {latest_roc:+.2f}pp over 63 days</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Not enough data for 63d rate of change.")

with div_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Curve vs credit spreads divergence</div>",
                    unsafe_allow_html=True)
        st.caption("Curve and HY OAS usually move together — both reflect growth/risk appetite. "
                   "When they diverge, one market isn't believing the other. "
                   "Watch for the lagging market to catch up.")

        div_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="div_range")
        c210_dv   = slice_series(curve_2_10, div_range)
        hy_dv     = slice_df(macro, div_range)["hy_oas"].dropna() \
                    if "hy_oas" in macro.columns else pd.Series(dtype=float)

        if not c210_dv.empty and not hy_dv.empty:
            common = c210_dv.index.intersection(hy_dv.index)
            if len(common) > 10:
                fig4 = make_subplots(specs=[[{"secondary_y": True}]])
                fig4.add_trace(go.Scatter(
                    x=c210_dv.loc[common].index,
                    y=c210_dv.loc[common].values,
                    mode="lines", name="2s10s (pp)",
                    line=dict(color="#1d4ed8", width=2),
                ), secondary_y=False)
                fig4.add_trace(go.Scatter(
                    x=hy_dv.loc[common].index,
                    y=hy_dv.loc[common].values,
                    mode="lines", name="HY OAS (%)",
                    line=dict(color="#dc2626", width=1.8, dash="dot"),
                ), secondary_y=True)
                fig4.add_hline(y=0, line_color="#94a3b8", line_width=1,
                               secondary_y=False)
                fig4.update_layout(
                    height=280, margin=dict(l=10, r=20, t=20, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    hovermode="x unified",
                )
                fig4.update_yaxes(title_text="2s10s (pp)", secondary_y=False,
                                  showgrid=True, gridcolor="#f1f5f9")
                fig4.update_yaxes(title_text="HY OAS (%)", secondary_y=True,
                                  showgrid=False)
                fig4.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig4, width='stretch')
        else:
            st.caption("Curve or HY OAS data unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — CROSS-ASSET CONFIRMATIONS
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Cross-asset confirmations</div>",
                unsafe_allow_html=True)
    st.caption(
        "The curve's regime should be reflected across asset classes. "
        "Divergences are the most interesting things to write about."
    )

    def _pct_ret(ticker, days):
        if px.empty or ticker not in px.columns:
            return None
        s = px[ticker].dropna()
        if len(s) < days + 2: return None
        end = s.index.max()
        prev_idx = s.index[s.index <= end - pd.Timedelta(days=days)]
        if len(prev_idx) == 0: return None
        return float(s.iloc[-1] / s.loc[prev_idx[-1]] - 1) * 100

    CROSS = [
        ("TLT",  "Long duration bonds",
         "Inverse of long yields. Falling TLT confirms bear steepening or curve pricing rate hikes.",
         False),
        ("XLF",  "Financials",
         "Banks profit from steeper curves (borrow short, lend long). XLF rising with curve = confirmation.",
         True),
        ("IWM",  "Small caps",
         "Rate-sensitive domestics. Hurt by tight credit from inversions; benefit from steepening.",
         True),
        ("HYG",  "High yield credit",
         "HY spreads should confirm curve stress. HYG falling while curve inverted = double warning.",
         True),
        ("SPY",  "S&P 500",
         "Equities can rally even with inversion initially. Divergence narrows as recession risk rises.",
         True),
    ]

    conf_cols = st.columns(5, gap="medium")
    for i, (ticker, name, note, up_good) in enumerate(CROSS):
        with conf_cols[i]:
            ret_1m = _pct_ret(ticker, 21)
            ret_3m = _pct_ret(ticker, 63)

            if ret_1m is None:
                status_color, status_bg, status_txt = "#6b7280", "#f3f4f6", "N/A"
            else:
                positive = ret_1m > 1
                if up_good:
                    status_color = "#1f7a4f" if positive else "#b42318"
                    status_bg    = "#dcfce7" if positive else "#fee2e2"
                    status_txt   = "Confirming ✓" if positive else "Diverging ✗"
                else:  # TLT — falling TLT = bear steepener, may or may not be bad
                    status_color = "#6b7280"; status_bg = "#f3f4f6"
                    status_txt   = f"{ret_1m:+.1f}% 1m"

            ret1m_str = f"{ret_1m:+.1f}%" if ret_1m is not None else "—"
            ret3m_str = f"{ret_3m:+.1f}%" if ret_3m is not None else "—"

            st.markdown(
                f"<div style='padding:12px;border-radius:12px;background:{status_bg};"
                f"border:1px solid rgba(0,0,0,0.06);height:100%;'>"
                f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.40);"
                f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>{ticker}</div>"
                f"<div style='font-size:13px;font-weight:800;color:rgba(0,0,0,0.85);"
                f"margin-bottom:6px;'>{name}</div>"
                f"<div style='font-size:20px;font-weight:900;color:{status_color};'>"
                f"{ret1m_str}</div>"
                f"<div style='font-size:11px;color:rgba(0,0,0,0.45);margin-top:2px;'>"
                f"3m: {ret3m_str}</div>"
                f"<div style='font-size:11px;font-weight:700;color:{status_color};"
                f"margin-top:4px;'>{status_txt}</div>"
                f"<div style='font-size:10px;color:rgba(0,0,0,0.40);margin-top:6px;"
                f"line-height:1.4;'>{note}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6 — PLAYBOOK CARDS
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Curve regime playbook</div>",
                unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4, gap="medium")

    def playbook_card(col, regime, trigger, implication, watch, report_angle, color, bg):
        is_active = crv_label.lower() in regime.lower() or regime.lower() in crv_label.lower()
        border = f"2px solid {color}" if is_active else "1px solid rgba(0,0,0,0.08)"
        badge  = (
            f"<span style='font-size:10px;font-weight:700;color:{color};"
            f"background:{bg};padding:2px 7px;border-radius:8px;margin-left:6px;'>CURRENT</span>"
            if is_active else ""
        )
        col.markdown(
            f"<div style='padding:12px 14px;border-radius:12px;background:{bg};"
            f"border:{border};height:100%;'>"
            f"<div style='font-weight:800;font-size:13px;color:{color};margin-bottom:10px;'>"
            f"{regime}{badge}</div>"
            f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.40);"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:2px;'>Trigger</div>"
            f"<div style='font-size:12px;color:rgba(0,0,0,0.72);margin-bottom:8px;'>{trigger}</div>"
            f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.40);"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:2px;'>Implication</div>"
            f"<div style='font-size:12px;color:rgba(0,0,0,0.72);margin-bottom:8px;'>{implication}</div>"
            f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.40);"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:2px;'>Watch for</div>"
            f"<div style='font-size:12px;color:rgba(0,0,0,0.72);margin-bottom:8px;'>{watch}</div>"
            f"<div style='font-size:10px;font-weight:700;color:{color};"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:2px;'>Report angle</div>"
            f"<div style='font-size:12px;color:{color};font-weight:600;'>{report_angle}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    playbook_card(
        p1,
        regime="Steep (≥0.75pp)",
        trigger="2s10s ≥ 0.75pp, long end above short",
        implication="Expansion. Banks earn on carry. Credit accessible. Cycle mid-to-early.",
        watch="Curve starting to flatten = Fed hiking or late-cycle signal",
        report_angle="Overweight cyclicals, XLF, IWM. Underweight long duration TLT.",
        color="#1f7a4f", bg="#dcfce7",
    )
    playbook_card(
        p2,
        regime="Flat / Mildly positive",
        trigger="2s10s 0–0.75pp",
        implication="Ambiguous. Late cycle or early pivot. Direction of change is the signal.",
        watch="Bear vs bull steepener/flattener — see momentum chart above",
        report_angle="Monitor 3m10y closely. Reduce duration risk. Quality over beta.",
        color="#6b7280", bg="#f3f4f6",
    )
    playbook_card(
        p3,
        regime="Shallow inversion",
        trigger="2s10s –0.25 to 0pp",
        implication="Market pricing Fed cuts eventually. Recession risk rising. 6–18m lead time.",
        watch="3m10y spread — if this also inverts, signal strengthens significantly",
        report_angle="Start adding duration (TLT). Reduce credit exposure. Raise cash.",
        color="#d97706", bg="#fef9c3",
    )
    playbook_card(
        p4,
        regime="Deep inversion",
        trigger="2s10s < –0.25pp",
        implication="Historically the strongest recession signal. 12–24m lead time average.",
        watch="Re-steepening after deep inversion = historically the actual recession onset",
        report_angle="Defensive positioning. Max duration. Quality credit. GLD as hedge.",
        color="#b42318", bg="#fee2e2",
    )

st.markdown("")
st.caption(
    "2s10s = 10-year minus 2-year Treasury yield. "
    "3m10y = 10-year minus 3-month Treasury yield (Fed preferred). "
    "Breakeven = 10y nominal minus 10y TIPS real yield. "
    "Data: FRED DGS2, DGS10, DGS3MO, DFII10, FEDFUNDS."
)
st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)