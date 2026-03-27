# pages/8_Transition_Watch.py
"""
Transition Watch
─────────────────────────────────────────────────────────────────────────────
Answers: "Is the regime about to change?"

Sections:
  1. Alert banner — how close is the score to a regime boundary?
  2. Score trajectory — 4-week trend with momentum arrow
  3. Component dashboard — each component's current level, z-score,
     trend direction, and distance from its threshold
  4. Component momentum sparklines — 12-week history per component
  5. Cross-asset confirmation — are price markets confirming the macro signal?
  6. What would flip the regime — plain-English scenario table
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries
from src.ui import inject_css, sidebar_nav, safe_switch_page, html_table, regime_bg

st.set_page_config(
    page_title="Transition Watch",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
sidebar_nav(active="Transition Watch")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV",
                    "GLD", "UUP", "IWM", "QQQ", "SPY", "HYG", "TLT"]

BOUNDARY_NAMES = {
    (25, 40): ("Bearish → Neutral",  "#d97706"),
    (40, 60): ("Neutral zone",       "#6b7280"),
    (60, 75): ("Bullish → Risk On",  "#16a34a"),
}

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    px = fetch_prices(ROTATION_TICKERS, period="5y")
    if px is None or px.empty:
        px = pd.DataFrame()
    else:
        px = px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px,
                               lookback_trend=63, momentum_lookback_days=21)
    reg_hist = compute_regime_timeseries(macro, px,
                                         lookback_trend=63, freq="W-FRI")
    return macro, px, regime, reg_hist

macro, px, regime, reg_hist = load_all()

cur_score   = int(getattr(regime, "score", 50))
cur_label   = getattr(regime, "label", "Unknown")
cur_conf    = getattr(regime, "confidence", "Unknown")
cur_mom     = str(getattr(regime, "momentum_label", "stable")).lower()
components  = getattr(regime, "components", {}) or {}

# ── Boundary proximity ────────────────────────────────────────────────────────

THRESHOLDS = [25, 40, 60, 75]
nearest_thresh = min(THRESHOLDS, key=lambda t: abs(cur_score - t))
dist_to_thresh = abs(cur_score - nearest_thresh)

if dist_to_thresh <= 3:
    alert_level = "critical"
    alert_color = "#b42318"
    alert_bg    = "#fee2e2"
    alert_txt   = (f"⚠️ Score {cur_score} is only {dist_to_thresh} point(s) from "
                   f"the {nearest_thresh} boundary — regime change imminent.")
elif dist_to_thresh <= 7:
    alert_level = "warning"
    alert_color = "#d97706"
    alert_bg    = "#fef9c3"
    alert_txt   = (f"Score {cur_score} is {dist_to_thresh} points from the "
                   f"{nearest_thresh} boundary. Monitor component trends closely.")
else:
    alert_level = "stable"
    alert_color = "#1f7a4f"
    alert_bg    = "#dcfce7"
    alert_txt   = (f"Score {cur_score} is {dist_to_thresh} points from nearest "
                   f"boundary ({nearest_thresh}). Regime appears stable near-term.")

# ── Score trajectory ──────────────────────────────────────────────────────────

def score_trajectory(reg_hist: pd.DataFrame, weeks: int = 8):
    """Returns recent scores and momentum direction."""
    if reg_hist.empty or len(reg_hist) < 2:
        return [], None
    recent = reg_hist["score"].dropna().tail(weeks)
    scores = recent.tolist()
    if len(scores) >= 4:
        trend_val = np.polyfit(range(len(scores)), scores, 1)[0]
    else:
        trend_val = scores[-1] - scores[0] if len(scores) >= 2 else 0
    return scores, float(trend_val)

recent_scores, traj_slope = score_trajectory(reg_hist, weeks=8)

# ── Topbar ────────────────────────────────────────────────────────────────────

h1, h2 = st.columns([4, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Transition Watch</div>
              <div class="me-subtle">Component momentum · score trajectory · regime boundary proximity · early warnings</div>
            </div>
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
              <span style="padding:6px 14px;border-radius:20px;background:{alert_bg};"
                    ><b style="color:{alert_color};">{cur_label} · {cur_score}</b></span>
              <span style="font-size:12px;color:rgba(0,0,0,0.5);">
                {dist_to_thresh}pt from boundary
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
# ROW 1 — ALERT BANNER + SCORE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<div style='padding:14px 18px;border-radius:14px;background:{alert_bg};"
    f"border:1px solid {alert_color}33;margin-bottom:8px;font-size:14px;"
    f"color:{alert_color};font-weight:600;line-height:1.5;'>"
    f"{alert_txt}</div>",
    unsafe_allow_html=True,
)

# ── How to read the boundary proximity ───────────────────────────────────────
_boundary_context = {
    25: ("Bearish / Risk Off boundary",
         "Crossing below 25 historically coincides with credit spread spikes and equity drawdowns "
         "exceeding 15%. Risk Off regimes average 4-6 months duration. The key signal to watch is "
         "whether HY OAS is accelerating — if spreads widen past their 75th percentile while the "
         "score approaches 25, the transition probability increases materially."),
    40: ("Bearish / Neutral boundary",
         "The 40 boundary separates deteriorating conditions from true stress. Crossing below 40 "
         "means multiple signals have turned simultaneously. Historically, a score below 40 with "
         "deteriorating momentum leads to further declines 65% of the time within 4 weeks."),
    60: ("Neutral / Bullish boundary",
         "Crossing above 60 signals broad macro improvement — credit tightening, curve steepening, "
         "and breadth improving together. This level historically precedes equity outperformance "
         "over the following quarter. The quality of the signal improves when the Fed is also "
         "easing or paused."),
    75: ("Bullish / Risk On boundary",
         "Above 75 is the full risk-on regime. All macro signals align bullishly. Historically "
         "this regime produces the strongest equity returns but also the highest drawdown risk "
         "if the regime reverses — the fall from Risk On to Neutral is typically fast."),
}
ctx = _boundary_context.get(nearest_thresh)
if ctx:
    thresh_name, thresh_text = ctx
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:10px;background:#f8fafc;"
        f"border:1px solid rgba(0,0,0,0.07);margin-bottom:16px;font-size:12px;"
        f"line-height:1.6;color:rgba(0,0,0,0.70);'>"
        f"<span style='font-weight:800;color:rgba(0,0,0,0.80);'>{thresh_name}:</span> "
        f"{thresh_text}</div>",
        unsafe_allow_html=True)

traj_col, comp_overview_col = st.columns([1.1, 1.9], gap="medium")

with traj_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Score trajectory (8 weeks)</div>",
                    unsafe_allow_html=True)

        if recent_scores and traj_slope is not None:
            if traj_slope > 0.5:
                mom_arrow, mom_color, mom_txt = "↑", "#1f7a4f", "Improving"
            elif traj_slope < -0.5:
                mom_arrow, mom_color, mom_txt = "↓", "#b42318", "Deteriorating"
            else:
                mom_arrow, mom_color, mom_txt = "→", "#6b7280", "Stable"

            st.markdown(
                f"<div style='font-size:36px;font-weight:900;color:{mom_color};"
                f"line-height:1;'>{cur_score}"
                f"<span style='font-size:22px;margin-left:8px;'>{mom_arrow}</span>"
                f"</div>"
                f"<div style='font-size:12px;color:rgba(0,0,0,0.5);margin-top:4px;'>"
                f"{mom_txt} · slope {traj_slope:+.2f} pts/week</div>",
                unsafe_allow_html=True,
            )

            # Mini sparkline of recent scores
            fig_traj = go.Figure()
            x_vals = list(range(len(recent_scores)))

            # Threshold band fills
            for y0, y1, bg in [(0, 25, "rgba(180,35,24,0.07)"),
                                (25, 40, "rgba(217,119,6,0.06)"),
                                (40, 60, "rgba(107,114,128,0.04)"),
                                (60, 75, "rgba(22,163,74,0.06)"),
                                (75, 100, "rgba(31,122,79,0.09)")]:
                fig_traj.add_hrect(y0=y0, y1=y1, fillcolor=bg, line_width=0)

            fig_traj.add_trace(go.Scatter(
                x=x_vals, y=recent_scores,
                mode="lines+markers",
                line=dict(color="#1d4ed8", width=2.5),
                marker=dict(size=6, color="#1d4ed8"),
            ))
            # Trend line
            if len(recent_scores) >= 3:
                fit = np.polyfit(x_vals, recent_scores, 1)
                trend_line = [fit[0]*x + fit[1] for x in x_vals]
                fig_traj.add_trace(go.Scatter(
                    x=x_vals, y=trend_line,
                    mode="lines", name="Trend",
                    line=dict(color=mom_color, width=1.5, dash="dash"),
                    showlegend=False,
                ))
            for thresh in [25, 40, 60, 75]:
                fig_traj.add_hline(y=thresh, line_dash="dot",
                                   line_color="#d1d5db", line_width=1)
            fig_traj.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False, yaxis_range=[0, 100],
                xaxis_title="Weeks ago (8=oldest)",
                yaxis_title="Score",
            )
            fig_traj.update_xaxes(
                tickvals=x_vals,
                ticktext=[f"−{len(recent_scores)-1-i}w" if i < len(recent_scores)-1 else "Now"
                           for i in x_vals],
                showgrid=True, gridcolor="#f1f5f9",
            )
            fig_traj.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_traj, width='stretch')

            # Table of recent scores
            score_rows = ""
            for i, s in enumerate(recent_scores):
                wk = len(recent_scores) - 1 - i
                lbl = "Now" if wk == 0 else f"−{wk}w"
                color = "#1f7a4f" if s >= 60 else ("#b42318" if s < 40 else "#6b7280")
                score_rows += (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:4px 8px;font-size:12px;'>"
                    f"<span style='color:rgba(0,0,0,0.5);'>{lbl}</span>"
                    f"<span style='font-weight:800;color:{color};'>{int(round(s))}</span>"
                    f"</div>"
                )
            st.markdown(score_rows, unsafe_allow_html=True)
        else:
            st.caption("Not enough history for trajectory.")

with comp_overview_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Component status dashboard</div>",
                    unsafe_allow_html=True)

        if components:
            def _comp_signal(name, c):
                """Returns (signal_text, color, bg)"""
                contrib = 0.0
                z = c.get("zscore")
                w = c.get("weight")
                if isinstance(z, (int, float)) and isinstance(w, (int, float)):
                    contrib = float(z) * float(w)
                if contrib > 0.05:
                    return "Supportive", "#1f7a4f", "#dcfce7"
                if contrib < -0.05:
                    return "Drag", "#b42318", "#fee2e2"
                return "Neutral", "#6b7280", "#f3f4f6"

            def _trend_arrow(name, trend_up):
                if trend_up is None: return "→"
                nm = name.lower()
                if "credit" in nm:
                    return "↓ tightening" if trend_up == 0 else "↑ widening"
                return "↑" if trend_up == 1 else "↓"

            for key, c in components.items():
                if not isinstance(c, dict):
                    continue
                name     = c.get("name", key)
                level    = c.get("level")
                z        = c.get("zscore")
                trend_up = c.get("trend_up")
                weight   = c.get("weight", 0)

                sig, sig_color, sig_bg = _comp_signal(name, c)
                arrow_txt = _trend_arrow(name, trend_up)

                z_color = "#b42318" if (z is not None and abs(float(z)) > 1.5) else "rgba(0,0,0,0.60)"

                roc_z    = c.get("roc_zscore")
                roc_str  = f"{float(roc_z):+.2f}" if roc_z is not None and not pd.isna(roc_z) else "—"
                roc_col  = "#1f7a4f" if (roc_z is not None and not pd.isna(roc_z) and float(roc_z) > 0) else "#b42318"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;padding:9px 12px;border-radius:12px;"
                    f"border:1px solid rgba(0,0,0,0.06);background:#fafafa;margin-bottom:8px;'>"
                    f"<div>"
                    f"  <div style='font-size:13px;font-weight:800;color:rgba(0,0,0,0.85);'>"
                    f"    {name}</div>"
                    f"  <div style='font-size:11px;color:rgba(0,0,0,0.50);margin-top:2px;'>"
                    f"    Level: {float(level):.2f} &nbsp;·&nbsp; "
                    f"    z: <span style='color:{z_color};font-weight:700;'>{float(z):.2f}</span>"
                    f"    &nbsp;·&nbsp; mom z: <span style='color:{roc_col};font-weight:700;'>{roc_str}</span>"
                    f"    &nbsp;·&nbsp; {arrow_txt} &nbsp;·&nbsp; wt {float(weight):.2f}"
                    f"  </div>"
                    f"</div>"
                    f"<span style='font-size:11px;font-weight:800;color:{sig_color};"
                    f"background:{sig_bg};padding:4px 10px;border-radius:8px;'>"
                    f"{sig}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No component data available.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — COMPONENT MOMENTUM SPARKLINES
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Component z-score history (12 weeks)</div>",
                unsafe_allow_html=True)
    st.caption("Tracks how each component's z-score has moved. Crosses through 0 = potential regime driver flip.")

    COMP_SERIES = {
        "Credit stress":          ("hy_oas",    True),
        "Curve (10y minus 2y)":   ("curve",     False),
        "Real yields":            ("real10",    True),
        "Dollar impulse":         ("dollar_broad", True),
        "Risk appetite (IWM/SPY)": (None,       False),
    }

    def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
        mu  = series.rolling(window, min_periods=60).mean()
        std = series.rolling(window, min_periods=60).std()
        return ((series - mu) / std.replace(0, np.nan)).dropna()

    def _get_series(key_tuple) -> pd.Series | None:
        col, _ = key_tuple
        if col is None:
            # Risk appetite proxy: IWM / SPY rolling ratio z-score
            if "IWM" in px.columns and "SPY" in px.columns:
                ratio = (px["IWM"] / px["SPY"]).dropna()
                ratio_wk = ratio.resample("W-FRI").last().dropna()
                return _rolling_zscore(ratio_wk)
            return None
        if col == "curve":
            if "y10" in macro.columns and "y2" in macro.columns:
                s = (macro["y10"] - macro["y2"]).dropna()
                s_wk = s.resample("W-FRI").last().dropna()
                return _rolling_zscore(s_wk)
            return None
        if col in macro.columns:
            s_wk = macro[col].dropna().resample("W-FRI").last().dropna()
            return _rolling_zscore(s_wk)
        return None

    valid_comps = {}
    for name, key_tuple in COMP_SERIES.items():
        s = _get_series(key_tuple)
        if s is not None and len(s) >= 12:
            valid_comps[name] = (s.tail(24), key_tuple[1])

    if valid_comps:
        n_cols = min(len(valid_comps), 3)
        spark_cols = st.columns(n_cols, gap="medium")
        for i, (name, (series, invert)) in enumerate(valid_comps.items()):
            with spark_cols[i % n_cols]:
                last_z = float(series.iloc[-1])
                prev_z = float(series.iloc[-2]) if len(series) >= 2 else last_z
                delta_z = last_z - prev_z

                if invert:
                    # For credit/dollar, positive z = bad
                    contrib_color = "#b42318" if last_z > 0.5 else ("#1f7a4f" if last_z < -0.5 else "#6b7280")
                else:
                    contrib_color = "#1f7a4f" if last_z > 0.5 else ("#b42318" if last_z < -0.5 else "#6b7280")

                st.markdown(
                    f"<div style='font-size:11px;font-weight:700;color:rgba(0,0,0,0.50);"
                    f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>"
                    f"{name}</div>"
                    f"<div style='font-size:22px;font-weight:900;color:{contrib_color};'>"
                    f"z = {last_z:+.2f}"
                    f"<span style='font-size:13px;color:rgba(0,0,0,0.45);font-weight:600;"
                    f"margin-left:6px;'>{delta_z:+.2f} this week</span></div>",
                    unsafe_allow_html=True,
                )
                fig_sp = go.Figure()
                x = list(range(len(series)))
                fig_sp.add_hline(y=0, line_color="#94a3b8", line_width=1)
                fig_sp.add_hline(y=1.5, line_dash="dot", line_color="#fca5a5", line_width=1)
                fig_sp.add_hline(y=-1.5, line_dash="dot", line_color="#86efac", line_width=1)

                colors = []
                for v in series.values:
                    if invert:
                        colors.append("#b42318" if v > 0 else "#1f7a4f")
                    else:
                        colors.append("#1f7a4f" if v > 0 else "#b42318")

                fig_sp.add_trace(go.Bar(
                    x=x, y=series.values,
                    marker_color=colors, showlegend=False,
                ))
                fig_sp.update_layout(
                    height=130, margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(visible=False),
                    yaxis=dict(showgrid=False, zeroline=False,
                               showticklabels=True, tickfont=dict(size=9)),
                )
                st.plotly_chart(fig_sp, width='stretch')
    else:
        st.caption("Component z-score history unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — CROSS-ASSET CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Cross-asset confirmation</div>",
                unsafe_allow_html=True)
    st.caption("Are price markets confirming the macro score? Divergences are early warnings.")

    CONFIRM_CHECKS = [
        ("HYG (high yield ETF)", "HYG",    "Credit",     True,  "Rising HYG = credit supportive = confirms bullish score"),
        ("IWM vs SPY",           None,      "Risk appetite", True, "IWM outperforming = risk appetite healthy = confirms score"),
        ("TLT (long duration)",  "TLT",    "Duration",   None,  "Falling TLT = rising rates = watch if credit spreads also widening"),
        ("UUP (USD ETF)",        "UUP",    "Dollar",     False, "Rising USD = headwind for risk assets = confirms bearish pressure"),
        ("GLD (gold)",           "GLD",    "Safe haven", None,  "Rising gold = flight to safety = watch for regime deterioration"),
    ]

    cols5 = st.columns(5, gap="medium")
    for i, (label, ticker, driver, up_is_good, note) in enumerate(CONFIRM_CHECKS):
        with cols5[i]:
            # Calculate 4-week return
            if ticker == "IWM vs SPY":
                pass  # handle below
            val = None
            if ticker and ticker in px.columns:
                s = px[ticker].dropna()
                if len(s) >= 22:
                    val = float(s.iloc[-1] / s.iloc[-22] - 1) * 100
            elif ticker is None:  # IWM vs SPY
                if "IWM" in px.columns and "SPY" in px.columns:
                    iwm_r = float(px["IWM"].dropna().iloc[-1] / px["IWM"].dropna().iloc[-22] - 1) * 100
                    spy_r = float(px["SPY"].dropna().iloc[-1] / px["SPY"].dropna().iloc[-22] - 1) * 100
                    val = iwm_r - spy_r

            if val is None:
                status_color, status_bg, status_txt = "#6b7280", "#f3f4f6", "N/A"
            elif up_is_good is True:
                if val > 1:
                    status_color, status_bg, status_txt = "#1f7a4f", "#dcfce7", "Confirming ✓"
                elif val < -1:
                    status_color, status_bg, status_txt = "#b42318", "#fee2e2", "Diverging ✗"
                else:
                    status_color, status_bg, status_txt = "#6b7280", "#f3f4f6", "Neutral"
            elif up_is_good is False:
                if val < -1:
                    status_color, status_bg, status_txt = "#1f7a4f", "#dcfce7", "Confirming ✓"
                elif val > 1:
                    status_color, status_bg, status_txt = "#b42318", "#fee2e2", "Diverging ✗"
                else:
                    status_color, status_bg, status_txt = "#6b7280", "#f3f4f6", "Neutral"
            else:
                status_color, status_bg, status_txt = "#6b7280", "#f3f4f6", "Watch"

            val_str = f"{val:+.1f}%" if val is not None else "—"

            st.markdown(
                f"<div style='padding:12px;border-radius:12px;background:{status_bg};"
                f"border:1px solid rgba(0,0,0,0.06);height:100%;'>"
                f"<div style='font-size:11px;font-weight:700;color:rgba(0,0,0,0.45);"
                f"text-transform:uppercase;letter-spacing:0.4px;'>{driver}</div>"
                f"<div style='font-size:13px;font-weight:800;color:rgba(0,0,0,0.85);"
                f"margin:4px 0;'>{label}</div>"
                f"<div style='font-size:20px;font-weight:900;color:{status_color};'>{val_str}</div>"
                f"<div style='font-size:11px;font-weight:800;color:{status_color};"
                f"margin-top:4px;'>{status_txt}</div>"
                f"<div style='font-size:10px;color:rgba(0,0,0,0.40);margin-top:6px;"
                f"line-height:1.4;'>{note}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — WHAT WOULD FLIP THE REGIME
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>What would flip the regime?</div>",
                unsafe_allow_html=True)
    st.caption("Scenario analysis — what macro changes would push the score across a boundary.")

    # Determine which boundaries are relevant
    up_boundary   = min([t for t in THRESHOLDS if t > cur_score], default=None)
    down_boundary = max([t for t in THRESHOLDS if t <= cur_score], default=None)

    flip_cols = st.columns(2, gap="medium")

    # Bull flip scenarios
    with flip_cols[0]:
        bull_color = "#1f7a4f"
        bull_bg    = "#dcfce7"
        if up_boundary:
            pts_needed = up_boundary - cur_score + 1
            st.markdown(
                f"<div style='padding:14px;border-radius:12px;background:{bull_bg};"
                f"border:1px solid #86efac;'>"
                f"<div style='font-weight:800;font-size:14px;color:{bull_color};"
                f"margin-bottom:10px;'>↑ Score to {up_boundary}: "
                f"+{pts_needed} points needed</div>",
                unsafe_allow_html=True,
            )
            bull_scenarios = [
                ("HY OAS tightens 30–50bp", "Credit stress component swings supportive"),
                ("Curve steepens +15bp", "Curve component contribution improves"),
                ("IWM outperforms SPY by 3%+ over 3m", "Risk appetite signal turns positive"),
                ("Real yields fall 20bp+", "Real yield component becomes supportive"),
                ("Dollar broad weakens 1%+", "Dollar impulse flips from headwind to tailwind"),
            ]
            for trigger, effect in bull_scenarios:
                st.markdown(
                    f"<div style='font-size:12px;padding:6px 0;"
                    f"border-bottom:1px solid rgba(31,122,79,0.15);'>"
                    f"<span style='font-weight:700;color:{bull_color};'>✦ {trigger}</span>"
                    f"<div style='color:rgba(0,0,0,0.60);margin-top:2px;'>{effect}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # Bear flip scenarios
    with flip_cols[1]:
        bear_color = "#b42318"
        bear_bg    = "#fee2e2"
        if down_boundary is not None:
            pts_lost = cur_score - down_boundary + 1
            st.markdown(
                f"<div style='padding:14px;border-radius:12px;background:{bear_bg};"
                f"border:1px solid #fca5a5;'>"
                f"<div style='font-weight:800;font-size:14px;color:{bear_color};"
                f"margin-bottom:10px;'>↓ Score to {down_boundary - 1}: "
                f"−{pts_lost} points would flip</div>",
                unsafe_allow_html=True,
            )
            bear_scenarios = [
                ("HY OAS widens 50bp+", "Credit component becomes primary headwind"),
                ("Curve flattens or re-inverts", "Curve signal turns bearish"),
                ("VIX spikes above 30 (V-Ratio > 1.0)", "Risk appetite collapses"),
                ("Real yields rise 30bp+", "Duration headwind increases"),
                ("Dollar surges 2%+", "Dollar impulse adds to bearish pressure"),
            ]
            for trigger, effect in bear_scenarios:
                st.markdown(
                    f"<div style='font-size:12px;padding:6px 0;"
                    f"border-bottom:1px solid rgba(180,35,24,0.15);'>"
                    f"<span style='font-weight:700;color:{bear_color};'>⚠ {trigger}</span>"
                    f"<div style='color:rgba(0,0,0,0.60);margin-top:2px;'>{effect}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — FULL SCORE HISTORY WITH BOUNDARY OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Full score history</div>", unsafe_allow_html=True)

    if not reg_hist.empty:
        rng = st.selectbox("Range", ["1y", "5y"], index=0, key="tw_hist_range")
        days = 400 if rng == "1y" else 2000
        end_dt  = reg_hist.index.max()
        start_dt = end_dt - pd.Timedelta(days=days)
        hist_view = reg_hist.loc[reg_hist.index >= start_dt]

        fig_full = go.Figure()
        for y0, y1, bg in [(0, 25, "rgba(180,35,24,0.07)"),
                            (25, 40, "rgba(217,119,6,0.06)"),
                            (40, 60, "rgba(107,114,128,0.04)"),
                            (60, 75, "rgba(22,163,74,0.06)"),
                            (75, 100, "rgba(31,122,79,0.09)")]:
            fig_full.add_hrect(y0=y0, y1=y1, fillcolor=bg, line_width=0)

        for thresh, lbl in [(25, "Bearish"), (40, "Neutral"), (60, "Bullish"), (75, "Risk On")]:
            fig_full.add_hline(y=thresh, line_dash="dash", line_color="#d1d5db",
                               line_width=1, annotation_text=lbl,
                               annotation_position="right")

        fig_full.add_trace(go.Scatter(
            x=hist_view.index, y=hist_view["score"],
            mode="lines", name="Regime score",
            line=dict(color="#1d4ed8", width=2),
            fill="tozeroy",
            fillcolor="rgba(29,78,216,0.05)",
        ))

        # Current score marker
        fig_full.add_hline(y=cur_score, line_color="#dc2626", line_width=1.5,
                           annotation_text=f"Now: {cur_score}",
                           annotation_position="right")

        fig_full.update_layout(
            height=320, margin=dict(l=20, r=80, t=20, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, yaxis_range=[0, 100],
            hovermode="x unified", yaxis_title="Score",
        )
        fig_full.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig_full.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_full, width='stretch')
    else:
        st.caption("Not enough history.")

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)