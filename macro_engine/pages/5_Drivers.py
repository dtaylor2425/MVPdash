# pages/5_Drivers.py
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3
from src.ui import inject_css, sidebar_nav, safe_switch_page, regime_color, regime_bg

st.set_page_config(page_title="Drivers", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Drivers")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

ROTATION_TICKERS = ["XLE","XLF","XLK","XLI","XLP","XLV","GLD","UUP","IWM","QQQ","SPY","RSP"]

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30*60, show_spinner=False)
def load_current_regime():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    px = fetch_prices(ROTATION_TICKERS, period="5y")
    px = pd.DataFrame() if px is None or px.empty else px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63,
                               momentum_lookback_days=21)
    return regime, macro, px

regime_obj, macro, px = load_current_regime()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _component_contribution(c: dict) -> float:
    if not isinstance(c, dict): return 0.0
    v = c.get("contribution")
    if isinstance(v, (int, float, np.floating)): return float(v)
    z, w = c.get("zscore"), c.get("weight")
    if isinstance(z, (int, float, np.floating)) and isinstance(w, (int, float, np.floating)):
        return float(z) * float(w)
    return 0.0

def _metric_target(name: str) -> str:
    n = (name or "").lower()
    if "credit" in n or "oas" in n:   return "Credit spreads"
    if "dollar" in n or "usd" in n:    return "Broad dollar"
    if "real" in n:                    return "Real yields"
    if "curve" in n or "term" in n:    return "Curve"
    if "inflation" in n or "cpi" in n: return "Inflation"
    if "vol" in n or "vix" in n:       return "Volatility"
    return "Risk appetite"

def _trend_phrase(name: str, trend_up) -> str:
    nm = (name or "").lower()
    if trend_up is None: return "steady"
    if "credit" in nm or "spread" in nm:
        return "widening" if int(trend_up) == 1 else "tightening"
    if "inflation" in nm or "cpi" in nm:
        return "rising" if int(trend_up) == 1 else "falling"
    if "dollar" in nm or "usd" in nm:
        return "strengthening" if int(trend_up) == 1 else "weakening"
    return "rising" if int(trend_up) == 1 else "falling"

def _z_signal(z, inverted=False) -> tuple[str, str]:
    """Returns (label, colour) for a z-score, adjusted for inversion."""
    if z is None or pd.isna(z): return "neutral", "#6b7280"
    fz = float(z)
    if inverted: fz = -fz
    if fz > 1.0:  return "strongly supportive", "#1f7a4f"
    if fz > 0.3:  return "mildly supportive",   "#16a34a"
    if fz < -1.0: return "strong drag",          "#b42318"
    if fz < -0.3: return "mild drag",            "#d97706"
    return "neutral",  "#6b7280"

# Inversion flag per component key — determines whether a high z = good or bad
_INVERTED = {
    "credit":       True,   # high OAS z = bad
    "real_yields":  True,   # high real yield z = bad
    "dollar":       True,   # high dollar z = bad
    "cpi_momentum": True,   # high inflation z = bad
    "curve":        False,  # high curve z = good (steeper)
    "risk_appetite":False,  # high IWM/SPY z = good
}

# ── Build driver cards ────────────────────────────────────────────────────────

def build_drivers(regime_obj):
    comps = getattr(regime_obj, "components", None)
    if not isinstance(comps, dict) or not comps:
        return [{"name": "Drivers unavailable",
                 "contrib": 0, "z": None, "roc_z": None,
                 "trend_up": None, "weight": 0, "target": "Risk appetite",
                 "inverted": False}]
    rows = []
    for k, c in comps.items():
        if not isinstance(c, dict): continue
        rows.append({
            "key":      k,
            "name":     c.get("name", k),
            "contrib":  _component_contribution(c),
            "z":        c.get("zscore"),
            "roc_z":    c.get("roc_zscore"),
            "trend_up": c.get("trend_up"),
            "weight":   float(c.get("weight", 0)),
            "level":    c.get("level"),
            "target":   _metric_target(c.get("name", k)),
            "inverted": _INVERTED.get(k, False),
        })
    return sorted(rows, key=lambda r: abs(r["contrib"]), reverse=True)

drivers = build_drivers(regime_obj)

# ── Build summary line ────────────────────────────────────────────────────────

label       = getattr(regime_obj, "label", "Unknown")
score_val   = int(getattr(regime_obj, "score", 0))
score_raw   = getattr(regime_obj, "score_raw", float(score_val))
conf        = getattr(regime_obj, "confidence", "Unknown")
mom         = getattr(regime_obj, "momentum_label", "Unknown")
score_color = regime_color(label)
score_bg_c  = regime_bg(label)

comps_list  = [(d["name"], d["contrib"]) for d in drivers]
supportive  = [n for n, v in comps_list if v > 0.01][:2]
adverse     = [n for n, v in comps_list if v < -0.01][-2:]
summary_line = f"Regime: {label} · Score: {score_val} ({score_raw:.1f}) · Confidence: {conf} · Momentum: {mom.lower()}"
summary_extra = ""
if supportive: summary_extra += "Supportive: " + ", ".join(supportive)
if adverse:    summary_extra += (" | " if summary_extra else "") + "Headwind: " + ", ".join(adverse)

# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════

h1, h2 = st.columns([4,1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Key drivers</div>
              <div class="me-subtle">Component z-scores · contributions · momentum signals</div>
            </div>
            <div style="padding:8px 16px;border-radius:20px;background:{score_bg_c};">
              <span style="font-weight:800;color:{score_color};font-size:14px;">
                {label} · {score_val}
              </span>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True)
with h2:
    if st.button("Regime deep dive →", width='stretch'):
        safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown(f"<span class='me-pill'>{summary_line}</span>", unsafe_allow_html=True)
if summary_extra:
    st.markdown(f"<div class='me-subtle' style='margin-top:6px;'>{summary_extra}</div>",
                unsafe_allow_html=True)
st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Driver cards (top 6 by |contribution|)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='me-rowtitle'>Component contributions</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
for i, d in enumerate(drivers[:6]):
    col = col1 if i % 2 == 0 else col2
    with col:
        with st.container(border=True):
            contrib   = d["contrib"]
            z         = d["z"]
            roc_z     = d["roc_z"]
            inverted  = d["inverted"]

            c_color = "#1f7a4f" if contrib > 0.01 else ("#b42318" if contrib < -0.01 else "#6b7280")
            c_bg    = "#dcfce7" if contrib > 0.01 else ("#fee2e2" if contrib < -0.01 else "#f3f4f6")
            contrib_arrow = "▲" if contrib > 0.01 else ("▼" if contrib < -0.01 else "●")

            z_sig_lbl, z_sig_col = _z_signal(z, inverted)

            # Mom z signal — not inverted (positive roc_z = accelerating in its natural direction)
            mom_lbl = "—"
            mom_col = "#6b7280"
            if roc_z is not None and not pd.isna(roc_z):
                frz = float(roc_z)
                # For inverted components, positive roc_z means condition worsening
                if inverted: frz = -frz
                if frz > 0.5:   mom_lbl, mom_col = "accelerating ↑", "#1f7a4f"
                elif frz < -0.5: mom_lbl, mom_col = "decelerating ↓", "#b42318"
                else:            mom_lbl, mom_col = "steady →",       "#6b7280"

            trend_txt = _trend_phrase(d["name"], d["trend_up"])

            # Header row: name + contribution badge
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
                f"margin-bottom:8px;'>"
                f"<div style='font-size:14px;font-weight:800;color:rgba(0,0,0,0.85);'>"
                f"  {d['name']}</div>"
                f"<span style='font-size:11px;font-weight:800;color:{c_color};"
                f"background:{c_bg};padding:3px 10px;border-radius:8px;white-space:nowrap;'>"
                f"  {contrib_arrow} {contrib:+.4f}</span>"
                f"</div>",
                unsafe_allow_html=True)

            # Z-score row
            z_str   = f"{float(z):+.2f}" if z is not None and not pd.isna(z) else "n/a"
            roc_str = f"{float(roc_z):+.2f}" if roc_z is not None and not pd.isna(roc_z) else "n/a"
            lvl_str = f"{float(d['level']):.2f}" if d['level'] is not None and not pd.isna(d['level']) else "n/a"

            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;"
                f"margin-bottom:8px;'>"
                f"<div style='background:#f8f9fa;border-radius:8px;padding:7px 10px;'>"
                f"  <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.40);"
                f"  text-transform:uppercase;letter-spacing:0.4px;'>Level</div>"
                f"  <div style='font-size:14px;font-weight:800;color:rgba(0,0,0,0.80);'>{lvl_str}</div>"
                f"</div>"
                f"<div style='background:#f8f9fa;border-radius:8px;padding:7px 10px;'>"
                f"  <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.40);"
                f"  text-transform:uppercase;letter-spacing:0.4px;'>Z (level)</div>"
                f"  <div style='font-size:14px;font-weight:800;color:{z_sig_col};'>{z_str}</div>"
                f"</div>"
                f"<div style='background:#f8f9fa;border-radius:8px;padding:7px 10px;'>"
                f"  <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.40);"
                f"  text-transform:uppercase;letter-spacing:0.4px;'>Z (mom)</div>"
                f"  <div style='font-size:14px;font-weight:800;color:{mom_col};'>{roc_str}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True)

            # Signal text
            st.markdown(
                f"<div style='font-size:12px;color:rgba(0,0,0,0.65);line-height:1.5;'>"
                f"  <span style='color:{z_sig_col};font-weight:700;'>{z_sig_lbl.capitalize()}</span>"
                f"  · {trend_txt} · momentum <span style='color:{mom_col};font-weight:700;'>"
                f"  {mom_lbl}</span> · weight {d['weight']:.2f}"
                f"</div>",
                unsafe_allow_html=True)

            st.markdown("")
            if st.button("Open related chart →", key=f"driver_{i}", width='stretch'):
                st.session_state["selected_metric"] = d["target"]
                safe_switch_page("pages/2_Macro_Charts.py")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Score decomposition waterfall + regime summary
# ══════════════════════════════════════════════════════════════════════════════

decomp_col, summary_col = st.columns([1.4, 1.0], gap="large")

with decomp_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Score decomposition</div>", unsafe_allow_html=True)
        st.caption("Signed contribution of each component to the final score. "
                   "Sum of contributions → normalised → mapped to [0, 100].")
        if drivers and drivers[0]["name"] != "Drivers unavailable":
            chart_data = pd.DataFrame([
                {"Component": d["name"][:20], "Contribution": d["contrib"]}
                for d in drivers
            ]).sort_values("Contribution")
            bar = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5,
                          cornerRadiusBottomLeft=5, cornerRadiusBottomRight=5)
                .encode(
                    y=alt.Y("Component:N", sort=None, title=None,
                            axis=alt.Axis(labelFontSize=11, labelLimit=150)),
                    x=alt.X("Contribution:Q", title="Contribution",
                            axis=alt.Axis(format=".3f")),
                    color=alt.condition(
                        alt.datum.Contribution > 0,
                        alt.value("#1f7a4f"), alt.value("#b42318")),
                    tooltip=["Component",
                             alt.Tooltip("Contribution:Q", format=".4f")],
                )
                .properties(height=220)
            )
            st.altair_chart(bar, width='stretch')
        else:
            st.caption("Component data unavailable.")

with summary_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Regime summary</div>", unsafe_allow_html=True)
        # Score gauge-style display
        st.markdown(
            f"<div style='text-align:center;padding:16px 0;'>"
            f"<div style='font-size:52px;font-weight:900;color:{score_color};"
            f"line-height:1;'>{score_val}</div>"
            f"<div style='font-size:13px;color:rgba(0,0,0,0.50);margin-top:4px;'>"
            f"raw {score_raw:.1f} · out of 100</div>"
            f"<div style='font-size:16px;font-weight:800;color:{score_color};"
            f"margin-top:8px;'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True)
        st.markdown("---")
        for line in [
            ("Confidence", conf),
            ("Momentum",   mom),
            ("Score Δ 21d", str(getattr(regime_obj, "score_delta", "—"))),
        ]:
            st.markdown(
                f"<div class='me-li'><span class='me-li-name'>{line[0]}</span>"
                f"<span class='me-li-right' style='font-weight:700;'>{line[1]}</span></div>",
                unsafe_allow_html=True)
        if summary_extra:
            st.caption(summary_extra)
        st.markdown("")
        if st.button("Transition Watch →", width='stretch', key="btn_tw"):
            safe_switch_page("pages/8_Transition_Watch.py")

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)