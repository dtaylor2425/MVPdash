# pages/1_Regime_Deep_Dive.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries
from src.ui import inject_css, sidebar_nav, safe_switch_page, html_table, regime_color, regime_bg

st.set_page_config(page_title="Regime deep dive", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Regime Engine")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

ROTATION_TICKERS = ["XLE","XLF","XLK","XLI","XLP","XLV","GLD","UUP","IWM","QQQ","SPY"]

@st.cache_data(ttl=30*60, show_spinner=False)
def load_macro():
    return get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro").sort_index()

@st.cache_data(ttl=30*60, show_spinner=False)
def load_prices(tickers, period="5y"):
    df = fetch_prices(tickers, period=period)
    return df.sort_index() if df is not None and not df.empty else pd.DataFrame()

@st.cache_data(ttl=30*60, show_spinner=False)
def load_current_regime():
    macro = load_macro()
    px    = load_prices(ROTATION_TICKERS)
    r     = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63, momentum_lookback_days=21)
    return r, macro, px

def fmt_num(x, nd=2):
    return "" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"

def _nearest_before(series, dt):
    s = series.dropna(); idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None

def delta_over_days(series, days):
    s = series.dropna()
    if s.empty: return None, None, None
    end = pd.Timestamp(s.index.max())
    ei  = _nearest_before(s, end)
    pi  = _nearest_before(s, end - pd.Timedelta(days=days))
    if ei is None or pi is None: return None, None, None
    lv, pv = float(s.loc[ei]), float(s.loc[pi])
    return lv, pv, float(lv - pv)

def plot_regime_history(score_df, spy, window):
    if score_df is None or score_df.empty or spy is None or spy.dropna().empty:
        return go.Figure().update_layout(height=300, margin=dict(l=20,r=20,t=40,b=20))
    days  = 400 if window == "1y" else 2000
    end   = max(score_df.index.max(), spy.index.max())
    start = end - pd.Timedelta(days=days)
    sv    = score_df.loc[score_df.index >= start]
    sp    = spy.loc[spy.index >= start].dropna()
    sp_r  = sp / sp.iloc[0] - 1.0
    fig   = go.Figure()
    for y0, y1, bg in [(0,25,"rgba(180,35,24,0.07)"),(25,40,"rgba(217,119,6,0.06)"),
                        (40,60,"rgba(107,114,128,0.04)"),(60,75,"rgba(22,163,74,0.06)"),
                        (75,100,"rgba(31,122,79,0.09)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=bg, line_width=0)
    for thresh, lbl in [(25,"Bearish"),(40,"Neutral"),(60,"Bullish"),(75,"Risk On")]:
        fig.add_hline(y=thresh, line_dash="dash", line_color="#d1d5db",
                      line_width=1, annotation_text=lbl, annotation_position="right")
    fig.add_trace(go.Scatter(x=sv.index, y=sv["score"], name="Regime score",
                             mode="lines", line=dict(color="#1d4ed8",width=2),
                             fill="tozeroy", fillcolor="rgba(29,78,216,0.05)"))
    fig.add_trace(go.Scatter(x=sp_r.index, y=sp_r.values*100, name="SPY return (%)",
                             mode="lines", line=dict(color="#94a3b8",width=1.5,dash="dot"),
                             yaxis="y2"))
    fig.update_layout(height=340, margin=dict(l=20,r=60,t=30,b=20),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
                      yaxis=dict(title="Score",range=[0,100],showgrid=True,gridcolor="#f1f5f9"),
                      yaxis2=dict(title="SPY return (%)",overlaying="y",side="right",showgrid=False),
                      plot_bgcolor="white",paper_bgcolor="white",hovermode="x unified")
    return fig

# ── Load ──────────────────────────────────────────────────────────────────────

regime, macro, px = load_current_regime()
label       = getattr(regime, "label", "Unknown")
score_val   = int(getattr(regime, "score", 0))
score_raw   = getattr(regime, "score_raw", float(score_val))
score_color = regime_color(label)
score_bg    = regime_bg(label)
dot         = score_color

# ── Topbar ────────────────────────────────────────────────────────────────────

tleft, tright = st.columns([4,1])
with tleft:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Regime deep dive</div>
              <div class="me-subtle">Score history · components · weekly deltas</div>
            </div>
            <div style="padding:8px 16px;border-radius:20px;background:{score_bg};">
              <span class="me-dot" style="background:{dot};display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle;"></span>
              <span style="color:{score_color};font-weight:800;font-size:14px;">{label} · {score_val}</span>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True)
with tright:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

st.markdown(
    f"<span class='me-pill'>Score: {score_val} ({score_raw:.1f} raw)</span>"
    f"<span class='me-pill'>Confidence: {getattr(regime,'confidence','')}</span>"
    f"<span class='me-pill'>Momentum: {str(getattr(regime,'momentum_label','')).lower()}</span>"
    f"<span class='me-pill'>Δ 21d: {getattr(regime,'score_delta','—')}</span>",
    unsafe_allow_html=True)
st.markdown("")

# ── Build weekly-change bullets ───────────────────────────────────────────────

bullets = []
if isinstance(macro, pd.DataFrame) and not macro.empty:
    for col, name, inverse in [
        ("hy_oas",       "Credit spreads (HY OAS)", True),
        ("curve",        "Curve (10y − 2y)",         False),
        ("real10",       "Real 10y yield",            True),
        ("dollar_broad", "Dollar broad",              True),
    ]:
        if col == "curve":
            series = (macro["y10"] - macro["y2"]).dropna() \
                     if "y10" in macro.columns and "y2" in macro.columns else None
        else:
            series = macro[col] if col in macro.columns else None
        if series is None: continue
        lv, pv, dlt = delta_over_days(series, 7)
        if dlt is not None: bullets.append((name, lv, pv, dlt, inverse))

# ── Build component table (v4: Level, Z, Mom-Z, Contribution, Weight) ─────────

def _z_bar(z):
    """Inline visual bar for z-score magnitude."""
    if z is None or pd.isna(z): return ""
    pct = min(abs(float(z)) / 2.5 * 100, 100)
    color = "#1f7a4f" if float(z) > 0 else "#b42318"
    return (f"<div style='display:inline-block;width:{pct:.0f}px;max-width:80px;height:6px;"
            f"background:{color};border-radius:3px;vertical-align:middle;margin-left:6px;'></div>")

def _fmt_z(z):
    if z is None or pd.isna(z): return "—"
    return f"{float(z):+.2f}"

def _contrib_color(v):
    try:
        f = float(v)
        if f > 0.01: return "#1f7a4f"
        if f < -0.01: return "#b42318"
    except Exception: pass
    return "#6b7280"

rows = []
components = getattr(regime, "components", {})
if isinstance(components, dict):
    for key, c in components.items():
        if not isinstance(c, dict): continue
        level    = c.get("level")
        z        = c.get("zscore")
        roc_z    = c.get("roc_zscore")
        trend_up = c.get("trend_up")
        contrib  = c.get("contribution")
        weight   = c.get("weight", 0.0)

        if trend_up is None:
            trend_txt = "—"
        elif "credit" in key or "inflation" in key.lower():
            trend_txt = "↓ tightening" if trend_up == 0 else "↑ widening"
        elif "dollar" in key:
            trend_txt = "↑ strong" if trend_up == 1 else "↓ weak"
        else:
            trend_txt = "↑" if trend_up == 1 else "↓"

        rows.append({
            "Component":   c.get("name", key),
            "Level":       fmt_num(level),
            "Z (level)":   _fmt_z(z),
            "Z (mom)":     _fmt_z(roc_z),
            "Trend":       trend_txt,
            "Contrib":     fmt_num(contrib, 3),
            "Wt":          f"{float(weight):.2f}",
        })
comp_df = pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Weekly changes | Component table
# ══════════════════════════════════════════════════════════════════════════════

topL, topR = st.columns([1, 1.6], gap="large")

with topL:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Weekly changes</div>", unsafe_allow_html=True)
        for name, lv, pv, dlt, inverse in bullets:
            good      = (dlt < 0) if inverse else (dlt > 0)
            badge_cls = "me-badge-green" if good else "me-badge-red"
            arrow     = "↑" if dlt > 0 else "↓"
            st.markdown(
                f"<div class='me-li'><div><div class='me-li-name'>{name}</div>"
                f"<div class='me-li-sub'>{fmt_num(pv)} → {fmt_num(lv)}</div></div>"
                f"<span class='me-badge {badge_cls}'>{arrow} {abs(dlt):.2f}</span></div>",
                unsafe_allow_html=True)
        st.markdown("")
        if st.button("Open weekly details →", width='stretch'):
            safe_switch_page("pages/4_Rotation_Setups.py")

with topR:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Component details — v4 continuous model</div>",
                    unsafe_allow_html=True)
        st.caption("Z (level) = z-score of the indicator vs trailing 252d · Z (mom) = z-score of 63d rate of change · "
                   "Contrib = contribution to score (clipped z × weight, sign-corrected)")

        # ── Component interpretation cards ────────────────────────────────────
        def _comp_interp(key, name, z, contrib, level):
            """Return plain-English interpretation of each component's current state."""
            if z is None: return None
            z = float(z); contrib = float(contrib or 0)
            direction = "bullish" if contrib > 0 else "bearish"
            col = "#1f7a4f" if contrib > 0 else "#b42318"
            bg  = "rgba(31,122,79,0.04)" if contrib > 0 else "rgba(180,35,24,0.04)"

            interps = {
                "credit": {
                    "pos": f"HY OAS is tightening (z {z:+.2f}) — credit markets are not pricing stress. "
                           "Tight spreads historically support equity multiples and risk appetite.",
                    "neg": f"HY OAS is elevated or widening (z {z:+.2f}) — credit stress is building. "
                           "Spreads lead equities by 4-8 weeks. This is the most important bearish signal.",
                },
                "real_yields": {
                    "pos": f"Real yield z-score is {z:+.2f} — conditions are easing in real terms. "
                           "Falling real rates expand equity multiples and support gold and duration.",
                    "neg": f"Real yield at {level:.2f}% (z {z:+.2f}) is in restrictive territory. "
                           "Every 100bp of real yield compresses equity P/E by roughly 1-2x. "
                           "This is the primary headwind on the score.",
                },
                "curve": {
                    "pos": f"Curve slope z {z:+.2f} — a steepening curve signals improving growth expectations. "
                           "Steep curve historically precedes credit tightening, cyclical outperformance, and higher equities.",
                    "neg": f"Curve z {z:+.2f} — flattening or inverted. Bear flattener means the market prices "
                           "Fed tightening without growth. This regime historically precedes credit stress by 6-12 months.",
                },
                "risk_appetite": {
                    "pos": f"IWM/SPY ratio z {z:+.2f} — small caps leading large caps. Broad participation "
                           "signals genuine risk appetite rather than a narrow mega-cap rally.",
                    "neg": f"IWM/SPY ratio z {z:+.2f} — small caps lagging. Breadth is narrowing, which historically "
                           "precedes broader market weakness. The rally is concentrated.",
                },
                "dollar": {
                    "pos": f"Dollar z {z:+.2f} is weakening — a falling dollar eases global financial conditions, "
                           "supports EM and commodities, and historically tailwinds risk assets.",
                    "neg": f"Dollar z {z:+.2f} is elevated or strengthening — a strong dollar tightens global "
                           "liquidity conditions. It acts as a global margin call for dollar-denominated borrowers.",
                },
                "cpi_momentum": {
                    "pos": f"Inflation momentum z {z:+.2f} is fading — easing price pressure gives the Fed "
                           "room to pause or cut, which is historically bullish for both bonds and equities.",
                    "neg": f"Inflation momentum z {z:+.2f} is elevated or re-accelerating — sticky inflation "
                           "keeps the Fed hawkish, compresses real returns, and is a headwind for duration.",
                },
            }
            key_clean = key.replace("_", "").lower()
            for k, v in interps.items():
                if k in key_clean or k in (name or "").lower():
                    text = v["pos"] if contrib >= 0 else v["neg"]
                    return col, bg, text
            return col, bg, f"{name} contributing {contrib:+.3f} to the score (z {z:+.2f})."

        if not comp_df.empty and isinstance(comps, dict):
            st.markdown("<div class='me-rowtitle' style='margin-top:14px;'>What each signal means right now</div>",
                        unsafe_allow_html=True)
            for key, c in comps.items():
                if not isinstance(c, dict): continue
                result = _comp_interp(
                    key, c.get("name",""), c.get("zscore"), c.get("contribution",0), c.get("level")
                )
                if not result: continue
                col, bg, text = result
                st.markdown(
                    f"<div style='padding:10px 14px;border-radius:10px;background:{bg};"
                    f"border-left:3px solid {col};margin-bottom:6px;font-size:12px;"
                    f"line-height:1.6;color:rgba(0,0,0,0.78);'>"
                    f"<span style='font-weight:800;color:{col};'>{c.get('name',key)}</span>"
                    f" — {text}</div>",
                    unsafe_allow_html=True)
        if comp_df.empty:
            st.caption("No component data yet.")
        else:
            # Render a richer HTML table with colour-coded Z and Contrib columns
            hdr_cells = "".join(
                f"<th style='padding:7px 10px;font-size:11px;font-weight:700;"
                f"color:rgba(0,0,0,0.45);text-transform:uppercase;letter-spacing:0.4px;"
                f"border-bottom:1px solid rgba(0,0,0,0.08);white-space:nowrap;'>{col}</th>"
                for col in comp_df.columns
            )
            rows_html = ""
            for _, row in comp_df.iterrows():
                cells = ""
                for col, val in row.items():
                    style = "padding:7px 10px;font-size:12px;border-bottom:1px solid rgba(0,0,0,0.04);"
                    extra_color = ""
                    if col in ("Z (level)", "Z (mom)"):
                        try:
                            fv = float(val)
                            extra_color = f"color:{'#1f7a4f' if fv > 0 else '#b42318'};font-weight:700;"
                        except Exception: pass
                    elif col == "Contrib":
                        try:
                            fv = float(val)
                            extra_color = f"color:{_contrib_color(val)};font-weight:800;"
                        except Exception: pass
                    elif col == "Component":
                        extra_color = "font-weight:700;color:rgba(0,0,0,0.85);"
                    cells += f"<td style='{style}{extra_color}'>{val}</td>"
                rows_html += f"<tr>{cells}</tr>"
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr>{hdr_cells}</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True)

            # Score contribution bar chart
            st.markdown("<div class='me-rowtitle' style='margin-top:14px;'>Contribution to score</div>",
                        unsafe_allow_html=True)
            import altair as alt
            chart_df = comp_df[["Component","Contrib"]].copy()
            chart_df["Contrib"] = pd.to_numeric(chart_df["Contrib"], errors="coerce")
            chart_df = chart_df.dropna().sort_values("Contrib")
            if not chart_df.empty:
                bar = (
                    alt.Chart(chart_df)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5,
                              cornerRadiusBottomLeft=5, cornerRadiusBottomRight=5)
                    .encode(
                        y=alt.Y("Component:N", sort=None, title=None,
                                axis=alt.Axis(labelFontSize=11, labelLimit=140)),
                        x=alt.X("Contrib:Q", title="Contribution", axis=alt.Axis(format=".3f")),
                        color=alt.condition(
                            alt.datum.Contrib > 0,
                            alt.value("#1f7a4f"), alt.value("#b42318")),
                        tooltip=["Component", alt.Tooltip("Contrib:Q", format=".4f")],
                    )
                    .properties(height=160)
                )
                st.altair_chart(bar, width='stretch')

        st.markdown("")
        if st.button("Score breakdown →", width='stretch', key="btn_score_bk"):
            safe_switch_page("pages/5_Drivers.py")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — History
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>History</div>", unsafe_allow_html=True)
    hist_range = st.selectbox("Range", ["1y","5y"], index=0)
    reg_hist   = compute_regime_timeseries(macro, px, lookback_trend=63, freq="W-FRI")
    spy_s      = px["SPY"] if isinstance(px,pd.DataFrame) and "SPY" in px.columns \
                 else pd.Series(dtype=float)
    st.plotly_chart(plot_regime_history(reg_hist, spy_s, hist_range), width='stretch')

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Allocation | Key drivers
# ══════════════════════════════════════════════════════════════════════════════

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Allocation and favoured groups</div>",
                    unsafe_allow_html=True)
        alloc  = getattr(regime, "allocation", {})
        stance = alloc.get("stance", {}) if isinstance(alloc, dict) else {}
        mix    = alloc.get("mix",    {}) if isinstance(alloc, dict) else {}
        for asset, s in stance.items():
            sv = str(s)
            if "over" in sv.lower():   bc, bg2 = "#166534", "#dcfce7"
            elif "under" in sv.lower(): bc, bg2 = "#991b1b", "#fee2e2"
            else:                       bc, bg2 = "#374151", "#f3f4f6"
            st.markdown(
                f"<div class='me-li'><span class='me-li-name'>{asset}</span>"
                f"<span class='me-badge' style='background:{bg2};color:{bc};'>{sv}</span></div>",
                unsafe_allow_html=True)
        if isinstance(mix, dict) and mix:
            st.caption("Suggested mix: " + ", ".join(f"{k} {v}%" for k,v in mix.items()))
        groups = getattr(regime, "favored_groups", []) or []
        if groups:
            chips = " ".join(
                f"<span style='display:inline-block;padding:4px 10px;margin:4px 4px 0 0;"
                f"border-radius:999px;background:#f2f3f5;font-size:12px;"
                f"color:rgba(0,0,0,0.75);'>{g}</span>"
                for g in groups)
            st.markdown("<div class='me-rowtitle' style='margin-top:10px;'>Favored groups</div>",
                        unsafe_allow_html=True)
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Regime Playbook →", width='stretch', key="btn_playbook"):
            safe_switch_page("pages/7_Regime_Playbook.py")

with right:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key drivers</div>", unsafe_allow_html=True)
        tilts = alloc.get("tilts", []) if isinstance(alloc, dict) else []
        if tilts:
            for t in tilts:
                st.markdown(
                    f"<div style='padding:7px 0;font-size:13px;color:rgba(0,0,0,0.80);"
                    f"border-bottom:1px solid rgba(0,0,0,0.05);'>• {t}</div>",
                    unsafe_allow_html=True)
        else:
            st.caption("No drivers available.")
        st.markdown("")
        if st.button("Transition Watch →", width='stretch', key="btn_tw"):
            safe_switch_page("pages/8_Transition_Watch.py")

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
