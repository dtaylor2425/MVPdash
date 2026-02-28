# pages/1_Regime_Deep_Dive.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries
from src.compute import component_contribution
from src.ui import (
    inject_css, sidebar_nav, safe_switch_page,
    regime_color, delta_badge_html, make_chip_row, SCORE_LEGEND_HTML,
)

st.set_page_config(
    page_title="Regime deep dive",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
sidebar_nav(active="Regime deep dive")

# ── Constants ─────────────────────────────────────────────────────────────────

ROTATION_TICKERS = [
    "XLE", "XLF", "XLK", "XLI", "XLP", "XLV",
    "GLD", "UUP", "IWM", "QQQ",
    "IGV", "SMH",   # software and semiconductors
    "SPY",
]
INVERSE_METRICS  = {"credit spreads", "hy oas", "spread"}

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_macro() -> pd.DataFrame:
    return get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro").sort_index()


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_prices(tickers: list, period: str = "5y") -> pd.DataFrame:
    df = fetch_prices(tickers, period=period)
    return pd.DataFrame() if (df is None or df.empty) else df.sort_index()


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_current_regime():
    macro  = load_macro()
    px     = load_prices(ROTATION_TICKERS)
    regime = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63, momentum_lookback_days=21)
    return regime, macro, px

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_num(x, nd: int = 2) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def _nearest_before(series: pd.Series, dt: pd.Timestamp):
    s   = series.dropna()
    idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None


def delta_over_days(series: pd.Series, days: int):
    s = series.dropna()
    if s.empty:
        return None, None, None
    end    = pd.Timestamp(s.index.max())
    end_i  = _nearest_before(s, end)
    prev_i = _nearest_before(s, end - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None, None, None
    latest = float(s.loc[end_i])
    prev   = float(s.loc[prev_i])
    return latest, prev, float(latest - prev)


def plot_regime_history(score_df: pd.DataFrame, spy: pd.Series, window: str) -> go.Figure:
    if score_df is None or score_df.empty or spy is None or spy.dropna().empty:
        fig = go.Figure()
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), title="Regime history")
        return fig

    look_days = 400 if window == "1y" else 2000
    end       = max(score_df.index.max(), spy.index.max())
    start     = end - pd.Timedelta(days=look_days)

    score_view = score_df.loc[score_df.index >= start].copy()
    spy_view   = spy.loc[spy.index >= start].dropna()
    spy_re     = spy_view / spy_view.iloc[0] - 1.0

    fig = go.Figure()

    # Regime band shading
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(31,122,79,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(180,35,24,0.08)", line_width=0)

    fig.add_trace(go.Scatter(x=score_view.index, y=score_view["score"],
                             name="Regime score", mode="lines",
                             line=dict(color="#1d4ed8", width=2)))
    fig.add_trace(go.Scatter(x=spy_re.index, y=spy_re.values * 100.0,
                             name="SPY return (pct)", mode="lines",
                             line=dict(color="#94a3b8", width=1.5, dash="dot"),
                             yaxis="y2"))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis =dict(title="Score", range=[0, 100]),
        yaxis2=dict(title="SPY return (pct)", overlaying="y", side="right"),
        title ="Regime score and SPY",
    )
    return fig

# ── Guard ─────────────────────────────────────────────────────────────────────

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

regime, macro, px = load_current_regime()

dot        = regime_color(getattr(regime, "label", "") or "")
score_val  = int(getattr(regime, "score", 0))
score_color = "#1f7a4f" if score_val >= 60 else ("#b42318" if score_val < 40 else "#6b7280")

# ═══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ═══════════════════════════════════════════════════════════════════════════════

tleft, tright = st.columns([3, 1], vertical_alignment="center")
with tleft:
    st.markdown(
        f"""
        <div class="me-topbar">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
            <div>
              <div class="me-title">Regime deep dive</div>
              <div class="me-subtle">Score history, drivers, components, and weekly deltas</div>
            </div>
            <div class="me-chip">
              <span class="me-dot" style="background:{dot}"></span>
              <span>{getattr(regime, "label", "Unknown")}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with tright:
    if st.button("Back to home", use_container_width=True):
        safe_switch_page("app.py")

st.markdown(
    f"""
    <span class="me-pill">Score: <strong style='color:{score_color};'>{score_val}</strong></span>
    <span class="me-pill">Confidence: {getattr(regime, "confidence", "")}</span>
    <span class="me-pill">Momentum: {str(getattr(regime, "momentum_label", "")).lower()}</span>
    """,
    unsafe_allow_html=True,
)
st.markdown(SCORE_LEGEND_HTML, unsafe_allow_html=True)
st.markdown("")

# ── Weekly bullets ────────────────────────────────────────────────────────────

bullets = []
if isinstance(macro, pd.DataFrame) and not macro.empty:
    checks = [
        ("hy_oas",     "Credit spreads (HY OAS)"),
        (None,         "Curve (10y minus 2y)"),   # special-cased below
        ("real10",     "Real 10y"),
        ("dollar_broad","Dollar broad"),
    ]
    if "hy_oas" in macro.columns:
        l, p, d = delta_over_days(macro["hy_oas"], 7)
        if d is not None: bullets.append(("Credit spreads (HY OAS)", l, p, d))
    if "y10" in macro.columns and "y2" in macro.columns:
        l, p, d = delta_over_days((macro["y10"] - macro["y2"]).dropna(), 7)
        if d is not None: bullets.append(("Curve (10y minus 2y)", l, p, d))
    if "real10" in macro.columns:
        l, p, d = delta_over_days(macro["real10"], 7)
        if d is not None: bullets.append(("Real 10y", l, p, d))
    if "dollar_broad" in macro.columns:
        l, p, d = delta_over_days(macro["dollar_broad"], 10)  # 10 days: dollar_broad is weekly
        if d is not None: bullets.append(("Dollar broad", l, p, d))

if not bullets:
    bullets = [("No weekly deltas available yet", None, None, None)]

# ── Component table ───────────────────────────────────────────────────────────

rows = []
components = getattr(regime, "components", {})
if isinstance(components, dict):
    for key, c in components.items():
        if not isinstance(c, dict):
            continue
        level    = c.get("level")
        z        = c.get("zscore")
        trend_up = c.get("trend_up")
        if trend_up is None:
            trend_txt = ""
        elif key == "credit":
            trend_txt = "tightening" if trend_up == 0 else "widening"
        else:
            trend_txt = "up" if trend_up == 1 else "down"
        rows.append({
            "Component": c.get("name", key),
            "Level":     "" if level is None else f"{float(level):.2f}",
            "Z":         "" if z     is None else f"{float(z):.2f}",
            "Trend":     trend_txt,
            "Weight":    f"{float(c.get('weight', 0.0)):.2f}",
        })
comp_df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# WEEKLY CHANGES  +  COMPONENT TABLE
# ═══════════════════════════════════════════════════════════════════════════════

topL, topR = st.columns([1, 1.35], gap="large")

with topL:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Weekly changes</div>", unsafe_allow_html=True)
        for name, latest, prev, dlt in bullets:
            if latest is None:
                st.markdown(
                    f"<div class='me-li'><div class='me-li-name'>{name}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                is_inverse = any(k in name.lower() for k in INVERSE_METRICS)
                badge      = delta_badge_html(dlt, inverse=is_inverse)
                st.markdown(
                    f"""
                    <div class="me-li">
                      <div>
                        <div class="me-li-name">{name}</div>
                        <div class="me-li-sub">{fmt_num(prev)} → {fmt_num(latest)}</div>
                      </div>
                      <div class="me-li-right">{badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("")
        if st.button("Open weekly details", use_container_width=True):
            safe_switch_page("pages/4_Rotation_Setups.py")

with topR:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Component details</div>", unsafe_allow_html=True)
        if comp_df.empty:
            st.caption("No component table available yet.")
        else:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            if "Weight" in comp_df.columns:
                try:
                    w        = pd.to_numeric(comp_df["Weight"], errors="coerce").fillna(0.0)
                    top_name = str(comp_df.loc[w.idxmax(), "Component"]) if len(w) else ""
                    st.caption(f"Highest weight: {top_name} ({float(w.max()):.2f})")
                except Exception:
                    pass
        st.markdown("")
        if st.button("Open score breakdown", use_container_width=True, key="btn_score_breakdown"):
            safe_switch_page("pages/5_Drivers.py")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY CHART
# ═══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>History</div>", unsafe_allow_html=True)
    hist_range = st.selectbox("Range", options=["1y", "5y"], index=0)
    reg_hist   = compute_regime_timeseries(macro, px, lookback_trend=63, freq="W-FRI")
    spy        = px["SPY"] if isinstance(px, pd.DataFrame) and "SPY" in px.columns else pd.Series(dtype=float)
    st.plotly_chart(plot_regime_history(reg_hist, spy, hist_range), use_container_width=True)

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# ALLOCATION  +  KEY DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Allocation and favoured groups</div>", unsafe_allow_html=True)
        alloc  = getattr(regime, "allocation", {})
        stance = alloc.get("stance", {}) if isinstance(alloc, dict) else {}
        mix    = alloc.get("mix",    {}) if isinstance(alloc, dict) else {}

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Equities",    stance.get("Equities",    "n/a"))
        c2.metric("Credit",      stance.get("Credit",      "n/a"))
        c3.metric("Duration",    stance.get("Duration",    "n/a"))
        c4.metric("USD",         stance.get("USD",         "n/a"))
        c5.metric("Commodities", stance.get("Commodities", "n/a"))

        if isinstance(mix, dict) and mix:
            st.caption("Suggested mix: " + ", ".join(f"{k} {v}%" for k, v in mix.items()))

        st.caption("Favoured groups")
        make_chip_row(getattr(regime, "favored_groups", []) or [])

with right:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key drivers</div>", unsafe_allow_html=True)
        tilts = alloc.get("tilts", []) if isinstance(alloc, dict) else []
        if tilts:
            for t in tilts:
                st.write(f"• {t}")
        else:
            st.write("• No drivers yet")

        st.markdown("")
        st.caption("Quick watchlist")
        make_chip_row(["HY OAS", "10y minus 2y", "Real 10y", "Dollar broad"])

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)