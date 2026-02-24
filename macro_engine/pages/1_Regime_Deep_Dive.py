# pages/1_Regime_Deep_Dive.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries

st.set_page_config(page_title="Regime deep dive", layout="wide")

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container {
  max-width: 1200px;
  padding-top: 4.8rem;   /* was ~1.2rem */
  padding-bottom: 2rem;
}

        /* Hide sidebar and its toggle */
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }

        /* Buttons */
        .stButton > button {
          border-radius: 12px !important;
          padding: 0.55rem 0.9rem !important;
          border: 1px solid rgba(0,0,0,0.10) !important;
        }

        .me-topbar {
  position: sticky;
  top: 1.5rem;          /* was 0 */
  z-index: 999;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 16px;
}

        .me-title {
          font-size: 24px;
          font-weight: 900;
          letter-spacing: -0.6px;
          margin: 0;
          line-height: 1.05;
        }

        .me-subtle {
          color: rgba(0,0,0,0.55);
          font-size: 12px;
          margin-top: 4px;
        }

        .me-chip {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          font-size: 14px;
          font-weight: 900;
          white-space: nowrap;
          background: rgba(0,0,0,0.01);
        }

        .me-dot {
          width: 10px;
          height: 10px;
          border-radius: 999px;
          display: inline-block;
        }

        .me-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          font-size: 12px;
          font-weight: 800;
          color: rgba(0,0,0,0.72);
          background: rgba(0,0,0,0.02);
          margin-right: 8px;
          margin-top: 6px;
        }

        .me-rowtitle {
          font-size: 13px;
          font-weight: 800;
          color: rgba(0,0,0,0.70);
          margin-bottom: 8px;
        }

        .me-li {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          border-radius: 12px;
          border: 1px solid rgba(0,0,0,0.06);
          background: #fff;
          margin-bottom: 8px;
        }

        .me-li:last-child { margin-bottom: 0; }

        .me-li-name {
          font-size: 13px;
          font-weight: 900;
          color: rgba(0,0,0,0.78);
          margin: 0;
        }

        .me-li-sub {
          font-size: 12px;
          color: rgba(0,0,0,0.55);
          margin: 0;
          margin-top: 2px;
        }

        .me-li-right {
          font-size: 12px;
          font-weight: 900;
          white-space: nowrap;
          color: rgba(0,0,0,0.72);
        }

        .me-kpi {
          font-size: 38px;
          font-weight: 900;
          letter-spacing: -1px;
          line-height: 1.0;
          margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]

def regime_color(regime: str) -> str:
    r = (regime or "").lower()
    if r == "risk on":
        return "#1f7a4f"
    if r == "risk off":
        return "#b42318"
    return "#6b7280"

@st.cache_data(ttl=12 * 60 * 60)
def load_macro() -> pd.DataFrame:
    df = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro")
    return df.sort_index()

@st.cache_data(ttl=6 * 60 * 60)
def load_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    df = fetch_prices(tickers, period=period)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_index()

def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"

def make_chip_row(items: list[str]) -> None:
    if not items:
        return
    chips = " ".join(
        [
            f"<span style='display:inline-block;padding:4px 10px;margin:4px 6px 0 0;border-radius:999px;background:#f2f3f5;font-size:13px;'>{t}</span>"
            for t in items
        ]
    )
    st.markdown(chips, unsafe_allow_html=True)

def _nearest_before(series: pd.Series, dt: pd.Timestamp):
    s = series.dropna()
    if s.empty:
        return None
    idx = s.index[s.index <= dt]
    if len(idx) == 0:
        return None
    return pd.Timestamp(idx.max())

def delta_over_days(series: pd.Series, days: int):
    s = series.dropna()
    if s.empty:
        return None, None, None

    end = pd.Timestamp(s.index.max())
    prev_dt = end - pd.Timedelta(days=days)

    end_i = _nearest_before(s, end)
    prev_i = _nearest_before(s, prev_dt)

    if end_i is None or prev_i is None:
        return None, None, None

    latest = float(s.loc[end_i])
    prev = float(s.loc[prev_i])
    return latest, prev, float(latest - prev)

def plot_regime_history(score_df: pd.DataFrame, spy: pd.Series, window: str) -> go.Figure:
    if score_df is None or score_df.empty or spy is None or spy.dropna().empty:
        fig = go.Figure()
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), title="Regime history")
        return fig

    look_days = 400 if window == "1y" else 2000
    end = max(score_df.index.max(), spy.index.max())
    start = end - pd.Timedelta(days=look_days)

    score_view = score_df.loc[score_df.index >= start].copy()
    spy_view = spy.loc[spy.index >= start].dropna().copy()
    spy_re = spy_view / spy_view.iloc[0] - 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score_view.index, y=score_view["score"], name="Regime score", mode="lines"))
    fig.add_trace(go.Scatter(x=spy_re.index, y=spy_re.values * 100.0, name="SPY return (pct)", mode="lines", yaxis="y2"))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Score"),
        yaxis2=dict(title="SPY return (pct)", overlaying="y", side="right"),
        title="Regime score and SPY",
    )
    return fig

# ---- Run ----
if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set. Add it as an environment variable locally and in Render.")
    st.stop()

macro = load_macro()
px = load_prices(ROTATION_TICKERS, period="5y")
regime = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63, momentum_lookback_days=21)

# Top bar (pretty)
dot = regime_color(getattr(regime, "label", "") or "")
tleft, tright = st.columns([3, 1], vertical_alignment="center")
with tleft:
    st.markdown(
        f"""
        <div class="me-topbar">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
            <div>
              <div class="me-title">Regime deep dive</div>
              <div class="me-subtle">Score history, drivers, components, and weekly deltas</div>
            </div>
            <div class="me-chip">
              <span class="me-dot" style="background:{dot}"></span>
              <span>{regime.label}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with tright:
    if st.button("Back to home", use_container_width=True):
        st.switch_page("app.py")

# Summary pills
st.markdown(
    f"""
    <span class="me-pill">Score: {regime.score}</span>
    <span class="me-pill">Confidence: {regime.confidence}</span>
    <span class="me-pill">Momentum: {regime.momentum_label.lower()}</span>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# Build weekly bullets and component rows first so we can place them at the top
days_look = 7
bullets = []

if "hy_oas" in macro.columns:
    latest, prev, dlt = delta_over_days(macro["hy_oas"], days_look)
    if dlt is not None:
        bullets.append(("Credit spreads (HY OAS)", latest, prev, dlt))
if "y10" in macro.columns and "y2" in macro.columns:
    curve = (macro["y10"] - macro["y2"]).dropna()
    latest, prev, dlt = delta_over_days(curve, days_look)
    if dlt is not None:
        bullets.append(("Curve (10y minus 2y)", latest, prev, dlt))
if "real10" in macro.columns:
    latest, prev, dlt = delta_over_days(macro["real10"], days_look)
    if dlt is not None:
        bullets.append(("Real 10y", latest, prev, dlt))
if "dollar_broad" in macro.columns:
    latest, prev, dlt = delta_over_days(macro["dollar_broad"], days_look)
    if dlt is not None:
        bullets.append(("Dollar broad", latest, prev, dlt))

if not bullets:
    bullets = [("No weekly deltas available yet", None, None, None)]

rows = []
for key, c in regime.components.items():
    level = c.get("level")
    z = c.get("zscore")
    trend_up = c.get("trend_up")

    if trend_up is None:
        trend_txt = ""
    else:
        if key == "credit":
            trend_txt = "tightening" if trend_up == 0 else "widening"
        else:
            trend_txt = "up" if trend_up == 1 else "down"

    rows.append(
        {
            "Component": c.get("name", key),
            "Level": "" if level is None else f"{float(level):.2f}",
            "Z": "" if z is None else f"{float(z):.2f}",
            "Trend": trend_txt,
            "Weight": f"{float(c.get('weight', 0.0)):.2f}",
        }
    )
comp_df = pd.DataFrame(rows)

# NEW: Top section with weekly changes + component details (moved to top)
topL, topR = st.columns([1, 1.35], gap="large")

with topL:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Weekly changes</div>", unsafe_allow_html=True)
        for name, latest, prev, dlt in bullets:
            if latest is None:
                st.markdown(
                    f"""
                    <div class="me-li">
                      <div><div class="me-li-name">{name}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                arrow = "↑" if dlt > 0 else "↓" if dlt < 0 else "→"
                st.markdown(
                    f"""
                    <div class="me-li">
                      <div>
                        <div class="me-li-name">{name}</div>
                        <div class="me-li-sub">{fmt_num(prev)} to {fmt_num(latest)}</div>
                      </div>
                      <div class="me-li-right">{arrow} {abs(dlt):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")
        if st.button("Open weekly details", use_container_width=True):
            st.switch_page("pages/4_Rotation_Setups.py")

with topR:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Component details</div>", unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Useful add: component concentration and biggest driver
        if not comp_df.empty and "Weight" in comp_df.columns:
            try:
                w = pd.to_numeric(comp_df["Weight"], errors="coerce").fillna(0.0)
                top_weight = float(w.max()) if len(w) else 0.0
                top_name = str(comp_df.loc[w.idxmax(), "Component"]) if len(w) else ""
                st.caption(f"Highest weight: {top_name} ({top_weight:.2f})")
            except Exception:
                pass

        st.markdown("")
        if st.button("Open score breakdown", use_container_width=True, key="btn_score_breakdown"):
            st.switch_page("pages/5_Drivers.py")

st.markdown("")

# History (kept, but now below the most actionable info)
with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>History</div>", unsafe_allow_html=True)
    hist_range = st.selectbox("Range", options=["1y", "5y"], index=0)
    reg_hist = compute_regime_timeseries(macro, px, lookback_trend=63, freq="W-FRI")
    spy = px["SPY"] if "SPY" in px.columns else pd.Series(dtype=float)
    st.plotly_chart(plot_regime_history(reg_hist, spy, hist_range), use_container_width=True)

st.markdown("")

# Allocation and favored groups (still useful, but below top actionables)
left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Allocation and favored groups</div>", unsafe_allow_html=True)

        stance = regime.allocation.get("stance", {}) if isinstance(regime.allocation, dict) else {}
        mix = regime.allocation.get("mix", {}) if isinstance(regime.allocation, dict) else {}
        tilts = regime.allocation.get("tilts", []) if isinstance(regime.allocation, dict) else []

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Equities", stance.get("Equities", "n/a"))
        s2.metric("Credit", stance.get("Credit", "n/a"))
        s3.metric("Duration", stance.get("Duration", "n/a"))
        s4.metric("USD", stance.get("USD", "n/a"))
        s5.metric("Commodities", stance.get("Commodities", "n/a"))

        if isinstance(mix, dict) and mix:
            mix_txt = ", ".join([f"{k} {v}%" for k, v in mix.items()])
            st.caption(f"Suggested mix: {mix_txt}")

        st.caption("Favored groups")
        make_chip_row(regime.favored_groups)

with right:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key drivers</div>", unsafe_allow_html=True)
        tilts = regime.allocation.get("tilts", []) if isinstance(regime.allocation, dict) else []
        if tilts:
            for t in tilts:
                st.write(f"• {t}")
        else:
            st.write("• No drivers yet")

        # Useful add: a quick watchlist CTA
        st.markdown("")
        st.caption("Quick watchlist")
        watch = ["HY OAS", "10y minus 2y", "Real 10y", "Dollar broad"]
        make_chip_row(watch)
