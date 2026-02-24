import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date

from src.data_sources import fetch_prices
from src.paths import data_path

st.set_page_config(
    page_title="Macro Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2.0rem;
          max-width: 1200px;
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
          top: 0;
          z-index: 999;
          background: rgba(255,255,255,0.92);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 16px;
          padding: 14px 16px;
          margin-bottom: 14px;
        }

        /* Bigger, more title-like */
        .me-title {
          font-size: 26px;
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

        /* Bigger regime chip */
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

        .me-kpi {
          font-size: 38px;
          font-weight: 900;
          letter-spacing: -1px;
          line-height: 1.0;
          margin-top: 2px;
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

        /* Nav tiles */
        .me-nav-title {
          font-weight: 900;
          font-size: 13px;
          margin: 0;
          line-height: 1.15;
        }
        .me-nav-desc {
          color: rgba(0,0,0,0.55);
          font-size: 12px;
          margin-top: 4px;
          line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

def regime_color(regime: str) -> str:
    r = (regime or "").lower()
    if r == "risk on":
        return "#1f7a4f"
    if r == "risk off":
        return "#b42318"
    return "#6b7280"

def safe_switch_page(path: str):
    try:
        st.switch_page(path)
    except Exception:
        st.error(f"Missing page: {path}. Make sure the file exists in your pages folder.")

def load_home_state():
    last_updated = date.today()

    regime = "Risk On"
    score = 65
    confidence = "High"
    momentum = "Stable"

    allocation_tilt = [
        ("Equities", "Overweight"),
        ("Credit", "Overweight"),
        ("Duration", "Underweight"),
        ("USD", "Underweight"),
        ("Commodities", "Overweight"),
    ]

    weekly_changes = [
        ("Credit spreads", 2.95, 2.86, -0.09, "tighter"),
        ("Curve (10y minus 2y)", 0.64, 0.60, -0.04, "flatter"),
        ("Real 10y", 1.77, 1.80, 0.03, "higher"),
        ("Dollar broad", 117.53, 117.99, 0.47, "stronger"),
    ]

    drivers = [
        ("Equities favored by regime score", "See credit and breadth"),
        ("Breadth improving", "See small caps and equal weight"),
        ("Credit spreads tight", "See HY OAS"),
    ]

    key_risk = "Watch inflation expectations, real yields, and credit for a reversal signal."

    return {
        "last_updated": last_updated,
        "regime": regime,
        "score": score,
        "confidence": confidence,
        "momentum": momentum,
        "allocation_tilt": allocation_tilt,
        "weekly_changes": weekly_changes,
        "drivers": drivers,
        "key_risk": key_risk,
    }

@st.cache_data(show_spinner=False)
def compute_rs_heatmap(period: str = "1y") -> pd.DataFrame:
    tickers = [
        "GLD", "XLE", "XLP", "XLI", "IWM", "RSP",
        "HYG", "TLT", "UUP", "QQQ", "XLK", "XLF", "XLV"
    ]
    base = "SPY"
    fetch_list = list(dict.fromkeys([base] + tickers))

    px = fetch_prices(fetch_list, period=period)
    if px is None or px.empty:
        return pd.DataFrame(index=tickers, columns=["1w", "1m", "3m"], data=np.nan)

    px = px.dropna(how="all")
    if base not in px.columns:
        return pd.DataFrame(index=tickers, columns=["1w", "1m", "3m"], data=np.nan)

    horizons = {"1w": 5, "1m": 21, "3m": 63}
    out = pd.DataFrame(index=tickers, columns=list(horizons.keys()), dtype=float)

    b = px[base].dropna()
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        idx = s.index.intersection(b.index)
        if len(idx) < 90:
            continue
        s = s.loc[idx]
        bb = b.loc[idx]
        for lab, n in horizons.items():
            if len(idx) <= n:
                out.loc[t, lab] = np.nan
            else:
                ret_t = (s.iloc[-1] / s.iloc[-1 - n]) - 1.0
                ret_b = (bb.iloc[-1] / bb.iloc[-1 - n]) - 1.0
                out.loc[t, lab] = float(ret_t - ret_b)

    if "SPY" in out.index:
        out.loc["SPY", :] = 0.0

    return out

def leaders_laggards(rs_df: pd.DataFrame, horizon: str):
    s = rs_df[horizon].dropna().sort_values(ascending=False)
    return s.head(5), s.tail(5)

home = load_home_state()

# Top bar
chip = home["regime"]
dot = regime_color(chip)

with st.container():
    c1, c2 = st.columns([2.2, 1.0], vertical_alignment="center")
    with c1:
        st.markdown(
            f"""
            <div class="me-topbar">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div>
                  <div class="me-title">Macro Engine</div>
                  <div class="me-subtle">Market regime and rotation signals</div>
                </div>
                <div class="me-chip">
                  <span class="me-dot" style="background:{dot}"></span>
                  <span>{chip}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='me-subtle' style='text-align:right;'>Last updated: {home['last_updated']}</div>",
            unsafe_allow_html=True,
        )

# Top navigation row with descriptors
nav_defs = [
    ("Home", "Overview and summary", None),
    ("Macro charts", "Rates, credit, liquidity, risk", "pages/2_Macro_Charts.py"),
    ("Weekly details", "Notes and deltas", "pages/1_Regime_Deep_Dive.py"),
    ("Drivers", "Narrative and chart links", "pages/5_Drivers.py"),
    ("Score", "Breakdown and history", "pages/1_Regime_Deep_Dive.py"),
]

nav_cols = st.columns(5, gap="small")
for i, (title, desc, path) in enumerate(nav_defs):
    with nav_cols[i]:
        with st.container(border=True):
            st.markdown(f"<div class='me-nav-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='me-nav-desc'>{desc}</div>", unsafe_allow_html=True)
            st.markdown("")
            if title == "Home":
                st.button("Open", use_container_width=True, disabled=True, key=f"topnav_{i}")
            else:
                if st.button("Open", use_container_width=True, key=f"topnav_{i}"):
                    safe_switch_page(path)

st.markdown("")

# Hero row (Key drivers moved above Regime and tilt)
h1, h2, h3, h4 = st.columns([1.0, 1.15, 1.2, 1.0], gap="large")

with h1:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Macro score</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='me-kpi'>{home['score']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
              <span class="me-pill">Confidence: {home['confidence']}</span>
              <span class="me-pill">Momentum: {home['momentum']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Open score breakdown", use_container_width=True, key="btn_score"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

with h2:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key drivers</div>", unsafe_allow_html=True)
        for i, (title, sub) in enumerate(home["drivers"], start=1):
            st.markdown(
                f"""
                <div class="me-li">
                  <div>
                    <div class="me-li-name">{i}. {title}</div>
                    <div class="me-li-sub">{sub}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")
        if st.button("Open drivers", use_container_width=True, key="btn_drivers"):
            safe_switch_page("pages/5_Drivers.py")

with h3:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>What changed this week</div>", unsafe_allow_html=True)
        for name, prev, cur, delta, label in home["weekly_changes"]:
            if delta > 0:
                arrow = "↑"
            elif delta < 0:
                arrow = "↓"
            else:
                arrow = "→"
            st.markdown(
                f"""
                <div class="me-li">
                  <div>
                    <div class="me-li-name">{name}</div>
                    <div class="me-li-sub">{prev:.2f} to {cur:.2f} ({label})</div>
                  </div>
                  <div class="me-li-right">{arrow} {abs(delta):.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")
        if st.button("Open weekly details", use_container_width=True, key="btn_weekly"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

with h4:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key risk</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:13px; color:rgba(0,0,0,0.72); line-height:1.35;'>{home['key_risk']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Open macro charts", use_container_width=True, key="btn_macro"):
            safe_switch_page("pages/2_Macro_Charts.py")

st.markdown("")

# Flows heatmap card
with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Leadership vs SPY</div>", unsafe_allow_html=True)

    rs = compute_rs_heatmap(period="1y")

    top_controls = st.columns([1.0, 1.0, 2.0, 1.2], gap="medium")
    with top_controls[0]:
        horizon = st.radio("Focus", ["1w", "1m", "3m"], horizontal=True, index=1, key="rs_focus")
    with top_controls[1]:
        view = st.radio("View", ["Heatmap", "Leaders and laggards"], horizontal=True, index=0, key="rs_view")
    with top_controls[2]:
        leaders, laggards = leaders_laggards(rs, horizon)
        ltxt = ", ".join(leaders.index.tolist()[:2]) if len(leaders) else "n a"
        gtxt = ", ".join(laggards.index.tolist()[:2]) if len(laggards) else "n a"
        st.markdown(
            f"<div class='me-subtle'>Takeaway</div><div style='font-weight:900;'>Leadership: {ltxt} | Laggards: {gtxt}</div>",
            unsafe_allow_html=True,
        )
    with top_controls[3]:
        if st.button("Open macro charts", use_container_width=True, key="btn_flows_to_macro"):
            safe_switch_page("pages/2_Macro_Charts.py")

    rs_long = rs.reset_index().melt(id_vars="index", var_name="Horizon", value_name="RS").rename(columns={"index": "Ticker"})
    rs_long = rs_long.dropna()
    rs_long = rs_long[rs_long["Horizon"] == horizon]

    if view == "Heatmap":
        heat = (
            alt.Chart(rs_long)
            .mark_rect(cornerRadius=6)
            .encode(
                x=alt.X("Horizon:N", title=None),
                y=alt.Y("Ticker:N", sort=alt.SortField(field="RS", order="descending"), title=None),
                color=alt.Color("RS:Q", title="RS"),
                tooltip=["Ticker", "Horizon", alt.Tooltip("RS:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        leaders, laggards = leaders_laggards(rs, horizon)
        cL, cR = st.columns([1, 1], gap="large")
        with cL:
            st.markdown("**Leaders**")
            for t, v in leaders.items():
                if st.button(f"{t} • {v:.3f}", use_container_width=True, key=f"lead_{horizon}_{t}"):
                    st.session_state["selected_ticker"] = t
                    safe_switch_page("pages/3_Ticker_Detail.py")
        with cR:
            st.markdown("**Laggards**")
            for t, v in laggards.items():
                if st.button(f"{t} • {v:.3f}", use_container_width=True, key=f"lag_{horizon}_{t}"):
                    st.session_state["selected_ticker"] = t
                    safe_switch_page("pages/3_Ticker_Detail.py")

st.markdown("")

# Weekly snapshot card
with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Weekly snapshot</div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Macro", "Markets"])

    macro_df = pd.DataFrame(
        {
            "Metric": ["Credit spreads", "Curve (10y minus 2y)", "Real 10y", "Dollar broad"],
            "Level": [2.86, 0.60, 1.80, 117.99],
            "Weekly change": [-0.09, -0.04, 0.03, 0.47],
        }
    )
    mkt_df = pd.DataFrame(
        {
            "Asset": ["SPY", "QQQ", "IWM", "HYG", "TLT", "DXY", "WTI", "Gold"],
            "Weekly change": [0.8, 1.2, 0.4, 0.3, -0.6, -0.5, 1.5, 0.7],
        }
    )

    with tab1:
        st.dataframe(macro_df, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(mkt_df, use_container_width=True, hide_index=True)

    st.markdown("")
    if st.button("Open weekly details", use_container_width=True, key="btn_weekly_bottom"):
        safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

# Bottom section: Regime and tilt + Why this regime
b1, b2 = st.columns([1.0, 1.2], gap="large")

with b1:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Regime and tilt</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:18px; font-weight:900; margin-bottom:10px;'>{home['regime']}</div>",
            unsafe_allow_html=True,
        )
        for k, v in home["allocation_tilt"]:
            st.markdown(
                f"""
                <div class="me-li">
                  <div>
                    <div class="me-li-name">{k}</div>
                    <div class="me-li-sub">Suggested stance</div>
                  </div>
                  <div class="me-li-right">{v}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with b2:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Why this regime</div>", unsafe_allow_html=True)

        comp = pd.DataFrame(
            {
                "Component": ["Credit stress", "Risk appetite", "Curve", "Real yields", "Dollar impulse"],
                "Contribution": [12, 8, 6, 4, -3],
            }
        ).sort_values("Contribution", ascending=False)

        ch = (
            alt.Chart(comp)
            .mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
                cornerRadiusBottomLeft=6,
                cornerRadiusBottomRight=6,
            )
            .encode(
                y=alt.Y("Component:N", sort="-x", title=None),
                x=alt.X("Contribution:Q", title=None),
                tooltip=["Component", "Contribution"],
            )
            .properties(height=240)
        )
        st.altair_chart(ch, use_container_width=True)

        st.markdown("")
        if st.button("Open component detail", use_container_width=True, key="btn_components"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

# Explore tiles
tiles = st.columns(5, gap="small")
tile_defs = [
    ("Macro charts", "Rates, credit, liquidity, risk", "pages/2_Macro_Charts.py"),
    ("Ticker drilldown", "RS, MAs, setup features", "pages/3_Ticker_Detail.py"),
    ("Weekly details", "Notes and deltas", "pages/1_Regime_Deep_Dive.py"),
    ("Drivers", "Narrative and chart links", "pages/5_Drivers.py"),
    ("Score", "Breakdown and history", "pages/1_Regime_Deep_Dive.py"),
]

for i, (t, d, path) in enumerate(tile_defs):
    with tiles[i]:
        with st.container(border=True):
            st.markdown(f"<div style='font-weight:900; font-size:13px;'>{t}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='me-subtle'>{d}</div>", unsafe_allow_html=True)
            st.markdown("")
            if st.button("Open", use_container_width=True, key=f"tile_open_{i}"):
                safe_switch_page(path)