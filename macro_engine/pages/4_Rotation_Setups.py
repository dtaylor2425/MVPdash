# pages/4_Rotation_Setups.py
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_sources import fetch_prices

st.set_page_config(page_title="Rotation and setups", layout="wide", initial_sidebar_state="expanded")


ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]
TRADING_DAYS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container {
          max-width: 1200px;
          padding-top: 4.8rem;
          padding-bottom: 5rem;
        }

        .stButton > button {
          border-radius: 12px !important;
          padding: 0.55rem 0.9rem !important;
          border: 1px solid rgba(0,0,0,0.10) !important;
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }

        .me-topbar {
          position: sticky;
          top: 1.5rem;
          z-index: 999;
          background: rgba(255,255,255,0.92);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 16px;
          padding: 14px 16px;
          margin-bottom: 16px;
        }

        .me-title {
          font-size: 26px;
          font-weight: 900;
          letter-spacing: -0.6px;
          margin: 0;
          line-height: 1.05;
          color: rgba(0,0,0,0.90);
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
          color: rgba(0,0,0,0.85);
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
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def safe_switch_page(path: str):
    try:
        st.switch_page(path)
    except Exception:
        st.error(f"Missing page: {path}. Check your pages folder.")


def sidebar_nav(active: str = "Rotation and setups"):
    st.sidebar.title("Macro Engine")
    st.sidebar.caption("Navigation")

    pages = {
        "Home": "app.py",
        "Macro charts": "pages/2_Macro_Charts.py",
        "Regime deep dive": "pages/1_Regime_Deep_Dive.py",
        "Rotation and setups": "pages/4_Rotation_Setups.py",
        "Drivers": "pages/5_Drivers.py",
        "Ticker drilldown": "pages/3_Ticker_Detail.py",
    }

    keys = list(pages.keys())
    idx = keys.index(active) if active in pages else 0
    choice = st.sidebar.selectbox("Go to", keys, index=idx, label_visibility="collapsed")
    if choice != active:
        safe_switch_page(pages[choice])


def trailing_return(series: pd.Series, n: int) -> float:
    s = series.dropna()
    if len(s) < n + 1:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1.0)


def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def _nearest_before(series: pd.Series, dt: pd.Timestamp):
    s = series.dropna()
    if s.empty:
        return None
    idx = s.index[s.index <= dt]
    if len(idx) == 0:
        return None
    return pd.Timestamp(idx.max())


def pct_return_over_days(series: pd.Series, days: int):
    s = series.dropna()
    if len(s) < days + 2:
        return None
    end = s.index.max()
    prev_dt = end - pd.Timedelta(days=days)
    end_i = _nearest_before(s, end)
    prev_i = _nearest_before(s, prev_dt)
    if end_i is None or prev_i is None:
        return None
    a = float(s.loc[end_i])
    b = float(s.loc[prev_i])
    if b == 0:
        return None
    return (a / b) - 1.0


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    df = fetch_prices(tickers, period=period)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_index()


def rs_over(px_df: pd.DataFrame, t: str, base: str, n: int) -> float:
    if t not in px_df.columns or base not in px_df.columns:
        return float("nan")
    return trailing_return(px_df[t], n) - trailing_return(px_df[base], n)


def compression_flag(series: pd.Series) -> int:
    r = series.dropna().pct_change().dropna()
    if len(r) < 126:
        return 0
    vol_recent = r.iloc[-63:].std()
    vol_prior = r.iloc[-126:-63].std()
    if pd.isna(vol_recent) or pd.isna(vol_prior) or vol_prior == 0:
        return 0
    return int(vol_recent < 0.7 * vol_prior)


def build_rotation_table(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ROTATION_TICKERS:
        if t not in px.columns:
            continue
        if t == "SPY":
            continue
        r = {"Ticker": t}
        for k, n in TRADING_DAYS.items():
            r[k] = trailing_return(px[t], n)
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Ticker")


def build_setups_table(px_u: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    out = []
    for t in universe:
        if t not in px_u.columns or "SPY" not in px_u.columns:
            continue
        s = px_u[t].dropna()
        if s.empty:
            continue

        rs3 = rs_over(px_u, t, "SPY", 63)
        rs6 = rs_over(px_u, t, "SPY", 126)

        ma200 = s.rolling(200).mean()
        trend = int(len(ma200.dropna()) > 0 and s.iloc[-1] > ma200.iloc[-1])

        comp = compression_flag(s)

        r21 = pct_return_over_days(s, 21)
        r63 = pct_return_over_days(s, 63)

        vol_recent = s.pct_change().iloc[-63:].std()
        vol_prior = s.pct_change().iloc[-252:-63].std() if len(s) >= 252 else np.nan
        vol_ratio = float(vol_recent / vol_prior) if (vol_prior is not None and not pd.isna(vol_prior) and vol_prior != 0) else np.nan

        score = 100.0 * (0.6 * rs6 + 0.4 * rs3)
        score += 15.0 if trend == 1 else 0.0
        score += 10.0 if comp == 1 else 0.0

        out.append(
            {
                "Ticker": t,
                "Score": round(float(score), 2),
                "RS 3m": float(rs3),
                "RS 6m": float(rs6),
                "1m return": np.nan if r21 is None else float(r21),
                "3m return": np.nan if r63 is None else float(r63),
                "Trend flag": int(trend),
                "Trend": "Above trend" if trend == 1 else "Below trend",
                "Compression flag": int(comp),
                "Volatility": "Compressed" if comp == 1 else "Not compressed",
                "Vol ratio": vol_ratio,
            }
        )

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def sparkline_chart(px_df: pd.DataFrame, ticker: str, days: int = 365):
    if px_df is None or px_df.empty or ticker not in px_df.columns:
        st.caption("No price history for selection.")
        return
    s = px_df[ticker].dropna()
    if s.empty:
        st.caption("No price history for selection.")
        return

    end = s.index.max()
    start = end - pd.Timedelta(days=days)
    view = s.loc[s.index >= start].copy()
    if view.empty:
        view = s.tail(252).copy()
    base = float(view.iloc[0])
    if base == 0:
        return
    ret = view / base - 1.0
    df = pd.DataFrame({"Date": ret.index, "Return": ret.values})

    ch = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Return:Q", title=None, axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(height=160)
    )
    st.altair_chart(ch, use_container_width=True)


sidebar_nav(active="Rotation and setups")

px = load_prices(ROTATION_TICKERS, period="5y")

top_left, top_right = st.columns([3, 1], vertical_alignment="center")
with top_left:
    st.markdown(
        """
        <div class="me-topbar">
          <div>
            <div class="me-title">Rotation and setups</div>
            <div class="me-subtle">Capital rotation snapshots, relative strength, trend, and volatility compression</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_right:
    if st.button("Back to home", use_container_width=True):
        safe_switch_page("app.py")

st.markdown("")

if px.empty:
    st.info("No price data available.")
    st.stop()

rot = build_rotation_table(px)
if rot.empty:
    st.info("No rotation table available.")
    st.stop()

st.markdown("<div class='me-rowtitle'>Capital rotation</div>", unsafe_allow_html=True)

cA, cB, cC, cD = st.columns([1.1, 1.1, 1.1, 1.1], gap="small")
with cA:
    focus_window = st.selectbox("Focus window", options=list(TRADING_DAYS.keys()), index=1)
with cB:
    order_by = st.selectbox("Order by", options=["Return", "Return vs SPY"], index=1)
with cC:
    show_vs_spy = st.checkbox("Show vs SPY", value=True)
with cD:
    show_heatmap = st.checkbox("Show heatmap", value=True)

rot_display = rot.copy()

if show_vs_spy and "SPY" in px.columns:
    for k, n in TRADING_DAYS.items():
        rot_display[k] = rot_display[k] - trailing_return(px["SPY"], n)

focus = rot_display[[focus_window]].copy()
focus.columns = ["Value"]
focus = focus.sort_values("Value", ascending=False)

leaders = focus.head(3)
laggards = focus.tail(3)

chips = []
if not leaders.empty:
    chips.append(f"Leaders ({focus_window}): " + ", ".join([f"{i} {leaders.loc[i,'Value']*100:.1f}%" for i in leaders.index]))
if not laggards.empty:
    chips.append(f"Laggards ({focus_window}): " + ", ".join([f"{i} {laggards.loc[i,'Value']*100:.1f}%" for i in laggards.index]))
for t in chips:
    st.markdown(f"<span class='me-pill'>{t}</span>", unsafe_allow_html=True)

rot_tbl = rot_display.copy()
rot_tbl = rot_tbl.reset_index()
for c in list(TRADING_DAYS.keys()):
    if c in rot_tbl.columns:
        rot_tbl[c] = pd.to_numeric(rot_tbl[c], errors="coerce")

if order_by == "Return":
    rot_tbl = rot_tbl.sort_values(focus_window, ascending=False)
else:
    rot_tbl = rot_tbl.sort_values(focus_window, ascending=False)

rot_tbl_disp = rot_tbl.copy()
for c in list(TRADING_DAYS.keys()):
    if c in rot_tbl_disp.columns:
        rot_tbl_disp[c] = rot_tbl_disp[c].apply(fmt_pct)

left, right = st.columns([1.6, 1.0], gap="large")
with left:
    st.dataframe(rot_tbl_disp, use_container_width=True, hide_index=True)
with right:
    st.markdown("<div class='me-rowtitle'>Rotation bar view</div>", unsafe_allow_html=True)
    bar_df = rot_tbl[["Ticker", focus_window]].rename(columns={focus_window: "Return"})
    bar_df["Return"] = pd.to_numeric(bar_df["Return"], errors="coerce")
    bar_df = bar_df.dropna().sort_values("Return", ascending=False)

    bar = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, cornerRadiusBottomLeft=6, cornerRadiusBottomRight=6)
        .encode(
            y=alt.Y("Ticker:N", sort="-x", title=None),
            x=alt.X("Return:Q", title=None, axis=alt.Axis(format="%")),
            tooltip=["Ticker", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(bar, use_container_width=True)

if show_heatmap:
    st.markdown("")
    st.markdown("<div class='me-rowtitle'>Rotation heatmap</div>", unsafe_allow_html=True)
    hm = rot_display.copy().reset_index().melt(id_vars="Ticker", var_name="Horizon", value_name="Return")
    hm["Return"] = pd.to_numeric(hm["Return"], errors="coerce")
    hm = hm.dropna()

    heat = (
        alt.Chart(hm)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("Horizon:N", title=None, sort=list(TRADING_DAYS.keys())),
            y=alt.Y("Ticker:N", title=None, sort=alt.SortField(field="Return", order="descending")),
            color=alt.Color("Return:Q", title="Return", scale=alt.Scale(scheme="redblue")),
            tooltip=["Ticker", "Horizon", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(heat, use_container_width=True)

st.divider()

st.markdown("<div class='me-rowtitle'>Top setups</div>", unsafe_allow_html=True)

universe = []
try:
    universe_path = Path(os.getcwd()) / "data" / "universe.csv"
    if universe_path.exists():
        df_u = pd.read_csv(universe_path)
        if "ticker" in df_u.columns:
            universe = df_u["ticker"].dropna().astype(str).str.upper().unique().tolist()
except Exception:
    universe = []

if not universe:
    st.info("No universe.csv found in data folder.")
    st.stop()

px_u = load_prices(list(dict.fromkeys(universe + ["SPY"])), period="5y")
if px_u.empty:
    st.info("No price data for universe.")
    st.stop()

df = build_setups_table(px_u, universe)
if df.empty:
    st.info("No setups could be computed for this universe.")
    st.stop()

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.2, 1.0, 1.0], gap="small")
with ctrl1:
    min_score = st.slider("Min score", min_value=float(df["Score"].min()), max_value=float(df["Score"].max()), value=float(df["Score"].quantile(0.6)))
with ctrl2:
    trend_only = st.selectbox("Trend filter", options=["All", "Above trend only", "Below trend only"], index=1)
with ctrl3:
    comp_only = st.selectbox("Compression", options=["All", "Compressed only", "Not compressed only"], index=0)
with ctrl4:
    top_n = st.selectbox("Show", options=[25, 50, 100], index=1)

flt = df[df["Score"] >= min_score].copy()
if trend_only == "Above trend only":
    flt = flt[flt["Trend flag"] == 1]
elif trend_only == "Below trend only":
    flt = flt[flt["Trend flag"] == 0]

if comp_only == "Compressed only":
    flt = flt[flt["Compression flag"] == 1]
elif comp_only == "Not compressed only":
    flt = flt[flt["Compression flag"] == 0]

flt = flt.head(int(top_n)).copy()

pick_row = st.columns([2.2, 1.0, 6.0])
with pick_row[0]:
    pick = st.selectbox("Ticker", options=flt["Ticker"].tolist(), index=0, label_visibility="collapsed")
with pick_row[1]:
    if st.button("Open", use_container_width=True):
        st.session_state["selected_ticker"] = pick
        safe_switch_page("pages/3_Ticker_Detail.py")
with pick_row[2]:
    st.caption("Opens the ticker drilldown page for the selected setup.")

k1, k2, k3, k4 = st.columns(4, gap="small")
sel = df[df["Ticker"] == pick].head(1)
if not sel.empty:
    k1.metric("Score", f"{float(sel['Score'].iloc[0]):.2f}")
    k2.metric("RS 6m", fmt_pct(float(sel["RS 6m"].iloc[0])))
    k3.metric("Trend", str(sel["Trend"].iloc[0]))
    vr = sel["Vol ratio"].iloc[0]
    k4.metric("Vol ratio", "" if pd.isna(vr) else f"{float(vr):.2f}")

left, right = st.columns([1.3, 1.0], gap="large")
with left:
    disp = flt.copy()
    disp["RS 3m"] = disp["RS 3m"].apply(fmt_pct)
    disp["RS 6m"] = disp["RS 6m"].apply(fmt_pct)
    disp["1m return"] = disp["1m return"].apply(fmt_pct)
    disp["3m return"] = disp["3m return"].apply(fmt_pct)
    st.dataframe(
        disp[["Ticker", "Score", "RS 3m", "RS 6m", "1m return", "3m return", "Trend", "Volatility"]],
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown("<div class='me-rowtitle'>Setup scatter</div>", unsafe_allow_html=True)
    sc = flt.copy()
    sc["RS 6m"] = pd.to_numeric(sc["RS 6m"], errors="coerce")
    sc["Vol ratio"] = pd.to_numeric(sc["Vol ratio"], errors="coerce")
    sc = sc.dropna(subset=["RS 6m", "Vol ratio"])

    scatter = (
        alt.Chart(sc)
        .mark_circle(size=110, opacity=0.8)
        .encode(
            x=alt.X("RS 6m:Q", title="RS 6m vs SPY", axis=alt.Axis(format="%")),
            y=alt.Y("Vol ratio:Q", title="Vol ratio (recent vs prior)"),
            tooltip=[
                "Ticker",
                alt.Tooltip("Score:Q", format=".2f"),
                alt.Tooltip("RS 6m:Q", format=".2%"),
                alt.Tooltip("Vol ratio:Q", format=".2f"),
                "Trend",
                "Volatility",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)

st.markdown("")
st.markdown("<div class='me-rowtitle'>Selected setup price action</div>", unsafe_allow_html=True)
sparkline_chart(px_u, pick, days=365)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
