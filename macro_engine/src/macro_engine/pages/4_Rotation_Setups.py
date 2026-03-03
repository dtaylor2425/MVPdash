# pages/4_Rotation_Setups.py
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_sources import fetch_prices
from src.ui import inject_css, sidebar_nav, safe_switch_page

st.set_page_config(page_title="Rotation and setups", layout="wide", initial_sidebar_state="expanded")

inject_css()
sidebar_nav(active="Rotation & setups")

# ── Constants ─────────────────────────────────────────────────────────────────

ROTATION_TICKERS = [
    "XLE", "XLF", "XLK", "XLI", "XLP", "XLV",
    "GLD", "UUP", "IWM", "QQQ",
    "IGV", "SMH",   # software and semiconductors
    "SPY",
]
TRADING_DAYS     = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}

# ── Helpers ───────────────────────────────────────────────────────────────────

def trailing_return(series: pd.Series, n: int) -> float:
    s = series.dropna()
    return float("nan") if len(s) < n + 1 else float(s.iloc[-1] / s.iloc[-(n+1)] - 1.0)


def fmt_pct(x) -> str:
    return "" if (x is None or pd.isna(x)) else f"{x * 100:.2f}%"


def _nearest_before(series: pd.Series, dt: pd.Timestamp):
    s = series.dropna()
    if s.empty:
        return None
    idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None


def pct_return_over_days(series: pd.Series, days: int):
    s = series.dropna()
    if len(s) < days + 2:
        return None
    end_i  = _nearest_before(s, s.index.max())
    prev_i = _nearest_before(s, s.index.max() - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None
    a, b = float(s.loc[end_i]), float(s.loc[prev_i])
    return None if b == 0 else (a / b) - 1.0


def compression_flag(series: pd.Series) -> int:
    r = series.dropna().pct_change().dropna()
    if len(r) < 126:
        return 0
    vol_recent = r.iloc[-63:].std()
    vol_prior  = r.iloc[-126:-63].std()
    if pd.isna(vol_recent) or pd.isna(vol_prior) or vol_prior == 0:
        return 0
    return int(vol_recent < 0.7 * vol_prior)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_prices(tickers: list, period: str = "5y") -> pd.DataFrame:
    df = fetch_prices(tickers, period=period)
    return pd.DataFrame() if (df is None or df.empty) else df.sort_index()


def rs_over(px_df, t, base, n) -> float:
    if t not in px_df.columns or base not in px_df.columns:
        return float("nan")
    return trailing_return(px_df[t], n) - trailing_return(px_df[base], n)


def build_rotation_table(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ROTATION_TICKERS:
        if t not in px.columns or t == "SPY":
            continue
        r = {"Ticker": t}
        for k, n in TRADING_DAYS.items():
            r[k] = trailing_return(px[t], n)
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Ticker")


def build_setups_table(px: pd.DataFrame, universe: list) -> pd.DataFrame:
    rows = []
    if "SPY" not in px.columns:
        return pd.DataFrame()
    for t in universe:
        if t == "SPY" or t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) < 260:
            continue
        rs_3m = float(s.pct_change(63).iloc[-1]  - px["SPY"].pct_change(63).iloc[-1])
        rs_6m = float(s.pct_change(126).iloc[-1] - px["SPY"].pct_change(126).iloc[-1])
        ma200 = s.rolling(200).mean().iloc[-1]
        trend = int(s.iloc[-1] > ma200)
        comp  = compression_flag(s)
        r_1m  = pct_return_over_days(s, 21)
        r_3m  = pct_return_over_days(s, 63)

        r = s.pct_change().dropna()
        vol_ratio = float("nan")
        if len(r) >= 126:
            vol_recent = r.iloc[-63:].std()
            vol_prior  = r.iloc[-126:-63].std()
            if vol_prior and not pd.isna(vol_prior):
                vol_ratio = float(vol_recent / vol_prior)

        score = 50 + 200 * rs_3m + 150 * rs_6m + 10 * trend + 10 * comp
        rows.append({
            "Ticker":           t,
            "Score":            float(score),
            "RS 3m":            rs_3m,
            "RS 6m":            rs_6m,
            "1m return":        r_1m if r_1m is not None else float("nan"),
            "3m return":        r_3m if r_3m is not None else float("nan"),
            "Trend flag":       trend,
            "Compression flag": comp,
            "Trend":            "Bullish" if trend == 1 else "Bearish",
            "Volatility":       "Compressed" if comp == 1 else "Normal",
            "Vol ratio":        vol_ratio,
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False) if rows else pd.DataFrame()


def sparkline_chart(px_df: pd.DataFrame, ticker: str, days: int = 365):
    if ticker not in px_df.columns:
        st.caption("No price data.")
        return
    s   = px_df[ticker].dropna()
    end = s.index.max()
    s   = s.loc[s.index >= end - pd.Timedelta(days=days)]
    df  = pd.DataFrame({"Date": s.index, "Return": (s / s.iloc[0] - 1.0).values})
    ch  = (
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

# ── Data ──────────────────────────────────────────────────────────────────────

px = load_prices(ROTATION_TICKERS, period="5y")

# ── Topbar ────────────────────────────────────────────────────────────────────

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

# ═══════════════════════════════════════════════════════════════════════════════
# ROTATION CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='me-rowtitle'>Capital rotation</div>", unsafe_allow_html=True)

cA, cB, cC, cD = st.columns([1.1, 1.1, 1.1, 1.1], gap="small")
with cA:
    focus_window = st.selectbox("Focus window", options=list(TRADING_DAYS.keys()), index=1)
with cB:
    # FIX: was copy-paste bug — both branches sorted the same column.
    order_by = st.selectbox("Order by", options=["Return vs SPY", "Absolute return"], index=0)
with cC:
    show_vs_spy = st.checkbox("Show vs SPY", value=True)
with cD:
    show_heatmap = st.checkbox("Show heatmap", value=True)

rot_display = rot.copy()
if show_vs_spy and "SPY" in px.columns:
    for k, n in TRADING_DAYS.items():
        rot_display[k] = rot_display[k] - trailing_return(px["SPY"], n)

# FIX: now actually switches sort key between vs-SPY and absolute return
if order_by == "Return vs SPY":
    rot_sorted = rot_display.sort_values(focus_window, ascending=False)
else:
    rot_sorted = rot.sort_values(focus_window, ascending=False)

leaders  = rot_sorted.head(3)
laggards = rot_sorted.tail(3)

chips = []
if not leaders.empty:
    chips.append(
        f"Leaders ({focus_window}): "
        + ", ".join(f"{i} {leaders.loc[i, focus_window]*100:.1f}%" for i in leaders.index)
    )
if not laggards.empty:
    chips.append(
        f"Laggards ({focus_window}): "
        + ", ".join(f"{i} {laggards.loc[i, focus_window]*100:.1f}%" for i in laggards.index)
    )
for t in chips:
    st.markdown(f"<span class='me-pill'>{t}</span>", unsafe_allow_html=True)
st.markdown("")

# Format for display
rot_tbl_disp = rot_sorted.reset_index().copy()
for c in list(TRADING_DAYS.keys()):
    if c in rot_tbl_disp.columns:
        rot_tbl_disp[c] = pd.to_numeric(rot_tbl_disp[c], errors="coerce").apply(fmt_pct)

left, right = st.columns([1.6, 1.0], gap="large")
with left:
    st.dataframe(rot_tbl_disp, use_container_width=True, hide_index=True)

with right:
    st.markdown("<div class='me-rowtitle'>Rotation bar view</div>", unsafe_allow_html=True)
    bar_df = rot_sorted.reset_index()[["Ticker", focus_window]].rename(columns={focus_window: "Return"})
    bar_df["Return"] = pd.to_numeric(bar_df["Return"], errors="coerce")
    bar_df = bar_df.dropna().sort_values("Return", ascending=False)

    bar = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6,
                  cornerRadiusBottomLeft=6, cornerRadiusBottomRight=6)
        .encode(
            y=alt.Y("Ticker:N", sort="-x", title=None),
            x=alt.X("Return:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.condition(
                alt.datum.Return > 0,
                alt.value("#1f7a4f"),
                alt.value("#b42318"),
            ),
            tooltip=["Ticker", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(bar, use_container_width=True)

if show_heatmap:
    st.markdown("")
    st.markdown("<div class='me-rowtitle'>Rotation heatmap</div>", unsafe_allow_html=True)
    hm = rot_display.reset_index().melt(id_vars="Ticker", var_name="Horizon", value_name="Return")
    hm["Return"] = pd.to_numeric(hm["Return"], errors="coerce")
    hm = hm.dropna()
    heat = (
        alt.Chart(hm)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("Horizon:N", title=None, sort=list(TRADING_DAYS.keys())),
            y=alt.Y("Ticker:N",  title=None, sort=alt.SortField(field="Return", order="descending")),
            color=alt.Color("Return:Q", title="Return", scale=alt.Scale(scheme="redblue")),
            tooltip=["Ticker", "Horizon", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(heat, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TOP SETUPS
# ═══════════════════════════════════════════════════════════════════════════════

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
    st.info("No universe.csv found in data/ folder.")
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
    min_score = st.slider(
        "Min score",
        min_value=float(df["Score"].min()),
        max_value=float(df["Score"].max()),
        value=float(df["Score"].quantile(0.6)),
    )
with ctrl2:
    trend_only = st.selectbox("Trend filter", ["All", "Above trend only", "Below trend only"], index=1)
with ctrl3:
    comp_only = st.selectbox("Compression", ["All", "Compressed only", "Not compressed only"], index=0)
with ctrl4:
    top_n = st.selectbox("Show", [25, 50, 100], index=1)

flt = df[df["Score"] >= min_score].copy()
if trend_only == "Above trend only":
    flt = flt[flt["Trend flag"] == 1]
elif trend_only == "Below trend only":
    flt = flt[flt["Trend flag"] == 0]
if comp_only == "Compressed only":
    flt = flt[flt["Compression flag"] == 1]
elif comp_only == "Not compressed only":
    flt = flt[flt["Compression flag"] == 0]
flt = flt.head(int(top_n))

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
    k1.metric("Score",   f"{float(sel['Score'].iloc[0]):.2f}")
    k2.metric("RS 6m",   fmt_pct(float(sel["RS 6m"].iloc[0])))
    k3.metric("Trend",   str(sel["Trend"].iloc[0]))
    vr = sel["Vol ratio"].iloc[0]
    k4.metric("Vol ratio", "" if pd.isna(vr) else f"{float(vr):.2f}")

left, right = st.columns([1.3, 1.0], gap="large")
with left:
    disp = flt.copy()
    for col in ["RS 3m", "RS 6m", "1m return", "3m return"]:
        disp[col] = disp[col].apply(fmt_pct)
    st.dataframe(
        disp[["Ticker", "Score", "RS 3m", "RS 6m", "1m return", "3m return", "Trend", "Volatility"]],
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown("<div class='me-rowtitle'>Setup scatter</div>", unsafe_allow_html=True)
    sc = flt.copy()
    sc["RS 6m"]    = pd.to_numeric(sc["RS 6m"],    errors="coerce")
    sc["Vol ratio"] = pd.to_numeric(sc["Vol ratio"], errors="coerce")
    sc = sc.dropna(subset=["RS 6m", "Vol ratio"])

    scatter = (
        alt.Chart(sc)
        .mark_circle(size=110, opacity=0.8)
        .encode(
            x=alt.X("RS 6m:Q",    title="RS 6m vs SPY",                axis=alt.Axis(format="%")),
            y=alt.Y("Vol ratio:Q", title="Vol ratio (recent vs prior)"),
            color=alt.condition(
                alt.datum["RS 6m"] > 0,
                alt.value("#1f7a4f"),
                alt.value("#b42318"),
            ),
            tooltip=[
                "Ticker",
                alt.Tooltip("Score:Q",    format=".2f"),
                alt.Tooltip("RS 6m:Q",    format=".2%"),
                alt.Tooltip("Vol ratio:Q", format=".2f"),
                "Trend", "Volatility",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)

st.markdown("")
st.markdown("<div class='me-rowtitle'>Selected setup price action</div>", unsafe_allow_html=True)
sparkline_chart(px_u, pick, days=365)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)