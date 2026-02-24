# app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3

st.set_page_config(page_title="Macro Engine", layout="wide")
st.title("Macro Engine")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set. Add it as an environment variable locally and in Render.")
    st.stop()

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]
FLOWS_COLUMNS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
}

PAGE_REGIME = "pages/1_Regime_Deep_Dive.py"
PAGE_MACRO = "pages/2_Macro_Charts.py"
PAGE_ROTATION = "pages/4_Rotation_Setups.py"
PAGE_TICKER = "pages/3_Ticker_Detail.py"
PAGE_WRITING = "pages/5_Writing.py"


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


def trailing_return(series: pd.Series, n: int) -> float:
    s = series.dropna()
    if len(s) < n + 1:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1.0)


def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x*100:.2f}%"


def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


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


def build_flows_heatmap(px: pd.DataFrame, rows: list[str], base: str = "SPY") -> pd.DataFrame:
    if px is None or px.empty or base not in px.columns:
        return pd.DataFrame()

    out = []
    for t in rows:
        if t not in px.columns or t == base:
            continue
        r = {"Ticker": t}
        for label, n in FLOWS_COLUMNS.items():
            r_t = trailing_return(px[t], n)
            r_b = trailing_return(px[base], n)
            r[label] = r_t - r_b
        out.append(r)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).set_index("Ticker")
    return df


def plot_flows_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), title=title)
        return fig

    z = df.values
    y = df.index.tolist()
    x = df.columns.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            colorbar=dict(title="RS vs SPY"),
            hovertemplate="Ticker %{y}<br>Horizon %{x}<br>RS %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=45, b=20),
        title=title,
        yaxis=dict(title=""),
        xaxis=dict(title=""),
    )
    return fig


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


macro = load_macro()
proxy_prices = load_prices(ROTATION_TICKERS, period="5y")
macro_last_date = macro.dropna(how="all").index.max()

# Top summary row
left, right = st.columns([2, 1])
with left:
    regime = compute_regime_v3(
        macro=macro,
        proxies=proxy_prices,
        lookback_trend=63,
        momentum_lookback_days=21,
    )

    if regime.score_delta is None:
        momentum_txt = "Momentum: stable"
    elif regime.score_delta == 0:
        momentum_txt = "Momentum: flat"
    else:
        sign = "+" if regime.score_delta > 0 else ""
        momentum_txt = f"Momentum: {sign}{regime.score_delta} in 1m"

    st.caption("Regime")
    st.subheader(regime.label)
    st.caption(f"Score {regime.score} | Confidence {regime.confidence} | {momentum_txt}")

with right:
    last_txt = macro_last_date.date() if pd.notna(macro_last_date) else "unknown"
    st.caption("Data")
    st.write(f"Last updated: {last_txt}")
    if regime.confidence == "Low":
        st.warning("Some inputs may be missing")
    elif regime.confidence == "Medium":
        st.info("Lighter conviction")

st.divider()

# Hero flows heatmap
st.subheader("Flows")
flows_df = build_flows_heatmap(proxy_prices, [t for t in ROTATION_TICKERS if t != "SPY"], base="SPY")
st.plotly_chart(plot_flows_heatmap(flows_df, "Relative strength vs SPY"), use_container_width=True)

# Drivers + weekly changes, kept short and clean, with clear jump buttons
col_a, col_b = st.columns([1, 1])

with col_a:
    st.caption("Drivers")
    make_chip_row(regime.favored_groups[:8])

    st.caption("Quick read")
    stance = regime.allocation.get("stance", {}) if isinstance(regime.allocation, dict) else {}
    decision = (
        f"Equities {stance.get('Equities','n/a')}, "
        f"Credit {stance.get('Credit','n/a')}, "
        f"Duration {stance.get('Duration','n/a')}, "
        f"USD {stance.get('USD','n/a')}, "
        f"Commodities {stance.get('Commodities','n/a')}"
    )
    st.write(decision)

    if st.button("Open drivers and allocation", use_container_width=True):
        st.switch_page(PAGE_REGIME)

with col_b:
    st.caption("Weekly changes")

    bullets = []

    days_look = 7
    if "hy_oas" in macro.columns:
        latest, prev, dlt = delta_over_days(macro["hy_oas"], days_look)
        if dlt is not None:
            bullets.append(f"Credit spreads {fmt_num(latest)} vs {fmt_num(prev)} change {fmt_num(dlt)}")

    if "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        latest, prev, dlt = delta_over_days(curve, days_look)
        if dlt is not None:
            bullets.append(f"Curve {fmt_num(latest)} vs {fmt_num(prev)} change {fmt_num(dlt)}")

    if "real10" in macro.columns:
        latest, prev, dlt = delta_over_days(macro["real10"], days_look)
        if dlt is not None:
            bullets.append(f"Real 10y {fmt_num(latest)} vs {fmt_num(prev)} change {fmt_num(dlt)}")

    if "dollar_broad" in macro.columns:
        latest, prev, dlt = delta_over_days(macro["dollar_broad"], days_look)
        if dlt is not None:
            bullets.append(f"Dollar broad {fmt_num(latest)} vs {fmt_num(prev)} change {fmt_num(dlt)}")

    if not bullets:
        bullets = ["No weekly deltas available yet"]

    for b in bullets[:4]:
        st.write(f"• {b}")

    if st.button("Open weekly details", use_container_width=True):
        st.switch_page(PAGE_REGIME)

st.divider()

# Explore section (signals depth without calling it a table of contents)
st.subheader("Explore")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.caption("Regime deep dive")
    st.write("Score history, components, allocation")
    if st.button("Open", key="go_regime", use_container_width=True):
        st.switch_page(PAGE_REGIME)

with c2:
    st.caption("Macro charts")
    st.write("Rates, credit, liquidity, risk")
    if st.button("Open", key="go_macro", use_container_width=True):
        st.switch_page(PAGE_MACRO)

with c3:
    st.caption("Rotation and setups")
    st.write("Leaders, laggards, heatmaps")
    if st.button("Open", key="go_rotation", use_container_width=True):
        st.switch_page(PAGE_ROTATION)

with c4:
    st.caption("Ticker detail")
    st.write("Price, RS vs SPY, features")
    if st.button("Open", key="go_ticker", use_container_width=True):
        st.switch_page(PAGE_TICKER)

with c5:
    st.caption("Writing")
    st.write("Draft notes for your weekly post")
    if st.button("Open", key="go_writing", use_container_width=True):
        st.switch_page(PAGE_WRITING)