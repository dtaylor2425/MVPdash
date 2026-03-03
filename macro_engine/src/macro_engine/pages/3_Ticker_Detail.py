# pages/3_Ticker_Detail.py
import pandas as pd
import streamlit as st

from src.data_sources import fetch_prices
from src.compute import weekly_pct_change
from src.paths import data_path
from src.ranges import RANGES, slice_df

st.set_page_config(page_title="Ticker detail", layout="wide")
st.title("Ticker detail")

# Navigation
nav_l, nav_r = st.columns([1, 6])
with nav_l:
    if st.button("Back to home"):
        st.switch_page("app.py")
with nav_r:
    st.caption("Open a ticker from Top setups, or pick one here.")

# Load universe (robust path)
universe_file = data_path("universe.csv")
try:
    df_u = pd.read_csv(universe_file)
    universe = df_u["ticker"].dropna().astype(str).str.upper().unique().tolist()
except Exception:
    universe = []

# Remove SPY to avoid SPY vs SPY edge case
universe = [t for t in universe if t != "SPY"]
if not universe:
    universe = ["QQQ", "IWM"]

default_ticker = st.session_state.get("selected_ticker", universe[0])
if default_ticker not in universe:
    default_ticker = universe[0]

ticker = st.selectbox("Ticker", options=universe, index=universe.index(default_ticker))
base = "SPY"

# Fetch prices for ticker and base
tickers_to_fetch = list(dict.fromkeys([ticker, base]))
px = fetch_prices(tickers_to_fetch, period="5y")

if px is None or px.empty:
    st.error("No price data returned. Try another ticker or check your data source.")
    st.stop()

if ticker not in px.columns or base not in px.columns:
    st.error("Price data missing for this ticker or SPY. Try another ticker.")
    st.write("Columns returned:", list(px.columns))
    st.stop()

s_full = px[ticker].dropna()
b_full = px[base].dropna()

idx_full = s_full.index.intersection(b_full.index)
if len(idx_full) < 260:
    st.error("Not enough history to compute features. Pick a different ticker.")
    st.stop()

s_full = s_full.loc[idx_full]
b_full = b_full.loc[idx_full]

ratio_full = (s_full / b_full).dropna()

# Feature metrics (kept fixed windows for comparability)
last = float(s_full.iloc[-1])
wk = weekly_pct_change(s_full, n=5)
mo = weekly_pct_change(s_full, n=21)

rs_3m = float(s_full.pct_change(63).iloc[-1] - b_full.pct_change(63).iloc[-1])
rs_6m = float(s_full.pct_change(126).iloc[-1] - b_full.pct_change(126).iloc[-1])

r = s_full.pct_change().dropna()
if len(r) >= 126:
    vol_recent = r.iloc[-63:].std()
    vol_prior = r.iloc[-126:-63].std()
    compression_flag = int(vol_recent < 0.7 * vol_prior)
else:
    compression_flag = 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last", f"{last:.2f}")
m2.metric("Week", "n/a" if wk is None else f"{wk*100:.2f}%")
m3.metric("Month", "n/a" if mo is None else f"{mo*100:.2f}%")
m4.metric("RS 3m", f"{rs_3m*100:.2f}%")
m5.metric("RS 6m", f"{rs_6m*100:.2f}%")

st.divider()

# Per chart time controls
ctrl_l, ctrl_r = st.columns([1, 1])

with ctrl_l:
    price_range = st.selectbox(
        "Price chart range",
        options=list(RANGES.keys()),
        index=list(RANGES.keys()).index("1y"),
        key="ticker_price_range",
    )
with ctrl_r:
    rs_range = st.selectbox(
        "Relative strength chart range",
        options=list(RANGES.keys()),
        index=list(RANGES.keys()).index("1y"),
        key="ticker_rs_range",
    )

# Moving averages adapt to the price chart range
spec_price = RANGES[price_range]
ma_fast_full = s_full.rolling(spec_price.ma_fast).mean()
ma_slow_full = s_full.rolling(spec_price.ma_slow).mean()

trend_flag = int(s_full.iloc[-1] > ma_slow_full.iloc[-1])

# Build plot frames and slice them so y-axis auto fits the selected time window
price_plot = pd.DataFrame(
    {
        ticker: s_full,
        f"MA{spec_price.ma_fast}": ma_fast_full,
        f"MA{spec_price.ma_slow}": ma_slow_full,
    }
)
price_plot = slice_df(price_plot, price_range)

ratio_plot = pd.DataFrame({f"{ticker}/SPY": ratio_full})
ratio_plot = slice_df(ratio_plot, rs_range)

# Charts
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Price")
    st.caption(f"MA windows: {spec_price.ma_fast} and {spec_price.ma_slow}")
    st.line_chart(price_plot)

with c2:
    st.subheader("Relative strength vs SPY")
    st.line_chart(ratio_plot)

st.divider()

st.subheader("Setup features")
feat_rows = [
    {"Metric": "Trend", "Value": "Bullish" if trend_flag == 1 else "Bearish"},
    {"Metric": "Volatility", "Value": "Compressed" if compression_flag == 1 else "Normal"},
    {"Metric": "RS 3m vs SPY", "Value": f"{rs_3m*100:.2f}%"},
    {"Metric": "RS 6m vs SPY", "Value": f"{rs_6m*100:.2f}%"},
]
st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)