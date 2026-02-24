# pages/4_Rotation_Setups.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_sources import fetch_prices

st.set_page_config(page_title="Rotation and setups", layout="wide")
st.title("Rotation and setups")

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]
TRADING_DAYS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}


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


if st.button("Back to home", use_container_width=True):
    st.switch_page("app.py")

px = load_prices(ROTATION_TICKERS, period="5y")

st.subheader("Capital rotation")
rows = []
for t in ROTATION_TICKERS:
    if t not in px.columns or t == "SPY":
        continue
    r = {"Ticker": t}
    for k, n in TRADING_DAYS.items():
        r[k] = trailing_return(px[t], n)
    rows.append(r)

rot = pd.DataFrame(rows).set_index("Ticker")
rot_disp = rot.copy()
for c in rot_disp.columns:
    rot_disp[c] = rot_disp[c].apply(fmt_pct)
st.dataframe(rot_disp.reset_index(), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Top setups")
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
    return int(vol_recent < 0.7 * vol_prior)

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

    score = 100.0 * (0.6 * rs6 + 0.4 * rs3)
    score += 15.0 if trend == 1 else 0.0
    score += 10.0 if comp == 1 else 0.0

    out.append(
        {
            "Ticker": t,
            "Score": round(float(score), 2),
            "RS 3m": rs3,
            "RS 6m": rs6,
            "Trend": "Above trend" if trend == 1 else "Below trend",
            "Volatility": "Compressed" if comp == 1 else "Not compressed",
        }
    )

df = pd.DataFrame(out).sort_values("Score", ascending=False).head(50).reset_index(drop=True)
df_disp = df.copy()
df_disp["RS 3m"] = df_disp["RS 3m"].apply(fmt_pct)
df_disp["RS 6m"] = df_disp["RS 6m"].apply(fmt_pct)

left, mid, right = st.columns([2, 1, 6])
with left:
    pick = st.selectbox("Ticker", options=df["Ticker"].tolist(), label_visibility="collapsed")
with mid:
    if st.button("Open", use_container_width=True):
        st.session_state["selected_ticker"] = pick
        st.switch_page("pages/3_Ticker_Detail.py")
with right:
    st.caption("Open the ticker detail page for the selected ticker.")

st.dataframe(df_disp, use_container_width=True, hide_index=True)