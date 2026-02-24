# pages/2_Macro.py
import streamlit as st
import pandas as pd

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.ranges import RANGES, slice_df
from src.charts import line_chart, single_line, ratio_chart

st.set_page_config(page_title="Macro", layout="wide")
st.title("Macro")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set. Add it as an environment variable locally and in Render.")
    st.stop()

@st.cache_data(ttl=12 * 60 * 60)
def load_macro() -> pd.DataFrame:
    return get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro")

@st.cache_data(ttl=6 * 60 * 60)
def load_proxies() -> pd.DataFrame:
    tickers = list(dict.fromkeys(list(YF_PROXIES.values())))
    return fetch_prices(tickers, period="5y")

macro = load_macro()
px = load_proxies()

last_macro_date = macro.dropna(how="all").index.max()
st.caption(f"Macro last updated: {last_macro_date.date() if pd.notna(last_macro_date) else 'unknown'}")

tabs = st.tabs(["Rates", "Credit and Liquidity", "Risk appetite"])

with tabs[0]:
    st.subheader("Rates")

    r1, r2 = st.columns([1, 1])
    with r1:
        yields_range = st.selectbox(
            "Treasury yields range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_yields_range",
        )
    with r2:
        curve_range = st.selectbox(
            "Curve spread range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_curve_range",
        )

    macro_yields_view = slice_df(macro, yields_range)
    macro_curve_view = slice_df(macro, curve_range)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = line_chart(
            macro_yields_view,
            title="Treasury yields",
            y_cols=["y3m", "y2", "y10"],
            y_axis_title="Percent",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "y10" in macro_curve_view.columns and "y2" in macro_curve_view.columns:
            curve = (macro_curve_view["y10"] - macro_curve_view["y2"]).dropna()
            st.plotly_chart(
                single_line(curve, title="Yield curve spread (10y minus 2y)", name="10y minus 2y", y_axis_title="Percent"),
                use_container_width=True,
            )
        else:
            st.info("Missing series for curve spread.")

    st.subheader("Real yields and dollar")

    r3, r4 = st.columns([1, 1])
    with r3:
        real_range = st.selectbox(
            "Real yield range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_real_range",
        )
    with r4:
        dollar_range = st.selectbox(
            "Dollar range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_dollar_range",
        )

    macro_real_view = slice_df(macro, real_range)
    macro_dollar_view = slice_df(macro, dollar_range)

    c3, c4 = st.columns([1, 1])
    with c3:
        if "real10" in macro_real_view.columns:
            st.plotly_chart(
                single_line(macro_real_view["real10"].dropna(), title="10y real yield (TIPS)", name="Real 10y", y_axis_title="Percent"),
                use_container_width=True,
            )
        else:
            st.info("Missing real yield series.")

    with c4:
        if "dollar_broad" in macro_dollar_view.columns:
            st.plotly_chart(
                single_line(macro_dollar_view["dollar_broad"].dropna(), title="Broad dollar index", name="Dollar broad", y_axis_title="Index"),
                use_container_width=True,
            )
        else:
            st.info("Missing dollar index series.")

with tabs[1]:
    st.subheader("Credit and Liquidity")

    r1, r2 = st.columns([1, 1])
    with r1:
        credit_range = st.selectbox(
            "Credit range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_credit_range",
        )
    with r2:
        fed_range = st.selectbox(
            "Fed assets range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("5y"),
            key="macro_fed_range",
        )

    macro_credit_view = slice_df(macro, credit_range)
    macro_fed_view = slice_df(macro, fed_range)

    c1, c2 = st.columns([1, 1])
    with c1:
        if "hy_oas" in macro_credit_view.columns:
            st.plotly_chart(
                single_line(macro_credit_view["hy_oas"].dropna(), title="High yield spread (OAS)", name="HY OAS", y_axis_title="Percent"),
                use_container_width=True,
            )
        else:
            st.info("Missing credit spread series.")

    with c2:
        if "fed_assets" in macro_fed_view.columns:
            st.plotly_chart(
                single_line(macro_fed_view["fed_assets"].dropna(), title="Fed total assets", name="Fed assets", y_axis_title="Millions of dollars"),
                use_container_width=True,
            )
        else:
            st.info("Missing Fed assets series.")

    st.subheader("Inflation impulse proxies")

    r3, r4 = st.columns([1, 1])
    with r3:
        oilgold_range = st.selectbox(
            "Oil over gold range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_oilg_range",
        )
    with r4:
        copgold_range = st.selectbox(
            "Copper over gold range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_copg_range",
        )

    px_oilg_view = slice_df(px, oilgold_range)
    px_copg_view = slice_df(px, copgold_range)

    oil_t = YF_PROXIES.get("oil", "USO")
    gold_t = YF_PROXIES.get("gold", "GLD")
    copper_t = YF_PROXIES.get("copper", "CPER")

    c3, c4 = st.columns([1, 1])
    with c3:
        if oil_t in px_oilg_view.columns and gold_t in px_oilg_view.columns:
            st.plotly_chart(
                ratio_chart(px_oilg_view[oil_t].dropna(), px_oilg_view[gold_t].dropna(), title="Oil over Gold ratio", name=f"{oil_t}/{gold_t}"),
                use_container_width=True,
            )
        else:
            st.info("Missing oil or gold proxy prices.")

    with c4:
        if copper_t in px_copg_view.columns and gold_t in px_copg_view.columns:
            st.plotly_chart(
                ratio_chart(px_copg_view[copper_t].dropna(), px_copg_view[gold_t].dropna(), title="Copper over Gold ratio", name=f"{copper_t}/{gold_t}"),
                use_container_width=True,
            )
        else:
            st.info("Missing copper or gold proxy prices.")

with tabs[2]:
    st.subheader("Risk appetite and breadth")

    r1, r2 = st.columns([1, 1])
    with r1:
        risk_range = st.selectbox(
            "IWM over SPY range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_risk_range",
        )
    with r2:
        breadth_range = st.selectbox(
            "Equal weight over SPY range",
            options=list(RANGES.keys()),
            index=list(RANGES.keys()).index("1y"),
            key="macro_breadth_range",
        )

    px_risk_view = slice_df(px, risk_range)
    px_breadth_view = slice_df(px, breadth_range)

    iwm_t = YF_PROXIES.get("iwm", "IWM")
    spy_t = YF_PROXIES.get("spy", "SPY")
    rsp_t = YF_PROXIES.get("rsp", "RSP")

    c1, c2 = st.columns([1, 1])
    with c1:
        if iwm_t in px_risk_view.columns and spy_t in px_risk_view.columns:
            st.plotly_chart(
                ratio_chart(px_risk_view[iwm_t].dropna(), px_risk_view[spy_t].dropna(), title="Small caps over SPY", name=f"{iwm_t}/{spy_t}"),
                use_container_width=True,
            )
        else:
            st.info("Missing IWM or SPY prices.")

    with c2:
        if rsp_t in px_breadth_view.columns and spy_t in px_breadth_view.columns:
            st.plotly_chart(
                ratio_chart(px_breadth_view[rsp_t].dropna(), px_breadth_view[spy_t].dropna(), title="Equal weight over SPY", name=f"{rsp_t}/{spy_t}"),
                use_container_width=True,
            )
        else:
            st.info("RSP not available, this is optional.")