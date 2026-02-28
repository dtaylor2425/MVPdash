# pages/2_Macro_Charts.py
import streamlit as st
import pandas as pd
import numpy as np

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.ranges import RANGES, slice_df
from src.charts import line_chart, single_line, ratio_chart, dual_axis_chart, vix_term_chart
from src.ui import inject_css, sidebar_nav

st.set_page_config(page_title="Macro charts", layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Macro charts")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_macro() -> pd.DataFrame:
    return get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro")

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_proxies() -> pd.DataFrame:
    tickers = list(dict.fromkeys(list(YF_PROXIES.values())))
    return fetch_prices(tickers, period="5y")

macro = load_macro()
px    = load_proxies()

last_macro_date = macro.dropna(how="all").index.max()

# ── Helpers ───────────────────────────────────────────────────────────────────

def cpi_yoy(df: pd.DataFrame) -> pd.Series | None:
    if "cpi" not in df.columns:
        return None
    c = df["cpi"].dropna()
    return (c.pct_change(12) * 100).dropna() if len(c) >= 13 else None

def _vix_series(name: str) -> pd.Series:
    ticker = YF_PROXIES.get(name)
    if ticker is None or ticker not in px.columns:
        return pd.Series(dtype=float)
    return px[ticker].dropna()

# ── Topbar ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""<div class="me-topbar">
      <div class="me-title">Macro charts</div>
      <div class="me-subtle">Last update: {last_macro_date.date() if pd.notna(last_macro_date) else 'unknown'}</div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Deep-link: ?tab=curve|rates|volatility|risk|credit ───────────────────────

TAB_NAMES = ["Curve context", "Rates", "Volatility", "Risk appetite", "Credit and Liquidity"]
TAB_SLUGS = ["curve", "rates", "volatility", "risk", "credit"]

_slug = st.query_params.get("tab", "")
try:
    _start = TAB_SLUGS.index(_slug)
except ValueError:
    _start = 0

# st.tabs does not support a programmatic default index natively, but we can
# use session_state to pre-select the tab by rendering a hidden radio that
# controls which tab content is shown, or simply rely on the user clicking.
# The cleanest working approach in Streamlit is to store the desired tab in
# session_state and use it as the default for a selectbox-driven tab switch.
# However st.tabs() itself renders all tab headers and the selected one is
# purely client-side after the first render.  The accepted workaround is to
# use st.session_state to remember the last tab and note the query param for
# external linking guidance.  We expose the slug in the URL so bookmarks work.

tabs = st.tabs(TAB_NAMES)

RKEYS = list(RANGES.keys())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — CURVE CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("<div class='me-rowtitle'>Curve regime context</div>", unsafe_allow_html=True)
    st.caption(
        "The yield curve doesn't move in isolation. Map it against inflation, the fed funds rate, "
        "and credit spreads to understand *why* the curve is where it is and what regime it signals."
    )

    ctx_range = st.selectbox("Range", RKEYS, index=RKEYS.index("5y"), key="curve_ctx_range")
    macro_ctx = slice_df(macro, ctx_range)

    # ── Chart 1: Curve vs Fed Funds vs CPI ───────────────────────────────────
    st.markdown("#### Curve vs Fed Funds rate vs CPI inflation")
    st.caption(
        "**How to read:** When the curve (blue) inverts, the Fed is typically tightening "
        "(fed funds rising, red dotted) or inflation has been running hot (green dotted). "
        "A steepening curve *while* the Fed holds or cuts = early recovery signal."
    )

    if "y10" in macro_ctx.columns and "y2" in macro_ctx.columns:
        curve_s = (macro_ctx["y10"] - macro_ctx["y2"]).dropna()
        right_ctx = []
        if "fed_funds" in macro_ctx.columns:
            right_ctx.append((macro_ctx["fed_funds"].dropna(), "Fed funds rate", "#dc2626"))
        cpi_s = cpi_yoy(macro_ctx)
        if cpi_s is not None and not cpi_s.empty:
            right_ctx.append((cpi_s, "CPI YoY %", "#059669"))
        if right_ctx:
            st.plotly_chart(dual_axis_chart(
                left_series=[(curve_s, "10y − 2y curve", "#1d4ed8")],
                right_series=right_ctx,
                title="Yield curve vs Fed Funds & CPI",
                left_title="Curve spread (ppt)", right_title="Rate / Inflation (%)",
                zero_line=True, height=400,
            ), use_container_width=True)
        else:
            st.plotly_chart(single_line(curve_s, "10y − 2y curve", "Curve", "Percent"),
                            use_container_width=True)
            st.info("Fed funds or CPI not in FRED_SERIES.")
    else:
        st.info("Missing y10 or y2.")

    st.divider()

    # ── Chart 2: Curve vs HY OAS ─────────────────────────────────────────────
    st.markdown("#### Curve vs Credit spreads (HY OAS)")
    st.caption(
        "**How to read:** Curve inversion with *tight* spreads = late-cycle complacency. "
        "Inversion with *wide* spreads = active stress / early recession."
    )

    if "y10" in macro_ctx.columns and "y2" in macro_ctx.columns:
        curve_s2 = (macro_ctx["y10"] - macro_ctx["y2"]).dropna()
        if "hy_oas" in macro_ctx.columns:
            st.plotly_chart(dual_axis_chart(
                left_series=[(curve_s2, "10y − 2y curve", "#1d4ed8")],
                right_series=[(macro_ctx["hy_oas"].dropna(), "HY OAS", "#dc2626")],
                title="Yield curve vs HY credit spread",
                left_title="Curve spread (ppt)", right_title="HY OAS (%)",
                zero_line=True, height=380,
            ), use_container_width=True)
        else:
            st.info("HY OAS not available.")
    else:
        st.info("Missing y10 or y2.")

    st.divider()

    # ── Curve regime classification ───────────────────────────────────────────
    st.markdown("#### Current curve regime")

    if "y10" in macro.columns and "y2" in macro.columns:
        curve_full = (macro["y10"] - macro["y2"]).dropna()
        c_now = float(curve_full.iloc[-1]) if not curve_full.empty else None
        ff_now = float(macro["fed_funds"].dropna().iloc[-1]) if "fed_funds" in macro.columns and not macro["fed_funds"].dropna().empty else None
        cpi_full = cpi_yoy(macro)

        if c_now is not None:
            if c_now >= 0.75:
                crv_regime, crv_bg = "🟢 Steep — early expansion or post-recession", "#dcfce7"
            elif c_now >= 0:
                crv_regime, crv_bg = "🟡 Flat to mildly positive — mid-cycle", "#fef9c3"
            elif c_now >= -0.25:
                crv_regime, crv_bg = "🟠 Shallow inversion — late-cycle caution", "#ffedd5"
            else:
                crv_regime, crv_bg = "🔴 Deep inversion — recession risk elevated", "#fee2e2"

            crv_trend = ("Steepening ↑" if c_now > float(curve_full.iloc[-64]) else "Flattening ↓") \
                if len(curve_full) > 64 else "—"

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Curve level", f"{c_now:.2f} ppt")
            col_b.metric("Trend (vs 3m ago)", crv_trend)
            if ff_now is not None:
                col_c.metric("Fed funds", f"{ff_now:.2f}%")
            if cpi_full is not None and not cpi_full.empty:
                col_d.metric("CPI YoY", f"{float(cpi_full.iloc[-1]):.1f}%")

            st.markdown(
                f"<div style='margin-top:10px;padding:12px 16px;border-radius:12px;"
                f"background:{crv_bg};font-size:14px;font-weight:700;'>"
                f"Current regime: {crv_regime}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Need y10 and y2 in FRED_SERIES.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RATES
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("<div class='me-rowtitle'>Treasury yields</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        yields_range = st.selectbox("Treasury yields range", RKEYS, index=RKEYS.index("1y"),
                                    key="macro_yields_range")
    with r2:
        curve_range = st.selectbox("Curve spread range", RKEYS, index=RKEYS.index("1y"),
                                   key="macro_curve_range")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            line_chart(slice_df(macro, yields_range), "Treasury yields",
                       ["y3m", "y2", "y10"], "Percent"),
            use_container_width=True,
        )
    with c2:
        mv = slice_df(macro, curve_range)
        if "y10" in mv.columns and "y2" in mv.columns:
            st.plotly_chart(
                single_line((mv["y10"] - mv["y2"]).dropna(),
                            "Yield curve spread (10y minus 2y)", "10y − 2y", "Percent"),
                use_container_width=True,
            )
        else:
            st.info("Missing curve series.")

    st.markdown("<div class='me-rowtitle' style='margin-top:16px;'>Real yields and dollar</div>",
                unsafe_allow_html=True)

    r3, r4 = st.columns(2)
    with r3:
        real_range = st.selectbox("Real yield range", RKEYS, index=RKEYS.index("1y"),
                                  key="macro_real_range")
    with r4:
        dollar_range = st.selectbox("Dollar range", RKEYS, index=RKEYS.index("1y"),
                                    key="macro_dollar_range")

    c3, c4 = st.columns(2)
    with c3:
        if "real10" in macro.columns:
            st.plotly_chart(
                single_line(slice_df(macro, real_range)["real10"].dropna(),
                            "10y real yield (TIPS)", "Real 10y", "Percent"),
                use_container_width=True,
            )
        else:
            st.info("Missing real yield series.")
    with c4:
        if "dollar_broad" in macro.columns:
            st.plotly_chart(
                single_line(slice_df(macro, dollar_range)["dollar_broad"].dropna(),
                            "Broad dollar index", "Dollar broad", "Index"),
                use_container_width=True,
            )
        else:
            st.info("Missing dollar index series.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("<div class='me-rowtitle'>Volatility term structure</div>", unsafe_allow_html=True)
    st.caption(
        "**V-Ratio = VIX ÷ VIX3M.** Normally VIX < VIX3M (mean-reversion expectation → V-Ratio < 1). "
        "V-Ratio > 1 = spot fear exceeds 3-month expectation = **panic mode**. "
        "V-Ratio < 0.9 with low VIX = complacency."
    )

    vix_s   = _vix_series("vix")
    vix3m_s = _vix_series("vix3m")
    vix6m_s = _vix_series("vix6m")

    if vix_s.empty or vix3m_s.empty:
        st.warning("VIX or VIX3M data unavailable. Check ^VIX and ^VIX3M are in YF_PROXIES.")
    else:
        vix_range = st.selectbox("VIX chart range", RKEYS, index=RKEYS.index("1y"), key="vix_range")

        vix_s   = slice_df(vix_s.to_frame("v"),   vix_range)["v"]
        vix3m_s = slice_df(vix3m_s.to_frame("v"), vix_range)["v"]
        if not vix6m_s.empty:
            vix6m_s = slice_df(vix6m_s.to_frame("v"), vix_range)["v"]

        st.plotly_chart(
            vix_term_chart(vix_s, vix3m_s, vix6m_s if not vix6m_s.empty else None),
            use_container_width=True,
        )

        # Live V-Ratio callout
        idx = vix_s.index.intersection(vix3m_s.index)
        if len(idx) > 0:
            v_last   = float(vix_s.loc[idx[-1]])
            v3m_last = float(vix3m_s.loc[idx[-1]])
            vratio   = v_last / v3m_last if v3m_last != 0 else None

            col1, col2, col3 = st.columns(3)
            col1.metric("VIX (spot)", f"{v_last:.1f}")
            col2.metric("VIX3M",      f"{v3m_last:.1f}")

            if vratio is not None:
                if vratio > 1.0:
                    signal, bg = "🔴 Panic — spot fear > 3m expectation", "#fee2e2"
                elif vratio > 0.9:
                    signal, bg = "🟡 Elevated — approaching panic zone", "#fef9c3"
                else:
                    signal, bg = "🟢 Calm — market expects mean reversion", "#dcfce7"

                col3.metric("V-Ratio", f"{vratio:.3f}")
                st.markdown(
                    f"<div style='margin-top:10px;padding:12px 16px;border-radius:12px;"
                    f"background:{bg};font-size:14px;font-weight:700;'>{signal}</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("#### V-Ratio in context")
        idx_full = vix_s.index.intersection(vix3m_s.index)
        if len(idx_full) > 5:
            vr_full  = (vix_s.loc[idx_full] / vix3m_s.loc[idx_full]).dropna()
            panic_pct = float((vr_full > 1.0).mean() * 100)
            st.caption(
                f"In this window, V-Ratio was above 1 on **{panic_pct:.1f}%** of trading days."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK APPETITE
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("<div class='me-rowtitle'>Risk appetite and breadth</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        risk_range = st.selectbox("IWM over SPY range", RKEYS, index=RKEYS.index("1y"),
                                  key="macro_risk_range")
    with r2:
        breadth_range = st.selectbox("Equal weight over SPY range", RKEYS, index=RKEYS.index("1y"),
                                     key="macro_breadth_range")

    iwm_t = YF_PROXIES.get("iwm", "IWM")
    spy_t = YF_PROXIES.get("spy", "SPY")
    rsp_t = YF_PROXIES.get("rsp", "RSP")

    c1, c2 = st.columns(2)
    with c1:
        px_rv = slice_df(px, risk_range)
        if iwm_t in px_rv.columns and spy_t in px_rv.columns:
            st.plotly_chart(
                ratio_chart(px_rv[iwm_t].dropna(), px_rv[spy_t].dropna(),
                            "Small caps over SPY", f"{iwm_t}/{spy_t}"),
                use_container_width=True,
            )
        else:
            st.info("Missing IWM or SPY.")
    with c2:
        px_bv = slice_df(px, breadth_range)
        if rsp_t in px_bv.columns and spy_t in px_bv.columns:
            st.plotly_chart(
                ratio_chart(px_bv[rsp_t].dropna(), px_bv[spy_t].dropna(),
                            "Equal weight over SPY", f"{rsp_t}/{spy_t}"),
                use_container_width=True,
            )
        else:
            st.info("RSP not available — optional breadth proxy.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CREDIT AND LIQUIDITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("<div class='me-rowtitle'>Credit and Liquidity</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        credit_range = st.selectbox("Credit range", RKEYS, index=RKEYS.index("1y"),
                                    key="macro_credit_range")
    with r2:
        fed_range = st.selectbox("Fed assets range", RKEYS, index=RKEYS.index("5y"),
                                 key="macro_fed_range")

    c1, c2 = st.columns(2)
    with c1:
        if "hy_oas" in macro.columns:
            st.plotly_chart(
                single_line(slice_df(macro, credit_range)["hy_oas"].dropna(),
                            "High yield spread (OAS)", "HY OAS", "Percent"),
                use_container_width=True,
            )
        else:
            st.info("Missing credit spread series.")
    with c2:
        if "fed_assets" in macro.columns:
            st.plotly_chart(
                single_line(slice_df(macro, fed_range)["fed_assets"].dropna(),
                            "Fed total assets", "Fed assets", "Millions USD"),
                use_container_width=True,
            )
        else:
            st.info("Missing Fed assets series.")

    st.markdown("<div class='me-rowtitle' style='margin-top:16px;'>Inflation impulse proxies</div>",
                unsafe_allow_html=True)

    r3, r4 = st.columns(2)
    with r3:
        oilgold_range = st.selectbox("Oil over gold range", RKEYS, index=RKEYS.index("1y"),
                                     key="macro_oilg_range")
    with r4:
        copgold_range = st.selectbox("Copper over gold range", RKEYS, index=RKEYS.index("1y"),
                                     key="macro_copg_range")

    oil_t    = YF_PROXIES.get("oil",    "USO")
    gold_t   = YF_PROXIES.get("gold",   "GLD")
    copper_t = YF_PROXIES.get("copper", "CPER")

    c3, c4 = st.columns(2)
    with c3:
        px_og = slice_df(px, oilgold_range)
        if oil_t in px_og.columns and gold_t in px_og.columns:
            st.plotly_chart(
                ratio_chart(px_og[oil_t].dropna(), px_og[gold_t].dropna(),
                            "Oil over Gold ratio", f"{oil_t}/{gold_t}"),
                use_container_width=True,
            )
        else:
            st.info("Missing oil or gold proxy.")
    with c4:
        px_cg = slice_df(px, copgold_range)
        if copper_t in px_cg.columns and gold_t in px_cg.columns:
            st.plotly_chart(
                ratio_chart(px_cg[copper_t].dropna(), px_cg[gold_t].dropna(),
                            "Copper over Gold ratio", f"{copper_t}/{gold_t}"),
                use_container_width=True,
            )
        else:
            st.info("Missing copper or gold proxy.")