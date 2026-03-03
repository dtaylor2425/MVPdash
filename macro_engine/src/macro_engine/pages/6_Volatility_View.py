# pages/6_Volatility_View.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.ranges import RANGES, slice_df
from src.charts import vix_term_chart, single_line
from src.ui import inject_css, sidebar_nav, safe_switch_page

st.set_page_config(
    page_title="Volatility View",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
sidebar_nav(active="Volatility View")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_data():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro")
    vix_tickers = [v for v in [
        YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m"), YF_PROXIES.get("vix6m"),
        YF_PROXIES.get("spy"), YF_PROXIES.get("hyg"),
    ] if v]
    px = fetch_prices(list(dict.fromkeys(vix_tickers)), period="5y")
    return macro, px if (px is not None and not px.empty) else pd.DataFrame()

macro, px = load_data()

def _s(name: str) -> pd.Series:
    t = YF_PROXIES.get(name)
    if t is None or px.empty or t not in px.columns:
        return pd.Series(dtype=float, name=name)
    return px[t].dropna().rename(name)

vix_s   = _s("vix")
vix3m_s = _s("vix3m")
vix6m_s = _s("vix6m")
spy_s   = _s("spy")
hyg_s   = _s("hyg")

RKEYS = list(RANGES.keys())

# ── Live regime calculations ───────────────────────────────────────────────────

def vratio_now(v, v3m):
    idx = v.index.intersection(v3m.index)
    if len(idx) == 0:
        return None
    return float(v.loc[idx[-1]]) / float(v3m.loc[idx[-1]])

def vix_pct_rank(v: pd.Series, lookback=252) -> float | None:
    """Where does today's VIX sit in the past `lookback` days? 0=low 100=high"""
    if len(v) < lookback:
        return None
    window = v.iloc[-lookback:]
    return float((window < v.iloc[-1]).mean() * 100)

def regime_from_vix(vix_val, vr):
    if vix_val is None:
        return "Unknown", "#6b7280", "#f3f4f6"
    if vix_val > 30 or (vr is not None and vr > 1.0):
        return "Stress / Panic", "#b42318", "#fee2e2"
    if vix_val > 20:
        return "Elevated concern", "#d97706", "#fef9c3"
    if vix_val > 15:
        return "Neutral", "#6b7280", "#f3f4f6"
    return "Complacency", "#1f7a4f", "#dcfce7"

vix_now   = float(vix_s.iloc[-1])   if not vix_s.empty   else None
vix3m_now = float(vix3m_s.iloc[-1]) if not vix3m_s.empty else None
vix6m_now = float(vix6m_s.iloc[-1]) if not vix6m_s.empty else None
vr_now    = vratio_now(vix_s, vix3m_s)
pct_rank  = vix_pct_rank(vix_s) if not vix_s.empty else None
vol_label, vol_color, vol_bg = regime_from_vix(vix_now, vr_now)

# ── Topbar ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""<div class="me-topbar">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <div>
          <div class="me-title">Volatility View</div>
          <div class="me-subtle">VIX term structure &nbsp;·&nbsp; V-Ratio &nbsp;·&nbsp;
          regime context &nbsp;·&nbsp; cross-asset stress</div>
        </div>
        <div style="padding:8px 16px;border-radius:20px;background:{vol_bg};">
          <span style="font-weight:800;color:{vol_color};font-size:14px;">{vol_label}</span>
        </div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

def kpi_card(col, label, value, sub, color="#0f172a", bg="#f8fafc"):
    col.markdown(
        f"<div style='padding:12px 14px;border-radius:12px;background:{bg};"
        f"border:1px solid rgba(0,0,0,0.07);'>"
        f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.4);"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-size:24px;font-weight:900;color:{color};line-height:1.1;'>{value}</div>"
        f"<div style='font-size:11px;color:rgba(0,0,0,0.5);margin-top:3px;'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

kpi_card(k1, "VIX Spot",
         f"{vix_now:.1f}" if vix_now else "—",
         "30-day implied vol",
         color=vol_color, bg=vol_bg)

kpi_card(k2, "VIX3M",
         f"{vix3m_now:.1f}" if vix3m_now else "—",
         "3-month implied vol",
         color="#1d4ed8", bg="#eff6ff")

kpi_card(k3, "VIX6M",
         f"{vix6m_now:.1f}" if vix6m_now else "—",
         "6-month implied vol",
         color="#7c3aed", bg="#f5f3ff")

if vr_now is not None:
    vr_color = "#b42318" if vr_now > 1.0 else ("#d97706" if vr_now > 0.9 else "#1f7a4f")
    vr_bg    = "#fee2e2" if vr_now > 1.0 else ("#fef9c3" if vr_now > 0.9 else "#dcfce7")
    vr_label = "PANIC" if vr_now > 1.0 else ("Elevated" if vr_now > 0.9 else "Calm")
    kpi_card(k4, "V-Ratio (VIX÷VIX3M)",
             f"{vr_now:.3f}",
             vr_label,
             color=vr_color, bg=vr_bg)
else:
    kpi_card(k4, "V-Ratio", "—", "Unavailable")

if pct_rank is not None:
    pr_color = "#b42318" if pct_rank > 75 else ("#d97706" if pct_rank > 50 else "#1f7a4f")
    pr_bg    = "#fee2e2" if pct_rank > 75 else ("#fef9c3" if pct_rank > 50 else "#dcfce7")
    kpi_card(k5, "VIX 1Y Percentile",
             f"{pct_rank:.0f}th",
             "vs past 252 trading days",
             color=pr_color, bg=pr_bg)
else:
    kpi_card(k5, "VIX Percentile", "—", "Need 252d history")

st.markdown("")

# ── Regime interpretation banner ──────────────────────────────────────────────

if vr_now is not None and vix_now is not None:
    if vr_now > 1.0:
        interp = (
            f"**Panic signal active.** Spot VIX ({vix_now:.1f}) exceeds 3-month VIX ({vix3m_now:.1f}) — "
            "the market is pricing more fear today than it expects over the next quarter. "
            "Historically this coincides with capitulation events and short-term bottoms. "
            "Watch for V-Ratio reverting below 1.0 as a potential all-clear."
        )
        banner_color = "#fee2e2"
        banner_border = "#fca5a5"
    elif vr_now > 0.9:
        interp = (
            f"**Volatility elevated but not panic.** V-Ratio at {vr_now:.3f} — approaching the panic zone. "
            f"VIX spot ({vix_now:.1f}) is running close to 3-month expectations. "
            "Risk-off pressure is building; watch credit spreads and equity breadth for confirmation."
        )
        banner_color = "#fef9c3"
        banner_border = "#fde68a"
    elif vix_now < 15:
        interp = (
            f"**Complacency zone.** VIX at {vix_now:.1f} with a calm V-Ratio ({vr_now:.3f}) suggests "
            "markets are pricing minimal near-term risk. Low VIX environments can persist but are "
            "also when tail risks build undetected. Monitor for sudden V-Ratio spikes."
        )
        banner_color = "#dcfce7"
        banner_border = "#86efac"
    else:
        interp = (
            f"VIX at {vix_now:.1f} with V-Ratio {vr_now:.3f} — volatility is in a normal range. "
            "The term structure is in contango (VIX3M > VIX), which is the baseline healthy state."
        )
        banner_color = "#f3f4f6"
        banner_border = "#d1d5db"

    st.markdown(
        f"<div style='padding:12px 16px;border-radius:12px;background:{banner_color};"
        f"border:1px solid {banner_border};font-size:13px;line-height:1.6;margin-bottom:16px;'>"
        f"{interp}</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — MAIN CHART + TERM STRUCTURE TABLE
# ══════════════════════════════════════════════════════════════════════════════

chart_col, table_col = st.columns([2.8, 1.0], gap="medium")

with chart_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>VIX term structure & V-Ratio</div>",
                    unsafe_allow_html=True)
        if vix_s.empty or vix3m_s.empty:
            st.warning("VIX data unavailable. Check ^VIX and ^VIX3M in YF_PROXIES.")
        else:
            vix_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="vol_range")
            vs  = slice_df(vix_s.to_frame("v"),   vix_range)["v"]
            v3s = slice_df(vix3m_s.to_frame("v"), vix_range)["v"]
            v6s = slice_df(vix6m_s.to_frame("v"), vix_range)["v"] if not vix6m_s.empty else pd.Series(dtype=float)
            st.plotly_chart(
                vix_term_chart(vs, v3s, v6s if not v6s.empty else None),
                use_container_width=True,
            )

with table_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Term structure snapshot</div>",
                    unsafe_allow_html=True)

        rows = []
        if vix_now:
            rows.append({"Tenor": "VIX (30d)", "Level": f"{vix_now:.2f}"})
        if vix3m_now:
            rows.append({"Tenor": "VIX3M (90d)", "Level": f"{vix3m_now:.2f}"})
        if vix6m_now:
            rows.append({"Tenor": "VIX6M (180d)", "Level": f"{vix6m_now:.2f}"})

        if len(rows) >= 2 and vix_now and vix3m_now:
            slope_30_90 = vix3m_now - vix_now
            rows.append({"Tenor": "Slope (3m−spot)", "Level": f"{slope_30_90:+.2f}"})
            structure = "Contango ↑" if slope_30_90 > 0 else "Backwardation ↓"
            rows.append({"Tenor": "Structure", "Level": structure})

        if rows:
            for r in rows:
                st.markdown(
                    f"<div class='me-li'>"
                    f"<div class='me-li-name'>{r['Tenor']}</div>"
                    f"<div class='me-li-right' style='font-weight:700;'>{r['Level']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:11px;color:rgba(0,0,0,0.5);line-height:1.6;'>"
            "<b>Contango</b> (3m > spot) = normal. "
            "Market expects vol to mean-revert higher over time.<br><br>"
            "<b>Backwardation</b> (spot > 3m) = stress. "
            "Market pricing more fear today than it expects going forward. "
            "V-Ratio > 1 = backwardation active.</div>",
            unsafe_allow_html=True,
        )

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — VIX PERCENTILE HISTORY + ROLLING REALIZED VOL
# ══════════════════════════════════════════════════════════════════════════════

hist_col, rvol_col = st.columns(2, gap="medium")

with hist_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>VIX percentile over time</div>",
                    unsafe_allow_html=True)
        if not vix_s.empty and len(vix_s) > 252:
            pct_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="pct_range")
            vix_sliced = slice_df(vix_s.to_frame("v"), pct_range)["v"]

            # Rolling 252-day percentile rank
            roll_pct = vix_sliced.rolling(252, min_periods=63).apply(
                lambda x: float((x[:-1] < x[-1]).mean() * 100) if len(x) > 1 else np.nan
            )
            roll_pct = roll_pct.dropna()

            if not roll_pct.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=roll_pct.index, y=roll_pct.values,
                    mode="lines", name="VIX %ile",
                    line=dict(color="#1d4ed8", width=1.8),
                    fill="tozeroy", fillcolor="rgba(29,78,216,0.06)",
                ))
                # Bands
                for level, lcolor, label in [(75, "#dc2626", "Elevated"), (50, "#94a3b8", "Median"), (25, "#1f7a4f", "Low")]:
                    fig.add_hline(y=level, line_dash="dash", line_color=lcolor,
                                  line_width=1, annotation_text=label,
                                  annotation_position="right")
                fig.update_layout(
                    height=300, margin=dict(l=10, r=60, t=20, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis_title="Percentile", yaxis_range=[0, 100],
                    showlegend=False, hovermode="x unified",
                )
                fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Not enough history for percentile calculation.")
        else:
            st.caption("Need 252+ days of VIX data.")

with rvol_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Realized vol vs VIX (implied)</div>",
                    unsafe_allow_html=True)
        st.caption("When realized vol > VIX, options were cheap. When VIX > realized, options were expensive.")

        if not spy_s.empty and not vix_s.empty:
            rvol_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="rvol_range")

            spy_sl  = slice_df(spy_s.to_frame("v"), rvol_range)["v"]
            vix_sl  = slice_df(vix_s.to_frame("v"), rvol_range)["v"]

            # 21-day rolling realized vol annualised
            rvol_21 = (spy_sl.pct_change().rolling(21).std() * np.sqrt(252) * 100).dropna()

            if not rvol_21.empty:
                # Align
                common = rvol_21.index.intersection(vix_sl.index)
                rvol_a = rvol_21.loc[common]
                vix_a  = vix_sl.loc[common]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=vix_a.index, y=vix_a.values,
                    mode="lines", name="VIX (implied)",
                    line=dict(color="#dc2626", width=1.8),
                ))
                fig2.add_trace(go.Scatter(
                    x=rvol_a.index, y=rvol_a.values,
                    mode="lines", name="Realized vol 21d",
                    line=dict(color="#1d4ed8", width=1.8, dash="dot"),
                ))
                fig2.update_layout(
                    height=300, margin=dict(l=10, r=20, t=20, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis_title="Annualized vol (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    hovermode="x unified",
                )
                fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.caption("Not enough SPY data for realized vol.")
        else:
            st.caption("SPY or VIX data unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — VRATIO HISTORY + VIX vs CREDIT SPREADS
# ══════════════════════════════════════════════════════════════════════════════

vr_col, credit_col = st.columns(2, gap="medium")

with vr_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>V-Ratio history</div>", unsafe_allow_html=True)
        st.caption("Above 1.0 = backwardation = panic zone. The longer it stays above 1, the more severe the stress.")

        if not vix_s.empty and not vix3m_s.empty:
            vr_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="vr_hist_range")
            vs2  = slice_df(vix_s.to_frame("v"),   vr_range)["v"]
            v3s2 = slice_df(vix3m_s.to_frame("v"), vr_range)["v"]
            idx2 = vs2.index.intersection(v3s2.index)

            if len(idx2) > 5:
                vr_hist = (vs2.loc[idx2] / v3s2.loc[idx2]).dropna()
                panic_days = int((vr_hist > 1.0).sum())
                total_days = len(vr_hist)

                fig3 = go.Figure()
                # Baseline for fill
                vr_lo = float(vr_hist.min()) * 0.97
                fig3.add_trace(go.Scatter(
                    x=vr_hist.index, y=[vr_lo] * len(vr_hist),
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig3.add_trace(go.Scatter(
                    x=vr_hist.index, y=vr_hist.values,
                    mode="lines", name="V-Ratio",
                    line=dict(color="#ea580c", width=2),
                    fill="tonexty", fillcolor="rgba(234,88,12,0.08)",
                ))
                fig3.add_hline(y=1.0, line_color="#dc2626", line_width=1.5,
                               annotation_text="Panic threshold",
                               annotation_position="right")
                fig3.add_hline(y=0.9, line_color="#94a3b8", line_dash="dash",
                               line_width=1, annotation_text="Calm",
                               annotation_position="right")
                fig3.update_layout(
                    height=280, margin=dict(l=10, r=80, t=20, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False, hovermode="x unified",
                    yaxis_title="V-Ratio",
                )
                fig3.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(
                    f"Panic days in window: **{panic_days}** of {total_days} "
                    f"({100*panic_days/total_days:.1f}%)"
                )
        else:
            st.caption("VIX / VIX3M unavailable.")

with credit_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>VIX vs Credit spreads (HY OAS)</div>",
                    unsafe_allow_html=True)
        st.caption("VIX and HY spreads confirm each other during stress. Divergence = one market isn't believing the other.")

        if not vix_s.empty and "hy_oas" in macro.columns:
            cs_range = st.selectbox("Range", RKEYS, index=RKEYS.index("1y"), key="cs_range")
            vix_cs  = slice_df(vix_s.to_frame("v"),  cs_range)["v"]
            hy_cs   = slice_df(macro, cs_range)["hy_oas"].dropna()
            common  = vix_cs.index.intersection(hy_cs.index)

            if len(common) > 10:
                from plotly.subplots import make_subplots as _msp
                fig4 = _msp(specs=[[{"secondary_y": True}]])
                fig4.add_trace(go.Scatter(
                    x=vix_cs.loc[common].index, y=vix_cs.loc[common].values,
                    mode="lines", name="VIX",
                    line=dict(color="#dc2626", width=1.8),
                ), secondary_y=False)
                fig4.add_trace(go.Scatter(
                    x=hy_cs.loc[common].index, y=hy_cs.loc[common].values,
                    mode="lines", name="HY OAS",
                    line=dict(color="#7c3aed", width=1.8, dash="dot"),
                ), secondary_y=True)
                fig4.update_layout(
                    height=280, margin=dict(l=10, r=20, t=20, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    hovermode="x unified",
                )
                fig4.update_yaxes(title_text="VIX", secondary_y=False,
                                  showgrid=True, gridcolor="#f1f5f9")
                fig4.update_yaxes(title_text="HY OAS (%)", secondary_y=True,
                                  showgrid=False)
                fig4.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.caption("VIX or HY OAS unavailable.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Volatility regime playbook</div>",
                unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4, gap="medium")

    def playbook_card(col, regime, trigger, implication, action, color, bg):
        is_active = vol_label.lower() in regime.lower()
        border = f"2px solid {color}" if is_active else "1px solid rgba(0,0,0,0.08)"
        active_badge = (
            f"<span style='font-size:10px;font-weight:700;color:{color};"
            f"background:{bg};padding:2px 7px;border-radius:8px;margin-left:6px;'>CURRENT</span>"
            if is_active else ""
        )
        col.markdown(
            f"<div style='padding:12px 14px;border-radius:12px;background:{bg};"
            f"border:{border};height:100%;'>"
            f"<div style='font-weight:800;font-size:13px;color:{color};margin-bottom:8px;'>"
            f"{regime}{active_badge}</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.5);font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:3px;'>Trigger</div>"
            f"<div style='font-size:12px;color:rgba(0,0,0,0.7);margin-bottom:8px;'>{trigger}</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.5);font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:3px;'>Implication</div>"
            f"<div style='font-size:12px;color:rgba(0,0,0,0.7);margin-bottom:8px;'>{implication}</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.5);font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:3px;'>Watch for</div>"
            f"<div style='font-size:12px;color:rgba(0,0,0,0.7);'>{action}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    playbook_card(p1,
        "Complacency", "VIX < 15, V-Ratio < 0.9",
        "Markets pricing minimal risk. Often precedes vol expansion.",
        "Sudden V-Ratio spike or credit spread widening",
        "#1f7a4f", "#dcfce7")

    playbook_card(p2,
        "Neutral", "VIX 15–20, V-Ratio 0.9–1.0",
        "Normal healthy market environment. Term structure in contango.",
        "V-Ratio drifting toward 1.0 or VIX breaking above 20",
        "#6b7280", "#f3f4f6")

    playbook_card(p3,
        "Elevated concern", "VIX 20–30, V-Ratio near 1.0",
        "Risk-off building. Options pricing elevated near-term fear.",
        "V-Ratio crossing 1.0 or credit spreads diverging from VIX",
        "#d97706", "#fef9c3")

    playbook_card(p4,
        "Stress / Panic", "VIX > 30 or V-Ratio > 1.0",
        "Backwardation active. Capitulation-like conditions.",
        "V-Ratio reverting below 1.0 = potential all-clear signal",
        "#b42318", "#fee2e2")

st.markdown("")
st.caption(
    "V-Ratio = VIX ÷ VIX3M. Normally < 1 (contango). "
    "Crosses above 1 during stress events when near-term fear spikes. "
    "Data: ^VIX, ^VIX3M, ^VIX6M via Yahoo Finance."
)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)