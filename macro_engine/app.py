# app.py  — Home page
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from datetime import date

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3
from src.compute import component_contribution
from src.ui import (
    inject_css, sidebar_nav, safe_switch_page,
    regime_color, regime_bg, delta_badge_html, SCORE_LEGEND_HTML, html_table,
)

st.set_page_config(page_title="Macro Engine", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Ticker lists ──────────────────────────────────────────────────────────────

ROTATION_TICKERS = ["XLE","XLF","XLK","XLI","XLP","XLV","GLD","UUP","IWM","QQQ","IGV","SMH","SPY"]
MARKET_SNAPSHOT  = ["SPY","QQQ","IWM","HYG","TLT","UUP","GLD","XLE"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nearest_before(series, dt):
    s   = series.dropna()
    idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None

def delta_over_days(series, days):
    s = series.dropna()
    if s.empty: return None, None, None
    end_i = _nearest_before(s, s.index.max())
    pre_i = _nearest_before(s, s.index.max() - pd.Timedelta(days=days))
    if end_i is None or pre_i is None: return None, None, None
    lv = float(s.loc[end_i]); pv = float(s.loc[pre_i])
    return lv, pv, float(lv - pv)

def pct_return_over_days(series, days):
    s = series.dropna()
    if len(s) < 2: return None
    end_i = _nearest_before(s, s.index.max())
    pre_i = _nearest_before(s, s.index.max() - pd.Timedelta(days=days))
    if end_i is None or pre_i is None: return None
    a, b = float(s.loc[end_i]), float(s.loc[pre_i])
    return None if b == 0 else (a / b) - 1.0

def _cc(c): return component_contribution(c)

def _trend_phrase(name, trend_up):
    nm = (name or "").lower()
    if trend_up is None: return "steady"
    if "credit" in nm or "spread" in nm: return "widening" if int(trend_up)==1 else "tightening"
    if "dollar" in nm or "usd" in nm:    return "strengthening" if int(trend_up)==1 else "weakening"
    return "rising" if int(trend_up)==1 else "falling"

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30*60, show_spinner=False)
def load_current_regime():
    if not FRED_API_KEY: raise RuntimeError("FRED_API_KEY not set")
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    vix_t = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
    all_t = list(dict.fromkeys(ROTATION_TICKERS + MARKET_SNAPSHOT + vix_t))
    px    = fetch_prices(all_t, period="5y")
    px    = pd.DataFrame() if (px is None or px.empty) else px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px,
                               lookback_trend=63, momentum_lookback_days=63)
    return regime, macro, px

@st.cache_data(show_spinner=False)
def compute_rs_heatmap(period="1y"):
    tickers = ["GLD","XLE","XLP","XLI","IWM","RSP","HYG","TLT","UUP","QQQ","XLK","XLF","XLV","IGV","SMH"]
    base = "SPY"
    px = fetch_prices(list(dict.fromkeys([base]+tickers)), period=period)
    if px is None or px.empty or base not in px.columns:
        return pd.DataFrame(index=tickers, columns=["1w","1m","3m"], data=np.nan)
    px = px.dropna(how="all"); b = px[base].dropna()
    horiz = {"1w":5,"1m":21,"3m":63}
    out   = pd.DataFrame(index=tickers, columns=list(horiz.keys()), dtype=float)
    for t in tickers:
        if t not in px.columns: continue
        s = px[t].dropna(); idx = s.index.intersection(b.index)
        if len(idx) < 90: continue
        s, bb = s.loc[idx], b.loc[idx]
        for lab, n in horiz.items():
            if len(idx) > n:
                out.loc[t, lab] = float((s.iloc[-1]/s.iloc[-1-n]) - (bb.iloc[-1]/bb.iloc[-1-n]))
    return out

# ── Snapshot builders ─────────────────────────────────────────────────────────

def build_allocation_tilt(r):
    alloc  = getattr(r, "allocation", None)
    stance = {}
    if isinstance(alloc, dict):
        s = alloc.get("stance", {}); stance = s if isinstance(s, dict) else {}
    return [(k, str(stance[k])) for k in ["Equities","Credit","Duration","USD","Commodities"]
            if k in stance and stance[k] is not None]

def build_weekly_macro_changes(macro):
    items = []
    if not isinstance(macro, pd.DataFrame) or macro.empty: return items
    for name, series, days, inv, pos_lbl, neg_lbl in [
        ("Credit spreads", macro.get("hy_oas",    pd.Series(dtype=float)), 7,  True,  "tighter","wider"),
        ("Curve 10y−2y",   (macro["y10"]-macro["y2"]).dropna()
                           if "y10" in macro.columns and "y2" in macro.columns
                           else pd.Series(dtype=float),                     7,  False, "steeper","flatter"),
        ("Real 10y yield", macro.get("real10",    pd.Series(dtype=float)), 7,  True,  "lower",  "higher"),
        ("Dollar broad",   macro.get("dollar_broad",pd.Series(dtype=float)),10, False, "weaker", "stronger"),
    ]:
        if hasattr(series, 'dropna') and not series.dropna().empty:
            l, p, d = delta_over_days(series, days)
            if d is not None:
                items.append((name, p, l, d, pos_lbl if d<0 else neg_lbl, inv))
    return items

def get_curve_snapshot(macro):
    if "y10" not in macro.columns or "y2" not in macro.columns: return None
    curve = (macro["y10"] - macro["y2"]).dropna()
    if curve.empty: return None
    c_now = float(curve.iloc[-1])
    c_63  = float(curve.iloc[-64]) if len(curve) > 64 else None
    trend = ("Steepening ↑" if c_now > c_63 else "Flattening ↓") if c_63 else "—"
    ff    = float(macro["fed_funds"].dropna().iloc[-1]) \
            if "fed_funds" in macro.columns and not macro["fed_funds"].dropna().empty else None
    if c_now >= 0.75:    label,color,bg = "Steep",             "#1f7a4f","#dcfce7"
    elif c_now >= 0:     label,color,bg = "Flat / mildly pos", "#6b7280","#f3f4f6"
    elif c_now >= -0.25: label,color,bg = "Shallow inversion", "#d97706","#fef9c3"
    else:                label,color,bg = "Deep inversion",    "#b42318","#fee2e2"
    return {"level":c_now,"trend":trend,"label":label,"color":color,"bg":bg,"ff":ff}

def get_vix_snapshot(px):
    vt = YF_PROXIES.get("vix","^VIX"); v3t = YF_PROXIES.get("vix3m","^VIX3M")
    if vt not in px.columns or v3t not in px.columns: return None
    v = px[vt].dropna(); v3m = px[v3t].dropna()
    if v.empty or v3m.empty: return None
    idx = v.index.intersection(v3m.index)
    if len(idx) == 0: return None
    vn = float(v.loc[idx[-1]]); v3n = float(v3m.loc[idx[-1]])
    vr = vn/v3n if v3n != 0 else None
    if vr is None:   sig,col,bg = "—",      "#6b7280","#f3f4f6"
    elif vr > 1.0:   sig,col,bg = "Panic",  "#b42318","#fee2e2"
    elif vr > 0.9:   sig,col,bg = "Elevated","#d97706","#fef9c3"
    else:            sig,col,bg = "Calm",   "#1f7a4f","#dcfce7"
    return {"vix":vn,"vix3m":v3n,"vratio":vr,"signal":sig,"color":col,"bg":bg}

def load_home_state():
    regime_obj, macro, px = load_current_regime()
    try:
        last_updated = macro.dropna(how="all").index.max().date()
    except Exception:
        last_updated = date.today()
    mkt_snap = []
    if isinstance(px, pd.DataFrame) and not px.empty:
        for t in MARKET_SNAPSHOT:
            if t not in px.columns: continue
            r = pct_return_over_days(px[t], 7)
            if r is not None:
                mkt_snap.append({"Asset":t, "Weekly %": round(100.0*float(r),2)})
    return {
        "last_updated":    last_updated,
        "regime":          getattr(regime_obj, "label",          "Unknown"),
        "score":           int(getattr(regime_obj, "score",      0)),
        "score_raw":       getattr(regime_obj, "score_raw",      0.0),
        "confidence":      getattr(regime_obj, "confidence",     "Unknown"),
        "momentum":        getattr(regime_obj, "momentum_label", "Unknown"),
        "score_delta":     getattr(regime_obj, "score_delta",    0) or 0,
        "allocation_tilt": build_allocation_tilt(regime_obj),
        "weekly_changes":  build_weekly_macro_changes(macro),
        "market_snapshot": pd.DataFrame(mkt_snap).sort_values("Weekly %", ascending=False) if mkt_snap else pd.DataFrame(),
        "macro":           macro,
        "px":              px,
        "regime_obj":      regime_obj,
        "curve":           get_curve_snapshot(macro),
        "vix":             get_vix_snapshot(px),
    }

# ── Guard ─────────────────────────────────────────────────────────────────────

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

home        = load_home_state()
chip        = home["regime"]
score_val   = home["score"]
score_raw   = home["score_raw"]
score_color = regime_color(chip)
score_bg_c  = regime_bg(chip)
score_delta = home["score_delta"]
mom         = str(home["momentum"]).lower()
mom_arrow   = "▲" if "improv" in mom else ("▼" if "deterio" in mom else "▶")
mom_color   = "#1f7a4f" if "improv" in mom else ("#b42318" if "deterio" in mom else "#6b7280")
macro       = home["macro"]
px          = home["px"]
crv         = home["curve"]
vix         = home["vix"]

sidebar_nav(active="Home")

# Sidebar live signals
with st.sidebar:
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:28px;font-weight:900;color:{score_color};line-height:1;'>{score_val}</div>"
        f"<div style='font-size:11px;color:rgba(0,0,0,0.55);margin-top:2px;'>{chip} · {home['confidence']}</div>",
        unsafe_allow_html=True)
    if score_delta:
        dc = "#1f7a4f" if score_delta > 0 else "#b42318"
        st.markdown(f"<div style='font-size:10px;color:{dc};font-weight:700;'>Δ {score_delta:+d} vs 21d</div>",
                    unsafe_allow_html=True)
    if vix:
        st.markdown(
            f"<div style='margin-top:8px;padding:6px 10px;border-radius:8px;background:{vix['bg']};font-size:11px;'>"
            f"<b style='color:{vix['color']};'>VIX {vix['vix']:.1f}</b> · V-Ratio "
            f"<b style='color:{vix['color']};'>{vix['vratio']:.3f}</b> · {vix['signal']}</div>",
            unsafe_allow_html=True)
    if crv:
        st.markdown(
            f"<div style='margin-top:4px;padding:6px 10px;border-radius:8px;background:{crv['bg']};font-size:11px;'>"
            f"<b style='color:{crv['color']};'>Curve {crv['level']:+.2f}pp</b> · {crv['label']}</div>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════

today_str = date.today().strftime("%A, %B %d, %Y").replace(" 0"," ")
st.markdown(
    f"""<div class="me-topbar" style="margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <div>
          <div class="me-title">Macro Engine</div>
          <div class="me-subtle">{today_str} &nbsp;·&nbsp; Updated {home['last_updated']}</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <div style="padding:7px 16px;border-radius:20px;background:{score_bg_c};border:1px solid {score_color}44;">
            <span style="font-weight:900;color:{score_color};font-size:13px;">{chip} · {score_val}</span>
          </div>
          <div style="padding:7px 14px;border-radius:20px;background:#f8fafc;border:1px solid rgba(0,0,0,0.08);">
            <span style="font-size:12px;color:{mom_color};font-weight:700;">{mom_arrow} {mom}</span>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — REGIME HERO (score + signals + allocation) full-width attention row
# ══════════════════════════════════════════════════════════════════════════════

hero_l, hero_r = st.columns([1.0, 2.0], gap="medium")

with hero_l:
    # Large score card with left border
    st.markdown(
        f"""<div style='padding:22px 20px;border-radius:16px;background:#ffffff;
              border:1px solid rgba(0,0,0,0.09);border-left:5px solid {score_color};
              box-shadow:0 2px 14px rgba(0,0,0,0.06);'>
          <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.35);
                      text-transform:uppercase;letter-spacing:0.7px;margin-bottom:8px;'>
            Macro score</div>
          <div style='display:flex;align-items:baseline;gap:10px;margin-bottom:10px;'>
            <div style='font-size:76px;font-weight:900;color:{score_color};line-height:1;
                        letter-spacing:-3px;'>{score_val}</div>
            <div style='font-size:14px;color:rgba(0,0,0,0.35);'>/ 100</div>
          </div>
          <div style='padding:8px 12px;border-radius:10px;background:{score_bg_c};margin-bottom:10px;'>
            <div style='font-size:14px;font-weight:800;color:{score_color};'>{chip}</div>
            <div style='display:flex;gap:6px;margin-top:5px;flex-wrap:wrap;'>
              <span style='font-size:10px;background:rgba(0,0,0,0.06);padding:2px 8px;border-radius:5px;color:rgba(0,0,0,0.55);'>{home['confidence']} confidence</span>
              <span style='font-size:10px;color:{mom_color};font-weight:800;background:rgba(0,0,0,0.04);padding:2px 8px;border-radius:5px;'>{mom_arrow} {mom}</span>
              <span style='font-size:10px;background:rgba(0,0,0,0.05);padding:2px 8px;border-radius:5px;color:rgba(0,0,0,0.50);'>Δ21d {score_delta:+d}</span>
            </div>
          </div>
          <div style='margin-bottom:10px;'>{SCORE_LEGEND_HTML}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("")
    if st.button("Score breakdown →", use_container_width=True, key="btn_score"):
        safe_switch_page("pages/1_Regime_Deep_Dive.py")

with hero_r:
    # Allocation tilt + key signals side by side
    ha, hb = st.columns(2, gap="medium")

    with ha:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Score drivers</div>", unsafe_allow_html=True)
            st.caption("What is moving the regime score right now.")
            comps = getattr(home["regime_obj"], "components", None)
            if isinstance(comps, dict) and comps:
                rows = sorted(
                    [{"name": c.get("name",k)[:18],
                      "contrib": float(_cc(c)),
                      "z": float(c.get("zscore") or 0)}
                     for k, c in comps.items() if isinstance(c, dict)],
                    key=lambda x: x["contrib"]
                )
                max_c = max(abs(r["contrib"]) for r in rows) or 0.1
                for r in rows:
                    bar_pct = min(abs(r["contrib"]) / max_c * 100, 100)
                    bc      = "#1f7a4f" if r["contrib"] >= 0 else "#b42318"
                    bg_row  = "rgba(31,122,79,0.04)" if r["contrib"]>=0 else "rgba(180,35,24,0.04)"
                    st.markdown(
                        f"<div style='padding:6px 8px;border-radius:8px;background:{bg_row};"
                        f"margin-bottom:4px;'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;margin-bottom:3px;'>"
                        f"<span style='font-size:11px;font-weight:700;color:rgba(0,0,0,0.75);'>{r['name']}</span>"
                        f"<div style='display:flex;gap:8px;align-items:center;'>"
                        f"<span style='font-size:9px;color:rgba(0,0,0,0.40);'>z {r['z']:+.2f}</span>"
                        f"<span style='font-size:11px;font-weight:900;color:{bc};'>{r['contrib']:+.3f}</span>"
                        f"</div></div>"
                        f"<div style='background:rgba(0,0,0,0.06);border-radius:3px;height:4px;overflow:hidden;'>"
                        f"<div style='width:{bar_pct:.0f}%;height:100%;background:{bc};border-radius:3px;'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True)
            else:
                st.caption("Component data unavailable.")
            st.markdown("")
            if st.button("Regime Playbook →", use_container_width=True, key="btn_playbook"):
                safe_switch_page("pages/7_Regime_Playbook.py")

    with hb:
        with st.container(border=True):
            st.markdown("<div class='me-rowtitle'>Live signals</div>", unsafe_allow_html=True)

            def _signal_row(label, value, sub, color):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:7px 0;border-bottom:1px solid rgba(0,0,0,0.04);'>"
                    f"<div><div style='font-size:11px;font-weight:700;color:rgba(0,0,0,0.75);'>{label}</div>"
                    f"<div style='font-size:9px;color:rgba(0,0,0,0.40);'>{sub}</div></div>"
                    f"<div style='font-size:14px;font-weight:900;color:{color};'>{value}</div>"
                    f"</div>", unsafe_allow_html=True)

            # Curve
            if crv:
                c_col = crv["color"]
                _signal_row("Curve 2s10s", f"{crv['level']:+.2f}pp", crv["label"], c_col)
            # Real yield
            if "real10" in macro.columns:
                rv = float(macro["real10"].dropna().iloc[-1])
                rv_c = "#b42318" if rv > 1.5 else ("#d97706" if rv > 0.5 else "#1f7a4f")
                _signal_row("Real yield", f"{rv:.2f}%", "Restrictive >1.5%", rv_c)
            # HY OAS
            if "hy_oas" in macro.columns:
                hy = float(macro["hy_oas"].dropna().iloc[-1])
                _, _, hy_d = delta_over_days(macro["hy_oas"], 7)
                hy_sub = f"7d {hy_d:+.2f}pp" if hy_d else ""
                hy_c = "#b42318" if hy > 5 else ("#d97706" if hy > 4 else "#1f7a4f")
                _signal_row("HY OAS", f"{hy:.2f}%", hy_sub, hy_c)
            # VIX / V-Ratio
            if vix:
                vc = vix["color"]
                _signal_row("V-Ratio", f"{vix['vratio']:.3f}", vix["signal"], vc)

            st.markdown("")
            if st.button("Transition Watch →", use_container_width=True, key="btn_tw"):
                safe_switch_page("pages/8_Transition_Watch.py")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — TWO KEY CHARTS + WHAT CHANGED
# The two most important macro charts at a glance
# ══════════════════════════════════════════════════════════════════════════════

chart_l, chart_m, chart_r = st.columns([1.0, 1.0, 1.1], gap="medium")

def _mini_chart_layout(height=175):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=6, r=6, t=28, b=6),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=8),
                   tickformat="%b '%y"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=8),
                   zeroline=False),
    )
    return fig

# Chart 1 — HY OAS (credit stress — the leading indicator)
with chart_l:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>HY Credit spreads (OAS)</div>",
                    unsafe_allow_html=True)
        if "hy_oas" in macro.columns:
            hy_s = macro["hy_oas"].dropna()
            # 90-day slice
            hy_90 = hy_s.loc[hy_s.index >= hy_s.index.max() - pd.Timedelta(days=120)]
            hy_now_v = float(hy_90.iloc[-1])
            hy_1m_v  = float(hy_90.iloc[-22]) if len(hy_90) >= 22 else hy_now_v
            hy_d     = hy_now_v - hy_1m_v
            hy_col   = "#b42318" if hy_now_v > 5 else ("#d97706" if hy_now_v > 4 else "#1f7a4f")

            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"margin-bottom:4px;'>"
                f"<span style='font-size:20px;font-weight:900;color:{hy_col};'>{hy_now_v:.2f}%</span>"
                f"<span style='font-size:11px;font-weight:700;color:{'#b42318' if hy_d>0 else '#1f7a4f'};'>"
                f"{'↑' if hy_d>0 else '↓'} {abs(hy_d):.2f}pp 1m</span>"
                f"</div>", unsafe_allow_html=True)

            fig_hy = _mini_chart_layout(185)
            # Percentile bands
            p25 = float(hy_s.quantile(0.25)); p75 = float(hy_s.quantile(0.75))
            fig_hy.add_hrect(y0=0,   y1=p25, fillcolor="rgba(31,122,79,0.06)",  line_width=0)
            fig_hy.add_hrect(y0=p75, y1=15,  fillcolor="rgba(180,35,24,0.06)", line_width=0)
            fig_hy.add_trace(go.Scatter(
                x=hy_90.index, y=hy_90.values,
                mode="lines", line=dict(color=hy_col, width=2),
                fill="tozeroy", fillcolor=f"rgba({int(hy_col[1:3],16)},{int(hy_col[3:5],16)},{int(hy_col[5:7],16)},0.06)"
            ))
            # Now line
            fig_hy.add_hline(y=hy_now_v, line_color=hy_col, line_width=1.2, line_dash="dot",
                             annotation_text=f"{hy_now_v:.2f}%",
                             annotation_position="right", annotation_font_size=8)
            vals_hy = list(hy_90.values)
            lo = min(vals_hy) - max((max(vals_hy)-min(vals_hy))*0.12, 0.05)
            hi = max(vals_hy) + max((max(vals_hy)-min(vals_hy))*0.12, 0.05)
            fig_hy.update_yaxes(range=[lo, hi])
            st.plotly_chart(fig_hy, use_container_width=True)
        else:
            st.caption("HY OAS unavailable.")
        if st.button("Credit & Macro →", use_container_width=True, key="btn_credit"):
            safe_switch_page("pages/2_Macro_Charts.py")

# Chart 2 — 2s10s Yield curve
with chart_m:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Yield curve (2s10s)</div>",
                    unsafe_allow_html=True)
        if "y10" in macro.columns and "y2" in macro.columns:
            curve_s  = (macro["y10"] - macro["y2"]).dropna()
            curve_90 = curve_s.loc[curve_s.index >= curve_s.index.max() - pd.Timedelta(days=120)]
            c_now_v  = float(curve_90.iloc[-1])
            c_1m_v   = float(curve_90.iloc[-22]) if len(curve_90) >= 22 else c_now_v
            c_d      = c_now_v - c_1m_v
            c_col    = "#1f7a4f" if c_now_v >= 0.75 else ("#6b7280" if c_now_v >= 0 else "#b42318")

            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"margin-bottom:4px;'>"
                f"<span style='font-size:20px;font-weight:900;color:{c_col};'>{c_now_v:+.2f}pp</span>"
                f"<span style='font-size:11px;font-weight:700;color:{'#1f7a4f' if c_d>0 else '#b42318'};'>"
                f"{'↑' if c_d>0 else '↓'} {abs(c_d):.2f}pp 1m</span>"
                f"</div>", unsafe_allow_html=True)

            fig_cv = _mini_chart_layout(185)
            # Regime shading
            fig_cv.add_hrect(y0=-5,  y1=0,    fillcolor="rgba(180,35,24,0.06)",  line_width=0)
            fig_cv.add_hrect(y0=0,   y1=0.75, fillcolor="rgba(217,119,6,0.04)", line_width=0)
            fig_cv.add_hrect(y0=0.75,y1=3,    fillcolor="rgba(31,122,79,0.05)", line_width=0)
            fig_cv.add_hline(y=0, line_color="#94a3b8", line_width=1.5)
            fig_cv.add_trace(go.Scatter(
                x=curve_90.index, y=curve_90.values,
                mode="lines", line=dict(color="#1d4ed8", width=2),
                fill="tozeroy", fillcolor="rgba(29,78,216,0.06)"
            ))
            fig_cv.add_hline(y=c_now_v, line_color=c_col, line_width=1.2, line_dash="dot",
                             annotation_text=f"{c_now_v:+.2f}pp",
                             annotation_position="right", annotation_font_size=8)
            vals_cv = list(curve_90.values)
            lo_c = min(vals_cv) - max((max(vals_cv)-min(vals_cv))*0.15, 0.05)
            hi_c = max(vals_cv) + max((max(vals_cv)-min(vals_cv))*0.15, 0.05)
            fig_cv.update_yaxes(range=[lo_c, hi_c])
            st.plotly_chart(fig_cv, use_container_width=True)
        else:
            st.caption("Curve data unavailable.")
        if st.button("Curve View →", use_container_width=True, key="btn_curve"):
            safe_switch_page("pages/9_Curve_View.py")

# What changed
with chart_r:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>What changed this week</div>",
                    unsafe_allow_html=True)
        changes = home["weekly_changes"]
        if changes:
            for name, prev, cur, delta, label, inverse in changes:
                good  = (delta < 0) if inverse else (delta > 0)
                d_col = "#1f7a4f" if good else "#b42318"
                d_bg  = "rgba(31,122,79,0.04)" if good else "rgba(180,35,24,0.04)"
                arrow = "↓" if delta < 0 else "↑"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:9px 10px;border-radius:9px;background:{d_bg};margin-bottom:5px;'>"
                    f"<div><div style='font-size:12px;font-weight:700;color:rgba(0,0,0,0.80);'>{name}</div>"
                    f"<div style='font-size:10px;color:rgba(0,0,0,0.40);'>"
                    f"{prev:.2f} → {cur:.2f} ({label})</div></div>"
                    f"<span style='font-size:15px;font-weight:900;color:{d_col};'>"
                    f"{arrow} {abs(delta):.2f}</span>"
                    f"</div>", unsafe_allow_html=True)
        else:
            st.caption("Weekly changes unavailable.")
        st.markdown("")
        if st.button("Morning Brief →", use_container_width=True, key="btn_brief"):
            safe_switch_page("pages/0_Morning_Brief.py")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — PAGE NAVIGATION GRID
# This is what the current home page is missing entirely
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.45);"
    "text-transform:uppercase;letter-spacing:0.8px;margin:4px 0 12px;"
    "padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,0.07);'>"
    "Explore</div>", unsafe_allow_html=True)

# 4 columns of nav cards — icon, title, description, button
NAV_CARDS = [
    ("📋", "Regime Playbook",
     "Ranked trade signals · Kelly sizing · factor rotation · divergence alerts",
     "pages/7_Regime_Playbook.py", "#1d4ed8", "#eff6ff"),
    ("🔔", "Transition Watch",
     "Markov transition probabilities · score trajectory · flip scenarios",
     "pages/8_Transition_Watch.py", "#d97706", "#fef9c3"),
    ("📐", "Curve View",
     "Full term structure · steepener/flattener type · real rate decomposition",
     "pages/9_Curve_View.py", "#1f7a4f", "#dcfce7"),
    ("⚡", "Volatility View",
     "VIX term structure · V-Ratio · realized vol · stress regimes",
     "pages/6_Volatility_View.py", "#7c3aed", "#f5f3ff"),
    ("🏦", "Fed & Liquidity",
     "Balance sheet impulse · real rate path · dollar cycle · policy stance",
     "pages/10_Fed_Liquidity.py", "#0e7490", "#ecfeff"),
    ("📊", "Credit & Macro",
     "HY OAS bands · curve vs credit · breadth ratios · inflation proxies",
     "pages/2_Macro_Charts.py", "#b42318", "#fee2e2"),
    ("🔄", "Rotation & Setups",
     "Pair signals · Clayton copula risk · RRG quadrant · pair explorer",
     "pages/4_Rotation_Setups.py", "#f97316", "#fff7ed"),
    ("☀️", "Morning Brief",
     "Dominant trade · thesis · falsification conditions · daily synthesis",
     "pages/0_Morning_Brief.py", "#6b7280", "#f8fafc"),
]

nav_cols = st.columns(4, gap="medium")
for i, (icon, title, desc, path, color, bg) in enumerate(NAV_CARDS):
    col = nav_cols[i % 4]
    with col:
        st.markdown(
            f"<div style='padding:14px 16px;border-radius:13px;background:{bg};"
            f"border:1px solid {color}33;margin-bottom:12px;min-height:90px;'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            f"<span style='font-size:16px;'>{icon}</span>"
            f"<span style='font-size:13px;font-weight:800;color:{color};'>{title}</span>"
            f"</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.55);line-height:1.4;'>{desc}</div>"
            f"</div>", unsafe_allow_html=True)
        if col.button(f"Open →", key=f"nav_{i}", use_container_width=True):
            safe_switch_page(path)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — RELATIVE STRENGTH BAR CHART (compact)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.45);"
    "text-transform:uppercase;letter-spacing:0.8px;margin:4px 0 10px;"
    "padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,0.07);'>"
    "Market rotation — relative strength vs SPY</div>", unsafe_allow_html=True)

rs_col, mkt_col = st.columns([1.6, 1.0], gap="large")

with rs_col:
    rs = compute_rs_heatmap(period="1y")
    hz_opts = ["1w","1m","3m"]
    ctl1, ctl2, ctl3 = st.columns([1,1,3], gap="small")
    with ctl1:
        hz = st.radio("Horizon", hz_opts, horizontal=True, index=1, key="rs_hz")
    with ctl2:
        pass
    with ctl3:
        ldr, lag = rs[hz].dropna().sort_values(ascending=False).head(3), \
                   rs[hz].dropna().sort_values().head(3)
        ltxt = ", ".join(ldr.index.tolist()); gtxt = ", ".join(lag.index.tolist())
        st.markdown(
            f"<div style='font-size:11px;margin-top:6px;'>"
            f"Leaders: <b style='color:#1f7a4f;'>{ltxt}</b> &nbsp;·&nbsp; "
            f"Laggards: <b style='color:#b42318;'>{gtxt}</b></div>",
            unsafe_allow_html=True)

    bar_df = rs[[hz]].copy().reset_index()
    bar_df.columns = ["Ticker","RS"]
    bar_df["RS"] = pd.to_numeric(bar_df["RS"], errors="coerce")
    bar_df = bar_df.dropna().sort_values("RS", ascending=False)

    chart = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                  cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Ticker:N", sort="-x", title=None),
            x=alt.X("RS:Q", title="RS vs SPY", axis=alt.Axis(format="+.1%")),
            color=alt.condition(alt.datum.RS > 0,
                                alt.value("#1f7a4f"), alt.value("#b42318")),
            tooltip=["Ticker", alt.Tooltip("RS:Q", format="+.2%", title="RS vs SPY")],
        ).properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    if st.button("Rotation & setups →", use_container_width=False, key="btn_rot"):
        safe_switch_page("pages/4_Rotation_Setups.py")

with mkt_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Market returns this week</div>",
                    unsafe_allow_html=True)
        mkt_df = home["market_snapshot"]
        if isinstance(mkt_df, pd.DataFrame) and not mkt_df.empty:
            sorted_ret = mkt_df.to_dict("records")
            max_abs    = max(abs(r["Weekly %"]) for r in sorted_ret) or 1
            for row in sorted_ret:
                t = row["Asset"]; r = row["Weekly %"]
                rc    = "#1f7a4f" if r > 0 else "#b42318"
                bar_w = min(abs(r) / max_abs * 80, 80)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"padding:6px 0;border-bottom:1px solid rgba(0,0,0,0.04);'>"
                    f"<span style='font-size:12px;font-weight:700;color:rgba(0,0,0,0.65);"
                    f"width:36px;flex-shrink:0;'>{t}</span>"
                    f"<div style='width:{bar_w:.0f}px;height:5px;border-radius:3px;"
                    f"background:{rc};flex-shrink:0;'></div>"
                    f"<span style='font-size:12px;font-weight:800;color:{rc};'>{r:+.1f}%</span>"
                    f"</div>", unsafe_allow_html=True)
        else:
            st.caption("Market data unavailable.")
        st.markdown("")
        if st.button("Fed & Liquidity →", use_container_width=True, key="btn_fed"):
            safe_switch_page("pages/10_Fed_Liquidity.py")

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)