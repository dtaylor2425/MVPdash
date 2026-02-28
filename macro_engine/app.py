# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3
from src.compute import component_contribution
from src.ui import (
    inject_css, sidebar_nav, safe_switch_page,
    regime_color, delta_badge_html, SCORE_LEGEND_HTML,
)

st.set_page_config(
    page_title="Macro Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Constants ─────────────────────────────────────────────────────────────────

ROTATION_TICKERS = [
    "XLE", "XLF", "XLK", "XLI", "XLP", "XLV",
    "GLD", "UUP", "IWM", "QQQ",
    "IGV", "SMH",
    "SPY",
]
MARKET_SNAPSHOT = ["SPY", "QQQ", "IWM", "HYG", "TLT", "UUP", "GLD", "XLE"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nearest_before(series, dt):
    s = series.dropna()
    idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None

def delta_over_days(series, days):
    s = series.dropna()
    if s.empty:
        return None, None, None
    end = pd.Timestamp(s.index.max())
    end_i  = _nearest_before(s, end)
    prev_i = _nearest_before(s, end - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None, None, None
    latest = float(s.loc[end_i])
    prev   = float(s.loc[prev_i])
    return latest, prev, float(latest - prev)

def pct_return_over_days(series, days):
    s = series.dropna()
    if len(s) < 2:
        return None
    end_i  = _nearest_before(s, s.index.max())
    prev_i = _nearest_before(s, s.index.max() - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None
    a, b = float(s.loc[end_i]), float(s.loc[prev_i])
    return None if b == 0 else (a / b) - 1.0

def _component_contribution(c):
    return component_contribution(c)

def _trend_phrase(name, trend_up):
    nm = (name or "").lower()
    if trend_up is None:
        return "steady"
    if "credit" in nm or "spread" in nm:
        return "widening" if int(trend_up) == 1 else "tightening"
    if "dollar" in nm or "usd" in nm:
        return "strengthening" if int(trend_up) == 1 else "weakening"
    return "rising" if int(trend_up) == 1 else "falling"

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_current_regime():
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set")
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro").sort_index()
    vix_tickers = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
    all_tickers = list(dict.fromkeys(ROTATION_TICKERS + MARKET_SNAPSHOT + vix_tickers))
    px = fetch_prices(all_tickers, period="5y")
    px = pd.DataFrame() if (px is None or px.empty) else px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63, momentum_lookback_days=63)
    return regime, macro, px

@st.cache_data(show_spinner=False)
def compute_rs_heatmap(period="1y"):
    tickers = [
        "GLD", "XLE", "XLP", "XLI", "IWM", "RSP",
        "HYG", "TLT", "UUP", "QQQ", "XLK", "XLF", "XLV", "IGV", "SMH",
    ]
    base = "SPY"
    px = fetch_prices(list(dict.fromkeys([base] + tickers)), period=period)
    if px is None or px.empty or base not in px.columns:
        return pd.DataFrame(index=tickers, columns=["1w", "1m", "3m"], data=np.nan)
    px   = px.dropna(how="all")
    horiz = {"1w": 5, "1m": 21, "3m": 63}
    out  = pd.DataFrame(index=tickers, columns=list(horiz.keys()), dtype=float)
    b    = px[base].dropna()
    for t in tickers:
        if t not in px.columns:
            continue
        s   = px[t].dropna()
        idx = s.index.intersection(b.index)
        if len(idx) < 90:
            continue
        s, bb = s.loc[idx], b.loc[idx]
        for lab, n in horiz.items():
            if len(idx) > n:
                out.loc[t, lab] = float((s.iloc[-1]/s.iloc[-1-n]) - (bb.iloc[-1]/bb.iloc[-1-n]))
    return out

def leaders_laggards(rs_df, horizon):
    s = rs_df[horizon].dropna().sort_values(ascending=False)
    return s.head(5), s.tail(5)

# ── Business logic ────────────────────────────────────────────────────────────

def build_dynamic_drivers(regime_obj, macro, px):
    comps = getattr(regime_obj, "components", None)
    drivers = []
    if isinstance(comps, dict) and comps:
        rows = sorted(
            [(c.get("name", k), _component_contribution(c), c.get("trend_up"), c.get("zscore"))
             for k, c in comps.items() if isinstance(c, dict)],
            key=lambda x: abs(x[1]), reverse=True,
        )
        for nm, contrib, trend_up, z in rows[:3]:
            direction = "supportive" if contrib >= 0 else "drag"
            trend     = _trend_phrase(nm, trend_up)
            ztxt      = "" if z is None or pd.isna(z) else f"z {float(z):.2f}"
            subtitle  = f"{trend} \u2022 {direction}" + (f" \u2022 {ztxt}" if ztxt else "")
            drivers.append((str(nm), subtitle))
    return drivers[:3] or [("Drivers unavailable", "No components returned")]

def build_allocation_tilt(regime_obj):
    alloc  = getattr(regime_obj, "allocation", None)
    stance = {}
    if isinstance(alloc, dict):
        s = alloc.get("stance", {})
        stance = s if isinstance(s, dict) else {}
    return [(k, str(stance[k])) for k in ["Equities", "Credit", "Duration", "USD", "Commodities"]
            if k in stance and stance[k] is not None]

def build_key_risk(regime_obj):
    comps = getattr(regime_obj, "components", None)
    if isinstance(comps, dict) and comps:
        rows = sorted(
            [(c.get("name", k), _component_contribution(c), c.get("zscore"))
             for k, c in comps.items() if isinstance(c, dict)],
            key=lambda x: x[1],
        )
        nm, contrib, z = rows[0]
        ztxt = "" if z is None or pd.isna(z) else f" (z {float(z):.2f})"
        verb = "staying adverse" if contrib < 0 else "reversing against signal"
        return f"{nm} {verb}{ztxt}"
    return "Regime momentum reversing on a growth or credit shock."

def build_weekly_macro_changes(macro):
    items = []
    if not isinstance(macro, pd.DataFrame) or macro.empty:
        return items
    if "hy_oas" in macro.columns:
        l, p, d = delta_over_days(macro["hy_oas"], 7)
        if d is not None:
            items.append(("Credit spreads (HY OAS)", p, l, d, "tighter" if d < 0 else "wider", True))
    if "y10" in macro.columns and "y2" in macro.columns:
        l, p, d = delta_over_days((macro["y10"] - macro["y2"]).dropna(), 7)
        if d is not None:
            items.append(("Curve (10y \u2212 2y)", p, l, d, "steeper" if d > 0 else "flatter", False))
    if "real10" in macro.columns:
        l, p, d = delta_over_days(macro["real10"], 7)
        if d is not None:
            items.append(("Real 10y yield", p, l, d, "higher" if d > 0 else "lower", True))
    if "dollar_broad" in macro.columns:
        l, p, d = delta_over_days(macro["dollar_broad"], 10)
        if d is not None:
            items.append(("Dollar broad", p, l, d, "stronger" if d > 0 else "weaker", False))
    return items

def cpi_yoy(macro):
    if "cpi" not in macro.columns:
        return None
    c = macro["cpi"].dropna()
    if len(c) < 13:
        return None
    return (c.pct_change(12) * 100).dropna()

def get_curve_snapshot(macro):
    if "y10" not in macro.columns or "y2" not in macro.columns:
        return None
    curve = (macro["y10"] - macro["y2"]).dropna()
    if curve.empty:
        return None
    c_now = float(curve.iloc[-1])
    c_63  = float(curve.iloc[-64]) if len(curve) > 64 else None
    trend = ("Steepening \u2191" if c_now > c_63 else "Flattening \u2193") if c_63 is not None else "\u2014"
    ff    = float(macro["fed_funds"].dropna().iloc[-1]) if "fed_funds" in macro.columns and not macro["fed_funds"].dropna().empty else None
    cpi   = cpi_yoy(macro)
    cpi_v = float(cpi.iloc[-1]) if cpi is not None and not cpi.empty else None
    if c_now >= 0.75:
        label, color, bg = "Steep", "#1f7a4f", "#dcfce7"
    elif c_now >= 0:
        label, color, bg = "Flat / mildly positive", "#6b7280", "#f3f4f6"
    elif c_now >= -0.25:
        label, color, bg = "Shallow inversion", "#d97706", "#fef9c3"
    else:
        label, color, bg = "Deep inversion", "#b42318", "#fee2e2"
    return {"level": c_now, "trend": trend, "label": label, "color": color,
            "bg": bg, "ff": ff, "cpi": cpi_v}

def get_vix_snapshot(px):
    vix_t   = YF_PROXIES.get("vix",   "^VIX")
    vix3m_t = YF_PROXIES.get("vix3m", "^VIX3M")
    if vix_t not in px.columns or vix3m_t not in px.columns:
        return None
    v   = px[vix_t].dropna()
    v3m = px[vix3m_t].dropna()
    if v.empty or v3m.empty:
        return None
    idx = v.index.intersection(v3m.index)
    if len(idx) == 0:
        return None
    vix_now   = float(v.loc[idx[-1]])
    vix3m_now = float(v3m.loc[idx[-1]])
    vratio    = vix_now / vix3m_now if vix3m_now != 0 else None
    if vratio is None:
        signal, color, bg = "\u2014", "#6b7280", "#f3f4f6"
    elif vratio > 1.0:
        signal, color, bg = "Panic", "#b42318", "#fee2e2"
    elif vratio > 0.9:
        signal, color, bg = "Elevated", "#d97706", "#fef9c3"
    else:
        signal, color, bg = "Calm", "#1f7a4f", "#dcfce7"
    return {"vix": vix_now, "vix3m": vix3m_now, "vratio": vratio,
            "signal": signal, "color": color, "bg": bg}

def load_home_state():
    regime_obj, macro, px = load_current_regime()
    try:
        last_updated = macro.index.max().date()
    except Exception:
        last_updated = date.today()
    macro_snap, mkt_snap = [], []
    for name, prev, cur, dlt, _lab, _inv in build_weekly_macro_changes(macro):
        macro_snap.append({"Metric": name, "Level": round(cur, 2), "\u0394 1w": round(dlt, 3)})
    if isinstance(px, pd.DataFrame) and not px.empty:
        for t in MARKET_SNAPSHOT:
            if t not in px.columns:
                continue
            r = pct_return_over_days(px[t], 7)
            if r is not None:
                mkt_snap.append({"Asset": t, "Weekly %": round(100.0 * float(r), 2)})
    mkt_df = pd.DataFrame(mkt_snap).sort_values("Weekly %", ascending=False) if mkt_snap else pd.DataFrame()
    return {
        "last_updated":    last_updated,
        "regime":          getattr(regime_obj, "label",          "Unknown"),
        "score":           int(getattr(regime_obj, "score",      0)),
        "confidence":      getattr(regime_obj, "confidence",     "Unknown"),
        "momentum":        getattr(regime_obj, "momentum_label", "Unknown"),
        "allocation_tilt": build_allocation_tilt(regime_obj),
        "weekly_changes":  build_weekly_macro_changes(macro),
        "drivers":         build_dynamic_drivers(regime_obj, macro, px),
        "key_risk":        build_key_risk(regime_obj),
        "macro_snapshot":  pd.DataFrame(macro_snap) if macro_snap else pd.DataFrame(),
        "market_snapshot": mkt_df,
        "macro":           macro,
        "px":              px,
        "regime_obj":      regime_obj,
        "curve":           get_curve_snapshot(macro),
        "vix":             get_vix_snapshot(px),
    }

# ── Guard ─────────────────────────────────────────────────────────────────────

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

home        = load_home_state()
chip        = home["regime"]
dot         = regime_color(chip)
score_val   = home["score"]
score_color = "#1f7a4f" if score_val >= 60 else ("#b42318" if score_val < 40 else "#6b7280")

# ── Sidebar live signals ──────────────────────────────────────────────────────

sidebar_nav(active="Home")

with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:rgba(0,0,0,0.4);"
        "text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>Live signals</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:22px;font-weight:900;color:{score_color};'>{score_val}</div>"
        f"<div style='font-size:12px;color:rgba(0,0,0,0.55);margin-top:2px;'>"
        f"{chip} &middot; {home['confidence']} confidence</div>",
        unsafe_allow_html=True,
    )
    vix_snap = home.get("vix")
    if vix_snap:
        vix_color = vix_snap["color"]
        vix_bg    = vix_snap["bg"]
        st.markdown(
            f"<div style='margin-top:10px;padding:8px 10px;border-radius:10px;"
            f"background:{vix_bg};font-size:12px;'>"
            f"<span style='font-weight:800;color:{vix_color};'>VIX {vix_snap['vix']:.1f}</span>"
            f"<span style='color:rgba(0,0,0,0.55);'> &middot; V-Ratio "
            f"<b style='color:{vix_color};'>{vix_snap['vratio']:.3f}</b>"
            f" &middot; {vix_snap['signal']}</span></div>",
            unsafe_allow_html=True,
        )
    crv_snap = home.get("curve")
    if crv_snap:
        crv_color = crv_snap["color"]
        crv_bg    = crv_snap["bg"]
        st.markdown(
            f"<div style='margin-top:6px;padding:8px 10px;border-radius:10px;"
            f"background:{crv_bg};font-size:12px;'>"
            f"<span style='font-weight:800;color:{crv_color};'>"
            f"Curve {crv_snap['level']:+.2f}pp</span>"
            f"<span style='color:rgba(0,0,0,0.55);'> &middot; {crv_snap['label']}</span></div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="me-topbar">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  gap:12px;flex-wrap:wrap;">
        <div>
          <div class="me-title">Macro Engine</div>
          <div class="me-subtle">Regime &nbsp;·&nbsp; rotation &nbsp;·&nbsp; volatility &nbsp;·&nbsp; curve</div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
          <div class="me-chip">
            <span class="me-dot" style="background:{dot}"></span>
            <span style="color:{score_color};font-weight:700;">{chip}</span>
          </div>
          <div class="me-subtle">Updated {home['last_updated']}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1  ·  Score  |  Allocation tilt  |  Curve  |  Volatility
# ══════════════════════════════════════════════════════════════════════════════

c_score, c_chg_top, c_curve, c_vix = st.columns([1.0, 1.35, 1.1, 1.1], gap="medium")

# ─── Score ────────────────────────────────────────────────────────────────────
with c_score:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Macro score</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='me-kpi' style='color:{score_color};'>{score_val}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(SCORE_LEGEND_HTML, unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;'>"
            f"<span class='me-pill'>Confidence: {home['confidence']}</span>"
            f"<span class='me-pill'>Momentum: {str(home['momentum']).lower()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Score breakdown →", use_container_width=True, key="btn_score"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

# ─── What changed this week (row 1, position 2) ──────────────────────────────
with c_chg_top:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>What changed this week</div>", unsafe_allow_html=True)
        if home["weekly_changes"]:
            for name, prev, cur, delta, label, is_inverse in home["weekly_changes"]:
                badge = delta_badge_html(delta, inverse=is_inverse)
                st.markdown(
                    f"<div class='me-li'><div>"
                    f"<div class='me-li-name'>{name}</div>"
                    f"<div class='me-li-sub'>{prev:.2f} → {cur:.2f} ({label})</div>"
                    f"</div>"
                    f"<div class='me-li-right'>{badge}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Weekly changes unavailable.")
        st.markdown("")
        if st.button("Deep dive →", use_container_width=True, key="btn_weekly_r1"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

# ─── Yield curve snapshot ─────────────────────────────────────────────────────
with c_curve:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Yield curve</div>", unsafe_allow_html=True)
        crv = home.get("curve")
        if crv:
            crv_col = crv["color"]
            st.markdown(
                f"<div style='padding:10px 12px;border-radius:10px;background:{crv['bg']};"
                f"margin-bottom:10px;'>"
                f"<div style='font-size:10px;font-weight:700;color:{crv_col};"
                f"text-transform:uppercase;letter-spacing:0.5px;'>{crv['label']}</div>"
                f"<div style='font-size:26px;font-weight:900;color:{crv_col};line-height:1.1;'>"
                f"{crv['level']:+.2f}<span style='font-size:13px;font-weight:500;'>pp</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='me-li'><div class='me-li-name'>Trend (3m)</div>"
                f"<div class='me-li-right' style='font-weight:700;'>{crv['trend']}</div></div>",
                unsafe_allow_html=True,
            )
            if crv["ff"] is not None:
                st.markdown(
                    f"<div class='me-li'><div class='me-li-name'>Fed funds</div>"
                    f"<div class='me-li-right'>{crv['ff']:.2f}%</div></div>",
                    unsafe_allow_html=True,
                )
            if crv["cpi"] is not None:
                st.markdown(
                    f"<div class='me-li'><div class='me-li-name'>CPI YoY</div>"
                    f"<div class='me-li-right'>{crv['cpi']:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Curve data unavailable.")
        st.markdown("")
        if st.button("Curve context →", use_container_width=True, key="btn_curve"):
            safe_switch_page("pages/2_Macro_Charts.py", tab="curve")

# ─── VIX / V-Ratio ───────────────────────────────────────────────────────────
with c_vix:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Volatility</div>", unsafe_allow_html=True)
        vix = home.get("vix")
        if vix:
            vc = vix["color"]
            vb = vix["bg"]
            st.markdown(
                f"<div style='padding:10px 12px;border-radius:10px;background:{vb};"
                f"margin-bottom:10px;'>"
                f"<div style='font-size:10px;font-weight:700;color:{vc};"
                f"text-transform:uppercase;letter-spacing:0.5px;'>"
                f"V-Ratio &middot; {vix['signal']}</div>"
                f"<div style='font-size:26px;font-weight:900;color:{vc};line-height:1.1;'>"
                f"{vix['vratio']:.3f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='me-li'><div class='me-li-name'>VIX spot</div>"
                f"<div class='me-li-right' style='font-weight:700;'>{vix['vix']:.1f}</div></div>"
                f"<div class='me-li'><div class='me-li-name'>VIX3M</div>"
                f"<div class='me-li-right'>{vix['vix3m']:.1f}</div></div>",
                unsafe_allow_html=True,
            )
            st.caption("V-Ratio > 1 = panic. < 0.9 = calm.")
        else:
            st.caption("VIX unavailable — check ^VIX / ^VIX3M in config.")
        st.markdown("")
        if st.button("VIX term structure →", use_container_width=True, key="btn_vix"):
            safe_switch_page("pages/2_Macro_Charts.py", tab="volatility")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2  ·  Key drivers  |  What changed  |  Component contributions
# ══════════════════════════════════════════════════════════════════════════════

c_drv, c_tilt, c_why = st.columns([1.0, 1.3, 1.2], gap="medium")

# ─── Key drivers ─────────────────────────────────────────────────────────────
with c_drv:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key drivers</div>", unsafe_allow_html=True)
        for i, (title, sub) in enumerate(home["drivers"], start=1):
            st.markdown(
                f"<div class='me-li'><div>"
                f"<div class='me-li-name'>{i}. {title}</div>"
                f"<div class='me-li-sub'>{sub}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("")
        if st.button("All drivers →", use_container_width=True, key="btn_drivers"):
            safe_switch_page("pages/5_Drivers.py")

# ─── Allocation tilt (row 2) ──────────────────────────────────────────────────
with c_tilt:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Allocation tilt</div>", unsafe_allow_html=True)
        if home["allocation_tilt"]:
            for k, v in home["allocation_tilt"]:
                vc = v.lower()
                if "overweight" in vc:
                    badge_cls = "me-badge-green"
                elif "underweight" in vc:
                    badge_cls = "me-badge-red"
                else:
                    badge_cls = "me-badge-neutral"
                st.markdown(
                    f"<div class='me-li'>"
                    f"<div class='me-li-name'>{k}</div>"
                    f"<span class='me-badge {badge_cls}'>{v}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Tilt unavailable.")
        st.markdown("")
        if st.button("Regime deep dive →", use_container_width=True, key="btn_tilt"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

# ─── Why this regime (bar chart) ─────────────────────────────────────────────
with c_why:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Why this regime</div>", unsafe_allow_html=True)
        comps = getattr(home["regime_obj"], "components", None)
        if isinstance(comps, dict) and comps:
            rows = [
                {"Component": c.get("name", k)[:16], "Contribution": float(_component_contribution(c))}
                for k, c in comps.items() if isinstance(c, dict)
            ]
            comp_df = pd.DataFrame(rows).sort_values("Contribution", ascending=True)
            chart = (
                alt.Chart(comp_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                          cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("Component:N", sort=None, title=None,
                            axis=alt.Axis(labelFontSize=11, labelLimit=120)),
                    x=alt.X("Contribution:Q", title=None,
                            axis=alt.Axis(labelFontSize=10, format=".2f")),
                    color=alt.condition(
                        alt.datum.Contribution > 0,
                        alt.value("#1f7a4f"),
                        alt.value("#b42318"),
                    ),
                    tooltip=["Component",
                             alt.Tooltip("Contribution:Q", format=".3f", title="Contribution")],
                )
                .properties(height=180)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Component data unavailable.")
        st.markdown(
            f"<div style='font-size:12px;color:rgba(0,0,0,0.6);line-height:1.4;'>"
            f"&#9888;&#xFE0E; {home['key_risk']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Score breakdown →", use_container_width=True, key="btn_why"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3  ·  Leadership vs SPY  (full width — most actionable)
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown("<div class='me-rowtitle'>Leadership vs SPY</div>", unsafe_allow_html=True)
    with hdr_r:
        if st.button("Rotation setups →", use_container_width=True, key="btn_rot"):
            safe_switch_page("pages/4_Rotation_Setups.py")

    rs = compute_rs_heatmap(period="1y")

    ctrl_l, ctrl_m, ctrl_r = st.columns([1.2, 1.2, 3.0], gap="medium")
    with ctrl_l:
        horizon = st.radio("Focus", ["1w", "1m", "3m"], horizontal=True, index=1, key="rs_focus")
    with ctrl_m:
        view = st.radio("View", ["Heatmap", "Bar"], horizontal=True, index=0, key="rs_view")
    with ctrl_r:
        leaders, laggards = leaders_laggards(rs, horizon)
        ltxt = ", ".join(leaders.index.tolist()[:3]) if len(leaders) else "n/a"
        gtxt = ", ".join(laggards.index.tolist()[:3]) if len(laggards) else "n/a"
        st.markdown(
            f"<div class='me-subtle' style='margin-top:6px;'>"
            f"Leaders: <strong style='color:#1f7a4f;'>{ltxt}</strong>"
            f" &nbsp;&middot;&nbsp; "
            f"Laggards: <strong style='color:#b42318;'>{gtxt}</strong></div>",
            unsafe_allow_html=True,
        )

    rs_long = (
        rs.reset_index()
        .melt(id_vars="index", var_name="Horizon", value_name="RS")
        .rename(columns={"index": "Ticker"})
        .dropna()
    )
    rs_h = rs_long[rs_long["Horizon"] == horizon]

    if view == "Heatmap":
        chart = (
            alt.Chart(rs_h)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Horizon:N", title=None),
                y=alt.Y("Ticker:N", sort=alt.SortField(field="RS", order="descending"), title=None),
                color=alt.Color("RS:Q", title="RS vs SPY",
                                scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=["Ticker", "Horizon",
                         alt.Tooltip("RS:Q", format="+.2%", title="RS vs SPY")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        bar_df = rs_h.copy()
        bar_df["RS"] = pd.to_numeric(bar_df["RS"], errors="coerce")
        bar_df = bar_df.dropna().sort_values("RS", ascending=False)
        chart = (
            alt.Chart(bar_df)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5,
                      cornerRadiusBottomLeft=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("Ticker:N", sort="-x", title=None),
                x=alt.X("RS:Q", title="RS vs SPY", axis=alt.Axis(format="+.1%")),
                color=alt.condition(
                    alt.datum.RS > 0,
                    alt.value("#1f7a4f"),
                    alt.value("#b42318"),
                ),
                tooltip=["Ticker",
                         alt.Tooltip("RS:Q", format="+.2%", title="RS vs SPY")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4  ·  Macro snapshot  |  Market returns
# ══════════════════════════════════════════════════════════════════════════════

c_msnap, c_mktsnap = st.columns([1.1, 1.0], gap="medium")

with c_msnap:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Macro snapshot this week</div>", unsafe_allow_html=True)
        macro_df = home["macro_snapshot"]
        if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
            st.dataframe(macro_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Macro snapshot unavailable.")
        st.markdown("")
        if st.button("Macro charts →", use_container_width=True, key="btn_macro_snap"):
            safe_switch_page("pages/2_Macro_Charts.py", tab="rates")

with c_mktsnap:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Market returns this week</div>", unsafe_allow_html=True)
        mkt_df = home["market_snapshot"]
        if isinstance(mkt_df, pd.DataFrame) and not mkt_df.empty:
            st.dataframe(mkt_df, use_container_width=True, hide_index=True)
            st.caption("7-day percent return.")
        else:
            st.caption("Market snapshot unavailable.")
        st.markdown("")
        if st.button("Ticker drilldown →", use_container_width=True, key="btn_ticker"):
            safe_switch_page("pages/3_Ticker_Detail.py")

st.markdown("<div style='height:56px;'></div>", unsafe_allow_html=True)