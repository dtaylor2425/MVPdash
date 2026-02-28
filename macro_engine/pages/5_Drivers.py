# pages/5_Drivers.py
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3
from src.compute import component_contribution
from src.ui import inject_css, sidebar_nav, safe_switch_page, SCORE_LEGEND_HTML

st.set_page_config(
    page_title="Drivers",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
sidebar_nav(active="Drivers")

# ── Constants ─────────────────────────────────────────────────────────────────

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]
BREADTH_TICKERS  = ["RSP"]

# ── Data ──────────────────────────────────────────────────────────────────────

def _nearest_before(series: pd.Series, dt: pd.Timestamp):
    s   = series.dropna()
    idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None


def delta_over_days(series: pd.Series, days: int):
    s = series.dropna()
    if s.empty:
        return None, None, None
    end    = pd.Timestamp(s.index.max())
    end_i  = _nearest_before(s, end)
    prev_i = _nearest_before(s, end - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None, None, None
    latest = float(s.loc[end_i])
    prev   = float(s.loc[prev_i])
    return latest, prev, float(latest - prev)


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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_current_regime():
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set")
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro").sort_index()
    tickers = list(dict.fromkeys(ROTATION_TICKERS + BREADTH_TICKERS))
    px = fetch_prices(tickers, period="5y")
    if px is None or px.empty:
        px = pd.DataFrame()
    else:
        px = px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px, lookback_trend=63, momentum_lookback_days=21)
    return regime, macro, px

# ── Logic ─────────────────────────────────────────────────────────────────────

def _component_contribution(c: dict) -> float:
    return component_contribution(c)


def _trend_phrase(name: str, trend_up) -> str:
    nm = (name or "").lower()
    if trend_up is None:
        return "steady"
    if "credit" in nm or "spread" in nm:
        return "widening" if int(trend_up) == 1 else "tightening"
    if "dollar" in nm or "usd" in nm:
        return "strengthening" if int(trend_up) == 1 else "weakening"
    if "real" in nm or "yield" in nm or "rates" in nm:
        return "rising" if int(trend_up) == 1 else "falling"
    return "rising" if int(trend_up) == 1 else "falling"


def _metric_target_for_component(name: str) -> str:
    n = (name or "").lower()
    if "credit" in n or "spread" in n or "oas" in n:
        return "Credit spreads"
    if "dollar" in n or "usd" in n:
        return "Broad dollar"
    if "real" in n:
        return "Real yields"
    if "curve" in n or "term" in n:
        return "Curve"
    if "liquidity" in n:
        return "Liquidity"
    if "inflation" in n or "breakeven" in n:
        return "Inflation expectations"
    if "vol" in n or "vix" in n:
        return "Volatility"
    return "Risk appetite"


def build_dynamic_drivers(regime_obj, macro: pd.DataFrame, px: pd.DataFrame):
    drivers = []
    comps = getattr(regime_obj, "components", None)
    rows  = []
    if isinstance(comps, dict) and comps:
        for k, c in comps.items():
            if not isinstance(c, dict):
                continue
            nm      = c.get("name", k)
            contrib = _component_contribution(c)
            rows.append({
                "name":      str(nm),
                "contrib":   float(contrib),
                "trend_up":  c.get("trend_up"),
                "z":         c.get("zscore"),
                "target":    _metric_target_for_component(str(nm)),
            })

    if rows:
        rows = sorted(rows, key=lambda r: abs(r["contrib"]), reverse=True)
        for r in rows[:6]:
            direction = "supportive" if r["contrib"] >= 0 else "drag"
            trend     = _trend_phrase(r["name"], r["trend_up"])
            ztxt      = ""
            if r["z"] is not None and not pd.isna(r["z"]):
                ztxt = f"z {float(r['z']):.2f}"
            desc = f"{trend} • {direction}" + (f" • {ztxt}" if ztxt else "")
            drivers.append((r["name"], desc, r["target"]))
    else:
        if isinstance(macro, pd.DataFrame) and not macro.empty and "hy_oas" in macro.columns:
            _, _, dlt = delta_over_days(macro["hy_oas"], 7)
            if dlt is not None:
                drivers.append(("Credit conditions", "tightening" if dlt < 0 else "widening", "Credit spreads"))
        if isinstance(px, pd.DataFrame) and not px.empty and "IWM" in px.columns:
            r_iwm = pct_return_over_days(px["IWM"], 21)
            if r_iwm is not None:
                drivers.append(("Breadth proxy", "small caps leading" if r_iwm > 0 else "small caps lagging", "Risk appetite"))

    return drivers or [("Drivers unavailable", "Regime engine returned no components", "Risk appetite")]


def build_regime_summary(regime_obj):
    label = getattr(regime_obj, "label",          "Unknown")
    score = getattr(regime_obj, "score",           None)
    conf  = getattr(regime_obj, "confidence",      None)
    mom   = getattr(regime_obj, "momentum_label",  None)

    comps      = getattr(regime_obj, "components", None)
    supportive = []
    adverse    = []
    if isinstance(comps, dict) and comps:
        rows = sorted(
            [(c.get("name", k), float(_component_contribution(c)))
             for k, c in comps.items() if isinstance(c, dict)],
            key=lambda x: x[1], reverse=True,
        )
        supportive = [n for n, v in rows[:2] if v > 0]
        adverse    = [n for n, v in rows[-2:] if v < 0]

    parts   = [f"Regime: {label}"]
    if score is not None: parts.append(f"Score: {int(score)}")
    if conf  is not None: parts.append(f"Confidence: {conf}")
    if mom   is not None: parts.append(f"Momentum: {str(mom).lower()}")
    summary = " • ".join(parts)

    extra = []
    if supportive: extra.append("Supportive: " + ", ".join(supportive))
    if adverse:    extra.append("Headwind: "   + ", ".join(adverse))
    return summary, (" | ".join(extra) if extra else "")

# ── Guard ─────────────────────────────────────────────────────────────────────

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set. Add it as an environment variable.")
    st.stop()

regime_obj, macro, px = load_current_regime()
summary_line, summary_extra = build_regime_summary(regime_obj)

score_val   = int(getattr(regime_obj, "score", 0))
score_color = "#1f7a4f" if score_val >= 60 else ("#b42318" if score_val < 40 else "#6b7280")

# ═══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ═══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([3, 1], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div class="me-topbar">
          <div class="me-title">Key drivers</div>
          <div class="me-subtle">Derived from the current regime engine output and latest macro inputs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<span class='me-pill'>{summary_line}</span>", unsafe_allow_html=True)
    st.markdown(SCORE_LEGEND_HTML, unsafe_allow_html=True)
    if summary_extra:
        st.markdown(f"<div class='me-subtle' style='margin-top:8px;'>{summary_extra}</div>", unsafe_allow_html=True)
with right:
    if st.button("Open regime deep dive", use_container_width=True):
        safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER CARDS
# ═══════════════════════════════════════════════════════════════════════════════

drivers = build_dynamic_drivers(regime_obj, macro, px)

col1, col2 = st.columns(2, gap="large")
for i, (title, desc, target) in enumerate(drivers):
    col = col1 if i % 2 == 0 else col2
    with col:
        with st.container(border=True):
            st.markdown(f"<div class='driver-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='driver-desc'>{desc}</div>",   unsafe_allow_html=True)
            st.markdown(f"<div class='driver-meta'>Related: {target}</div>", unsafe_allow_html=True)
            st.markdown("")
            if st.button("Open related chart", key=f"driver_{i}", use_container_width=True):
                st.session_state["selected_metric"] = target
                safe_switch_page("pages/2_Macro_Charts.py")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Regime summary</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:rgba(0,0,0,0.80);'>{summary_line}</div>",
        unsafe_allow_html=True,
    )
    if summary_extra:
        st.caption(summary_extra)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)