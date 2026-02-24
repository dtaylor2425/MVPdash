# pages/5_Drivers.py
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3

st.set_page_config(
    page_title="Drivers",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]
BREADTH_TICKERS = ["RSP"]


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container {
          max-width: 1200px;
          padding-top: 4.8rem;
          padding-bottom: 5rem;
        }

        .me-topbar {
          position: sticky;
          top: 1.5rem;
          z-index: 999;
          background: rgba(255,255,255,0.92);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 16px;
          padding: 14px 16px;
          margin-bottom: 16px;
        }

        .me-title {
          font-size: 26px;
          font-weight: 900;
          letter-spacing: -0.6px;
          margin: 0;
          line-height: 1.05;
        }

        .me-subtle {
          color: rgba(0,0,0,0.55);
          font-size: 12px;
          margin-top: 4px;
        }

        .me-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          font-size: 12px;
          font-weight: 800;
          color: rgba(0,0,0,0.72);
          background: rgba(0,0,0,0.02);
          margin-right: 8px;
          margin-top: 6px;
        }

        .me-rowtitle {
          font-size: 13px;
          font-weight: 800;
          color: rgba(0,0,0,0.70);
          margin-bottom: 8px;
        }

        .driver-title {
          font-size: 14px;
          font-weight: 900;
          color: rgba(0,0,0,0.85);
          margin: 0;
        }

        .driver-desc {
          font-size: 13px;
          color: rgba(0,0,0,0.65);
          margin-top: 6px;
          line-height: 1.35;
        }

        .driver-meta {
          font-size: 12px;
          color: rgba(0,0,0,0.55);
          margin-top: 10px;
        }

        .stButton > button {
          border-radius: 12px !important;
          border: 1px solid rgba(0,0,0,0.10) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def safe_switch_page(path: str):
    try:
        st.switch_page(path)
    except Exception:
        st.error(f"Missing page: {path}. Check your pages folder.")


def sidebar_nav(active: str = "Drivers"):
    st.sidebar.title("Macro Engine")
    st.sidebar.caption("Navigation")

    pages = {
        "Home": "app.py",
        "Macro charts": "pages/2_Macro_Charts.py",
        "Regime deep dive": "pages/1_Regime_Deep_Dive.py",
        "Drivers": "pages/5_Drivers.py",
        "Ticker drilldown": "pages/3_Ticker_Detail.py",
    }

    keys = list(pages.keys())
    idx = keys.index(active) if active in pages else 0
    choice = st.sidebar.selectbox("Go to", keys, index=idx, label_visibility="collapsed")
    if choice != active:
        safe_switch_page(pages[choice])


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


def pct_return_over_days(series: pd.Series, days: int):
    s = series.dropna()
    if len(s) < days + 2:
        return None
    end = s.index.max()
    prev_dt = end - pd.Timedelta(days=days)
    end_i = _nearest_before(s, end)
    prev_i = _nearest_before(s, prev_dt)
    if end_i is None or prev_i is None:
        return None
    a = float(s.loc[end_i])
    b = float(s.loc[prev_i])
    if b == 0:
        return None
    return (a / b) - 1.0


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


def _component_contribution(c: dict) -> float:
    if not isinstance(c, dict):
        return 0.0
    if isinstance(c.get("contribution"), (int, float, np.floating)):
        return float(c["contribution"])
    z = c.get("zscore")
    w = c.get("weight")
    if isinstance(z, (int, float, np.floating)) and isinstance(w, (int, float, np.floating)):
        return float(z) * float(w)
    return 0.0


def _trend_phrase(name: str, trend_up):
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


def _metric_target_for_component(name: str):
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
    rows = []
    if isinstance(comps, dict) and comps:
        for k, c in comps.items():
            if not isinstance(c, dict):
                continue
            nm = c.get("name", k)
            contrib = _component_contribution(c)
            rows.append(
                {
                    "name": str(nm),
                    "contrib": float(contrib),
                    "trend_up": c.get("trend_up"),
                    "z": c.get("zscore"),
                    "target": _metric_target_for_component(str(nm)),
                }
            )

    if rows:
        rows = sorted(rows, key=lambda r: abs(r["contrib"]), reverse=True)
        for r in rows[:6]:
            direction = "supportive" if r["contrib"] >= 0 else "drag"
            trend = _trend_phrase(r["name"], r["trend_up"])
            ztxt = ""
            if r["z"] is not None and not pd.isna(r["z"]):
                ztxt = f"z {float(r['z']):.2f}"
            desc = f"{trend} • {direction}" + (f" • {ztxt}" if ztxt else "")
            drivers.append((r["name"], desc, r["target"]))
    else:
        if isinstance(macro, pd.DataFrame) and not macro.empty:
            if "hy_oas" in macro.columns:
                latest, prev, dlt = delta_over_days(macro["hy_oas"], 7)
                if dlt is not None:
                    drivers.append(("Credit conditions", "tightening" if dlt < 0 else "widening", "Credit spreads"))
        if isinstance(px, pd.DataFrame) and not px.empty and "IWM" in px.columns:
            r_iwm = pct_return_over_days(px["IWM"], 21)
            if r_iwm is not None:
                drivers.append(("Breadth proxy", "small caps leading" if r_iwm > 0 else "small caps lagging", "Risk appetite"))

    if not drivers:
        drivers = [("Drivers unavailable", "Regime engine returned no components", "Risk appetite")]

    return drivers


def build_regime_summary(regime_obj):
    label = getattr(regime_obj, "label", "Unknown")
    score = getattr(regime_obj, "score", None)
    conf = getattr(regime_obj, "confidence", None)
    mom = getattr(regime_obj, "momentum_label", None)

    comps = getattr(regime_obj, "components", None)
    supportive = []
    adverse = []
    if isinstance(comps, dict) and comps:
        rows = []
        for k, c in comps.items():
            if not isinstance(c, dict):
                continue
            nm = c.get("name", k)
            contrib = _component_contribution(c)
            rows.append((str(nm), float(contrib)))
        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        supportive = [n for n, v in rows[:2] if v > 0]
        adverse = [n for n, v in rows[-2:] if v < 0]

    parts = [f"Regime: {label}"]
    if score is not None:
        parts.append(f"Score: {int(score)}")
    if conf is not None:
        parts.append(f"Confidence: {conf}")
    if mom is not None:
        parts.append(f"Momentum: {str(mom).lower()}")

    summary = " • ".join(parts)

    extra = []
    if supportive:
        extra.append("Supportive: " + ", ".join(supportive))
    if adverse:
        extra.append("Headwind: " + ", ".join(adverse))

    return summary, (" | ".join(extra) if extra else "")


sidebar_nav(active="Drivers")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set. Add it as an environment variable locally and in your host.")
    st.stop()

regime_obj, macro, px = load_current_regime()
summary_line, summary_extra = build_regime_summary(regime_obj)

with st.container():
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div class="me-topbar">
              <div>
                <div class="me-title">Key drivers</div>
                <div class="me-subtle">Derived from the current regime engine output and latest macro inputs</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<span class='me-pill'>{summary_line}</span>", unsafe_allow_html=True)
        if summary_extra:
            st.markdown(f"<div class='me-subtle' style='margin-top:8px;'>{summary_extra}</div>", unsafe_allow_html=True)
    with right:
        if st.button("Open regime deep dive", use_container_width=True):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

drivers = build_dynamic_drivers(regime_obj, macro, px)

col1, col2 = st.columns(2, gap="large")
for i, (title, desc, target) in enumerate(drivers):
    col = col1 if i % 2 == 0 else col2
    with col:
        with st.container(border=True):
            st.markdown(f"<div class='driver-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='driver-desc'>{desc}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='driver-meta'>Related: {target}</div>", unsafe_allow_html=True)
            st.markdown("")
            if st.button("Open related chart", key=f"driver_{i}", use_container_width=True):
                st.session_state["selected_metric"] = target
                safe_switch_page("pages/2_Macro_Charts.py")

st.markdown("")

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Regime summary</div>", unsafe_allow_html=True)
    st.write(summary_line)
    if summary_extra:
        st.caption(summary_extra)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)