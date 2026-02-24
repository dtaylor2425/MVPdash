import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3

st.set_page_config(
    page_title="Macro Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        html, body {
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }

        [data-testid="stAppViewContainer"] { background: #ffffff !important; }
        [data-testid="stAppViewContainer"] > .main { background: #ffffff !important; }

        .block-container {
          max-width: 1200px;
          padding-top: 4.8rem;
          padding-bottom: 5rem;
        }

        .stButton > button {
          border-radius: 12px !important;
          padding: 0.55rem 0.9rem !important;
          border: 1px solid rgba(0,0,0,0.10) !important;
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }

        input, textarea {
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }

        .stDataFrame, .stTable, [data-testid="stTable"] {
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }

        .me-topbar {
          position: sticky;
          top: 1.5rem;
          z-index: 999;
          background: rgba(255,255,255,0.92);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 20px;
          padding: 18px 20px;
          margin-bottom: 20px;
        }

        .me-title {
          font-size: 30px;
          font-weight: 900;
          letter-spacing: -0.6px;
          margin: 0;
          line-height: 1.05;
          color: rgba(0,0,0,0.90);
        }

        .me-subtle {
          color: rgba(0,0,0,0.55);
          font-size: 12px;
          margin-top: 4px;
        }

        .me-chip {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          font-size: 20px;
          font-weight: 900;
          white-space: nowrap;
          background: rgba(0,0,0,0.01);
          color: rgba(0,0,0,0.85);
        }

        .me-dot {
          width: 10px;
          height: 10px;
          border-radius: 999px;
          display: inline-block;
        }

        .me-kpi {
          font-size: 38px;
          font-weight: 900;
          letter-spacing: -1px;
          line-height: 1.0;
          margin-top: 2px;
          color: rgba(0,0,0,0.90);
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
        }

        .me-rowtitle {
          font-size: 13px;
          font-weight: 800;
          color: rgba(0,0,0,0.70);
          margin-bottom: 8px;
        }

        .me-li {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          border-radius: 12px;
          border: 1px solid rgba(0,0,0,0.06);
          background: #fff;
          margin-bottom: 8px;
        }

        .me-li:last-child { margin-bottom: 0; }

        .me-li-name {
          font-size: 13px;
          font-weight: 900;
          color: rgba(0,0,0,0.78);
          margin: 0;
        }

        .me-li-sub {
          font-size: 12px;
          color: rgba(0,0,0,0.55);
          margin: 0;
          margin-top: 2px;
        }

        .me-li-right {
          font-size: 12px;
          font-weight: 900;
          white-space: nowrap;
          color: rgba(0,0,0,0.72);
        }

        .me-nav-title {
          font-weight: 900;
          font-size: 13px;
          margin: 0;
          line-height: 1.15;
          color: rgba(0,0,0,0.85);
        }
        .me-nav-desc {
          color: rgba(0,0,0,0.55);
          font-size: 12px;
          margin-top: 4px;
          line-height: 1.25;
        }

        @media (max-width: 700px) {

          .block-container {
            padding-top: 4.0rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
          }

          .me-topbar {
            top: 0.75rem !important;
            padding: 12px 12px !important;
            border-radius: 16px !important;
            margin-bottom: 14px !important;
          }

          .me-title {
            font-size: 22px !important;
            letter-spacing: -0.3px !important;
          }

          .me-chip {
            font-size: 14px !important;
            padding: 8px 10px !important;
            gap: 8px !important;
          }

          .me-kpi { font-size: 32px !important; }

          div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 10px !important;
          }
          div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
          }

          div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 280px;
          }

          .stButton > button {
            width: 100% !important;
            padding: 0.65rem 0.9rem !important;
          }

          .stTabs [data-baseweb="tab"] { font-size: 14px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def regime_color(regime: str) -> str:
    r = (regime or "").lower()
    if r == "risk on":
        return "#1f7a4f"
    if r == "risk off":
        return "#b42318"
    return "#6b7280"


def safe_switch_page(path: str):
    try:
        st.switch_page(path)
    except Exception:
        st.error(f"Missing page: {path}. Make sure the file exists in your pages folder.")


def sidebar_nav(active: str = "Home"):
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


ROTATION_TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLV", "GLD", "UUP", "IWM", "QQQ", "SPY"]
MARKET_SNAPSHOT = ["SPY", "QQQ", "IWM", "HYG", "TLT", "UUP", "GLD", "XLE"]


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
    px = fetch_prices(list(dict.fromkeys(ROTATION_TICKERS + MARKET_SNAPSHOT)), period="5y")
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


def build_dynamic_drivers(regime_obj, macro: pd.DataFrame, px: pd.DataFrame):
    drivers = []

    comps = getattr(regime_obj, "components", None)
    if isinstance(comps, dict) and comps:
        rows = []
        for k, c in comps.items():
            if not isinstance(c, dict):
                continue
            nm = c.get("name", k)
            contrib = _component_contribution(c)
            rows.append((nm, contrib, c.get("trend_up"), c.get("zscore")))
        rows = sorted(rows, key=lambda x: abs(x[1]), reverse=True)

        for nm, contrib, trend_up, z in rows[:3]:
            direction = "supportive" if contrib >= 0 else "drag"
            trend = _trend_phrase(nm, trend_up)
            ztxt = "" if z is None or pd.isna(z) else f"z {float(z):.2f}"
            subtitle = f"{trend} • {direction}" + (f" • {ztxt}" if ztxt else "")
            drivers.append((str(nm), subtitle))

    if len(drivers) < 3 and isinstance(macro, pd.DataFrame) and not macro.empty:
        if "hy_oas" in macro.columns:
            latest, prev, dlt = delta_over_days(macro["hy_oas"], 7)
            if dlt is not None:
                drivers.append(("Credit conditions", "tightening" if dlt < 0 else "widening"))

        if isinstance(px, pd.DataFrame) and not px.empty and "IWM" in px.columns and "RSP" in px.columns:
            r_iwm = pct_return_over_days(px["IWM"], 21)
            r_rsp = pct_return_over_days(px["RSP"], 21) if "RSP" in px.columns else None
            if r_iwm is not None:
                msg = "small caps leading" if r_iwm > 0 else "small caps lagging"
                if r_rsp is not None:
                    msg = msg + f" • equal weight {('up' if r_rsp > 0 else 'down')}"
                drivers.append(("Breadth proxy", msg))

    drivers = drivers[:3]
    if not drivers:
        drivers = [("Drivers unavailable", "Regime engine returned no components")]

    return drivers


def build_allocation_tilt(regime_obj):
    alloc = getattr(regime_obj, "allocation", None)
    stance = {}
    if isinstance(alloc, dict):
        stance = alloc.get("stance", {}) if isinstance(alloc.get("stance", {}), dict) else {}

    keys = ["Equities", "Credit", "Duration", "USD", "Commodities"]
    out = []
    for k in keys:
        v = stance.get(k)
        if v is None:
            continue
        out.append((k, str(v)))
    return out


def build_key_risk(regime_obj):
    comps = getattr(regime_obj, "components", None)
    if isinstance(comps, dict) and comps:
        rows = []
        for k, c in comps.items():
            if not isinstance(c, dict):
                continue
            nm = c.get("name", k)
            contrib = _component_contribution(c)
            z = c.get("zscore")
            rows.append((nm, contrib, z))
        rows = sorted(rows, key=lambda x: x[1])
        nm, contrib, z = rows[0]
        ztxt = "" if z is None or pd.isna(z) else f"z {float(z):.2f}"
        if contrib < 0:
            return f"Primary risk is {nm} staying adverse" + (f" ({ztxt})" if ztxt else "")
        return f"Primary risk is {nm} reversing against the signal" + (f" ({ztxt})" if ztxt else "")
    return "Primary risk is regime momentum reversing on a growth or credit shock."


def build_weekly_macro_changes(macro: pd.DataFrame):
    items = []

    if not isinstance(macro, pd.DataFrame) or macro.empty:
        return items

    if "hy_oas" in macro.columns:
        latest, prev, dlt = delta_over_days(macro["hy_oas"], 7)
        if dlt is not None:
            items.append(("Credit spreads (HY OAS)", prev, latest, dlt, "tighter" if dlt < 0 else "wider"))

    if "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        latest, prev, dlt = delta_over_days(curve, 7)
        if dlt is not None:
            items.append(("Curve (10y minus 2y)", prev, latest, dlt, "steeper" if dlt > 0 else "flatter"))

    if "real10" in macro.columns:
        latest, prev, dlt = delta_over_days(macro["real10"], 7)
        if dlt is not None:
            items.append(("Real 10y", prev, latest, dlt, "higher" if dlt > 0 else "lower"))

    if "dollar_broad" in macro.columns:
        latest, prev, dlt = delta_over_days(macro["dollar_broad"], 7)
        if dlt is not None:
            items.append(("Dollar broad", prev, latest, dlt, "stronger" if dlt > 0 else "weaker"))

    return items


def build_weekly_snapshot_tables(macro: pd.DataFrame, px: pd.DataFrame):
    macro_rows = []
    mkt_rows = []

    for name, prev, cur, dlt, _lab in build_weekly_macro_changes(macro):
        macro_rows.append(
            {
                "Metric": name,
                "Level": cur,
                "Weekly change": dlt,
            }
        )

    if isinstance(px, pd.DataFrame) and not px.empty:
        for t in MARKET_SNAPSHOT:
            if t not in px.columns:
                continue
            r = pct_return_over_days(px[t], 7)
            if r is None:
                continue
            mkt_rows.append({"Asset": t, "Weekly change": 100.0 * float(r)})

    macro_df = pd.DataFrame(macro_rows)
    mkt_df = pd.DataFrame(mkt_rows).sort_values("Weekly change", ascending=False) if mkt_rows else pd.DataFrame()

    return macro_df, mkt_df


def load_home_state():
    regime_obj, macro, px = load_current_regime()

    if isinstance(macro, pd.DataFrame) and not macro.empty:
        try:
            last_updated = macro.index.max().date()
        except Exception:
            last_updated = date.today()
    else:
        last_updated = date.today()

    regime = getattr(regime_obj, "label", "Unknown")
    score = int(getattr(regime_obj, "score", 0))
    confidence = getattr(regime_obj, "confidence", "Unknown")
    momentum = getattr(regime_obj, "momentum_label", "Unknown")

    allocation_tilt = build_allocation_tilt(regime_obj)
    weekly_changes = build_weekly_macro_changes(macro)
    drivers = build_dynamic_drivers(regime_obj, macro, px)
    key_risk = build_key_risk(regime_obj)

    macro_df, mkt_df = build_weekly_snapshot_tables(macro, px)

    return {
        "last_updated": last_updated,
        "regime": regime,
        "score": score,
        "confidence": confidence,
        "momentum": momentum,
        "allocation_tilt": allocation_tilt,
        "weekly_changes": weekly_changes,
        "drivers": drivers,
        "key_risk": key_risk,
        "macro_snapshot": macro_df,
        "market_snapshot": mkt_df,
        "macro": macro,
        "px": px,
        "regime_obj": regime_obj,
    }


@st.cache_data(show_spinner=False)
def compute_rs_heatmap(period: str = "1y") -> pd.DataFrame:
    tickers = [
        "GLD", "XLE", "XLP", "XLI", "IWM", "RSP",
        "HYG", "TLT", "UUP", "QQQ", "XLK", "XLF", "XLV"
    ]
    base = "SPY"
    fetch_list = list(dict.fromkeys([base] + tickers))

    px = fetch_prices(fetch_list, period=period)
    if px is None or px.empty:
        return pd.DataFrame(index=tickers, columns=["1w", "1m", "3m"], data=np.nan)

    px = px.dropna(how="all")
    if base not in px.columns:
        return pd.DataFrame(index=tickers, columns=["1w", "1m", "3m"], data=np.nan)

    horizons = {"1w": 5, "1m": 21, "3m": 63}
    out = pd.DataFrame(index=tickers, columns=list(horizons.keys()), dtype=float)

    b = px[base].dropna()
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        idx = s.index.intersection(b.index)
        if len(idx) < 90:
            continue
        s = s.loc[idx]
        bb = b.loc[idx]
        for lab, n in horizons.items():
            if len(idx) <= n:
                out.loc[t, lab] = np.nan
            else:
                ret_t = (s.iloc[-1] / s.iloc[-1 - n]) - 1.0
                ret_b = (bb.iloc[-1] / bb.iloc[-1 - n]) - 1.0
                out.loc[t, lab] = float(ret_t - ret_b)

    if "SPY" in out.index:
        out.loc["SPY", :] = 0.0

    return out


def leaders_laggards(rs_df: pd.DataFrame, horizon: str):
    s = rs_df[horizon].dropna().sort_values(ascending=False)
    return s.head(5), s.tail(5)


if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set. Add it as an environment variable locally and in your host.")
    st.stop()

sidebar_nav(active="Home")
home = load_home_state()

chip = home["regime"]
dot = regime_color(chip)

with st.container():
    c1, c2 = st.columns([2.2, 1.0], vertical_alignment="center")
    with c1:
        st.markdown(
            f"""
            <div class="me-topbar">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div>
                  <div class="me-title">Macro Engine</div>
                  <div class="me-subtle">Market regime and rotation signals</div>
                </div>
                <div class="me-chip">
                  <span class="me-dot" style="background:{dot}"></span>
                  <span>{chip}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='me-subtle' style='text-align:right;'>Last updated: {home['last_updated']}</div>",
            unsafe_allow_html=True,
        )

nav_defs = [
    ("Home", "Overview and summary", None),
    ("Macro charts", "Rates, credit, liquidity, risk", "pages/2_Macro_Charts.py"),
    ("Regime deep dive", "Notes, deltas, components", "pages/1_Regime_Deep_Dive.py"),
    ("Drivers", "Narrative and chart links", "pages/5_Drivers.py"),
    ("Ticker drilldown", "RS, MAs, setups", "pages/3_Ticker_Detail.py"),
]

nav_cols = st.columns(5, gap="small")
for i, (title, desc, path) in enumerate(nav_defs):
    with nav_cols[i]:
        with st.container(border=True):
            st.markdown(f"<div class='me-nav-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='me-nav-desc'>{desc}</div>", unsafe_allow_html=True)
            st.markdown("")
            if title == "Home":
                st.button("Open", use_container_width=True, disabled=True, key=f"topnav_{i}")
            else:
                if st.button("Open", use_container_width=True, key=f"topnav_{i}"):
                    safe_switch_page(path)

st.markdown("")

h1, h2, h3, h4 = st.columns([1.0, 1.15, 1.2, 1.0], gap="large")

with h1:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Macro score</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='me-kpi'>{home['score']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
              <span class="me-pill">Confidence: {home['confidence']}</span>
              <span class="me-pill">Momentum: {str(home['momentum']).lower()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Open score breakdown", use_container_width=True, key="btn_score"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

with h2:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key drivers</div>", unsafe_allow_html=True)
        for i, (title, sub) in enumerate(home["drivers"], start=1):
            st.markdown(
                f"""
                <div class="me-li">
                  <div>
                    <div class="me-li-name">{i}. {title}</div>
                    <div class="me-li-sub">{sub}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")
        if st.button("Open drivers", use_container_width=True, key="btn_drivers"):
            safe_switch_page("pages/5_Drivers.py")

with h3:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>What changed this week</div>", unsafe_allow_html=True)

        if home["weekly_changes"]:
            for name, prev, cur, delta, label in home["weekly_changes"]:
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                st.markdown(
                    f"""
                    <div class="me-li">
                      <div>
                        <div class="me-li-name">{name}</div>
                        <div class="me-li-sub">{prev:.2f} to {cur:.2f} ({label})</div>
                      </div>
                      <div class="me-li-right">{arrow} {abs(delta):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Weekly changes unavailable for the current macro dataset.")

        st.markdown("")
        if st.button("Open weekly details", use_container_width=True, key="btn_weekly"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

with h4:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Key risk</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:13px; color:rgba(0,0,0,0.72); line-height:1.35;'>{home['key_risk']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Open macro charts", use_container_width=True, key="btn_macro"):
            safe_switch_page("pages/2_Macro_Charts.py")

st.markdown("")

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Leadership vs SPY</div>", unsafe_allow_html=True)

    rs = compute_rs_heatmap(period="1y")

    top_controls = st.columns([1.0, 1.0, 2.0, 1.2], gap="medium")
    with top_controls[0]:
        horizon = st.radio("Focus", ["1w", "1m", "3m"], horizontal=True, index=1, key="rs_focus")
    with top_controls[1]:
        view = st.radio("View", ["Heatmap", "Leaders and laggards"], horizontal=True, index=0, key="rs_view")
    with top_controls[2]:
        leaders, laggards = leaders_laggards(rs, horizon)
        ltxt = ", ".join(leaders.index.tolist()[:2]) if len(leaders) else "n a"
        gtxt = ", ".join(laggards.index.tolist()[:2]) if len(laggards) else "n a"
        st.markdown(
            f"<div class='me-subtle'>Takeaway</div><div style='font-weight:900; color:rgba(0,0,0,0.85);'>Leadership: {ltxt} | Laggards: {gtxt}</div>",
            unsafe_allow_html=True,
        )
    with top_controls[3]:
        if st.button("Open macro charts", use_container_width=True, key="btn_flows_to_macro"):
            safe_switch_page("pages/2_Macro_Charts.py")

    rs_long = rs.reset_index().melt(id_vars="index", var_name="Horizon", value_name="RS").rename(columns={"index": "Ticker"})
    rs_long = rs_long.dropna()
    rs_long = rs_long[rs_long["Horizon"] == horizon]

    if view == "Heatmap":
        heat = (
            alt.Chart(rs_long)
            .mark_rect(cornerRadius=6)
            .encode(
                x=alt.X("Horizon:N", title=None),
                y=alt.Y("Ticker:N", sort=alt.SortField(field="RS", order="descending"), title=None),
                color=alt.Color("RS:Q", title="RS"),
                tooltip=["Ticker", "Horizon", alt.Tooltip("RS:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        leaders, laggards = leaders_laggards(rs, horizon)
        cL, cR = st.columns([1, 1], gap="large")
        with cL:
            st.markdown("**Leaders**")
            for t, v in leaders.items():
                if st.button(f"{t} • {v:.3f}", use_container_width=True, key=f"lead_{horizon}_{t}"):
                    st.session_state["selected_ticker"] = t
                    safe_switch_page("pages/3_Ticker_Detail.py")
        with cR:
            st.markdown("**Laggards**")
            for t, v in laggards.items():
                if st.button(f"{t} • {v:.3f}", use_container_width=True, key=f"lag_{horizon}_{t}"):
                    st.session_state["selected_ticker"] = t
                    safe_switch_page("pages/3_Ticker_Detail.py")

st.markdown("")

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Weekly snapshot</div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Macro", "Markets"])

    macro_df = home["macro_snapshot"]
    mkt_df = home["market_snapshot"]

    with tab1:
        if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
            show = macro_df.copy()
            for c in ["Level", "Weekly change"]:
                if c in show.columns:
                    show[c] = pd.to_numeric(show[c], errors="coerce")
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.caption("Macro snapshot unavailable for the current dataset.")

    with tab2:
        if isinstance(mkt_df, pd.DataFrame) and not mkt_df.empty:
            show = mkt_df.copy()
            if "Weekly change" in show.columns:
                show["Weekly change"] = pd.to_numeric(show["Weekly change"], errors="coerce")
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("Market weekly change is percent return over the last week window.")
        else:
            st.caption("Market snapshot unavailable for the current price dataset.")

    st.markdown("")
    if st.button("Open regime deep dive", use_container_width=True, key="btn_weekly_bottom"):
        safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

b1, b2 = st.columns([1.0, 1.2], gap="large")

with b1:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Regime and tilt</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:18px; font-weight:900; margin-bottom:10px; color:rgba(0,0,0,0.85);'>{home['regime']}</div>",
            unsafe_allow_html=True,
        )

        if home["allocation_tilt"]:
            for k, v in home["allocation_tilt"]:
                st.markdown(
                    f"""
                    <div class="me-li">
                      <div>
                        <div class="me-li-name">{k}</div>
                        <div class="me-li-sub">Suggested stance</div>
                      </div>
                      <div class="me-li-right">{v}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Allocation tilt unavailable from the current regime output.")

with b2:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Why this regime</div>", unsafe_allow_html=True)

        comps = getattr(home["regime_obj"], "components", None)
        if isinstance(comps, dict) and comps:
            rows = []
            for k, c in comps.items():
                if not isinstance(c, dict):
                    continue
                nm = c.get("name", k)
                rows.append({"Component": str(nm), "Contribution": float(_component_contribution(c))})
            comp = pd.DataFrame(rows).sort_values("Contribution", ascending=False)

            ch = (
                alt.Chart(comp)
                .mark_bar(
                    cornerRadiusTopLeft=6,
                    cornerRadiusTopRight=6,
                    cornerRadiusBottomLeft=6,
                    cornerRadiusBottomRight=6,
                )
                .encode(
                    y=alt.Y("Component:N", sort="-x", title=None),
                    x=alt.X("Contribution:Q", title=None),
                    tooltip=["Component", alt.Tooltip("Contribution:Q", format=".2f")],
                )
                .properties(height=240)
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            st.caption("Component contributions unavailable from the current regime output.")

        st.markdown("")
        if st.button("Open component detail", use_container_width=True, key="btn_components"):
            safe_switch_page("pages/1_Regime_Deep_Dive.py")

st.markdown("")

tiles = st.columns(5, gap="small")
tile_defs = [
    ("Macro charts", "Rates, credit, liquidity, risk", "pages/2_Macro_Charts.py"),
    ("Ticker drilldown", "RS, MAs, setup features", "pages/3_Ticker_Detail.py"),
    ("Regime deep dive", "Notes, deltas, components", "pages/1_Regime_Deep_Dive.py"),
    ("Drivers", "Narrative and chart links", "pages/5_Drivers.py"),
    ("Score", "Breakdown and history", "pages/1_Regime_Deep_Dive.py"),
]

for i, (t, d, path) in enumerate(tile_defs):
    with tiles[i]:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-weight:900; font-size:13px; color:rgba(0,0,0,0.85);'>{t}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='me-subtle'>{d}</div>", unsafe_allow_html=True)
            st.markdown("")
            if st.button("Open", use_container_width=True, key=f"tile_open_{i}"):
                safe_switch_page(path)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)