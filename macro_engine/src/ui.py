# src/ui.py
# Shared UI helpers — import from every page to eliminate duplication.
import streamlit as st


# ── Colour helpers ────────────────────────────────────────────────────────────

def regime_color(regime: str) -> str:
    """Dot/text colour for each of the 5 v4 regime labels."""
    r = (regime or "").lower()
    if "risk on" in r:
        return "#1f7a4f"
    if "bullish" in r:
        return "#16a34a"
    if "bearish" in r:
        return "#d97706"
    if "risk off" in r:
        return "#b42318"
    return "#6b7280"   # Neutral


def regime_bg(regime: str) -> str:
    """Light background pill colour to pair with regime_color."""
    r = (regime or "").lower()
    if "risk on" in r:
        return "#dcfce7"
    if "bullish" in r:
        return "#dcfce7"
    if "bearish" in r:
        return "#fef9c3"
    if "risk off" in r:
        return "#fee2e2"
    return "#f3f4f6"


def delta_color(value: float, inverse: bool = False) -> str:
    """Green/red badge colour.  inverse=True for series where up = bad (e.g. credit spreads)."""
    if value is None:
        return "#6b7280"
    good = value > 0
    if inverse:
        good = not good
    return "#1f7a4f" if good else "#b42318"


# ── Navigation ────────────────────────────────────────────────────────────────

_PAGES = {
    # ── Command centre ────────────────────────────────
    "Home":              "app.py",
    "Morning Brief":     "pages/0_Morning_Brief.py",
    # ── Analysis ──────────────────────────────────────
    "Regime Playbook":   "pages/7_Regime_Playbook.py",
    "Transition Watch":  "pages/8_Transition_Watch.py",
    # ── Market views ──────────────────────────────────
    "Fed & Liquidity":   "pages/10_Fed_Liquidity.py",
    "Asset Monitor":     "pages/3_Asset_Monitor.py",
    "Curve View":        "pages/9_Curve_View.py",
    "Volatility View":   "pages/6_Volatility_View.py",
    "Credit & Macro":    "pages/2_Macro_Charts.py",
    "Rotation & setups": "pages/4_Rotation_Setups.py",
    # ── Reference ─────────────────────────────────────
    "Regime Engine":     "pages/1_Regime_Deep_Dive.py",
}


def safe_switch_page(path: str, tab: str | None = None):
    """Navigate to a page. If tab is supplied it is written to st.query_params
    before switching so the target page can open directly to the right tab."""
    try:
        if tab is not None:
            st.query_params["tab"] = tab
        else:
            st.query_params.pop("tab", None)
        st.switch_page(path)
    except Exception:
        st.error(f"Missing page: {path}. Check your pages folder.")


# Navigation metadata — icons, section headers, descriptions
_NAV_META = {
    # page_name: (icon, section, short description)
    "Home":              ("🏠", "core",      "Command centre"),
    "Morning Brief":     ("☀️", "core",      "Daily synthesis"),
    "Regime Playbook":   ("📋", "analysis",  "Actionable signals"),
    "Transition Watch":  ("🔔", "analysis",  "Regime change alerts"),
    "Fed & Liquidity":   ("🏦", "markets",   "Policy backbone"),
    "Asset Monitor":     ("📡", "markets",   "SPY QQQ GLD BTC SLV"),
    "Curve View":        ("📐", "markets",   "Yield curve"),
    "Volatility View":   ("⚡", "markets",   "VIX & stress"),
    "Credit & Macro":    ("📊", "markets",   "Credit & spreads"),
    "Rotation & setups": ("🔄", "markets",   "Pair signals"),
    "Regime Engine":     ("🔍", "reference", "Score breakdown"),
}

_SECTION_LABELS = {
    "core":      "",
    "analysis":  "ANALYSIS",
    "markets":   "MARKET VIEWS",
    "reference": "REFERENCE",
}

def sidebar_nav(active: str = "Home"):
    # Title block
    st.sidebar.markdown(
        """<div style="padding:16px 12px 10px;border-bottom:1px solid rgba(0,0,0,0.07);
            margin-bottom:8px;">
          <div style="font-size:18px;font-weight:900;letter-spacing:-0.4px;
                      color:rgba(0,0,0,0.88);">Macro Engine</div>
          <div style="font-size:10px;color:rgba(0,0,0,0.38);margin-top:2px;
                      text-transform:uppercase;letter-spacing:0.6px;">
            Signal · Analysis · Alpha
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    last_section = None
    for name, path in _PAGES.items():
        meta       = _NAV_META.get(name, ("·", "core", ""))
        icon, section, desc = meta
        is_active  = name == active

        # Section header
        if section != last_section and _SECTION_LABELS.get(section):
            st.sidebar.markdown(
                f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.32);"
                f"letter-spacing:0.8px;text-transform:uppercase;"
                f"padding:10px 12px 4px;'>{_SECTION_LABELS[section]}</div>",
                unsafe_allow_html=True,
            )
        last_section = section

        # Active vs inactive styling via markdown injection + button
        if is_active:
            st.sidebar.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"padding:8px 12px;margin:1px 0;border-radius:10px;"
                f"background:rgba(0,0,0,0.07);cursor:default;'>"
                f"<span style='font-size:14px;'>{icon}</span>"
                f"<div>"
                f"  <div style='font-size:12px;font-weight:800;color:rgba(0,0,0,0.88);'>"
                f"  {name}</div>"
                f"  <div style='font-size:9px;color:rgba(0,0,0,0.42);'>{desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            # Use a button but style it to look like a nav item
            clicked = st.sidebar.button(
                f"{icon}  {name}",
                key=f"sidenav_{name}",
                width='stretch',
            )
            if clicked:
                safe_switch_page(path)

    st.sidebar.markdown(
        "<div style='border-top:1px solid rgba(0,0,0,0.06);margin-top:12px;"
        "padding:10px 12px;'>"
        "<div style='font-size:9px;color:rgba(0,0,0,0.28);text-transform:uppercase;"
        "letter-spacing:0.6px;'>Macro Engine v4</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Score legend ──────────────────────────────────────────────────────────────

SCORE_LEGEND_HTML = (
    "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;'>"
    "<span style='font-size:10px;color:#b42318;font-weight:700;'>▼&lt;25 Risk Off</span>"
    "<span style='font-size:10px;color:#d97706;font-weight:700;'>▼ 25–40 Bearish</span>"
    "<span style='font-size:10px;color:#6b7280;font-weight:700;'>● 40–60 Neutral</span>"
    "<span style='font-size:10px;color:#16a34a;font-weight:700;'>▲ 60–75 Bullish</span>"
    "<span style='font-size:10px;color:#1f7a4f;font-weight:700;'>▲&gt;75 Risk On</span>"
    "</div>"
)


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        /* ── Hide Streamlit auto pages nav (we use our own) ── */
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavItems"],
        [data-testid="stSidebarNavSeparator"] {
          display: none !important;
        }

        /* ── Base ── */
        html, body, [class*="css"] {
          font-family: 'Inter', sans-serif;
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
          background: #ffffff !important;
        }

        /* ── Widget labels — surgical fix for mobile dark mode ── */
        /* These are the specific elements that go invisible on mobile;
           we do NOT touch dataframe cells or plotly which manage their own colours */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label span,
        [data-testid="stRadio"] > div > label,
        [data-baseweb="radio"] ~ div label,
        .stRadio label { color: rgba(0,0,0,0.85) !important; }

        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] > label { color: rgba(0,0,0,0.85) !important; }

        [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
        [data-baseweb="select"] span:not([data-testid]),
        [data-baseweb="popover"] li { color: rgba(0,0,0,0.85) !important; }

        [data-baseweb="tab"] span,
        [data-baseweb="tab-list"] button { color: rgba(0,0,0,0.70) !important; }
        [data-baseweb="tab"][aria-selected="true"] span { color: rgba(0,0,0,0.90) !important; }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] span,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div { color: rgba(0,0,0,0.85) !important; }

        [data-testid="stCaptionContainer"] p { color: rgba(0,0,0,0.55) !important; }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: rgba(0,0,0,0.85) !important; }

        /* ── Mobile dark mode: only suppress on the specific widget selectors above ── */
        @media (prefers-color-scheme: dark) {
          html, body,
          [data-testid="stAppViewContainer"],
          [data-testid="stAppViewContainer"] > .main {
            background: #ffffff !important;
          }
          [data-testid="stRadio"] label,
          [data-testid="stRadio"] label span,
          .stRadio label,
          [data-testid="stSelectbox"] label,
          [data-baseweb="tab"] span,
          [data-baseweb="tab-list"] button,
          [data-testid="stMetricLabel"],
          [data-testid="stMetricLabel"] span,
          [data-testid="stMetricValue"],
          [data-testid="stMetricValue"] div,
          [data-testid="stCaptionContainer"] p,
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li { color: rgba(0,0,0,0.85) !important; }
        }

        .block-container {
          max-width: 1280px;
          padding-top: 3.5rem;
          padding-bottom: 4rem;
          padding-left: 2rem !important;
          padding-right: 2rem !important;
        }

        /* ── Buttons ── */
        .stButton > button {
          border-radius: 12px !important;
          padding: 0.55rem 0.9rem !important;
          border: 1px solid rgba(0,0,0,0.10) !important;
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
          font-weight: 600 !important;
          width: 100%;
        }
        .stButton > button:hover {
          background: #f5f5f5 !important;
        }

        /* ── Sidebar nav buttons ── */
        [data-testid="stSidebar"] {
          background: #f9fafb !important;
        }
        [data-testid="stSidebar"] .stButton > button {
          text-align: left !important;
          border: none !important;
          background: transparent !important;
          color: rgba(0,0,0,0.65) !important;
          font-size: 12px !important;
          font-weight: 500 !important;
          padding: 7px 12px !important;
          border-radius: 8px !important;
          margin: 1px 0 !important;
          transition: background 0.12s ease !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
          background: rgba(0,0,0,0.06) !important;
          color: rgba(0,0,0,0.88) !important;
        }
        /* Remove focus ring on sidebar buttons */
        [data-testid="stSidebar"] .stButton > button:focus {
          box-shadow: none !important;
          outline: none !important;
        }

        /* ── Inputs / tables ── */
        input, textarea {
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }
        .stDataFrame, .stTable, [data-testid="stTable"] {
          background: #ffffff !important;
          color: rgba(0,0,0,0.85) !important;
        }

        /* ── Topbar ── */
        .me-topbar {
          position: sticky;
          top: 1.5rem;
          z-index: 999;
          background: rgba(255,255,255,0.96);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(0,0,0,0.07);
          border-radius: 20px;
          padding: 16px 20px;
          margin-bottom: 20px;
        }

        /* ── Typography ── */
        .me-title {
          font-size: 26px;
          font-weight: 900;
          letter-spacing: -0.5px;
          margin: 0;
          line-height: 1.05;
          color: rgba(0,0,0,0.90);
        }
        .me-subtle {
          color: rgba(0,0,0,0.50);
          font-size: 12px;
          margin-top: 4px;
        }
        .me-rowtitle {
          font-size: 12px;
          font-weight: 800;
          letter-spacing: 0.5px;
          text-transform: uppercase;
          color: rgba(0,0,0,0.45);
          margin-bottom: 10px;
        }
        .me-kpi {
          font-size: 38px;
          font-weight: 900;
          letter-spacing: -1px;
          line-height: 1.0;
          margin-top: 2px;
          color: rgba(0,0,0,0.90);
        }

        /* ── Chip / pill ── */
        .me-chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          font-size: 16px;
          font-weight: 900;
          white-space: nowrap;
          background: rgba(0,0,0,0.01);
          color: rgba(0,0,0,0.85);
        }
        .me-dot {
          width: 9px;
          height: 9px;
          border-radius: 999px;
          display: inline-block;
          flex-shrink: 0;
        }
        .me-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 5px 10px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          font-size: 12px;
          font-weight: 700;
          color: rgba(0,0,0,0.70);
          background: rgba(0,0,0,0.02);
          margin-right: 6px;
          margin-top: 6px;
        }

        /* ── Delta badge ── */
        .me-badge {
          display: inline-block;
          padding: 3px 8px;
          border-radius: 6px;
          font-size: 12px;
          font-weight: 800;
          white-space: nowrap;
        }
        .me-badge-green  { background: #dcfce7; color: #166534; }
        .me-badge-red    { background: #fee2e2; color: #991b1b; }
        .me-badge-neutral{ background: #f3f4f6; color: #374151; }

        /* ── List items ── */
        .me-li {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          padding: 9px 12px;
          border-radius: 12px;
          border: 1px solid rgba(0,0,0,0.06);
          background: #fafafa;
          margin-bottom: 8px;
        }
        .me-li:last-child { margin-bottom: 0; }
        .me-li-name {
          font-size: 13px;
          font-weight: 700;
          color: rgba(0,0,0,0.80);
          margin: 0;
        }
        .me-li-sub {
          font-size: 12px;
          color: rgba(0,0,0,0.50);
          margin: 0;
          margin-top: 2px;
        }
        .me-li-right {
          font-size: 12px;
          font-weight: 800;
          white-space: nowrap;
          color: rgba(0,0,0,0.70);
        }

        /* ── Nav card ── */
        .me-nav-title {
          font-weight: 800;
          font-size: 13px;
          margin: 0;
          line-height: 1.15;
          color: rgba(0,0,0,0.85);
        }
        .me-nav-desc {
          color: rgba(0,0,0,0.50);
          font-size: 11px;
          margin-top: 3px;
          line-height: 1.3;
        }

        /* ── Driver cards ── */
        .driver-title {
          font-size: 14px;
          font-weight: 800;
          color: rgba(0,0,0,0.85);
          margin: 0;
        }
        .driver-desc {
          font-size: 13px;
          color: rgba(0,0,0,0.60);
          margin-top: 5px;
          line-height: 1.35;
        }
        .driver-meta {
          font-size: 11px;
          color: rgba(0,0,0,0.45);
          margin-top: 8px;
        }

        /* ══════════════════════════════════════════
           MOBILE  (≤ 640 px)
           ══════════════════════════════════════════ */
        @media (max-width: 640px) {

          .block-container {
            padding-top: 3.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
          }

          /* Stack every Streamlit column to full width */
          div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
          }
          div[data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
          }

          /* Topbar */
          .me-topbar {
            top: 0.5rem !important;
            padding: 12px 14px !important;
            border-radius: 14px !important;
            margin-bottom: 12px !important;
            /* prevent chip overflow */
            display: flex;
            flex-direction: column;
            gap: 8px;
          }

          /* Typography scale-down */
          .me-title  { font-size: 20px !important; letter-spacing: -0.2px !important; }
          .me-chip   { font-size: 13px !important; padding: 6px 10px !important; }
          .me-kpi    { font-size: 30px !important; }
          .me-pill   { font-size: 11px !important; padding: 4px 8px !important; }
          .me-subtle { font-size: 11px !important; }

          /* List items: stack label + value vertically */
          .me-li {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 4px !important;
          }
          .me-li-right {
            font-size: 13px !important;
            color: rgba(0,0,0,0.85) !important;
          }

          /* Buttons full-width, larger touch target */
          .stButton > button {
            padding: 0.75rem 1rem !important;
            font-size: 14px !important;
          }

          /* Tabs readable */
          .stTabs [data-baseweb="tab"] {
            font-size: 13px !important;
            padding: 8px 10px !important;
          }

          /* Plotly charts – prevent horizontal overflow */
          .js-plotly-plot, .plotly { max-width: 100% !important; }

          /* Altair charts */
          canvas { max-width: 100% !important; }

          /* Score legend compact */
          .me-score-legend span { font-size: 10px !important; }

          /* Nav cards – hide description on very small screens to save space */
          .me-nav-desc { display: none !important; }
        }

        /* ══════════════════════════════════════════
           SMALL TABLET  (641 – 900 px)
           ══════════════════════════════════════════ */
        @media (min-width: 641px) and (max-width: 900px) {
          .block-container {
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
          }
          .me-title { font-size: 22px !important; }
          .me-kpi   { font-size: 34px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Reusable HTML fragments ───────────────────────────────────────────────────

def delta_badge_html(value: float, inverse: bool = False) -> str:
    """Return a coloured badge HTML string for a numeric delta."""
    if value is None:
        return "<span class='me-badge me-badge-neutral'>–</span>"
    good = value > 0
    if inverse:
        good = not good
    cls = "me-badge-green" if good else "me-badge-red"
    arrow = "↑" if value > 0 else "↓"
    return f"<span class='me-badge {cls}'>{arrow} {abs(value):.2f}</span>"


def html_table(df: "pd.DataFrame", value_col: str | None = None,
               value_color_fn=None) -> str:
    """
    Render a pandas DataFrame as a plain HTML table that is fully controlled
    by our CSS — no iframe, no shadow DOM, works correctly in both desktop
    and mobile dark mode.

    value_col:      name of the column whose cells get colour treatment
    value_color_fn: callable(float) -> hex colour string, applied to value_col
    """
    import pandas as pd

    if df is None or df.empty:
        return "<p style='color:rgba(0,0,0,0.5);font-size:13px;'>No data.</p>"

    header_cells = "".join(
        f"<th style='text-align:left;padding:7px 10px;font-size:11px;"
        f"font-weight:700;color:rgba(0,0,0,0.45);text-transform:uppercase;"
        f"letter-spacing:0.4px;border-bottom:1px solid rgba(0,0,0,0.08);'>{c}</th>"
        for c in df.columns
    )

    rows_html = ""
    for i, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            # Decide text colour
            if col == value_col and value_color_fn is not None:
                try:
                    color = value_color_fn(float(val))
                except Exception:
                    color = "rgba(0,0,0,0.85)"
                weight = "800"
            else:
                color = "rgba(0,0,0,0.80)"
                weight = "600"
            # Format numeric values
            if isinstance(val, float):
                display = f"{val:+.2f}" if col == value_col else f"{val:.2f}"
            else:
                display = str(val)
            cells += (
                f"<td style='padding:7px 10px;font-size:13px;"
                f"font-weight:{weight};color:{color};'>{display}</td>"
            )
        bg = "#fafafa" if i % 2 == 0 else "#ffffff"
        rows_html += f"<tr style='background:{bg};'>{cells}</tr>"

    return (
        "<table style='width:100%;border-collapse:collapse;"
        "border-radius:8px;overflow:hidden;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


def make_chip_row(items: list) -> None:
    if not items:
        return
    chips = " ".join(
        f"<span style='display:inline-block;padding:4px 10px;margin:4px 4px 0 0;"
        f"border-radius:999px;background:#f2f3f5;font-size:12px;color:rgba(0,0,0,0.75);'>{t}</span>"
        for t in items
    )
    st.markdown(chips, unsafe_allow_html=True)
