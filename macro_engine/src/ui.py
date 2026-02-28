# src/ui.py
# Shared UI helpers — import from every page to eliminate duplication.
import streamlit as st


# ── Colour helpers ────────────────────────────────────────────────────────────

def regime_color(regime: str) -> str:
    r = (regime or "").lower()
    if r == "risk on":
        return "#1f7a4f"
    if r == "risk off":
        return "#b42318"
    return "#6b7280"


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
    "Home":             "app.py",
    "Macro charts":     "pages/2_Macro_Charts.py",
    "Regime deep dive": "pages/1_Regime_Deep_Dive.py",
    "Rotation & setups":"pages/4_Rotation_Setups.py",
    "Drivers":          "pages/5_Drivers.py",
    "Ticker drilldown": "pages/3_Ticker_Detail.py",
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


def sidebar_nav(active: str = "Home"):
    st.sidebar.title("Macro Engine")
    st.sidebar.markdown(
        "<div style='font-size:11px;color:rgba(0,0,0,0.45);margin-bottom:6px;'>Navigation</div>",
        unsafe_allow_html=True,
    )
    for name, path in _PAGES.items():
        is_active = name == active
        label = f"**{name}**" if is_active else name
        if st.sidebar.button(label, key=f"sidenav_{name}", use_container_width=True, disabled=is_active):
            safe_switch_page(path)


# ── Score legend ──────────────────────────────────────────────────────────────

SCORE_LEGEND_HTML = (
    "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;'>"
    "<span style='font-size:11px;color:#b42318;font-weight:700;'>▼ &lt;40 Bearish</span>"
    "<span style='font-size:11px;color:#6b7280;font-weight:700;'>● 40–60 Neutral</span>"
    "<span style='font-size:11px;color:#1f7a4f;font-weight:700;'>▲ &gt;60 Bullish</span>"
    "</div>"
)


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

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

        .block-container {
          max-width: 1200px;
          padding-top: 4.8rem;
          padding-bottom: 5rem;
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
        [data-testid="stSidebar"] .stButton > button {
          text-align: left !important;
          border: none !important;
          background: transparent !important;
          color: rgba(0,0,0,0.75) !important;
          font-size: 13px !important;
          padding: 0.4rem 0.7rem !important;
          border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
          background: rgba(0,0,0,0.05) !important;
        }
        [data-testid="stSidebar"] .stButton > button:disabled {
          background: rgba(0,0,0,0.06) !important;
          color: rgba(0,0,0,0.85) !important;
          font-weight: 800 !important;
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


def make_chip_row(items: list) -> None:
    if not items:
        return
    chips = " ".join(
        f"<span style='display:inline-block;padding:4px 10px;margin:4px 4px 0 0;"
        f"border-radius:999px;background:#f2f3f5;font-size:12px;color:rgba(0,0,0,0.75);'>{t}</span>"
        for t in items
    )
    st.markdown(chips, unsafe_allow_html=True)