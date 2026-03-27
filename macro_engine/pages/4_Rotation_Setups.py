# pages/4_Rotation_Setups.py
"""
Rotation & Setups — v3  (pair signals + copula risk layer)
═══════════════════════════════════════════════════════════════════════════════

PAIR SCORING  (what makes a good relative trade signal)
────────────────────────────────────────────────────────
For every A/B combination we compute ratio r = price_A / price_B then score:

  Signal 1 — Ratio momentum z  (wt 0.35)
    Z-score of the 21d rate-of-change of r vs trailing 252d history.
    Positive = A accelerating vs B.

  Signal 2 — Ratio trend  (wt 0.25)
    r above its own 63d moving average → persistent structural rotation.

  Signal 3 — Mean-reversion component  (wt 0.25)
    Negative of the ratio level z-score. A very negative z (A cheap vs B)
    is an *additional* tailwind if momentum is also turning up — and a caution
    flag if z is extremely positive (A stretched vs B).

  Signal 4 — 1m confirmation  (wt 0.15)
    1-month return of the ratio. Filters false breakouts.

Raw score mapped to [-100, +100]. Positive = A outperforms. Negative = B outperforms.

COPULA RISK LAYER  (why copulas matter for pair trades)
────────────────────────────────────────────────────────
Standard correlation (Pearson/Spearman) only captures *average* dependence.
It misses what actually kills pair trades: assets that are decorrelated on
normal days but co-crash during stress — *lower tail dependence*.

We fit a bivariate Clayton copula to each pair's weekly returns.
Clayton's parameter θ is estimated analytically from Kendall's τ:
    θ = 2τ / (1 − τ)
The lower tail dependence coefficient λ_L = 2^(−1/θ) gives the probability
that both assets tank simultaneously in the worst quantiles.

We also compute a **Copula regime flag**: is the current 63-day rolling
Spearman ρ materially above the historically expected ρ from the fitted copula?
If yes, the pair is in an "anomalous co-movement" state — a forced correlation
that often snaps back but makes the trade riskier in the meantime.

How copulas modify the pair score:
  • High tail dependence (λ_L > 0.35) → score penalised 15–25%
    (long A / short B blows up in drawdowns — both legs fall)
  • Anomalous regime flag → additional 10% penalty + warning
  • Low tail dependence (λ_L < 0.15) → score boosted 5%
    (structurally decorrelated pair = safer relative value trade)

Basket
───────
XLE XLF XLK XLI XLP XLV GLD IWM QQQ TLT HYG BTC XBI UUP SPY
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import altair as alt
import streamlit as st

from src.data_sources import fetch_prices
from src.ui import inject_css, sidebar_nav, safe_switch_page

st.set_page_config(
    page_title="Rotation & Setups",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
sidebar_nav(active="Rotation & setups")

# ── Basket ────────────────────────────────────────────────────────────────────

BASKET = [
    "XLE", "XLF", "XLK", "XLI", "XLP", "XLV",
    "GLD", "IWM", "QQQ", "TLT", "HYG",
    "BTC", "XBI", "UUP", "SPY",
]

LABELS = {
    "XLE": "Energy",     "XLF": "Financials",  "XLK": "Technology",
    "XLI": "Industrials","XLP": "Staples",      "XLV": "Healthcare",
    "GLD": "Gold",       "IWM": "Small caps",   "QQQ": "Nasdaq",
    "TLT": "Long bonds", "HYG": "High yield",   "BTC": "Bitcoin",
    "XBI": "Biotech",    "UUP": "USD",          "SPY": "S&P 500",
}

TICKER_COLORS = {
    "XLE": "#f97316", "XLF": "#3b82f6", "XLK": "#8b5cf6", "XLI": "#64748b",
    "XLP": "#22c55e", "XLV": "#06b6d4", "GLD": "#eab308", "IWM": "#ec4899",
    "QQQ": "#6366f1", "TLT": "#0ea5e9", "HYG": "#a855f7", "BTC": "#f59e0b",
    "XBI": "#10b981", "UUP": "#ef4444", "SPY": "#1d4ed8",
}

HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}

# ── Pair universe rules ───────────────────────────────────────────────────────
# SECTOR_ETFs: equity sectors — these pair against each other freely.
# ALTERNATES:  non-equity assets (crypto, gold, bonds, credit, USD).
#              These pair against each other AND against sector ETFs, but
#              macro anchors (UUP, TLT, SPY) are capped at MAX_APPEARANCES
#              in the displayed results so they don't flood the cards.
# SPY is kept for RS/RRG context but excluded from pair signal cards entirely.

SECTOR_ETFS  = {"XLE","XLF","XLK","XLI","XLP","XLV","IWM","QQQ","XBI"}
ALTERNATES   = {"GLD","HYG","BTC","TLT","UUP"}
# SPY excluded from pair cards — it's the benchmark, not a trade leg
PAIR_EXCLUDE = {"SPY"}
# Macro anchors (directional dollar/rates bets) — capped more aggressively
# so they don't crowd out genuine sector rotation signals
MACRO_ANCHORS = {"UUP", "TLT", "GLD"}
# Max appearances per ticker in the displayed top-pairs grid
MAX_PER_TICKER        = 2   # sector ETFs / alts
MAX_PER_MACRO_ANCHOR  = 1   # UUP, TLT, GLD each appear at most once


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_prices() -> pd.DataFrame:
    df = fetch_prices(BASKET, period="5y")
    return df.sort_index() if df is not None and not df.empty else pd.DataFrame()

px = load_prices()
if px.empty:
    st.error("No price data available.")
    st.stop()

available = [t for t in BASKET if t in px.columns]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _zscore(s: pd.Series, window: int = 252) -> float | None:
    s = s.dropna()
    if len(s) < max(30, window // 4): return None
    w = s.iloc[-min(window, len(s)):]
    sd = float(w.std())
    return float((w.iloc[-1] - w.mean()) / sd) if sd > 0 else 0.0

def _roc_zscore(s: pd.Series, roc_days: int = 21, window: int = 252) -> float | None:
    s = s.dropna()
    if len(s) < roc_days + max(30, window // 4): return None
    return _zscore(s.diff(roc_days).dropna(), window)

def _above_ma(s: pd.Series, window: int = 63) -> float | None:
    s = s.dropna()
    if len(s) < window: return None
    return 1.0 if float(s.iloc[-1]) > float(s.rolling(window).mean().iloc[-1]) else -1.0

def _ret(s: pd.Series, days: int) -> float | None:
    s = s.dropna()
    if len(s) < days + 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] / s.loc[prev[-1]] - 1) if len(prev) else None

def _clip(z, cap: float = 2.5) -> float:
    return float(np.clip(z if z is not None else 0.0, -cap, cap))


# ══════════════════════════════════════════════════════════════════════════════
# COPULA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _to_uniform(returns: pd.Series) -> np.ndarray:
    """Empirical CDF transform — maps returns to [0,1] (pseudo-observations)."""
    n = len(returns)
    return stats.rankdata(returns) / (n + 1)

def fit_clayton(u: np.ndarray, v: np.ndarray) -> dict:
    """
    Fit a bivariate Clayton copula to pseudo-observations (u, v).

    Steps:
      1. Compute Kendall's tau from the pseudo-observations.
      2. Analytically solve for Clayton theta: theta = 2*tau / (1-tau).
         (MLE is impractical here; tau inversion is exact for Clayton.)
      3. Compute lower tail dependence: lambda_L = 2^(-1/theta).
         Interpretation: probability both assets tank together in the
         worst quantiles.
      4. Compute expected Spearman rho from theta: rho = 3*tau - 1 ≈ approx.
         Then compare with the actual 63d rolling Spearman to detect
         anomalous co-movement.

    Returns dict with: theta, lambda_L, tau, expected_rho, regime_flag, regime_delta.
    """
    if len(u) < 30 or len(v) < 30:
        return {}

    # Kendall's tau (nonparametric, robust)
    tau, _ = stats.kendalltau(u, v)
    tau = float(tau)

    # ── Clayton requires positive tau (models lower tail, i.e. co-crash risk).
    # Three regimes:
    #   tau > 0.05 → fit Clayton normally; compute λ_L and regime flag.
    #   0 ≤ tau ≤ 0.05 → near-independence; no meaningful tail dependence,
    #                     but also no bonus — these are uncorrelated bets.
    #   tau < 0 → negatively correlated pair (e.g. XLE/TLT, sector/UUP).
    #             Clayton doesn't apply. We record lambda_l=0 but apply a
    #             mild penalty because negative-tau pairs are macro bets that
    #             can violently reverse — NOT structurally "safe" decorrelation.
    #
    # Regime flag is ONLY meaningful when tau > 0 (copula expected_rho is
    # defined). For negative-tau pairs we suppress the flag entirely to avoid
    # the 20 spurious anomalous-regime counts seen when UUP floods the pairs.

    if tau > 0.05:
        theta = max(float(2 * tau / (1 - tau)), 0.01)
        lambda_l = float(2 ** (-1.0 / theta))
        expected_rho = float(tau * (tau + 2) / 3)   # Clayton → Spearman approx
        recent_n = min(63, len(u))
        current_rho, _ = stats.spearmanr(u[-recent_n:], v[-recent_n:])
        current_rho = float(current_rho)
        regime_delta = current_rho - expected_rho
        # Only flag if the shift is large AND tau suggests meaningful copula fit
        regime_flag = bool(abs(regime_delta) > 0.22 and tau > 0.15)
    elif tau >= 0:
        # Near-independence — Clayton undefined, no bonus, no flag
        theta = 0.0; lambda_l = 0.0; expected_rho = 0.0
        recent_n = min(63, len(u))
        current_rho, _ = stats.spearmanr(u[-recent_n:], v[-recent_n:])
        current_rho = float(current_rho)
        regime_delta = 0.0; regime_flag = False
    else:
        # Negative tau — anti-correlated (macro bet). No Clayton fit.
        theta = 0.0; lambda_l = 0.0; expected_rho = float(tau)
        recent_n = min(63, len(u))
        current_rho, _ = stats.spearmanr(u[-recent_n:], v[-recent_n:])
        current_rho = float(current_rho)
        regime_delta = 0.0; regime_flag = False

    return {
        "theta":        theta,
        "lambda_l":     lambda_l,
        "tau":          tau,
        "expected_rho": expected_rho,
        "current_rho":  current_rho,
        "regime_delta": regime_delta,
        "regime_flag":  regime_flag,
        "negative_tau": tau < 0,   # flag for penalty logic below
    }

def copula_risk_penalty(copula: dict) -> float:
    """
    Returns a multiplier in [0.60, 1.05] to apply to the raw pair score.

    Positive tau, high λ_L → both legs crash together → penalise.
    Positive tau, low  λ_L → structurally safer pair → small bonus.
    Negative tau         → anti-correlated macro bet → mild penalty
                           (NOT a bonus — these can violently reverse).
    Near-zero tau        → independent pair, no copula signal → neutral.
    Anomalous regime     → forced co-movement → penalise.
    """
    if not copula:
        return 1.0

    tau   = copula.get("tau", 0.0) or 0.0
    lam   = copula.get("lambda_l", 0.0) or 0.0
    flag  = copula.get("regime_flag", False)
    delta = copula.get("regime_delta", 0.0) or 0.0
    neg   = copula.get("negative_tau", False)

    penalty = 1.0

    if neg:
        # Negatively correlated pair — macro bet, not structural rotation
        penalty = 0.88
    elif tau > 0.05:
        # Clayton applies — use tail dependence
        if lam > 0.45:
            penalty *= 0.72
        elif lam > 0.35:
            penalty *= 0.82
        elif lam > 0.25:
            penalty *= 0.91
        elif lam < 0.10:
            penalty *= 1.04   # genuine low-risk decorrelated pair
    # tau in [0, 0.05]: near-independent → neutral (1.0)

    # Anomalous regime penalty (only when copula fit is meaningful)
    if flag and tau > 0.15:
        extra = min(abs(delta) / 0.22 * 0.10, 0.15)
        penalty *= (1.0 - extra)

    return float(np.clip(penalty, 0.60, 1.05))


# ══════════════════════════════════════════════════════════════════════════════
# PAIR SCORING  (momentum signals + copula adjustment)
# ══════════════════════════════════════════════════════════════════════════════

def score_pair(a: str, b: str, px: pd.DataFrame) -> dict | None:
    if a not in px.columns or b not in px.columns: return None

    # Build ratio
    common = px[a].dropna().index.intersection(px[b].dropna().index)
    if len(common) < 90: return None
    sa = px[a].loc[common]; sb = px[b].loc[common]
    ratio = (sa / sb).dropna()
    if len(ratio) < 90: return None

    # ── Momentum signals ──────────────────────────────────────────────────
    mom_z   = _roc_zscore(ratio, 21, 252)
    trend   = _above_ma(ratio, 63)
    level_z = _zscore(ratio, 252)
    r1m     = _ret(ratio, 21)

    s1 = _clip(mom_z)   * 0.35
    s2 = (trend or 0.0) * 0.25 * 2.5
    s3 = _clip(-(level_z or 0.0)) * 0.25
    s4 = _clip((r1m or 0.0) * 100 / 2.5) * 2.5 * 0.15
    raw = s1 + s2 + s3 + s4
    raw_score = float(np.clip(raw / 2.5 * 100, -100, 100))

    # ── Copula layer ──────────────────────────────────────────────────────
    # Use weekly returns for copula (less noise than daily)
    wkly_a = sa.resample("W-FRI").last().pct_change().dropna()
    wkly_b = sb.resample("W-FRI").last().pct_change().dropna()
    cidx   = wkly_a.index.intersection(wkly_b.index)
    copula = {}
    if len(cidx) >= 52:
        u = _to_uniform(wkly_a.loc[cidx].values)
        v = _to_uniform(wkly_b.loc[cidx].values)
        copula = fit_clayton(u, v)

    penalty = copula_risk_penalty(copula)
    adj_score = float(np.clip(raw_score * penalty, -100, 100))

    abs_s = abs(adj_score)
    conviction = "Strong" if abs_s >= 65 else ("Moderate" if abs_s >= 45 else "Weak")
    leader  = a if adj_score > 0 else b
    laggard = b if adj_score > 0 else a

    return {
        "a": a, "b": b,
        "raw_score":  raw_score,
        "adj_score":  adj_score,
        "abs_score":  abs_s,
        "penalty":    penalty,
        "conviction": conviction,
        "leader":  leader, "laggard": laggard,
        "mom_z":   mom_z,  "trend":   trend,
        "level_z": level_z,"r1m":     r1m,
        "ratio":   ratio,
        # Copula fields
        "lambda_l":     copula.get("lambda_l"),
        "tau":          copula.get("tau"),
        "theta":        copula.get("theta"),
        "expected_rho": copula.get("expected_rho"),
        "current_rho":  copula.get("current_rho"),
        "regime_delta": copula.get("regime_delta"),
        "regime_flag":  copula.get("regime_flag", False),
    }


def _valid_pair(a: str, b: str) -> bool:
    """Return True if this pair should be scored and shown in pair cards.

    Rules:
    - SPY is never a pair leg (it's the benchmark)
    - UUP/TLT vs a sector ETF is a macro-beta trade not a rotation trade;
      we keep them in the pool but MAX_PER_TICKER handles dominance at display time.
    - Everything else is valid.
    """
    if a in PAIR_EXCLUDE or b in PAIR_EXCLUDE:
        return False
    return True


@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_all_pairs(_px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    combos    = [(a, b) for a, b in itertools.combinations(available, 2)
                 if _valid_pair(a, b)]
    raw_pairs = [score_pair(a, b, _px) for a, b in combos]
    valid     = [p for p in raw_pairs if p is not None]
    ratio_store = {(p["a"], p["b"]): p["ratio"] for p in valid}
    rows = [{k: v for k, v in p.items() if k != "ratio"} for p in valid]
    df   = pd.DataFrame(rows).sort_values("abs_score", ascending=False).reset_index(drop=True)
    return df, ratio_store

pairs_df, ratio_store = build_all_pairs(px)


# ── RS vs SPY table ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_rs_table(_px: pd.DataFrame) -> pd.DataFrame:
    if "SPY" not in _px.columns: return pd.DataFrame()
    rows = []
    for t in available:
        if t == "SPY": continue
        row = {"Ticker": t, "Name": LABELS.get(t, t)}
        for lbl, n in HORIZONS.items():
            ra = _ret(_px[t], n); rb = _ret(_px["SPY"], n)
            row[lbl] = float(ra - rb) if ra is not None and rb is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")

rs_df = build_rs_table(px)


# ── RRG coordinates ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_rrg(_px: pd.DataFrame) -> pd.DataFrame:
    if "SPY" not in _px.columns: return pd.DataFrame()
    spy = _px["SPY"].dropna()
    out = []
    for t in available:
        if t == "SPY": continue
        s = _px[t].dropna()
        cidx = s.index.intersection(spy.index)
        if len(cidx) < 60: continue
        ratio   = (s.loc[cidx] / spy.loc[cidx]).dropna()
        smooth  = ratio.rolling(50, min_periods=20).mean().dropna()
        if len(smooth) < 22: continue
        rs52    = float(smooth.rolling(252, min_periods=50).mean().iloc[-1]) \
                  if len(smooth) >= 50 else float(smooth.mean())
        rs_ratio = float(smooth.iloc[-1]) / rs52 * 100 if rs52 != 0 else 100.0
        rs_4w    = float(smooth.iloc[-21]) if len(smooth) >= 21 else float(smooth.iloc[0])
        rs_mom   = float(smooth.iloc[-1]) / rs_4w * 100 if rs_4w != 0 else 100.0
        out.append({"Ticker": t, "Name": LABELS.get(t, t),
                    "RS_ratio": rs_ratio, "RS_mom": rs_mom})
    return pd.DataFrame(out)

rrg_df = build_rrg(px)


# ── Copula summary table ──────────────────────────────────────────────────────
# Aggregate copula stats for the full pair matrix

@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_copula_matrix(_px: pd.DataFrame) -> pd.DataFrame:
    """Build a ticker × ticker DataFrame of lower tail dependence coefficients."""
    n  = len(available)
    mat = pd.DataFrame(np.nan, index=available, columns=available)
    for a, b in itertools.combinations(available, 2):
        row = pairs_df[(pairs_df["a"] == a) & (pairs_df["b"] == b)]
        if row.empty:
            row = pairs_df[(pairs_df["a"] == b) & (pairs_df["b"] == a)]
        if not row.empty:
            lam = row.iloc[0]["lambda_l"]
            if lam is not None and not (isinstance(lam, float) and np.isnan(lam)):
                mat.loc[a, b] = float(lam)
                mat.loc[b, a] = float(lam)
    np.fill_diagonal(mat.values, 1.0)
    return mat

copula_mat = build_copula_matrix(px)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — Plotly-safe."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(29,78,216,{alpha})"   # fallback blue


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x)*100:+.1f}%"

def fmt_f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x):.{nd}f}"

def score_color(s):
    if abs(s) >= 65: return "#1f7a4f" if s > 0 else "#b42318"
    if abs(s) >= 45: return "#d97706"
    return "#94a3b8"

def conviction_palette(c):
    return {
        "Strong":   ("#1f7a4f", "#dcfce7"),
        "Moderate": ("#d97706", "#fef9c3"),
        "Weak":     ("#94a3b8", "#f3f4f6"),
    }.get(c, ("#94a3b8", "#f3f4f6"))

def lambda_color(lam):
    if lam is None or np.isnan(lam): return "#6b7280", "#f3f4f6"
    if lam > 0.40: return "#b42318", "#fee2e2"
    if lam > 0.25: return "#d97706", "#fef9c3"
    return "#1f7a4f", "#dcfce7"

def kpi_card(col, label, value, sub, color="#0f172a", bg="#f8fafc"):
    col.markdown(
        f"<div style='padding:12px 14px;border-radius:12px;background:{bg};"
        f"border:1px solid rgba(0,0,0,0.07);'>"
        f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.4);"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-size:20px;font-weight:900;color:{color};line-height:1.1;'>{value}</div>"
        f"<div style='font-size:11px;color:rgba(0,0,0,0.5);margin-top:3px;'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)


# ── Summary stats ─────────────────────────────────────────────────────────────

top_leaders  = rs_df["1m"].dropna().sort_values(ascending=False).head(3) if not rs_df.empty else pd.Series()
top_laggards = rs_df["1m"].dropna().sort_values(ascending=True).head(3)  if not rs_df.empty else pd.Series()
top_pair     = pairs_df.iloc[0] if not pairs_df.empty else None
strong_n     = int((pairs_df["abs_score"] >= 65).sum()) if not pairs_df.empty else 0
flagged_n    = int(pairs_df["regime_flag"].sum()) if not pairs_df.empty else 0


# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        """<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Rotation &amp; Setups</div>
              <div class="me-subtle">
                Pair trade signals &nbsp;·&nbsp; Clayton copula risk &nbsp;·&nbsp;
                relative strength &nbsp;·&nbsp; RRG quadrants
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ── KPI strip ─────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

leader_txt = " · ".join(f"{t} {fmt_pct(v)}" for t, v in top_leaders.items()) \
             if not top_leaders.empty else "—"
laggard_txt = " · ".join(f"{t} {fmt_pct(v)}" for t, v in top_laggards.items()) \
              if not top_laggards.empty else "—"

kpi_card(k1, "1m Leaders vs SPY",
         top_leaders.index[0] if not top_leaders.empty else "—",
         leader_txt, color="#1f7a4f", bg="#dcfce7")
kpi_card(k2, "1m Laggards vs SPY",
         top_laggards.index[0] if not top_laggards.empty else "—",
         laggard_txt, color="#b42318", bg="#fee2e2")
if top_pair is not None:
    kpi_card(k3, "Top pair signal",
             f"{top_pair['leader']} > {top_pair['laggard']}",
             f"Score {top_pair['adj_score']:.0f} · {top_pair['conviction']}",
             color=score_color(top_pair["adj_score"]),
             bg=conviction_palette(top_pair["conviction"])[1])
else:
    kpi_card(k3, "Top pair signal", "—", "")
kpi_card(k4, "Strong pairs", str(strong_n),
         f"of {len(pairs_df)} pairs · |score| ≥ 65",
         color="#1d4ed8", bg="#eff6ff")
kpi_card(k5, "Anomalous regimes", str(flagged_n),
         "pairs in unusual correlation state",
         color="#d97706" if flagged_n > 3 else "#6b7280",
         bg="#fef9c3" if flagged_n > 3 else "#f3f4f6")

st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOP PAIR SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='me-rowtitle'>Top pair signals</div>", unsafe_allow_html=True)
st.caption(
    "Positive score = left ticker expected to outperform. "
    "Score already adjusted by the Clayton copula tail-dependence penalty — "
    "pairs where both assets crash together are penalised. "
    "λ_L = lower tail dependence (high = risky pair trade in drawdowns).")

# ── How to read pair signals ──────────────────────────────────────────────────
_n_strong   = int((pairs_df["conviction"] == "Strong").sum())  if not pairs_df.empty else 0
_n_flagged  = int(pairs_df["regime_flag"].sum()) if not pairs_df.empty and "regime_flag" in pairs_df.columns else 0
_top_score  = float(pairs_df["adj_score"].abs().max()) if not pairs_df.empty else 0

if _n_flagged > 4:
    _mkt_context = (
        f"⚠ {_n_flagged} pairs are showing anomalous correlation regimes. "
        "When pairs that are normally decorrelated start moving together, "
        "it signals a stress event or macro shift is forcing correlation. "
        "These pairs carry hidden tail risk — the copula penalty has been applied but "
        "consider reducing position sizes on flagged pairs by 30-50%."
    )
    _ctx_color = "#d97706"; _ctx_bg = "#fef9c3"
elif _n_strong >= 6:
    _mkt_context = (
        f"{_n_strong} strong pair signals — above average rotation opportunity. "
        "Multiple pairs with high conviction suggests a clear sector rotation is underway. "
        "Strong signals with low λ_L (tail dependence) are the cleanest trades: "
        "momentum is present and the pair is structurally decorrelated in drawdowns."
    )
    _ctx_color = "#1f7a4f"; _ctx_bg = "#dcfce7"
else:
    _mkt_context = (
        "Pair signals interpret relative momentum between assets in the basket. "
        "A score of +65 or above (Strong) means momentum, trend, and confirmation all align. "
        "The copula adjustment penalises pairs where both assets tend to crash together — "
        "a high raw score on a high-λ_L pair is a weaker trade than a moderate score on a low-λ_L pair."
    )
    _ctx_color = "#6b7280"; _ctx_bg = "#f3f4f6"

st.markdown(
    f"<div style='padding:11px 16px;border-radius:11px;background:{_ctx_bg};"
    f"border-left:4px solid {_ctx_color};font-size:12px;color:rgba(0,0,0,0.75);"
    f"line-height:1.6;margin-bottom:12px;'>{_mkt_context}</div>",
    unsafe_allow_html=True)

# ── What the copula stats mean ────────────────────────────────────────────────
with st.expander("How to read the copula stats on each card"):
    st.markdown("""
**λ_L (lower tail dependence)** — the probability that both assets in the pair fall simultaneously 
during a market stress event. Derived from a fitted Clayton copula, which captures left-tail 
dependence that standard correlation misses.

- **λ_L < 0.15** — structurally decorrelated in drawdowns. The pair is a clean relative value trade.
- **λ_L 0.15–0.35** — moderate tail dependence. Use standard position sizing.
- **λ_L > 0.35** — both legs tend to fall together in drawdowns. Score is penalised 15-25%.

**⚠ REGIME flag** — the current 63-day rolling correlation is materially above the copula's expected 
level. This means the pair is in a forced co-movement state — they are moving together more than 
their long-run relationship predicts. This often snaps back but makes the trade higher risk 
in the near term.

**Raw vs adjusted score** — the raw score is the pure signal. The adjusted score reflects the 
copula penalty. A large gap (e.g. Raw +80, Adj +58) means the signal is strong but the risk 
structure is unfavourable. Prefer pairs where raw and adjusted scores are close.
    """)

if not pairs_df.empty:
    def _dedup_by_ticker(df_in: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        """Walk the ranked list and keep a row only if:
          - neither regular ticker has appeared >= MAX_PER_TICKER times, AND
          - neither macro anchor has appeared >= MAX_PER_MACRO_ANCHOR times.
        This prevents UUP/TLT/GLD from flooding the card grid while still
        allowing them to appear once if they genuinely have the top signal."""
        counts: dict[str, int] = {}
        kept = []
        for _, row in df_in.iterrows():
            a, b = row["a"], row["b"]
            cap_a = MAX_PER_MACRO_ANCHOR if a in MACRO_ANCHORS else MAX_PER_TICKER
            cap_b = MAX_PER_MACRO_ANCHOR if b in MACRO_ANCHORS else MAX_PER_TICKER
            if counts.get(a, 0) >= cap_a or counts.get(b, 0) >= cap_b:
                continue
            kept.append(row)
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
            if len(kept) >= n:
                break
        return pd.DataFrame(kept) if kept else pd.DataFrame(columns=df_in.columns)

    strong_df   = _dedup_by_ticker(pairs_df[pairs_df["conviction"] == "Strong"],  9)
    moderate_df = _dedup_by_ticker(pairs_df[pairs_df["conviction"] == "Moderate"], 9)

    tab_s, tab_m = st.tabs([
        f"⚡ Strong  ({len(strong_df)})",
        f"· Moderate  ({len(moderate_df)})",
    ])

    def _mini_bar(label: str, val, *, invert: bool = False,
                  fmt_fn=None, cap: float = 2.5):
        """Single signal row with inline bar."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return (f"<div style='display:flex;justify-content:space-between;"
                    f"padding:3px 0;font-size:11px;border-bottom:1px solid rgba(0,0,0,0.04);'>"
                    f"<span style='color:rgba(0,0,0,0.40);width:80px;flex-shrink:0;'>{label}</span>"
                    f"<span style='color:#94a3b8;'>—</span></div>")
        fv  = float(val) * (-1 if invert else 1)
        pct = min(abs(fv) / cap * 52, 52)
        bc  = "#1f7a4f" if fv > 0 else "#b42318"
        vs  = fmt_fn(val) if fmt_fn else f"{float(val):+.2f}"
        return (
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:3px 0;font-size:11px;border-bottom:1px solid rgba(0,0,0,0.04);'>"
            f"<span style='color:rgba(0,0,0,0.45);width:80px;flex-shrink:0;'>{label}</span>"
            f"<div style='flex:1;display:flex;align-items:center;gap:5px;justify-content:flex-end;'>"
            f"<div style='width:{pct:.0f}px;height:5px;border-radius:3px;"
            f"background:{bc};flex-shrink:0;'></div>"
            f"<span style='font-weight:700;color:{bc};min-width:46px;text-align:right;'>{vs}</span>"
            f"</div></div>")

    def _pair_grid(df_p: pd.DataFrame, ncols: int = 3):
        if df_p.empty:
            st.caption("No pairs at this conviction level.")
            return
        cols = st.columns(ncols, gap="medium")
        for i, (_, row) in enumerate(df_p.iterrows()):
            a, b    = row["a"], row["b"]
            s       = row["adj_score"]
            rs      = row["raw_score"]
            conv    = row["conviction"]
            leader  = row["leader"]
            laggard = row["laggard"]
            pen     = row["penalty"]
            lam     = row["lambda_l"]
            rflag   = row["regime_flag"]
            sc      = score_color(s)
            cc, cbg = conviction_palette(conv)
            lam_c, lam_bg = lambda_color(lam if lam is not None else 0.0)

            # Signals HTML
            sigs = (
                _mini_bar("Mom z",    row["mom_z"]) +
                _mini_bar("Trend",    row["trend"],
                          fmt_fn=lambda v: "Above ↑" if float(v) > 0 else "Below ↓") +
                _mini_bar("Level z",  row["level_z"], invert=True) +
                _mini_bar("1m ratio", row["r1m"],
                          fmt_fn=lambda v: f"{float(v)*100:+.1f}%", cap=0.05)
            )

            # Copula row
            lam_str  = fmt_f(lam, 3) if lam is not None else "—"
            tau_str  = fmt_f(row.get("tau"), 2) if row.get("tau") is not None else "—"
            pen_str  = f"{pen:.2f}×" if pen is not None else "—"
            flag_html = (
                f"<span style='font-size:9px;font-weight:800;color:#d97706;"
                f"background:#fef9c3;padding:1px 5px;border-radius:4px;margin-left:4px;'>"
                f"⚠ REGIME</span>" if rflag else ""
            )

            adj_note = (
                f"<div style='font-size:9px;color:rgba(0,0,0,0.40);margin-top:3px;'>"
                f"Raw {rs:+.0f} → copula adj {s:+.0f} (×{pen:.2f})</div>"
            ) if abs(rs - s) > 1 else ""

            a_c = TICKER_COLORS.get(a, "#6b7280")
            b_c = TICKER_COLORS.get(b, "#6b7280")

            with cols[i % ncols]:
                st.markdown(
                    f"<div style='padding:14px 14px 10px;border-radius:14px;background:#fafafa;"
                    f"border:1.5px solid {sc}44;margin-bottom:14px;'>"

                    # ─ Title row ─
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:8px;'>"
                    f"<span style='font-size:16px;font-weight:900;'>"
                    f"<span style='color:{a_c};'>{a}</span>"
                    f"<span style='color:rgba(0,0,0,0.28);font-size:12px;'> / </span>"
                    f"<span style='color:{b_c};'>{b}</span></span>"
                    f"<span style='font-size:20px;font-weight:900;color:{sc};'>{s:+.0f}</span>"
                    f"</div>"

                    # ─ Direction badge ─
                    f"<div style='padding:6px 10px;border-radius:8px;background:{cbg};"
                    f"font-size:12px;font-weight:700;color:{cc};margin-bottom:8px;'>"
                    f"▲ {leader} <span style='opacity:0.55;font-weight:600;'>outperforms ▼ {laggard}</span>"
                    f"</div>"

                    # ─ Sector labels ─
                    f"<div style='font-size:10px;color:rgba(0,0,0,0.38);margin-bottom:8px;'>"
                    f"{LABELS.get(a,'')} vs {LABELS.get(b,'')}</div>"

                    # ─ Momentum signals ─
                    f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.38);"
                    f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>"
                    f"Momentum signals</div>"
                    f"{sigs}"

                    # ─ Copula block ─
                    f"<div style='margin-top:10px;padding:8px 10px;border-radius:8px;"
                    f"background:{lam_bg};'>"
                    f"<div style='font-size:10px;font-weight:700;color:{lam_c};"
                    f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>"
                    f"Clayton copula risk{flag_html}</div>"
                    f"<div style='display:flex;gap:14px;font-size:11px;'>"
                    f"<div><span style='color:rgba(0,0,0,0.45);'>λ_L</span> "
                    f"<span style='font-weight:800;color:{lam_c};'>{lam_str}</span></div>"
                    f"<div><span style='color:rgba(0,0,0,0.45);'>τ</span> "
                    f"<span style='font-weight:700;'>{tau_str}</span></div>"
                    f"<div><span style='color:rgba(0,0,0,0.45);'>penalty</span> "
                    f"<span style='font-weight:700;'>{pen_str}</span></div>"
                    f"</div>"
                    f"<div style='font-size:10px;color:{lam_c};margin-top:3px;'>"
                    f"{'⚠ High co-crash risk' if lam and lam > 0.35 else ('✓ Low co-crash risk' if lam and lam < 0.15 else 'Moderate co-crash risk')}"
                    f"</div>"
                    f"</div>"

                    f"{adj_note}"
                    f"</div>",
                    unsafe_allow_html=True)

    with tab_s:
        _pair_grid(strong_df, ncols=3)
    with tab_m:
        _pair_grid(moderate_df, ncols=3)

st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RS CHART + RRG QUADRANT
# ══════════════════════════════════════════════════════════════════════════════

rs_col, rrg_col = st.columns([1.05, 1.0], gap="large")

with rs_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Relative strength vs SPY</div>",
                    unsafe_allow_html=True)
        if not rs_df.empty:
            hz = st.radio("Horizon", list(HORIZONS.keys()),
                          horizontal=True, index=1, key="rs_hz")
            rs_sorted = rs_df[hz].dropna().sort_values(ascending=False).reset_index()
            rs_sorted.columns = ["Ticker", "RS"]

            fig_rs = go.Figure()
            for _, row_r in rs_sorted.iterrows():
                t = row_r["Ticker"]; v = float(row_r["RS"])
                fig_rs.add_trace(go.Bar(
                    x=[v], y=[t], orientation="h",
                    marker_color=TICKER_COLORS.get(t, "#6b7280"),
                    marker_opacity=0.85, showlegend=False,
                    text=f"{v*100:+.1f}%", textposition="outside",
                    textfont=dict(size=10),
                    hovertemplate=f"<b>{t}</b> {LABELS.get(t,'')}<br>{hz} RS: {v*100:+.1f}%<extra></extra>",
                ))
            fig_rs.add_vline(x=0, line_color="#94a3b8", line_width=1.5)
            vals = rs_sorted["RS"].values
            lo = float(min(vals)); hi = float(max(vals))
            pad = max((hi - lo) * 0.28, 0.004)
            fig_rs.update_layout(
                height=max(300, len(rs_sorted) * 26),
                margin=dict(l=10, r=60, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(tickformat="+.1%", range=[lo - pad, hi + pad],
                           showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(showgrid=False, autorange="reversed"),
                hovermode="y unified",
            )
            st.plotly_chart(fig_rs, width='stretch')

with rrg_col:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Rotation quadrant (RRG-style)</div>",
                    unsafe_allow_html=True)
        st.caption("X = RS level vs 52w avg · Y = RS 4w momentum · "
                   "Leading = strong & improving")
        if not rrg_df.empty:
            fig_rrg = go.Figure()
            # Quadrant shading
            for x0, x1, y0, y1, fc in [
                (100,120,100,120,"rgba(31,122,79,0.07)"),   # Leading
                (100,120, 80,100,"rgba(217,119,6,0.06)"),   # Weakening
                ( 80,100,100,120,"rgba(29,78,216,0.06)"),   # Improving
                ( 80,100, 80,100,"rgba(180,35,24,0.06)"),   # Lagging
            ]:
                fig_rrg.add_shape(type="rect", x0=x0,x1=x1,y0=y0,y1=y1,
                                  fillcolor=fc, line_width=0, layer="below")
            fig_rrg.add_hline(y=100, line_color="#cbd5e1", line_width=1.5)
            fig_rrg.add_vline(x=100, line_color="#cbd5e1", line_width=1.5)
            # Quadrant text labels
            for tx,ty,lbl,tc in [
                (110,110,"Leading",   "#1f7a4f"),
                (110, 90,"Weakening", "#d97706"),
                ( 90,110,"Improving", "#1d4ed8"),
                ( 90, 90,"Lagging",   "#b42318"),
            ]:
                fig_rrg.add_annotation(x=tx, y=ty, text=lbl, showarrow=False,
                                       font=dict(size=9, color=tc), opacity=0.55)
            # Points
            for _, row_r in rrg_df.iterrows():
                t  = row_r["Ticker"]
                rx = float(row_r["RS_ratio"])
                ry = float(row_r["RS_mom"])
                tc = TICKER_COLORS.get(t, "#6b7280")
                fig_rrg.add_trace(go.Scatter(
                    x=[rx], y=[ry], mode="markers+text",
                    text=[t], textposition="top center",
                    textfont=dict(size=10, color=tc),
                    marker=dict(size=12, color=tc, opacity=0.9,
                                line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate=f"<b>{t}</b> {LABELS.get(t,'')}<br>"
                                  f"RS ratio: {rx:.1f}<br>RS mom: {ry:.1f}<extra></extra>",
                ))
            # Axis range from data
            all_x = rrg_df["RS_ratio"].dropna().values
            all_y = rrg_df["RS_mom"].dropna().values
            xlo = min(float(min(all_x))*0.96, 88); xhi = max(float(max(all_x))*1.03, 112)
            ylo = min(float(min(all_y))*0.96, 88); yhi = max(float(max(all_y))*1.03, 112)
            fig_rrg.update_layout(
                height=420, margin=dict(l=10, r=10, t=10, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(range=[xlo, xhi], showgrid=True, gridcolor="#f1f5f9",
                           title="RS level (vs 52w avg)"),
                yaxis=dict(range=[ylo, yhi], showgrid=True, gridcolor="#f1f5f9",
                           title="RS momentum (4w)"),
                hovermode="closest",
            )
            st.plotly_chart(fig_rrg, width='stretch')

st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COPULA RISK MATRIX
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Tail dependence matrix (Clayton λ_L)</div>",
                unsafe_allow_html=True)
    st.caption(
        "Lower tail dependence coefficient from a fitted Clayton copula on weekly returns. "
        "High λ_L (red) = both assets tend to crash together — avoid these as pair trades "
        "during drawdown periods. Low λ_L (green) = structurally decorrelated = safer pairs. "
        "Values on diagonal = 1.0 (self). Greyed = insufficient data.")

    if not copula_mat.empty:
        tickers_in_mat = [t for t in available if t in copula_mat.index]
        mat_display = copula_mat.loc[tickers_in_mat, tickers_in_mat].copy()

        # Build plotly heatmap
        z_vals = mat_display.values.tolist()
        x_lbl  = tickers_in_mat
        y_lbl  = tickers_in_mat

        fig_cop = go.Figure(go.Heatmap(
            z=z_vals, x=x_lbl, y=y_lbl,
            colorscale=[
                [0.0, "#dcfce7"], [0.15, "#bbf7d0"],
                [0.30, "#fef9c3"], [0.50, "#fee2e2"],
                [0.75, "#fca5a5"], [1.0, "#b42318"],
            ],
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" if not np.isnan(v) else "—"
                   for v in row] for row in z_vals],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{x} / %{y}</b><br>λ_L = %{z:.3f}<extra></extra>",
            colorbar=dict(title="λ_L", thickness=12, len=0.8),
        ))
        fig_cop.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, side="bottom"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_cop, width='stretch')

        # Flagged pairs
        flagged = pairs_df[pairs_df["regime_flag"] == True][
            ["a","b","lambda_l","current_rho","expected_rho","regime_delta","adj_score"]
        ].head(8)
        if not flagged.empty:
            st.markdown(
                "<div style='font-size:12px;font-weight:700;color:#d97706;"
                "margin-bottom:6px;'>⚠ Anomalous co-movement pairs</div>",
                unsafe_allow_html=True)
            for _, fr in flagged.iterrows():
                delta = fr["regime_delta"]
                d_color = "#b42318" if delta > 0 else "#1f7a4f"
                st.markdown(
                    f"<div class='me-li'>"
                    f"<div><div class='me-li-name'>{fr['a']} / {fr['b']}</div>"
                    f"<div class='me-li-sub'>"
                    f"Expected ρ {fmt_f(fr['expected_rho'])} · "
                    f"Current ρ {fmt_f(fr['current_rho'])} · "
                    f"λ_L {fmt_f(fr['lambda_l'],3)}</div></div>"
                    f"<span class='me-badge' style='background:{'#fee2e2' if delta>0 else '#dcfce7'};"
                    f"color:{d_color};'>"
                    f"Δρ {delta:+.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PAIR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Pair explorer</div>", unsafe_allow_html=True)
    st.caption("Select any two tickers to inspect the ratio, signals, and copula detail.")

    exp_a, exp_b, _ = st.columns([1, 1, 3], gap="medium")
    with exp_a:
        sel_a = st.selectbox("Ticker A (numerator)",   available, index=0, key="exp_a")
    with exp_b:
        sel_b = st.selectbox("Ticker B (denominator)", available,
                             index=min(2, len(available)-1), key="exp_b")

    if sel_a == sel_b:
        st.caption("Select two different tickers.")
    else:
        # Fetch pair data (from cache if exists, else recompute)
        a_ord, b_ord = (sel_a, sel_b) if (sel_a, sel_b) in ratio_store else \
                       ((sel_b, sel_a) if (sel_b, sel_a) in ratio_store else (sel_a, sel_b))
        ratio_s = ratio_store.get((a_ord, b_ord))
        if ratio_s is None and sel_a in px.columns and sel_b in px.columns:
            cidx = px[sel_a].dropna().index.intersection(px[sel_b].dropna().index)
            ratio_s = (px[sel_a].loc[cidx] / px[sel_b].loc[cidx]).dropna()

        pair_row = pairs_df[
            ((pairs_df["a"]==sel_a)&(pairs_df["b"]==sel_b)) |
            ((pairs_df["a"]==sel_b)&(pairs_df["b"]==sel_a))
        ]

        ec1, ec2 = st.columns([1.6, 1.0], gap="large")

        with ec1:
            if ratio_s is not None and not ratio_s.empty:
                rng_opts = list(HORIZONS.keys()) + ["5y"]
                rng_days = {"1w":7,"1m":35,"3m":110,"6m":185,"5y":2000}
                exp_range = st.selectbox("Range", rng_opts, index=2, key="exp_range")
                days_cut  = rng_days.get(exp_range, 110)
                r_view = ratio_s.loc[ratio_s.index >= ratio_s.index.max() -
                                     pd.Timedelta(days=days_cut)]
                # Dynamic axis
                lo_r = float(r_view.min()); hi_r = float(r_view.max())
                pad_r = max((hi_r - lo_r) * 0.12, hi_r * 0.01)
                ma63  = r_view.rolling(63, min_periods=10).mean()
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(
                    x=r_view.index, y=r_view.values,
                    mode="lines", name=f"{sel_a}/{sel_b}",
                    line=dict(color=TICKER_COLORS.get(sel_a, "#1d4ed8"), width=2),
                    fill="tozeroy", fillcolor=_hex_to_rgba(TICKER_COLORS.get(sel_a,'#1d4ed8'), 0.09),
                ))
                if not ma63.dropna().empty:
                    fig_e.add_trace(go.Scatter(
                        x=ma63.index, y=ma63.values,
                        mode="lines", name="63d MA",
                        line=dict(color="#94a3b8", width=1.5, dash="dash"),
                    ))
                fig_e.update_layout(
                    height=280, margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(range=[lo_r-pad_r, hi_r+pad_r],
                               showgrid=True, gridcolor="#f1f5f9"),
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                    legend=dict(orientation="h", y=1.05, x=0),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_e, width='stretch')
            else:
                st.caption("Ratio data unavailable for this pair.")

        with ec2:
            if not pair_row.empty:
                row = pair_row.iloc[0]
                s   = row["adj_score"]
                cc, cbg = conviction_palette(row["conviction"])
                lam  = row["lambda_l"]
                lc, lbg = lambda_color(lam if lam is not None else 0.0)

                st.markdown(
                    f"<div style='padding:12px;border-radius:12px;background:#f9fafb;"
                    f"border:1px solid rgba(0,0,0,0.07);'>"
                    f"<div style='font-size:24px;font-weight:900;color:{score_color(s)};'>"
                    f"{s:+.0f}</div>"
                    f"<div style='font-size:12px;font-weight:700;color:{cc};"
                    f"background:{cbg};display:inline-block;padding:2px 8px;"
                    f"border-radius:6px;margin:4px 0 10px;'>{row['conviction']}"
                    f" · {row['leader']} > {row['laggard']}</div>"
                    f"<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.40);"
                    f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:6px;'>"
                    f"Momentum signals</div>"
                    + _mini_bar("Mom z",   row["mom_z"])
                    + _mini_bar("Trend",   row["trend"],
                                fmt_fn=lambda v: "Above ↑" if float(v)>0 else "Below ↓")
                    + _mini_bar("Level z", row["level_z"], invert=True)
                    + _mini_bar("1m",      row["r1m"],
                                fmt_fn=lambda v: f"{float(v)*100:+.1f}%", cap=0.05)
                    + f"<div style='margin-top:10px;padding:8px;border-radius:8px;"
                      f"background:{lbg};'>"
                    + f"<div style='font-size:10px;font-weight:700;color:{lc};"
                      f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>"
                      f"Copula stats</div>"
                    + f"<div style='font-size:11px;display:grid;grid-template-columns:1fr 1fr;gap:4px;'>"
                    + f"<div><span style='color:rgba(0,0,0,0.40);'>λ_L</span> "
                      f"<span style='font-weight:800;color:{lc};'>{fmt_f(lam,3)}</span></div>"
                    + f"<div><span style='color:rgba(0,0,0,0.40);'>θ</span> "
                      f"<span style='font-weight:700;'>{fmt_f(row.get('theta'),2)}</span></div>"
                    + f"<div><span style='color:rgba(0,0,0,0.40);'>τ</span> "
                      f"<span style='font-weight:700;'>{fmt_f(row.get('tau'),2)}</span></div>"
                    + f"<div><span style='color:rgba(0,0,0,0.40);'>cur ρ</span> "
                      f"<span style='font-weight:700;'>{fmt_f(row.get('current_rho'),2)}</span></div>"
                    + f"<div><span style='color:rgba(0,0,0,0.40);'>exp ρ</span> "
                      f"<span style='font-weight:700;'>{fmt_f(row.get('expected_rho'),2)}</span></div>"
                    + f"<div><span style='color:rgba(0,0,0,0.40);'>penalty</span> "
                      f"<span style='font-weight:700;'>{fmt_f(row.get('penalty'),2)}×</span></div>"
                    + f"</div>"
                    + (f"<div style='font-size:10px;color:#d97706;margin-top:4px;'>"
                       f"⚠ Anomalous co-movement: Δρ {row['regime_delta']:+.2f}</div>"
                       if row.get("regime_flag") else "")
                    + f"</div></div>",
                    unsafe_allow_html=True)
            else:
                st.caption("No pre-computed score for this pair. Pair may have insufficient history.")

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)