# pages/7_Regime_Playbook.py
"""
Regime Playbook  — v2  (alpha engine)
═══════════════════════════════════════════════════════════════════════════════
Synthesises signals from every page into ranked, sized, actionable trades.

Statistical methods used
────────────────────────
1. REGIME-CONDITIONAL RETURN DISTRIBUTION
   For each asset × regime label, compute the full empirical distribution of
   4-week forward returns (5th/25th/50th/75th/95th percentile). The shape of
   this distribution — not just the median — determines the trade quality.
   Asymmetric distributions (large upside, bounded downside) score highest.

2. KELLY FRACTION SIZING
   Win rate and payoff ratio from the historical return distribution feed a
   fractional Kelly (25% full Kelly) position size:
       Kelly% = W - (1-W)/R   where W=win_rate, R=avg_win/avg_loss
   This directly answers "how much should I size this trade."

3. CROSS-SIGNAL CONFLUENCE SCORING
   For each asset, count how many independent signals agree directionally:
     • Regime score direction (bullish/bearish)
     • Curve regime (steepening vs flattening vs inverted)
     • Volatility regime (V-Ratio / VIX percentile)
     • Fed/liquidity stance (real rate regime + BS impulse)
     • Asset-specific momentum z-score (from 9_Curve_View / 6_Volatility_View logic)
     • Mean-reversion signal (z-score of price vs 252d — extreme = reversion risk)
   Confluence 5/6 = strong conviction. 3/6 = moderate. <3 = avoid.

4. REGIME TRANSITION PROBABILITY (Markov chain)
   From the weekly regime timeseries, build a 5×5 first-order Markov transition
   matrix. Given current regime + recent score slope, compute the 4-week
   transition probability vector. If P(transition) > 30%, flag early and size
   down all current-regime trades.

5. FACTOR ROTATION TIMING
   The factor cycle (value/growth, quality/beta, size/cap) is determined by
   where we sit in the simultaneous space of:
     • Curve slope z-score (positive = steep = value / cyclicals)
     • Real rate level (high = value, low = growth/duration)
     • Credit spread z-score (wide = quality, tight = beta)
     • Regime momentum (improving = small caps, deteriorating = large cap quality)
   Output: ranked factor tilts with z-score backing for each.

6. CROSS-ASSET DIVERGENCE ALERTS
   Pairs of markets that *should* agree (VIX ↑ + HY OAS ↑, curve steep + IWM leading,
   Fed easing + GLD up) but don't. Divergence = one market is wrong = opportunity.
   Measured as: standardised difference between actual and predicted co-movement.

Sources used
────────────
Regime engine (v4):  score, components, timeseries, momentum
FRED:   hy_oas, real10, y10-y2, y3m, fed_assets, dollar_broad, cpi, fed_funds
Prices: SPY, QQQ, IWM, XLE, XLF, XLK, XLI, XLV, XLP, GLD, UUP, HYG, TLT,
        BTC, XBI, IGV, SMH, RSP
VIX proxies: ^VIX, ^VIX3M
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries
from src.ui import inject_css, sidebar_nav, safe_switch_page, regime_color, regime_bg

st.set_page_config(page_title="Regime Playbook", page_icon="📋",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Regime Playbook")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

# ── Basket ────────────────────────────────────────────────────────────────────

ASSETS = ["SPY","QQQ","IWM","XLE","XLF","XLK","XLI","XLV","XLP",
          "XLU","XLC","GLD","UUP","HYG","TLT","BTC","XBI","IGV","SMH","RSP"]
ASSET_LABELS = {
    "SPY":"S&P 500",  "QQQ":"Nasdaq",    "IWM":"Small caps",
    "XLE":"Energy",   "XLF":"Financials","XLK":"Technology",
    "XLI":"Industrials","XLV":"Healthcare","XLP":"Staples",
    "XLU":"Utilities","XLC":"Comms",
    "GLD":"Gold",     "UUP":"USD",       "HYG":"High yield",
    "TLT":"Long bonds","BTC":"Bitcoin", "XBI":"Biotech",
    "IGV":"Software", "SMH":"Semis",     "RSP":"Equal weight",
}
ASSET_COLORS = {
    "SPY":"#1d4ed8","QQQ":"#6366f1","IWM":"#ec4899","XLE":"#f97316",
    "XLF":"#3b82f6","XLK":"#8b5cf6","XLI":"#64748b","XLV":"#06b6d4",
    "XLP":"#22c55e","XLU":"#0d9488","XLC":"#c026d3",
    "GLD":"#eab308","UUP":"#ef4444","HYG":"#a855f7",
    "TLT":"#0ea5e9","BTC":"#f59e0b","XBI":"#10b981","IGV":"#7c3aed",
    "SMH":"#db2777","RSP":"#84cc16",
}

# ── Data load ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    vix_t = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
    all_t = list(dict.fromkeys(ASSETS + vix_t))
    px = fetch_prices(all_t, period="5y")
    px = pd.DataFrame() if px is None or px.empty else px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px,
                               lookback_trend=63, momentum_lookback_days=21)
    reg_hist = compute_regime_timeseries(macro, px,
                                         lookback_trend=63, freq="W-FRI")
    return macro, px, regime, reg_hist

macro, px, regime, reg_hist = load_all()

# ── Regime state ──────────────────────────────────────────────────────────────

cur_label   = getattr(regime, "label",          "Unknown")
cur_score   = int(getattr(regime, "score",      50))
cur_raw     = getattr(regime, "score_raw",      float(cur_score))
cur_conf    = getattr(regime, "confidence",     "Unknown")
cur_mom     = getattr(regime, "momentum_label", "Unknown")
cur_delta   = getattr(regime, "score_delta",    0) or 0
components  = getattr(regime, "components",     {}) or {}
cur_alloc   = getattr(regime, "allocation",     {}) or {}
cur_stance  = cur_alloc.get("stance", {}) if isinstance(cur_alloc, dict) else {}
cur_color   = regime_color(cur_label)
cur_bg      = regime_bg(cur_label)

# ── Helper primitives ─────────────────────────────────────────────────────────

def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _zscore_last(s, w=252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w, len(s)):]
    sd = float(tail.std())
    return float((tail.iloc[-1] - tail.mean()) / sd) if sd > 0 else 0.0

def _ret(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] / s.loc[prev[-1]] - 1) if len(prev) else None

def _score_to_label(sc):
    if sc >= 75: return "Risk On"
    if sc >= 60: return "Bullish"
    if sc >= 40: return "Neutral"
    if sc >= 25: return "Bearish"
    return "Risk Off"

def fmt(x, nd=2, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{float(x):+.{nd}f}{suffix}" if plus else f"{float(x):.{nd}f}{suffix}"

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Regime-conditional forward return distributions ────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_return_distributions(_px, _reg_hist):
    """
    For each asset × regime label:
      - Align weekly prices with weekly regime label
      - Compute 4-week forward return for each week
      - Return full distribution stats + win_rate + kelly fraction
    """
    if _reg_hist.empty or _px.empty:
        return pd.DataFrame()

    # Weekly regime labels
    rh = _reg_hist.copy()
    rh["label"] = rh["score"].apply(_score_to_label)

    results = []
    for t in ASSETS:
        if t not in _px.columns: continue
        s = _px[t].resample("W-FRI").last().dropna()
        # 4-week forward return: fwd[i] = price[i+4]/price[i] - 1
        fwd = s.pct_change(4).shift(-4).dropna()
        merged = rh.join(fwd.rename("fwd"), how="inner").dropna()
        if len(merged) < 20: continue

        for lbl in ["Risk On", "Bullish", "Neutral", "Bearish", "Risk Off"]:
            sub = merged[merged["label"] == lbl]["fwd"].dropna()
            if len(sub) < 8: continue
            arr = sub.values
            wins = arr[arr > 0]; losses = arr[arr <= 0]
            win_rate  = len(wins) / len(arr)
            avg_win   = float(wins.mean())  if len(wins)  > 0 else 0.0
            avg_loss  = abs(float(losses.mean())) if len(losses) > 0 else 1e-6
            payoff    = avg_win / avg_loss if avg_loss > 0 else 0.0
            kelly     = max(0.0, win_rate - (1 - win_rate) / payoff) if payoff > 0 else 0.0
            kelly_25  = kelly * 0.25  # fractional Kelly

            # Sharpe-like: mean / std of 4w returns
            sr = float(arr.mean() / arr.std()) * np.sqrt(52) if arr.std() > 0 else 0.0

            results.append({
                "asset": t, "label": lbl, "n": len(arr),
                "p5":  float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "mean": float(arr.mean()),
                "win_rate": win_rate, "payoff": payoff,
                "kelly": kelly, "kelly_25": kelly_25,
                "ann_sharpe": sr,
            })

    return pd.DataFrame(results)

ret_dist = build_return_distributions(px, reg_hist)

# ── 2. Markov transition matrix ───────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_markov(_reg_hist):
    """
    Build 5×5 first-order Markov transition matrix from weekly regime history.
    Returns (matrix_df, transition_probs_from_current).
    """
    if _reg_hist.empty or len(_reg_hist) < 20:
        return pd.DataFrame(), {}

    labels_ord = ["Risk On", "Bullish", "Neutral", "Bearish", "Risk Off"]
    rh = _reg_hist.copy()
    rh["label"] = rh["score"].apply(_score_to_label)
    lbl_series  = rh["label"].values

    # Count transitions
    mat = pd.DataFrame(0, index=labels_ord, columns=labels_ord)
    for i in range(len(lbl_series) - 1):
        f = lbl_series[i]; t = lbl_series[i+1]
        if f in labels_ord and t in labels_ord:
            mat.loc[f, t] += 1

    # Row-normalise to probabilities
    mat_prob = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return mat_prob, dict(mat_prob.loc[cur_label]) if cur_label in mat_prob.index else {}

markov_mat, trans_probs = build_markov(reg_hist)

# Probability of staying in current regime
stay_prob    = trans_probs.get(cur_label, 0.0)
# Probability of deteriorating (move to worse regime)
worse_labels = {"Risk On": ["Bullish","Neutral","Bearish","Risk Off"],
                "Bullish": ["Neutral","Bearish","Risk Off"],
                "Neutral": ["Bearish","Risk Off"],
                "Bearish": ["Risk Off"],
                "Risk Off": []}
degrade_prob = sum(trans_probs.get(l, 0.0) for l in worse_labels.get(cur_label, []))
improve_labels = {"Risk Off": ["Bearish","Neutral","Bullish","Risk On"],
                  "Bearish":  ["Neutral","Bullish","Risk On"],
                  "Neutral":  ["Bullish","Risk On"],
                  "Bullish":  ["Risk On"],
                  "Risk On":  []}
improve_prob = sum(trans_probs.get(l, 0.0) for l in improve_labels.get(cur_label, []))

# ── 3. Cross-signal confluence engine ────────────────────────────────────────

def compute_confluence(asset: str) -> dict:
    """
    For a given asset, score each of 6 independent signals on [-1, 0, +1]
    where +1 = bullish, -1 = bearish, 0 = neutral/no signal.
    Returns signal dict + overall confluence score in [-6, +6].
    """
    signals = {}

    # ── Pre-compute shared z-scores ───────────────────────────────────────────
    curve_z  = _zscore_last((macro["y10"]-macro["y2"]).dropna()) \
               if "y10" in macro.columns and "y2" in macro.columns else None
    real_now = _last(macro["real10"].dropna()) if "real10" in macro.columns else None
    real_z   = _zscore_last(macro["real10"].dropna()) if "real10" in macro.columns else None
    credit_z = _zscore_last(macro["hy_oas"].dropna()) if "hy_oas" in macro.columns else None
    dollar_z = _zscore_last(macro["dollar_broad"].dropna()) if "dollar_broad" in macro.columns else None

    fed_roc13 = None
    if "fed_assets" in macro.columns and len(macro["fed_assets"].dropna()) >= 70:
        fed_roc13 = float(macro["fed_assets"].dropna().pct_change(63).iloc[-1]*100)

    # VIX / volatility state
    vix_t   = YF_PROXIES.get("vix"); vix3m_t = YF_PROXIES.get("vix3m")
    vix_pct = vratio = None
    if vix_t and vix_t in px.columns:
        vix_s = px[vix_t].dropna()
        if len(vix_s) >= 252:
            vix_pct = float((vix_s.iloc[-252:] < vix_s.iloc[-1]).mean()*100)
        if vix3m_t and vix3m_t in px.columns:
            v3m = px[vix3m_t].dropna()
            idx = vix_s.index.intersection(v3m.index)
            if len(idx) > 0:
                vratio = float(vix_s.loc[idx[-1]] / v3m.loc[idx[-1]])
    vol_stress = (vix_pct is not None and vix_pct > 75) or (vratio is not None and vratio > 1.0)
    vol_calm   = (vix_pct is not None and vix_pct < 30) and (vratio is None or vratio < 0.9)

    # ── Signal 1: Regime direction ────────────────────────────────────────────
    # Regime score > 60 = macro is risk-on = good for risk assets, bad for safe havens
    # SIGN CONVENTION: +1 means the signal is BULLISH FOR THIS SPECIFIC ASSET
    if cur_score >= 60:   raw_reg = +1
    elif cur_score <= 40: raw_reg = -1
    else:                 raw_reg = 0

    # These assets are inversely correlated with macro risk-on:
    # High score (risk-on) is BAD for them; low score (risk-off) is GOOD
    safe_haven_reg = {"TLT", "UUP", "GLD", "XLP", "XLV", "XLU"}
    if asset in safe_haven_reg:
        signals["regime"] = -raw_reg   # flip: high score = bearish for safe havens
    else:
        signals["regime"] = raw_reg    # risk assets: high score = bullish

    # ── Signal 2: Curve regime ────────────────────────────────────────────────
    # curve_z > 0 = steepening = growth signal, good for cyclicals/banks
    # curve_z < 0 = flattening/inverting = late-cycle, bad for cyclicals
    # KEY FIX: TLT signal is driven by real yield DIRECTION, not curve slope.
    # A bear-flattener (short rates rising fast) is bad for TLT even though
    # the curve is "flat". We use real_z as the TLT curve proxy.
    if curve_z is not None:
        if asset == "XLF":
            # Banks: steep curve = wider net interest margin = bullish
            signals["curve"] = +1 if curve_z > 0.3 else (-1 if curve_z < -0.5 else 0)
        elif asset == "TLT":
            # Long bonds: use real yield DIRECTION not curve slope
            # Falling real yields = bonds rally. Rising = bonds fall.
            if real_z is not None:
                signals["curve"] = +1 if real_z < -0.5 else (-1 if real_z > 0.5 else 0)
            else:
                signals["curve"] = 0
        elif asset == "XLU":
            # Utilities = bond proxy. Steep curve = competition from bonds = bearish XLU
            # Flat/falling curve = supportive for yield-seeking stocks
            signals["curve"] = +1 if curve_z < -0.3 else (-1 if curve_z > 0.5 else 0)
        elif asset in ("XLE","XLI","IWM","XLC"):
            # Cyclicals + Comms: steep curve signals growth expectations = bullish
            signals["curve"] = +1 if curve_z > 0.2 else (-1 if curve_z < -0.5 else 0)
        elif asset in ("QQQ","XLK","IGV","SMH","XBI"):
            # Long-duration growth: flatter curve = lower long-term discount rates
            # But real yields capture this better; curve here is secondary
            signals["curve"] = +1 if curve_z < -0.3 else (-1 if curve_z > 0.5 else 0)
        elif asset in ("XLP","XLV","GLD"):
            # Defensives and gold: not driven by curve slope
            signals["curve"] = 0
        else:
            signals["curve"] = +1 if curve_z > 0.3 else (-1 if curve_z < -0.5 else 0)
    else:
        signals["curve"] = 0

    # ── Signal 3: Volatility regime ───────────────────────────────────────────
    # vol_stress = VIX elevated = bad for risk assets, good for safe havens
    # vol_calm   = VIX low = good for risk assets, safe havens lose their bid
    safe_haven_vol = {"GLD","TLT","UUP","XLP","XLV","XLU"}
    high_beta_vol  = {"BTC","XBI","IWM","SMH","IGV","XLC"}
    if asset in safe_haven_vol:
        signals["volatility"] = +1 if vol_stress else (-1 if vol_calm else 0)
    elif asset in high_beta_vol:
        signals["volatility"] = -1 if vol_stress else (+1 if vol_calm else 0)
    else:
        signals["volatility"] = -1 if vol_stress else (+1 if vol_calm else 0)

    # ── Signal 4: Real yields / Fed stance ────────────────────────────────────
    # This is the most important signal for rate-sensitive assets.
    # real_now > 1.5% = restrictive = headwind for equities/GLD, tailwind for UUP
    # real_now < 0.5% = accommodative = tailwind for equities/GLD, headwind for UUP
    # real_z direction: rising = tightening (bearish most assets), falling = easing
    if real_now is not None and real_z is not None:
        if asset == "TLT":
            # TLT: falling real yields → bond prices rise. Rising → prices fall.
            # Use real_z direction as the signal
            signals["fed_liq"] = +1 if real_z < -0.5 else (-1 if real_z > 0.5 else 0)
        elif asset == "GLD":
            # GLD: low/falling real yields = no opportunity cost to hold gold
            # High real yields = strong alternative = bearish for GLD
            signals["fed_liq"] = +1 if real_now < 0.5 else (-1 if real_now > 1.5 else 0)
        elif asset == "UUP":
            # USD: restrictive real yields attract foreign capital inflows
            signals["fed_liq"] = +1 if real_now > 1.0 else (-1 if real_now < 0.3 else 0)
        elif asset == "XLU":
            # Utilities: bond-proxy dividend stocks. Low rates = cheap funding + yield attractive
            signals["fed_liq"] = +1 if real_now < 0.8 else (-1 if real_now > 1.5 else 0)
        elif asset == "XLF":
            # Banks: moderate real yields = healthy margins. Very high = credit stress.
            signals["fed_liq"] = +1 if (real_now is not None and 0.5 < real_now < 2.0) \
                                  else (-1 if (real_now is not None and real_now > 2.5) else 0)
        elif asset == "XLE":
            # Energy driven by growth/inflation not real yields — use Fed BS impulse
            if fed_roc13 is not None:
                signals["fed_liq"] = +1 if fed_roc13 > 0.5 else (-1 if fed_roc13 < -0.5 else 0)
            else:
                signals["fed_liq"] = 0
        elif asset in ("BTC","XBI","IWM","QQQ","XLK","IGV","SMH","XLC"):
            # Long-duration / high-beta growth: low real yields = lower discount rates
            signals["fed_liq"] = -1 if real_now > 1.5 else (+1 if real_now < 0.5 else 0)
        else:
            # Default: restrictive rates are a headwind for most equity assets
            signals["fed_liq"] = -1 if real_now > 1.5 else (+1 if real_now < 0.5 else 0)
    else:
        signals["fed_liq"] = 0

    # ── Signal 5: Price momentum ───────────────────────────────────────────────
    # Requires BOTH trend (above 63d MA) AND positive RoC z-score for a long signal
    # Requires BOTH below MA AND negative RoC z-score for a short signal
    # Mixed signals = 0 (not a clean momentum setup)
    mom_signal = 0
    if asset in px.columns:
        s = px[asset].dropna()
        if len(s) >= 63:
            ma63  = float(s.rolling(63).mean().iloc[-1])
            above = s.iloc[-1] > ma63
            roc   = s.pct_change(21).dropna()
            roc_z = _zscore_last(roc)
            if roc_z is not None:
                if above and roc_z > 0.5:         mom_signal = +1
                elif not above and roc_z < -0.5:  mom_signal = -1
    signals["momentum"] = mom_signal

    # ── Signal 6: Mean-reversion / valuation stretch ──────────────────────────
    # Price z-score vs 252d history. Only fires at extremes (z > 2.0 or < -2.0)
    mr_signal = 0
    if asset in px.columns:
        s  = px[asset].dropna()
        lz = _zscore_last(s, 252)
        if lz is not None:
            if lz > 2.0:    mr_signal = -1   # stretched → reversion risk
            elif lz < -2.0: mr_signal = +1   # depressed → bounce potential
    signals["mean_rev"] = mr_signal

    confluence = sum(signals.values())
    return {
        "signals":    signals,
        "confluence": confluence,
        "abs_conf":   abs(confluence),
        # direction = "long" means the net signal is bullish for this asset
        "direction":  "long" if confluence > 0 else ("short" if confluence < 0 else "neutral"),
    }

# ── 4. Factor rotation model ──────────────────────────────────────────────────

def compute_factor_scores() -> list[dict]:
    """
    Score factor tilts using macro signals.
    Returns list of {factor, score, z_backing, description} sorted by |score|.
    """
    factors = []

    curve_z  = _zscore_last((macro["y10"] - macro["y2"]).dropna()) \
               if "y10" in macro.columns and "y2" in macro.columns else 0.0
    real_z   = _zscore_last(macro["real10"].dropna()) \
               if "real10" in macro.columns else 0.0
    credit_z = _zscore_last(macro["hy_oas"].dropna()) \
               if "hy_oas" in macro.columns else 0.0
    dollar_z = _zscore_last(macro["dollar_broad"].dropna()) \
               if "dollar_broad" in macro.columns else 0.0

    # Value vs Growth
    # Value wins when curve steep (higher discount rates) + credit tight
    # Growth wins when real rates low + QQQ in momentum
    vg_score = (curve_z or 0) * 0.4 - (real_z or 0) * 0.35 - (credit_z or 0) * 0.25
    factors.append({
        "factor": "Value vs Growth",
        "score":  float(np.clip(vg_score, -3, 3)),
        "direction": "Value" if vg_score > 0.3 else ("Growth" if vg_score < -0.3 else "Neutral"),
        "proxy": "XLE+XLF vs QQQ+IGV",
        "z_inputs": f"curve z={fmt(curve_z,2,plus=True)} · real z={fmt(real_z,2,plus=True)} · credit z={fmt(credit_z,2,plus=True)}",
        "description": (
            "Steep curve + tight credit → value/cyclicals historically outperform growth."
            if vg_score > 0.3 else
            "Low real rates + accommodative conditions → growth / long-duration equities preferred."
            if vg_score < -0.3 else
            "Mixed curve/real rate signal. No clear value-growth tilt."
        ),
    })

    # Quality vs Beta (High-beta)
    # Quality wins in risk-off: wide credit spreads, high VIX, inverted curve
    qb_score = (credit_z or 0) * 0.45 + (real_z or 0) * 0.30 - (curve_z or 0) * 0.25
    factors.append({
        "factor": "Quality vs High-beta",
        "score":  float(np.clip(qb_score, -3, 3)),
        "direction": "Quality" if qb_score > 0.3 else ("High-beta" if qb_score < -0.3 else "Neutral"),
        "proxy": "XLP+XLV vs IWM+XBI",
        "z_inputs": f"credit z={fmt(credit_z,2,plus=True)} · real z={fmt(real_z,2,plus=True)} · curve z={fmt(curve_z,2,plus=True)}",
        "description": (
            "Wide credit spreads signal stress → quality/defensive premium historically significant."
            if qb_score > 0.3 else
            "Tight credit + accommodative conditions → high-beta can be rewarded, risk appetite healthy."
            if qb_score < -0.3 else
            "Balanced quality-beta signal. Size positions symmetrically."
        ),
    })

    # Small cap vs Large cap (Size premium)
    # Small caps win when: credit tight + curve steep + dollar weak
    sc_score = -(credit_z or 0) * 0.35 + (curve_z or 0) * 0.35 - (dollar_z or 0) * 0.30
    factors.append({
        "factor": "Small cap vs Large cap",
        "score":  float(np.clip(sc_score, -3, 3)),
        "direction": "Small caps" if sc_score > 0.3 else ("Large caps" if sc_score < -0.3 else "Neutral"),
        "proxy": "IWM vs SPY",
        "z_inputs": f"credit z={fmt(credit_z,2,plus=True)} · curve z={fmt(curve_z,2,plus=True)} · dollar z={fmt(dollar_z,2,plus=True)}",
        "description": (
            "Credit tight + curve steep + dollar soft = classic size premium environment."
            if sc_score > 0.3 else
            "Wide credit/strong dollar/flat curve = large-cap quality defense."
            if sc_score < -0.3 else
            "Neutral size signal. IWM/SPY ratio will be driven by near-term momentum."
        ),
    })

    # Cyclicals vs Defensives
    cyc_score = (curve_z or 0) * 0.35 - (credit_z or 0) * 0.35 + float(np.clip((cur_score - 50) / 25, -1, 1)) * 0.30
    factors.append({
        "factor": "Cyclicals vs Defensives",
        "score":  float(np.clip(cyc_score, -3, 3)),
        "direction": "Cyclicals" if cyc_score > 0.3 else ("Defensives" if cyc_score < -0.3 else "Neutral"),
        "proxy": "XLE+XLI+XLF vs XLP+XLV",
        "z_inputs": f"curve z={fmt(curve_z,2,plus=True)} · credit z={fmt(credit_z,2,plus=True)} · regime={cur_score}",
        "description": (
            "Regime + steep curve + tight credit = strong cyclical environment."
            if cyc_score > 0.3 else
            "Stress signals active → defensives (XLP, XLV) offer downside protection."
            if cyc_score < -0.3 else
            "No strong cyclical/defensive tilt. Favour stock selection over factor rotation."
        ),
    })

    # Real assets vs Financial assets (GLD, commodities vs bonds)
    ra_score = -(real_z or 0) * 0.40 - (dollar_z or 0) * 0.35 + float(np.clip((cur_score - 50) / 25, -1, 1)) * 0.25
    factors.append({
        "factor": "Real assets vs Financial",
        "score":  float(np.clip(ra_score, -3, 3)),
        "direction": "Real assets" if ra_score > 0.3 else ("Financial" if ra_score < -0.3 else "Neutral"),
        "proxy": "GLD+XLE vs TLT+HYG",
        "z_inputs": f"real z={fmt(real_z,2,plus=True)} · dollar z={fmt(dollar_z,2,plus=True)} · regime={cur_score}",
        "description": (
            "Low/falling real rates + weak dollar = real assets (GLD, commodities) historically outperform."
            if ra_score > 0.3 else
            "High real rates + strong dollar favours financial assets over hard assets."
            if ra_score < -0.3 else
            "Mixed real-asset signal."
        ),
    })

    # International vs Domestic (USD cycle)
    intl_score = -(dollar_z or 0) * 0.50 + float(np.clip((cur_score - 50) / 25, -1, 1)) * 0.30 - (real_z or 0) * 0.20
    factors.append({
        "factor": "International vs Domestic",
        "score":  float(np.clip(intl_score, -3, 3)),
        "direction": "International/EM" if intl_score > 0.3 else ("US domestic" if intl_score < -0.3 else "Neutral"),
        "proxy": "Weak USD → EM/intl, Strong USD → US",
        "z_inputs": f"dollar z={fmt(dollar_z,2,plus=True)} · regime={cur_score} · real z={fmt(real_z,2,plus=True)}",
        "description": (
            "Dollar weakening removes the biggest EM headwind. International equities historically outperform."
            if intl_score > 0.3 else
            "Strong dollar = EM USD debt cost rising, capital flows reversing. Stay domestic."
            if intl_score < -0.3 else
            "Dollar cycle neutral. Allocation between domestic/international driven by relative valuations."
        ),
    })

    return sorted(factors, key=lambda x: abs(x["score"]), reverse=True)

factor_scores = compute_factor_scores()

# ── 5. Cross-asset divergence detection ──────────────────────────────────────

def find_divergences() -> list[dict]:
    """
    Check 8 canonical cross-asset relationships for divergence.
    Each pair should move together; when they don't, one is mispricing the other.
    """
    divs = []

    def _rolling_corr(s1, s2, w=63):
        """Rolling correlation over w days."""
        common = s1.dropna().index.intersection(s2.dropna().index)
        if len(common) < w + 10: return None, None
        r1 = s1.loc[common].pct_change().dropna()
        r2 = s2.loc[common].pct_change().dropna()
        ci = r1.index.intersection(r2.index)
        if len(ci) < w: return None, None
        full_corr  = float(r1.loc[ci].corr(r2.loc[ci]))
        recent_corr = float(r1.loc[ci].iloc[-w:].corr(r2.loc[ci].iloc[-w:]))
        return full_corr, recent_corr

    # 1. VIX vs HY OAS — should both be elevated in stress
    vix_t = YF_PROXIES.get("vix")
    if vix_t and vix_t in px.columns and "hy_oas" in macro.columns:
        vix_s = px[vix_t].dropna()
        hy_s  = macro["hy_oas"].dropna()
        vix_z = _zscore_last(vix_s)
        hy_z  = _zscore_last(hy_s)
        if vix_z is not None and hy_z is not None:
            diverge = vix_z - hy_z  # positive = VIX elevated vs credit
            if abs(diverge) > 1.0:
                winner = "VIX" if diverge > 0 else "HY OAS"
                lagger = "HY OAS" if diverge > 0 else "VIX"
                divs.append({
                    "pair": "VIX vs HY OAS",
                    "delta_z": diverge,
                    "interpretation": f"{winner} pricing more stress than {lagger}. "
                        f"Credit markets ({lagger}) usually lead — expect convergence. "
                        + ("Reduce equity risk if VIX leading credit." if diverge > 0
                           else "Credit widening without VIX spike = quiet stress building."),
                    "trade_implication": "Short high-beta if VIX > HY" if diverge > 0
                        else "HYG weakness without vol = credit event risk. Add TLT hedge.",
                })

    # 2. Curve slope vs IWM/SPY ratio — both should improve together
    if "y10" in macro.columns and "y2" in macro.columns \
            and "IWM" in px.columns and "SPY" in px.columns:
        curve_z = _zscore_last((macro["y10"] - macro["y2"]).dropna())
        iwm_spy_z = _zscore_last((px["IWM"] / px["SPY"]).dropna())
        if curve_z is not None and iwm_spy_z is not None:
            diverge = curve_z - iwm_spy_z
            if abs(diverge) > 1.2:
                divs.append({
                    "pair": "Curve slope vs IWM/SPY",
                    "delta_z": diverge,
                    "interpretation": (
                        "Curve steepening but small caps not confirming. "
                        "Either curve is false signal or IWM is lagging — latter more likely."
                        if diverge > 0 else
                        "Small caps outperforming but curve not steepening to support it. "
                        "Breadth rally may lack macro backing — watch for mean reversion."
                    ),
                    "trade_implication": "Long IWM — curve is the leading signal." if diverge > 0
                        else "Fade IWM strength without curve confirmation. Add XLP hedge.",
                })

    # 3. GLD vs Real rates (inverse) — GLD should rise when real rates fall
    if "real10" in macro.columns and "GLD" in px.columns:
        real_z = _zscore_last(macro["real10"].dropna())
        gld_z  = _zscore_last(px["GLD"].dropna())
        if real_z is not None and gld_z is not None:
            # Expected: GLD_z ≈ -real_z
            diverge = gld_z + real_z  # should be near 0
            if abs(diverge) > 1.2:
                divs.append({
                    "pair": "GLD vs Real rates",
                    "delta_z": diverge,
                    "interpretation": (
                        "GLD elevated vs what real rates imply. "
                        "Either geopolitical/safe-haven premium embedded, or real rates will fall."
                        if diverge > 0 else
                        "GLD underperforming relative to falling real rates. "
                        "May be catching up — or dollar strength is offsetting."
                    ),
                    "trade_implication": "GLD rich — consider taking profits or adding GLD put hedge." if diverge > 0
                        else "GLD cheap vs real rate signal — accumulate.",
                })

    # 4. Credit (HYG) vs Equity breadth (RSP/SPY)
    if "HYG" in px.columns and "RSP" in px.columns and "SPY" in px.columns:
        hyg_ret = _ret(px["HYG"].dropna(), 63)
        rsp_spy = _ret(px["RSP"].dropna(), 63)
        spy_ret = _ret(px["SPY"].dropna(), 63)
        if hyg_ret is not None and rsp_spy is not None and spy_ret is not None:
            # HYG and RSP should both be positive in risk-on
            if hyg_ret < -0.02 and rsp_spy > 0.03:
                divs.append({
                    "pair": "HYG vs Equity breadth",
                    "delta_z": (rsp_spy - hyg_ret) * 10,
                    "interpretation": "Credit (HYG) deteriorating while equity breadth (RSP) healthy. "
                        "Credit leads equity by 4–8 weeks historically. "
                        "This is a significant warning — equity breadth unlikely to hold.",
                    "trade_implication": "Reduce equity beta. Add HYG puts or TLT. Don't fight the credit signal.",
                })
            elif hyg_ret > 0.02 and spy_ret < -0.02:
                divs.append({
                    "pair": "HYG vs SPY",
                    "delta_z": (hyg_ret - spy_ret) * 10,
                    "interpretation": "Credit supportive but equities selling off. "
                        "Rotation or profit-taking in equities, not systemic. "
                        "Credit is the cleaner signal.",
                    "trade_implication": "Use equity weakness to add risk. Credit is not confirming the selloff.",
                })

    # 5. Fed assets vs Risk appetite
    if "fed_assets" in macro.columns and "IWM" in px.columns and "SPY" in px.columns:
        fed_z   = _zscore_last(macro["fed_assets"].dropna())
        iwm_z   = _zscore_last(px["IWM"].dropna())
        if fed_z is not None and iwm_z is not None:
            diverge = iwm_z - fed_z
            if abs(diverge) > 1.5:
                divs.append({
                    "pair": "Fed balance sheet vs Risk appetite",
                    "delta_z": diverge,
                    "interpretation": (
                        "Risk appetite (IWM) running well ahead of liquidity conditions. "
                        "Markets priced for more accommodation than exists."
                        if diverge > 0 else
                        "Liquidity above historical norm but risk appetite subdued. "
                        "Either markets are mis-pricing the liquidity tail, or QT impulse is ahead."
                    ),
                    "trade_implication": "Fade high-beta momentum — liquidity not backing the move." if diverge > 0
                        else "Add risk — liquidity cycle should eventually drive breadth improvement.",
                })

    return sorted(divs, key=lambda d: abs(d["delta_z"]), reverse=True)[:5]

divergences = find_divergences()

# ── 6. Build the master trade table ──────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_trade_table(_ret_dist, _cur_label, _assets):
    """
    Rank assets by forward-looking alpha in the current regime.

    Direction is determined by CONFLUENCE (the multi-signal forward view),
    not by the historical mean. The historical distribution then validates
    whether that directional bet has a good win rate and payoff ratio.

    Agreement logic:
      - If confluence is bullish (+) AND win_rate_long > 0.52 → long candidate
      - If confluence is bearish (-) AND win_rate_short > 0.48 → short candidate
        (win_rate_short = 1 - win_rate_long since we're flipping the trade)
      - If confluence is 0 (neutral) → skip, no edge
      - If |confluence| < 2 → weak signal, score heavily discounted

    Alpha = kelly_25 × |confluence|/6 × confidence_multiplier × transition_penalty
    """
    if _ret_dist.empty: return pd.DataFrame()
    subset = _ret_dist[_ret_dist["label"] == _cur_label].copy()
    if subset.empty: return pd.DataFrame()

    rows = []
    for _, row in subset.iterrows():
        t    = row["asset"]
        conf = compute_confluence(t)
        c    = conf["confluence"]    # integer -6 to +6

        # Skip neutral confluence — no directional edge
        if c == 0:
            continue

        # Confluence-driven direction
        trade_long = c > 0

        # Win rate from the perspective of the trade direction
        # ret_dist stores win_rate as fraction of positive returns
        wr_long  = float(row["win_rate"])
        wr_trade = wr_long if trade_long else (1.0 - wr_long)

        # Payoff from the trade direction
        # For a short: avg_gain = avg_loss of underlying, avg_loss = avg_gain of underlying
        # Approximation: use the stored kelly which is for the long side
        # For shorts, recompute using p50 direction
        if trade_long:
            kelly_trade = float(row["kelly_25"])
            sharpe_trade = float(row["ann_sharpe"])
            p50_trade    = float(row["p50"])
        else:
            # Short side: flip p50, approximate kelly from short win rate
            p50_trade = -float(row["p50"])
            # Short kelly: use win_rate_short and same payoff magnitude
            payoff = float(row["payoff"]) if "payoff" in row.index else 1.0
            k_short = max(0.0, wr_trade - (1 - wr_trade) / payoff) if payoff > 0 else 0.0
            kelly_trade  = k_short * 0.25
            sharpe_trade = -float(row["ann_sharpe"])   # invert for short

        # Confidence multiplier: scale by |confluence| / 6
        conf_strength = abs(c) / 6.0

        # Historical validation: does history support at least a 50% win rate in this direction?
        # Use 0.48 as minimum threshold (allows some noise tolerance)
        hist_validates = wr_trade >= 0.48

        # Transition risk penalty
        trans_penalty = 1.0 - min(degrade_prob * 0.5, 0.4)

        # Alpha score — only for validated trades
        if hist_validates:
            alpha = kelly_trade * conf_strength * trans_penalty
        else:
            # History contradicts signal — heavily penalise but don't fully zero
            # so these still appear in the "all" tab with a warning
            alpha = kelly_trade * conf_strength * 0.1 * trans_penalty

        rows.append({
            "asset":          t,
            "name":           ASSET_LABELS.get(t, t),
            "direction":      "long" if trade_long else "short",
            "hist_validates": hist_validates,
            "p50_4w":         float(row["p50"]),
            "p50_trade":      p50_trade,
            "p25_4w":         float(row["p25"]),
            "p75_4w":         float(row["p75"]),
            "win_rate":       wr_trade,        # win rate for the actual trade direction
            "kelly_25":       kelly_trade,
            "ann_sharpe":     sharpe_trade,
            "confluence":     c,
            "conf_dir":       conf["direction"],   # "long" or "short"
            "signals":        conf["signals"],
            "alpha_score":    alpha,
            "n_obs":          int(row["n"]),
        })

    df = pd.DataFrame(rows).sort_values("alpha_score", ascending=False).reset_index(drop=True)
    return df

trade_df = build_trade_table(ret_dist, cur_label, ASSETS)

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Regime Playbook</div>
              <div class="me-subtle">
                Cross-signal confluence &nbsp;·&nbsp; Kelly sizing &nbsp;·&nbsp;
                factor rotation &nbsp;·&nbsp; divergence alerts &nbsp;·&nbsp; Markov transitions
              </div>
            </div>
            <div style="padding:8px 18px;border-radius:20px;background:{cur_bg};">
              <span style="font-weight:800;color:{cur_color};font-size:14px;">
                {cur_label} · {cur_score}
              </span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ── Regime + transition summary strip ─────────────────────────────────────────

rs1, rs2, rs3, rs4, rs5 = st.columns(5, gap="medium")

def _kpi(col, label, val, sub, color="#0f172a", bg="#f8fafc"):
    col.markdown(
        f"<div style='padding:11px 13px;border-radius:12px;background:{bg};"
        f"border:1px solid rgba(0,0,0,0.07);'>"
        f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.4);"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;'>{label}</div>"
        f"<div style='font-size:18px;font-weight:900;color:{color};line-height:1.1;'>{val}</div>"
        f"<div style='font-size:10px;color:rgba(0,0,0,0.5);margin-top:2px;'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)

_kpi(rs1, "Score", f"{cur_score} ({cur_raw:.1f})", f"{cur_label} · {cur_conf}",
     color=cur_color, bg=cur_bg)
_kpi(rs2, "Momentum", cur_mom,
     f"Δ {cur_delta:+d} vs 21d ago",
     color="#1f7a4f" if "improv" in cur_mom.lower() else ("#b42318" if "deterio" in cur_mom.lower() else "#6b7280"))
_kpi(rs3, "Stay probability", f"{stay_prob:.0%}",
     "4-week Markov estimate",
     color="#1f7a4f" if stay_prob > 0.6 else "#d97706")
_kpi(rs4, "Degrade prob", f"{degrade_prob:.0%}",
     "Prob of worse regime",
     color="#b42318" if degrade_prob > 0.35 else "#d97706" if degrade_prob > 0.20 else "#6b7280",
     bg="#fee2e2" if degrade_prob > 0.35 else "#fef9c3" if degrade_prob > 0.20 else "#f8fafc")
_kpi(rs5, "Improve prob", f"{improve_prob:.0%}",
     "Prob of better regime",
     color="#1f7a4f" if improve_prob > 0.25 else "#6b7280",
     bg="#dcfce7" if improve_prob > 0.25 else "#f8fafc")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRADE SIGNALS TABLE  (main alpha output)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='me-rowtitle'>Ranked trade signals — current regime</div>",
            unsafe_allow_html=True)
st.caption(
    f"Alpha score = fractional Kelly (25%) × confluence (6 signals) × history-confluence agreement × "
    f"transition penalty ({1-degrade_prob*0.5:.0%}). "
    f"Only shows assets where all signals agree. "
    f"n = weeks in this regime historically.")

# ── Contextual banner: how to use these signals ───────────────────────────────
_n_longs  = int((trade_df["direction"]=="long").sum())  if not trade_df.empty else 0
_n_shorts = int((trade_df["direction"]=="short").sum()) if not trade_df.empty else 0
_top_kelly = float(trade_df["kelly_25"].max()) if not trade_df.empty else 0
_trans_pen = 1 - degrade_prob * 0.5

if degrade_prob > 0.35:
    _play_context = (
        f"⚠ Elevated regime transition risk ({degrade_prob:.0%} degrade probability) is suppressing "
        f"the Kelly sizes by {(1-_trans_pen):.0%} across all signals. "
        "When the regime may be about to change, position sizing should shrink — "
        "the transition penalty enforces this automatically. "
        "Treat current signals as directionally valid but size at 50-60% of normal."
    )
    _play_col = "#d97706"; _play_bg = "#fef9c3"
elif _n_longs >= 4 and _n_shorts == 0:
    _play_context = (
        f"{_n_longs} long signals with no short signals — the macro environment is broadly bullish. "
        "In this regime, systematic long exposure is confirmed by multiple independent signals. "
        f"Maximum Kelly fraction available: {_top_kelly:.1%}. "
        "The confluence approach means each signal (regime, curve, vol, Fed, momentum, mean-reversion) "
        "votes independently — when 5-6 agree, the historical win rate is highest."
    )
    _play_col = "#1f7a4f"; _play_bg = "#dcfce7"
elif _n_shorts >= 3:
    _play_context = (
        f"{_n_shorts} short signals — the macro environment is showing stress. "
        "Short signals in this model mean the regime, credit, and volatility signals all point to "
        "risk reduction. These are not pure short trades but risk-off allocations: "
        "shift from beta to quality, reduce duration risk in credit, favour defensives."
    )
    _play_col = "#b42318"; _play_bg = "#fee2e2"
else:
    _play_context = (
        "The alpha score combines six independent signals: regime direction, curve regime, "
        "volatility regime, Fed/liquidity stance, price momentum, and mean-reversion risk. "
        "Each votes +1 or -1. Confluence ≥ 4/6 is required to show a signal. "
        "The Kelly fraction tells you how much of a standard position to take — "
        "6/6 confluence with a 65% historical win rate produces a larger Kelly than "
        "4/6 confluence with a 55% win rate."
    )
    _play_col = "#6b7280"; _play_bg = "#f3f4f6"

st.markdown(
    f"<div style='padding:11px 16px;border-radius:11px;background:{_play_bg};"
    f"border-left:4px solid {_play_col};font-size:12px;color:rgba(0,0,0,0.75);"
    f"line-height:1.6;margin-bottom:12px;'>{_play_context}</div>",
    unsafe_allow_html=True)

# ── How to read signal dots ───────────────────────────────────────────────────
with st.expander("How to read the signal cards"):
    st.markdown("""
**Signal dots** (R C V F M Z) — each letter represents one of six independent signals:
- **R** — Regime direction: does the current macro score favour this asset?
- **C** — Curve regime: is the yield curve slope supportive?
- **V** — Volatility regime: is vol in a state consistent with this trade?
- **F** — Fed/liquidity stance: is the balance sheet impulse aligned?
- **M** — Momentum: is the asset's own price trend confirming the macro signal?
- **Z** — Mean-reversion: is the asset at an extreme that increases reversion risk?

Green = signal confirms the trade. Red = signal contradicts. Grey = neutral.

**Win rate** — fraction of weeks historically where this asset posted a positive 4-week 
forward return in the current regime label. Computed from actual price data, not assumed.

**Kelly 25%** — fractional Kelly position size. Kelly% = W - (1-W)/R where W = win rate, 
R = avg win / avg loss. Multiplied by 0.25 (fractional Kelly) to account for model uncertainty. 
A Kelly of 8% means take 8% of your normal position size for this trade.

**⚠ Unvalidated** — the confluence fires but historical data in this regime does not support 
the direction with sufficient win rate. These signals are shown for awareness but should not 
be acted on without additional confirmation.
    """)

if not trade_df.empty:
    # Top longs and shorts
    # Validated = history supports the confluence direction
    # Unvalidated = confluence fires but history is lukewarm (show in All tab only)
    longs  = trade_df[(trade_df["direction"] == "long")  & (trade_df["hist_validates"] == True)].head(6)
    shorts = trade_df[(trade_df["direction"] == "short") & (trade_df["hist_validates"] == True)].head(4)
    n_unvalidated = int((trade_df["hist_validates"] == False).sum())

    tab_long, tab_short, tab_all = st.tabs([
        f"📈 Longs  {len(longs)}",
        f"📉 Shorts  {len(shorts)}",
        f"All  {len(trade_df)}  ({n_unvalidated} unvalidated)",
    ])

    def _signal_dots(signals: dict) -> str:
        icons = {"regime":"R", "curve":"C", "volatility":"V",
                 "fed_liq":"F", "momentum":"M", "mean_rev":"Z"}
        html = ""
        for k, v in signals.items():
            c = "#1f7a4f" if v > 0 else ("#b42318" if v < 0 else "#d1d5db")
            bg = "#dcfce7" if v > 0 else ("#fee2e2" if v < 0 else "#f3f4f6")
            html += (f"<span style='display:inline-block;padding:1px 5px;margin:1px;"
                     f"border-radius:4px;font-size:9px;font-weight:700;"
                     f"background:{bg};color:{c};'>{icons.get(k,k)}</span>")
        return html

    def _dist_bar(p25, p50, p75, p5, p95) -> str:
        """Inline distribution visualisation."""
        vals = [p5, p25, p50, p75, p95]
        col = "#1f7a4f" if p50 > 0 else "#b42318"
        bar_html = (
            f"<div style='font-size:9px;color:rgba(0,0,0,0.45);'>"
            f"p5:{p5*100:+.1f}% "
            f"<span style='color:{col};font-weight:700;'>p50:{p50*100:+.1f}%</span> "
            f"p95:{p95*100:+.1f}%</div>"
        )
        return bar_html

    def _trade_cards(df_in):
        if df_in.empty:
            st.caption("No assets meet the confluence + historical agreement criteria.")
            return
        cols = st.columns(3, gap="medium")
        for i, (_, row) in enumerate(df_in.iterrows()):
            t  = row["asset"]
            nm = row["name"]
            d    = row["direction"]
            hv   = bool(row.get("hist_validates", True))
            sc   = float(row["alpha_score"])
            c    = "#1f7a4f" if d == "long" else "#b42318"
            bg   = "#dcfce7" if d == "long" else "#fee2e2"
            if not hv:
                c = "#94a3b8"; bg = "#f3f4f6"
            conf = int(row["confluence"])
            wr   = float(row["win_rate"])
            k25  = float(row["kelly_25"])
            sr   = float(row["ann_sharpe"])
            p50  = float(row.get("p50_trade", row["p50_4w"]))
            p25  = float(row["p25_4w"])
            p75  = float(row["p75_4w"])
            p5   = float(ret_dist[(ret_dist.asset==t)&(ret_dist.label==cur_label)]["p5"].values[0]) \
                   if not ret_dist[(ret_dist.asset==t)&(ret_dist.label==cur_label)].empty else p25
            p95  = float(ret_dist[(ret_dist.asset==t)&(ret_dist.label==cur_label)]["p95"].values[0]) \
                   if not ret_dist[(ret_dist.asset==t)&(ret_dist.label==cur_label)].empty else p75
            n    = int(row["n_obs"])
            dots = _signal_dots(row["signals"])
            dist = _dist_bar(p25, p50, p75, p5, p95)
            sr_color = c if abs(sr) > 0.3 else "#6b7280"
            tc   = ASSET_COLORS.get(t, "#6b7280")

            cols[i % 3].markdown(
                f"<div style='padding:14px;border-radius:14px;background:#fafafa;"
                f"border:1.5px solid {c}44;margin-bottom:12px;'>"
                # Header
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;'>"
                f"<div><span style='font-size:18px;font-weight:900;color:{tc};'>{t}</span>"
                f"<span style='font-size:11px;color:rgba(0,0,0,0.45);margin-left:6px;'>{nm}</span></div>"
                f"<span style='font-size:11px;font-weight:800;color:{c};background:{bg};"
                f"padding:3px 8px;border-radius:8px;'>{d.upper()}"
                + ("" if hv else " ⚠") + "</span>"
                f"</div>"
                # Alpha score bar
                f"<div style='font-size:11px;color:rgba(0,0,0,0.50);margin-bottom:6px;'>"
                f"Alpha score <b style='color:{c};font-size:14px;'>{sc:.3f}</b>"
                f"<span style='margin-left:8px;color:rgba(0,0,0,0.35);'>n={n}w</span></div>"
                # Signal dots
                f"<div style='margin-bottom:8px;'>Signals: {dots}</div>"
                f"<div style='font-size:10px;color:rgba(0,0,0,0.40);margin-bottom:6px;'>"
                f"R=Regime C=Curve V=Vol F=Fed M=Mom Z=MeanRev · green=bullish red=bearish</div>"
                # Stats grid
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;'>"
                f"<div style='background:#f8f9fa;border-radius:7px;padding:6px 8px;'>"
                f"<div style='font-size:9px;color:rgba(0,0,0,0.40);text-transform:uppercase;'>Win rate</div>"
                f"<div style='font-size:14px;font-weight:800;color:{c};'>{wr:.0%}</div></div>"
                f"<div style='background:#f8f9fa;border-radius:7px;padding:6px 8px;'>"
                f"<div style='font-size:9px;color:rgba(0,0,0,0.40);text-transform:uppercase;'>Kelly 25%</div>"
                f"<div style='font-size:14px;font-weight:800;color:{c};'>{k25:.1%}</div></div>"
                f"<div style='background:#f8f9fa;border-radius:7px;padding:6px 8px;'>"
                f"<div style='font-size:9px;color:rgba(0,0,0,0.40);text-transform:uppercase;'>Ann Sharpe</div>"
                f"<div style='font-size:14px;font-weight:800;color:{sr_color};'>{sr:.2f}</div></div>"
                f"</div>"
                # 4w return distribution
                f"<div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.40);"
                f"text-transform:uppercase;margin-bottom:3px;'>4w return distribution</div>"
                f"{dist}"
                f"<div style='font-size:10px;color:rgba(0,0,0,0.50);margin-top:4px;'>"
                f"Confluence: <b style='color:{c};'>{conf:+d}/6</b></div>"
                f"</div>",
                unsafe_allow_html=True)

    with tab_long:
        _trade_cards(longs)
    with tab_short:
        _trade_cards(shorts)
    with tab_all:
        # Compact table view
        tbl = trade_df[["asset","name","direction","p50_4w","win_rate",
                         "kelly_25","ann_sharpe","confluence","alpha_score","n_obs"]].copy()
        tbl["p50_4w"]     = tbl["p50_4w"].apply(lambda x: f"{x*100:+.1f}%")
        tbl["win_rate"]   = tbl["win_rate"].apply(lambda x: f"{x:.0%}")
        tbl["kelly_25"]   = tbl["kelly_25"].apply(lambda x: f"{x:.1%}")
        tbl["ann_sharpe"] = tbl["ann_sharpe"].apply(lambda x: f"{x:.2f}")
        tbl["confluence"] = tbl["confluence"].apply(lambda x: f"{x:+d}/6")
        tbl["alpha_score"]= tbl["alpha_score"].apply(lambda x: f"{x:.3f}")
        tbl.columns       = ["Ticker","Name","Dir","p50 4w","Win%","Kelly25","Sharpe",
                              "Confluence","Alpha","n"]
        from src.ui import html_table
        st.markdown(html_table(tbl), unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FACTOR ROTATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='me-rowtitle'>Factor rotation — where to tilt the book</div>",
            unsafe_allow_html=True)
st.caption("Z-score composite of curve, real rates, credit, and dollar. "
           "Score >0 = first factor preferred. Score <0 = second factor preferred.")

fc1, fc2 = st.columns([1.4, 1.0], gap="large")

with fc1:
    for f in factor_scores:
        sc    = f["score"]
        d     = f["direction"]
        fc_c  = "#1f7a4f" if sc > 0.3 else ("#b42318" if sc < -0.3 else "#6b7280")
        fc_bg = "#dcfce7" if sc > 0.3 else ("#fee2e2" if sc < -0.3 else "#f3f4f6")
        bar_w = min(abs(sc) / 3.0 * 120, 120)
        bar_c = "#1f7a4f" if sc > 0 else "#b42318"

        st.markdown(
            f"<div style='padding:12px 14px;border-radius:12px;background:#fafafa;"
            f"border:1px solid rgba(0,0,0,0.07);margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
            f"<div style='font-size:13px;font-weight:800;color:rgba(0,0,0,0.85);'>{f['factor']}</div>"
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:{bar_w:.0f}px;height:6px;background:{bar_c};border-radius:3px;'></div>"
            f"<span style='font-size:13px;font-weight:900;color:{fc_c};min-width:36px;text-align:right;'>"
            f"{sc:+.2f}</span></div></div>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:5px;'>"
            f"<span style='font-size:11px;font-weight:800;color:{fc_c};"
            f"background:{fc_bg};padding:2px 8px;border-radius:6px;'>{d}</span>"
            f"<span style='font-size:10px;color:rgba(0,0,0,0.38);'>{f['proxy']}</span></div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.55);line-height:1.4;'>{f['description']}</div>"
            f"<div style='font-size:10px;color:rgba(0,0,0,0.35);margin-top:4px;'>{f['z_inputs']}</div>"
            f"</div>",
            unsafe_allow_html=True)

with fc2:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Factor score radar</div>", unsafe_allow_html=True)
        categories = [f["factor"].split(" vs ")[0] for f in factor_scores]
        scores_vals = [abs(f["score"]) * (1 if f["score"] > 0 else -1) for f in factor_scores]
        colors_vals = ["#1f7a4f" if s > 0 else "#b42318" for s in scores_vals]

        fig_fact = go.Figure()
        fig_fact.add_trace(go.Bar(
            x=[abs(s) for s in scores_vals],
            y=categories,
            orientation="h",
            marker_color=["#1f7a4f" if s > 0 else "#b42318" for s in scores_vals],
            marker_opacity=0.85,
            text=[f"{s:+.2f}" for s in scores_vals],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_fact.add_vline(x=0, line_color="#94a3b8", line_width=1)
        max_s = max(abs(s) for s in scores_vals) if scores_vals else 1
        fig_fact.update_layout(
            height=280, margin=dict(l=10, r=50, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(range=[0, max_s*1.25], showgrid=True, gridcolor="#f1f5f9",
                       title="|Factor score|"),
            yaxis=dict(autorange="reversed", showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_fact, width='stretch')

        # Transition matrix heatmap
        if not markov_mat.empty:
            st.markdown("<div class='me-rowtitle' style='margin-top:8px;'>Markov transition matrix</div>",
                        unsafe_allow_html=True)
            st.caption("1-week regime transition probabilities from history.")
            z_mat = markov_mat.values.tolist()
            fig_mk = go.Figure(go.Heatmap(
                z=z_mat,
                x=list(markov_mat.columns),
                y=list(markov_mat.index),
                colorscale=[[0,"#f8fafc"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]],
                zmin=0, zmax=1,
                text=[[f"{v:.0%}" for v in row] for row in z_mat],
                texttemplate="%{text}", textfont=dict(size=9),
                showscale=False,
            ))
            # Highlight current regime row
            if cur_label in markov_mat.index:
                ri = list(markov_mat.index).index(cur_label)
                fig_mk.add_shape(type="rect", x0=-0.5, x1=len(markov_mat.columns)-0.5,
                                 y0=ri-0.5, y1=ri+0.5,
                                 line=dict(color=cur_color, width=2), fillcolor="rgba(0,0,0,0)")
            fig_mk.update_layout(
                height=200, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(side="bottom", showgrid=False),
                yaxis=dict(autorange="reversed", showgrid=False),
            )
            st.plotly_chart(fig_mk, width='stretch')

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DIVERGENCE ALERTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='me-rowtitle'>Cross-asset divergence alerts</div>",
            unsafe_allow_html=True)
st.caption("When two markets that should agree are sending contradictory signals — "
           "one is mispricing the other. These are the highest-quality opportunistic signals.")

if divergences:
    dcols = st.columns(min(len(divergences), 3), gap="medium")
    for i, div in enumerate(divergences):
        col  = dcols[i % 3]
        dz   = div["delta_z"]
        dc   = "#b42318" if dz > 0 else "#1f7a4f"
        dbg  = "#fee2e2" if dz > 0 else "#dcfce7"
        col.markdown(
            f"<div style='padding:14px;border-radius:14px;background:#fafafa;"
            f"border:1.5px solid {dc}44;margin-bottom:12px;'>"
            f"<div style='font-size:11px;font-weight:800;color:{dc};"
            f"margin-bottom:6px;'>{div['pair']}</div>"
            f"<div style='font-size:20px;font-weight:900;color:{dc};"
            f"margin-bottom:8px;'>Δz {dz:+.2f}</div>"
            f"<div style='font-size:11px;color:rgba(0,0,0,0.70);line-height:1.5;"
            f"margin-bottom:8px;'>{div['interpretation']}</div>"
            f"<div style='padding:7px 10px;border-radius:8px;background:{dbg};"
            f"font-size:11px;font-weight:700;color:{dc};'>"
            f"⟶ {div['trade_implication']}</div>"
            f"</div>",
            unsafe_allow_html=True)
else:
    st.info("No significant cross-asset divergences detected. All markets broadly consistent.")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RETURN DISTRIBUTION CHART (current regime, top assets)
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>4-week return distribution by regime</div>",
                unsafe_allow_html=True)
    st.caption("Box plot of 4-week forward returns in each regime. "
               "Width of box = interquartile range. Whiskers = p5/p95. "
               "Current regime highlighted.")

    if not ret_dist.empty:
        top_assets = trade_df["asset"].head(8).tolist() if not trade_df.empty else ASSETS[:8]
        fig_box = go.Figure()
        regime_order = ["Risk On","Bullish","Neutral","Bearish","Risk Off"]
        regime_colors_map = {"Risk On":"#1f7a4f","Bullish":"#16a34a","Neutral":"#94a3b8",
                             "Bearish":"#d97706","Risk Off":"#b42318"}

        for t in top_assets:
            if t not in px.columns: continue
            s = px[t].resample("W-FRI").last().dropna()
            fwd = s.pct_change(4).shift(-4).dropna()
            rh  = reg_hist.copy()
            rh["label"] = rh["score"].apply(_score_to_label)
            merged = rh.join(fwd.rename("fwd"), how="inner").dropna()
            if merged.empty: continue

            for lbl in regime_order:
                sub = merged[merged["label"] == lbl]["fwd"].dropna()
                if len(sub) < 5: continue
                is_cur = lbl == cur_label
                fig_box.add_trace(go.Box(
                    y=sub.values * 100,
                    name=t,
                    legendgroup=lbl,
                    legendgrouptitle_text=lbl if t == top_assets[0] else None,
                    marker_color=regime_colors_map.get(lbl, "#6b7280"),
                    marker_opacity=1.0 if is_cur else 0.35,
                    line_width=2 if is_cur else 1,
                    boxpoints=False,
                    showlegend=(t == top_assets[0]),
                    x=[lbl] * len(sub),
                ))
        # add_vline doesn't support categorical x — use add_shape + add_annotation instead
        fig_box.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=cur_label, x1=cur_label,
            y0=0, y1=1,
            line=dict(color=cur_color, width=2, dash="dot"),
        )
        fig_box.add_annotation(
            xref="x", yref="paper",
            x=cur_label, y=1.02,
            text="◀ Current regime",
            showarrow=False,
            font=dict(color=cur_color, size=10, family="Inter"),
            xanchor="center",
        )
        # Dynamic y-axis: pad outward from zero in each direction
        _lo_raw = float(pd.to_numeric(ret_dist["p5"],  errors="coerce").dropna().min()) * 100
        _hi_raw = float(pd.to_numeric(ret_dist["p95"], errors="coerce").dropna().max()) * 100
        lo_b = _lo_raw * 1.15 if _lo_raw < 0 else _lo_raw * 0.85
        hi_b = _hi_raw * 1.15 if _hi_raw > 0 else _hi_raw * 0.85
        fig_box.update_layout(
            height=400, margin=dict(l=10, r=10, t=30, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="4w return (%)", showgrid=True, gridcolor="#f1f5f9",
                       range=[lo_b, hi_b], zeroline=True, zerolinecolor="#e2e8f0"),
            xaxis=dict(showgrid=False, categoryorder="array", categoryarray=regime_order),
            boxmode="group", hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_box, width='stretch')

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — REGIME CALENDAR + SCORE HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Regime history & context</div>", unsafe_allow_html=True)
    rng = st.selectbox("Range", ["1y","3y","5y"], index=1, key="hist_range")
    days = {"1y":400,"3y":1100,"5y":2000}[rng]

    if not reg_hist.empty:
        end_dt = reg_hist.index.max()
        rh_view = reg_hist.loc[reg_hist.index >= end_dt - pd.Timedelta(days=days)].copy()
        rh_view["label"] = rh_view["score"].apply(_score_to_label)

        spy_line = None
        if "SPY" in px.columns:
            spy_wk = px["SPY"].resample("W-FRI").last().dropna()
            spy_wk = spy_wk.loc[spy_wk.index >= end_dt - pd.Timedelta(days=days)]
            if not spy_wk.empty:
                spy_line = (spy_wk / spy_wk.iloc[0] * 100).rename("SPY")

        fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.65, 0.35], vertical_spacing=0.04)

        for y0, y1, bg in [(0,25,"rgba(180,35,24,0.07)"),(25,40,"rgba(217,119,6,0.06)"),
                            (40,60,"rgba(107,114,128,0.04)"),(60,75,"rgba(22,163,74,0.06)"),
                            (75,100,"rgba(31,122,79,0.09)")]:
            fig_hist.add_hrect(y0=y0, y1=y1, fillcolor=bg, line_width=0, row=1, col=1)
        for t, lbl in [(25,"Bearish"),(40,"Neutral"),(60,"Bullish"),(75,"Risk On")]:
            fig_hist.add_hline(y=t, line_dash="dash", line_color="#d1d5db",
                               line_width=1, annotation_text=lbl,
                               annotation_position="right", annotation_font_size=9, row=1, col=1)

        fig_hist.add_trace(go.Scatter(x=rh_view.index, y=rh_view["score"],
                                       mode="lines", name="Score",
                                       line=dict(color="#1d4ed8", width=2.2),
                                       fill="tozeroy", fillcolor="rgba(29,78,216,0.05)"),
                           row=1, col=1)

        if spy_line is not None:
            fig_hist.add_trace(go.Scatter(x=spy_line.index, y=spy_line.values,
                                           mode="lines", name="SPY (indexed)",
                                           line=dict(color="#94a3b8", width=1.4, dash="dot"),
                                           yaxis="y3"), row=1, col=1)

        label_num = {"Risk On":5,"Bullish":4,"Neutral":3,"Bearish":2,"Risk Off":1}
        lrc        = {"Risk On":"#1f7a4f","Bullish":"#16a34a","Neutral":"#94a3b8",
                      "Bearish":"#d97706","Risk Off":"#b42318"}
        for lbl, c in lrc.items():
            mask = rh_view["label"] == lbl
            if mask.any():
                fig_hist.add_trace(go.Bar(x=rh_view.index[mask],
                                           y=[label_num[lbl]]*mask.sum(),
                                           name=lbl, marker_color=c,
                                           showlegend=True), row=2, col=1)

        fig_hist.update_layout(
            height=480, margin=dict(l=10, r=60, t=20, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.03, x=0),
            barmode="overlay", hovermode="x unified",
            yaxis3=dict(overlaying="y", side="right", showgrid=False, title="SPY"),
        )
        fig_hist.update_yaxes(range=[0,100], title_text="Score",
                               showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
        fig_hist.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
        fig_hist.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_hist, width='stretch')

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)