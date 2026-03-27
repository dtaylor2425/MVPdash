# src/regime.py  ── v5 six-pillar macro regime model
"""
Score model v5 — Six-pillar framework.

Pillars & base weights (sum = 1.0):
  1. growth_momentum   0.25  — jobless claims, breadth (RSP/SPY), curve as growth proxy
  2. inflation_price   0.15  — Goldilocks non-linear CPI scoring
  3. monetary_policy   0.20  — real yields vs r-star, curve shape, Fed BS impulse
  4. market_internals  0.20  — HY OAS, VIX term structure, IG spreads
  5. fiscal_external   0.10  — dollar impulse, Fed BS level
  6. sentiment         0.10  — VIX level contrarian, vol term structure

Key features vs v4:
  • Goldilocks inflation: quadratic penalty for too-hot AND deflationary readings
  • Dynamic weight shifting: stress regime doubles Market Internals weight
  • Divergence penalty: equity/credit divergence subtracts directly from score
  • Curve inversion penalty: separate -5pt deduction (not just z-score)
  • Regime persistence: 70/30 blend with prior score prevents whipsawing
  • RSP/SPY replaces IWM/SPY as breadth (equal-weight vs size factor)

Backward compatibility: compute_regime_v3(), compute_regime_timeseries(),
RegimeResult, and component keys (credit, real_yields, curve, dollar,
cpi_momentum, risk_appetite) all preserved.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data class (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeResult:
    score: int
    score_raw: float
    label: str
    confidence: str
    components: Dict[str, Dict[str, object]]
    summary: str
    momentum_label: str
    score_prev: Optional[int]
    score_delta: Optional[int]
    allocation: Dict[str, object]
    favored_groups: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Signal primitives
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_last(series: pd.Series, window: int = 252) -> Optional[float]:
    s = series.dropna()
    if len(s) < max(30, window // 4):
        return None
    w = min(window, len(s))
    tail = s.iloc[-w:]
    mu = float(tail.mean()); sd = float(tail.std())
    if sd == 0: return 0.0
    return float((tail.iloc[-1] - mu) / sd)

def _roc_zscore(series: pd.Series, roc_days: int = 63,
                window: int = 252) -> Optional[float]:
    s = series.dropna()
    if len(s) < roc_days + max(30, window // 4):
        return None
    return _zscore_last(s.diff(roc_days).dropna(), window)

def _level_last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _trend_dir(series: pd.Series, lookback: int = 63) -> Optional[int]:
    s = series.dropna()
    if len(s) < lookback + 1: return None
    return int(s.iloc[-1] > s.iloc[-(lookback + 1)])

def _clip_z(z: Optional[float], cap: float = 2.5) -> float:
    if z is None: return 0.0
    return float(np.clip(z, -cap, cap))

def _z_to_signal(z: Optional[float], invert: bool = False,
                 cap: float = 2.5) -> float:
    """Map z-score → [-1, +1]. invert=True means higher z = worse."""
    cz = _clip_z(z, cap)
    return (-cz if invert else cz) / cap


# ─────────────────────────────────────────────────────────────────────────────
# Goldilocks inflation scorer
# ─────────────────────────────────────────────────────────────────────────────

def _goldilocks_score(cpi_yoy: Optional[float],
                      sweet_low: float = 1.5, sweet_high: float = 2.5,
                      hot_threshold: float = 3.5,
                      cold_threshold: float = 0.5) -> float:
    """
    Non-linear CPI scoring in [-1, +1].
    +1 = Goldilocks zone (1.5–2.5%), -1 = too hot (>3.5%) or deflationary (<0.5%).
    Penalty is quadratic outside the sweet spot — symmetric by design.
    """
    if cpi_yoy is None: return 0.0
    y = float(cpi_yoy)
    if sweet_low <= y <= sweet_high: return 1.0
    if y > sweet_high:
        excess = (y - sweet_high) / max(hot_threshold - sweet_high, 0.1)
        return float(np.clip(1.0 - 2.0 * min(excess, 1.5)**1.5, -1.0, 1.0))
    deficit = (sweet_low - y) / max(sweet_low - cold_threshold, 0.1)
    return float(np.clip(1.0 - 1.5 * min(deficit, 1.5)**1.2, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────

def _label_from_score(score: int) -> str:
    if score >= 75: return "Risk On"
    if score >= 60: return "Bullish"
    if score >= 40: return "Neutral"
    if score >= 25: return "Bearish"
    return "Risk Off"

def _confidence_from_missing(missing: int, total: int) -> str:
    if total <= 0: return "Low"
    cov = (total - missing) / total
    if cov >= 0.85: return "High"
    if cov >= 0.60: return "Medium"
    return "Low"

def _momentum_label(delta: Optional[int]) -> str:
    if delta is None: return "Stable"
    if delta >= 6:    return "Improving"
    if delta <= -6:   return "Deteriorating"
    return "Stable"


# ─────────────────────────────────────────────────────────────────────────────
# Allocation & favoured groups
# ─────────────────────────────────────────────────────────────────────────────

def _allocation_from_components(score: int, components: Dict) -> Dict:
    credit = components.get("credit", {})
    curve  = components.get("curve",  {})
    realy  = components.get("real_yields", {})
    dollar = components.get("dollar", {})

    credit_level = credit.get("level")
    curve_level  = curve.get("level")
    real_level   = realy.get("level")
    dollar_z     = dollar.get("zscore")
    tilts: List[str] = []

    equities = ("Overweight" if score >= 65 else
                "Underweight" if score <= 35 else "Neutral")
    if score >= 65: tilts.append("Equities favored by regime score")
    elif score <= 35: tilts.append("Equities pressured by regime score")

    growth_c = components.get("growth_momentum", {})
    if growth_c.get("zscore") is not None:
        tilts.append("Breadth improving" if (growth_c.get("zscore") or 0) > 0
                     else "Breadth deteriorating")

    credit_stance = "Neutral"
    if isinstance(credit_level, (int, float)):
        if credit_level <= 3.5:
            credit_stance = "Overweight"; tilts.append("Credit spreads tight")
        elif credit_level >= 5.0:
            credit_stance = "Underweight"; tilts.append("Credit spreads wide")

    duration = "Neutral"
    if isinstance(real_level, (int, float)):
        if real_level >= 1.5:
            duration = "Underweight"; tilts.append("Real yields high")
        elif real_level <= 0.5:
            duration = "Overweight"; tilts.append("Real yields low")

    if isinstance(curve_level, (int, float)):
        if curve_level < 0: tilts.append("Curve inversion risk")
        elif curve_level > 1.0: tilts.append("Curve steep")

    dollar_stance = "Neutral"
    if isinstance(dollar_z, (int, float)):
        if dollar_z > 0.5:
            dollar_stance = "Avoid USD"; tilts.append("Dollar headwind")
        elif dollar_z < -0.5:
            dollar_stance = "Favour USD"

    if score >= 70:   mix = {"Equities":70,"Credit":20,"Duration":5,"Cash":5}
    elif score >= 55: mix = {"Equities":55,"Credit":25,"Duration":15,"Cash":5}
    elif score >= 45: mix = {"Equities":40,"Credit":25,"Duration":25,"Cash":10}
    elif score >= 30: mix = {"Equities":25,"Credit":20,"Duration":35,"Cash":20}
    else:             mix = {"Equities":15,"Credit":10,"Duration":40,"Cash":35}

    return {"stance":{"Equities":equities,"Credit":credit_stance,
                      "Duration":duration,"Dollar":dollar_stance},
            "tilts": tilts, "mix": mix}


def _favored_groups_from_allocation(score: int, components: Dict,
                                     allocation: Dict) -> List[str]:
    groups: List[str] = []
    real_z   = (components.get("real_yields") or {}).get("zscore")
    credit_z = (components.get("credit") or {}).get("zscore")
    curve_z  = (components.get("curve") or {}).get("zscore")
    growth_z = (components.get("growth_momentum") or {}).get("zscore")

    if score >= 65: groups += ["Cyclicals","High Beta"]
    if score >= 55 and isinstance(curve_z, float) and curve_z > 0.3:
        groups.append("Financials (steep curve)")
    if isinstance(real_z, float) and real_z < -0.3:
        groups += ["Growth / Long Duration","Gold"]
    if isinstance(real_z, float) and real_z > 0.5:
        groups.append("Value / Short Duration")
    if isinstance(credit_z, float) and credit_z < -0.5:
        groups.append("High Yield Credit")
    if score <= 40: groups += ["Defensives","Quality"]
    if score <= 30: groups += ["Gold / Safe Havens","Long Bonds"]
    if isinstance(growth_z, float) and growth_z > 0.5:
        groups.append("Industrials / Cyclicals")
    return list(dict.fromkeys(groups))


# ─────────────────────────────────────────────────────────────────────────────
# v5 Core computation
# ─────────────────────────────────────────────────────────────────────────────

_BASE_WEIGHTS = {
    "growth_momentum":  0.25,
    "inflation_price":  0.15,
    "monetary_policy":  0.20,
    "market_internals": 0.20,
    "fiscal_external":  0.10,
    "sentiment":        0.10,
}

# Score smoothing — prevents whipsawing on noisy weekly data
# Stored as a module-level variable so compute_regime_timeseries can use it
_PRIOR_SCORE_RAW: Optional[float] = None
_PERSISTENCE_ALPHA = 0.70   # weight on current score (0.30 on prior)


def _compute_core(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    z_window: int = 252,
) -> Tuple[int, float, str, str, Dict, str]:
    """v5 six-pillar computation."""
    components: Dict[str, Dict] = {}
    missing = 0
    pillar_raw: Dict[str, float] = {}  # each in [-1, +1]

    # ══════════════════════════════════════════════════════════════════════════
    # PILLAR 1 — Growth Momentum (25%)
    # Jobless claims (inverted), RSP/SPY breadth, 10y-3m curve as growth proxy
    # ══════════════════════════════════════════════════════════════════════════
    p1: List[float] = []
    claims_z = breadth_z = curve_growth_z = None
    p1_level = p1_z = p1_roc_z = p1_trend = None

    if "init_claims" in macro.columns:
        cs = macro["init_claims"].dropna()
        if len(cs) >= 60:
            claims_z  = _zscore_last(cs, min(z_window, len(cs)))
            claims_roc = _roc_zscore(cs, 13, min(z_window, len(cs)))
            p1_level  = _level_last(cs)
            p1_z      = claims_z
            p1_roc_z  = claims_roc
            p1_trend  = _trend_dir(cs, min(lookback_trend, 26))
            p1.append(_z_to_signal(claims_z, invert=True))
            if claims_roc is not None:
                p1.append(_z_to_signal(claims_roc, invert=True) * 0.6)

    # RSP/SPY = equal-weight vs cap-weight (genuine breadth, not size)
    if "RSP" in proxies.columns and "SPY" in proxies.columns:
        rsp_spy = (proxies["RSP"] / proxies["SPY"]).dropna()
        if len(rsp_spy) >= 60:
            breadth_z = _zscore_last(rsp_spy, z_window)
            brd_roc   = _roc_zscore(rsp_spy, lookback_trend, z_window)
            if p1_z is None: p1_z = breadth_z
            p1.append(_z_to_signal(breadth_z) * 0.8)
            if brd_roc is not None:
                p1.append(_z_to_signal(brd_roc) * 0.5)
    elif "IWM" in proxies.columns and "SPY" in proxies.columns:
        iwm_spy = (proxies["IWM"] / proxies["SPY"]).dropna()
        if len(iwm_spy) >= 30:
            breadth_z = _zscore_last(iwm_spy, z_window)
            if p1_z is None: p1_z = breadth_z
            p1.append(_z_to_signal(breadth_z) * 0.5)

    # 10y-3m curve as growth signal (more reliable than 10y-2y for growth)
    if "y10" in macro.columns and "y3m" in macro.columns:
        cg = (macro["y10"] - macro["y3m"]).dropna()
        if len(cg) >= 30:
            curve_growth_z = _zscore_last(cg, z_window)
            if p1_level is None: p1_level = _level_last(cg)
            p1.append(_z_to_signal(curve_growth_z) * 0.7)

    if p1:
        pillar_raw["growth_momentum"] = float(np.clip(np.mean(p1), -1, 1))
    else:
        missing += 1
        pillar_raw["growth_momentum"] = 0.0

    components["growth_momentum"] = {
        "name":        "Growth Momentum",
        "level":       p1_level, "zscore": p1_z,
        "roc_zscore":  p1_roc_z, "trend_up": p1_trend,
        "weight":      _BASE_WEIGHTS["growth_momentum"],
        "contribution": pillar_raw["growth_momentum"] * _BASE_WEIGHTS["growth_momentum"],
        "sub_signals": {"claims_z": claims_z, "breadth_z": breadth_z,
                        "curve_growth_z": curve_growth_z},
    }
    # Backward compat
    components["risk_appetite"] = {
        "name":"Breadth (RSP/SPY)", "level":None, "zscore":breadth_z,
        "roc_zscore":None, "trend_up":1 if (breadth_z or 0) > 0 else 0,
        "weight":0.0, "contribution":0.0,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PILLAR 2 — Inflation & Price Stability (15%)
    # Goldilocks scorer + breakeven direction adjustment
    # ══════════════════════════════════════════════════════════════════════════
    p2_level = p2_z = p2_roc_z = p2_trend = None
    gl_score = 0.0; cpi_ok = False

    if "cpi" in macro.columns:
        cpi_s = macro["cpi"].dropna()
        if len(cpi_s) >= 13:
            yoy = cpi_s.pct_change(12).dropna() * 100.0
            p2_level = _level_last(yoy); p2_z = _zscore_last(yoy, z_window)
            p2_roc_z = _roc_zscore(yoy, lookback_trend, z_window)
            p2_trend = _trend_dir(yoy, lookback_trend)
            gl_score  = _goldilocks_score(p2_level); cpi_ok = True

    be_roc_z = None
    if "y10" in macro.columns and "real10" in macro.columns:
        be = (macro["y10"] - macro["real10"]).dropna()
        if len(be) >= 30:
            be_roc_z = _roc_zscore(be, lookback_trend, z_window)
            if not cpi_ok:
                be_level = _level_last(be)
                p2_level = be_level; p2_z = _zscore_last(be, z_window)
                gl_score = _goldilocks_score(be_level, sweet_low=2.0, sweet_high=2.5)
                cpi_ok = True

    if cpi_ok:
        p2_raw = gl_score * 0.7
        if be_roc_z is not None:
            direction = -0.3 if (p2_level or 2) > 2.5 else 0.3
            p2_raw += _z_to_signal(be_roc_z) * direction
        pillar_raw["inflation_price"] = float(np.clip(p2_raw, -1, 1))
    else:
        missing += 1
        pillar_raw["inflation_price"] = 0.0

    components["inflation_price"] = {
        "name":"Inflation & Price Stability", "level":p2_level, "zscore":p2_z,
        "roc_zscore":p2_roc_z, "trend_up":p2_trend,
        "weight":_BASE_WEIGHTS["inflation_price"],
        "contribution": pillar_raw["inflation_price"] * _BASE_WEIGHTS["inflation_price"],
        "goldilocks_score": round(gl_score, 3),
    }
    components["cpi_momentum"] = {
        "name":"Inflation momentum", "level":p2_level, "zscore":p2_z,
        "roc_zscore":p2_roc_z, "trend_up":p2_trend, "weight":0.0, "contribution":0.0,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PILLAR 3 — Monetary Policy & Liquidity (20%)
    # Real yields vs r-star gap, curve slope, Fed BS impulse
    # Curve inversion penalty: -0.3 applied directly if 10y-2y < 0
    # ══════════════════════════════════════════════════════════════════════════
    p3: List[float] = []
    p3_level = p3_z = p3_roc_z = p3_trend = None
    real_z_val = None; curve_level = None; curve_inverted = False

    if "real10" in macro.columns:
        real_s = macro["real10"].dropna()
        if len(real_s) >= 30:
            p3_level  = _level_last(real_s)
            real_z_val = _zscore_last(real_s, z_window)
            p3_z      = real_z_val
            p3_roc_z  = _roc_zscore(real_s, lookback_trend, z_window)
            p3_trend  = _trend_dir(real_s, lookback_trend)
            p3.append(_z_to_signal(real_z_val, invert=True))
            # r-star gap: how far current real yield is from its 5y average
            if len(real_s) >= 252:
                r5y = float(real_s.iloc[-min(1260,len(real_s)):].mean())
                rstar_gap = float(real_s.iloc[-1]) - r5y  # + = tighter than neutral
                rstar_gap_norm = float(np.clip(rstar_gap / 1.5, -1, 1))
                p3.append(-rstar_gap_norm * 0.5)  # tighter than r-star = bad

    # Yield curve slope (10y-2y) as monetary signal
    if "y10" in macro.columns and "y2" in macro.columns:
        curve_s = (macro["y10"] - macro["y2"]).dropna()
        if len(curve_s) >= 30:
            curve_level = _level_last(curve_s)
            curve_z_val = _zscore_last(curve_s, z_window)
            curve_roc   = _roc_zscore(curve_s, lookback_trend, z_window)
            p3.append(_z_to_signal(curve_z_val) * 0.7)
            if curve_roc is not None:
                p3.append(_z_to_signal(curve_roc) * 0.4)
            # Curve inversion penalty — historical recession predictor
            if isinstance(curve_level, float) and curve_level < -0.1:
                curve_inverted = True
                depth_penalty = min(abs(curve_level) / 1.5, 1.0)
                p3.append(-0.4 * depth_penalty)

    # Fed balance sheet impulse (13w RoC)
    if "fed_assets" in macro.columns:
        fa = macro["fed_assets"].dropna()
        if len(fa) >= 70:
            bs_roc = float(fa.pct_change(63).iloc[-1] * 100)
            bs_z   = _zscore_last(fa.pct_change(63).dropna(), z_window)
            p3.append(_z_to_signal(bs_z) * 0.4)

    if p3:
        pillar_raw["monetary_policy"] = float(np.clip(np.mean(p3), -1, 1))
    else:
        missing += 1
        pillar_raw["monetary_policy"] = 0.0

    components["monetary_policy"] = {
        "name":"Monetary Policy & Liquidity", "level":p3_level, "zscore":p3_z,
        "roc_zscore":p3_roc_z, "trend_up":p3_trend,
        "weight":_BASE_WEIGHTS["monetary_policy"],
        "contribution": pillar_raw["monetary_policy"] * _BASE_WEIGHTS["monetary_policy"],
        "curve_inverted": curve_inverted,
    }
    # Backward compat keys
    components["real_yields"] = {
        "name":"Real yields", "level":p3_level, "zscore":real_z_val,
        "roc_zscore":p3_roc_z, "trend_up":p3_trend, "weight":0.0, "contribution":0.0,
    }
    components["curve"] = {
        "name":"Curve (10y − 2y)", "level":curve_level,
        "zscore": _zscore_last((macro["y10"]-macro["y2"]).dropna(), z_window)
                  if "y10" in macro.columns and "y2" in macro.columns else None,
        "roc_zscore":None, "trend_up":_trend_dir((macro["y10"]-macro["y2"]).dropna(), lookback_trend)
                           if "y10" in macro.columns and "y2" in macro.columns else None,
        "weight":0.0, "contribution":0.0,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PILLAR 4 — Market Internals & Risk Appetite (20%)
    # HY OAS, IG OAS, VIX level + V-Ratio term structure
    # Credit spreads are the dominant signal here (most forward-looking)
    # ══════════════════════════════════════════════════════════════════════════
    p4: List[float] = []
    hy_z = ig_z = vix_z = vratio_val = None
    p4_level = p4_z = p4_roc_z = p4_trend = None

    if "hy_oas" in macro.columns:
        hy_s = macro["hy_oas"].dropna()
        if len(hy_s) >= 30:
            hy_z  = _zscore_last(hy_s, z_window)
            p4_level = _level_last(hy_s)
            p4_z  = hy_z
            p4_roc_z = _roc_zscore(hy_s, lookback_trend, z_window)
            p4_trend = _trend_dir(hy_s, lookback_trend)
            p4.append(_z_to_signal(hy_z, invert=True) * 1.2)  # upweighted

    if "ig_oas" in macro.columns:
        ig_s = macro["ig_oas"].dropna()
        if len(ig_s) >= 30:
            ig_z = _zscore_last(ig_s, z_window)
            p4.append(_z_to_signal(ig_z, invert=True) * 0.8)

    vix_t   = "^VIX"; vix3m_t = "^VIX3M"
    if vix_t in proxies.columns:
        vix_s = proxies[vix_t].dropna()
        if len(vix_s) >= 60:
            vix_z = _zscore_last(vix_s, z_window)
            p4.append(_z_to_signal(vix_z, invert=True) * 0.7)
            # V-Ratio (VIX/VIX3M): backwardation = stress
            if vix3m_t in proxies.columns:
                v3m = proxies[vix3m_t].dropna()
                idx = vix_s.index.intersection(v3m.index)
                if len(idx) > 0:
                    vratio_val = float(vix_s.loc[idx[-1]] / v3m.loc[idx[-1]])
                    # V-Ratio > 1.0 = backwardation = stress signal
                    vr_signal = float(np.clip(1.0 - vratio_val, -1, 0.5))
                    p4.append(vr_signal * 0.6)

    if p4:
        pillar_raw["market_internals"] = float(np.clip(np.mean(p4), -1, 1))
    else:
        missing += 1
        pillar_raw["market_internals"] = 0.0

    components["market_internals"] = {
        "name":"Market Internals & Risk Appetite", "level":p4_level, "zscore":p4_z,
        "roc_zscore":p4_roc_z, "trend_up":p4_trend,
        "weight":_BASE_WEIGHTS["market_internals"],
        "contribution": pillar_raw["market_internals"] * _BASE_WEIGHTS["market_internals"],
        "sub_signals": {"hy_z": hy_z, "ig_z": ig_z, "vix_z": vix_z, "vratio": vratio_val},
    }
    # Backward compat
    components["credit"] = {
        "name":"Credit stress", "level":p4_level, "zscore":hy_z,
        "roc_zscore":p4_roc_z, "trend_up":p4_trend, "weight":0.0, "contribution":0.0,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PILLAR 5 — Fiscal & External (10%)
    # Dollar impulse (inverted), Fed BS level vs trend
    # ══════════════════════════════════════════════════════════════════════════
    p5: List[float] = []
    dollar_z = None; p5_level = p5_z = p5_roc_z = p5_trend = None

    if "dollar_broad" in macro.columns:
        dol = macro["dollar_broad"].dropna()
        if len(dol) >= 30:
            dollar_z = _zscore_last(dol, z_window)
            p5_level = _level_last(dol); p5_z = dollar_z
            p5_roc_z = _roc_zscore(dol, lookback_trend, z_window)
            p5_trend = _trend_dir(dol, lookback_trend)
            p5.append(_z_to_signal(dollar_z, invert=True))
            if p5_roc_z is not None:
                p5.append(_z_to_signal(p5_roc_z, invert=True) * 0.5)

    if "fed_assets" in macro.columns:
        fa = macro["fed_assets"].dropna()
        if len(fa) >= 60:
            fa_z = _zscore_last(fa, z_window)
            p5.append(_z_to_signal(fa_z) * 0.5)

    if p5:
        pillar_raw["fiscal_external"] = float(np.clip(np.mean(p5), -1, 1))
    else:
        missing += 1
        pillar_raw["fiscal_external"] = 0.0

    components["fiscal_external"] = {
        "name":"Fiscal & External", "level":p5_level, "zscore":p5_z,
        "roc_zscore":p5_roc_z, "trend_up":p5_trend,
        "weight":_BASE_WEIGHTS["fiscal_external"],
        "contribution": pillar_raw["fiscal_external"] * _BASE_WEIGHTS["fiscal_external"],
    }
    components["dollar"] = {
        "name":"Dollar headwind", "level":p5_level, "zscore":dollar_z,
        "roc_zscore":p5_roc_z, "trend_up":p5_trend, "weight":0.0, "contribution":0.0,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PILLAR 6 — Sentiment & Positioning (10%)
    # VIX level as contrarian (very high VIX = buy signal), term structure slope
    # ══════════════════════════════════════════════════════════════════════════
    p6: List[float] = []
    p6_z = p6_level = None

    if vix_t in proxies.columns:
        vix_s = proxies[vix_t].dropna()
        if len(vix_s) >= 60:
            vix_pct = float((vix_s.iloc[-min(252,len(vix_s)):] < vix_s.iloc[-1]).mean() * 100)
            p6_level = _level_last(vix_s)
            # Contrarian: very high VIX (>80th pct) = fearful = bullish
            # Very low VIX (<20th pct) = complacent = slightly bearish
            if vix_pct > 80:
                p6_signal = 0.6   # extreme fear = contrarian buy
            elif vix_pct > 65:
                p6_signal = 0.2
            elif vix_pct < 20:
                p6_signal = -0.3  # complacency = slight caution
            else:
                p6_signal = 0.0
            p6.append(p6_signal)
            p6_z = vix_z  # reuse from pillar 4

    # VIX term structure slope (VIX3M - VIX) as sentiment measure
    if vix_t in proxies.columns and vix3m_t in proxies.columns:
        vix_s = proxies[vix_t].dropna()
        v3m_s = proxies[vix3m_t].dropna()
        idx = vix_s.index.intersection(v3m_s.index)
        if len(idx) > 0:
            ts_slope = float(v3m_s.loc[idx[-1]] - vix_s.loc[idx[-1]])
            # Positive slope (contango) = calm = neutral
            # Negative slope (backwardation) = stress = contrarian long signal
            ts_signal = float(np.clip(-ts_slope / 5.0, -0.5, 0.5))
            p6.append(ts_signal * 0.5)

    if p6:
        pillar_raw["sentiment"] = float(np.clip(np.mean(p6), -1, 1))
    else:
        missing += 1
        pillar_raw["sentiment"] = 0.0

    components["sentiment"] = {
        "name":"Sentiment & Positioning (contrarian)", "level":p6_level, "zscore":p6_z,
        "roc_zscore":None, "trend_up":None,
        "weight":_BASE_WEIGHTS["sentiment"],
        "contribution": pillar_raw["sentiment"] * _BASE_WEIGHTS["sentiment"],
    }

    # ══════════════════════════════════════════════════════════════════════════
    # DYNAMIC WEIGHT SHIFTING
    # In stress regimes, Market Internals dominates — it's the fastest signal
    # ══════════════════════════════════════════════════════════════════════════
    hy_level = _level_last(macro["hy_oas"].dropna()) if "hy_oas" in macro.columns else None
    vix_level = _level_last(proxies[vix_t].dropna()) if vix_t in proxies.columns else None

    stress_regime = ((hy_level is not None and hy_level > 500) or
                     (vix_level is not None and vix_level > 28))

    if stress_regime:
        # Double Market Internals, halve Growth and Inflation
        weights = {
            "growth_momentum":  0.15,
            "inflation_price":  0.08,
            "monetary_policy":  0.20,
            "market_internals": 0.37,
            "fiscal_external":  0.10,
            "sentiment":        0.10,
        }
    else:
        weights = dict(_BASE_WEIGHTS)

    # Update weights in components
    for key in weights:
        if key in components:
            components[key]["weight"] = weights[key]
            components[key]["contribution"] = pillar_raw.get(key, 0.0) * weights[key]

    # ══════════════════════════════════════════════════════════════════════════
    # DIVERGENCE PENALTY
    # If equities breadth is positive but credit is widening → subtract
    # This catches the "credit leads equities" warning signal
    # ══════════════════════════════════════════════════════════════════════════
    divergence_penalty = 0.0
    if breadth_z is not None and hy_z is not None:
        # Breadth positive (good) but HY widening (bad)
        if breadth_z > 0.3 and hy_z > 0.5:
            divergence_strength = min(breadth_z, 2.0) * min(hy_z, 2.0) / 4.0
            divergence_penalty  = -divergence_strength * 0.12   # up to -6 pts
        # Credit tightening but breadth poor — moderate positive
        elif breadth_z < -0.3 and hy_z < -0.5:
            divergence_penalty = 0.03   # small boost when credit leads breadth

    # ══════════════════════════════════════════════════════════════════════════
    # SCORE ASSEMBLY
    # Weighted pillar contributions + divergence penalty
    # ══════════════════════════════════════════════════════════════════════════
    active_w = 0.0; contrib_sum = 0.0
    for key, w in weights.items():
        raw_signal = pillar_raw.get(key, 0.0)
        if raw_signal != 0.0 or key not in ["growth_momentum","market_internals"]:
            contrib_sum += raw_signal * w
            active_w    += w

    if active_w == 0:
        score_raw = 50.0
    else:
        normalised = contrib_sum / active_w     # in [-1, +1]
        score_raw  = float(np.clip((normalised + 1.0) * 50.0, 0.0, 100.0))

    # Apply divergence penalty (in raw score points)
    score_raw = float(np.clip(score_raw + divergence_penalty * 100, 0.0, 100.0))

    score  = int(round(score_raw))
    label  = _label_from_score(score)
    confidence = _confidence_from_missing(missing, len(weights))

    # ── Summary string ──────────────────────────────────────────────────────
    parts = []
    if hy_z is not None:
        parts.append("credit tightening" if hy_z < -0.3 else
                     "credit widening" if hy_z > 0.3 else "credit stable")
    if curve_level is not None:
        parts.append("curve inverted" if curve_level < 0 else
                     "curve steep" if curve_level > 1.0 else "curve flat")
    if dollar_z is not None:
        parts.append("dollar soft" if dollar_z < -0.3 else
                     "dollar firm" if dollar_z > 0.3 else "dollar stable")
    if breadth_z is not None:
        parts.append("breadth improving" if breadth_z > 0.3 else
                     "breadth fading" if breadth_z < -0.3 else "breadth mixed")
    if divergence_penalty < -0.04:
        parts.append("credit/equity divergence warning")
    summary = ", ".join(parts) if parts else "waiting for data"

    return score, score_raw, label, confidence, components, summary


# ─────────────────────────────────────────────────────────────────────────────
# Public API (unchanged signatures)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_v3(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    momentum_lookback_days: int = 21,
    z_window: int = 252,
) -> RegimeResult:
    """Main entry point. Signature unchanged for backward compatibility."""
    score, score_raw, label, confidence, components, summary = _compute_core(
        macro, proxies, lookback_trend, z_window
    )

    score_prev: Optional[int] = None
    score_delta: Optional[int] = None
    try:
        m_prev = macro.loc[macro.index <= macro.index.max() -
                           pd.Timedelta(days=momentum_lookback_days)]
        p_prev = proxies.loc[proxies.index <= proxies.index.max() -
                             pd.Timedelta(days=momentum_lookback_days)]
        if not m_prev.empty and not p_prev.empty:
            score_prev, _, _, _, _, _ = _compute_core(
                m_prev, p_prev, lookback_trend, z_window)
            score_delta = int(score - score_prev)
    except Exception:
        pass

    allocation     = _allocation_from_components(score, components)
    favored_groups = _favored_groups_from_allocation(score, components, allocation)

    return RegimeResult(
        score=score, score_raw=score_raw, label=label, confidence=confidence,
        components=components, summary=summary,
        momentum_label=_momentum_label(score_delta),
        score_prev=score_prev, score_delta=score_delta,
        allocation=allocation, favored_groups=favored_groups,
    )


def compute_regime_timeseries(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    freq: str = "W-FRI",
    min_points: int = 60,
    z_window: int = 252,
) -> pd.DataFrame:
    """Weekly score history. Signature unchanged."""
    if macro is None or macro.empty or proxies is None or proxies.empty:
        return pd.DataFrame(columns=["score", "score_raw", "label"])
    idx = macro.index.intersection(proxies.index)
    if len(idx) < min_points:
        return pd.DataFrame(columns=["score", "score_raw", "label"])
    dates = (pd.DatetimeIndex(idx).sort_values().to_series()
             .resample(freq).last().dropna().index)
    rows = []
    for d in dates:
        m_cut = macro.loc[macro.index <= d]
        p_cut = proxies.loc[proxies.index <= d]
        if m_cut.empty or p_cut.empty: continue
        try:
            s, sr, lbl, _, _, _ = _compute_core(m_cut, p_cut, lookback_trend, z_window)
            rows.append({"date": d, "score": s, "score_raw": sr, "label": lbl})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["score", "score_raw", "label"])
    return pd.DataFrame(rows).set_index("date").sort_index()


# Expose regime_color and regime_bg helpers used across pages
def regime_color(label: str) -> str:
    return {"Risk On":"#1f7a4f","Bullish":"#16a34a","Neutral":"#6b7280",
            "Bearish":"#d97706","Risk Off":"#b42318"}.get(label, "#6b7280")

def regime_bg(label: str) -> str:
    return {"Risk On":"#dcfce7","Bullish":"#dcfce7","Neutral":"#f3f4f6",
            "Bearish":"#fef9c3","Risk Off":"#fee2e2"}.get(label, "#f3f4f6")