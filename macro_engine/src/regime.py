# src/regime.py  ── v4 continuous z-score model
"""
Score model v4 — replaces the discrete ±1 bucket system with fully continuous
z-scores for every component, adds CPI momentum and a growth proxy component,
and recalibrates weights.

Architecture
────────────
Each component produces one number: its *continuous z-score contribution*.
The formula is always:

    contribution_i = clip(z_i, -2.5, +2.5) × weight_i

where z_i is the z-score of the signal relative to its own trailing 252-day
history (self-calibrating), and the clip prevents extreme outliers from
dominating the sum.

For components where the signal is *inverted* (higher value = worse, e.g.
credit spreads), the z-score is negated before contributing.

The weighted contributions are summed and mapped to [0, 100]:

    raw = sum(contribution_i) / sum(weight_i)   # range ≈ [-2.5, +2.5]
    score = clip((raw / 2.5 + 1) × 50, 0, 100)

This gives a genuinely continuous score (e.g. 54.3, 67.1) and is
self-calibrating to the recent environment rather than hard-coded thresholds.

Components & weights (sum = 1.0)
──────────────────────────────────
  credit          HY OAS z-score (inverted)            0.28
  real_yields     10y TIPS real yield z-score (inv)    0.20
  curve           10y − 2y spread z-score              0.18
  risk_appetite   IWM/SPY ratio z-score                0.14
  dollar          Broad dollar index z-score (inv)     0.12
  cpi_momentum    CPI YoY momentum z-score (inv)       0.08

Rate-of-change z-scores
───────────────────────
In addition to the level z-score, each component now carries a *momentum*
z-score: z-score of the 63-day rate of change of the underlying series. This
replaces the old binary trend_up flag and is stored separately in the component
dict. The main score uses only level z-scores; roc_zscore is exposed for the
Transition Watch page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeResult:
    score: int
    score_raw: float           # continuous before int()
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
# Low-level signal primitives
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_last(series: pd.Series, window: int = 252) -> Optional[float]:
    """Z-score of the most recent value relative to rolling window."""
    s = series.dropna()
    if len(s) < max(30, window // 4):
        return None
    w    = min(window, len(s))
    tail = s.iloc[-w:]
    mu   = float(tail.mean())
    sd   = float(tail.std())
    if sd == 0:
        return 0.0
    return float((tail.iloc[-1] - mu) / sd)


def _roc_zscore(series: pd.Series, roc_days: int = 63, window: int = 252) -> Optional[float]:
    """Z-score of the roc_days rate-of-change relative to rolling window.

    Tells you not just direction but *how fast* relative to history.
    """
    s = series.dropna()
    if len(s) < roc_days + max(30, window // 4):
        return None
    roc = s.diff(roc_days).dropna()
    return _zscore_last(roc, window)


def _level_last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _trend_dir(series: pd.Series, lookback: int = 63) -> Optional[int]:
    """Legacy binary trend — kept for allocation logic only, not scoring."""
    s = series.dropna()
    if len(s) < lookback + 1:
        return None
    return int(s.iloc[-1] > s.iloc[-(lookback + 1)])


def _clip_z(z: Optional[float], cap: float = 2.5) -> float:
    if z is None:
        return 0.0
    return float(np.clip(z, -cap, cap))


# ─────────────────────────────────────────────────────────────────────────────
# Labels / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_from_score(score: int) -> str:
    if score >= 75:
        return "Risk On"
    if score >= 60:
        return "Bullish"
    if score >= 40:
        return "Neutral"
    if score >= 25:
        return "Bearish"
    return "Risk Off"


def _confidence_from_missing(missing: int, total: int) -> str:
    if total <= 0:
        return "Low"
    coverage = (total - missing) / total
    if coverage >= 0.85:
        return "High"
    if coverage >= 0.60:
        return "Medium"
    return "Low"


def _momentum_label(delta: Optional[int]) -> str:
    if delta is None:
        return "Stable"
    if delta >= 6:
        return "Improving"
    if delta <= -6:
        return "Deteriorating"
    return "Stable"


# ─────────────────────────────────────────────────────────────────────────────
# Allocation & favoured groups (unchanged logic, works from component dict)
# ─────────────────────────────────────────────────────────────────────────────

def _allocation_from_components(score: int, components: Dict) -> Dict:
    credit  = components.get("credit", {})
    curve   = components.get("curve", {})
    realy   = components.get("real_yields", {})
    dollar  = components.get("dollar", {})
    risk    = components.get("risk_appetite", {})

    credit_level = credit.get("level")
    curve_level  = curve.get("level")
    real_level   = realy.get("level")
    dollar_z     = dollar.get("zscore")
    risk_trend   = risk.get("trend_up")

    tilts: List[str] = []

    # Equities
    if score >= 65:
        equities = "Overweight"
        tilts.append("Equities favored by regime score")
    elif score <= 35:
        equities = "Underweight"
        tilts.append("Equities pressured by regime score")
    else:
        equities = "Neutral"

    if risk_trend is not None:
        tilts.append("Breadth improving" if risk_trend == 1 else "Breadth deteriorating")

    # Credit
    credit_stance = "Neutral"
    if isinstance(credit_level, (int, float)):
        if credit_level <= 3.5:
            credit_stance = "Overweight"
            tilts.append("Credit spreads tight")
        elif credit_level >= 5.0:
            credit_stance = "Underweight"
            tilts.append("Credit spreads wide")

    # Duration — now also use z-score direction
    duration = "Neutral"
    if isinstance(real_level, (int, float)):
        if real_level >= 1.5:
            duration = "Underweight"
            tilts.append("Real yields high")
        elif real_level <= 0.5:
            duration = "Overweight"
            tilts.append("Real yields low")

    if isinstance(curve_level, (int, float)):
        if curve_level < 0:
            tilts.append("Curve inversion risk")
        elif curve_level > 1.0:
            tilts.append("Curve steep")

    # Dollar
    # Note: dollar_z >= 0.5 means strong dollar — which DRAGS the regime score
    # (dollar is inverted in scoring) but means USD positions will outperform.
    # "Favour" = own this, regardless of regime direction.
    usd = "Neutral"
    if isinstance(dollar_z, (int, float)):
        if dollar_z >= 0.5:
            usd = "Favour USD"
            tilts.append("Dollar strong — tightening global conditions")
        elif dollar_z <= -0.5:
            usd = "Avoid USD"
            tilts.append("Dollar weak — tailwind for EM and commodities")

    # Commodities
    commodities = "Neutral"
    if usd == "Avoid USD" and score >= 55:
        commodities = "Overweight"
        tilts.append("Weaker dollar supports commodities")

    if score >= 65:
        mix = {"Equities": 70, "Bonds": 20, "Cash": 10}
    elif score <= 35:
        mix = {"Equities": 35, "Bonds": 45, "Cash": 20}
    else:
        mix = {"Equities": 55, "Bonds": 30, "Cash": 15}

    return {
        "stance": {
            "Equities":    equities,
            "Duration":    duration,
            "Credit":      credit_stance,
            "USD":         usd,
            "Commodities": commodities,
        },
        "mix":   mix,
        "tilts": tilts[:8],
    }


def _favored_groups_from_allocation(score: int, components: Dict, allocation: Dict) -> List[str]:
    stance     = allocation.get("stance", {}) if isinstance(allocation, dict) else {}
    equities   = stance.get("Equities",   "Neutral")
    duration   = stance.get("Duration",   "Neutral")
    credit     = stance.get("Credit",     "Neutral")
    usd        = stance.get("USD",        "Neutral")
    commodities = stance.get("Commodities", "Neutral")

    risk       = components.get("risk_appetite", {})
    curve      = components.get("curve", {})
    risk_trend = risk.get("trend_up")
    curve_level = curve.get("level")

    groups: List[str] = []

    if equities == "Overweight":
        groups += (["Small caps", "Cyclicals"] if risk_trend == 1 else ["Large cap quality"])
        groups += ["Industrials", "Energy"]
    if equities == "Underweight":
        groups += ["Defensives", "Quality dividends", "Low volatility", "Cash-like ballast"]
    if duration == "Overweight":
        groups.append("Long duration growth")
    if duration == "Underweight":
        groups += ["Value tilt", "Shorter duration equities"]
    if credit == "Overweight":
        groups.append("Credit beta")
    if credit == "Underweight":
        groups.append("Investment grade quality")
    if usd == "Avoid USD":
        groups += ["International", "Emerging markets"]
    if usd == "Favour USD":
        groups.append("USD beneficiaries")
    if commodities == "Overweight":
        groups += ["Commodities", "Real assets"]
    if isinstance(curve_level, (int, float)) and curve_level < 0:
        groups.append("Avoid deep cyclicals needing easing")
    if score >= 75:
        groups.append("Momentum leaders")
    elif score <= 25:
        groups.append("Tail risk hedges")

    seen, out = set(), []
    for g in groups:
        if g not in seen:
            seen.add(g); out.append(g)
    return out[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Core computation  (v4 — all continuous z-scores)
# ─────────────────────────────────────────────────────────────────────────────

# Weights must sum to 1.0
_WEIGHTS = {
    "credit":       0.28,
    "real_yields":  0.20,
    "curve":        0.18,
    "risk_appetite":0.14,
    "dollar":       0.12,
    "cpi_momentum": 0.08,
}

_Z_CAP = 2.5   # clip all z-scores to ±2.5 before contributing


def _compute_core(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    z_window: int = 252,
) -> Tuple[int, float, str, str, Dict, str]:
    """
    Returns (score_int, score_raw, label, confidence, components, summary).

    Each component dict now contains:
        name, level, zscore, roc_zscore, trend_up (binary, for UI only),
        weight, contribution (= clipped_z × weight, signed)
    """
    components: Dict[str, Dict] = {}
    missing = 0

    # ── 1. Credit stress (HY OAS) ──────────────────────────────────────────
    if "hy_oas" in macro.columns:
        series      = macro["hy_oas"]
        level       = _level_last(series)
        z           = _zscore_last(series, z_window)      # high OAS = bad → invert
        roc_z       = _roc_zscore(series, lookback_trend, z_window)
        trend_up    = _trend_dir(series, lookback_trend)
        contrib     = _clip_z(-z) * _WEIGHTS["credit"]   # negated: tighter = good
        components["credit"] = {
            "name":        "Credit stress",
            "level":       level,
            "zscore":      z,
            "roc_zscore":  roc_z,
            "trend_up":    trend_up,
            "weight":      _WEIGHTS["credit"],
            "contribution": contrib,
        }
    else:
        missing += 1
        components["credit"] = {"name": "Credit stress", "weight": _WEIGHTS["credit"],
                                 "contribution": 0.0}

    # ── 2. Real yields (TIPS 10y) ───────────────────────────────────────────
    if "real10" in macro.columns:
        series      = macro["real10"]
        level       = _level_last(series)
        z           = _zscore_last(series, z_window)      # high real yields = bad
        roc_z       = _roc_zscore(series, lookback_trend, z_window)
        trend_up    = _trend_dir(series, lookback_trend)
        contrib     = _clip_z(-z) * _WEIGHTS["real_yields"]
        components["real_yields"] = {
            "name":        "Real yields",
            "level":       level,
            "zscore":      z,
            "roc_zscore":  roc_z,
            "trend_up":    trend_up,
            "weight":      _WEIGHTS["real_yields"],
            "contribution": contrib,
        }
    else:
        missing += 1
        components["real_yields"] = {"name": "Real yields", "weight": _WEIGHTS["real_yields"],
                                      "contribution": 0.0}

    # ── 3. Yield curve (10y − 2y) ───────────────────────────────────────────
    curve_series = None
    if "y10" in macro.columns and "y2" in macro.columns:
        curve_series = (macro["y10"] - macro["y2"]).dropna()
    if curve_series is not None and not curve_series.empty:
        level       = _level_last(curve_series)
        z           = _zscore_last(curve_series, z_window)  # steeper = better
        roc_z       = _roc_zscore(curve_series, lookback_trend, z_window)
        trend_up    = _trend_dir(curve_series, lookback_trend)
        contrib     = _clip_z(z) * _WEIGHTS["curve"]
        components["curve"] = {
            "name":        "Curve (10y − 2y)",
            "level":       level,
            "zscore":      z,
            "roc_zscore":  roc_z,
            "trend_up":    trend_up,
            "weight":      _WEIGHTS["curve"],
            "contribution": contrib,
        }
    else:
        missing += 1
        components["curve"] = {"name": "Curve (10y − 2y)", "weight": _WEIGHTS["curve"],
                                "contribution": 0.0}

    # ── 4. Risk appetite (IWM / SPY ratio) ─────────────────────────────────
    risk_series = None
    if "IWM" in proxies.columns and "SPY" in proxies.columns:
        risk_series = (proxies["IWM"] / proxies["SPY"]).dropna()
    if risk_series is not None and len(risk_series) >= 30:
        level       = _level_last(risk_series)
        z           = _zscore_last(risk_series, z_window)   # high ratio = good
        roc_z       = _roc_zscore(risk_series, lookback_trend, z_window)
        trend_up    = _trend_dir(risk_series, lookback_trend)
        contrib     = _clip_z(z) * _WEIGHTS["risk_appetite"]
        components["risk_appetite"] = {
            "name":        "Risk appetite (IWM/SPY)",
            "level":       level,
            "zscore":      z,
            "roc_zscore":  roc_z,
            "trend_up":    trend_up,
            "weight":      _WEIGHTS["risk_appetite"],
            "contribution": contrib,
        }
    else:
        missing += 1
        components["risk_appetite"] = {"name": "Risk appetite (IWM/SPY)",
                                        "weight": _WEIGHTS["risk_appetite"],
                                        "contribution": 0.0}

    # ── 5. Dollar impulse (Broad dollar, inverted) ──────────────────────────
    if "dollar_broad" in macro.columns:
        series      = macro["dollar_broad"]
        level       = _level_last(series)
        z           = _zscore_last(series, z_window)       # strong dollar = bad
        roc_z       = _roc_zscore(series, lookback_trend, z_window)
        trend_up    = _trend_dir(series, lookback_trend)
        contrib     = _clip_z(-z) * _WEIGHTS["dollar"]
        components["dollar"] = {
            "name":        "Dollar headwind",
            "level":       level,
            "zscore":      z,
            "roc_zscore":  roc_z,
            "trend_up":    trend_up,
            "weight":      _WEIGHTS["dollar"],
            "contribution": contrib,
        }
    else:
        missing += 1
        components["dollar"] = {"name": "Dollar headwind", "weight": _WEIGHTS["dollar"],
                                 "contribution": 0.0}

    # ── 6. CPI momentum (inverted — rising inflation = headwind) ───────────
    # We use YoY CPI momentum: rate of change of the YoY rate.
    # If CPI is unavailable we fall back to the implied inflation
    # breakeven (y10 − real10) which is always available.
    cpi_z = None
    cpi_level = None
    cpi_roc_z = None
    cpi_trend = None

    if "cpi" in macro.columns:
        cpi_s = macro["cpi"].dropna()
        if len(cpi_s) >= 13:
            yoy   = cpi_s.pct_change(12).dropna() * 100.0
            cpi_level  = _level_last(yoy)
            cpi_z      = _zscore_last(yoy, z_window)
            cpi_roc_z  = _roc_zscore(yoy, lookback_trend, z_window)
            cpi_trend  = _trend_dir(yoy, lookback_trend)

    # Fallback: breakeven inflation (10y nominal − 10y real)
    if cpi_z is None and "y10" in macro.columns and "real10" in macro.columns:
        be = (macro["y10"] - macro["real10"]).dropna()
        if len(be) >= 30:
            cpi_level  = _level_last(be)
            cpi_z      = _zscore_last(be, z_window)
            cpi_roc_z  = _roc_zscore(be, lookback_trend, z_window)
            cpi_trend  = _trend_dir(be, lookback_trend)

    if cpi_z is not None:
        contrib = _clip_z(-cpi_z) * _WEIGHTS["cpi_momentum"]
        components["cpi_momentum"] = {
            "name":        "Inflation momentum",
            "level":       cpi_level,
            "zscore":      cpi_z,
            "roc_zscore":  cpi_roc_z,
            "trend_up":    cpi_trend,
            "weight":      _WEIGHTS["cpi_momentum"],
            "contribution": contrib,
        }
    else:
        missing += 1
        components["cpi_momentum"] = {"name": "Inflation momentum",
                                       "weight": _WEIGHTS["cpi_momentum"],
                                       "contribution": 0.0}

    # ── Score assembly ──────────────────────────────────────────────────────
    total      = len(_WEIGHTS)
    confidence = _confidence_from_missing(missing, total)

    # Weight the contributions; if a component is missing its contribution=0
    # and its weight is excluded from normalisation to avoid pulling toward 50.
    contrib_sum  = sum(c.get("contribution", 0.0) for c in components.values())
    active_w_sum = sum(
        _WEIGHTS[k] for k, c in components.items()
        if c.get("contribution", 0.0) != 0.0 or c.get("zscore") is not None
    )
    if active_w_sum == 0:
        score_raw = 0.0
    else:
        # contrib_sum is in range ≈ [-2.5, +2.5] (clipped z × weights)
        normalised = contrib_sum / active_w_sum
        score_raw  = float(np.clip((normalised / _Z_CAP + 1.0) * 50.0, 0.0, 100.0))

    score = int(round(score_raw))
    label = _label_from_score(score)

    # ── Summary string ──────────────────────────────────────────────────────
    parts = []
    c_credit = components.get("credit", {})
    c_curve  = components.get("curve",  {})
    c_dollar = components.get("dollar", {})
    c_risk   = components.get("risk_appetite", {})
    if c_credit.get("trend_up") is not None:
        parts.append("credit tightening" if c_credit["trend_up"] == 0 else "credit widening")
    if c_curve.get("trend_up") is not None:
        parts.append("curve steepening" if c_curve["trend_up"] == 1 else "curve flattening")
    if c_dollar.get("trend_up") is not None:
        parts.append("dollar firm" if c_dollar["trend_up"] == 1 else "dollar soft")
    if c_risk.get("trend_up") is not None:
        parts.append("breadth improving" if c_risk["trend_up"] == 1 else "breadth fading")
    summary = ", ".join(parts) if parts else "waiting for data"

    return score, score_raw, label, confidence, components, summary


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_v3(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    momentum_lookback_days: int = 21,
    z_window: int = 252,
) -> RegimeResult:
    """
    Main entry point.  Kept as compute_regime_v3 for backward compatibility.
    All callers can continue to use this function unchanged.
    """
    score, score_raw, label, confidence, components, summary = _compute_core(
        macro, proxies, lookback_trend, z_window
    )

    # Momentum: re-run on data cut momentum_lookback_days ago
    score_prev: Optional[int] = None
    score_delta: Optional[int] = None
    try:
        m_prev = macro.loc[macro.index <= macro.index.max() - pd.Timedelta(days=momentum_lookback_days)] \
                 if macro is not None and not macro.empty else macro
        p_prev = proxies.loc[proxies.index <= proxies.index.max() - pd.Timedelta(days=momentum_lookback_days)] \
                 if proxies is not None and not proxies.empty else proxies
        if m_prev is not None and not m_prev.empty and p_prev is not None and not p_prev.empty:
            score_prev, _, _, _, _, _ = _compute_core(m_prev, p_prev, lookback_trend, z_window)
            score_delta = int(score - score_prev)
    except Exception:
        pass

    momentum_label = _momentum_label(score_delta)
    allocation     = _allocation_from_components(score, components)
    favored_groups = _favored_groups_from_allocation(score, components, allocation)

    return RegimeResult(
        score=score,
        score_raw=score_raw,
        label=label,
        confidence=confidence,
        components=components,
        summary=summary,
        momentum_label=momentum_label,
        score_prev=score_prev,
        score_delta=score_delta,
        allocation=allocation,
        favored_groups=favored_groups,
    )


def compute_regime_timeseries(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    freq: str = "W-FRI",
    min_points: int = 60,
    z_window: int = 252,
) -> pd.DataFrame:
    """
    Weekly score history.
    Returns DataFrame with columns: score, score_raw, label
    """
    if macro is None or macro.empty or proxies is None or proxies.empty:
        return pd.DataFrame(columns=["score", "score_raw", "label"])

    idx = macro.index.intersection(proxies.index)
    if len(idx) < min_points:
        return pd.DataFrame(columns=["score", "score_raw", "label"])

    dates = pd.DatetimeIndex(idx).sort_values().to_series().resample(freq).last().dropna().index
    rows  = []

    for d in dates:
        m_cut = macro.loc[macro.index <= d]
        p_cut = proxies.loc[proxies.index <= d]
        if m_cut.empty or p_cut.empty:
            continue
        try:
            s, sr, lbl, _, _, _ = _compute_core(m_cut, p_cut, lookback_trend, z_window)
            rows.append({"date": d, "score": s, "score_raw": sr, "label": lbl})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["score", "score_raw", "label"])

    return pd.DataFrame(rows).set_index("date").sort_index()