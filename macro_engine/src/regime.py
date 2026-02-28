# src/regime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeResult:
    score: int
    label: str
    confidence: str
    components: Dict[str, Dict[str, object]]
    summary: str
    momentum_label: str
    score_prev: Optional[int]
    score_delta: Optional[int]
    allocation: Dict[str, object]
    favored_groups: List[str]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _trend_dir(series: pd.Series, lookback: int = 63) -> Optional[int]:
    """1 if series is higher now than `lookback` bars ago, else 0. None if not enough data."""
    s = series.dropna()
    if len(s) < lookback + 1:
        return None
    return int(s.iloc[-1] > s.iloc[-(lookback + 1)])


def _zscore_last(series: pd.Series, window: int = 252) -> Optional[float]:
    """Z-score of the latest value relative to a trailing window."""
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


def _level_last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return None if s.empty else float(s.iloc[-1])


def _bucket_score(value: Optional[float], thresholds: Tuple[float, float], invert: bool = False) -> Optional[int]:
    """
    Returns +1 (good), 0 (neutral), or -1 (bad) based on level vs thresholds.
    invert=True flips the sign (used for series where high value = bad, e.g. credit spreads).
    """
    if value is None:
        return None
    low, high = thresholds
    if value < low:
        raw = 1        # below lower threshold = good (e.g. low credit spreads)
    elif value > high:
        raw = -1       # above upper threshold = bad (e.g. high credit spreads)
    else:
        raw = 0
    return -raw if invert else raw


def _confidence_from_missing(missing: int, total: int) -> str:
    if total <= 0:
        return "Low"
    coverage = (total - missing) / total
    if coverage >= 0.85:
        return "High"
    if coverage >= 0.60:
        return "Medium"
    return "Low"


def _label_from_score(score: int) -> str:
    if score >= 75:
        return "Risk on expansion"
    if score >= 60:
        return "Risk on"
    if score >= 40:
        return "Neutral"
    if score >= 25:
        return "Risk off"
    return "Risk off stress"


def _momentum_label(delta: Optional[int]) -> str:
    if delta is None:
        return "Stable"
    # ±10 threshold on a 63-day window avoids false signals from short-term noise
    if delta >= 10:
        return "Improving"
    if delta <= -10:
        return "Deteriorating"
    return "Stable"


# ── Allocation ────────────────────────────────────────────────────────────────

def _allocation_from_components(score: int, components: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    credit = components.get("credit",        {})
    curve  = components.get("curve",         {})
    realy  = components.get("real_yields",   {})
    dollar = components.get("dollar",        {})
    risk   = components.get("risk_appetite", {})

    credit_level = credit.get("level")
    curve_level  = curve.get("level")
    real_level   = realy.get("level")
    dollar_z     = dollar.get("zscore")
    risk_trend   = risk.get("trend_up")

    tilts: List[str] = []

    # Equities stance from overall regime score
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

    # Credit: tight spreads = overweight credit, wide = underweight
    credit_stance = "Neutral"
    if isinstance(credit_level, (int, float)):
        if credit_level <= 3.5:
            credit_stance = "Overweight"
            tilts.append("Credit spreads tight — favor credit")
        elif credit_level >= 5.0:
            credit_stance = "Underweight"
            tilts.append("Credit spreads wide — avoid credit risk")

    # Duration: high real yields = underweight duration (bonds expensive when yields high)
    duration = "Neutral"
    if isinstance(real_level, (int, float)):
        if real_level >= 1.5:
            duration = "Underweight"
            tilts.append("Real yields elevated — duration headwind")
        elif real_level <= 0.5:
            duration = "Overweight"
            tilts.append("Real yields low — duration supportive")

    if isinstance(curve_level, (int, float)):
        if curve_level < 0:
            tilts.append("Inverted curve — late-cycle caution")
        elif curve_level > 1.0:
            tilts.append("Steep curve — cyclicals and financials supported")

    # USD: z-score based — strong dollar headwind for international/commodities
    usd = "Neutral"
    if isinstance(dollar_z, (int, float)):
        if dollar_z >= 0.5:
            usd = "Overweight"
            tilts.append("Dollar strong relative to history")
        elif dollar_z <= -0.5:
            usd = "Underweight"
            tilts.append("Dollar weak relative to history")

    # Commodities: weak dollar + healthy regime = supportive
    commodities = "Neutral"
    if usd == "Underweight" and score >= 55:
        commodities = "Overweight"
        tilts.append("Weaker dollar supports commodities")

    # Suggested portfolio mix
    if equities == "Overweight":
        mix = {"Equities": 70, "Bonds": 20, "Cash": 10}
    elif equities == "Underweight":
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


def _favored_groups_from_allocation(
    score: int,
    components: Dict[str, Dict[str, object]],
    allocation: Dict[str, object],
) -> List[str]:
    stance = allocation.get("stance", {}) if isinstance(allocation, dict) else {}

    equities    = stance.get("Equities",    "Neutral")
    duration    = stance.get("Duration",    "Neutral")
    credit      = stance.get("Credit",      "Neutral")
    usd         = stance.get("USD",         "Neutral")
    commodities = stance.get("Commodities", "Neutral")

    risk  = components.get("risk_appetite", {})
    curve = components.get("curve",         {})
    risk_trend  = risk.get("trend_up")
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

    if usd == "Underweight":
        groups += ["International", "Emerging markets"]
    if usd == "Overweight":
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
            seen.add(g)
            out.append(g)
    return out[:10]


# ── Core score engine ─────────────────────────────────────────────────────────

def _compute_core(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
) -> Tuple[int, str, str, Dict[str, Dict[str, object]], str]:
    """
    Builds 5 macro components, scores each on level + trend, then computes a
    weighted average score mapped to 0-100.

    Score mapping:
      Each component produces a level_score and trend_score, each in {-1, 0, +1}.
      comp_score = mean([level_score, trend_score])  →  in [-1, +1]
      weighted_sum = Σ weight_i * comp_score_i        →  in [-1, +1] (if weights sum to 1)
      final_score = clip((weighted_sum + 1) * 50, 0, 100)
        → -1 maps to 0, 0 maps to 50, +1 maps to 100

    IMPORTANT: contribution stored on each component = comp_score * weight
    This allows the UI to display a correct signed bar chart.
    """
    components: Dict[str, Dict[str, object]] = {}

    # ── 1. Credit spreads (HY OAS) — weight 0.30 ─────────────────────────────
    # High spreads = bad; rising spreads = bad
    # Level: <3.5 = good (+1), 3.5-5.0 = neutral (0), >5.0 = bad (-1)
    credit_level = _level_last(macro["hy_oas"]) if "hy_oas" in macro.columns else None
    credit_trend = _trend_dir(macro["hy_oas"], lookback_trend) if "hy_oas" in macro.columns else None
    credit_level_score = _bucket_score(credit_level, thresholds=(3.5, 5.0), invert=True)
    credit_trend_score = None if credit_trend is None else (-1 if credit_trend == 1 else 1)

    credit_part_scores = [s for s in [credit_level_score, credit_trend_score] if s is not None]
    credit_comp_score  = float(np.mean(credit_part_scores)) if credit_part_scores else None
    credit_contrib     = credit_comp_score * 0.30 if credit_comp_score is not None else 0.0

    components["credit"] = {
        "name":         "Credit stress",
        "level":        credit_level,
        "trend_up":     credit_trend,
        "level_score":  credit_level_score,
        "trend_score":  credit_trend_score,
        "weight":       0.30,
        "contribution": credit_contrib,
    }

    # ── 2. Yield curve (10y minus 2y) — weight 0.20 ──────────────────────────
    # Steep curve = good; steepening = good
    # Level: <-0.25 = bad (-1), -0.25 to 0.75 = neutral (0), >0.75 = good (+1)
    curve = None
    if "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
    curve_level = _level_last(curve) if curve is not None else None
    curve_trend = _trend_dir(curve, lookback_trend) if curve is not None else None
    curve_level_score = _bucket_score(curve_level, thresholds=(-0.25, 0.75), invert=False)
    curve_trend_score = None if curve_trend is None else (1 if curve_trend == 1 else -1)

    curve_part_scores = [s for s in [curve_level_score, curve_trend_score] if s is not None]
    curve_comp_score  = float(np.mean(curve_part_scores)) if curve_part_scores else None
    curve_contrib     = curve_comp_score * 0.20 if curve_comp_score is not None else 0.0

    components["curve"] = {
        "name":         "Curve (10y minus 2y)",
        "level":        curve_level,
        "trend_up":     curve_trend,
        "level_score":  curve_level_score,
        "trend_score":  curve_trend_score,
        "weight":       0.20,
        "contribution": curve_contrib,
    }

    # ── 3. Real yields (10y TIPS) — weight 0.15 ──────────────────────────────
    # High real yields = bad (expensive to borrow, competes with equities)
    # Level: <0.5 = good (+1), 0.5-1.5 = neutral (0), >1.5 = bad (-1)
    real_level = _level_last(macro["real10"]) if "real10" in macro.columns else None
    real_trend = _trend_dir(macro["real10"], lookback_trend) if "real10" in macro.columns else None
    real_level_score = _bucket_score(real_level, thresholds=(0.5, 1.5), invert=True)
    real_trend_score = None if real_trend is None else (-1 if real_trend == 1 else 1)

    real_part_scores = [s for s in [real_level_score, real_trend_score] if s is not None]
    real_comp_score  = float(np.mean(real_part_scores)) if real_part_scores else None
    real_contrib     = real_comp_score * 0.15 if real_comp_score is not None else 0.0

    components["real_yields"] = {
        "name":         "Real yields",
        "level":        real_level,
        "trend_up":     real_trend,
        "level_score":  real_level_score,
        "trend_score":  real_trend_score,
        "weight":       0.15,
        "contribution": real_contrib,
    }

    # ── 4. Broad dollar index — weight 0.15 ──────────────────────────────────
    # Strong dollar = headwind for risk assets and commodities
    # Scored via z-score: z > 0.5 = bad (-1), z < -0.5 = good (+1)
    dollar_series  = macro["dollar_broad"] if "dollar_broad" in macro.columns else None
    dollar_level   = _level_last(dollar_series)     if dollar_series is not None else None
    dollar_trend   = _trend_dir(dollar_series, lookback_trend) if dollar_series is not None else None
    dollar_z       = _zscore_last(dollar_series, 252) if dollar_series is not None else None
    dollar_level_score = _bucket_score(dollar_z, thresholds=(-0.5, 0.5), invert=True)
    dollar_trend_score = None if dollar_trend is None else (-1 if dollar_trend == 1 else 1)

    dollar_part_scores = [s for s in [dollar_level_score, dollar_trend_score] if s is not None]
    dollar_comp_score  = float(np.mean(dollar_part_scores)) if dollar_part_scores else None
    dollar_contrib     = dollar_comp_score * 0.15 if dollar_comp_score is not None else 0.0

    components["dollar"] = {
        "name":         "Dollar impulse",
        "level":        dollar_level,
        "zscore":       dollar_z,
        "trend_up":     dollar_trend,
        "level_score":  dollar_level_score,
        "trend_score":  dollar_trend_score,
        "weight":       0.15,
        "contribution": dollar_contrib,
    }

    # ── 5. Risk appetite (IWM / SPY ratio) — weight 0.20 ────────────────────
    # Small caps outperforming = risk-on = good
    # No level score (ratio level has no absolute meaning); trend only
    risk_ratio = None
    if isinstance(proxies, pd.DataFrame) and "IWM" in proxies.columns and "SPY" in proxies.columns:
        risk_ratio = (proxies["IWM"] / proxies["SPY"]).dropna()
    risk_level = _level_last(risk_ratio) if risk_ratio is not None else None
    risk_trend = _trend_dir(risk_ratio, lookback_trend) if risk_ratio is not None else None
    risk_trend_score = None if risk_trend is None else (1 if risk_trend == 1 else -1)

    risk_part_scores = [s for s in [risk_trend_score] if s is not None]
    risk_comp_score  = float(np.mean(risk_part_scores)) if risk_part_scores else None
    risk_contrib     = risk_comp_score * 0.20 if risk_comp_score is not None else 0.0

    components["risk_appetite"] = {
        "name":         "Risk appetite (IWM over SPY)",
        "level":        risk_level,
        "trend_up":     risk_trend,
        "level_score":  None,
        "trend_score":  risk_trend_score,
        "weight":       0.20,
        "contribution": risk_contrib,
    }

    # ── Aggregate score ───────────────────────────────────────────────────────
    missing      = 0
    total        = 0
    weighted_sum = 0.0
    weight_sum   = 0.0

    for comp in components.values():
        w           = float(comp["weight"])
        part_scores = [comp.get("level_score"), comp.get("trend_score")]
        part_scores = [s for s in part_scores if s is not None]
        total      += 1
        if not part_scores:
            missing += 1
            continue
        comp_score    = float(np.mean(part_scores))
        weighted_sum += w * comp_score
        weight_sum   += w

    confidence = _confidence_from_missing(missing, total)

    if weight_sum == 0:
        score = 50
    else:
        normalized = weighted_sum / weight_sum
        # Map [-1, +1] → [0, 100]
        score = int(np.clip((normalized + 1) * 50, 0, 100))

    label = _label_from_score(score)

    # ── Summary phrase ────────────────────────────────────────────────────────
    summary_parts = []
    # Credit: trend_up=1 means spreads widening (bad); trend_up=0 means tightening (good)
    if components["credit"]["trend_up"] is not None:
        summary_parts.append(
            "credit widening" if components["credit"]["trend_up"] == 1 else "credit tightening"
        )
    if components["curve"]["trend_up"] is not None:
        summary_parts.append("curve steepening" if components["curve"]["trend_up"] == 1 else "curve flattening")
    if components["dollar"]["trend_up"] is not None:
        summary_parts.append("dollar strengthening" if components["dollar"]["trend_up"] == 1 else "dollar weakening")
    if components["risk_appetite"]["trend_up"] is not None:
        summary_parts.append(
            "breadth improving" if components["risk_appetite"]["trend_up"] == 1 else "breadth deteriorating"
        )

    summary = ", ".join(summary_parts) if summary_parts else "waiting for data"
    return score, label, confidence, components, summary


# ── Public API ────────────────────────────────────────────────────────────────

def compute_regime_v3(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    momentum_lookback_days: int = 63,   # CHANGED: was 21, now 63 to reduce noise
) -> RegimeResult:
    score, label, confidence, components, summary = _compute_core(macro, proxies, lookback_trend)

    score_prev:  Optional[int] = None
    score_delta: Optional[int] = None

    # Slice data to momentum_lookback_days ago and re-score
    macro_prev   = macro
    proxies_prev = proxies

    if macro is not None and not macro.empty:
        cut        = macro.index.max() - pd.Timedelta(days=momentum_lookback_days)
        macro_prev = macro.loc[macro.index <= cut]

    if proxies is not None and not proxies.empty:
        cutp         = proxies.index.max() - pd.Timedelta(days=momentum_lookback_days)
        proxies_prev = proxies.loc[proxies.index <= cutp]

    try:
        if (macro_prev is not None and not macro_prev.empty
                and proxies_prev is not None and not proxies_prev.empty):
            score_prev, _, _, _, _ = _compute_core(macro_prev, proxies_prev, lookback_trend)
            score_delta = int(score - score_prev)
    except Exception:
        score_prev  = None
        score_delta = None

    momentum_label = _momentum_label(score_delta)
    allocation     = _allocation_from_components(score, components)
    favored_groups = _favored_groups_from_allocation(score, components, allocation)

    return RegimeResult(
        score          = score,
        label          = label,
        confidence     = confidence,
        components     = components,
        summary        = summary,
        momentum_label = momentum_label,
        score_prev     = score_prev,
        score_delta    = score_delta,
        allocation     = allocation,
        favored_groups = favored_groups,
    )


def compute_regime_timeseries(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    lookback_trend: int = 63,
    freq: str = "W-FRI",
    min_points: int = 60,
) -> pd.DataFrame:
    """Weekly score history used by the history chart."""
    if macro is None or macro.empty or proxies is None or proxies.empty:
        return pd.DataFrame(columns=["score"])

    idx = macro.index.intersection(proxies.index)
    if len(idx) < min_points:
        return pd.DataFrame(columns=["score"])

    dates    = pd.DatetimeIndex(idx).sort_values().to_series().resample(freq).last().dropna().index
    scores   = []
    out_dates = []

    for d in dates:
        m_cut = macro.loc[macro.index <= d]
        p_cut = proxies.loc[proxies.index <= d]
        if m_cut.empty or p_cut.empty:
            continue
        try:
            s, _, _, _, _ = _compute_core(m_cut, p_cut, lookback_trend)
            scores.append(int(s))
            out_dates.append(d)
        except Exception:
            continue

    if not out_dates:
        return pd.DataFrame(columns=["score"])

    return pd.DataFrame({"score": scores}, index=pd.DatetimeIndex(out_dates)).sort_index()