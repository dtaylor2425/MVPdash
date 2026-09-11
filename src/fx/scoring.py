"""
src/fx/scoring.py

Cross-sectional z-scoring, winsorisation, component drop + reweight, composite,
0-100 display mapping, and antisymmetric pair derivation (spec sections 4.2-4.4).

Invariants this module guarantees (checked in tests/test_fx.py):
  * weights used for a currency always sum to 1.0 (after reweighting)
  * no NaN in any output -- unusable inputs are simply absent
  * pair(A,B).z == -pair(B,A).z and pair(A,A).z == 0 by construction
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.fx.components import COMPONENTS
from src.fx.series_map import INTERVENTION_WATCH, SCORED

# Five active components. macroVsMandate (originally 20%) was dropped entirely
# (see src/fx/components.py) -- the remaining five weights are the original
# ones each scaled by 100/80 so they still sum to 1.0.
COMPONENT_WEIGHTS: Dict[str, float] = {
    "carry": 0.3125,
    "policyMomentum": 0.3125,
    "termsOfTrade": 0.125,
    "trend": 0.125,
    "valuation": 0.125,
}

MIN_CURRENCIES_PER_COMPONENT = 6      # spec 4.2
WINSOR_SIGMA = 2.5                    # spec 4.2
SIGMA_TO_DISPLAY = 12.5               # spec 4.2 -- score = 50 + 12.5 * z
NEUTRAL = 50.0

CARRY_TREND_CONFLICT_CARRY_Z = 0.5
CARRY_TREND_CONFLICT_TREND_Z = -0.5
INTERVENTION_TREND_Z = 1.5


def _finite(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# cross-sectional z
# ---------------------------------------------------------------------------
def cross_sectional_z(raw_by_ccy: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Z-score across currencies for one component at one date.

    Returns {} to signal "drop this component" when fewer than
    MIN_CURRENCIES_PER_COMPONENT currencies have valid data.
    Winsorised at +/- WINSOR_SIGMA before returning.
    """
    clean = {k: v for k in raw_by_ccy for v in [_finite(raw_by_ccy[k])] if v is not None}
    if len(clean) < MIN_CURRENCIES_PER_COMPONENT:
        return {}
    arr = np.array(list(clean.values()), dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std == 0.0 or not math.isfinite(std):
        return {k: 0.0 for k in clean}
    out: Dict[str, float] = {}
    for k, v in clean.items():
        z = (v - mean) / std
        out[k] = max(-WINSOR_SIGMA, min(WINSOR_SIGMA, z))
    return out


def z_from_components(comp_raw: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, float]]:
    """
    comp_raw: {component -> {ccy -> ComponentValue}}
    returns:  {component -> {ccy -> z}}   (empty inner dict == dropped)
    """
    result: Dict[str, Dict[str, float]] = {}
    for comp in COMPONENTS:
        by_ccy = comp_raw.get(comp, {})
        raw = {
            ccy: (cv.get("raw") if cv.get("available") else None)
            for ccy, cv in by_ccy.items()
        }
        result[comp] = cross_sectional_z(raw)
    return result


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------
def composite_scores(
    comp_z: Dict[str, Dict[str, float]],
    currencies: List[str] = None,
) -> Dict[str, dict]:
    """
    Weighted sum of component z-scores per currency, reweighting across the
    components actually available for that currency.

    Returns {ccy -> {"z", "display", "weights", "componentZ", "droppedComponents"}}.
    """
    currencies = currencies or SCORED
    globally_available = {c for c in COMPONENTS if comp_z.get(c)}
    out: Dict[str, dict] = {}
    for ccy in currencies:
        have: Dict[str, float] = {}
        for comp in COMPONENTS:
            z = comp_z.get(comp, {}).get(ccy)
            zf = _finite(z)
            if zf is not None:
                have[comp] = zf
        if not have:
            out[ccy] = {
                "z": 0.0,
                "display": NEUTRAL,
                "weights": {},
                "componentZ": {},
                "droppedComponents": sorted(globally_available),
            }
            continue
        wsum = sum(COMPONENT_WEIGHTS[c] for c in have)
        weights = {c: COMPONENT_WEIGHTS[c] / wsum for c in have}
        cz = sum(weights[c] * have[c] for c in have)
        out[ccy] = {
            "z": float(cz),
            "display": _to_display(cz),
            "weights": weights,
            "componentZ": have,
            "droppedComponents": sorted(globally_available - set(have)),
        }
    return out


def _to_display(z: float) -> float:
    return round(max(0.0, min(100.0, NEUTRAL + SIGMA_TO_DISPLAY * z)), 1)


def rank_currencies(composites: Dict[str, dict]) -> Dict[str, int]:
    order = sorted(composites.keys(), key=lambda c: composites[c]["z"], reverse=True)
    return {c: i + 1 for i, c in enumerate(order)}


# ---------------------------------------------------------------------------
# flags (spec 4.4)
# ---------------------------------------------------------------------------
def currency_flags(
    ccy: str,
    comp_z: Dict[str, Dict[str, float]],
    composite: dict,
    trend_1m_z: Dict[str, float],
    comp_raw: Dict[str, Dict[str, dict]],
) -> dict:
    carry_z = comp_z.get("carry", {}).get(ccy)
    trend_z = comp_z.get("trend", {}).get(ccy)
    conflict = (
        carry_z is not None and trend_z is not None
        and carry_z > CARRY_TREND_CONFLICT_CARRY_Z
        and trend_z < CARRY_TREND_CONFLICT_TREND_Z
    )

    t1 = trend_1m_z.get(ccy)
    intervention = (
        ccy in INTERVENTION_WATCH and t1 is not None and abs(t1) > INTERVENTION_TREND_Z
    )

    proxy_inputs = sorted(
        comp for comp in COMPONENTS
        if comp_raw.get(comp, {}).get(ccy, {}).get("available")
        and comp_raw.get(comp, {}).get(ccy, {}).get("proxy")
    )

    return {
        "carryTrendConflict": bool(conflict),
        "interventionRisk": bool(intervention),
        "dataIncomplete": bool(composite.get("droppedComponents")),
        "proxyInputs": proxy_inputs,
    }


# ---------------------------------------------------------------------------
# pairs (spec 4.3) -- one triangle, antisymmetric by construction
# ---------------------------------------------------------------------------
def derive_pairs(
    composites: Dict[str, dict],
    flags_by_ccy: Dict[str, dict],
    currencies: List[str] = None,
) -> List[dict]:
    currencies = currencies or [c for c in SCORED if c in composites]
    pairs: List[dict] = []
    for i, base in enumerate(currencies):
        for quote in currencies[i + 1:]:
            z = composites[base]["z"] - composites[quote]["z"]
            bf = flags_by_ccy.get(base, {})
            qf = flags_by_ccy.get(quote, {})
            pairs.append({
                "base": base,
                "quote": quote,
                "z": round(float(z), 4),
                "display": _to_display(z),
                "flags": {
                    "interventionRisk": bool(bf.get("interventionRisk") or qf.get("interventionRisk")),
                    "carryTrendConflict": bool(bf.get("carryTrendConflict") or qf.get("carryTrendConflict")),
                    "dataIncomplete": bool(bf.get("dataIncomplete") or qf.get("dataIncomplete")),
                },
            })
    return pairs


def pair_lookup(pairs: List[dict]) -> Dict[Tuple[str, str], float]:
    """Full antisymmetric matrix (both directions + zero diagonal) for tests / API."""
    m: Dict[Tuple[str, str], float] = {}
    seen = set()
    for p in pairs:
        m[(p["base"], p["quote"])] = p["z"]
        m[(p["quote"], p["base"])] = -p["z"]
        seen.add(p["base"])
        seen.add(p["quote"])
    for c in seen:
        m[(c, c)] = 0.0
    return m


def mirror_pairs_for_base(pairs: List[dict], base: str) -> List[dict]:
    """Every pair involving `base`, oriented with `base` first."""
    out: List[dict] = []
    for p in pairs:
        if p["base"] == base:
            out.append(dict(p))
        elif p["quote"] == base:
            out.append({
                "base": base,
                "quote": p["base"],
                "z": round(-p["z"], 4),
                "display": _to_display(-p["z"]),
                "flags": p["flags"],
            })
    out.sort(key=lambda x: x["z"], reverse=True)
    return out


def weights_sum_ok(composites: Dict[str, dict], tol: float = 1e-9) -> bool:
    for c in composites.values():
        w = c.get("weights") or {}
        if w and abs(sum(w.values()) - 1.0) > tol:
            return False
    return True
