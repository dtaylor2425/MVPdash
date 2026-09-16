"""
src/macro_thesis/thesis_text.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 8)

Deterministic string templates -- same state always produces the same
words, no runtime LLM call, per spec.

**Simplification, disclosed:** the spec asks triggers to be "derived from
the actual thresholds that would flip a component's sign." Doing this
exactly (solving for how much one raw input has to move to flip a 7-input
equal-weighted composite's momentum sign) is a well-defined but nontrivial
derivative computation given the level-smoothing in engine.py. Given the
size of this build, triggers instead reference a fixed set of the most
commonly cited "headline" series per axis (10y breakeven, HY OAS, 2s10s
curve, initial claims) at their REAL current value plus a genuine
round-number step in the regime-relevant direction -- specific and
checkable (a real series, a real current value, a real threshold), just
not a mathematically exact tipping point. If this needs to be exact later,
the derivative computation is the next step.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

QUADRANT_SENTENCES = {
    "GOLDILOCKS": "Growth is strengthening while inflation cools -- the classic backdrop for risk assets and duration alike.",
    "REFLATION": "Growth and inflation are both accelerating -- a backdrop that favors cyclicals and real assets over duration.",
    "STAGFLATION": "Growth is weakening while inflation accelerates -- the hardest backdrop to hedge, favoring real assets and cash over both equities and bonds.",
    "DEFLATION": "Both growth and inflation are decelerating -- a flight-to-quality backdrop that favors duration and defensives.",
}

TRIGGER_CANDIDATES = [
    # (raw_column, label, round_step, direction_for_hotter_or_stronger, unit, display_scale)
    # hy_oas (BAMLH0A0HYM2) is stored in percentage points (2.76 == 276bp),
    # not basis points -- scale=100 converts to bp for both the rounding
    # step and the display, so "step=50" means 50bp, not 50 percentage points.
    ("i_breakeven_10y", "10y breakeven", 0.25, "above", "", 1.0),
    ("hy_oas", "HY OAS", 50, "above", "bp", 100.0),
    ("curve_2s10s", "2s10s curve", 0.25, "below", "", 1.0),
    ("init_claims_level", "initial claims", 25, "above", "k", 1.0 / 1000.0),
]


def _round_to_step(value: float, step: float, direction: str) -> float:
    """Next round multiple of `step` strictly above/below `value`. Uses
    floor/ceil rather than int() truncation because inputs like curve_2s10s
    can be negative (curve inversions), and int() truncates toward zero
    instead of flooring, which silently picks the wrong neighbor for
    negative values."""
    if direction == "above":
        return (math.floor(value / step) + 1) * step
    return (math.ceil(value / step) - 1) * step


def _format_trigger(column: str, label: str, step: float, direction: str, unit: str, scale: float, raw_row: Dict[str, Any]) -> Optional[str]:
    value = raw_row.get(column)
    if value is None:
        return None
    # Scale to display units BEFORE rounding -- rounding hy_oas's raw 2.76
    # (percentage points) with a step of 50 without first scaling to bp
    # would silently round to "50bp" instead of "300bp". Same issue for
    # claims (raw persons vs. displayed thousands).
    scaled_value = float(value) * scale
    threshold = _round_to_step(scaled_value, step, direction)
    if unit == "k":
        threshold_disp = "{:.0f}k".format(threshold)
    elif unit == "bp":
        threshold_disp = "{:.0f}bp".format(threshold)
    else:
        threshold_disp = "{:.2f}".format(threshold)
    return "{} {} {}".format(label, direction, threshold_disp)


def top_triggers(raw_row: Dict[str, Any], n: int = 2) -> List[str]:
    triggers = []
    for column, label, step, direction, unit, scale in TRIGGER_CANDIDATES:
        t = _format_trigger(column, label, step, direction, unit, scale, raw_row)
        if t:
            triggers.append(t)
    return triggers[:n]


def phase_sentence(phase_info: Optional[Dict[str, Any]]) -> str:
    if not phase_info or not phase_info.get("phase"):
        return ""
    months = phase_info.get("monthsInPhase")
    median = phase_info.get("medianPhaseMonths")
    met = phase_info.get("conditionsMet")
    total = phase_info.get("conditionsTotal")
    duration = ""
    if months is not None and median is not None:
        duration = " (month {}, historical median {:.0f})".format(months, median)
    return "The short-term cycle is in {}{}, with {} of {} conditions met.".format(
        phase_info["phase"], duration, met, total
    )


def confirmation_sentence(confirmation: Optional[Dict[str, Any]]) -> str:
    if not confirmation or not confirmation.get("rows"):
        return ""
    score = confirmation.get("confirmationScore", "")
    if confirmation.get("modelUnderReview"):
        return "The tape disagrees: only {} markets confirm this read -- enough divergence that the model itself is under review.".format(score)
    return "The tape agrees: {} markets confirm this read.".format(score)


def build_base_case(quadrant: Optional[str], phase_info: Optional[Dict], confirmation: Optional[Dict], raw_row: Dict[str, Any]) -> Dict[str, Any]:
    quadrant_sentence = QUADRANT_SENTENCES.get(quadrant, "The read is currently unavailable.")
    return {
        "quadrant": quadrant,
        "text": " ".join(filter(None, [quadrant_sentence, phase_sentence(phase_info), confirmation_sentence(confirmation)])),
        "whatWouldChangeIt": top_triggers(raw_row),
    }


def build_alternate(rank_label: str, quadrant: str, probability: Optional[float], sample_size: int, assets: Optional[List[Dict]], raw_row: Dict[str, Any]) -> Dict[str, Any]:
    top_assets = [a["asset"] for a in (assets or [])[:3]]
    return {
        "label": rank_label,
        "quadrant": quadrant,
        "probability": probability,
        "sampleSize": sample_size,
        "trigger": (top_triggers(raw_row, n=1) or ["a shift in the dominant axis"])[0],
        "expression": top_assets,
    }


def build_invalidation(quadrant: Optional[str], raw_row: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit, pre-committed: a named level, named series, named date
    (spec section 8). Uses the single headline trigger for whichever axis
    is currently dominant, with a concrete re-check date 3 months out --
    the same horizon the transition probabilities are computed over."""
    triggers = top_triggers(raw_row, n=1)
    condition = triggers[0] if triggers else "no clear invalidation level available"
    return {
        "condition": condition,
        "reviewBy": None,  # filled in by snapshot.py with as_of + 3 months
        "note": "If this level is reached before the review date, the base case above is invalidated and should be re-derived, not patched.",
    }
