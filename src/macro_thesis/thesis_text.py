"""
src/macro_thesis/thesis_text.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 8)

Deterministic string templates -- same state always produces the same
words, no runtime LLM call, per spec.

**Rewritten 2026-09-16 in response to a real logic bug (found independently
by the user and by the frontend session testing live data):** the previous
version picked triggers from a single fixed candidate list regardless of
which quadrant a scenario was even describing, so base/alternate1/
alternate2/invalidation all emitted the literal same string ("10y breakeven
above 2.50") even when Alternate 1 (Reflation) and Alternate 2 (Deflation)
are opposite outcomes that cannot share a directional trigger. Triggers are
now derived per-scenario from which axis (growth, inflation, or both)
actually differs between the current quadrant and the target quadrant --
see `build_scenario_trigger`.

**Base case reframing:** the base case is "the current regime persists",
which can have a LOW probability (regimes don't always persist) -- showing
an 8% "base case" next to two 46% "alternates" read as a framing bug, not
a low number. Relabeled CURRENT REGIME vs MOST LIKELY NEXT with an explicit
persistence-note sentence when persistence is genuinely unlikely, per the
user's own diagnosis of the right fix (chose interpretation (b): rename
rather than reorder, since spec section 8 defines the base case as "the
current state's narrative", not "the modal forecast").

**Level-aware narrative:** momentum sets the quadrant, but a level far from
where momentum points changes what a reader should take from it --
"growth strengthening from a well below-trend base" is a different claim
than "growth strengthening" alone. Every quadrant sentence now says both.

**Simplification, still disclosed:** the spec asks triggers to be "derived
from the actual thresholds that would flip a component's sign." This picks
ONE headline series per axis (10y breakeven for inflation, initial claims
for growth) rather than solving the exact multi-input tipping point --
specific and checkable (a real series, a real current value, a real
threshold), just not a mathematically exact composite-level derivative.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# (growth_positive, inflation_positive) required for each quadrant --
# the same sign rule as engine.assign_quadrant, inverted for lookup.
QUADRANT_SIGNS: Dict[str, Tuple[bool, bool]] = {
    "GOLDILOCKS": (True, False),
    "REFLATION": (True, True),
    "STAGFLATION": (False, True),
    "DEFLATION": (False, False),
}

QUADRANT_TAILS = {
    "GOLDILOCKS": "the classic backdrop for risk assets and duration alike.",
    "REFLATION": "a backdrop that favors cyclicals and real assets over duration.",
    "STAGFLATION": "the hardest backdrop to hedge, favoring real assets and cash over both equities and bonds.",
    "DEFLATION": "a flight-to-quality backdrop that favors duration and defensives.",
}

# One headline, checkable series per axis. (column, label, base_step, unit, display_scale)
GROWTH_TRIGGER = ("init_claims_level", "initial claims", 25, "k", 1.0 / 1000.0)
INFLATION_TRIGGER = ("i_breakeven_10y", "10y breakeven", 0.25, "", 1.0)


def _round_to_step(value: float, step: float, direction: str) -> float:
    """Next round multiple of `step` strictly above/below `value`. Uses
    floor/ceil rather than int() truncation because inputs like curve_2s10s
    can be negative, and int() truncates toward zero instead of flooring,
    which silently picks the wrong neighbor for negative values."""
    if direction == "above":
        return (math.floor(value / step) + 1) * step
    return (math.ceil(value / step) - 1) * step


def _format_trigger(column: str, label: str, step: float, direction: str, unit: str, scale: float, raw_row: Dict[str, Any]) -> Optional[str]:
    value = raw_row.get(column)
    if value is None:
        return None
    # Scale to display units BEFORE rounding -- see engine/confirmation
    # docstrings on the hy_oas/claims unit bugs this guards against.
    scaled_value = float(value) * scale
    threshold = _round_to_step(scaled_value, step, direction)
    if unit == "k":
        threshold_disp = "{:.0f}k".format(threshold)
    elif unit == "bp":
        threshold_disp = "{:.0f}bp".format(threshold)
    else:
        threshold_disp = "{:.2f}".format(threshold)
    return "{} {} {}".format(label, direction, threshold_disp)


def _axis_trigger(axis: str, want_positive: bool, raw_row: Dict[str, Any], step_multiplier: float = 1.0) -> Optional[str]:
    if axis == "growth":
        column, label, step, unit, scale = GROWTH_TRIGGER
        direction = "below" if want_positive else "above"  # growth+ means claims LOW/falling
    else:
        column, label, step, unit, scale = INFLATION_TRIGGER
        direction = "above" if want_positive else "below"  # inflation+ means breakeven HIGH/rising
    return _format_trigger(column, label, step * step_multiplier, direction, unit, scale, raw_row)


def build_scenario_trigger(
    current_quadrant: Optional[str],
    target_quadrant: Optional[str],
    raw_row: Dict[str, Any],
    step_multiplier: float = 1.0,
) -> Optional[str]:
    """The condition that would move FROM current TO target: only the
    axis/axes that actually differ between the two quadrants' required
    signs get a trigger. This is what fixes the "every scenario shares one
    trigger" bug -- Reflation (inflation flips up from Goldilocks) and
    Deflation (growth flips down from Goldilocks) now get genuinely
    different, direction-correct triggers instead of both defaulting to
    the same fixed candidate."""
    if not current_quadrant or not target_quadrant:
        return None
    if current_quadrant not in QUADRANT_SIGNS or target_quadrant not in QUADRANT_SIGNS:
        return None
    if current_quadrant == target_quadrant:
        return None

    cur_g, cur_i = QUADRANT_SIGNS[current_quadrant]
    tgt_g, tgt_i = QUADRANT_SIGNS[target_quadrant]

    parts = []
    if cur_g != tgt_g:
        t = _axis_trigger("growth", tgt_g, raw_row, step_multiplier)
        if t:
            parts.append(t)
    if cur_i != tgt_i:
        t = _axis_trigger("inflation", tgt_i, raw_row, step_multiplier)
        if t:
            parts.append(t)
    return " and ".join(parts) if parts else None


def _level_phrase(level: Optional[float]) -> str:
    if level is None:
        return "an unclear base"
    if level > 0.5:
        return "a well above-trend base"
    if level > 0.15:
        return "an above-trend base"
    if level > -0.15:
        return "a near-trend base"
    if level > -0.5:
        return "a below-trend base"
    return "a deeply below-trend base"


def quadrant_sentence(
    quadrant: Optional[str],
    growth_level: Optional[float],
    growth_momentum: Optional[float],
    inflation_level: Optional[float],
    inflation_momentum: Optional[float],
) -> str:
    if quadrant not in QUADRANT_TAILS:
        return "The read is currently unavailable."
    growth_verb = "strengthening" if (growth_momentum or 0) >= 0 else "weakening"
    inflation_verb = "accelerating" if (inflation_momentum or 0) >= 0 else "cooling"
    return "Growth is {} from {}, while inflation is {} from {} -- {}".format(
        growth_verb, _level_phrase(growth_level), inflation_verb, _level_phrase(inflation_level),
        QUADRANT_TAILS[quadrant],
    )


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
    """Uses the confirms/diverges/neutral structure directly rather than a
    single opaque score string -- "the tape agrees" is only true when
    confirms actually outnumbers diverges, not whenever diverges < 3."""
    if not confirmation or not confirmation.get("rows"):
        return ""
    if confirmation.get("tapeNotExpressingView"):
        return "The tape isn't expressing a view right now: {}.".format(confirmation.get("confirmationLabel", ""))
    if confirmation.get("modelUnderReview"):
        return "The tape disagrees: {} -- enough divergence that the model itself is under review.".format(
            confirmation.get("confirmationLabel", "")
        )
    confirms, diverges = confirmation.get("confirms", 0), confirmation.get("diverges", 0)
    if confirms > diverges:
        return "The tape agrees: {}.".format(confirmation.get("confirmationLabel", ""))
    if diverges > confirms:
        return "The tape leans against this read: {}.".format(confirmation.get("confirmationLabel", ""))
    return "The tape is mixed: {}.".format(confirmation.get("confirmationLabel", ""))


def _persistence_note(probability: Optional[float]) -> Optional[str]:
    if probability is None:
        return None
    if probability < 0.20:
        return "This regime rarely persists from here -- historically it has been more likely to shift than to continue over the next 3 months."
    if probability < 0.40:
        return "This regime persists less often than not from here over the next 3 months."
    return None


def build_base_case(
    quadrant: Optional[str],
    phase_info: Optional[Dict],
    confirmation: Optional[Dict],
    raw_row: Dict[str, Any],
    growth_level: Optional[float] = None,
    growth_momentum: Optional[float] = None,
    inflation_level: Optional[float] = None,
    inflation_momentum: Optional[float] = None,
    probability: Optional[float] = None,
    assets: Optional[List[Dict]] = None,
    what_would_change_it: Optional[str] = None,
) -> Dict[str, Any]:
    """`probability` is P(still in this quadrant at t+3m). `expression`
    mirrors alternates[].expression: top-ranked assets for the current
    quadrant. `whatWouldChangeIt` is now the SAME trigger as the top-ranked
    alternate (the most likely way the regime actually changes), not an
    independently-derived generic pick -- deriving it from the same source
    as alternates[0].trigger is what guarantees they can never silently
    diverge into two different "what changes it" claims."""
    sentence = quadrant_sentence(quadrant, growth_level, growth_momentum, inflation_level, inflation_momentum)
    note = _persistence_note(probability)
    text = " ".join(filter(None, [sentence, note, phase_sentence(phase_info), confirmation_sentence(confirmation)]))
    return {
        "label": "CURRENT REGIME",
        "quadrant": quadrant,
        "text": text,
        "whatWouldChangeIt": [what_would_change_it] if what_would_change_it else [],
        "probability": probability,
        "persistenceNote": note,
        "expression": [a["asset"] for a in (assets or [])[:3]],
    }


def build_alternate(
    rank_label: str,
    current_quadrant: Optional[str],
    quadrant: str,
    probability: Optional[float],
    sample_size: int,
    assets: Optional[List[Dict]],
    raw_row: Dict[str, Any],
) -> Dict[str, Any]:
    top_assets = [a["asset"] for a in (assets or [])[:3]]
    trigger = build_scenario_trigger(current_quadrant, quadrant, raw_row) or "a shift in the dominant axis"
    return {
        "label": rank_label,
        "quadrant": quadrant,
        "probability": probability,
        "sampleSize": sample_size,
        "trigger": trigger,
        "expression": top_assets,
    }


def build_invalidation(
    current_quadrant: Optional[str],
    primary_target_quadrant: Optional[str],
    raw_row: Dict[str, Any],
    existing_triggers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Explicit, pre-committed: a named level, named series, named date
    (spec section 8), and per the user's explicit requirement, DISTINCT
    from every scenario trigger on the page -- this is "the framework is
    wrong" level, not "the view shifts" level. Uses a 2x (then 3x, 4x if
    still colliding) step multiple on the same axis the most likely
    alternate would flip, so it's a decisive break past that trigger, not
    the same threshold restated."""
    existing = set(existing_triggers or [])
    condition = None
    for multiplier in (2.0, 3.0, 4.0):
        candidate = build_scenario_trigger(current_quadrant, primary_target_quadrant, raw_row, step_multiplier=multiplier)
        if candidate and candidate not in existing:
            condition = candidate
            break
    if condition is None:
        condition = build_scenario_trigger(current_quadrant, primary_target_quadrant, raw_row, step_multiplier=2.0)

    return {
        "condition": condition or "no clear invalidation level available",
        "reviewBy": None,  # filled in by snapshot.py with as_of + 3 months
        "note": (
            "A decisive break past the level that would merely shift the view to the "
            "next-most-likely scenario -- not the same threshold. If reached before the "
            "review date, treat the whole framework as wrong and re-derive it, not patch it."
        ),
    }
