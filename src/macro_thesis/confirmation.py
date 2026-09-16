"""
src/macro_thesis/confirmation.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 7)

Compares the model's quadrant-implied direction for five markets against
what price has actually done over the last 20 sessions.

**Complete implication table (fixed 2026-09-16):** the spec only names 2 of
4 quadrants for duration/dollar/commodities; the other cells used to read
"neutral". A neutral MODEL IMPLICATION means that market can never confirm
or diverge from anything, which defeats the point of a 5-market check when
3 of 5 are neutral 60% of the time. Completed using standard cross-asset
priors for the missing cells (documented per-cell below) -- these are a
judgment call, not spec text, and are disclosed as such.

**Unit bug fixed 2026-09-16:** `hy_oas` and `y10` are stored in percentage
points (2.76 == 276bp), not basis points -- same class of bug as the one
found in thesis_text.py's triggers. `actual20d` for credit/duration is now
returned already converted to real basis points so a consumer never has to
guess the multiplier from the label string.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

# Direction per quadrant. Cells marked "spec" are direct spec-section-7
# text; cells marked "prior" are a standard cross-asset judgment call for
# the quadrants the spec didn't name, added so no market implication is
# ever neutral (a neutral implication can't confirm or diverge from
# anything, so having one for 60% of markets defeated the confirmation
# check's purpose).
MODEL_IMPLICATIONS = {
    # spec: risk-on in Goldilocks/Reflation -> converse is risk-off/down
    "equities": {"GOLDILOCKS": "up", "REFLATION": "up", "STAGFLATION": "down", "DEFLATION": "down"},
    # spec: spreads tighten in risk-on quadrants -> converse is widen
    "credit": {"GOLDILOCKS": "tighten", "REFLATION": "tighten", "STAGFLATION": "widen", "DEFLATION": "widen"},
    # spec names DEFLATION=fall, REFLATION=rise. GOLDILOCKS (prior): growth+/inflation-
    # is broadly duration-supportive (falling term/inflation premium) -> fall.
    # STAGFLATION (prior): inflation/term premium historically dominates weak-growth
    # pull in real episodes (e.g. 2022) -> rise.
    "duration": {"GOLDILOCKS": "fall", "REFLATION": "rise", "STAGFLATION": "rise", "DEFLATION": "fall"},
    # spec names DEFLATION=up, STAGFLATION=up (flight to safety both times).
    # GOLDILOCKS/REFLATION (prior): broad risk-on capital rotation out of the
    # dollar into cyclicals/EM -> down in both.
    "dollar": {"GOLDILOCKS": "down", "REFLATION": "down", "STAGFLATION": "up", "DEFLATION": "up"},
    # spec names REFLATION=up, STAGFLATION=up (supply-driven or reflationary demand).
    # GOLDILOCKS/DEFLATION (prior): soft-landing/demand-destruction backdrops are
    # commodity-bearish -> down in both.
    "commodities": {"GOLDILOCKS": "down", "REFLATION": "up", "STAGFLATION": "up", "DEFLATION": "down"},
}

# Below this absolute 20-session move, call it "neutral" rather than reading
# noise as confirmation or divergence. Judgment calls, disclosed here. Units
# match what's now returned in `actual20d` (bp for credit/duration).
NEUTRAL_THRESHOLDS = {
    "equities": 0.01,      # 1% 20d return
    "credit": 10.0,        # 10bp OAS change
    "duration": 10.0,      # 10bp yield change
    "dollar": 0.005,       # 0.5% 20d change
    "commodities": 0.02,   # 2% 20d return
}

UP_LIKE = {"up", "rise", "widen"}
DOWN_LIKE = {"down", "fall", "tighten"}


def _verdict(implication: str, actual_change: Optional[float], threshold: float) -> str:
    if implication == "neutral" or actual_change is None or pd.isna(actual_change):
        return "neutral"
    if abs(actual_change) < threshold:
        return "neutral"
    actual_up = actual_change > 0
    expected_up = implication in UP_LIKE
    return "confirms" if actual_up == expected_up else "diverges"


def _chg_20d(series: pd.Series, is_pct: bool, scale: float = 1.0) -> Optional[float]:
    s = series.dropna()
    if len(s) < 21:
        return None
    if is_pct:
        return float(s.iloc[-1] / s.iloc[-21] - 1)
    return float((s.iloc[-1] - s.iloc[-21]) * scale)


def compute_cross_asset_confirmation(quadrant: str, macro: pd.DataFrame, prices: pd.DataFrame) -> Dict:
    if quadrant not in MODEL_IMPLICATIONS["equities"]:
        return {
            "rows": [], "confirms": 0, "diverges": 0, "neutral": 0,
            "confirmationScore": None, "confirmationLabel": None,
            "tapeNotExpressingView": False, "modelUnderReview": False,
        }

    actuals = {
        "equities": _chg_20d(prices["SPY"], is_pct=True) if "SPY" in prices.columns else None,
        # scale=100: hy_oas/y10 are percentage points, not bp -- see module docstring.
        "credit": _chg_20d(macro["hy_oas"], is_pct=False, scale=100.0) if "hy_oas" in macro.columns else None,
        "duration": _chg_20d(macro["y10"], is_pct=False, scale=100.0) if "y10" in macro.columns else None,
        "dollar": _chg_20d(macro["dollar_broad"], is_pct=True) if "dollar_broad" in macro.columns else None,
        "commodities": _chg_20d(prices["DBC"], is_pct=True) if "DBC" in prices.columns else None,
    }

    labels = {
        "equities": ("Equities (SPY)", "20d return"),
        "credit": ("Credit (HY OAS)", "20d change (bp)"),
        "duration": ("Duration (10y yield)", "20d change (bp)"),
        "dollar": ("Dollar", "20d change"),
        "commodities": ("Commodities (DBC)", "20d return"),
    }

    rows = []
    for market, (label, actual_label) in labels.items():
        implication = MODEL_IMPLICATIONS[market][quadrant]
        actual = actuals[market]
        verdict = _verdict(implication, actual, NEUTRAL_THRESHOLDS[market])
        rows.append({
            "market": label,
            "modelImplication": implication,
            "actual20d": round(actual, 4) if actual is not None else None,
            "actualLabel": actual_label,
            "state": verdict,
        })

    confirming = sum(1 for r in rows if r["state"] == "confirms")
    diverging = sum(1 for r in rows if r["state"] == "diverges")
    neutral = sum(1 for r in rows if r["state"] == "neutral")
    total = len(rows)

    parts = []
    if diverging:
        parts.append("{} diverge{}".format(diverging, "" if diverging == 1 else "s"))
    if neutral:
        parts.append("{} neutral".format(neutral))
    if confirming:
        parts.append("{} confirm{}".format(confirming, "" if confirming == 1 else "s"))
    confirmation_label = ", ".join(parts) if parts else "no data"

    tape_not_expressing_view = neutral >= 4
    non_neutral = confirming + diverging
    confirmation_ratio = round(confirming / non_neutral, 3) if non_neutral > 0 else None

    return {
        "rows": rows,
        "confirms": confirming,
        "diverges": diverging,
        "neutral": neutral,
        # confirmationScore is computed over NON-neutral markets only (spec
        # intent: a market with nothing to say shouldn't count against the
        # model) -- None (not a misleading "0 of 5") when neutral dominates.
        "confirmationScore": None if tape_not_expressing_view else confirmation_ratio,
        "confirmationLabel": "Tape is not expressing a view" if tape_not_expressing_view else confirmation_label,
        "tapeNotExpressingView": tape_not_expressing_view,
        "modelUnderReview": diverging >= 3,
    }
