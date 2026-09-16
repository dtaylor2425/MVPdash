"""
src/macro_thesis/confirmation.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 7)

Compares the model's quadrant-implied direction for five markets against
what price has actually done over the last 20 sessions. Only the quadrants
the spec explicitly names for a market get a directional implication --
the other two are marked "neutral" (no claim) rather than inventing a prior
the spec never stated, EXCEPT for equities and credit, where "risk-on in
Goldilocks/Reflation" is a complete binary framing and the converse
(risk-off -> down / widen) for Stagflation/Deflation is a direct, not
invented, reading of the same sentence.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

# direction per quadrant; "neutral" = spec states no prior for this cell
MODEL_IMPLICATIONS = {
    "equities": {"GOLDILOCKS": "up", "REFLATION": "up", "STAGFLATION": "down", "DEFLATION": "down"},
    "credit": {"GOLDILOCKS": "tighten", "REFLATION": "tighten", "STAGFLATION": "widen", "DEFLATION": "widen"},
    "duration": {"GOLDILOCKS": "neutral", "REFLATION": "rise", "STAGFLATION": "neutral", "DEFLATION": "fall"},
    "dollar": {"GOLDILOCKS": "neutral", "REFLATION": "neutral", "STAGFLATION": "up", "DEFLATION": "up"},
    "commodities": {"GOLDILOCKS": "neutral", "REFLATION": "up", "STAGFLATION": "up", "DEFLATION": "neutral"},
}

# Below this absolute 20-session move, call it "neutral" rather than reading
# noise as confirmation or divergence. Judgment calls, disclosed here.
NEUTRAL_THRESHOLDS = {
    "equities": 0.01,      # 1% 20d return
    "credit": 0.10,        # 10bp OAS change
    "duration": 0.10,      # 10bp yield change
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


def _chg_20d(series: pd.Series, is_pct: bool) -> Optional[float]:
    s = series.dropna()
    if len(s) < 21:
        return None
    if is_pct:
        return float(s.iloc[-1] / s.iloc[-21] - 1)
    return float(s.iloc[-1] - s.iloc[-21])


def compute_cross_asset_confirmation(quadrant: str, macro: pd.DataFrame, prices: pd.DataFrame) -> Dict:
    if quadrant not in MODEL_IMPLICATIONS["equities"]:
        return {"rows": [], "confirmationScore": None, "modelUnderReview": False}

    actuals = {
        "equities": _chg_20d(prices["SPY"], is_pct=True) if "SPY" in prices.columns else None,
        "credit": _chg_20d(macro["hy_oas"], is_pct=False) if "hy_oas" in macro.columns else None,
        "duration": _chg_20d(macro["y10"], is_pct=False) if "y10" in macro.columns else None,
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

    return {
        "rows": rows,
        "confirmationScore": "{} of {}".format(confirming, len(rows)),
        "confirmationRatio": round(confirming / len(rows), 3),
        "modelUnderReview": diverging >= 3,
    }
