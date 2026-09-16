"""
src/macro_thesis/history.py  (docs/MACRO-THESIS-BACKEND-SPEC.md sections 5, 6)

Empirical transition probabilities and asset returns by quadrant. Both are
computed once from the quadrant history and refreshed monthly per spec
section 9 -- static tables, not live signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

MIN_TRANSITION_N = 12
MIN_ASSET_EPISODES = 24
QUADRANTS = ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]


def build_transition_frame(quadrant_series: pd.Series, phase_series: pd.Series, horizon_months: int) -> pd.DataFrame:
    idx = quadrant_series.index
    frame = pd.DataFrame({
        "from_quadrant": quadrant_series,
        "from_phase": phase_series.reindex(idx) if phase_series is not None else None,
        "to_quadrant": quadrant_series.shift(-horizon_months),
    })
    return frame.dropna(subset=["from_quadrant", "to_quadrant"])


def transition_probabilities(
    quadrant_series: pd.Series,
    phase_series: Optional[pd.Series],
    current_quadrant: str,
    current_phase: Optional[str],
    horizon_months: int,
    min_n: int = MIN_TRANSITION_N,
) -> Dict:
    """P(quadrant at t+horizon | quadrant at t [, phase at t]). Conditions on
    (quadrant, phase) only if that finer cut still clears min_n -- spec 5:
    "condition on the current state where sample size allows: same quadrant
    AND same short-cycle phase" -- otherwise falls back to quadrant alone.
    These are historical frequencies, not forecasts (spec's own wording,
    carried into the payload label)."""
    frame = build_transition_frame(quadrant_series, phase_series, horizon_months)
    conditioned_on = ["quadrant"]
    subset = frame[frame["from_quadrant"] == current_quadrant]

    if current_phase is not None and phase_series is not None:
        finer = subset[subset["from_phase"] == current_phase]
        if len(finer) >= min_n:
            subset = finer
            conditioned_on = ["quadrant", "phase"]

    n = len(subset)
    if n < min_n:
        return {
            "horizonMonths": horizon_months, "from": current_quadrant,
            "conditionedOn": conditioned_on, "sampleSize": n,
            "probabilities": None, "suppressed": True,
            "note": "Historical frequency, not a forecast. Suppressed: fewer than {} observations.".format(min_n),
        }

    counts = subset["to_quadrant"].value_counts(normalize=True)
    return {
        "horizonMonths": horizon_months, "from": current_quadrant,
        "conditionedOn": conditioned_on, "sampleSize": n,
        "probabilities": {q: round(float(counts.get(q, 0.0)), 3) for q in QUADRANTS},
        "suppressed": False,
        "note": "Historical frequency, not a forecast.",
    }


def compute_asset_returns_by_quadrant(
    quadrant_series: pd.Series,
    monthly_prices: pd.DataFrame,
    universe: List[str],
    forward_months: int = 3,
    min_episodes: int = MIN_ASSET_EPISODES,
) -> Dict[str, List[Dict]]:
    """Forward-only: the quadrant at t maps to the return from t to t+3m.
    `pct_change(3).shift(-3)` at row t0 yields (P[t0+3]-P[t0])/P[t0] --
    realized strictly after t0, using only the price level established AT
    t0 as the baseline (see spec validation #4)."""
    fwd_return = monthly_prices.pct_change(forward_months).shift(-forward_months)

    results: Dict[str, List[Dict]] = {}
    for quadrant in QUADRANTS:
        mask = quadrant_series.reindex(fwd_return.index) == quadrant
        rows = []
        for ticker in universe:
            if ticker not in fwd_return.columns:
                continue
            rets = fwd_return.loc[mask, ticker].dropna()
            n = len(rets)
            if n < min_episodes:
                continue
            rows.append({
                "asset": ticker,
                "avg3mFwd": round(float(rets.mean()), 4),
                "hitRate": round(float((rets > 0).mean()), 4),
                "episodes": n,
            })
        rows.sort(key=lambda r: -r["avg3mFwd"])
        results[quadrant] = rows
    return results
