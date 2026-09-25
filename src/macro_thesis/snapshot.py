"""
src/macro_thesis/snapshot.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 9)

Orchestrates engine.py + history.py + confirmation.py + thesis_text.py into
the single payload GET /api/macro/thesis serves. Compute happens in the
nightly job (jobs/macro_thesis_job.py); this module is pure computation, no
DB access, so it's unit-testable without Postgres.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.macro_thesis import confirmation as confirmation_mod
from src.macro_thesis import engine
from src.macro_thesis import history as history_mod
from src.macro_thesis import thesis_text
from src.macro_thesis.series_map import ASSET_UNIVERSE

# Which pillar page each evidence-table input belongs to, for the frontend
# to link out to (spec section 9's evidence table: "identifying which
# pillar page it belongs to").
INPUT_PILLAR = {
    "g_init_claims": "growth", "g_cont_claims": "growth", "g_breadth": "growth",
    "g_nfci": "credit", "g_copper_gold": "growth", "g_curve_3m10y": "rates", "g_umich": "growth",
    "i_breakeven_10y": "inflation", "i_breakeven_5y": "inflation", "i_breakeven_5y5y": "inflation",
    "i_cpi_yoy": "inflation", "i_commodity_basket": "inflation", "i_dollar": "inflation",
    "i_real_fed_funds": "rates",
}
INPUT_LABEL = {
    "g_init_claims": "Initial claims", "g_cont_claims": "Continuing claims", "g_breadth": "Breadth (RSP/SPY)",
    "g_nfci": "NFCI", "g_copper_gold": "Copper/Gold", "g_curve_3m10y": "3m10y curve", "g_umich": "Consumer sentiment",
    "i_breakeven_10y": "10y breakeven", "i_breakeven_5y": "5y breakeven", "i_breakeven_5y5y": "5y5y forward",
    "i_cpi_yoy": "CPI YoY", "i_commodity_basket": "Commodity basket (DBC)", "i_dollar": "Dollar (DXY-broad)",
    "i_real_fed_funds": "Real fed funds",
}


def _evidence_table(monthly: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"growth": [], "inflation": []}
    for col in engine.GROWTH_INPUTS + engine.INFLATION_INPUTS:
        if col not in monthly.columns:
            continue
        s = monthly[col].dropna()
        if s.empty:
            continue
        z = engine._expanding_zscore(s)
        current = float(s.iloc[-1])
        z_now = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None
        change_3m = float(s.iloc[-1] - s.iloc[-4]) if len(s) > 3 else None
        axis = "growth" if col.startswith("g_") else "inflation"
        out[axis].append({
            "series": INPUT_LABEL.get(col, col),
            "currentValue": round(current, 3),
            "zscore": round(z_now, 3) if z_now is not None else None,
            "change3m": round(change_3m, 3) if change_3m is not None else None,
            "contributionSign": "positive" if (z_now or 0) >= 0 else "negative",
            "pillar": INPUT_PILLAR.get(col, "growth"),
        })
    return out


def _growth_inflation_trail(quadrant_history: pd.DataFrame, months: int = 12) -> List[Dict[str, Any]]:
    tail = quadrant_history.dropna(subset=["growth_momentum", "inflation_momentum"]).tail(months)
    return [
        {
            "month": ts.strftime("%Y-%m"),
            "growthMomentum": round(float(row["growth_momentum"]), 3),
            "inflationMomentum": round(float(row["inflation_momentum"]), 3),
        }
        for ts, row in tail.iterrows()
    ]


def build_thesis_snapshot(
    macro: pd.DataFrame,
    extra: pd.DataFrame,
    prices_full: pd.DataFrame,
    dsr: Optional[pd.Series],
    house_view: Optional[Dict[str, Any]] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    as_of = as_of or pd.Timestamp.now().normalize()

    monthly = engine.build_monthly_raw(macro, extra, prices_full, dsr, as_of=as_of)
    quadrant_history = engine.compute_quadrant_history(monthly)
    phases = engine.phase_series(quadrant_history, monthly)
    current = engine.current_quadrant_read(quadrant_history)
    quadrant = current.get("quadrant")

    duration = engine.months_in_phase_and_median(phases)
    current_phase = phases.iloc[-1] if not phases.empty else None
    phase_detail = None
    if quadrant_history.dropna(subset=["quadrant"]).shape[0] > 0:
        idx = quadrant_history.dropna(subset=["quadrant"]).index
        last_ts, prior_ts = idx[-1], (idx[-2] if len(idx) > 1 else None)
        prior_row = monthly.loc[prior_ts] if prior_ts is not None and prior_ts in monthly.index else None
        phase_detail = engine.classify_phase(quadrant_history.loc[last_ts], monthly.loc[last_ts], prior_row)
        phase_detail.update(duration)

    gauge = engine.compute_long_term_gauge(monthly)

    transitions_3m = history_mod.transition_probabilities(
        quadrant_history["quadrant"], phases, quadrant, current_phase, 3
    ) if quadrant else None
    transitions_6m = history_mod.transition_probabilities(
        quadrant_history["quadrant"], phases, quadrant, current_phase, 6
    ) if quadrant else None

    monthly_prices = prices_full.resample("ME").last()
    assets_by_quadrant = history_mod.compute_asset_returns_by_quadrant(
        quadrant_history["quadrant"], monthly_prices, ASSET_UNIVERSE
    )
    # Reconciles "n=13 months classified" (transition sample, conditioned on
    # quadrant+phase) against "not enough history" on the asset-return table
    # (conditioned on quadrant alone, needs a REALIZED forward return per
    # episode) -- these are different, both-correct denominators for
    # different statistical questions, not a contradiction, but they need
    # to be reconcilable on the page rather than silently disagreeing.
    quadrant_month_counts = quadrant_history["quadrant"].value_counts().to_dict()
    assets_by_quadrant_meta = {
        q: {
            "monthsClassified": int(quadrant_month_counts.get(q, 0)),
            "assetEpisodesRequired": history_mod.MIN_ASSET_EPISODES,
            "suppressed": len(assets_by_quadrant.get(q, [])) == 0,
        }
        for q in history_mod.QUADRANTS
    }

    confirmation = confirmation_mod.compute_cross_asset_confirmation(quadrant, macro, prices_full) if quadrant else {
        "rows": [], "confirms": 0, "diverges": 0, "neutral": 0,
        "confirmationScore": None, "confirmationLabel": None,
        "tapeNotExpressingView": False, "modelUnderReview": False,
    }

    raw_row = monthly.iloc[-1].to_dict() if not monthly.empty else {}
    base_probability = None
    ranked_alternates: List = []
    if transitions_3m and not transitions_3m.get("suppressed") and transitions_3m.get("probabilities"):
        base_probability = transitions_3m["probabilities"].get(quadrant)
        ranked_alternates = sorted(
            ((q, p) for q, p in transitions_3m["probabilities"].items() if q != quadrant),
            key=lambda kv: -kv[1],
        )

    alternates = []
    rank_labels = ["MOST LIKELY NEXT", "SECOND MOST LIKELY"]
    for (alt_quadrant, prob), rank_label in zip(ranked_alternates[:2], rank_labels):
        alternates.append(
            thesis_text.build_alternate(
                rank_label, quadrant, alt_quadrant, prob, transitions_3m["sampleSize"],
                assets_by_quadrant.get(alt_quadrant), raw_row,
            )
        )

    # what_would_change_it reuses alternates[0]'s trigger exactly (not a
    # second, independently-derived pick) -- "what would change the current
    # regime" and "the trigger for the most likely next regime" describe the
    # same real-world threshold crossing by construction, so deriving them
    # from one call is what guarantees they can't silently disagree.
    what_would_change_it = alternates[0]["trigger"] if alternates else None
    primary_target_quadrant = alternates[0]["quadrant"] if alternates else None

    base_case = thesis_text.build_base_case(
        quadrant, phase_detail, confirmation, raw_row,
        growth_level=current.get("growth", {}).get("level"),
        growth_momentum=current.get("growth", {}).get("momentum"),
        inflation_level=current.get("inflation", {}).get("level"),
        inflation_momentum=current.get("inflation", {}).get("momentum"),
        probability=base_probability,
        assets=assets_by_quadrant.get(quadrant) if quadrant else None,
        what_would_change_it=what_would_change_it,
    )

    existing_triggers = [a["trigger"] for a in alternates] + ([what_would_change_it] if what_would_change_it else [])
    invalidation = thesis_text.build_invalidation(quadrant, primary_target_quadrant, raw_row, existing_triggers)
    invalidation["reviewBy"] = (as_of.date() + timedelta(days=90)).isoformat()

    conviction = engine.compute_conviction(
        current.get("quadrantStrength"), confirmation.get("confirms", 0), confirmation.get("diverges", 0)
    )

    model_read = {
        "quadrant": quadrant,
        "quadrantStrength": current.get("quadrantStrength"),
        "transitioning": current.get("transitioning"),
        "weeksSinceCrossing": current.get("weeksSinceCrossing"),
        "conviction": conviction,
        "modelUnderReview": confirmation.get("modelUnderReview", False),
    }

    return {
        "asOf": as_of.date().isoformat(),
        "dataDates": {
            "modelMonth": monthly.index.max().strftime("%Y-%m") if not monthly.empty else None,
            "marketPricesThrough": prices_full.dropna(how="all").index.max().date().isoformat() if not prices_full.dropna(how="all").empty else None,
        },
        "quadrant": {
            "name": quadrant,
            "quadrantStrength": current.get("quadrantStrength"),
            "transitioning": current.get("transitioning"),
            "weeksSinceCrossing": current.get("weeksSinceCrossing"),
            "conviction": conviction,
            "modelUnderReview": confirmation.get("modelUnderReview", False),
        },
        # Spec section 8: model and house view are never merged -- two
        # separate top-level keys, shown side by side, not resolved.
        "modelRead": model_read,
        "houseView": house_view,  # None until an admin publishes one (section 8)
        # confirmationScore is a ratio over NON-neutral markets only (None
        # when neutral dominates -- see confirmation.py). confirms/diverges/
        # neutral are the three raw counts; confirmationLabel is the display
        # string ("1 diverges, 4 neutral" or "Tape is not expressing a view").
        "confirmationScore": confirmation.get("confirmationScore"),
        "confirmationLabel": confirmation.get("confirmationLabel"),
        "confirms": confirmation.get("confirms", 0),
        "diverges": confirmation.get("diverges", 0),
        "neutral": confirmation.get("neutral", 0),
        "tapeNotExpressingView": confirmation.get("tapeNotExpressingView", False),
        "growth": {
            "level": current.get("growth", {}).get("level"),
            "momentum": current.get("growth", {}).get("momentum"),
            "trail": _growth_inflation_trail(quadrant_history),
        },
        "inflation": {
            "level": current.get("inflation", {}).get("level"),
            "momentum": current.get("inflation", {}).get("momentum"),
        },
        "cycleMap": {
            "longTerm": {
                "position": gauge.get("position"),
                "label": gauge.get("label"),
                "trend": gauge.get("trend"),
                "mostExtended": gauge.get("mostExtended"),
            },
            "shortTerm": {
                "phase": phase_detail.get("phase") if phase_detail else None,
                "position": (phase_detail.get("phaseConfidence") if phase_detail else None),
                "monthsInPhase": duration.get("monthsInPhase"),
                "medianPhaseMonths": duration.get("medianPhaseMonths"),
                "conditionsMet": phase_detail.get("conditionsMet") if phase_detail else None,
                "conditionsTotal": phase_detail.get("conditionsTotal") if phase_detail else None,
                "conditions": phase_detail.get("conditions") if phase_detail else [],
            },
        },
        "scenarios": {
            "base": {**base_case, "n": transitions_3m.get("sampleSize") if transitions_3m else None},
            "alternates": alternates,
            "invalidation": invalidation,
        },
        "transitions": {"3m": transitions_3m, "6m": transitions_6m},
        "assetsByQuadrant": assets_by_quadrant.get(quadrant, []) if quadrant else [],
        "assetsByQuadrantAll": assets_by_quadrant,
        "assetsByQuadrantMeta": assets_by_quadrant_meta,
        "crossAssetCheck": confirmation.get("rows", []),
        "evidenceTable": _evidence_table(monthly),
        "methodology": {
            "revisionCaveat": (
                "Uses current-vintage FRED data, so historical quadrants benefit from "
                "revisions unavailable in real time. Not corrected for real-time vintage."
            ),
            "calibrationDisclosure": engine.CALIBRATION_DISCLOSURE,
            "calibrationEpisodes": engine.CALIBRATION_EPISODES,
        },
    }
