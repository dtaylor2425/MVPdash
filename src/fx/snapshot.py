"""
src/fx/snapshot.py

Orchestrator: ingest -> score -> assemble the published snapshot payload
(spec section 6). Computed on a schedule by jobs/fx_snapshot_job.py; the API
routes only ever read the result.

`build_fx_snapshot()` returns the exact response shape of GET /api/fx/snapshot,
plus a per-currency `history` block so GET /api/fx/currency/[code] can be served
from the same document.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.fx import breadth as breadthmod
from src.fx import components as comp
from src.fx import frankfurter as fxmod
from src.fx.components import COMPONENTS, InputBundle
from src.fx.fred_client import build_fx_fred_frame
from src.fx.reconciliation import build_reconciliation
from src.fx.reer import load_reer
from src.fx.scoring import (
    COMPONENT_WEIGHTS,
    NEUTRAL,
    composite_scores,
    cross_sectional_z,
    currency_flags,
    derive_pairs,
    rank_currencies,
    z_from_components,
)
from src.fx.series_map import (
    CB_INFLATION_TARGET,
    CURRENCY_NAMES,
    DISPLAY_ONLY,
    FRED_SERIES,
    SCORED,
)

SOURCE_TEXT = "ECB reference fixing via Frankfurter; rates and macro via FRED"
HISTORY_MONTHS = 27          # >= 2y (spec: currency detail history)
MAX_INCOMPLETE = 3           # spec 5 -- never publish if > 3 of 10 incomplete


# ---------------------------------------------------------------------------
# scoring at a single date
# ---------------------------------------------------------------------------
def _score_at(bundle: InputBundle, ts: pd.Timestamp) -> dict:
    comp_raw = comp.compute_all(bundle, ts)
    comp_z = z_from_components(comp_raw)
    composites = composite_scores(comp_z, SCORED)
    trend_1m_raw = comp.trend_1m_z_inputs(bundle, ts)
    trend_1m_z = cross_sectional_z(trend_1m_raw)
    return {
        "comp_raw": comp_raw,
        "comp_z": comp_z,
        "composites": composites,
        "trend_1m_z": trend_1m_z,
    }


def _month_ends(as_of: pd.Timestamp, months: int) -> List[pd.Timestamp]:
    grid = list(pd.date_range(end=as_of.normalize(), periods=months, freq="ME"))
    if not grid or grid[-1].date() != as_of.date():
        grid.append(as_of.normalize())
    return grid


# ---------------------------------------------------------------------------
# per-currency history
# ---------------------------------------------------------------------------
def _series_to_points(series: pd.Series, as_of: pd.Timestamp, years: float = 2.0) -> List[dict]:
    if series is None or series.empty:
        return []
    s = series.dropna()
    s = s[s.index <= as_of]
    s = s[s.index >= (as_of - pd.Timedelta(days=int(years * 365) + 40))]
    if s.empty:
        return []
    monthly = s.resample("ME").last().dropna()
    return [
        {"date": idx.date().isoformat(), "value": round(float(v), 4)}
        for idx, v in monthly.items()
    ]


def _yield_history(bundle: InputBundle, ccy: str, as_of: pd.Timestamp) -> List[dict]:
    m = FRED_SERIES.get(ccy, {})
    if m.get("y2"):
        s = bundle.col(f"{ccy}__y2")
        if not s.empty:
            return _series_to_points(s, as_of)
    if m.get("y10"):
        return _series_to_points(bundle.col(f"{ccy}__y10"), as_of)
    return []


# ---------------------------------------------------------------------------
# currency payload
# ---------------------------------------------------------------------------
def _policy_rate_label(ccy: str, rate: Optional[float]) -> Optional[str]:
    if rate is None:
        return None
    if ccy == "USD":
        lo = np.floor(rate / 0.25) * 0.25
        return f"{lo:.2f}-{lo + 0.25:.2f}%"
    return f"{rate:.2f}%"


def _read_text(components_payload: Dict[str, dict], flags: dict) -> str:
    ranked = sorted(
        ((c, v["z"]) for c, v in components_payload.items() if v.get("available")),
        key=lambda x: x[1],
        reverse=True,
    )
    if not ranked:
        return "Insufficient component data to form a read."
    labels = {
        "carry": "carry", "policyMomentum": "policy momentum",
        "macroVsMandate": "inflation vs mandate", "termsOfTrade": "terms of trade",
        "trend": "trend", "valuation": "valuation",
    }
    top = [labels[c] for c, z in ranked[:2] if z > 0.15]
    drag = [labels[c] for c, z in ranked[::-1] if z < -0.15][:1]
    bits = []
    if len(top) == 2:
        bits.append(f"{top[0].capitalize()} and {top[1]} both supportive")
    elif len(top) == 1:
        bits.append(f"{top[0].capitalize()} supportive")
    else:
        bits.append("No component is clearly supportive")
    if drag:
        bits.append(f"{drag[0]} is the drag")
    if flags.get("carryTrendConflict"):
        bits.append("carry/trend conflict flagged (pre-unwind setup)")
    if flags.get("interventionRisk"):
        bits.append("intervention risk elevated")
    return "; ".join(bits) + "."


def _currency_entry(
    ccy: str,
    latest: dict,
    prev_week: Optional[dict],
    score_history: Dict[str, List[dict]],
    bundle: InputBundle,
    as_of: pd.Timestamp,
) -> dict:
    comp_raw = latest["comp_raw"]
    comp_z = latest["comp_z"]
    composite = latest["composites"][ccy]

    components_payload: Dict[str, dict] = {}
    for c in COMPONENTS:
        cv = comp_raw.get(c, {}).get(ccy, {})
        z = comp_z.get(c, {}).get(ccy)
        available = bool(cv.get("available")) and z is not None
        components_payload[c] = {
            "z": round(float(z), 3) if z is not None else None,
            "display": round(float(NEUTRAL + 12.5 * z), 1) if z is not None else None,
            "proxy": bool(cv.get("proxy")),
            "available": available,
        }

    flags = currency_flags(ccy, comp_z, composite, latest["trend_1m_z"], comp_raw)

    # top-level raw macro fields
    ts = as_of
    policy_logical = f"{ccy}__policy_rate"
    policy = comp._asof(bundle.col(policy_logical), ts, comp._max_age(policy_logical))
    yld_series, _, yld_logical = comp._yield_series(bundle, ccy)
    y2 = comp._asof(yld_series, ts, comp._max_age(yld_logical))
    cpi = comp._cpi_yoy(bundle, ccy, ts)

    change_1w = None
    if prev_week is not None and ccy in prev_week["composites"]:
        change_1w = round(
            composite["display"] - prev_week["composites"][ccy]["display"], 1
        )

    return {
        "code": ccy,
        "name": CURRENCY_NAMES.get(ccy, ccy),
        "scored": True,
        "score": round(composite["display"]),
        "rank": None,  # filled by caller
        "change1w": change_1w,
        "components": components_payload,
        "flags": flags,
        "policyRate": None if policy is None else round(policy, 3),
        "policyRateLabel": _policy_rate_label(ccy, policy),
        "cpiYoY": None if cpi is None else round(cpi, 2),
        "cbTarget": CB_INFLATION_TARGET.get(ccy),
        "yield2y": None if y2 is None else round(y2, 3),
        "read": _read_text(components_payload, flags),
        "compositeZ": round(float(composite["z"]), 4),
        "droppedComponents": composite.get("droppedComponents", []),
        "history": {
            "score": score_history.get(ccy, []),
            "spotUsd": _series_to_points(bundle.fx.spot_usd(ccy), as_of),
            "yield2y": _yield_history(bundle, ccy, as_of),
        },
    }


def _cny_entry(bundle: InputBundle, as_of: pd.Timestamp) -> dict:
    ts = as_of
    policy = comp._asof(bundle.col("CNY__policy_rate"), ts, comp._max_age("CNY__policy_rate"))
    cpi = comp._cpi_yoy(bundle, "CNY", ts)
    return {
        "code": "CNY",
        "name": CURRENCY_NAMES["CNY"],
        "scored": False,
        "label": "managed — not scored",
        "score": None,
        "rank": None,
        "policyRate": None if policy is None else round(policy, 3),
        "policyRateLabel": None if policy is None else f"{policy:.2f}%",
        "cpiYoY": None if cpi is None else round(cpi, 2),
        "cbTarget": None,
        "yield2y": None,
        "read": "Managed float. Data shown for context; a free-float model would "
                "produce confident nonsense here.",
        "history": {
            "spotUsd": _series_to_points(bundle.fx.spot_usd("CNY"), as_of),
        },
    }


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def build_bundle(use_cache: bool = True) -> InputBundle:
    fred = build_fx_fred_frame(use_cache=use_cache)
    fx = fxmod.default_history()
    reer = load_reer(use_cache=use_cache)
    return InputBundle(fred=fred, fx=fx, reer=reer)


def build_fx_snapshot(
    bundle: Optional[InputBundle] = None,
    as_of: Optional[date] = None,
    history_months: int = HISTORY_MONTHS,
    use_cache: bool = True,
) -> dict:
    bundle = bundle or build_bundle(use_cache=use_cache)

    obs_date = bundle.fx.observation_date
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
    elif obs_date is not None:
        as_of_ts = pd.Timestamp(obs_date)
    else:
        as_of_ts = pd.Timestamp(datetime.now(timezone.utc).date())

    # --- history grid: score every currency at each month-end ---------------
    grid = _month_ends(as_of_ts, history_months)
    score_history: Dict[str, List[dict]] = {c: [] for c in SCORED}
    for ts in grid:
        snap = _score_at(bundle, ts)
        for ccy in SCORED:
            disp = snap["composites"][ccy]["display"]
            score_history[ccy].append(
                {"date": ts.date().isoformat(), "value": round(disp)}
            )

    latest = _score_at(bundle, as_of_ts)
    prev_week = _score_at(bundle, as_of_ts - pd.Timedelta(days=7))

    # --- currency entries -------------------------------------------------- --
    currencies: List[dict] = []
    for ccy in SCORED:
        currencies.append(
            _currency_entry(ccy, latest, prev_week, score_history, bundle, as_of_ts)
        )
    ranks = rank_currencies(latest["composites"])
    for entry in currencies:
        entry["rank"] = ranks.get(entry["code"])
    currencies.sort(key=lambda e: e["rank"] or 99)

    incomplete = [c["code"] for c in currencies if c["flags"]["dataIncomplete"]]

    cny = _cny_entry(bundle, as_of_ts)
    all_currencies = currencies + [cny]

    # --- pairs ---------------------------------------------------------------
    flags_by_ccy = {c["code"]: c["flags"] for c in currencies}
    pairs = derive_pairs(latest["composites"], flags_by_ccy, SCORED)

    # --- reconciliation ----------------------------------------------------- --
    usd_score = next((c["score"] for c in currencies if c["code"] == "USD"), None)
    reconciliation = build_reconciliation(usd_score, bundle.fred)

    # --- breadth & attribution (spec 4A) -- independent model, own failure domain
    try:
        breadth = breadthmod.compute_breadth_snapshot(bundle.fx)
    except Exception as exc:  # never let this block the composite snapshot
        breadth = {
            "universe": breadthmod.DEFAULT_UNIVERSE, "horizon": breadthmod.DEFAULT_HORIZON,
            "trendWindow": breadthmod.DEFAULT_TREND_WINDOW, "threshold": breadthmod.DEFAULT_THRESHOLD,
            "strengths": {}, "byBase": {}, "history": {}, "error": str(exc),
        }

    # --- component availability roll-up ------------------------------------- --
    component_availability = {
        c: sorted(k for k, z in latest["comp_z"].get(c, {}).items())
        for c in COMPONENTS
    }
    dropped_components = [c for c in COMPONENTS if not latest["comp_z"].get(c)]

    return {
        "asOf": datetime.now(timezone.utc).date().isoformat(),
        "observationDate": obs_date or as_of_ts.date().isoformat(),
        "source": SOURCE_TEXT,
        "stale": False,
        "currencies": all_currencies,
        "pairs": pairs,
        "reconciliation": reconciliation,
        "breadth": breadth,
        "meta": {
            "componentWeights": dict(COMPONENT_WEIGHTS),
            "universe": list(SCORED),
            "displayOnly": list(DISPLAY_ONLY),
            "historyMonths": history_months,
            "incompleteCurrencies": incomplete,
            "droppedComponents": dropped_components,
            "componentAvailability": component_availability,
            "methodology": {
                "zScore": "cross-sectional across the 10 scored currencies per component "
                          "per date; winsorised at ±2.5σ; component dropped and reweighted "
                          "below 6 valid currencies",
                "displayMapping": "score = 50 + 12.5 · composite_z, clamped 0-100 (fixed sigma "
                                  "mapping, comparable week to week)",
                "pairs": "pairScore = baseComposite_z − quoteComposite_z; one triangle stored, "
                         "antisymmetric, mirror for display",
                "proxies": "policyMomentum uses Δ2y government yield as an OIS-path proxy; "
                           "non-USD carry/momentum fall back to the 10y yield where no 2y "
                           "series exists; trend uses static trade-weights",
                "fixings": "Frankfurter returns daily central-bank reference fixings, not live "
                           "quotes; observationDate carries the fixing date",
            },
        },
    }


def snapshot_is_publishable(payload: dict) -> tuple[bool, str]:
    incomplete = payload.get("meta", {}).get("incompleteCurrencies", [])
    if len(incomplete) > MAX_INCOMPLETE:
        return False, (
            f"{len(incomplete)} of {len(SCORED)} currencies incomplete "
            f"({', '.join(incomplete)}); spec caps at {MAX_INCOMPLETE}. Aborting."
        )
    scored = [c for c in payload.get("currencies", []) if c.get("scored")]
    if len(scored) < len(SCORED):
        return False, f"only {len(scored)} scored currencies in payload; expected {len(SCORED)}."
    return True, "ok"
