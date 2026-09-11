"""
src/fx/components.py

The five per-currency component calculators (spec section 4.1; a sixth,
macroVsMandate, was dropped -- see the note above termsOfTrade below). Each
returns a dict {currency -> ComponentValue}:

    {"raw": float | None,      # the pre-z quantity, in natural units
     "proxy": bool,            # computed from a proxy series?
     "available": bool,        # is `raw` usable?
     "detail": {...}}          # sub-inputs, for the detail page / debugging

Cross-sectional z-scoring, winsorisation, dropping and reweighting all happen
downstream in src/fx/scoring.py. Calculators never emit NaN -- an unusable
input yields available:false.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.fx import frankfurter as fxmod
from src.fx.series_map import (
    FRED_SERIES,
    SCORED,
    STALE_DAYS,
    TERMS_OF_TRADE_BASKET,
    TREND_PARTNER_WEIGHTS,
    logical_max_age,
    policy_rate_provenance,
)

_MAX_AGE = logical_max_age()


def _max_age(logical: str) -> int:
    return _MAX_AGE.get(logical, 120)

COMPONENTS = ["carry", "policyMomentum", "termsOfTrade", "trend", "valuation"]


# ---------------------------------------------------------------------------
# input bundle
# ---------------------------------------------------------------------------
@dataclass
class InputBundle:
    fred: pd.DataFrame
    fx: "fxmod.FxData"
    reer: Dict[str, pd.Series] = field(default_factory=dict)
    # BIS-announced central bank policy rates (src/fx/policy_rates.py), keyed
    # by ISO currency. Preferred over the FRED money-market proxy in
    # FRED_SERIES[ccy]["policy_rate"] wherever available -- see carry() and
    # snapshot._currency_entry.
    policy_rates: Dict[str, pd.Series] = field(default_factory=dict)

    def col(self, logical: str) -> pd.Series:
        if logical in self.fred.columns:
            return self.fred[logical].dropna()
        return pd.Series(dtype="float64")


def _val(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _asof(series: pd.Series, ts: pd.Timestamp, max_age_days: Optional[int] = None) -> Optional[float]:
    """
    Point-in-time lookup: the most recent real (non-forward-filled) observation
    at or before `ts`. If `max_age_days` is given and that observation is older
    than that, treat the series as unavailable rather than silently reporting a
    frozen value -- this is what keeps a dead FRED series from scoring forever
    (spec 3.3).
    """
    if series is None or series.empty:
        return None
    s = series.dropna()
    s = s[s.index <= ts]
    if s.empty:
        return None
    if max_age_days is not None and (ts - s.index[-1]).days > max_age_days:
        return None
    return _val(s.iloc[-1])


def _asof_lag(series: pd.Series, ts: pd.Timestamp, days: int,
              max_age_days: Optional[int] = None) -> Optional[float]:
    return _asof(series, ts - pd.Timedelta(days=days), max_age_days)


def _yoy(series: pd.Series, ts: pd.Timestamp, kind: str,
         max_age_days: Optional[int] = None) -> Optional[float]:
    if kind == "yoy":
        return _asof(series, ts, max_age_days)
    now = _asof(series, ts, max_age_days)
    year_ago = _asof_lag(series, ts, 365, max_age_days)
    if now is None or year_ago in (None, 0):
        return None
    return (now / year_ago - 1.0) * 100.0


def _pct_change(series: pd.Series, ts: pd.Timestamp, days: int,
                 max_age_days: Optional[int] = None) -> Optional[float]:
    now = _asof(series, ts, max_age_days)
    then = _asof_lag(series, ts, days, max_age_days)
    if now is None or then in (None, 0):
        return None
    return (now / then - 1.0) * 100.0


def _blank(proxy: bool = False) -> Dict[str, object]:
    return {"raw": None, "proxy": proxy, "available": False, "detail": {}}


# ---------------------------------------------------------------------------
# helpers shared across components
# ---------------------------------------------------------------------------
def _yield_series(b: InputBundle, ccy: str):
    """Return (series, is_proxy, logical_name). Prefers 2y; falls back to 10y (proxy=True)."""
    m = FRED_SERIES.get(ccy, {})
    if m.get("y2"):
        logical = f"{ccy}__y2"
        s = b.col(logical)
        if not s.empty:
            return s, False, logical
    if m.get("y10"):
        logical = f"{ccy}__y10"
        return b.col(logical), True, logical
    return pd.Series(dtype="float64"), True, f"{ccy}__y10"


def _cpi_yoy(b: InputBundle, ccy: str, ts: pd.Timestamp) -> Optional[float]:
    m = FRED_SERIES.get(ccy, {})
    if not m.get("cpi"):
        return None
    logical = f"{ccy}__cpi"
    return _yoy(b.col(logical), ts, m.get("cpi_kind", "index"), _max_age(logical))


def _policy_rate(b: InputBundle, ccy: str, ts: pd.Timestamp):
    """
    Return (rate, is_proxy, instrument_label). Prefers the BIS-announced
    central bank policy rate (src/fx/policy_rates.py); falls back to the
    FRED money-market rate in FRED_SERIES when BIS has nothing fresh for
    this currency (fix-list item 5 -- the FRED rate is a real proxy, not the
    announced rate, for every currency except USD/EUR).
    """
    bis_series = b.policy_rates.get(ccy)
    if bis_series is not None and not bis_series.empty:
        val = _asof(bis_series, ts, STALE_DAYS["daily"])
        if val is not None:
            return val, False, "central bank policy rate"
    policy_logical = f"{ccy}__policy_rate"
    fred_series_id = FRED_SERIES.get(ccy, {}).get("policy_rate")
    val = _asof(b.col(policy_logical), ts, _max_age(policy_logical))
    provenance = policy_rate_provenance(fred_series_id)
    return val, provenance["isProxy"], provenance["instrument"]


# ---------------------------------------------------------------------------
# 1. carry -- policy rate; 2y yield; real policy rate (policy - YoY CPI)
# ---------------------------------------------------------------------------
def carry(b: InputBundle, ts: pd.Timestamp) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ccy in SCORED:
        policy, policy_is_proxy, policy_instrument = _policy_rate(b, ccy, ts)
        yld_series, yld_is_proxy, yld_logical = _yield_series(b, ccy)
        yld = _asof(yld_series, ts, _max_age(yld_logical))
        cpi = _cpi_yoy(b, ccy, ts)
        real_policy = (policy - cpi) if (policy is not None and cpi is not None) else None

        parts = [p for p in (policy, yld, real_policy) if p is not None]
        is_proxy = bool(policy_is_proxy or yld_is_proxy)
        if not parts:
            out[ccy] = _blank(proxy=is_proxy)
            continue
        out[ccy] = {
            "raw": float(np.mean(parts)),
            "proxy": is_proxy,
            "available": True,
            "detail": {
                "policyRate": policy,
                "policyRateIsProxy": policy_is_proxy,
                "policyRateInstrument": policy_instrument,
                "yield": yld,
                "yieldIsProxy": yld_is_proxy,
                "realPolicyRate": real_policy,
                "cpiYoY": cpi,
            },
        }
    return out


# ---------------------------------------------------------------------------
# 2. policyMomentum -- change in 2y yield over 1m and 3m (proxy: govt yield)
# ---------------------------------------------------------------------------
def policy_momentum(b: InputBundle, ts: pd.Timestamp) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ccy in SCORED:
        s, is_proxy, logical = _yield_series(b, ccy)
        age = _max_age(logical)
        now = _asof(s, ts, age)
        d1 = None if now is None else _asof_lag(s, ts, 30, age)
        d3 = None if now is None else _asof_lag(s, ts, 90, age)
        chg1 = (now - d1) if (now is not None and d1 is not None) else None
        chg3 = (now - d3) if (now is not None and d3 is not None) else None
        parts = [p for p in (chg1, chg3) if p is not None]
        if not parts:
            out[ccy] = _blank(proxy=True)
            continue
        out[ccy] = {
            "raw": float(np.mean(parts)),
            # 2y govt yield is a proxy for the OIS-implied path in all cases.
            "proxy": True,
            "available": True,
            "detail": {"change1m": chg1, "change3m": chg3, "usedLongYield": is_proxy},
        }
    return out


# ---------------------------------------------------------------------------
# termsOfTrade -- export-weighted commodity basket return, 3m
#
# NOTE: a sixth component, macroVsMandate (CPI YoY vs central-bank target),
# was dropped from the model entirely -- not just at runtime. The OECD MEI CPI
# feed on FRED is dead for every scored currency except USD and EUR (verified;
# 528-618 days stale), so the component could only ever score 2 of 10
# currencies -- permanently below the 6-currency minimum. Free G10 CPI
# coverage sufficient to score isn't available, so shipping this as a
# sometimes-scored sixth component would mean shipping a component that's
# always dropped for 8 of 10 currencies. CPI YoY / CB target / mandate gap are
# still surfaced as unscored display-only context for USD and EUR in
# src/fx/snapshot.py (`mandate` field).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def terms_of_trade(b: InputBundle, ts: pd.Timestamp) -> Dict[str, dict]:
    returns_3m: Dict[str, Optional[float]] = {}
    for name in {c for basket in TERMS_OF_TRADE_BASKET.values() for c in basket}:
        logical = f"cmdty__{name}"
        returns_3m[name] = _pct_change(b.col(logical), ts, 90, _max_age(logical))

    out: Dict[str, dict] = {}
    for ccy in SCORED:
        basket = TERMS_OF_TRADE_BASKET.get(ccy, {})
        legs = {k: (w, returns_3m.get(k)) for k, w in basket.items()}
        used = {k: (w, r) for k, (w, r) in legs.items() if r is not None}
        if not used:
            out[ccy] = _blank()
            continue
        raw = sum(w * r for (w, r) in used.values())
        out[ccy] = {
            "raw": float(raw),
            "proxy": False,
            "available": True,
            "detail": {"legs": {k: {"weight": w, "return3m": r} for k, (w, r) in used.items()}},
        }
    return out


# ---------------------------------------------------------------------------
# 5. trend -- trade-weighted index momentum, 1m/3m/6m blend
# ---------------------------------------------------------------------------
def _trend_metrics(b: InputBundle, ccy: str, ts: pd.Timestamp):
    idx = b.fx.effective_index(ccy, TREND_PARTNER_WEIGHTS.get(ccy, {}))
    if idx.empty:
        return None, None
    m1 = _pct_change(idx, ts, 30)
    m3 = _pct_change(idx, ts, 90)
    m6 = _pct_change(idx, ts, 180)
    parts = [p for p in (m1, m3, m6) if p is not None]
    blend = float(np.mean(parts)) if parts else None
    return blend, {"m1": m1, "m3": m3, "m6": m6}


def trend(b: InputBundle, ts: pd.Timestamp) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ccy in SCORED:
        blend, detail = _trend_metrics(b, ccy, ts)
        if blend is None:
            out[ccy] = _blank(proxy=True)
            continue
        out[ccy] = {
            "raw": blend,
            # static partner weights, not a true published NEER
            "proxy": True,
            "available": True,
            "detail": detail,
        }
    return out


def trend_1m_z_inputs(b: InputBundle, ts: pd.Timestamp) -> Dict[str, Optional[float]]:
    """Raw 1m effective-index return per currency -- feeds interventionRisk."""
    vals: Dict[str, Optional[float]] = {}
    for ccy in SCORED:
        idx = b.fx.effective_index(ccy, TREND_PARTNER_WEIGHTS.get(ccy, {}))
        vals[ccy] = _pct_change(idx, ts, 30) if not idx.empty else None
    return vals


# ---------------------------------------------------------------------------
# 6. valuation -- REER deviation from 10y average, SIGN INVERTED
# ---------------------------------------------------------------------------
def valuation(b: InputBundle, ts: pd.Timestamp) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ccy in SCORED:
        series = b.reer.get(ccy)
        if series is None or series.empty:
            out[ccy] = _blank()
            continue
        s = series.dropna()
        s = s[s.index <= ts]
        if s.empty:
            out[ccy] = _blank()
            continue
        current = _val(s.iloc[-1])
        window = s[s.index >= (ts - pd.Timedelta(days=3653))]
        avg = _val(window.mean()) if not window.empty else None
        if current is None or avg in (None, 0):
            out[ccy] = _blank()
            continue
        deviation = (current / avg - 1.0) * 100.0
        out[ccy] = {
            # inverted: an expensive currency (deviation > 0) scores negative.
            "raw": float(-deviation),
            "proxy": False,
            "available": True,
            "detail": {"reer": current, "reer10yAvg": avg, "deviationPct": deviation},
        }
    return out


# ---------------------------------------------------------------------------
# run all
# ---------------------------------------------------------------------------
_CALCULATORS = {
    "carry": carry,
    "policyMomentum": policy_momentum,
    "termsOfTrade": terms_of_trade,
    "trend": trend,
    "valuation": valuation,
}


def compute_all(b: InputBundle, ts: pd.Timestamp) -> Dict[str, Dict[str, dict]]:
    """{component -> {currency -> ComponentValue}}."""
    return {name: fn(b, ts) for name, fn in _CALCULATORS.items()}
