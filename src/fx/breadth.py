"""
src/fx/breadth.py  (spec section 4A -- "Breadth and attribution module")

A second, independent model from the composite score in scoring.py. The
composite is a forecast (what should happen, from carry/policy/valuation).
This module is realised attribution: what did happen over the last N bars,
and which currency caused it. Do not merge the two.

Ported from a Pine Script implementation (CFR FX Breadth); the maths is exact,
not approximate. No FRED dependency -- Frankfurter cross-rate history only.

Core identity (section 4A.1), using log prices L(c) = -log(usd_per(c)) so that
log(a/b) = L(a) - L(b) exactly:

    strength(c) = (1/N) * sum over the full universe (incl. c itself) of log(c/x)
                = L(c) - mean(L)                     <- per date, row-demeaned

Demeaning per date makes the two hard invariants automatic:
  * strength(a) - strength(b) == log(a/b) exactly (same subtraction either way)
  * sum(strength) == 0 exactly (mean subtracted off)
and, because the additive term introduced by fetching from any other base
cancels in the demeaning, strengths are base-invariant by construction --
verified in tests/test_fx_breadth.py using an independently-derived cross-rate
path (log_prices_via_base), not just algebraically assumed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.fx.frankfurter import FxData
from src.fx.series_map import SCORED

MAJORS = ["USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD"]
G10 = list(SCORED)
UNIVERSES: Dict[str, List[str]] = {"majors": MAJORS, "g10": G10}

DEFAULT_UNIVERSE = "majors"
DEFAULT_HORIZON = 21          # bars (~1 trading month)
DEFAULT_TREND_WINDOW = 63     # bars (~1 trading quarter)
DEFAULT_THRESHOLD = 0.0       # log-return noise filter for breadth counting
HISTORY_MONTHS = 25


# ---------------------------------------------------------------------------
# log-price panel and strength (4A.1)
# ---------------------------------------------------------------------------
def log_prices_usd(fx: FxData, universe: List[str]) -> pd.DataFrame:
    """L(c) = -log(usd_per(c)) = log(price of c in USD), aligned across universe."""
    cols = {}
    for c in universe:
        s = fx.usd_per(c)
        if not s.empty:
            cols[c] = np.log(s.astype(float))
    if not cols:
        return pd.DataFrame()
    df = -pd.DataFrame(cols)
    return df.dropna(how="any").sort_index()


def log_prices_via_base(fx: FxData, universe: List[str], base: str) -> pd.DataFrame:
    """
    Same log-price panel, derived through an explicit `base` cross-rate instead
    of the USD table directly. Used only to prove base-invariance (4A.7 #5) --
    the production path uses log_prices_usd because it needs one fetch, not N.
    """
    cols = {}
    for c in universe:
        cr = fx.cross(base, c)  # units of c per 1 base
        if not cr.empty:
            cols[c] = np.log(cr.astype(float))
    if not cols:
        return pd.DataFrame()
    df = -pd.DataFrame(cols)
    return df.dropna(how="any").sort_index()


def strengths_frame(L: pd.DataFrame) -> pd.DataFrame:
    """Row-wise demean -- strength(c, t) for every date and currency."""
    if L.empty:
        return L
    return L.sub(L.mean(axis=1), axis=0)


def latest_strengths(S: pd.DataFrame) -> Dict[str, float]:
    if S.empty:
        return {}
    row = S.iloc[-1]
    return {c: round(float(v), 6) for c, v in row.items() if math.isfinite(v)}


# ---------------------------------------------------------------------------
# verdict taxonomy (4A.4)
# ---------------------------------------------------------------------------
def _verdict(breadth: int, share: float, m: int) -> str:
    extreme = abs(breadth) >= max(m - 1, 0)
    if extreme and share >= 0.5:
        return "BASE_DRIVEN"
    if extreme:
        return "BASE_PLUS_STRESS"
    if share < 0.5:
        return "OTHERS_MOVING"
    return "QUIET_MIXED"


# ---------------------------------------------------------------------------
# per-base metrics (4A.3, 4A.5)
# ---------------------------------------------------------------------------
def base_metrics(
    L: pd.DataFrame,
    S: pd.DataFrame,
    base: str,
    universe: List[str],
    horizon: int = DEFAULT_HORIZON,
    trend_window: int = DEFAULT_TREND_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    others = [c for c in universe if c != base and c in L.columns]
    if base not in L.columns or len(others) < 2 or len(L) <= horizon:
        return {"available": False}

    # x_i(t) = log(base/other_i) = L(base,t) - L(other_i,t) -- identical whether
    # read off L directly or off strengths S (the per-date constant cancels).
    x = pd.DataFrame({o: L[base] - L[o] for o in others})

    r = x.iloc[-1] - x.iloc[-1 - horizon]           # horizon log return, per cross
    up_n = int((r > threshold).sum())
    down_n = int((r < -threshold).sum())
    breadth = up_n - down_n
    m = len(others)

    mean_r = float(r.mean())
    msq = float((r ** 2).mean())
    rms = math.sqrt(msq) if msq > 0 else 0.0
    # mean(r)^2 <= mean(r^2) always (Cauchy-Schwarz) -> share is naturally in [0,1];
    # clamp only to absorb float noise, never to hide an actual sign/logic error.
    share = min(1.0, max(0.0, (mean_r ** 2) / msq)) if msq > 0 else 0.0

    d = x.diff().dropna(how="all")
    trend_share: Optional[float] = None
    cumulative_ad = 0
    if len(d) >= 2:
        num = m * (d.mean(axis=1) ** 2)
        den = (d ** 2).sum(axis=1)
        w = min(trend_window, len(d))
        tail_den = float(den.iloc[-w:].sum())
        if tail_den > 0:
            trend_share = round(float(num.iloc[-w:].sum()) / tail_den, 4)
        ad_step = np.sign(d).sum(axis=1)
        cumulative_ad = int(ad_step.sum())

    verdict = _verdict(breadth, share, m)

    strengths_latest = S.iloc[-1] if not S.empty else None
    crosses = []
    if strengths_latest is not None:
        fb = float(strengths_latest[base])
        for o in others:
            fq = -float(strengths_latest[o])
            denom = abs(fb) + abs(strengths_latest[o])
            base_share_pair = round(abs(fb) / denom, 4) if denom > 0 else 0.5
            crosses.append({
                "quote": o,
                "move": round(fb + fq, 6),
                "fromBase": round(fb, 6),
                "fromQuote": round(fq, 6),
                "baseShare": base_share_pair,
            })
        crosses.sort(key=lambda c: abs(c["move"]), reverse=True)

    return {
        "available": True,
        "breadth": breadth,
        "upN": up_n,
        "downN": down_n,
        "meanMove": round(mean_r, 6),
        "rms": round(rms, 6),
        "share": round(share, 4),
        "trendShare": trend_share,
        "cumulativeAD": cumulative_ad,
        "verdict": verdict,
        "crosses": crosses,
    }


# ---------------------------------------------------------------------------
# history grid (mirrors src/fx/snapshot.py's monthly backfill)
# ---------------------------------------------------------------------------
def _month_ends(as_of: pd.Timestamp, months: int) -> List[pd.Timestamp]:
    grid = list(pd.date_range(end=as_of.normalize(), periods=months, freq="ME"))
    if not grid or grid[-1].date() != as_of.date():
        grid.append(as_of.normalize())
    return grid


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def compute_breadth_snapshot(
    fx: FxData,
    universe_name: str = DEFAULT_UNIVERSE,
    horizon: int = DEFAULT_HORIZON,
    trend_window: int = DEFAULT_TREND_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    history_months: int = HISTORY_MONTHS,
) -> dict:
    universe = UNIVERSES.get(universe_name, MAJORS)
    L_full = log_prices_usd(fx, universe)
    empty = {
        "universe": universe_name, "horizon": horizon, "trendWindow": trend_window,
        "threshold": threshold, "strengths": {}, "byBase": {}, "history": {},
    }
    if L_full.empty or len(L_full) <= horizon:
        return empty

    S_full = strengths_frame(L_full)
    by_base = {
        base: base_metrics(L_full, S_full, base, universe, horizon, trend_window, threshold)
        for base in universe
    }

    history: Dict[str, dict] = {c: {"share": [], "breadth": [], "cumulativeAD": []} for c in universe}
    grid = _month_ends(L_full.index[-1], history_months)
    for ts in grid:
        sub = L_full[L_full.index <= ts]
        if len(sub) <= horizon:
            continue
        sub_s = strengths_frame(sub)
        d_iso = pd.Timestamp(ts).date().isoformat()
        for base in universe:
            m = base_metrics(sub, sub_s, base, universe, horizon, trend_window, threshold)
            if not m.get("available"):
                continue
            history[base]["share"].append({"date": d_iso, "value": m["share"]})
            history[base]["breadth"].append({"date": d_iso, "value": m["breadth"]})
            history[base]["cumulativeAD"].append({"date": d_iso, "value": m["cumulativeAD"]})

    return {
        "universe": universe_name,
        "horizon": horizon,
        "trendWindow": trend_window,
        "threshold": threshold,
        "strengths": latest_strengths(S_full),
        "byBase": by_base,
        "history": history,
    }
