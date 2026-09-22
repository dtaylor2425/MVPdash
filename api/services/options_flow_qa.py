"""
api/services/options_flow_qa.py

Pure QA/reporting math for the historical Options Flow backfill (Phase 1 QA report,
distribution QA, reconciliation, cross-ETF comparability, research-dataset row-building).
No Postgres, no ThetaData -- callers (scripts/generate_options_flow_qa_report.py,
scripts/export_options_flow_research_dataset.py) do the querying and pass in plain dicts
(parsed run/snapshot rows). Read-only: nothing here writes anything or changes any
calculation from options_flow_metrics.py.

A "row" here is {"ticker": str, "marketDate": date, "status": "success"|"partial",
"payload": <the stored JSON payload>, "diagnostics": <the stored JSON diagnostics>}.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PERCENTILES = (0, 1, 5, 50, 95, 99, 100)
PERCENTILE_LABELS = {0: "min", 1: "p1", 5: "p5", 50: "median", 95: "p95", 99: "p99", 100: "max"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _nums(vals: Sequence[Any]) -> np.ndarray:
    return np.array([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))],
                    dtype="float64")


def _get(d: Optional[Dict[str, Any]], *path: str) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def describe(vals: Sequence[Any]) -> Dict[str, Any]:
    """min/p1/p5/median/mean/p95/p99/max/n, dropping None/NaN. All-None -> all fields None, n=0."""
    a = _nums(vals)
    if len(a) == 0:
        out = {PERCENTILE_LABELS[p]: None for p in PERCENTILES}
        out["mean"] = None
        out["n"] = 0
        return out
    out = {PERCENTILE_LABELS[p]: float(np.percentile(a, p)) for p in PERCENTILES}
    out["mean"] = float(a.mean())
    out["n"] = int(len(a))
    return out


def _median(vals: Sequence[Any]) -> Optional[float]:
    a = _nums(vals)
    return float(np.median(a)) if len(a) else None


# --------------------------------------------------------------------------
# per-ticker-day extraction (payload -> flat fields used everywhere below)
# --------------------------------------------------------------------------

def is_full_flow(row: Dict[str, Any]) -> bool:
    return _get(row["payload"], "mode") == "full_flow" or (
        _get(row["payload"], "mode") is None and _get(row["payload"], "sentiment") is not None)


def is_iv_warmup(row: Dict[str, Any]) -> bool:
    return _get(row["payload"], "mode") == "iv_warmup"


def extract_full_flow_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    p = row["payload"]
    diag = row.get("diagnostics") or {}
    return {
        "ticker": row["ticker"], "marketDate": row["marketDate"], "status": row["status"],
        "netTradeSentiment": _get(p, "sentiment", "value"),
        "deltaImbalanceRatio": _get(p, "delta", "ratio"),
        "netDollarDelta": _get(p, "delta", "netDollar"),
        "grossPremium": _get(p, "premium", "gross"),
        "atmIv": _get(p, "iv", "atm"), "iv7d": _get(p, "iv", "iv7d"), "iv30d": _get(p, "iv", "iv30d"),
        "iv60d": _get(p, "iv", "iv60d"), "skew25d30d": _get(p, "iv", "skew25d30d"),
        "pct20d": _get(p, "iv", "percentiles", "20d"), "pct60d": _get(p, "iv", "percentiles", "60d"),
        "pct126d": _get(p, "iv", "percentiles", "126d"), "pct252d": _get(p, "iv", "percentiles", "252d"),
        "zeroDteShare": _get(p, "dte", "zeroDteShare"),
        "classifiedPctPremium": _get(p, "quality", "classifiedPctPremium"),
        "greekMatchedPctPremium": _get(p, "quality", "deltaMatchedPctPremium"),
        "oiMatchedPctPremium": _get(p, "quality", "oiMatchedPctPremium"),
        "trades": _get(p, "quality", "trades"), "eligibleTrades": _get(p, "quality", "eligibleTrades"),
        "callBought": _get(p, "aggression", "callBought"), "callSold": _get(p, "aggression", "callSold"),
        "putBought": _get(p, "aggression", "putBought"), "putSold": _get(p, "aggression", "putSold"),
        "dteBuckets": _get(p, "dte", "buckets") or [],
        "thetaRequests": diag.get("thetaRequests"), "runtimeSec": diag.get("runtimeSec"),
    }


def extract_iv_warmup_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    p = row["payload"]
    diag = row.get("diagnostics") or {}
    return {
        "ticker": row["ticker"], "marketDate": row["marketDate"], "status": row["status"],
        "atmIv": _get(p, "iv", "atm"), "iv7d": _get(p, "iv", "iv7d"), "iv30d": _get(p, "iv", "iv30d"),
        "iv60d": _get(p, "iv", "iv60d"), "skew25d30d": _get(p, "iv", "skew25d30d"),
        "pct20d": _get(p, "iv", "percentiles", "20d"), "pct60d": _get(p, "iv", "percentiles", "60d"),
        "pct126d": _get(p, "iv", "percentiles", "126d"), "pct252d": _get(p, "iv", "percentiles", "252d"),
        "thetaRequests": diag.get("thetaRequests"), "runtimeSec": diag.get("runtimeSec"),
    }


# --------------------------------------------------------------------------
# item 5: per-ETF + combined summary
# --------------------------------------------------------------------------

def summarize_ticker(
    ticker: str, requested_full: int, full_rows: List[Dict[str, Any]], full_failed: int,
    requested_warmup: int, warmup_rows: List[Dict[str, Any]], warmup_failed: int,
) -> Dict[str, Any]:
    """One row of the item-5 report for a single ticker (or the combined universe when
    ticker='ALL' and the row lists are the union across tickers)."""
    fs = [r for r in full_rows if r["status"] == "success"]
    fp = [r for r in full_rows if r["status"] == "partial"]
    ws = [r for r in warmup_rows if r["status"] == "success"]
    wp = [r for r in warmup_rows if r["status"] == "partial"]
    all_rows = full_rows + warmup_rows  # percentile completeness draws on both modes

    theta_all = [r["thetaRequests"] for r in full_rows + warmup_rows if r.get("thetaRequests") is not None]
    runtime_all = [r["runtimeSec"] for r in full_rows + warmup_rows if r.get("runtimeSec") is not None]

    n_full = max(1, len(full_rows))
    n_all = max(1, len(all_rows))
    return {
        "ticker": ticker,
        "fullFlowRequested": requested_full, "fullFlowSuccess": len(fs), "fullFlowPartial": len(fp),
        "fullFlowFailed": full_failed,
        "warmupRequested": requested_warmup, "warmupSuccess": len(ws), "warmupPartial": len(wp),
        "warmupFailed": warmup_failed,
        "totalThetaRequests": int(sum(theta_all)) if theta_all else 0,
        "medianRequestsPerTickerDay": _median(theta_all),
        "medianRuntimeSecPerTickerDay": _median(runtime_all),
        "totalWallClockSec": float(sum(runtime_all)) if runtime_all else 0.0,
        "medianTradesDownloaded": _median([r.get("trades") for r in full_rows]),
        "medianEligibleTrades": _median([r.get("eligibleTrades") for r in full_rows]),
        "medianGrossPremium": _median([r.get("grossPremium") for r in full_rows]),
        "medianClassifiedPctPremium": _median([r.get("classifiedPctPremium") for r in full_rows]),
        "medianGreekMatchedPctPremium": _median([r.get("greekMatchedPctPremium") for r in full_rows]),
        "medianOiMatchedPctPremium": _median([r.get("oiMatchedPctPremium") for r in full_rows]),
        "missingAtmIv": sum(1 for r in full_rows if r.get("atmIv") is None),
        "missingIv7d": sum(1 for r in full_rows if r.get("iv7d") is None),
        "missingIv30d": sum(1 for r in full_rows if r.get("iv30d") is None),
        "missingIv60d": sum(1 for r in full_rows if r.get("iv60d") is None),
        "missingSkew": sum(1 for r in full_rows if r.get("skew25d30d") is None),
        "missingDeltaImbalance": sum(1 for r in full_rows if r.get("netDollarDelta") is None),
        "missingPct20d": sum(1 for r in all_rows if r.get("pct20d") is None),
        "missingPct60d": sum(1 for r in all_rows if r.get("pct60d") is None),
        "missingPct126d": sum(1 for r in all_rows if r.get("pct126d") is None),
        "missingPct252d": sum(1 for r in all_rows if r.get("pct252d") is None),
        "missingAtmIvPct": sum(1 for r in full_rows if r.get("atmIv") is None) / n_full,
        "missingPct252dPct": sum(1 for r in all_rows if r.get("pct252d") is None) / n_all,
    }


# --------------------------------------------------------------------------
# item 6: distribution QA (per ETF)
# --------------------------------------------------------------------------

DISTRIBUTION_FIELDS = [
    ("netTradeSentiment", "net_trade_sentiment"), ("deltaImbalanceRatio", "delta_imbalance_ratio"),
    ("netDollarDelta", "net_dollar_delta_imbalance"), ("grossPremium", "gross_premium"),
    ("atmIv", "atm_iv"), ("skew25d30d", "put_skew_25d"), ("zeroDteShare", "zero_dte_share"),
    ("classifiedPctPremium", "classification_coverage"), ("greekMatchedPctPremium", "greek_match_coverage"),
    ("oiMatchedPctPremium", "oi_match_coverage"),
]


def distribution_qa(full_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{label: describe(values)} for every DISTRIBUTION_FIELDS metric, full-flow rows only."""
    return {label: describe([r.get(key) for r in full_rows]) for key, label in DISTRIBUTION_FIELDS}


EXTREME_Z = 4.0   # |value - median| / MAD-based robust sigma beyond this -> flagged, never dropped


def flag_extreme_observations(full_rows: List[Dict[str, Any]], field: str = "netTradeSentiment") -> List[Dict[str, Any]]:
    """Robust (median/MAD) outlier flags for one field -- flags, never removes, matching
    'flag extreme observations rather than deleting them'."""
    vals = _nums([r.get(field) for r in full_rows])
    if len(vals) < 8:
        return []
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) or 1e-12
    robust_sigma = mad * 1.4826
    flagged = []
    for r in full_rows:
        v = r.get(field)
        if v is None:
            continue
        z = abs(v - med) / robust_sigma
        if z >= EXTREME_Z:
            flagged.append({"ticker": r["ticker"], "marketDate": r["marketDate"], "field": field,
                            "value": v, "median": med, "robustZ": round(z, 2)})
    return sorted(flagged, key=lambda x: -x["robustZ"])


# --------------------------------------------------------------------------
# item 7: reconciliation
# --------------------------------------------------------------------------

def reconciliation_errors(full_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Two checks against `premium.gross` (total classified/eligible premium):
      aggression:  call_bought + call_sold + put_bought + put_sold
                   -- this is EXPECTED to be <= gross, not equal: inside-spread trades
                   carry a fractional aggressor score in [-1, 1], so the difference
                   measures the premium of trades that printed between the bid and ask,
                   not a bug. Reported as a signed and relative gap either way.
      dte_buckets: sum(bucket.grossPremium for bucket in dte.buckets)
                   -- buckets PARTITION eligible trades by DTE with no overlap, so this
                   SHOULD equal gross to floating-point precision; a nonzero gap here is
                   a genuine reconciliation problem, not an expected artefact.
    """
    agg_diffs, agg_rel = [], []
    dte_diffs, dte_rel = [], []
    for r in full_rows:
        gross = r.get("grossPremium")
        if gross is None:
            continue
        agg_sum = sum(v for v in (r.get("callBought"), r.get("callSold"), r.get("putBought"), r.get("putSold"))
                     if v is not None)
        agg_diffs.append(gross - agg_sum)
        if gross > 0:
            agg_rel.append((gross - agg_sum) / gross)
        dte_sum = sum((b.get("grossPremium") or 0.0) for b in r.get("dteBuckets") or [])
        dte_diffs.append(gross - dte_sum)
        if gross > 0:
            dte_rel.append((gross - dte_sum) / gross)
    return {
        "aggression": {
            "note": "call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) "
                    "premium; a positive gap is expected, not a bug.",
            "medianAbsDiff": _median([abs(x) for x in agg_diffs]), "worstAbsDiff": max((abs(x) for x in agg_diffs), default=None),
            "medianRelDiff": _median(agg_rel), "worstRelDiff": max((abs(x) for x in agg_rel), default=None),
            "n": len(agg_diffs),
        },
        "dteBuckets": {
            "note": "DTE buckets partition eligible premium with no overlap; should reconcile to "
                    "floating-point precision. A material gap here is a real bug.",
            "medianAbsDiff": _median([abs(x) for x in dte_diffs]), "worstAbsDiff": max((abs(x) for x in dte_diffs), default=None),
            "medianRelDiff": _median(dte_rel), "worstRelDiff": max((abs(x) for x in dte_rel), default=None),
            "n": len(dte_diffs),
        },
    }


# --------------------------------------------------------------------------
# item 8: cross-ETF comparability
# --------------------------------------------------------------------------

def cross_etf_comparability(rows_by_ticker: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for ticker, rows in rows_by_ticker.items():
        out.append({
            "ticker": ticker,
            "medianGrossPremium": _median([r.get("grossPremium") for r in rows]),
            "medianNetDollarDelta": _median([r.get("netDollarDelta") for r in rows]),
            "medianAbsNetDollarDelta": _median([abs(r["netDollarDelta"]) for r in rows if r.get("netDollarDelta") is not None]),
            "medianTrades": _median([r.get("trades") for r in rows]),
        })
    out.sort(key=lambda x: -(x["medianGrossPremium"] or 0))
    return out


def comparability_verdict(comparability: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """(raw_magnitudes_comparable, note). Never invents a normalization -- just measures
    the spread of raw magnitudes across tickers and states the factual conclusion."""
    prem = [c["medianGrossPremium"] for c in comparability if c["medianGrossPremium"]]
    if len(prem) < 2:
        return True, "not enough tickers with data yet to assess."
    ratio = max(prem) / max(min(prem), 1e-9)
    comparable = ratio < 3.0
    note = ("Median gross premium spans a {:.0f}x range across tickers ({}); raw dollar magnitudes "
           "are {} directly comparable cross-sectionally. Use net_trade_sentiment / "
           "delta_imbalance_ratio (already normalized to [-1, 1] / scale-free) for cross-sectional "
           "ranking, not raw premium or dollar delta.").format(
        ratio, ", ".join("{}=${:,.0f}".format(c["ticker"], c["medianGrossPremium"]) for c in comparability),
        "not" if not comparable else "marginally")
    return comparable, note


# --------------------------------------------------------------------------
# item 9: research dataset rows (full-flow only, point-in-time)
# --------------------------------------------------------------------------

def research_row(row: Dict[str, Any], methodology_version: str) -> Dict[str, Any]:
    """One (date, ticker) row for the research dataset. Only fields knowable by market close
    that date -- no forward-looking data of any kind."""
    p = row["payload"]
    f = extract_full_flow_fields(row)
    buckets = {b["bucket"]: b for b in f["dteBuckets"]}

    def bucket_val(name: str, key: str) -> Optional[float]:
        b = buckets.get(name)
        return None if b is None else b.get(key)

    return {
        "date": row["marketDate"], "ticker": row["ticker"], "methodology_version": methodology_version,
        "net_trade_sentiment": f["netTradeSentiment"],
        "net_directional_premium": _get(p, "premium", "netDirectional"),
        "gross_premium": f["grossPremium"],
        "delta_imbalance_ratio": f["deltaImbalanceRatio"], "net_dollar_delta": f["netDollarDelta"],
        "call_bought_premium": f["callBought"], "call_sold_premium": f["callSold"],
        "put_bought_premium": f["putBought"], "put_sold_premium": f["putSold"],
        "zero_dte_share": f["zeroDteShare"],
        "dte_0dte_gross_premium": bucket_val("0DTE", "grossPremium"),
        "dte_1_7d_gross_premium": bucket_val("1-7D", "grossPremium"),
        "dte_8_30d_gross_premium": bucket_val("8-30D", "grossPremium"),
        "dte_31_60d_gross_premium": bucket_val("31-60D", "grossPremium"),
        "dte_60d_plus_gross_premium": bucket_val("60D+", "grossPremium"),
        "atm_iv": f["atmIv"], "iv_7d": f["iv7d"], "iv_30d": f["iv30d"], "iv_60d": f["iv60d"],
        "iv_term_slope_7v30": _get(p, "iv", "spread7v30"), "iv_term_slope_30v60": _get(p, "iv", "spread30v60"),
        "put_skew_25d": f["skew25d30d"],
        "iv_percentile_20d": f["pct20d"], "iv_percentile_60d": f["pct60d"],
        "iv_percentile_126d": f["pct126d"], "iv_percentile_252d": f["pct252d"],
        "classification_coverage": f["classifiedPctPremium"], "greek_match_coverage": f["greekMatchedPctPremium"],
        "oi_match_coverage": f["oiMatchedPctPremium"], "trade_count": f["trades"],
    }
