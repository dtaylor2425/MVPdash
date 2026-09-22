"""
api/services/options_flow_status.py

Pure formatting + ETA math for `jobs/options_flow_backfill.py --status`. No Postgres, no
ThetaData -- the CLI does the querying (lock probe, heartbeat read, ticker-day counts) and
passes plain values in here. Medians (not means) are used throughout so one slow outlier day
(e.g. a 335s SPY day) doesn't distort the estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


def _median(vals: Sequence[float]) -> Optional[float]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2.0


def median_runtime_by_ticker(
    successful_runtimes: Sequence[Dict[str, Any]], min_ticker_observations: int = 5,
) -> Dict[str, Optional[float]]:
    """
    successful_runtimes: [{"ticker": str, "runtimeSec": float}, ...] -- SUCCESSFUL ticker-days
    only (never failed/partial: a failed day's runtime says nothing about how long a real one
    takes). Returns {ticker: median_runtime_or_None, "__overall__": overall_median_or_None}.
    A ticker's own median is used once it has >= min_ticker_observations; below that its
    'own' entry is None and callers should fall back to "__overall__".
    """
    by_ticker: Dict[str, List[float]] = {}
    all_vals: List[float] = []
    for r in successful_runtimes:
        rt = r.get("runtimeSec")
        if rt is None:
            continue
        by_ticker.setdefault(r["ticker"], []).append(rt)
        all_vals.append(rt)
    out: Dict[str, Optional[float]] = {
        t: (_median(vals) if len(vals) >= min_ticker_observations else None) for t, vals in by_ticker.items()
    }
    out["__overall__"] = _median(all_vals)
    return out


def eta_for_ticker(medians: Dict[str, Optional[float]], ticker: str, remaining: int) -> Optional[float]:
    """Seconds remaining for one ticker: its own median if it has enough observations, else
    the overall (all-tickers) median for this mode. None (unknown) only if neither exists."""
    if remaining <= 0:
        return 0.0
    m = medians.get(ticker) or medians.get("__overall__")
    return None if m is None else m * remaining


def remaining_by_ticker(
    tickers: Sequence[str], requested_per_ticker: int, published_rows: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    """Per-ticker remaining ticker-days (requested - successful - partial; a failed day is
    still remaining, same semantics as mode_progress)."""
    done = {t: 0 for t in tickers}
    for r in published_rows:
        if r.get("status") in ("success", "partial") and r.get("ticker") in done:
            done[r["ticker"]] += 1
    return {t: max(0, requested_per_ticker - done[t]) for t in tickers}


def total_eta_seconds(medians: Dict[str, Optional[float]], remaining: Dict[str, int]) -> Optional[float]:
    """Sum of eta_for_ticker across tickers. None if some ticker has remaining work but no
    median at all (own or overall) to estimate it from -- never silently understate the total."""
    total = 0.0
    for t, rem in remaining.items():
        if rem <= 0:
            continue
        e = eta_for_ticker(medians, t, rem)
        if e is None:
            return None
        total += e
    return total


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "{}h {:02d}m".format(h, m)
    if m:
        return "{}m {:02d}s".format(m, s)
    return "{}s".format(s)


def mode_progress(requested: int, successful: int, partial: int, failed: int) -> Dict[str, int]:
    """'remaining' = not yet successfully or partially completed -- failed days still count as
    remaining work (they are retried on the next --resume, never treated as done)."""
    return {"requested": requested, "successful": successful, "partial": partial, "failed": failed,
            "remaining": max(0, requested - successful - partial)}


def format_status(
    active: bool, heartbeat: Optional[Dict[str, Any]], methodology_version: str,
    full_flow: Dict[str, int], iv_warmup: Dict[str, int],
    full_flow_eta_sec: Optional[float], full_flow_median_sec: Optional[float],
    warmup_eta_sec: Optional[float], warmup_median_sec: Optional[float],
    now: Optional[datetime] = None,
) -> str:
    now = now or datetime.now(timezone.utc)
    hb = heartbeat or {}
    combined_eta = None
    if full_flow_eta_sec is not None or warmup_eta_sec is not None:
        combined_eta = (full_flow_eta_sec or 0.0) + (warmup_eta_sec or 0.0)

    lines = [
        "OPTIONS FLOW BACKFILL STATUS", "",
        "Active process: {}".format("YES" if active else "NO"),
        "Active run ID: {}".format(hb.get("run_process_id") or "n/a"),
        "Phase: {}".format(hb.get("phase") or "none"),
        "Current ticker: {}".format(hb.get("ticker") or "n/a"),
        "Current date: {}".format(hb.get("market_date").isoformat() if hb.get("market_date") else "n/a"),
        "Last heartbeat: {}".format(hb.get("last_heartbeat").isoformat() if hb.get("last_heartbeat") else "n/a"),
        "",
        "FULL FLOW",
        "Successful: {} / {}".format(full_flow["successful"], full_flow["requested"]),
        "Partial: {}".format(full_flow["partial"]),
        "Failed: {}".format(full_flow["failed"]),
        "Remaining: {}".format(full_flow["remaining"]),
        "Median successful runtime: {}".format("n/a" if full_flow_median_sec is None else "{:.0f}s".format(full_flow_median_sec)),
        "ETA (estimate): {}".format(format_duration(full_flow_eta_sec)),
        "",
        "IV WARMUP",
        "Successful: {} / {}".format(iv_warmup["successful"], iv_warmup["requested"]),
        "Partial: {}".format(iv_warmup["partial"]),
        "Failed: {}".format(iv_warmup["failed"]),
        "Remaining: {}".format(iv_warmup["remaining"]),
        "Median successful runtime: {}".format("n/a" if warmup_median_sec is None else "{:.0f}s".format(warmup_median_sec)),
        "ETA (estimate): {}".format(format_duration(warmup_eta_sec)),
        "",
        "Combined ETA (estimate): ~{}".format(format_duration(combined_eta)),
        "",
        "Methodology: {}".format(methodology_version),
    ]
    return "\n".join(lines)
