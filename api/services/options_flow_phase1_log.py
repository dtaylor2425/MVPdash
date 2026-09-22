"""
api/services/options_flow_phase1_log.py

Pure formatting/parsing for the Phase 1 progress log (logs/options-flow-phase1.log).
No Postgres, no ThetaData -- scripts/log_options_flow_phase1_progress.py does the
querying and file I/O; this module only formats lines and decides checkpoint timing,
so it is fully unit-testable without a database.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s+(?P<ticker>[A-Z0-9.\-]+)\s+(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<mode>\S+)\s+(?P<status>\S+)\s+(?P<rest>.*)$"
)


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else "{:.1%}".format(x)


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    sign = "+" if x >= 0 else "-"
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return "{}${:.2f}{}".format(sign, a / div, suf)
    return "{}${:.0f}".format(sign, a)


def _fmt_num(x: Optional[float]) -> str:
    return "n/a" if x is None else "{:,.0f}".format(x)


def _fmt_val(x: Optional[float]) -> str:
    return "n/a" if x is None else "{:+.4f}".format(x)


def format_line(
    ticker: str, market_date: date, mode: str, status: str, runtime_sec: Optional[float],
    theta_requests: Optional[int], trades: Optional[int], classified_pct: Optional[float],
    greek_pct: Optional[float], oi_pct: Optional[float], sentiment: Optional[float],
    delta_imbalance: Optional[float], reason: Optional[str] = None, now: Optional[datetime] = None,
) -> str:
    """One line, fixed field order, always the same fields (n/a where a mode/status has none --
    never a manufactured 0)."""
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    parts = [
        ts, ticker, market_date.isoformat(), mode, status,
        "runtime={}".format("n/a" if runtime_sec is None else "{:.1f}s".format(runtime_sec)),
        "requests={}".format("n/a" if theta_requests is None else theta_requests),
        "trades={}".format(_fmt_num(trades)),
        "classCov={}".format(_fmt_pct(classified_pct)),
        "greekCov={}".format(_fmt_pct(greek_pct)),
        "oiCov={}".format(_fmt_pct(oi_pct)),
        "sentiment={}".format(_fmt_val(sentiment)),
        "deltaImb={}".format(_fmt_money(delta_imbalance)),
    ]
    line = "  ".join(parts)
    if status == "failed" and reason:
        line += "  reason={}".format(reason)
    return line


def parse_logged_keys(log_text: str) -> set:
    """{(ticker, date_iso, mode)} already present in an existing log file, for dedup on resume."""
    keys = set()
    for line in log_text.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            keys.add((m.group("ticker"), m.group("date"), m.group("mode")))
    return keys


def format_checkpoint(
    full_success: int, full_partial: int, full_failed: int, full_requested: int,
    warm_success: int, warm_partial: int, warm_failed: int, warm_requested: int,
    started_at: Optional[datetime], now: Optional[datetime] = None,
) -> str:
    now = now or datetime.now(timezone.utc)
    elapsed = "n/a"
    if started_at is not None:
        secs = (now - started_at).total_seconds()
        h, rem = divmod(int(secs), 3600)
        m, s = divmod(rem, 60)
        elapsed = "{}h{:02d}m{:02d}s".format(h, m, s)
    full_done = full_success + full_partial + full_failed
    warm_done = warm_success + warm_partial + warm_failed
    return "\n".join([
        "PHASE 1",
        "Full flow: {} / {}".format(full_done, full_requested),
        "Warmup: {} / {}".format(warm_done, warm_requested),
        "Failed: {}".format(full_failed + warm_failed),
        "Partial: {}".format(full_partial + warm_partial),
        "Elapsed: {}".format(elapsed),
    ])


def checkpoint_boundaries_crossed(prev_total: int, new_total: int, every: int = 10) -> List[int]:
    """Which multiples of `every` fall strictly between prev_total (exclusive) and new_total
    (inclusive) -- lets the caller emit exactly one checkpoint per 10-ticker-day boundary even
    when a poll discovers several new rows at once."""
    if new_total <= prev_total:
        return []
    first = (prev_total // every + 1) * every
    return list(range(first, new_total + 1, every)) if first <= new_total else []
