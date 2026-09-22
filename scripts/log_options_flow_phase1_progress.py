"""
scripts/log_options_flow_phase1_progress.py

Read-only Phase 1 progress logger. Polls Postgres on an interval (never ThetaData, never
writes to Postgres) and appends newly-published/failed ticker-days to a persistent log,
default logs/options-flow-phase1.log, in append-only, resume-safe fashion: on each poll it
re-derives "already logged" from the log file itself (api.services.options_flow_phase1_log
.parse_logged_keys), so starting, stopping, or running this alongside the backfill any
number of times never duplicates a line. Emits a PHASE 1 checkpoint block every 10
ticker-days (combined full-flow + iv-warmup).

Entirely independent of the backfill process -- it does not touch, restart, or depend on
jobs/options_flow_backfill.py in any way, so it is safe to run while a backfill is live and
never counts as "another ThetaData process."

    python scripts/log_options_flow_phase1_progress.py                       # poll every 30s until Ctrl-C
    python scripts/log_options_flow_phase1_progress.py --once                # one poll, exit
    python scripts/log_options_flow_phase1_progress.py --interval 60
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_phase1_log as plog  # noqa: E402
from api.services.options_flow_calendar import sessions_between  # noqa: E402
from scripts.generate_options_flow_qa_report import (  # noqa: E402
    DEFAULT_FULL_FLOW,
    DEFAULT_TICKERS,
    DEFAULT_WARMUP,
    fetch_failed,
    fetch_published,
    reconcile_failed,
)

LOG_PATH = ROOT / "logs" / "options-flow-phase1.log"


def _q(r: Dict[str, Any]) -> Dict[str, Any]:
    return r["payload"].get("quality") or {}


def rows_to_events(published: List[Dict[str, Any]], failed: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Merge published + failed rows into one chronological event list (sorted by createdAt)."""
    events = []
    for r in published:
        p, q = r["payload"], _q(r)
        events.append({
            "createdAt": r.get("createdAt") or datetime.min.replace(tzinfo=timezone.utc),
            "ticker": r["ticker"], "marketDate": r["marketDate"], "mode": mode, "status": r["status"],
            "runtimeSec": r["diagnostics"].get("runtimeSec"), "thetaRequests": q.get("thetaRequests") or r["diagnostics"].get("thetaRequests"),
            "trades": q.get("trades"), "classifiedPct": q.get("classifiedPctPremium"),
            "greekPct": q.get("deltaMatchedPctPremium"), "oiPct": q.get("oiMatchedPctPremium"),
            "sentiment": (p.get("sentiment") or {}).get("value"), "deltaImbalance": (p.get("delta") or {}).get("netDollar"),
            "reason": None,
        })
    for f in failed:
        d = f["diagnostics"]
        events.append({
            "createdAt": f.get("createdAt") or datetime.min.replace(tzinfo=timezone.utc),
            "ticker": f["ticker"], "marketDate": f["marketDate"], "mode": mode, "status": "failed",
            "runtimeSec": d.get("runtimeSec"), "thetaRequests": d.get("thetaRequests"),
            "trades": None, "classifiedPct": None, "greekPct": None, "oiPct": None,
            "sentiment": None, "deltaImbalance": None,
            "reason": (d.get("missingReasons") or [d.get("error")])[0],
        })
    events.sort(key=lambda e: e["createdAt"])
    return events


def poll_once(conn, tickers: List[str], full_range, warmup_range, log_path: Path) -> int:
    """Appends any newly-completed ticker-day to log_path. Returns how many lines were appended."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    already = plog.parse_logged_keys(existing_text)

    all_events: List[Dict[str, Any]] = []
    for mode, rng in (("full_flow", full_range), ("iv_warmup", warmup_range)):
        pub = fetch_published(conn, tickers, mode, *rng)
        fail = reconcile_failed(fetch_failed(conn, tickers, mode, *rng), pub)
        all_events += rows_to_events(pub, fail, mode)
    all_events.sort(key=lambda e: e["createdAt"])

    new_events = [e for e in all_events if (e["ticker"], e["marketDate"].isoformat(), e["mode"]) not in already]
    if not new_events:
        return 0

    # running total (for checkpoint boundaries) = count of every distinct ticker-day-mode ever logged
    prior_total = len(already)
    started_at = min((e["createdAt"] for e in all_events), default=None)
    if started_at == datetime.min.replace(tzinfo=timezone.utc):
        started_at = None

    lines: List[str] = []
    running_total = prior_total
    for e in new_events:
        lines.append(plog.format_line(
            e["ticker"], e["marketDate"], e["mode"], e["status"], e["runtimeSec"], e["thetaRequests"],
            e["trades"], e["classifiedPct"], e["greekPct"], e["oiPct"], e["sentiment"], e["deltaImbalance"],
            reason=e["reason"], now=e["createdAt"] if e["createdAt"] != datetime.min.replace(tzinfo=timezone.utc) else None,
        ))
        running_total += 1
        if running_total % 10 == 0:
            # recompute cumulative status counts from ALL events up to and including this one
            seen = {(x["ticker"], x["marketDate"].isoformat(), x["mode"]) for x in new_events[: new_events.index(e) + 1]} | already
            counted = [x for x in all_events if (x["ticker"], x["marketDate"].isoformat(), x["mode"]) in seen]
            fs = sum(1 for x in counted if x["mode"] == "full_flow" and x["status"] == "success")
            fp = sum(1 for x in counted if x["mode"] == "full_flow" and x["status"] == "partial")
            ff = sum(1 for x in counted if x["mode"] == "full_flow" and x["status"] == "failed")
            ws = sum(1 for x in counted if x["mode"] == "iv_warmup" and x["status"] == "success")
            wp = sum(1 for x in counted if x["mode"] == "iv_warmup" and x["status"] == "partial")
            wf = sum(1 for x in counted if x["mode"] == "iv_warmup" and x["status"] == "failed")
            lines.append("")
            lines.append(plog.format_checkpoint(fs, fp, ff, len(full_range_sessions) * len(tickers), ws, wp, wf,
                                                len(warmup_range_sessions) * len(tickers), started_at, e["createdAt"]))
            lines.append("")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(new_events)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--full-flow-start", default=DEFAULT_FULL_FLOW[0].isoformat())
    ap.add_argument("--full-flow-end", default=DEFAULT_FULL_FLOW[1].isoformat())
    ap.add_argument("--warmup-start", default=DEFAULT_WARMUP[0].isoformat())
    ap.add_argument("--warmup-end", default=DEFAULT_WARMUP[1].isoformat())
    ap.add_argument("--log-path", default=str(LOG_PATH))
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between polls")
    ap.add_argument("--once", action="store_true", help="poll once and exit instead of looping")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    full_range = (date.fromisoformat(args.full_flow_start), date.fromisoformat(args.full_flow_end))
    warmup_range = (date.fromisoformat(args.warmup_start), date.fromisoformat(args.warmup_end))
    log_path = Path(args.log_path)

    global full_range_sessions, warmup_range_sessions  # simplest way to share with poll_once without a bigger refactor
    full_range_sessions = sessions_between(*full_range)
    warmup_range_sessions = sessions_between(*warmup_range)

    from api.db import get_connection

    print("Logging Phase 1 progress to {} (poll every {:.0f}s, Ctrl-C to stop)".format(log_path, args.interval))
    while True:
        try:
            with get_connection() as conn:
                n = poll_once(conn, tickers, full_range, warmup_range, log_path)
            if n:
                print("appended {} new ticker-day line(s)".format(n))
        except Exception as e:  # noqa: BLE001 -- a transient DB hiccup must not kill the logger
            print("poll failed (will retry): {}: {}".format(type(e).__name__, e))
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
