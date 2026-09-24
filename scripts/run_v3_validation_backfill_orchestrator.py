"""
scripts/run_v3_validation_backfill_orchestrator.py

Signal Validation v3's backfill driver. `jobs/options_flow_backfill.py` refuses to combine
--resume and --overwrite (they are mutually exclusive), but this job needs BOTH behaviors at
once: every target date already has a pre-existing iv_warmup row (archived separately -- see
scripts/archive_preexisting_iv_warmup.py), so writing full_flow there requires --overwrite, yet
a crash partway through must not re-do ticker-days that were already converted. This orchestrator
provides that by re-computing, per ticker, the first NOT-YET-full_flow date before every attempt,
and invoking the real CLI with a narrowed --start so nothing already converted is repeated.

Singleton lock, heartbeat, --status, and fatal-error classification are all still the real
jobs/options_flow_backfill.py's -- this script only decides WHICH --tickers/--start/--end to
pass it and loops per ticker; it never talks to ThetaData or Postgres for data itself (only the
one read-only "what's already full_flow" query below, via the same store.backfill_existing used
by --resume).

    python scripts/run_v3_validation_backfill_orchestrator.py                 # run to completion
    python scripts/run_v3_validation_backfill_orchestrator.py --status-only   # just print progress
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
RANGE_START = date(2025, 8, 21)
RANGE_END = date(2026, 6, 25)
PY = sys.executable


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    out = subprocess.run(["railway", "variables", "--service", "Postgres", "--kv"],
                         capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.startswith("DATABASE_PUBLIC_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Could not resolve DATABASE_URL")


def trading_sessions(start: date, end: date) -> List[date]:
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=str(start), end_date=str(end))
    return sorted(sched.index.date.tolist())


def first_remaining_date(conn, ticker: str, sessions: List[date]) -> Optional[date]:
    """First session in `sessions` that does NOT yet have a full_flow success/partial row."""
    from api.services.options_flow_store import backfill_existing
    done = backfill_existing(conn, [ticker], sessions[0], sessions[-1], mode="full_flow")
    for d in sessions:
        if (ticker, d) not in done:
            return d
    return None


def count_done(conn, ticker: str, sessions: List[date]) -> int:
    """How many sessions currently have a full_flow success/partial row -- used (instead of
    "is the first remaining date the same") to detect genuine stalls. A single persistently
    flaky date (e.g. one transient ThetaData 502) sitting at the FRONT of the remaining range
    would make "first remaining date" look unchanged forever even while every later date keeps
    succeeding -- comparing the total count avoids that false stall and avoids one bad date
    blocking the whole ticker (let alone the whole orchestrator)."""
    from api.services.options_flow_store import backfill_existing
    return len(backfill_existing(conn, [ticker], sessions[0], sessions[-1], mode="full_flow"))


def print_status(sessions: List[date]) -> None:
    import psycopg
    from psycopg.rows import dict_row
    from api.services.options_flow_store import backfill_existing

    conn = psycopg.connect(_get_database_url(), row_factory=dict_row, autocommit=True)
    try:
        print(f"v3 validation backfill status -- target range {sessions[0]} -> {sessions[-1]} "
             f"({len(sessions)} sessions x {len(TICKERS)} tickers = {len(sessions) * len(TICKERS)} ticker-days)")
        total_done = 0
        for t in TICKERS:
            done = backfill_existing(conn, [t], sessions[0], sessions[-1], mode="full_flow")
            n_done = len(done)
            total_done += n_done
            nxt = first_remaining_date(conn, t, sessions)
            print(f"  {t}: {n_done}/{len(sessions)} full_flow done"
                 + (f", next remaining: {nxt}" if nxt else " -- COMPLETE"))
        print(f"TOTAL: {total_done}/{len(sessions) * len(TICKERS)} ticker-days")
    finally:
        conn.close()


def run_one_ticker_leg(ticker: str, start: date, end: date) -> int:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "thetadl" / "site"))
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [PY, "jobs/options_flow_backfill.py", "--tickers", ticker,
          "--start", str(start), "--end", str(end), "--overwrite"]
    print(f"=== v3 backfill leg: {ticker} {start} -> {end} (--overwrite) ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print(f"=== leg exit code: {proc.returncode} ===", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()

    sessions = trading_sessions(RANGE_START, RANGE_END)
    print(f"v3 validation+warmup target: {len(sessions)} NYSE sessions, {sessions[0]} -> {sessions[-1]}")
    if len(sessions) != 212:
        print(f"WARNING: expected 212 sessions, computed {len(sessions)} -- trading calendar may "
             f"have changed since the manifest was written. Proceeding with the computed set.")

    if args.status_only:
        print_status(sessions)
        return 0

    MAX_RETRIES_PER_TICKER = 6
    incomplete_tickers: List[str] = []

    import psycopg
    from psycopg.rows import dict_row
    conn = psycopg.connect(_get_database_url(), row_factory=dict_row, autocommit=True)
    try:
        for ticker in TICKERS:
            attempt = 0
            while True:
                nxt = first_remaining_date(conn, ticker, sessions)
                if nxt is None:
                    print(f"{ticker}: already complete for the full v3 range.")
                    break
                attempt += 1
                if attempt > MAX_RETRIES_PER_TICKER:
                    still_missing = len(sessions) - count_done(conn, ticker, sessions)
                    print(f"{ticker}: still {still_missing} date(s) unconverted (first: {nxt}) "
                         f"after {MAX_RETRIES_PER_TICKER} attempts -- a persistently failing "
                         f"date, not an infinite loop (other dates keep succeeding). Moving on "
                         f"to the next ticker; investigate {ticker} {nxt} separately, then "
                         f"re-run this script (it will pick up exactly what's missing).")
                    incomplete_tickers.append(ticker)
                    break
                total_before = count_done(conn, ticker, sessions)
                code = run_one_ticker_leg(ticker, nxt, sessions[-1])
                if code == 3:
                    print(f"FATAL error classification from the backfill CLI for {ticker} "
                         f"starting {nxt} (exit 3) -- stopping the ENTIRE orchestrator, no "
                         f"fallthrough to other tickers.")
                    return 3
                if code == 130:
                    print("Backfill CLI was interrupted (exit 130) -- stopping the orchestrator.")
                    return 130
                # 0 (all done), 1 (nothing published/refused), 2 (partial) all fall through to
                # re-check what's actually left in Postgres and continue -- ground truth (the
                # TOTAL count of converted dates, not just whether the first remaining date
                # happens to be unchanged), not the exit code, decides whether to keep looping.
                total_after = count_done(conn, ticker, sessions)
                if total_after == total_before:
                    print(f"{ticker}: no additional dates converted on attempt {attempt} "
                         f"(leg exit {code}, still stuck at {nxt}) -- will retry "
                         f"({attempt}/{MAX_RETRIES_PER_TICKER}).")
                else:
                    print(f"{ticker}: {total_after - total_before} more date(s) converted this "
                         f"attempt ({total_after}/{len(sessions)} total).")
    finally:
        conn.close()

    if incomplete_tickers:
        print(f"v3 validation+warmup backfill: STOPPED WITH GAPS in {incomplete_tickers} -- "
             f"re-run this script after investigating; every other ticker is complete.")
        return 2

    print("v3 validation+warmup backfill: ALL TICKERS COMPLETE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
