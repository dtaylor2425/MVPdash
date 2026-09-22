"""
api/services/options_flow_heartbeat.py

A single-row "what is the active backfill process doing right now" table, purely for
observability (jobs/options_flow_backfill.py --status). Never read to decide whether a new
process may start -- api/services/options_flow_lock.py's advisory lock is the sole authority
for that; a stale heartbeat here must never be treated as evidence a process is or isn't
running. This table is entirely separate from options_flow_runs / options_flow_symbol_snapshots
and never touches them -- writing a heartbeat can never alter an analytics snapshot.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

# Schema lives in sql/009_options_flow_control_plane.sql, applied via
# options_flow_store.ensure_schema (its DDL_PATHS already includes it) -- no separate
# ensure_schema here, to keep one canonical place that creates every options-flow table.


def new_run_process_id() -> str:
    """One identifier per process start (not per ticker-day), shown as --status's 'Active run ID'."""
    return uuid.uuid4().hex[:12]


def upsert_heartbeat(
    conn: Any, run_process_id: str, phase: Optional[str], ticker: Optional[str],
    market_date: Optional[date], methodology_version: str, started_at: datetime,
    now: Optional[datetime] = None,
) -> None:
    now = now or datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO options_flow_backfill_heartbeat
                (id, run_process_id, phase, ticker, market_date, hostname, pid,
                 started_at, last_heartbeat, methodology_version)
            VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                run_process_id = EXCLUDED.run_process_id,
                phase = EXCLUDED.phase,
                ticker = EXCLUDED.ticker,
                market_date = EXCLUDED.market_date,
                hostname = EXCLUDED.hostname,
                pid = EXCLUDED.pid,
                started_at = EXCLUDED.started_at,
                last_heartbeat = EXCLUDED.last_heartbeat,
                methodology_version = EXCLUDED.methodology_version
            """,
            (run_process_id, phase, ticker, market_date, socket.gethostname(), os.getpid(),
             started_at, now, methodology_version),
        )
    conn.commit()


def read_heartbeat(conn: Any) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM options_flow_backfill_heartbeat WHERE id = 1")
        row = cur.fetchone()
    return dict(row) if row else None
