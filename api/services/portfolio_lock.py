"""
api/services/portfolio_lock.py

Postgres session-level advisory lock, one per strategy, used by the nightly portfolio
refresh (jobs/nightly_portfolio_refresh.py) so two overlapping invocations building the SAME
strategy can never race each other. Mirrors api/services/options_flow_lock.py's pattern
exactly: a session-level advisory lock is held by the Postgres BACKEND for as long as the
connection that acquired it stays open, and Postgres releases it automatically the moment that
connection closes or drops -- for any reason, including the process dying -- so there is
nothing to clean up on crash. No table, no PID file.

This deliberately replaces the job's old pg_advisory_xact_lock(...) call. That variant BLOCKS
until the lock is free, which is exactly what let one stuck invocation wedge every later cron
trigger behind it for hours. Every caller here must use try_acquire (non-blocking) and treat a
False result as "another invocation already owns this strategy right now -- skip cleanly," never
wait for it.

Callers acquire the lock on a DEDICATED connection held open for the whole strategy run (never
used for queries), separate from the connection(s) used to read/write portfolio data -- so a
lock probe or a data-connection hiccup can never be confused with "another process holds the
lock."
"""

from __future__ import annotations

import zlib
from typing import Any

LOCK_NAME_PREFIX = "macro_engine_portfolio"


def _lock_key(strategy: str) -> int:
    # Deterministic across processes/machines (crc32, not Python's salted hash()); fits in a
    # Postgres bigint with room to spare. One key per strategy so stock_alpha and smid_growth
    # never contend with each other.
    return zlib.crc32(f"{LOCK_NAME_PREFIX}:{strategy}".encode("utf-8"))


def try_acquire(conn: Any, strategy: str) -> bool:
    """Non-blocking. True if the lock is now held on `conn`'s session for this strategy.
    False if another session already holds it -- callers must skip this strategy, not wait."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s) AS ok", (_lock_key(strategy),))
        return bool(cur.fetchone()["ok"])


def release(conn: Any, strategy: str) -> None:
    """Explicit release (also happens automatically when `conn` is closed). Safe to call even
    if this session never held the lock -- Postgres just returns false."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s) AS ok", (_lock_key(strategy),))


def probe_active(conn: Any, strategy: str) -> bool:
    """
    Non-mutating liveness check: True if some OTHER session currently holds the lock for this
    strategy. Implemented as try-then-immediately-release so this never blocks and never holds
    the lock itself -- a status check must never contend with (or count as) the real run.
    """
    acquired_here = try_acquire(conn, strategy)
    if acquired_here:
        release(conn, strategy)
        return False
    return True
