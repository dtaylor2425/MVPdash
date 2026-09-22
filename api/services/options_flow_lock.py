"""
api/services/options_flow_lock.py

Postgres session-level advisory lock, used as a singleton-process lock for the options-flow
backfill (jobs/options_flow_backfill.py) so two ThetaData sessions can never be opened at
once. A session-level advisory lock is held by the Postgres BACKEND for as long as the
connection that acquired it stays open, and Postgres releases it automatically the moment
that connection closes or drops -- for any reason, including the process dying -- so there is
nothing to clean up on crash. No table, no PID file: this is deliberately DB-backed (not a
local PID file) so it works unchanged if this job later runs on Railway.

Callers acquire the lock on a DEDICATED connection held open for the whole run (never used
for queries), separate from the connection(s) used to read/write backfill data -- so a lock
probe or a data-connection hiccup can never be confused with "another process holds the lock."
"""

from __future__ import annotations

import zlib
from typing import Any

LOCK_NAME = "macro_engine_options_flow_backfill"
# Deterministic across processes/machines (crc32, not Python's salted hash()); fits in a
# Postgres bigint with room to spare.
LOCK_KEY = zlib.crc32(LOCK_NAME.encode("utf-8"))


def try_acquire(conn: Any) -> bool:
    """Non-blocking. True if the lock is now held on `conn`'s session. False if another
    session already holds it -- callers must not proceed to create a ThetaData client."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s) AS ok", (LOCK_KEY,))
        return bool(cur.fetchone()["ok"])


def release(conn: Any) -> None:
    """Explicit release (also happens automatically when `conn` is closed). Safe to call
    even if this session never held the lock -- Postgres just returns false."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s) AS ok", (LOCK_KEY,))


def probe_active(conn: Any) -> bool:
    """
    Non-mutating liveness check for --status: True if some OTHER session currently holds the
    lock. Implemented as try-then-immediately-release so this never blocks and never holds
    the lock itself -- a status check must never contend with (or count as) the real backfill.
    """
    acquired_here = try_acquire(conn)
    if acquired_here:
        release(conn)
        return False
    return True
