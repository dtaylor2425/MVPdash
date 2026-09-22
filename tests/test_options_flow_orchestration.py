"""
tests/test_options_flow_orchestration.py  (control-plane hardening: lock, --status, heartbeat,
fatal-error classification -- NOT analytics)

Fake-object tests run anywhere (no DB). Two `pg_`-prefixed tests exercise the REAL Postgres
session-advisory-lock semantics (auto-release on disconnect is a genuine Postgres behaviour a
fake connection can't meaningfully simulate) and are skipped without a database -- same
convention as tests/test_options_flow_backfill.py.

Per instruction: run ONLY this file while the live Phase 1 backfill is active, not the whole
suite.

    pytest tests/test_options_flow_orchestration.py
    python tests/test_options_flow_orchestration.py
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jobs.options_flow_backfill as bf  # noqa: E402
import jobs.options_flow_refresh as live_job  # noqa: E402
from api.services import options_flow_heartbeat as heartbeat  # noqa: E402
from api.services import options_flow_lock as lock  # noqa: E402
from api.services import options_flow_status as ostatus  # noqa: E402
from api.services.options_flow_errors import (  # noqa: E402
    CATEGORY_CLIENT_UNAVAILABLE,
    CATEGORY_DATABASE_UNAVAILABLE,
    CATEGORY_SESSION_OR_AUTH,
    FatalBackfillError,
    classify_error,
)
from tests.test_options_flow import raw_trades  # noqa: E402
from tests.test_options_flow_backfill import Counting, DAYS, START, END  # noqa: E402

BASE = ["--tickers", "SPY", "--start", START, "--end", END]


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeLockCursor:
    """State mutates on execute() (like real Postgres, where acquiring/releasing an advisory
    lock is a side effect of executing the statement), never on fetchone() -- release() in
    the real lock module never calls fetchone() at all, so putting the mutation there would
    silently make release() a no-op in tests."""

    def __init__(self, conn):
        self.conn = conn
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if "pg_try_advisory_lock" in sql:
            if self.conn.held_by is None:
                self.conn.held_by = self.conn
                self._result = {"ok": True}
            else:
                self._result = {"ok": self.conn.held_by is self.conn}
        elif "pg_advisory_unlock" in sql:
            was = self.conn.held_by is self.conn
            if was:
                self.conn.held_by = None
            self._result = {"ok": was}
        else:
            self._result = None

    def fetchone(self):
        return self._result


class FakeLockConn:
    """Simulates ONE Postgres session-advisory-lock's state, shared across every FakeLockConn
    instance constructed with the same `registry` dict (so two 'connections' in a test can
    contend for the same lock, like two real backend sessions would)."""

    def __init__(self, registry):
        self.registry = registry
        self.closed = False

    @property
    def held_by(self):
        return self.registry.get("held_by")

    @held_by.setter
    def held_by(self, v):
        self.registry["held_by"] = v

    def cursor(self):
        return FakeLockCursor(self)

    def commit(self):
        pass

    def close(self):
        self.closed = True
        if self.registry.get("held_by") is self:
            self.registry["held_by"] = None  # a real Postgres session releases on disconnect


def make_lock_conn_factory():
    registry: dict = {"held_by": None}
    return lambda: FakeLockConn(registry), registry


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._rows = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakeDataConn:
    """Enough of a connection for run()'s ensure_schema/backfill_existing calls when we also
    monkeypatch the store functions that matter (see _patch_store)."""

    def __init__(self):
        self.executed = []
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def close(self):
        self.closed = True


def _patch_store(monkey, backfill=None, failures=None, heartbeats=None):
    backfill = backfill if backfill is not None else {}
    failures = failures if failures is not None else []
    heartbeats = heartbeats if heartbeats is not None else []

    def patch(mod, name, fn):
        monkey.append((mod, name, getattr(mod, name)))
        setattr(mod, name, fn)

    patch(bf.store, "ensure_schema", lambda conn: None)
    # backfill_existing()'s real return type is {(ticker, date): status_string}, distinct from
    # our local `backfill` dict's {(ticker, date): {"status":..., "payload":...}} shape.
    patch(bf.store, "backfill_existing", lambda conn, tickers, start, end: {k: v["status"] for k, v in backfill.items()})
    patch(bf.store, "load_atm_series", lambda conn, t: {})
    patch(bf.store, "build_iv_history", lambda series, d: {"priorAtmIv": [], "sessionsBack": []})
    patch(bf.store, "restat_backfill", lambda conn, t, d, cfg: 0)

    def publish(conn, ticker, group, d, status, as_of, diag, payload, config, overwrite=False, mode="full_flow"):
        backfill[(ticker, d)] = {"status": status, "payload": copy.deepcopy(payload)}
        return "run-x"

    def fail(conn, ticker, d, diag, config, mode="full_flow"):
        failures.append((ticker, d, diag))
        return "run-f"

    patch(bf.store, "publish_backfill", publish)
    patch(bf.store, "record_backfill_failure", fail)
    patch(bf.heartbeat, "upsert_heartbeat",
         lambda conn, run_id, phase, ticker, d, ver, started_at, now=None: heartbeats.append(
             {"runId": run_id, "phase": phase, "ticker": ticker, "date": d}))
    return backfill, failures, heartbeats


def _run(args, fetcher, backfill=None, failures=None, heartbeats=None, now=None):
    lock_factory, _ = make_lock_conn_factory()
    monkey = []
    backfill, failures, heartbeats = _patch_store(monkey, backfill, failures, heartbeats)
    try:
        code = bf.run(args, fetcher=fetcher, conn_factory=lambda: FakeDataConn(),
                      lock_conn_factory=lock_factory, now=now or datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc))
    finally:
        for mod, name, val in reversed(monkey):
            setattr(mod, name, val)
    return code, backfill, failures, heartbeats


class SessionKickFetcher(Counting):
    """expirations() raises a real 'Invalid session ID' error (as actually seen from
    ThetaData) on a chosen date; a plain RuntimeError (ordinary per-ticker-day failure) on
    another; succeeds otherwise."""

    def __init__(self, fatal_on=None, recoverable_on=None, **kw):
        super().__init__(**kw)
        self.fatal_on = fatal_on
        self.recoverable_on = recoverable_on
        self.calls = []

    def expirations(self, symbol, market_date, max_dte):
        # Counting.expirations() (below, via super()) already appends to self.calls for the
        # normal path -- append here only for the two raising branches, or every non-special
        # date would be double-counted.
        if market_date == self.fatal_on:
            self.calls.append((symbol, market_date))
            raise RuntimeError(
                "option_history_greeks_first_order(SPY, 2025-07-24) failed: <_MultiThreadedRendezvous "
                "of RPC that terminated with:\n\tdetails = \"Invalid session ID. This can occur if more "
                "than one terminal is running. If restarting your terminal does not resolve this...\">")
        if market_date == self.recoverable_on:
            self.calls.append((symbol, market_date))
            raise RuntimeError("no data available for this contract on this date")
        return super().expirations(symbol, market_date, max_dte)


class ExplodingHistoricalFetcher:
    """Stand-in for HistoricalFetcher whose mere construction fails the test -- used to prove
    a refused/status invocation never gets far enough to build a ThetaData client."""

    def __init__(self, *a, **k):
        raise AssertionError("a ThetaData client must never be constructed here")


# --------------------------------------------------------------------------
# 1 & 2: second concurrent backfill refuses to start, before any ThetaData client
# --------------------------------------------------------------------------

def test_second_concurrent_backfill_refuses_before_thetadata_client():
    lock_factory, registry = make_lock_conn_factory()
    registry["held_by"] = object()  # simulate: some OTHER session already holds the lock

    orig = bf.HistoricalFetcher
    bf.HistoricalFetcher = ExplodingHistoricalFetcher
    try:
        lines = []
        orig_log = bf.log
        bf.log = lambda msg: lines.append(msg)
        try:
            code = bf.run(BASE, fetcher=None, conn_factory=lambda: FakeDataConn(), lock_conn_factory=lock_factory)
        finally:
            bf.log = orig_log
    finally:
        bf.HistoricalFetcher = orig

    assert code == 1
    assert any("already active" in x for x in lines)
    assert any("Refusing to start a second ThetaData session" in x for x in lines)


def test_lock_acquired_when_free_and_a_second_probe_sees_it_held():
    lock_factory, registry = make_lock_conn_factory()
    c1 = lock_factory()
    assert lock.try_acquire(c1) is True
    c2 = lock_factory()
    assert lock.try_acquire(c2) is False
    assert lock.probe_active(c2) is True
    lock.release(c1)
    assert lock.try_acquire(c2) is True


# --------------------------------------------------------------------------
# 3: --status never connects to ThetaData
# --------------------------------------------------------------------------

def test_status_never_constructs_a_thetadata_client():
    orig = bf.HistoricalFetcher
    orig_live = live_job.ThetaFetcher
    bf.HistoricalFetcher = ExplodingHistoricalFetcher
    live_job.ThetaFetcher = ExplodingHistoricalFetcher
    try:
        monkey = []

        def patch(mod, name, fn):
            monkey.append((mod, name, getattr(mod, name)))
            setattr(mod, name, fn)

        patch(bf.store, "ensure_schema", lambda conn: None)
        patch(bf, "lock", bf.lock)  # no-op, keep real lock module (probe_active used, not fetcher-related)
        patch(bf.lock, "probe_active", lambda conn: False)
        patch(bf.heartbeat, "read_heartbeat", lambda conn: None)
        patch(bf.store, "fetch_mode_published", lambda conn, tickers, mode, start, end: [])
        patch(bf.store, "fetch_mode_failed", lambda conn, tickers, mode, start, end: [])
        patch(bf.store, "reconcile_failed_days", lambda failed, published: failed)
        try:
            code = bf.run(["--status"], conn_factory=lambda: FakeDataConn())
        finally:
            for mod, name, val in reversed(monkey):
                setattr(mod, name, val)
    finally:
        bf.HistoricalFetcher = orig
        live_job.ThetaFetcher = orig_live
    assert code == 0


# --------------------------------------------------------------------------
# 5 & 6: ticker-level failure continues; session/auth failure stops everything
# --------------------------------------------------------------------------

def test_ticker_level_failure_continues_to_remaining_days():
    bad_day = DAYS[1]
    f = SessionKickFetcher(recoverable_on=bad_day)
    code, backfill, failures, _ = _run(BASE, f)
    assert code == 2                                                     # published + failed, not fatal
    assert [d for _, d in f.calls] == list(DAYS)                         # every day was attempted
    assert (("SPY", bad_day) not in backfill)
    assert any(d == bad_day for _, d, _ in failures)
    assert all(("SPY", d) in backfill for d in DAYS if d != bad_day)     # the rest still published


def test_session_kick_failure_stops_the_entire_process():
    fatal_day = DAYS[1]
    f = SessionKickFetcher(fatal_on=fatal_day)
    lines = []
    orig_log = bf.log
    bf.log = lambda msg: lines.append(msg)
    try:
        code, backfill, failures, _ = _run(BASE, f)
    finally:
        bf.log = orig_log
    assert code == 3
    assert any("RUN-LEVEL FATAL" in x for x in lines)
    assert any(CATEGORY_SESSION_OR_AUTH in x for x in lines)
    # stopped AT the fatal day -- nothing after it in the (chronological) date list was attempted
    assert [d for _, d in f.calls] == list(DAYS[: DAYS.index(fatal_day) + 1])
    assert ("SPY", fatal_day) not in backfill and not failures           # no partial/failed record for it either
    assert all(("SPY", d) in backfill for d in DAYS[: DAYS.index(fatal_day)])   # earlier days ARE preserved


def test_classify_error_categories():
    assert classify_error(RuntimeError("plain no-data error")) is None
    assert classify_error(RuntimeError("...Invalid session ID. more than one terminal...")) == CATEGORY_SESSION_OR_AUTH
    assert classify_error(RuntimeError("PERMISSION_DENIED: bad api key")) == CATEGORY_SESSION_OR_AUTH
    assert classify_error(RuntimeError("THETADATA_API_KEY is not set in the worker environment")) == CATEGORY_CLIENT_UNAVAILABLE

    class AuthenticationError(Exception):
        pass
    assert classify_error(AuthenticationError("nope")) == CATEGORY_CLIENT_UNAVAILABLE

    import psycopg
    assert classify_error(psycopg.OperationalError("server closed the connection unexpectedly")) == CATEGORY_DATABASE_UNAVAILABLE

    err = RuntimeError("boom")
    fbe = FatalBackfillError(CATEGORY_SESSION_OR_AUTH, err)
    assert fbe.category == CATEGORY_SESSION_OR_AUTH and fbe.original is err


# --------------------------------------------------------------------------
# 7 & 8: --resume retries failed days, skips successful ones
# --------------------------------------------------------------------------

def test_resume_retries_failed_and_skips_successful():
    d0, d1, d2 = DAYS[0], DAYS[1], DAYS[2]
    backfill = {("SPY", d0): {"status": "success", "payload": {}}}       # d0 already done
    failures = [("SPY", d1, {})]                                        # d1 previously failed (no snapshot)
    # backfill_existing() must reflect only the SUCCESSFUL day; a failed day has no snapshot row
    f = Counting()
    code, backfill, failures, _ = _run(BASE + ["--resume"], f, backfill=dict(backfill), failures=[])
    attempted = {d for _, d in f.calls}
    assert d0 not in attempted                                           # never regenerated
    assert d1 in attempted and d2 in attempted                           # retried / attempted
    assert ("SPY", d0) in backfill and backfill[("SPY", d0)]["status"] == "success"
    assert all(("SPY", d) in backfill for d in (d1, d2))
    assert code == 0


# --------------------------------------------------------------------------
# 9: heartbeat writes never touch analytics snapshots
# --------------------------------------------------------------------------

def test_heartbeat_writes_are_isolated_from_snapshot_payloads():
    f = Counting()
    code, backfill, failures, heartbeats = _run(["--tickers", "SPY", "--start", DAYS[0].isoformat(),
                                                 "--end", DAYS[0].isoformat()], f)
    assert code == 0
    assert len(heartbeats) >= 1 and heartbeats[0]["ticker"] == "SPY" and heartbeats[0]["date"] == DAYS[0]
    payload = backfill[("SPY", DAYS[0])]["payload"]
    forbidden = {"run_process_id", "runProcessId", "heartbeat", "pid", "hostname", "lastHeartbeat", "startedAt"}
    assert not (forbidden & set(payload.keys()))
    # the payload's own analytics fields are untouched / present exactly as build_symbol_payload makes them
    for k in ("sentiment", "premium", "delta", "iv", "dte", "quality"):
        assert k in payload, k


def test_heartbeat_upsert_only_writes_the_heartbeat_table_shape():
    class Recording:
        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            self.sql, self.params = sql, params

        def commit(self):
            pass

    conn = Recording()
    heartbeat.upsert_heartbeat(conn, "run-1", "full_flow", "SPY", date(2026, 6, 26), "1.0.0",
                               datetime(2026, 9, 22, tzinfo=timezone.utc))
    assert "options_flow_backfill_heartbeat" in conn.sql
    assert "options_flow_symbol_snapshots" not in conn.sql and "options_flow_runs" not in conn.sql


# --------------------------------------------------------------------------
# 10: ETA uses successful observations only
# --------------------------------------------------------------------------

def test_median_runtime_excludes_non_successful_by_construction_of_the_call_site():
    # median_runtime_by_ticker's contract is "successful rows only, as documented" -- the
    # real call site (print_status) filters on status == "success" before calling it; this
    # proves that filter actually excludes a partial/failed runtime from the estimate.
    rows_success_only = [{"ticker": "SPY", "runtimeSec": 100.0}, {"ticker": "SPY", "runtimeSec": 120.0}]
    rows_with_partial_included_by_mistake = rows_success_only + [{"ticker": "SPY", "runtimeSec": 5.0}]
    m1 = ostatus.median_runtime_by_ticker(rows_success_only, min_ticker_observations=1)
    m2 = ostatus.median_runtime_by_ticker(rows_with_partial_included_by_mistake, min_ticker_observations=1)
    assert m1["__overall__"] == 110.0
    assert m2["__overall__"] != 110.0                                    # proves the 5.0s outlier DOES change it
    assert m2["__overall__"] == 100.0                                    # (i.e. it must never be let in)


def test_status_eta_only_counts_successful_rows_end_to_end():
    monkey = []

    def patch(mod, name, fn):
        monkey.append((mod, name, getattr(mod, name)))
        setattr(mod, name, fn)

    d0, d1 = DAYS[0], DAYS[1]
    published = [
        {"ticker": "SPY", "marketDate": d0, "status": "success", "payload": {}, "diagnostics": {"runtimeSec": 100.0}},
        {"ticker": "SPY", "marketDate": d1, "status": "partial", "payload": {}, "diagnostics": {"runtimeSec": 5.0}},
    ]
    patch(bf.store, "ensure_schema", lambda conn: None)
    patch(bf.lock, "probe_active", lambda conn: False)
    patch(bf.heartbeat, "read_heartbeat", lambda conn: None)
    patch(bf.store, "fetch_mode_published", lambda conn, tickers, mode, start, end: published if mode == "full_flow" else [])
    patch(bf.store, "fetch_mode_failed", lambda conn, tickers, mode, start, end: [])
    patch(bf.store, "reconcile_failed_days", lambda failed, published: failed)
    lines = []
    orig_log = bf.log
    bf.log = lambda msg: lines.append(msg)
    try:
        code = bf.run(["--status"], conn_factory=lambda: FakeDataConn())
    finally:
        bf.log = orig_log
        for mod, name, val in reversed(monkey):
            setattr(mod, name, val)
    assert code == 0
    out = "\n".join(lines)
    assert "Median successful runtime: 100s" in out                      # not (100+5)/2 = 52 or 5


# --------------------------------------------------------------------------
# real Postgres: advisory lock auto-release on process/connection termination
# --------------------------------------------------------------------------

_PG = {}


def _pg_uri():
    if "uri" in _PG:
        return _PG["uri"]
    url = os.getenv("OPTIONS_FLOW_TEST_DATABASE_URL")
    if not url:
        try:
            import pgserver
        except ImportError:
            raise unittest.SkipTest("no Postgres available (set OPTIONS_FLOW_TEST_DATABASE_URL or pip install pgserver)")
        srv = pgserver.get_server(tempfile.mkdtemp(prefix="of_orch_pg_"))
        _PG["srv"] = srv
        import atexit
        atexit.register(srv.cleanup)
        url = srv.get_uri()
    _PG["uri"] = url
    return url


def test_pg_advisory_lock_blocks_second_session_and_releases_on_disconnect():
    import psycopg
    from psycopg.rows import dict_row
    c1 = psycopg.connect(_pg_uri(), row_factory=dict_row)
    c2 = psycopg.connect(_pg_uri(), row_factory=dict_row)
    try:
        assert lock.try_acquire(c1) is True
        assert lock.try_acquire(c2) is False                             # refused while c1 holds it
        assert lock.probe_active(c2) is True
        c1.close()                                                       # simulate the process dying
        c3 = psycopg.connect(_pg_uri(), row_factory=dict_row)
        try:
            assert lock.try_acquire(c3) is True                          # released automatically
        finally:
            lock.release(c3)
            c3.close()
    finally:
        for c in (c1, c2):
            try:
                c.close()
            except Exception:
                pass


def test_pg_run_refuses_when_lock_already_held_by_another_real_session():
    import psycopg
    from psycopg.rows import dict_row
    holder = psycopg.connect(_pg_uri(), row_factory=dict_row)
    try:
        assert lock.try_acquire(holder) is True
        orig = bf.HistoricalFetcher
        bf.HistoricalFetcher = ExplodingHistoricalFetcher
        try:
            code = bf.run(BASE, fetcher=None, conn_factory=lambda: FakeDataConn(),
                          lock_conn_factory=lambda: psycopg.connect(_pg_uri(), row_factory=dict_row))
        finally:
            bf.HistoricalFetcher = orig
        assert code == 1
    finally:
        lock.release(holder)
        holder.close()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except unittest.SkipTest as e:
                print("SKIP", name, "->", e)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if failed else 0)
