"""
tests/test_options_flow_backfill.py  (historical backfill for Macro Options Flow)

Two layers:
  * orchestration tests use an in-memory stand-in for the store, so they run anywhere;
  * "pg_" tests run the REAL SQL against a real Postgres and are skipped when none is
    available. Provide one via OPTIONS_FLOW_TEST_DATABASE_URL, or `pip install pgserver`
    (embedded Postgres; Python <= 3.13 wheels). This also covers the live-snapshot SQL
    (DISTINCT ON selection etc.) that the original test file could only guard textually.

    pytest tests/test_options_flow_backfill.py
    python tests/test_options_flow_backfill.py
"""

from __future__ import annotations

import atexit
import copy
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jobs.options_flow_backfill as bf  # noqa: E402
import jobs.options_flow_refresh as live_job  # noqa: E402
from api.services import options_flow_calendar as cal  # noqa: E402
from api.services import options_flow_metrics as m  # noqa: E402
from api.services import options_flow_store as store  # noqa: E402
from api.services.options_flow_config import load_config  # noqa: E402
from tests.test_options_flow import FakeFetcher, _Conn, env, raw_trades  # noqa: E402

NOW = datetime(2026, 9, 21, 22, 0, tzinfo=timezone.utc)      # Mon after the close: 2026-09-21 is complete
START, END = "2026-09-14", "2026-09-18"
DAYS = list(cal.sessions_between(date(2026, 9, 14), date(2026, 9, 18)))
CFG = load_config()


# --------------------------------------------------------------------------
# fetchers
# --------------------------------------------------------------------------

class Counting(FakeFetcher):
    """FakeFetcher that records which dates were actually fetched (one entry per ticker-date)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def expirations(self, symbol, market_date, max_dte):
        self.calls.append((symbol, market_date))
        return super().expirations(symbol, market_date, max_dte)


class FailOn(Counting):
    def __init__(self, bad_dates, **kw):
        super().__init__(**kw)
        self.bad = set(bad_dates)

    def expirations(self, symbol, market_date, max_dte):
        if market_date in self.bad:
            self.calls.append((symbol, market_date))
            raise RuntimeError("theta exploded on {}".format(market_date))
        return super().expirations(symbol, market_date, max_dte)


class NoTrades(Counting):
    def trade_quote(self, symbol, exp, d):
        return pd.DataFrame()          # what ThetaData NoDataFound turns into


class NoGreeks(Counting):
    def greeks(self, symbol, exp, d):
        return pd.DataFrame()


class Bearish(Counting):
    """Same data but every trade hits the bid -> opposite sentiment (to prove overwrite replaced the row)."""

    def trade_quote(self, symbol, exp, d):
        rows = [(pd.Timestamp(d.year, d.month, d.day, 10, 30 + i, tz="America/New_York"), "CALL", 500.0, 1.00, 200, 1.0, 1.2)
                for i in range(10)]
        return raw_trades(rows, exp.isoformat())


class Interrupting(Counting):
    def __init__(self, at_call, **kw):
        super().__init__(**kw)
        self.at = at_call

    def expirations(self, symbol, market_date, max_dte):
        if len(self.calls) + 1 == self.at:
            raise KeyboardInterrupt
        return super().expirations(symbol, market_date, max_dte)


# --------------------------------------------------------------------------
# in-memory stand-in for the store (orchestration tests only)
# --------------------------------------------------------------------------

class MemStore:
    def __init__(self):
        self.backfill = {}          # (ticker, date) -> {"status", "payload", "run"}
        self.live = {}              # (ticker, date) -> payload   (must never change)
        self.failures = []
        self.series = {}            # ticker -> {date: iv}   preloaded observations (may include future dates)
        self.n = 0

    def install(self, undo):
        def patch(name, fn):
            undo.append((store, name, getattr(store, name)))
            setattr(store, name, fn)

        def existing(conn, tickers, start, end, mode=None):
            return {k: v["status"] for k, v in self.backfill.items()
                    if k[0] in tickers and start <= k[1] <= end and (mode is None or v.get("mode") == mode)}

        def publish(conn, ticker, group, d, status, as_of, diag, payload, config, overwrite=False, mode="full_flow"):
            if (ticker, d) in self.backfill and not overwrite:
                raise store.DuplicateBackfillError("dup")
            self.n += 1
            self.backfill[(ticker, d)] = {"status": status, "payload": copy.deepcopy(payload), "run": self.n,
                                          "diag": diag, "mode": mode}
            return str(self.n)

        def failure(conn, ticker, d, diag, config, mode="full_flow"):
            self.failures.append((ticker, d, diag))
            return "f"

        def series(conn, ticker):
            s = dict(self.series.get(ticker, {}))
            s.update({k[1]: v["payload"]["iv"]["atm"] for k, v in self.backfill.items()
                      if k[0] == ticker and v["payload"]["iv"]["atm"] is not None})
            return s

        patch("ensure_schema", lambda conn: None)
        patch("backfill_existing", existing)
        patch("publish_backfill", publish)
        patch("record_backfill_failure", failure)
        patch("load_atm_series", series)
        patch("restat_backfill", lambda conn, t, d, cfg: 0)


class _AlwaysGrantLockConn:
    """Minimal fake for run()'s singleton-lock connection: always grants/releases, never
    contends with anything -- these tests aren't testing locking, they just need run() to get
    past the (now-mandatory) lock acquisition step without a real Postgres."""

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            pass

        def fetchone(self):
            return {"ok": True}

    def cursor(self):
        return self._Cur()

    def commit(self):
        pass

    def close(self):
        pass


def _run(args, fetcher, mem, now=NOW):
    undo = []
    mem.install(undo)
    try:
        return bf.run(args, fetcher=fetcher, conn_factory=lambda: _Conn(),
                      lock_conn_factory=lambda: _AlwaysGrantLockConn(), now=now)
    finally:
        for obj, name, val in reversed(undo):
            setattr(obj, name, val)


BASE = ["--tickers", "SPY", "--start", START, "--end", END]


# --------------------------------------------------------------------------
# same production code
# --------------------------------------------------------------------------

def test_backfill_uses_the_production_metric_functions():
    assert bf.process_ticker is live_job.process_ticker

    counts = {"payload": 0, "expiration": 0}
    orig_p, orig_e = live_job.build_symbol_payload, live_job.process_expiration

    def p(*a, **k):
        counts["payload"] += 1
        return orig_p(*a, **k)

    def e(*a, **k):
        counts["expiration"] += 1
        return orig_e(*a, **k)

    live_job.build_symbol_payload, live_job.process_expiration = p, e
    try:
        mem = MemStore()
        assert _run(BASE, Counting(), mem) == 0
    finally:
        live_job.build_symbol_payload, live_job.process_expiration = orig_p, orig_e
    assert counts["payload"] == len(DAYS) and counts["expiration"] == 2 * len(DAYS)


def test_backfill_payload_equals_live_payload_apart_from_provenance():
    d = DAYS[0]
    history = store.build_iv_history({}, d)
    fetcher = Counting()
    live_payload, _ = live_job.process_ticker(fetcher, "SPY", "INDEX", d, CFG, history)
    mem = MemStore()
    _run(["--tickers", "SPY", "--start", d.isoformat(), "--end", d.isoformat()], Counting(), mem)
    bfp = copy.deepcopy(mem.backfill[("SPY", d)]["payload"])

    volatile_q = {"thetaFetchedAt", "fetchSeconds", "dataStatus", "missingReasons"}
    for p in (live_payload, bfp):
        p["quality"] = {k: v for k, v in p["quality"].items() if k not in volatile_q}
    assert live_payload.pop("source") == "live" and bfp.pop("source") == "historical_backfill"
    assert live_payload == bfp


def test_historical_fetcher_never_uses_todays_contract_universe():
    calls = {}

    class Spy:
        def option_list_expirations(self, **kw):
            calls["live_list"] = kw
            return pd.DataFrame({"expiration": ["2099-01-01"]})

        def option_list_contracts(self, **kw):
            calls["dated_list"] = kw
            return pd.DataFrame({"expiration": ["2026-09-18", "2026-10-16", "2027-01-15"], "strike": 1.0, "right": "C"})

        def option_history_open_interest(self, **kw):
            calls["oi"] = kw
            return pd.DataFrame()

    f = bf.HistoricalFetcher(CFG, client=Spy())
    today = datetime.now(timezone.utc).date()
    # even for a date equal to *today* (where the live fetcher would use the live expiration list)
    exps = f.expirations("SPY", today, 60)
    assert "live_list" not in calls and "dated_list" in calls
    assert calls["dated_list"]["date"] == today and calls["dated_list"]["max_dte"] is None
    assert all(e >= today for e in exps)                       # filtered client-side, relative to t
    f.open_interest("SPY", date(2026, 9, 18))
    assert calls["oi"]["max_dte"] is None and calls["oi"]["date"] == date(2026, 9, 18)

    live = live_job.ThetaFetcher(CFG, client=Spy())
    calls.clear()
    live.expirations("SPY", today, 60)
    assert "live_list" in calls                                # live behaviour unchanged


# --------------------------------------------------------------------------
# point-in-time: rolling IV
# --------------------------------------------------------------------------

def _prior_series(n=40, end=date(2026, 9, 11), base=0.10, step=0.002):
    sessions = sorted(cal.sessions_before(end + timedelta(days=1), n))
    return {d: base + step * i for i, d in enumerate(sessions)}


def test_iv_history_only_contains_strictly_earlier_sessions():
    d = date(2026, 9, 16)
    series = _prior_series(60, end=date(2026, 9, 11))
    series.update({date(2026, 9, 16): 9.0, date(2026, 9, 17): 9.0, date(2026, 9, 21): 9.0, date(2027, 1, 4): 9.0})
    h = store.build_iv_history(series, d)
    assert all(date.fromisoformat(x) < d for x, _ in h["priorAtmIv"])
    assert all(date.fromisoformat(x) < d for x in h["sessionsBack"])
    assert 9.0 not in [v for _, v in h["priorAtmIv"]]
    assert h["sessionsBack"][0] == "2026-09-15"


def test_percentile_and_changes_ignore_future_observations():
    d = date(2026, 9, 16)
    past = _prior_series(60, end=date(2026, 9, 15))
    poisoned = {**past, date(2026, 9, 16): 9.9, date(2026, 9, 17): 0.001, date(2026, 9, 30): 5.0}
    a = m.iv_history_stats(0.15, store.build_iv_history(past, d))
    b = m.iv_history_stats(0.15, store.build_iv_history(poisoned, d))
    assert a == b
    assert a["percentiles"]["20d"] is not None and a["atmChange1d"] is not None


def test_percentile_for_a_date_moves_only_with_earlier_data():
    d = date(2026, 9, 16)
    past = _prior_series(60, end=date(2026, 9, 15))
    base = m.iv_history_stats(0.15, store.build_iv_history(past, d))["percentiles"]["60d"]
    changed_past = dict(past)
    changed_past[sorted(past)[-1]] = 0.01                  # the session right before d: SHOULD matter
    assert m.iv_history_stats(0.15, store.build_iv_history(changed_past, d))["percentiles"]["60d"] != base
    # ...but a session after d must not:
    changed_future = {**past, date(2026, 9, 17): 0.90}
    assert m.iv_history_stats(0.15, store.build_iv_history(changed_future, d))["percentiles"]["60d"] == base


def test_rolling_windows_1d_5d_20d_and_percentiles_match_manual_calc():
    d = date(2026, 9, 16)
    series = _prior_series(260, end=date(2026, 9, 15), base=0.10, step=0.0005)
    sessions = list(cal.sessions_before(d, 252))                     # newest first
    iv = 0.20
    s = m.iv_history_stats(iv, store.build_iv_history(series, d))
    assert abs(s["atmChange1d"] - (iv - series[sessions[0]])) < 1e-12
    assert abs(s["atmChange5d"] - (iv - series[sessions[4]])) < 1e-12
    assert abs(s["atmChange20d"] - (iv - series[sessions[19]])) < 1e-12
    # a window of size n = the (n-1) prior sessions PLUS today's own observation
    for n in (20, 60, 126, 252):
        vals = np.array([series[x] for x in sessions[: n - 1]] + [iv])
        assert abs(s["percentiles"]["%dd" % n] - float((vals <= iv).mean() * 100)) < 1e-9
        assert s["percentilesObs"]["%dd" % n] == n


def test_thin_or_gappy_history_gives_null_not_a_guess():
    d = date(2026, 9, 16)
    sessions = list(cal.sessions_before(d, 252))
    only_10 = {x: 0.1 for x in sessions[:10]}
    s = m.iv_history_stats(0.2, store.build_iv_history(only_10, d))
    assert s["percentiles"] == {"20d": None, "60d": None, "126d": None, "252d": None}
    # 10 prior obs + today's own observation = 11 total (window minimum for 20d is 15)
    assert s["percentilesObs"]["20d"] == 11 and s["atmChange1d"] is not None and s["atmChange20d"] is None
    # a missing session is a null change, never a bridge to an older observation
    gappy = {x: 0.1 for x in sessions[1:30]}                        # yesterday missing
    g = m.iv_history_stats(0.2, store.build_iv_history(gappy, d))
    assert g["atmChange1d"] is None and g["atmChange5d"] is not None
    # 14 prior sessions + today = 15 total, exactly the 20D minimum; 13 prior + today = 14 is not enough
    assert m.iv_history_stats(0.2, store.build_iv_history({x: 0.1 for x in sessions[:14]}, d))["percentiles"]["20d"] is not None
    assert m.iv_history_stats(0.2, store.build_iv_history({x: 0.1 for x in sessions[:13]}, d))["percentiles"]["20d"] is None
    assert m.iv_history_stats(None, store.build_iv_history(only_10, d))["percentiles"]["20d"] is None


def test_min_observation_thresholds_are_exact():
    d = date(2026, 9, 16)
    sessions = list(cal.sessions_before(d, 252))
    mins = {20: 15, 60: 45, 126: 95, 252: 190}
    for n, need in mins.items():
        below = {x: 0.1 for x in sessions[: need - 2]}       # (need-2) prior + current = need-1 total
        at = {x: 0.1 for x in sessions[: need - 1]}          # (need-1) prior + current = need total
        assert m.iv_history_stats(0.2, store.build_iv_history(below, d))["percentiles"]["%dd" % n] is None
        assert m.iv_history_stats(0.2, store.build_iv_history(at, d))["percentiles"]["%dd" % n] is not None


def test_live_history_format_still_works():
    prior = [("2026-09-18", 0.19)] + [("x%d" % i, 0.18) for i in range(30)]
    s = m.iv_history_stats(0.20, {"priorAtmIv": prior}, {"ivPercentileMinObs": 20})
    assert abs(s["atmChange1d"] - 0.01) < 1e-12 and abs(s["atmChange5d"] - 0.02) < 1e-12
    assert s["percentile"] == 100.0


def test_backfill_run_never_sees_future_iv_end_to_end():
    def run_with(extra):
        mem = MemStore()
        mem.series["SPY"] = {**_prior_series(40, end=date(2026, 9, 11)), **extra}
        assert _run(BASE, Counting(), mem) == 0
        return {k[1]: v["payload"]["iv"] for k, v in mem.backfill.items()}

    clean = run_with({})
    poisoned = run_with({date(2026, 9, 21): 7.7, date(2026, 9, 22): 0.0001, date(2026, 10, 1): 3.3})
    assert clean == poisoned
    first = clean[DAYS[0]]
    assert first["percentiles"]["20d"] is not None and first["atmChange20d"] is not None


# --------------------------------------------------------------------------
# missing data is missing, not zero
# --------------------------------------------------------------------------

def test_no_trade_data_publishes_nothing_and_records_the_reason():
    mem = MemStore()
    code = _run(BASE, NoTrades(), mem)
    assert code == 1 and mem.backfill == {}                # nothing manufactured
    assert len(mem.failures) == len(DAYS)
    assert all(f[2]["missingReasons"] == ["no_trade_data"] for f in mem.failures)


def test_missing_greeks_are_null_and_flagged_partial_not_zero():
    mem = MemStore()
    assert _run(BASE, NoGreeks(), mem) == 0
    assert len(mem.backfill) == len(DAYS)
    for v in mem.backfill.values():
        p = v["payload"]
        assert v["status"] == "partial"
        assert "greeks_unavailable" in p["quality"]["missingReasons"] and p["quality"]["dataStatus"] == "partial"
        assert p["delta"]["netDollar"] is None and p["delta"]["ratio"] is None and p["delta"]["netContracts"] is None
        assert p["iv"]["atm"] is None and p["iv"]["iv30d"] is None and p["iv"]["skew25d30d"] is None
        assert p["iv"]["percentiles"]["20d"] is None and p["iv"]["atmChange1d"] is None
        assert p["spot"] is None
        assert p["sentiment"]["value"] is not None            # what WAS measured is still there
        assert p["premium"]["gross"] > 0


def test_missing_open_interest_is_partial_with_reason():
    mem = MemStore()
    assert _run(BASE, Counting(no_oi=True), mem) == 0
    for v in mem.backfill.values():
        assert v["status"] == "partial" and "open_interest_unavailable" in v["payload"]["quality"]["missingReasons"]
        assert v["payload"]["openInterest"]["contractsToOi"] is None


def test_complete_day_is_success_and_flagged_complete():
    mem = MemStore()
    assert _run(BASE, Counting(), mem) == 0
    for v in mem.backfill.values():
        assert v["status"] == "success" and v["payload"]["quality"]["dataStatus"] == "complete"


def test_assess_payload_rules():
    def p(**over):
        base = {"quality": {"trades": 10, "eligibleTrades": 10, "deltaMatchedPctPremium": 0.95, "warnings": []},
                "iv": {"termStructure": [1], "atm": 0.2}, "delta": {"matchedTrades": 5}}
        for k, v in over.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) else v
        return base
    assert bf.assess_payload(p()) == ("success", [])
    assert bf.assess_payload(p(quality={"trades": 0})) == ("failed", ["no_trade_data"])
    assert bf.assess_payload(p(quality={"eligibleTrades": 0})) == ("partial", ["no_eligible_trades"])
    st, reasons = bf.assess_payload(p(quality={"deltaMatchedPctPremium": 0.5}))
    assert st == "partial" and reasons[0].startswith("low_greek_match")
    st, reasons = bf.assess_payload(p(iv={"termStructure": [], "atm": None}, delta={"matchedTrades": 0}))
    assert st == "partial" and reasons == ["greeks_unavailable"]
    assert bf.assess_payload(p(iv={"atm": None}))[1] == ["atm_iv_unavailable"]


# --------------------------------------------------------------------------
# duplicates / resume / overwrite / failure safety (orchestration)
# --------------------------------------------------------------------------

def test_no_duplicate_ticker_dates_are_generated():
    mem = MemStore()
    f = Counting()
    assert _run(["--tickers", "SPY,spy,SPY", "--start", START, "--end", END], f, mem) == 0
    assert len(mem.backfill) == len(DAYS) and len(f.calls) == len(DAYS)          # de-duplicated tickers
    assert len({(t, d) for t, d in f.calls}) == len(f.calls)

    before = copy.deepcopy(mem.backfill)
    f2 = Counting()
    assert _run(BASE, f2, mem) == 1                                             # existing rows, no flag: refuse
    assert f2.calls == [] and mem.backfill == before
    assert _run(BASE + ["--resume"], f2, mem) == 0                              # resume: nothing to do
    assert f2.calls == [] and mem.backfill == before


def test_resume_restarts_at_the_first_missing_day():
    mem = MemStore()
    assert _run(BASE, Counting(), mem) == 0
    killed = DAYS[2]                                         # simulate: process died before days 3..5 committed
    for d in DAYS[2:]:
        del mem.backfill[("SPY", d)]
    f = Counting()
    assert _run(BASE + ["--resume"], f, mem) == 0
    assert [d for _, d in f.calls] == DAYS[2:]               # first missing day onward, in order
    assert killed in [d for _, d in f.calls] and DAYS[0] not in [d for _, d in f.calls]
    assert len(mem.backfill) == len(DAYS)


def test_resume_skips_success_and_partial_unless_retry_partial():
    mem = MemStore()
    _run(BASE, NoGreeks(), mem)                              # all partial
    f = Counting()
    assert _run(BASE + ["--resume"], f, mem) == 0 and f.calls == []
    f2 = Counting()
    assert _run(BASE + ["--resume", "--retry-partial"], f2, mem) == 0
    assert len(f2.calls) == len(DAYS)
    # partial rows are replaced by the (now complete) rebuild, not duplicated
    assert len(mem.backfill) == len(DAYS) and all(v["status"] == "success" for v in mem.backfill.values())


def test_overwrite_replaces_backfill_rows_and_never_touches_live():
    mem = MemStore()
    live_payload = {"source": "live", "sentiment": {"value": 0.42}}
    mem.live = {("SPY", d): copy.deepcopy(live_payload) for d in DAYS}
    assert _run(BASE, Counting(), mem) == 0
    old = {k: v["payload"]["sentiment"]["value"] for k, v in mem.backfill.items()}
    assert all(x == 1.0 for x in old.values())

    assert _run(BASE + ["--overwrite"], Bearish(), mem) == 0
    new = {k: v["payload"]["sentiment"]["value"] for k, v in mem.backfill.items()}
    assert all(x == -1.0 for x in new.values()) and len(mem.backfill) == len(DAYS)     # replaced, not appended
    assert all(v == live_payload for v in mem.live.values())


def test_failed_day_does_not_destroy_the_previous_snapshot():
    mem = MemStore()
    assert _run(BASE, Counting(), mem) == 0
    before = copy.deepcopy(mem.backfill)
    bad = DAYS[1]
    code = _run(BASE + ["--overwrite"], _BearishFailOn([bad]), mem)
    assert code == 2                                                            # some published, one failed
    assert mem.backfill[("SPY", bad)] == before[("SPY", bad)]                    # untouched, still success
    for d in DAYS:
        if d != bad:
            assert mem.backfill[("SPY", d)]["payload"]["sentiment"]["value"] == -1.0
    assert [f[1] for f in mem.failures] == [bad] and "theta exploded" in mem.failures[0][2]["error"]


class _BearishFailOn(Bearish):
    def __init__(self, bad, **kw):
        super().__init__(**kw)
        self.bad = set(bad)

    def expirations(self, symbol, market_date, max_dte):
        if market_date in self.bad:
            raise RuntimeError("theta exploded on {}".format(market_date))
        return super().expirations(symbol, market_date, max_dte)


def test_failed_day_does_not_stop_the_rest_and_is_retried_on_resume():
    mem = MemStore()
    bad = DAYS[2]
    assert _run(BASE, FailOn([bad]), mem) == 2
    assert sorted(d for _, d in mem.backfill) == [d for d in DAYS if d != bad]
    f = Counting()
    assert _run(BASE + ["--resume"], f, mem) == 0
    assert [d for _, d in f.calls] == [bad] and len(mem.backfill) == len(DAYS)


def test_kill_mid_run_keeps_everything_committed_so_far():
    mem = MemStore()
    code = _run(BASE, Interrupting(at_call=3), mem)
    assert code == 130
    assert sorted(d for _, d in mem.backfill) == DAYS[:2]                        # days 1-2 committed, 3+ absent
    f = Counting()
    assert _run(BASE + ["--resume"], f, mem) == 0
    assert [d for _, d in f.calls] == DAYS[2:] and len(mem.backfill) == len(DAYS)


def test_backfilled_days_are_chronological_and_history_builds_up():
    mem = MemStore()
    _run(BASE, Counting(), mem)
    changes = [mem.backfill[("SPY", d)]["payload"]["iv"]["percentileObs"] for d in DAYS]
    assert changes == [0, 1, 2, 3, 4]                                            # each day sees only earlier days


def test_dry_run_writes_nothing_and_needs_no_database():
    # dry-run still opens a real ThetaData session, so it DOES need the lock connection (guards
    # against two concurrent dry-runs the same way any other invocation is guarded); it must
    # never open the DATA connection, since nothing is written.
    def boom():
        raise AssertionError("dry run opened the data DB connection")
    assert bf.run(["--tickers", "SPY", "--days", "3", "--dry-run"], fetcher=Counting(), conn_factory=boom,
                  lock_conn_factory=lambda: _AlwaysGrantLockConn(), now=NOW) == 0


# --------------------------------------------------------------------------
# scope / CLI planning
# --------------------------------------------------------------------------

def test_dates_default_to_completed_sessions_only():
    args = bf.build_parser().parse_args(["--days", "5"])
    dates, _ = bf.resolve_dates(args, NOW)
    assert dates == sorted(dates) and len(dates) == 5 and dates[-1] == date(2026, 9, 21)
    early = datetime(2026, 9, 21, 19, 0, tzinfo=timezone.utc)                    # 15:00 ET, session still open
    assert bf.resolve_dates(args, early)[0][-1] == date(2026, 9, 18)
    args = bf.build_parser().parse_args(["--start", "2026-09-14", "--end", "2026-12-31"])
    d2, notes = bf.resolve_dates(args, NOW)
    assert d2[-1] == date(2026, 9, 21) and any("clamped" in n for n in notes)
    for bad in (["--days", "5", "--start", "2026-09-01"], ["--days", "0"]):
        try:
            bf.resolve_dates(bf.build_parser().parse_args(bad), NOW)
            raise AssertionError(bad)
        except bf.ScopeError:
            pass


def test_default_universe_is_the_staged_phase_one():
    uni = bf.load_universe()
    t = bf.resolve_tickers(bf.build_parser().parse_args([]), uni)
    assert t == ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
    assert len(bf.resolve_tickers(bf.build_parser().parse_args(["--tickers", "all"]), uni)) == 22
    assert bf.resolve_tickers(bf.build_parser().parse_args(["--ticker", "spy"]), uni) == ["SPY"]
    for bad in (["--ticker", "SPY", "--tickers", "QQQ"], ["--tickers", "AAPL"]):
        try:
            bf.resolve_tickers(bf.build_parser().parse_args(bad), uni)
            raise AssertionError(bad)
        except bf.ScopeError:
            pass


def test_scope_guards_force_staging():
    bf.check_scope(6, 60, False)
    bf.check_scope(22, 60, False)                       # full universe, 60 days: allowed
    bf.check_scope(1, 252, False)
    for n_t, n_d in ((22, 252), (22, 127), (6, 252)):
        try:
            bf.check_scope(n_t, n_d, False)
            raise AssertionError((n_t, n_d))
        except bf.ScopeError:
            pass
        bf.check_scope(n_t, n_d, True)                  # explicit override works


def test_resume_and_overwrite_are_exclusive_and_cli_errors_are_nonzero():
    assert bf.run(BASE + ["--resume", "--overwrite"], fetcher=Counting(), conn_factory=lambda: _Conn(), now=NOW) == 1
    assert bf.run(["--tickers", "AAPL", "--days", "3"], fetcher=Counting(), conn_factory=lambda: _Conn(), now=NOW) == 1
    assert bf.run(["--tickers", "all", "--days", "252"], fetcher=Counting(), conn_factory=lambda: _Conn(), now=NOW) == 1


def test_build_diagnostics_rejects_bad_mode_and_branches_correctly():
    try:
        bf.build_diagnostics("SPY", DAYS[0], "success", [], None, None, 1.0, mode="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    full = {"quality": {"classifiedPctTrades": 0.9, "classifiedPctPremium": 0.9, "trades": 10, "eligibleTrades": 9,
                        "deltaMatchedPctPremium": 1.0, "oiMatchedPctPremium": 1.0, "contractsTraded": 1, "expirations": 1,
                        "thetaRequests": 1, "warnings": []}, "premium": {"gross": 1.0}}
    d1 = bf.build_diagnostics("SPY", DAYS[0], "success", [], full, None, 1.0, mode=bf.MODE_FULL_FLOW)
    assert d1["mode"] == "full_flow" and d1["classifiedPctTrades"] == 0.9

    warm = {"quality": {"expirations": 1, "contracts": 10, "thetaRequests": 1, "warnings": []},
            "iv": {"atm": 0.2, "skew25d30d": 0.01}}
    d2 = bf.build_diagnostics("SPY", DAYS[0], "success", [], warm, None, 1.0, mode=bf.MODE_IV_WARMUP)
    assert d2["mode"] == "iv_warmup" and d2["atmIv"] == 0.2 and "classifiedPctTrades" not in d2


def test_iv_warmup_mode_never_fetches_trades_or_open_interest():
    class Tracking(Counting):
        def trade_quote(self, *a, **k):
            raise AssertionError("iv-warmup must never fetch trade_quote")

        def open_interest(self, *a, **k):
            raise AssertionError("iv-warmup must never fetch open_interest")

    mem = MemStore()
    code = _run(BASE + ["--mode", "iv-warmup"], Tracking(), mem)
    assert code == 0 and len(mem.backfill) == len(DAYS)
    for v in mem.backfill.values():
        p = v["payload"]
        assert v["mode"] == "iv_warmup" and p["mode"] == "iv_warmup"
        assert p["iv"]["atm"] is not None
        for k in ("sentiment", "premium", "delta", "dte", "aggression", "openInterest"):
            assert p[k] is None, k
        assert p["intraday"] == [] and p["largeTrades"] == []
        assert p["methodologyVersion"] == bf.METHODOLOGY_VERSION == "1.0.0"
        assert v["status"] == "success"


def test_iv_warmup_no_greek_data_is_failed_not_published():
    mem = MemStore()
    code = _run(BASE + ["--mode", "iv-warmup"], NoGreeks(), mem)
    assert code == 1 and mem.backfill == {}
    assert all(f[2]["missingReasons"] == ["no_greek_data"] for f in mem.failures)
    assert all(f[2]["mode"] == "iv_warmup" for f in mem.failures)


def test_iv_warmup_then_full_flow_upgrade_needs_overwrite():
    d = DAYS[0]
    single = ["--tickers", "SPY", "--start", d.isoformat(), "--end", d.isoformat()]
    mem2 = MemStore()
    assert _run(single + ["--mode", "iv-warmup"], Counting(), mem2) == 0
    assert mem2.backfill[("SPY", d)]["mode"] == "iv_warmup"
    # backfill_existing() is mode-filtered (see test_pg_backfill_existing_does_not_leak_across_modes):
    # asking for full_flow correctly sees nothing published for full_flow yet, so this is NOT refused
    # upfront -- it proceeds, computes the real full_flow result, then the physical (ticker, date)
    # slot's still occupied by the iv_warmup row, so the write is safely skipped (DuplicateBackfillError),
    # exit 0, and the iv_warmup data is left untouched, exactly like any other "already published" skip.
    code = _run(single, Counting(), mem2)
    assert code == 0 and mem2.backfill[("SPY", d)]["mode"] == "iv_warmup"   # untouched without --overwrite
    assert _run(single + ["--overwrite"], Counting(), mem2) == 0
    assert mem2.backfill[("SPY", d)]["mode"] == "full_flow"
    assert mem2.backfill[("SPY", d)]["payload"]["sentiment"] is not None


def test_resume_with_nothing_to_do_never_constructs_a_theta_client():
    mem = MemStore()
    assert _run(BASE, Counting(), mem) == 0
    assert _run(BASE + ["--resume"], fetcher=None, mem=mem) == 0     # fetcher=None: would crash if actually used


def test_estimate_requests_iv_warmup_is_cheaper_than_full_flow():
    full = bf.estimate_requests("SPY", 60, 10, bf.MODE_FULL_FLOW)
    warm = bf.estimate_requests("SPY", 60, 10, bf.MODE_IV_WARMUP)
    assert warm < full
    assert bf.estimate_requests("SPY", 60, 1, bf.MODE_IV_WARMUP) == bf.expected_expirations("SPY", 60) + 1


def test_qa_report_stage_timing_is_actually_populated():
    """Regression guard: a run must record non-trivial per-stage timing for both modes,
    not just print an empty 'stage timing profile' header (caught once by eyeballing a
    real run's full log -- this pins it so a refactor can't silently drop the wiring)."""
    for extra, expect_stages in ((["--mode", "iv-warmup"], {"contract_lookup", "greeks_fetch", "normalize",
                                                             "iv_snapshot", "aggregate"}),
                                 ([], {"contract_lookup", "trade_quote_fetch", "greeks_fetch",
                                       "normalize", "classify", "greek_asof_join", "aggregate"})):
        lines = []
        orig = bf.log
        bf.log = lambda msg: lines.append(msg)
        try:
            mem = MemStore()
            assert _run(["--tickers", "SPY", "--start", START, "--end", START] + extra, Counting(), mem) == 0
        finally:
            bf.log = orig
        stage_lines = [x for x in lines if x.startswith("    ") and "median=" in x]
        seen = {x.split()[0] for x in stage_lines}
        # postgres_write is timed by run() itself around the store call, so it's present
        # even against the in-memory MemStore -- only open_interest_fetch is mode-specific.
        assert (expect_stages | {"postgres_write"}) <= seen, (extra, seen)
        if "--mode" in extra:
            assert "trade_quote_fetch" not in seen and "open_interest_fetch" not in seen


def test_median_and_fmt_bytes():
    assert bf._median([]) is None
    assert bf._median([3.0]) == 3.0
    assert bf._median([1.0, 2.0, 3.0]) == 2.0
    assert bf._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert bf.fmt_bytes(None) == "n/a"
    assert bf.fmt_bytes(0) == "+0B"
    assert bf.fmt_bytes(-5 * (1 << 20)) == "-5.0MB"


def test_spy_60_day_request_estimate():
    # measured on real ThetaData: ~34 requests per SPY day at max_dte=60 (~16 expirations)
    assert bf.expected_expirations("SPY", 60) == 16
    assert bf.estimate_requests("SPY", 60, 60) == 60 * 34 == 2040
    assert bf.estimate_requests("GLD", 60, 60) < bf.estimate_requests("SPY", 60, 60)


def test_open_interest_uses_per_expiration_requests_only_for_the_current_day():
    """ThetaData rejects wildcard-expiration OI for today ("Cannot fetch current-day data without specifying an expiration")."""
    from api.services.options_flow_calendar import NY
    calls = []

    class Client:
        def option_history_open_interest(self, **kw):
            calls.append(kw)
            return pd.DataFrame({"expiration": [str(kw["expiration"])], "strike": [500.0], "right": ["CALL"], "open_interest": [10]})

    f = live_job.ThetaFetcher(CFG, client=Client())
    today = datetime.now(NY).date()
    exps = [today, today + timedelta(days=7)]
    df = f.open_interest("SPY", today, exps)
    assert [c["expiration"] for c in calls] == exps and len(df) == 2       # one request per expiration
    assert all(c["max_dte"] is None for c in calls)

    calls.clear()
    f.open_interest("SPY", today - timedelta(days=3), exps)               # past date: single wildcard request
    assert [c["expiration"] for c in calls] == ["*"] and calls[0]["max_dte"] == 60
    calls.clear()
    bf.HistoricalFetcher(CFG, client=Client()).open_interest("SPY", today - timedelta(days=3), exps)
    assert [c["expiration"] for c in calls] == ["*"] and calls[0]["max_dte"] is None


def test_progress_line_format(capfd=None):
    mem = MemStore()
    lines = []
    orig = bf.log
    bf.log = lambda msg: lines.append(msg)
    try:
        _run(["--tickers", "SPY", "--start", START, "--end", START], Counting(), mem)
    finally:
        bf.log = orig
    row = [x for x in lines if x.startswith("[SPY 2026-09-14]")][0]
    import re
    assert re.match(r"\[SPY 2026-09-14\] trades=\d[\d,]* coverage=\d+\.\d% sentiment=[+-]\d\.\d\d delta=[+-]\$[\d.]+[KMBT]? \d+\.\ds$", row), row
    assert "  1/1 ticker-days complete" in lines


# --------------------------------------------------------------------------
# real Postgres
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
        srv = pgserver.get_server(tempfile.mkdtemp(prefix="of_pg_"))
        _PG["srv"] = srv
        atexit.register(srv.cleanup)
        url = srv.get_uri()
    _PG["uri"] = url
    return url


_OPEN = []


def pg_conn(fresh=True):
    import psycopg
    from psycopg.rows import dict_row
    if fresh:                                    # a previous test's open transaction would block DROP TABLE
        while _OPEN:
            try:
                _OPEN.pop().close()
            except Exception:
                pass
    conn = psycopg.connect(_pg_uri(), row_factory=dict_row, autocommit=False)
    _OPEN.append(conn)
    if fresh:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS options_flow_symbol_snapshots, options_flow_runs CASCADE")
        conn.commit()
        store.ensure_schema(conn)
    return conn


def _payload(ticker, d, sentiment=0.1, atm=0.2, source="live"):
    return {"source": source, "ticker": ticker, "marketDate": d.isoformat(),
            "sentiment": {"value": sentiment, "label": "NEUTRAL"},
            "iv": {"atm": atm, "percentile": None, "atmChange1d": None}, "premium": {}, "delta": {}, "dte": {}}


def _live_publish(conn, ticker, d, as_of, sentiment=0.1, atm=0.2, status="success"):
    rid = store.create_run(conn, d, {})
    store.finalize_run(conn, rid, status, as_of, {}, [
        {"ticker": ticker, "group": "INDEX", "as_of": as_of, "payload": _payload(ticker, d, sentiment, atm)}])
    return rid


def _count(conn, where="true"):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM options_flow_symbol_snapshots WHERE " + where)
        return cur.fetchone()["n"]


D1, D2 = date(2026, 9, 17), date(2026, 9, 18)


def _t(d, h=20, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=timezone.utc)


def test_pg_migration_is_idempotent_and_existing_rows_become_live():
    conn = pg_conn(fresh=False)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS options_flow_symbol_snapshots, options_flow_runs CASCADE")
        cur.execute(store.DDL_PATHS[0].read_text(encoding="utf-8"))            # only migration 005 -> old schema
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("INSERT INTO options_flow_runs (market_date, as_of_timestamp, status) VALUES (%s, now(), 'success') RETURNING id", (D1,))
        old_run = cur.fetchone()["id"]
        cur.execute("""INSERT INTO options_flow_symbol_snapshots (run_id, ticker, group_name, market_date, as_of_timestamp, payload)
                       VALUES (%s, 'SPY', 'INDEX', %s, now(), '{}')""", (old_run, D1))
    conn.commit()
    store.ensure_schema(conn)
    store.ensure_schema(conn)                                                  # idempotent
    with conn.cursor() as cur:
        cur.execute("SELECT source, mode FROM options_flow_symbol_snapshots")
        assert [tuple(r.values()) for r in cur.fetchall()] == [("live", "full_flow")]
        cur.execute("SELECT source, mode FROM options_flow_runs")
        assert [tuple(r.values()) for r in cur.fetchall()] == [("live", "full_flow")]
        try:
            cur.execute("UPDATE options_flow_runs SET source = 'bogus'")
            raise AssertionError("source CHECK constraint missing")
        except Exception as e:
            assert e.__class__.__name__ == "CheckViolation"
        conn.rollback()
        try:
            with conn.cursor() as cur2:
                cur2.execute("UPDATE options_flow_runs SET mode = 'bogus'")
            raise AssertionError("mode CHECK constraint missing")
        except Exception as e:
            assert e.__class__.__name__ == "CheckViolation"
    conn.rollback()


def test_pg_methodology_version_is_retroactive_on_old_rows_and_explicit_going_forward():
    """A row inserted before 008 (no methodology_version in the INSERT at all) must retroactively
    read as the CURRENT version once 008 is applied -- because nothing about the calculations
    changed between when that row was written and now, so '1.0.0' is factually correct for it,
    not merely a placeholder. A row inserted with the current code must carry the same value
    explicitly (not just via the column default), so a future version bump shows up correctly."""
    conn = pg_conn(fresh=False)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS options_flow_symbol_snapshots, options_flow_runs CASCADE")
        for path in store.DDL_PATHS[:-1]:                      # every migration EXCEPT 008
            cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO options_flow_runs (market_date, as_of_timestamp, status, source, mode)
               VALUES (%s, now(), 'success', 'historical_backfill', 'full_flow') RETURNING id""", (D1,))
        old_run = cur.fetchone()["id"]
        cur.execute(
            """INSERT INTO options_flow_symbol_snapshots
                   (run_id, ticker, group_name, market_date, as_of_timestamp, payload, source, mode)
               VALUES (%s, 'SPY', 'INDEX', %s, now(), '{}', 'historical_backfill', 'full_flow')""",
            (old_run, D1))
    conn.commit()

    store.ensure_schema(conn)                                   # now applies 008 too
    with conn.cursor() as cur:
        cur.execute("SELECT methodology_version FROM options_flow_symbol_snapshots WHERE run_id = %s", (old_run,))
        assert cur.fetchone()["methodology_version"] == "1.0.0"
        cur.execute("SELECT methodology_version FROM options_flow_runs WHERE id = %s", (old_run,))
        assert cur.fetchone()["methodology_version"] == "1.0.0"

    store.publish_backfill(conn, "QQQ", "INDEX", D2, "success", _t(D2), {},
                           _payload("QQQ", D2, source="historical_backfill"), {})
    with conn.cursor() as cur:
        cur.execute("SELECT r.methodology_version AS run_v, s.methodology_version AS snap_v "
                    "FROM options_flow_symbol_snapshots s JOIN options_flow_runs r ON r.id = s.run_id "
                    "WHERE s.ticker = 'QQQ'")
        row = cur.fetchone()
        assert row["run_v"] == row["snap_v"] == store.METHODOLOGY_VERSION == "1.0.0"


def test_pg_iv_warmup_mode_stored_and_upgradeable_to_full_flow():
    conn = pg_conn()
    args = dict(ticker="SPY", group="INDEX", market_date=D1, status="success", as_of=_t(D1), diagnostics={}, config={})
    store.publish_backfill(conn, payload=_payload("SPY", D1, None, 0.15, "historical_backfill"), mode="iv_warmup", **args)
    with conn.cursor() as cur:
        cur.execute("SELECT source, mode, sentiment FROM options_flow_symbol_snapshots")
        row = cur.fetchone()
        assert row["source"] == "historical_backfill" and row["mode"] == "iv_warmup" and row["sentiment"] is None
        cur.execute("SELECT mode FROM options_flow_runs")
        assert cur.fetchone()["mode"] == "iv_warmup"

    store.publish_backfill(conn, payload=_payload("SPY", D1, 0.4, 0.15, "historical_backfill"),
                           mode="full_flow", overwrite=True, **args)
    with conn.cursor() as cur:
        cur.execute("SELECT mode, sentiment FROM options_flow_symbol_snapshots")
        row = cur.fetchone()
        assert row["mode"] == "full_flow" and row["sentiment"] == 0.4

    try:
        store.publish_backfill(conn, payload=_payload("SPY", D1, 0.4), mode="bogus", **{**args, "overwrite": True})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pg_backfill_existing_does_not_leak_across_modes():
    """Regression: an iv_warmup row for a date must never make backfill_existing()/--resume
    treat that date as 'already done' for full_flow (or vice versa) -- found in Phase 1 when
    an ad-hoc iv-warmup smoke-test date fell inside the full-flow window and was silently,
    permanently skipped by every full-flow --resume afterward, with no error and no record."""
    conn = pg_conn()
    store.publish_backfill(conn, "SPY", "INDEX", D1, "success", _t(D1), {},
                           _payload("SPY", D1, None, 0.15, "historical_backfill"), {}, mode="iv_warmup")
    only_iv = store.backfill_existing(conn, ["SPY"], D1, D1, mode="iv_warmup")
    only_full = store.backfill_existing(conn, ["SPY"], D1, D1, mode="full_flow")
    unfiltered = store.backfill_existing(conn, ["SPY"], D1, D1)   # mode=None: legacy cross-mode view
    assert only_iv == {("SPY", D1): "success"}
    assert only_full == {}                                        # <-- the bug: this used to equal only_iv
    assert unfiltered == {("SPY", D1): "success"}


def test_pg_full_iv_warmup_backfill_run_writes_mode_column():
    conn0 = pg_conn()
    conn0.close()

    def go(args):
        return bf.run(args, fetcher=Counting(), conn_factory=lambda: pg_conn(fresh=False),
                      lock_conn_factory=lambda: pg_conn(fresh=False), now=NOW)

    assert go(BASE + ["--mode", "iv-warmup"]) == 0
    conn = pg_conn(fresh=False)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT mode FROM options_flow_symbol_snapshots WHERE source = 'historical_backfill'")
            assert [r["mode"] for r in cur.fetchall()] == ["iv_warmup"]
            cur.execute("SELECT payload->'sentiment' AS s, payload->'iv'->'atm' AS atm FROM options_flow_symbol_snapshots LIMIT 1")
            row = cur.fetchone()
            assert row["s"] is None and row["atm"] is not None
        latest = store.fetch_latest(conn, {"INDEX": ["SPY"]}, {})
        assert latest["tickers"][0]["sentiment"] is None and latest["tickers"][0]["iv"]["atm"] is not None
    finally:
        conn.close()


def test_pg_publish_backfill_duplicate_and_overwrite():
    conn = pg_conn()
    args = dict(ticker="SPY", group="INDEX", market_date=D1, status="success", as_of=_t(D1),
                diagnostics={"a": 1}, config={})
    store.publish_backfill(conn, payload=_payload("SPY", D1, 0.5, source="historical_backfill"), **args)
    try:
        store.publish_backfill(conn, payload=_payload("SPY", D1, 0.9, source="historical_backfill"), **args)
        raise AssertionError("duplicate backfill row accepted")
    except store.DuplicateBackfillError:
        pass
    assert _count(conn, "source = 'historical_backfill'") == 1                 # unchanged, connection still usable
    with conn.cursor() as cur:
        cur.execute("SELECT (payload->'sentiment'->>'value')::float AS v FROM options_flow_symbol_snapshots")
        assert cur.fetchone()["v"] == 0.5

    store.publish_backfill(conn, payload=_payload("SPY", D1, 0.9, source="historical_backfill"), overwrite=True, **args)
    assert _count(conn) == 1
    with conn.cursor() as cur:
        cur.execute("SELECT (payload->'sentiment'->>'value')::float AS v FROM options_flow_symbol_snapshots")
        assert cur.fetchone()["v"] == 0.9
        cur.execute("SELECT count(*) AS n FROM options_flow_runs WHERE source = 'historical_backfill' AND status <> 'failed'")
        assert cur.fetchone()["n"] == 1                                        # old run removed with its snapshot
    # DB-level guard even if application code were bypassed
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM options_flow_runs LIMIT 1")
            rid = cur.fetchone()["id"]
            cur.execute("""INSERT INTO options_flow_symbol_snapshots (run_id, ticker, group_name, market_date, as_of_timestamp, payload, source)
                           VALUES (%s, 'SPY', 'INDEX', %s, now(), '{}', 'historical_backfill')""", (rid, D1))
        raise AssertionError("unique index missing")
    except Exception as e:
        assert e.__class__.__name__ == "UniqueViolation"
    conn.rollback()


def test_pg_overwrite_is_atomic_failed_replacement_keeps_old_snapshot():
    conn = pg_conn()
    good = dict(ticker="SPY", group="INDEX", market_date=D1, status="success", as_of=_t(D1), diagnostics={}, config={})
    store.publish_backfill(conn, payload=_payload("SPY", D1, 0.5, source="historical_backfill"), **good)
    try:
        store.publish_backfill(conn, payload=_payload("SPY", D1, 0.9), overwrite=True, **{**good, "group": None})  # NOT NULL fails
        raise AssertionError("expected DB error")
    except store.DuplicateBackfillError:
        raise AssertionError("wrong error class")
    except Exception as e:
        assert e.__class__.__name__ == "NotNullViolation"
    with conn.cursor() as cur:
        cur.execute("SELECT (payload->'sentiment'->>'value')::float AS v FROM options_flow_symbol_snapshots WHERE source='historical_backfill'")
        rows = cur.fetchall()
    assert len(rows) == 1 and rows[0]["v"] == 0.5                              # old snapshot survived the rollback


def test_pg_failure_record_publishes_nothing_and_deletes_nothing():
    conn = pg_conn()
    store.publish_backfill(conn, "SPY", "INDEX", D1, "success", _t(D1), {}, _payload("SPY", D1, source="historical_backfill"), {})
    store.record_backfill_failure(conn, "SPY", D1, {"missingReasons": ["no_trade_data"]}, {})
    assert _count(conn) == 1
    assert store.backfill_existing(conn, ["SPY"], D1, D1) == {("SPY", D1): "success"}      # failed run isn't "published"
    latest = store.fetch_latest(conn, {"INDEX": ["SPY"]}, {})
    assert latest["tickers"][0]["ticker"] == "SPY"


def test_pg_live_and_backfill_coexist_and_live_wins_latest():
    conn = pg_conn()
    _live_publish(conn, "SPY", D1, _t(D1, 19, 0), sentiment=0.30, atm=0.21)                # live, same date
    store.publish_backfill(conn, "SPY", "INDEX", D1, "success", _t(D1, 20, 15), {},
                           _payload("SPY", D1, -0.9, 0.25, "historical_backfill"), {})     # later as_of on purpose
    store.publish_backfill(conn, "SPY", "INDEX", D2, "success", _t(D2), {}, _payload("SPY", D2, -0.5, source="historical_backfill"), {})
    assert _count(conn, "source = 'live'") == 1 and _count(conn, "source = 'historical_backfill'") == 2

    # same date: the live row is served even though the backfill row has a later as_of
    t = store.fetch_ticker(conn, "SPY", {"INDEX": ["SPY"]}, {})["ticker"]
    assert t["marketDate"] == D2.isoformat() and t["sentiment"]["value"] == -0.5          # newer market date wins first
    store.publish_backfill(conn, "QQQ", "INDEX", D1, "success", _t(D1, 20, 15), {}, _payload("QQQ", D1, -0.7, source="historical_backfill"), {})
    _live_publish(conn, "QQQ", D1, _t(D1, 19, 0), sentiment=0.4)
    q = store.fetch_ticker(conn, "QQQ", {"INDEX": ["QQQ"]}, {})["ticker"]
    assert q["sentiment"]["value"] == 0.4 and q["source"] == "live"

    # the live worker's status ignores the (many) backfill runs
    st = store.fetch_status(conn, {"INDEX": ["SPY"]}, {})
    assert st["lastRun"]["status"] == "success" and st["lastRun"]["marketDate"] == D1.isoformat()
    # history: one point per date, no duplicates from coexisting rows
    h = store.fetch_history(conn, "SPY", 10)
    assert [p["date"] for p in h["series"]] == [D1.isoformat(), D2.isoformat()]


def test_pg_live_latest_selection_and_failed_runs_never_shadow():
    conn = pg_conn()
    _live_publish(conn, "SPY", D1, _t(D1, 19, 0), sentiment=0.1)
    _live_publish(conn, "QQQ", D1, _t(D1, 19, 0), sentiment=0.2)
    r2 = _live_publish(conn, "SPY", D2, _t(D2, 15, 0), sentiment=0.3)              # newer date, SPY only
    failed = store.create_run(conn, D2, {})
    store.fail_run(conn, failed, {"error": "boom"})                                # newest run of all is a failure
    uni = {"INDEX": ["SPY", "QQQ", "XXX"]}
    out = store.fetch_latest(conn, uni, {})
    by = {t["ticker"]: t for t in out["tickers"]}
    assert by["SPY"]["sentiment"]["value"] == 0.3 and by["QQQ"]["sentiment"]["value"] == 0.2
    assert by["QQQ"]["carriedForward"] is True and by["SPY"]["carriedForward"] is False
    assert out["run"]["id"] == r2 and out["missing"] == ["XXX"]
    assert "intraday" not in by["SPY"] and "largeTrades" not in by["SPY"]         # stripped in SQL
    assert store.fetch_status(conn, uni, {})["lastRun"]["status"] == "failed"      # status still reports the failure


def test_pg_iv_series_and_restat_are_point_in_time():
    conn = pg_conn()
    sessions = list(sorted(cal.sessions_before(date(2026, 9, 19), 40)))            # 40 sessions up to 2026-09-18
    target = sessions[-1]
    for i, d in enumerate(sessions):
        p = _payload("SPY", d, 0.0, 0.10 + 0.002 * i, "historical_backfill")
        store.publish_backfill(conn, "SPY", "INDEX", d, "success", _t(d), {}, p, {})
    series = store.load_atm_series(conn, "SPY")
    assert len(series) == 40

    n = store.restat_backfill(conn, "SPY", target, CFG)
    assert n == 1
    with conn.cursor() as cur:
        cur.execute("SELECT payload->'iv' AS iv FROM options_flow_symbol_snapshots WHERE market_date = %s", (target,))
        iv = cur.fetchone()["iv"]
    prior = list(sorted(cal.sessions_before(target, 20)))
    manual = float(np.mean([series[x] <= series[target] for x in prior]) * 100)
    assert abs(iv["percentiles"]["20d"] - manual) < 1e-6 and iv["percentiles"]["20d"] == 100.0
    assert abs(iv["atmChange1d"] - (series[target] - series[sessions[-2]])) < 1e-9

    # a LATER date with a wild IV must not change the earlier row's stats when restat runs again
    later = date(2026, 9, 21)
    store.publish_backfill(conn, "SPY", "INDEX", later, "success", _t(later), {}, _payload("SPY", later, 0.0, 9.0, "historical_backfill"), {})
    store.restat_backfill(conn, "SPY", target, CFG)
    with conn.cursor() as cur:
        cur.execute("SELECT payload->'iv' AS iv FROM options_flow_symbol_snapshots WHERE market_date = %s", (target,))
        assert cur.fetchone()["iv"] == iv
        cur.execute("SELECT payload->'iv' AS iv FROM options_flow_symbol_snapshots WHERE market_date = %s", (later,))
        assert abs(cur.fetchone()["iv"]["atmChange1d"] - (9.0 - series[target])) < 1e-9
    # restat only ever touches backfill rows
    _live_publish(conn, "SPY", date(2026, 9, 22), _t(date(2026, 9, 22)), atm=0.5)
    before = _count(conn, "source = 'live'")
    store.restat_backfill(conn, "SPY", sessions[0], CFG)
    assert before == _count(conn, "source = 'live'")
    with conn.cursor() as cur:
        cur.execute("SELECT payload->'iv'->'percentile' AS p, payload->'iv'->'atmChange1d' AS c FROM options_flow_symbol_snapshots WHERE source = 'live'")
        assert cur.fetchone() == {"p": None, "c": None}


def test_pg_full_backfill_run_resume_overwrite_and_live_untouched():
    conn0 = pg_conn()
    live_rid = _live_publish(conn0, "SPY", DAYS[1], _t(DAYS[1], 19, 30), sentiment=0.77)
    conn0.close()

    def go(args, fetcher):
        return bf.run(args, fetcher=fetcher, conn_factory=lambda: pg_conn(fresh=False),
                      lock_conn_factory=lambda: pg_conn(fresh=False), now=NOW)

    f1 = Counting()
    assert go(BASE, f1) == 0 and len(f1.calls) == len(DAYS)
    f2 = Counting()
    assert go(BASE, f2) == 1 and f2.calls == []                                    # refuses without --resume/--overwrite
    assert go(BASE + ["--resume"], f2) == 0 and f2.calls == []
    assert go(BASE + ["--overwrite"], Bearish()) == 0
    bad = FailOn([DAYS[3]])
    assert go(BASE + ["--overwrite"], bad) == 2

    conn = pg_conn(fresh=False)
    try:
        assert _count(conn, "source = 'historical_backfill'") == len(DAYS)         # never duplicated
        assert _count(conn, "source = 'live'") == 1
        with conn.cursor() as cur:
            cur.execute("SELECT (payload->'sentiment'->>'value')::float AS v FROM options_flow_symbol_snapshots WHERE source='live'")
            assert cur.fetchone()["v"] == 0.77                                     # live row untouched
            cur.execute("SELECT count(*) AS n FROM options_flow_runs WHERE id = %s", (live_rid,))
            assert cur.fetchone()["n"] == 1
            cur.execute("SELECT (payload->'sentiment'->>'value')::float AS v, payload->'quality'->>'dataStatus' AS s "
                        "FROM options_flow_symbol_snapshots WHERE source='historical_backfill' AND market_date = %s", (DAYS[3],))
            row = cur.fetchone()
            assert row["v"] == -1.0                                                # failed re-run left the last good (bearish) row
            cur.execute("SELECT status, diagnostics->'missingReasons' AS r FROM options_flow_runs WHERE status = 'failed' AND market_date = %s", (DAYS[3],))
            fail = cur.fetchone()
            assert fail is not None and "theta exploded" in str(fail)
    finally:
        conn.close()


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
