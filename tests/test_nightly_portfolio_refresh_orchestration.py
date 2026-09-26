"""
Orchestration-only regression tests for jobs/nightly_portfolio_refresh.py.

These cover the non-blocking lock, the tri-state (published / failed / skipped) return from
_run_strategy, and the max-runtime watchdog -- the pieces added to fix a run that hung for
hours (see api/services/portfolio_lock.py's docstring for the incident this replaces).
Deliberately does NOT touch or re-derive portfolio methodology/holdings/scoring: everything
here mocks _run_strategy_locked and the actual build/publish helpers.
"""

import argparse
import threading
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

import jobs.nightly_portfolio_refresh as npr


def test_run_strategy_skips_cleanly_when_lock_already_held():
    """A duplicate/overlapping invocation must never wait -- it exits cleanly instead."""
    fake_conn = MagicMock()
    with patch.object(npr, "get_connection", return_value=fake_conn), \
         patch.object(npr, "_lock_try_acquire", return_value=False), \
         patch.object(npr, "_lock_release") as mock_release, \
         patch.object(npr, "_run_strategy_locked") as mock_locked:
        result = npr._run_strategy("stock_alpha", date(2026, 9, 25), dry_run=False)

    assert result is None
    mock_locked.assert_not_called()
    mock_release.assert_called_once_with(fake_conn, "stock_alpha")
    fake_conn.close.assert_called_once()


def test_run_strategy_proceeds_and_releases_lock_when_acquired():
    fake_conn = MagicMock()
    with patch.object(npr, "get_connection", return_value=fake_conn), \
         patch.object(npr, "_lock_try_acquire", return_value=True), \
         patch.object(npr, "_lock_release") as mock_release, \
         patch.object(npr, "_run_strategy_locked", return_value=True) as mock_locked:
        result = npr._run_strategy("stock_alpha", date(2026, 9, 25), dry_run=False)

    assert result is True
    mock_locked.assert_called_once_with("stock_alpha", date(2026, 9, 25), False)
    mock_release.assert_called_once_with(fake_conn, "stock_alpha")
    fake_conn.close.assert_called_once()


def test_run_strategy_releases_lock_even_if_the_locked_run_raises():
    """The old pg_advisory_xact_lock relied on transaction rollback to release on crash; the
    new lock is explicit, so this must hold even when the build/publish path blows up."""
    fake_conn = MagicMock()
    with patch.object(npr, "get_connection", return_value=fake_conn), \
         patch.object(npr, "_lock_try_acquire", return_value=True), \
         patch.object(npr, "_lock_release") as mock_release, \
         patch.object(npr, "_run_strategy_locked", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            npr._run_strategy("stock_alpha", date(2026, 9, 25), dry_run=False)

    mock_release.assert_called_once_with(fake_conn, "stock_alpha")
    fake_conn.close.assert_called_once()


def test_skipped_strategies_do_not_count_as_failure():
    """A strategy skipped due to lock contention must not trip had_failure / SystemExit(1) --
    it means another invocation is legitimately handling it, not that this run failed."""
    args = argparse.Namespace(
        strategy=None, all=True, force=True, dry_run=False, run_date="2026-09-25",
        scheduled_hour=None, max_runtime_seconds=60,
    )
    with patch.object(npr, "_run_strategy", return_value=None):
        npr._run_main(args)  # must not raise SystemExit


def test_one_failed_one_skipped_still_exits_nonzero():
    args = argparse.Namespace(
        strategy=None, all=True, force=True, dry_run=False, run_date="2026-09-25",
        scheduled_hour=None, max_runtime_seconds=60,
    )
    results = iter([False, None])
    with patch.object(npr, "_run_strategy", side_effect=lambda *a, **k: next(results)):
        with pytest.raises(SystemExit) as exc_info:
            npr._run_main(args)
    assert exc_info.value.code == 1


def test_runtime_guard_force_exits_past_deadline(monkeypatch):
    killed = threading.Event()
    monkeypatch.setattr(npr.os, "_exit", lambda code: killed.set())

    timer = npr._install_runtime_guard(0.05)
    try:
        assert killed.wait(timeout=2), "runtime guard did not fire"
    finally:
        timer.cancel()


def test_runtime_guard_cancelled_before_deadline_never_fires(monkeypatch):
    killed = threading.Event()
    monkeypatch.setattr(npr.os, "_exit", lambda code: killed.set())

    timer = npr._install_runtime_guard(60)
    timer.cancel()
    assert not killed.wait(timeout=0.2)


def test_stage_marker_prints_even_before_job_start_is_set(capsys):
    npr._START_MONOTONIC = None
    npr._stage("stock_alpha", "lock:acquire")
    out = capsys.readouterr().out
    assert "[stock_alpha] STAGE lock:acquire" in out
