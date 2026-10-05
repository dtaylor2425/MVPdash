"""Startup retries are distinct from the collector's per-request gRPC retries."""
import sys
from types import SimpleNamespace
import httpx
import pytest
from jobs import options_flow_refresh as job


@pytest.mark.parametrize("error", [httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_transient_startup_recovers_without_logging_auth(monkeypatch, capsys, error):
    calls, sleeps = [], []
    client = object()
    def connect(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise error("secret-test-key")
        return client
    monkeypatch.setenv("THETADATA_API_KEY", "secret-test-key")
    monkeypatch.setitem(sys.modules, "thetadata", SimpleNamespace(ThetaClient=connect))
    monkeypatch.setattr(job.time, "sleep", sleeps.append)
    assert job.ThetaFetcher._connect() is client
    assert len(calls) == 3 and sleeps == [2.0, 5.0]
    log = capsys.readouterr().out
    assert "secret-test-key" not in log
    assert "recovered on attempt 3" in log


def test_startup_retries_are_bounded(monkeypatch):
    calls, sleeps = [], []
    def connect(**kwargs):
        calls.append(kwargs)
        raise httpx.ConnectTimeout("temporary failure")
    monkeypatch.setenv("THETADATA_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "thetadata", SimpleNamespace(ThetaClient=connect))
    monkeypatch.setattr(job.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        job.ThetaFetcher._connect()
    assert len(calls) == 4 and sleeps == [2.0, 5.0, 10.0]


@pytest.mark.parametrize("error", [ValueError("invalid credentials"), httpx.HTTPStatusError(
    "Forbidden", request=httpx.Request("POST", "https://example.test"), response=httpx.Response(403))])
def test_permanent_startup_error_does_not_retry(monkeypatch, error):
    calls = []
    def connect(**kwargs):
        calls.append(kwargs)
        raise error
    monkeypatch.setenv("THETADATA_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "thetadata", SimpleNamespace(ThetaClient=connect))
    monkeypatch.setattr(job.time, "sleep", lambda _: pytest.fail("permanent error retried"))
    with pytest.raises(type(error)):
        job.ThetaFetcher._connect()
    assert len(calls) == 1
