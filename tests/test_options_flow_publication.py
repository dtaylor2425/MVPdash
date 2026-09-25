from datetime import date, datetime, timezone
from uuid import UUID
from types import SimpleNamespace
import pandas as pd
import pytest
from api.services import options_flow_publication as pub
from api.services import options_flow_store as store
from api.services import options_flow_market_activity as activity
from api.services import options_flow_issues as issues


def row(day=date(2026, 9, 21), state="final", source="live", gross=100, method="v1"):
    return {"id": "00000000-0000-0000-0000-000000000001", "run_id": "r1", "ticker": "SPY",
            "group_name": "Equity", "market_date": day, "source": source, "mode": "full_flow",
            "methodology_version": method, "run_status": "success",
            "run_config": {"sessionStart": "09:30:00", "sessionEnd": "16:15:00", "maxDte": 60, "strikeRange": 25},
            "created_at": datetime(2026, 9, 22, tzinfo=timezone.utc),
            "as_of_timestamp": datetime(2026, 9, 21, 20, 15, tzinfo=timezone.utc),
            "payload": {"publication": {"state": state}, "premium": {"gross": gross},
                        "quality": {"greekCoverage": None}, "largeTrades": [{"private": True}], "intraday": [1]}}


def test_partial_and_methodology_mismatch_never_generate_daily_delta():
    previous = row(gross=20)
    assert pub.comparisons([row(gross=100)], [previous])[0]["metrics"]["grossPremium"]["delta"] == 80
    for current in [row(state="partial"), row(method="v2")]:
        result = pub.comparisons([current], [previous])[0]
        assert result["comparable"] is False
        assert result["metrics"]["grossPremium"]["delta"] is None
    assert not pub.comparisons([row()], [])[0]["comparable"]


def test_provenance_detail_and_summary_preserve_quality_without_fabricating():
    r = row()
    assert pub.ticker_view(r)["quality"]["greekCoverage"] is None
    assert "intraday" not in pub.ticker_view(r)
    assert pub.ticker_view(r, heavy=True)["largeTrades"] == [{"private": True}]
    assert pub.provenance(row(source="historical_backfill"))["publicationState"] == "historical"
    r["payload"].pop("publication")
    assert pub.provenance(r)["comparable"] is False


def test_activity_does_not_attach_another_sessions_factor(monkeypatch):
    monkeypatch.setattr(activity, "compute_market_activity_snapshot", lambda *a, **kw: {"asOfDate": "2026-09-18"})
    monkeypatch.setattr(activity, "load_v4_research_summary", lambda: None)
    out = {"tickers": [{"ticker": "SPY"}]}
    store.attach_market_activity(None, out, date(2026, 9, 21))
    assert out["marketActivity"] is None
    assert out["tickers"][0]["activityZ"] is None


class Cursor:
    def __init__(self, rows): self.rows, self.calls = rows, []
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def execute(self, sql, params=None): self.calls.append((sql, params))
    def fetchall(self): return self.rows


class Connection:
    def __init__(self, rows): self.cur = Cursor(rows)
    def cursor(self): return self.cur


def test_operational_history_explicitly_enables_live_final_research_default_does_not():
    conn = Connection([])
    activity.load_gross_premium_history(conn)
    assert conn.cur.calls[-1][1]["include_live"] is False
    activity.load_gross_premium_history(conn, include_live=True)
    sql, params = conn.cur.calls[-1]
    assert params["include_live"] is True
    assert "->>'state' = 'final'" in sql
    assert "s.mode = 'full_flow'" in sql


def test_session_selector_has_no_carry_forward_and_history_has_upper_bound():
    conn = Connection([])
    pub.rows_for(conn, date(2026, 9, 21), "SPY")
    sql, params = conn.cur.calls[-1]
    assert "s.market_date=%(session)s" in sql
    assert params["session"] == date(2026, 9, 21)
    store.fetch_history(conn, "SPY", 20, date(2026, 9, 21))
    assert "s.market_date <= %(session)s" in conn.cur.calls[-1][0]
    for sql in [store.LATEST_SUMMARY_SQL, store.LATEST_TICKER_SQL, store.HISTORY_SQL]:
        assert "s.mode = 'full_flow'" in sql


def test_frozen_export_is_copied_deterministic_and_excludes_private_prints():
    r = row()
    snapshot = {"session": {"snapshotIds": [r["id"]]}, "tickers": [pub.ticker_view(r, heavy=True)],
                "collectionStatus": {"secretDiagnostic": 123}}
    evidence, digest = issues.freeze_evidence(snapshot, [UUID(r["id"])])
    assert "largeTrades" not in evidence["tickers"][0]
    assert "collectionStatus" not in evidence
    assert digest == issues.freeze_evidence(snapshot, [r["id"]])[1]
    snapshot["tickers"][0]["premium"]["gross"] = 999
    assert evidence["tickers"][0]["premium"]["gross"] == 100
    with pytest.raises(issues.SnapshotConflict):
        issues.freeze_evidence(snapshot, ["different"])


def test_evidence_hash_detects_prior_history_change_with_same_current_ids():
    r = row()
    snapshot = {"session": {"snapshotIds": [r["id"]]}, "tickers": [],
                "marketActivity": {"history": [{"date": "2026-09-18", "value": 1.0}]}}
    _, digest = issues.freeze_evidence(snapshot, [r["id"]])
    snapshot["session"]["evidenceSha256"] = digest
    assert issues.freeze_evidence(snapshot, [r["id"]])[1] == digest
    snapshot["marketActivity"]["history"][0]["value"] = 2.0
    request = SimpleNamespace(snapshotIds=[r["id"]], expectedEvidenceSha256=digest)
    with pytest.raises(issues.SnapshotConflict, match="Analytical evidence changed"):
        issues.create(None, request, snapshot)


def test_operational_activity_cache_invalidates_by_publication_identity(monkeypatch):
    class CacheCursor(Cursor):
        def fetchone(self): return {"identity": self.rows}
    conn = Connection([])
    conn.cur = CacheCursor("publication-a")
    calls = []
    def load(*args, **kwargs):
        calls.append(1)
        return pd.DataFrame([{"ticker": t, "date": d.date().isoformat(), "gross_premium": float(i*i+1)}
                             for t in activity.ACTIVITY_TICKERS
                             for i, d in enumerate(pd.bdate_range("2026-01-01", periods=45))])
    monkeypatch.setattr(activity, "load_gross_premium_history", load)
    activity._ACTIVITY_CACHE.clear()
    end = date(2026, 2, 27)
    a = activity.compute_market_activity_snapshot(conn, end, include_live=True)
    b = activity.compute_market_activity_snapshot(conn, end, include_live=True)
    assert a == b and len(calls) == 1
    conn.cur.rows = "publication-b"
    activity.compute_market_activity_snapshot(conn, end, include_live=True)
    assert len(calls) == 2
    assert all(p["date"] <= end.isoformat() for p in a["history"])


def test_private_archive_and_sessions_fail_closed(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import private_options_flow as router
    app = FastAPI()
    app.include_router(router.router)
    monkeypatch.setenv("INTERNAL_OPTIONS_API_SECRET", "test-token")
    monkeypatch.setattr(router, "get_connection", lambda: pytest.fail("Unauthorized route opened database"))
    client = TestClient(app)
    assert client.get("/api/private/options-flow/sessions").status_code == 404
    assert client.get("/api/private/options-flow/issues").status_code == 404
    assert client.post("/api/private/options-flow/issues", json={}).status_code == 404


def test_refresh_during_options_tail_is_not_a_final_close():
    from api.services.options_flow_calendar import compute_data_status
    day = date(2026, 9, 21)
    published = datetime(2026, 9, 21, 20, 5, tzinfo=timezone.utc)
    now = datetime(2026, 9, 21, 21, 0, tzinfo=timezone.utc)
    assert compute_data_status(day, published, now)["status"] == "STALE"


def test_worker_overlap_skips_before_schema_or_vendor(monkeypatch):
    from jobs import options_flow_refresh as job
    class LockedConnection:
        closed = False
        def cursor(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params=None): assert "pg_try_advisory_lock" in sql
        def fetchone(self): return {"acquired": False}
        def close(self): self.closed = True
    conn = LockedConnection()
    monkeypatch.setattr(store, "ensure_schema", lambda _: pytest.fail("overlap reached schema"))
    assert job.run(["--date", "2026-09-21"], conn_factory=lambda: conn) == 0
    assert conn.closed


def test_activity_baselines_restart_on_profile_changes_and_missing_sessions():
    from api.services.options_flow_calendar import sessions_between
    days = sessions_between(date(2026, 1, 2), date(2026, 3, 30))[:35]
    cfg = row()["run_config"]
    rows = [{"ticker": "SPY", "date": d.isoformat(), "gross_premium": float(i*i+1),
             "methodology_version": "v1", "collection_config": {**cfg, "maxDte": 60 if i < 15 else 30}}
            for i, d in enumerate(days)]
    frame = activity.market_activity_frame(activity.compatible_activity_segments(pd.DataFrame(rows)), min_tickers=1, baseline_group_col="activity_segment")
    assert pd.notna(frame.iloc[14]["z_gross_premium_20d"])
    assert frame.iloc[15:25]["z_gross_premium_20d"].isna().all()
    assert pd.notna(frame.iloc[25]["z_gross_premium_20d"])
    # Same profile after a missing exchange session still starts a new baseline.
    gap = rows[:28] + rows[29:]
    segmented = activity.compatible_activity_segments(pd.DataFrame(gap))
    assert segmented.iloc[27]["activity_segment"] != segmented.iloc[28]["activity_segment"]


def test_unknown_profiles_and_method_changes_never_pool_baselines():
    from api.services.options_flow_calendar import sessions_between
    days = sessions_between(date(2026, 1, 2), date(2026, 3, 30))[:30]
    rows = [{"ticker": "SPY", "date": d.isoformat(), "gross_premium": float(i+1),
             "methodology_version": "v1" if i < 15 else "v2", "collection_config": row()["run_config"]}
            for i, d in enumerate(days)]
    frame = activity.market_activity_frame(activity.compatible_activity_segments(pd.DataFrame(rows)), 1, "activity_segment")
    assert frame.iloc[15:25]["z_gross_premium_20d"].isna().all()
    for r in rows:
        r["collection_config"] = {}
    unknown = activity.market_activity_frame(activity.compatible_activity_segments(pd.DataFrame(rows)), 1, "activity_segment")
    assert unknown["z_gross_premium_20d"].isna().all()
