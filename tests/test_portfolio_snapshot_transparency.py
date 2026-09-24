from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone
from api.routers import portfolio_snapshots as snapshots


def test_legacy_entry_enrichment_preserves_stored_payload_and_curve(monkeypatch):
    records = [{"id": "first", "run_date": date(2026,8,10), "published_at": datetime(2026,8,11,tzinfo=timezone.utc),
                "holdings": [{"ticker": "A", "close": 100, "target_weight": .5}]}]
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params):
            assert "published_at <=" in sql
            assert params[1] == row["published_at"]
        def fetchall(self): return records
    class Connection:
        def cursor(self): return Cursor()
    @contextmanager
    def connection(): yield Connection()
    monkeypatch.setattr(snapshots, "get_connection", connection)
    row = dict(id="latest", strategy="stock_alpha", run_date=date(2026,9,22),
               published_at=datetime(2026,9,22,tzinfo=timezone.utc), status="published", is_published=True,
               payload={"holdings": [{"ticker": "A", "close": 110, "target_weight": .5}],
                        "stock_exposure": .95, "performance": {"series": [{"date": "2026-09-21", "model": .12345}]}})
    original = deepcopy(row)
    result = snapshots._entry_references(row, snapshots._row_to_payload(row, []))
    assert row == original
    assert result["holdings"][0]["entry_date"] == "2026-08-11"
    assert result["holdings"][0]["entry_price"] == 100
    assert round(result["holdings"][0]["unrealized_return_pct"], 8) == 10
    assert result["stock_exposure"] == result["cash_weight"] == .5
    assert result["performance"] == original["payload"]["performance"]


def test_internal_ledger_is_not_returned_and_anonymous_history_stays_gated():
    row = dict(id="id", strategy="stock_alpha", status="published", is_published=True,
               payload={"ledger": {"trades": ["private"]}, "official_rebalance": {"buys": ["A"]},
                        "trade_queue": ["private"], "performance": {"rebalance_log": ["private"]}})
    public = snapshots._row_to_payload(row, ["private"])
    assert "ledger" not in public
    anonymous = snapshots._truncate_for_anon(public)
    assert anonymous["trade_queue"] == anonymous["official_rebalance_log"] == []
    assert anonymous["performance"]["rebalance_log"] == []
    assert "official_rebalance" not in anonymous
    assert "ledger" in row["payload"]


def test_public_history_never_leaks_rebalance_diagnostics(monkeypatch):
    record = dict(id="published", run_date=date(2026,9,23), holdings=[{"ticker":"A"}],
                  diagnostics={"official_rebalance":{"exits":[{"ticker":"PRIVATE", "exit_return_pct":21}]}})
    class Cursor:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def execute(self,sql,params): assert "diagnostics" not in sql
        def fetchall(self): return [record]
    class Connection:
        def cursor(self): return Cursor()
    @contextmanager
    def connection(): yield Connection()
    monkeypatch.setattr(snapshots,"get_connection",connection)
    result = snapshots.get_portfolio_snapshot_history("stock_alpha",20)
    assert result["runs"][0]["diagnostics"] == {}
    assert "PRIVATE" not in str(result)
    assert result["runs"][0]["holding_count"] == 1
