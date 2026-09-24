from copy import deepcopy
from datetime import date
import pytest
from src.portfolio_ledger import build_ledger, first_entries, macro_target, volume_signal


def bar(day, price, **extra):
    return dict(date=day, open=price, close=price, volume=100, split=0, dividend=0, **extra)


def fixture():
    payload = {"holdings": [{"ticker": "AAA", "target_weight": .7, "close": 100}],
               "performance": {"series": [{"date": "2026-09-18", "model": .2, "benchmark": .1}],
                               "stats": {"model_return": .2}, "rebalance_log": [{"date": "2026-09-18"}]}}
    previous = {"id": "first", "run_date": "2026-09-18", "published_at": "2026-09-18T22:00:00Z", "payload": payload}
    candidate = {"holdings": [{"ticker": "AAA", "target_weight": .7}],
                 "risk_rules": {"max_single_position": 1, "max_sector_weight": 1}}
    market = {"AAA": [bar("2026-09-18", 100), bar("2026-09-21", 110), bar("2026-09-22", 121)],
              "SPY": [bar("2026-09-18", 400), bar("2026-09-21", 400), bar("2026-09-22", 400)]}
    return candidate, previous, market


def test_migration_preserves_history_entries_and_accounts_forward():
    candidate, previous, market = fixture()
    original = deepcopy(previous)
    result = build_ledger(candidate, previous, [previous], market, "2026-09-22", 50)
    assert previous == original
    assert result["performance"]["series"][0] == previous["payload"]["performance"]["series"][0]
    assert result["performance"]["series"][-1]["model"] == .2
    assert result["ledger"]["anchor_date"] == "2026-09-21"
    assert "unmeasured" in result["performance"]["methodology_note"]
    position = result["holdings"][0]
    assert position["entry_date"] == "2026-09-18"
    assert position["entry_price"] == 100
    assert position["unrealized_return_pct"] == pytest.approx(10)
    assert position["current_weight"] == pytest.approx(.7)
    assert result["trade_queue"] == []
    assert result["pending_allocation"]["execute_not_before"] == "2026-09-23"


def test_same_state_rerun_preserves_curve_and_moves():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-22", 50)
    second = build_ledger(c, {"payload": first}, [previous], market, "2026-09-22", 50)
    assert second["performance"] == first["performance"]
    assert second["ledger"] == first["ledger"]
    assert second["holdings"] == first["holdings"]


def test_next_session_execution_preserves_entry_and_previous_curve():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-23", 50)
    assert result["performance"]["series"][:1] == first["performance"]["series"]
    assert all(t["date"] == "2026-09-22" for t in result["trade_queue"])
    assert result["holdings"][0]["entry_price"] == 100
    assert result["holdings"][0]["current_weight"] == pytest.approx(.8)
    assert result["ledger"]["trades"] == result["trade_queue"]


def test_reentry_gets_new_reference_and_missing_is_not_invented():
    history = [dict(run_date="2026-01-01", holdings=[dict(ticker="A", close=10)]),
               dict(run_date="2026-01-02", holdings=[]),
               dict(run_date="2026-01-03", holdings=[dict(ticker="A", close=20), dict(ticker="B")])]
    entries = first_entries(history)
    assert entries["A"]["entry_date"] == "2026-01-03"
    assert entries["A"]["entry_price"] == 20
    assert entries["B"]["entry_price"] is None


def test_publication_date_not_backdated_model_date():
    rows = [dict(run_date="2026-01-01", published_at="2026-01-02T01:00:00Z", holdings=[dict(ticker="A", close=10)])]
    assert first_entries(rows)["A"]["entry_date"] == "2026-01-02"
    assert first_entries(rows)["A"]["entry_model_date"] == "2026-01-01"


@pytest.mark.parametrize("score,target", [(0,.65),(25,.65),(50,.8),(75,.95),(100,.95)])
def test_macro_range(score, target):
    assert macro_target(score) == target


def test_invalid_macro_fails_closed():
    for value in (None, float("nan"), -1, 101, True):
        with pytest.raises(ValueError): macro_target(value)


def test_caps_spy_sleeve_and_target_reconcile():
    from src.portfolio_ledger import target_holdings
    candidate = dict(holdings=[dict(ticker="A", target_weight=.5, sector="Tech"), dict(ticker="B", target_weight=.5, sector="Tech")],
                     risk_rules=dict(max_single_position=.12, max_sector_weight=.2))
    rows, target = target_holdings(candidate, {}, 100)
    assert sum(r["target_weight"] for r in rows) == pytest.approx(.95)
    assert all(r["target_weight"] <= .12 for r in rows if r["ticker"] != "SPY")
    assert sum(r["target_weight"] for r in rows if r.get("sector") == "Tech") == pytest.approx(.2)


@pytest.mark.parametrize("price,volume,label", [(101,200,"Strong buying volume"),(99,200,"Strong selling volume"),(101,50,"Weak buying volume"),(99,50,"Weak selling volume")])
def test_directional_volume(price, volume, label):
    bars = [dict(close=100, volume=100)] * 20 + [dict(close=price, volume=volume)]
    assert volume_signal(bars)["volume_signal"] == label


def test_split_preserves_nav_and_original_entry():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    market["AAA"][1].update(open=50, close=50, split=2)
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-22", 50)
    assert result["performance"]["series"][-1]["model"] == pytest.approx(.2)
    assert result["holdings"][0]["entry_price"] == 50
    assert result["holdings"][0]["original_entry_price"] == 100
    assert result["holdings"][0]["unrealized_return_pct"] == 0


def test_dividend_books_cash_without_faking_price_gain():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    market["AAA"][1].update(open=99, close=99, dividend=1)
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-22", 50)
    assert result["performance"]["series"][-1]["model"] == pytest.approx(.2)
    assert result["cash_weight"] == pytest.approx(.307)
    assert result["holdings"][0]["unrealized_return_pct"] == pytest.approx(-1)


def test_missing_price_cannot_publish_a_false_gain():
    c, previous, market = fixture()
    market["AAA"] = market["AAA"][:1]
    with pytest.raises(ValueError, match="Missing valid market bar"):
        build_ledger(c, previous, [previous], market, "2026-09-22", 50)


def test_publisher_same_day_skips_builder(monkeypatch):
    from jobs import nightly_portfolio_refresh as job
    monkeypatch.setattr(job, "_load_latest_published", lambda _: {"id": "frozen", "run_date": date(2026,9,22)})
    monkeypatch.setattr(job, "_build_strategy_payload", lambda _: pytest.fail("rerun rebuilt portfolio"))
    assert job._run_strategy_locked("stock_alpha", date(2026,9,22), False)


def test_market_drift_is_not_logged_as_trade():
    from jobs.nightly_portfolio_refresh import _official_rebalance_diff
    prior = dict(holdings=[dict(ticker="A", current_weight=.7)])
    current = dict(holdings=[dict(ticker="A", current_weight=.8)], ledger={"version":"test"}, trade_queue=[])
    diff = _official_rebalance_diff(prior, current, date(2026,9,22))
    assert diff["adds"] == diff["trims"] == []
    assert diff["turnover"] == 0


def test_revised_provider_anchor_fails_without_rewriting_history():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    saved = deepcopy(first)
    market["AAA"][0]["close"] = 50
    with pytest.raises(ValueError, match="reconciliation required"):
        build_ledger(c, {"payload": first}, [previous], market, "2026-09-22", 50)
    assert first == saved


@pytest.mark.parametrize("exit_price,expected", [(121,21),(80,-20),(100,0)])
def test_full_exit_pnl_is_frozen_from_entry_reference(exit_price, expected):
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    first["ledger"]["pending"]["holdings"] = [{"ticker": "SPY", "target_weight": .8}]
    market["AAA"][2].update(open=exit_price, close=exit_price)
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-23", 50)
    exit = next(t for t in result["trade_queue"] if t["ticker"] == "AAA")
    assert exit["exit_type"] == "Full exit"
    assert exit["entry_date"] == "2026-09-18"
    assert exit["entry_price"] == 100
    assert exit["exit_price"] == exit_price
    assert exit["exit_return_pct"] == pytest.approx(expected)
    from jobs.nightly_portfolio_refresh import _official_rebalance_diff
    official = _official_rebalance_diff(first, result, date(2026,9,23))
    assert official["exits"] == [exit]
    rerun = build_ledger(c, {"payload": result}, [previous], market, "2026-09-23", 50)
    assert rerun["ledger"]["trades"] == result["ledger"]["trades"]


def test_trim_and_missing_entry_are_explicit():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    first["ledger"]["pending"]["holdings"] = [{"ticker": "AAA", "target_weight": .4}, {"ticker": "SPY", "target_weight": .4}]
    first["ledger"]["positions"]["AAA"]["entry_price"] = None
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-23", 50)
    exit = next(t for t in result["trade_queue"] if t["ticker"] == "AAA")
    assert exit["exit_type"] == "Trim"
    assert exit["exit_return_pct"] is None


def test_split_adjusted_exit_does_not_report_false_loss():
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    first["ledger"]["pending"]["holdings"] = [{"ticker":"SPY", "target_weight":.8}]
    market["AAA"][2].update(open=50, close=50, split=2)
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-23", 50)
    exit = next(t for t in result["trade_queue"] if t["ticker"] == "AAA")
    assert exit["entry_price"] == 50
    assert exit["exit_return_pct"] == 0
