from datetime import date, timedelta
import pytest
from api.services.stock_participation import build_participation_payload
from api.services.stock_research_score import build_research_score
from api.services.stock_participation import enrich_price_bars, session_anatomy


def test_moving_averages_have_real_warmup_and_no_future_input():
    data = bars(280)
    result = enrich_price_bars(data)
    assert result[48]['sma50'] is None
    assert result[49]['sma50'] == 125.5
    assert result[198]['sma200'] is None
    assert result[199]['sma200'] == 200.5
    data[-1]['close'] = 99999
    assert enrich_price_bars(data)[199]['sma200'] == result[199]['sma200']
    assert enrich_price_bars(data)[-1]['sma200'] is None


def test_session_activity_respects_early_close_and_missing_volume():
    # Friday after Thanksgiving closes at 13:00 ET, not 16:00.
    data = [dict(timestamp='2026-11-27T'+t, open=100, high=102, low=99, close=101, volume=v)
            for t,v in [('09:30:00',100),('10:00:00',200),('12:30:00',300)]]
    out = session_anatomy(data, '2026-11-27')
    assert out['regular_volume'] == 600
    assert out['opening_30m_volume_share_pct'] == pytest.approx(100/6)
    assert out['closing_30m_volume_share_pct'] == 50
    assert out['intraday_range_pct'] == 3
    assert out['regular_dollar_turnover_estimate'] == pytest.approx(60400)
    data[1]['volume'] = None
    assert session_anatomy(data, '2026-11-27')['regular_volume'] is None


def test_old_snapshot_enrichment_preserves_stored_score_and_inputs(monkeypatch):
    import copy
    from datetime import datetime, timezone
    from fastapi import Response
    from api.routers.stock_participation import stock_participation
    from api.services import theta_stock_store as store
    daily = bars(280)
    payload = build_participation_payload('NET', daily, [], daily)
    payload.pop('analytics_view_version')
    payload['bars'] = daily[-260:]
    saved = copy.deepcopy(payload)
    monkeypatch.setattr(store, 'latest', lambda symbol: dict(payload=payload,daily_bars=daily,
        market_date=date.fromisoformat(daily[-1]['date']),updated_at=datetime.now(timezone.utc)))
    monkeypatch.setattr(store, 'load_daily_cache', lambda symbol: dict(daily_bars=daily))
    result = stock_participation('NET', Response())
    assert result['sector_symbol'] == 'XLK'
    assert result['metrics']['relative_strength_sector_20d_pct'] == 0
    assert result['bars'][-1]['sma200'] is not None
    assert result['score'] == saved['score']
    assert payload == saved


def bars(n=70):
    return [{"date":str(date(2026,1,1)+timedelta(days=i)),"open":100+i,"high":102+i,"low":99+i,"close":101+i,"volume":100} for i in range(n)]


def intraday(d,close):
    return [{"timestamp":d+"T09:30:00","open":close-1,"high":close+1,"low":close-2,"close":close,"volume":200}]


def test_volume_baseline_excludes_current_and_uses_eod_only():
    daily=bars(); daily[-1]["volume"]=400
    result=build_participation_payload("TEST",daily,intraday(daily[-1]["date"],168),daily)
    assert result["metrics"]["relative_volume"] == 4
    assert result["metrics"]["relative_strength_spy_20d_pct"] == 0
    assert result["metrics"]["gap_pct"] is None
    assert result["score"]["coverage_pct"] == 100


def test_regular_session_gap_does_not_use_after_hours_eod_close():
    daily=bars()
    regular=intraday(daily[-2]["date"],150)+intraday(daily[-1]["date"],153)
    out=build_participation_payload("TEST",daily,regular,daily)
    assert out["metrics"]["gap_pct"] == pytest.approx((152/150-1)*100,abs=.0001)
    assert out["metrics"]["regular_return_1d_pct"] == 2
    assert out["intraday"][0]["timestamp"].startswith(daily[-1]["date"])


def test_missing_history_never_fabricates_relative_volume_or_score():
    daily=bars(1)
    out=build_participation_payload("IPO",daily,[],[])
    assert out["score"]["total"] is None
    assert out["metrics"]["relative_volume"] is None
    assert out["metrics"]["realized_volatility_20d_pct"] is None


def test_large_discontinuity_withholds_multi_session_score():
    daily=bars(); daily[-1].update(open=40,high=45,low=39,close=42)
    out=build_participation_payload("SPLT",daily,[],daily)
    assert out["score"]["total"] is None
    assert out["metrics"]["relative_strength_spy_20d_pct"] is None
    assert any("corporate-action" in x for x in out["observations"])


def test_unmatched_benchmark_dates_are_not_forward_filled():
    daily=bars()
    out=build_participation_payload("TEST",daily,[],daily[:-1])
    assert out["metrics"]["relative_strength_spy_20d_pct"] is None


def test_company_missing_inputs_have_no_neutral_score():
    out=build_research_score({})
    assert out["total"] is None
    assert out["coverage_pct"] == 0
    assert all(c["value"] is None for c in out["components"])


def test_company_zero_inputs_are_evidence_not_missing():
    out=build_research_score({"operatingMargins":0,"returnOnAssets":0,"revenueGrowth":0,"freeCashflow":0,"totalRevenue":100})
    assert out["total"] is not None
    assert out["coverage_pct"] == 65
    scores={c["key"]:c["value"] for c in out["components"]}
    assert scores["profitability"] == 0
    assert scores["cash"] == 0
    assert scores["growth"] == 20


def test_financial_sector_does_not_reward_generic_leverage_ratios():
    out=build_research_score({"sector":"Financial Services","currentRatio":10,"debtToEquity":0,"returnOnEquity":.1,"revenueGrowth":.1,"forwardPE":15,"freeCashflow":100,"totalRevenue":100})
    scores={c["key"]:c["value"] for c in out["components"]}
    assert scores["balance"] is None
    assert scores["cash"] is None
    assert out["coverage_pct"] == 65
