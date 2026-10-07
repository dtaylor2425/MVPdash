from datetime import date, timedelta
import pytest
from api.services.stock_participation import build_participation_payload
from api.services.stock_research_score import build_research_score


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
