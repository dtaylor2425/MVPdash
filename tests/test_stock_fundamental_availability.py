import pandas as pd
import pytest
from types import SimpleNamespace
from api.routers import stock_intelligence as stock


def points(dates, values):
    return [{"date": d, "value": v} for d, v in zip(dates, values)]


def test_yoy_matches_year_not_four_positions():
    data=points(["2024-03-31","2024-06-30","2024-12-31","2025-03-31"], [100,110,120,150])
    assert stock._yoy_series(data)[-1]["yoy_growth"] == pytest.approx(.5)
    data=points(["2023-12-31","2024-03-31","2024-06-30","2024-12-31","2025-03-31"], [50,100,110,120,150])
    assert stock._yoy_series(data)[-1]["yoy_growth"] == pytest.approx(.5)


def test_fiscal_53_week_year_supported():
    data=points(["2024-01-28","2025-02-02"], [100,150])
    assert stock._yoy_series(data)[-1]["yoy_growth"] == pytest.approx(.5)


def test_acceleration_needs_latest_adjacent_rates():
    series=[{"date":"2025-03-31","yoy_growth":.1},{"date":"2025-06-30","yoy_growth":.2}]
    assert stock._latest_acceleration(series) == pytest.approx(.1)
    assert stock._latest_acceleration(series+[{"date":"2025-09-30","yoy_growth":None}]) is None
    series[-1]["date"]="2025-09-30"
    assert stock._latest_acceleration(series) is None


def ticker_with_quarters(n):
    cols=pd.date_range("2024-03-31",periods=n,freq="QE")
    frame=pd.DataFrame([range(100,100+n),range(10,10+n)],index=["Total Revenue","Diluted EPS"],columns=cols)
    return SimpleNamespace(quarterly_income_stmt=frame,quarterly_cashflow=pd.DataFrame())


def test_five_quarters_explains_missing_acceleration():
    result=stock._fundamental_velocity(ticker_with_quarters(5))
    assert result["revenue_acceleration"] is None
    assert result["availability"]["revenue_acceleration"]["valid_quarters"] == 5
    assert not result["availability"]["revenue_acceleration"]["available"]
    assert result["availability"]["revenue_acceleration"]["reason"]
    assert result["availability"]["score"]["available"]


def test_six_quarters_can_supply_acceleration():
    result=stock._fundamental_velocity(ticker_with_quarters(6))
    assert result["revenue_acceleration"] is not None
    assert result["availability"]["revenue_acceleration"]["available"]


@pytest.mark.parametrize("capex",[-20,20])
def test_velocity_fcf_derives_from_same_period(capex):
    frame=pd.DataFrame({pd.Timestamp("2025-03-31"):[100,capex]},index=["Operating Cash Flow","Capital Expenditure"])
    assert stock._quarterly_series(frame,stock.STATEMENT_ALIASES["free_cash_flow"])[0]["value"] == 80


def test_ttm_is_not_compared_with_previous_fiscal_year():
    ticker=ticker_with_quarters(5)
    ticker.income_stmt=pd.DataFrame({pd.Timestamp("2023-12-31"):[300,3],pd.Timestamp("2024-12-31"):[400,4]},index=["Total Revenue","Diluted EPS"])
    model=stock._build_financial_model(ticker)
    annual=model["records"][-2]
    assert annual["revenue_growth"] == pytest.approx(1/3)
    assert model["records"][-1]["label"] == "TTM"
    assert model["records"][-1]["revenue_growth"] is None
    assert model["records"][-1]["eps_growth"] is None


def test_no_evidence_is_not_neutral():
    result=stock._fundamental_velocity(SimpleNamespace())
    assert result["score"] is None
    assert result["label"] == "Unavailable"
    assert result["availability"]["gross_margin_expansion"]["reason"]
    assert not result["availability"]["series"]["revenue"]["available"]
    assert result["availability"]["series"]["revenue"]["growth_reason"]


def test_skipped_fiscal_year_is_not_yoy_growth():
    ticker=ticker_with_quarters(5)
    ticker.income_stmt=pd.DataFrame({pd.Timestamp("2022-12-31"):[300,3],pd.Timestamp("2024-12-31"):[400,4]},index=["Total Revenue","Diluted EPS"])
    model=stock._build_financial_model(ticker)
    assert model["records"][-2]["revenue_growth"] is None
    assert model["availability"]["FY24"]["revenue_growth"]["reason"]
    assert "trailing" in model["availability"]["TTM"]["revenue_growth"]["reason"]
