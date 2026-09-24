"""Offline regression coverage for native-monthly CPI and its consumers."""

import numpy as np
import pandas as pd
import pytest

from api.routers.charts import _build_inline_map
from src.derived import cpi_yoy, real_fed_funds
from src.macro_thesis.engine import build_monthly_raw
from src.monthly_data import align_macro_observations, monthly_year_over_year
from src.regime import compute_regime_v3


def macro_frame():
    days = pd.date_range("2024-01-01", "2026-02-28", freq="D")
    dates = pd.date_range("2024-01-01", "2026-02-01", freq="MS")
    cpi = pd.Series(100.0 + np.arange(len(dates)), index=dates)
    frame = pd.DataFrame({"cpi": cpi.reindex(days), "fed_funds": 5.0}, index=days)
    return align_macro_observations(frame)


def test_native_cpi_is_preserved_while_other_inputs_are_carried():
    raw = pd.DataFrame({"cpi": [100.0, np.nan], "y10": [4.0, np.nan]},
                       index=pd.date_range("2024-01-01", periods=2))
    aligned = align_macro_observations(raw)
    assert pd.isna(aligned.iloc[1]["cpi"])
    assert aligned.iloc[1]["y10"] == 4.0


def test_cpi_annual_change_and_real_rate_agree_across_live_consumers():
    macro = macro_frame()
    expected = (125.0 / 113.0 - 1) * 100
    direct = cpi_yoy(macro, pd.DataFrame())
    chart = _build_inline_map(macro, pd.DataFrame())["cpi_yoy"]()
    assert direct.iloc[-1] == pytest.approx(expected)
    pd.testing.assert_series_equal(direct, chart)
    assert direct.loc["2026-02-02"] == direct.loc["2026-02-28"]
    assert real_fed_funds(macro, pd.DataFrame()).iloc[-1] == pytest.approx(5 - expected)
    regime = compute_regime_v3(macro, pd.DataFrame())
    assert regime.components["cpi_momentum"]["level"] == pytest.approx(expected)


def test_thesis_monthly_inputs_use_annual_cpi_and_real_policy_rate():
    macro = macro_frame()
    for column in ["y10", "y5", "y3m", "real10", "real5", "init_claims",
                   "cont_claims", "nfci", "umich", "dollar_broad", "hy_oas",
                   "ig_oas", "y2", "fed_assets"]:
        macro[column] = 1.0
    extra = pd.DataFrame(1.0, index=macro.index, columns=[
        "breakeven_5y5y", "credit_to_gdp", "fed_debt_pub_pct_gdp",
        "nominal_gdp", "term_premium_10y"])
    prices = pd.DataFrame(100.0, index=macro.index, columns=["RSP", "SPY", "CPER", "GLD", "DBC"])
    raw = build_monthly_raw(macro, extra, prices, None,
                            as_of=pd.Timestamp("2026-02-28"))
    expected = (125.0 / 113.0 - 1) * 100
    assert raw.iloc[-1]["i_cpi_yoy"] == pytest.approx(expected)
    assert raw.iloc[-1]["real_policy_rate"] == pytest.approx(5 - expected)


def test_missing_month_is_not_compressed_to_twelve_observations():
    dates = pd.date_range("2024-01-01", "2025-03-01", freq="MS")
    values = pd.Series(100.0 + np.arange(len(dates)), index=dates)
    values.loc["2024-02-01"] = np.nan
    actual = monthly_year_over_year(values)
    assert pd.isna(actual.loc["2025-02-01"])
    assert actual.loc["2025-03-01"] == pytest.approx((114 / 102 - 1) * 100)


def test_new_value_does_not_move_back_to_month_start_or_change_prefix():
    dates = pd.to_datetime(["2024-01-15", "2024-02-15", "2025-01-15", "2025-02-15"])
    values = pd.Series([100.0, 101.0, 110.0, 115.0], index=dates)
    grid = pd.date_range("2024-01-01", "2025-02-28", freq="D")
    sparse = values.reindex(grid)
    full = monthly_year_over_year(sparse)
    prefix = monthly_year_over_year(sparse.loc[:"2025-02-14"])
    pd.testing.assert_series_equal(full.loc[:"2025-02-14"], prefix)
    assert pd.isna(full.loc["2025-01-14"])
    assert full.loc["2025-02-14"] == pytest.approx(10.0)
    assert full.loc["2025-02-15"] == pytest.approx((115 / 101 - 1) * 100)


def test_zero_denominator_is_missing_not_infinite():
    values = pd.Series([0.0, 100.0], index=pd.to_datetime(["2024-01-01", "2025-01-01"]))
    assert monthly_year_over_year(values).isna().all()


def test_disk_loader_uses_versioned_native_cache_without_fetch(monkeypatch, tmp_path):
    from api import deps
    monkeypatch.setattr(deps, "CACHE_DIR", tmp_path)
    assert deps._parquet_path().name == "fred_macro_native_cpi_v1.parquet"
    frame = macro_frame()
    frame.to_parquet(deps._parquet_path())
    pd.testing.assert_series_equal(deps._load_from_disk()["cpi"], frame["cpi"], check_freq=False)


def test_fred_cache_writes_native_cpi_to_new_namespace(monkeypatch, tmp_path):
    from src import data_sources
    frame = macro_frame()
    monkeypatch.setattr(data_sources, "fetch_fred", lambda *args: frame)
    result = data_sources.get_fred_cached({"cpi": "CPIAUCSL"}, "unused", str(tmp_path))
    assert (tmp_path / "fred_macro_native_cpi_v1.parquet").exists()
    assert not (tmp_path / "fred_macro.parquet").exists()
    pd.testing.assert_series_equal(result["cpi"], frame["cpi"], check_freq=False)
