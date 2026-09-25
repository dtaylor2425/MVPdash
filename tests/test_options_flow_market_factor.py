"""
tests/test_options_flow_market_factor.py

Unit tests for api/services/options_flow_market_factor.py against synthetic data with
hand-computable answers. No DB/network.
"""

import numpy as np
import pandas as pd
import pytest

from api.services.options_flow_market_factor import (
    autocorrelation,
    average_forward_path,
    build_outcome_frame,
    forward_max_drawdown,
    forward_realized_vol,
    market_activity_factor,
    residual_activity_from_frame,
    vix_change,
)


def test_market_activity_factor_median_and_mean():
    df = pd.DataFrame({
        "date": ["d1", "d1", "d1", "d2", "d2"],
        "ticker": ["A", "B", "C", "A", "B"],
        "z": [1.0, 2.0, 9.0, 5.0, 7.0],
    })
    out = market_activity_factor(df, "z", min_tickers=2).set_index("date")
    assert out.loc["d1", "market_activity_median"] == 2.0  # median of 1,2,9
    assert out.loc["d1", "market_activity_mean"] == pytest.approx(4.0)  # mean of 1,2,9
    assert out.loc["d1", "n_tickers"] == 3
    assert out.loc["d2", "market_activity_median"] == 6.0  # median of 5,7
    assert out.loc["d2", "n_tickers"] == 2


def test_market_activity_factor_min_tickers_guard():
    df = pd.DataFrame({"date": ["d1"], "ticker": ["A"], "z": [1.0]})
    out = market_activity_factor(df, "z", min_tickers=3).set_index("date")
    assert pd.isna(out.loc["d1", "market_activity_median"])  # only 1 ticker, below min_tickers=3
    assert out.loc["d1", "n_tickers"] == 1


def test_residual_activity_from_frame():
    df = pd.DataFrame({"date": ["d1", "d1"], "ticker": ["A", "B"], "z": [5.0, 1.0]})
    market_by_date = pd.Series({"d1": 3.0})
    resid = residual_activity_from_frame(df, "z", market_by_date)
    assert list(resid) == [2.0, -2.0]


def test_residual_activity_propagates_missing():
    df = pd.DataFrame({"date": ["d1", "d2"], "ticker": ["A", "A"], "z": [5.0, None]})
    market_by_date = pd.Series({"d1": 3.0})  # d2 missing from the market series
    resid = residual_activity_from_frame(df, "z", market_by_date)
    assert resid.iloc[0] == 2.0
    assert pd.isna(resid.iloc[1])


def _synthetic_prices(n=40, seed=1):
    idx = pd.bdate_range("2026-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    rets[0] = 0.0
    prices = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    return prices


def test_forward_realized_vol_known_value():
    idx = pd.bdate_range("2026-01-01", periods=10)
    # constant +1% daily return -> forward vol should be exactly 0 (no variance)
    prices = pd.Series([100 * (1.01 ** i) for i in range(10)], index=idx)
    out = forward_realized_vol(prices, [idx[0].date()], horizon=5)
    assert out[str(idx[0].date())] == pytest.approx(0.0, abs=1e-9)


def test_forward_realized_vol_missing_when_insufficient_history():
    idx = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.Series([100.0] * 5, index=idx)
    out = forward_realized_vol(prices, [idx[-1].date()], horizon=5)
    assert out[str(idx[-1].date())] is None


def test_forward_max_drawdown_known_path():
    idx = pd.bdate_range("2026-01-01", periods=6)
    # 100 -> 110 -> 90 -> 95 -> 100 -> 105 : max drawdown from T=100 within next 5 sessions
    # peak becomes 110 at k=1, trough 90 at k=2 -> drawdown = 90/110 - 1
    prices = pd.Series([100.0, 110.0, 90.0, 95.0, 100.0, 105.0], index=idx)
    out = forward_max_drawdown(prices, [idx[0].date()], horizon=5)
    assert out[str(idx[0].date())] == pytest.approx(90.0 / 110.0 - 1.0)


def test_forward_max_drawdown_monotonic_up_is_zero():
    idx = pd.bdate_range("2026-01-01", periods=6)
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=idx)
    out = forward_max_drawdown(prices, [idx[0].date()], horizon=5)
    assert out[str(idx[0].date())] == pytest.approx(0.0, abs=1e-9)


def test_vix_change_known_value():
    idx = pd.bdate_range("2026-01-01", periods=10)
    vix = pd.Series(np.linspace(15, 24, 10), index=idx)
    out = vix_change(vix, [idx[0].date()], horizon=5)
    assert out[str(idx[0].date())] == pytest.approx(vix.iloc[5] - vix.iloc[0])


def test_vix_change_missing_beyond_history():
    idx = pd.bdate_range("2026-01-01", periods=5)
    vix = pd.Series([15.0] * 5, index=idx)
    out = vix_change(vix, [idx[-1].date()], horizon=5)
    assert out[str(idx[-1].date())] is None


def test_build_outcome_frame_shape():
    prices = _synthetic_prices(40)
    dates = [prices.index[5].date(), prices.index[10].date()]
    out = build_outcome_frame(prices, dates)
    assert len(out) == 2
    for h in (1, 3, 5, 10, 20):
        assert f"fwd_vol_{h}d" in out.columns
        assert f"fwd_max_drawdown_{h}d" in out.columns
        assert f"vix_change_{h}d" in out.columns  # None throughout since vix=None was passed


def test_autocorrelation_perfect_persistence():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = autocorrelation(s, lag=1)
    assert out["corr"] == pytest.approx(1.0)


def test_autocorrelation_insufficient_data():
    s = pd.Series([1.0, 2.0])
    out = autocorrelation(s, lag=1)
    assert out["corr"] is None


def test_average_forward_path_known_values():
    idx = pd.bdate_range("2026-01-01", periods=10)
    prices = pd.Series([100.0 * (1.01 ** i) for i in range(10)], index=idx)
    # single date at index 0 -> cumulative return at k should be 1.01^k - 1
    out = average_forward_path(prices, [idx[0].date()], max_k=3)
    assert out["nDates"] == 1
    expected = [1.01 ** k - 1 for k in (1, 2, 3)]
    for got, exp in zip(out["cumulativeReturn"]["avgByK"], expected):
        assert got == pytest.approx(exp, rel=1e-6)


def test_average_forward_path_n_by_k_shrinks_near_end_of_history():
    idx = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.Series([100.0] * 5, index=idx)
    out = average_forward_path(prices, [idx[-1].date()], max_k=3)  # no forward sessions exist
    assert out["cumulativeReturn"]["nByK"] == [0, 0, 0]
    assert out["cumulativeReturn"]["avgByK"] == [None, None, None]
