"""
tests/test_options_flow_regime_features.py

Unit tests for api/services/options_flow_regime_features.py against synthetic price series with
a known, hand-computable answer. No network/DB.
"""

import numpy as np
import pandas as pd
import pytest

from api.services.options_flow_regime_features import (
    chronological_half_split,
    iv_percentile_regime,
    market_wide_activity_factor,
    momentum_regime,
    realized_vol_regime,
    sma_regime,
)


def test_chronological_half_split_even():
    dates = ["2026-01-05", "2026-01-01", "2026-01-03", "2026-01-04"]  # deliberately unsorted
    out = chronological_half_split(dates)
    # sorted order: 01-01, 01-03, 01-04, 01-05 -> first 2 = first_half, last 2 = second_half
    mapping = dict(zip(dates, out))
    assert mapping["2026-01-01"] == "first_half"
    assert mapping["2026-01-03"] == "first_half"
    assert mapping["2026-01-04"] == "second_half"
    assert mapping["2026-01-05"] == "second_half"


def test_sma_regime_known_step_function():
    idx = pd.bdate_range("2026-01-01", periods=10)
    # constant 100 for first 5 sessions then jump to 200 at index 5. At index 6, the trailing
    # 3-session SMA (indices 4,5,6 = 100,200,200 -> mean 166.67) is still below the current
    # price (200), so the regime should read "above".
    prices = pd.Series([100] * 5 + [200] * 5, index=idx)
    dates = [idx[6].date()]
    out = sma_regime(prices, dates, window=3)
    assert out["byDate"][str(dates[0])] == "above"
    assert out["smaByDate"][str(dates[0])] == pytest.approx((100 + 200 + 200) / 3)


def test_sma_regime_insufficient_history_is_none():
    idx = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.Series([100.0] * 5, index=idx)
    out = sma_regime(prices, [idx[0].date()], window=200)
    assert out["byDate"][str(idx[0].date())] is None


def test_realized_vol_regime_median_split():
    idx = pd.bdate_range("2026-01-01", periods=50)
    rng = np.random.default_rng(1)
    # alternate low-vol and high-vol stretches so the median split is meaningful
    rets = np.concatenate([rng.normal(0, 0.001, 25), rng.normal(0, 0.05, 25)])
    prices = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    dates = [idx[30].date(), idx[45].date()]  # both inside the high-vol stretch's rolling window
    out = realized_vol_regime(prices, dates, window=10)
    assert out["medianSplit"] is not None
    assert set(out["byDate"].values()) <= {"high", "low"}


def test_momentum_regime_positive_and_negative():
    idx = pd.bdate_range("2026-01-01", periods=25)
    prices = pd.Series(np.linspace(100, 150, 25), index=idx)  # steadily rising
    out = momentum_regime(prices, [idx[-1].date()], window=20)
    assert out["byDate"][str(idx[-1].date())] == "positive"

    prices_down = pd.Series(np.linspace(150, 100, 25), index=idx)
    out_down = momentum_regime(prices_down, [idx[-1].date()], window=20)
    assert out_down["byDate"][str(idx[-1].date())] == "negative"


def test_iv_percentile_regime_median_split():
    df = pd.DataFrame({"iv_percentile_60d": [10.0, 20.0, 30.0, 70.0, 80.0, 90.0]})
    out = iv_percentile_regime(df)
    median = df["iv_percentile_60d"].median()  # 50.0
    expected = ["low" if v <= median else "high" for v in df["iv_percentile_60d"]]
    assert list(out) == expected


def test_iv_percentile_regime_handles_missing():
    df = pd.DataFrame({"iv_percentile_60d": [10.0, None, 90.0]})
    out = iv_percentile_regime(df)
    assert out.iloc[1] is None


def test_market_wide_activity_factor():
    df = pd.DataFrame({
        "date": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "ticker": ["A", "B", "C", "A", "B", "C"],
        "z": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    out = market_wide_activity_factor(df, "z")
    assert list(out) == [2.0, 2.0, 2.0, 20.0, 20.0, 20.0]  # per-date median broadcast to every row
