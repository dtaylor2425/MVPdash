"""
tests/test_options_flow_forward_returns.py

Unit tests for api/services/options_flow_forward_returns.py -- pure computation against a
synthetic price series, no network/DB.
"""

import pandas as pd
import pytest

from api.services.options_flow_forward_returns import (
    HORIZONS,
    compute_forward_returns,
    verify_no_future_prices_in_features,
)


def _synthetic_prices(n_sessions=30):
    idx = pd.bdate_range("2026-01-02", periods=n_sessions)
    # SPY: rises 1% every session (so ret_Nd is deterministic and easy to check)
    spy = [100.0 * (1.01 ** i) for i in range(n_sessions)]
    return pd.DataFrame({"SPY": spy}, index=idx)


def test_compute_forward_returns_known_values():
    prices = _synthetic_prices(30)
    dates = [prices.index[0].date(), prices.index[5].date()]
    out = compute_forward_returns(prices, dates, ["SPY"])
    assert len(out) == 2
    row0 = out[out["date"] == dates[0]].iloc[0]
    # 1.01^1 - 1 == ret_1d exactly, by construction
    assert row0["ret_1d"] == pytest.approx(0.01, rel=1e-9)
    assert row0["ret_5d"] == pytest.approx(1.01 ** 5 - 1, rel=1e-9)
    assert row0["ret_20d"] == pytest.approx(1.01 ** 20 - 1, rel=1e-9)


def test_compute_forward_returns_missing_future_session_is_none_not_zero():
    prices = _synthetic_prices(10)  # only 10 sessions total
    last_date = prices.index[-1].date()
    out = compute_forward_returns(prices, [last_date], ["SPY"])
    row = out.iloc[0]
    # no session at all exists after the last one -- every horizon must be None, never 0.0
    for h in HORIZONS:
        assert row[f"ret_{h}d"] is None


def test_compute_forward_returns_date_not_in_index():
    prices = _synthetic_prices(10)
    missing_date = pd.Timestamp("2026-01-01").date()  # a weekend, not a trading session
    out = compute_forward_returns(prices, [missing_date], ["SPY"])
    row = out.iloc[0]
    assert row["close"] is None
    for h in HORIZONS:
        assert row[f"ret_{h}d"] is None


def test_compute_forward_returns_ticker_not_in_prices():
    prices = _synthetic_prices(10)
    out = compute_forward_returns(prices, [prices.index[0].date()], ["NOTATICKER"])
    row = out.iloc[0]
    assert row["close"] is None
    for h in HORIZONS:
        assert row[f"ret_{h}d"] is None


def test_verify_no_future_prices_true_when_features_precede_price_history():
    result = verify_no_future_prices_in_features(["2026-01-01", "2026-01-05"], "2026-01-10")
    assert result["featuresPrecedeOrEqualPriceHistory"] is True


def test_verify_no_future_prices_false_when_a_feature_date_is_after():
    result = verify_no_future_prices_in_features(["2026-01-01", "2026-01-15"], "2026-01-10")
    assert result["featuresPrecedeOrEqualPriceHistory"] is False


def test_verify_no_future_prices_accepts_mixed_date_and_string_types():
    import datetime
    result = verify_no_future_prices_in_features(
        [datetime.date(2026, 1, 1), datetime.date(2026, 1, 5)], datetime.date(2026, 1, 10)
    )
    assert result["featuresPrecedeOrEqualPriceHistory"] is True
