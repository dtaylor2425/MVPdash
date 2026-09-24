from datetime import date
import pandas as pd
import pytest
from src.portfolio_market import historical_share_bars


def test_empty_multi_ticker_alignment_rows_do_not_invent_bars_or_splits():
    frame = pd.DataFrame({"Open": [50, None, 51], "Close": [50, None, 51],
                          "Volume": [200, None, 200], "Stock Splits": [0, None, 2],
                          "Dividends": [0, None, 0]},
                         index=pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23"]))
    bars = historical_share_bars(frame, date(2026, 9, 24))
    assert [b["date"] for b in bars] == ["2026-09-21", "2026-09-23"]
    assert [b["close"] for b in bars] == [100, 51]


@pytest.mark.parametrize("split", [float("nan"), -1])
def test_nonempty_rows_with_invalid_corporate_actions_still_fail(split):
    frame = pd.DataFrame({"Open": [50], "Close": [50], "Volume": [200],
                          "Stock Splits": [split], "Dividends": [0]},
                         index=pd.to_datetime(["2026-09-23"]))
    with pytest.raises(ValueError, match="Invalid corporate-action data"):
        historical_share_bars(frame, date(2026, 9, 24))


def test_split_adjusted_provider_prices_restored_to_historical_shares():
    frame = pd.DataFrame({"Open": [50, 50, 51], "Close": [50, 50, 51],
                          "Volume": [200, 200, 200], "Stock Splits": [0, 2, 0],
                          "Dividends": [.5, 0, 0]}, index=pd.to_datetime(["2026-09-18", "2026-09-21", "2026-09-22"]))
    bars = historical_share_bars(frame, date(2026,9,23))
    assert bars[0]["close"] == 100
    assert bars[0]["volume"] == 100
    assert bars[0]["dividend"] == 1
    assert bars[0]["signal_close"] == 50
    assert bars[0]["signal_volume"] == 200
    assert bars[1]["close"] == 50
    assert bars[1]["split"] == 2


def test_todays_split_normalizes_prior_prices_but_today_not_valued():
    frame = pd.DataFrame({"Open": [50, 50], "Close": [50, 50], "Volume": [200, 10],
                          "Stock Splits": [0, 2], "Dividends": [0, 0]},
                         index=pd.to_datetime(["2026-09-21", "2026-09-22"]))
    bars = historical_share_bars(frame, date(2026,9,22))
    assert len(bars) == 1
    assert bars[0]["close"] == 100


def test_multi_split_reverse_split_normalization():
    frame = pd.DataFrame({"Open": [100, 100, 100], "Close": [100, 100, 100],
                          "Volume": [100, 100, 100], "Stock Splits": [0, 2, .1], "Dividends": [0, 0, 0]},
                         index=pd.to_datetime(["2026-09-18", "2026-09-21", "2026-09-22"]))
    bars = historical_share_bars(frame, date(2026,9,23))
    assert [b["close"] for b in bars] == [20, 10, 100]


def test_real_share_basis_split_does_not_invent_ledger_return():
    from tests.test_portfolio_ledger import fixture
    from src.portfolio_ledger import build_ledger
    c, previous, market = fixture()
    first = build_ledger(c, previous, [previous], market, "2026-09-21", 50)
    frame = pd.DataFrame({"Open": [50, 50], "Close": [50, 50], "Volume": [200, 200],
                          "Stock Splits": [0, 2], "Dividends": [0, 0]},
                         index=pd.to_datetime(["2026-09-18", "2026-09-21"]))
    market["AAA"] = historical_share_bars(frame, date(2026,9,22))
    result = build_ledger(c, {"payload": first}, [previous], market, "2026-09-22", 50)
    assert result["performance"]["series"][-1]["model"] == pytest.approx(.2)
