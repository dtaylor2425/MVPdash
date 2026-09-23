"""
tests/test_options_flow_payload_features.py

Unit tests for scripts/extract_options_flow_payload_features.py's pure extraction logic against
a synthetic payload shaped like api/services/options_flow_metrics.py::build_symbol_payload's
real output (verified against a live row in Postgres during development). No DB/network.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_options_flow_payload_features import _hhi, _ratio, extract_row


def _synthetic_payload():
    return {
        "dte": {"buckets": [
            {"bucket": "0DTE", "trades": 100, "sharePct": 0.5, "sentiment": 0.1,
             "deltaRatio": 0.05, "signedPremium": 1000.0, "grossPremium": 10000.0},
            {"bucket": "1-7D", "trades": 50, "sharePct": 0.3, "sentiment": -0.2,
             "deltaRatio": -0.1, "signedPremium": -2000.0, "grossPremium": 6000.0},
            {"bucket": "8-30D", "trades": 20, "sharePct": 0.15, "sentiment": 0.0,
             "deltaRatio": 0.0, "signedPremium": 0.0, "grossPremium": 3000.0},
            {"bucket": "31-60D", "trades": 5, "sharePct": 0.05, "sentiment": 0.3,
             "deltaRatio": 0.2, "signedPremium": 300.0, "grossPremium": 1000.0},
            {"bucket": "60D+", "trades": 0, "sharePct": 0.0, "sentiment": None,
             "deltaRatio": None, "signedPremium": 0.0, "grossPremium": 0.0},
        ]},
        "aggression": {"callBought": 300.0, "callSold": 100.0, "putBought": 50.0, "putSold": 150.0},
        "premium": {"top10Concentration": 0.42},
        "largeTrades": [
            {"premium": 1000.0, "aggressor": 1.0, "right": "C", "dte": 0},
            {"premium": 500.0, "aggressor": -1.0, "right": "P", "dte": 5},
            {"premium": 250.0, "aggressor": 1.0, "right": "C", "dte": 0},
        ],
        "intraday": (
            [{"t": f"09:{15*i:02d}", "netPremium": 10.0, "grossPremium": 100.0} for i in range(4)]
            + [{"t": "mid", "netPremium": 5.0, "grossPremium": 50.0} for _ in range(5)]
            + [{"t": f"15:{15*i:02d}", "netPremium": -20.0, "grossPremium": 200.0} for i in range(4)]
        ),
    }


def test_ratio_basic():
    assert _ratio(1.0, 2.0) == 0.5
    assert _ratio(1.0, 0.0) is None
    assert _ratio(None, 2.0) is None
    assert _ratio(1.0, None) is None


def test_hhi():
    assert _hhi([50.0, 50.0]) == 0.5  # two equal shares -> 0.5^2 + 0.5^2
    assert _hhi([100.0]) == 1.0  # single trade -> maximally concentrated
    assert _hhi([]) is None
    assert _hhi([0.0, 0.0]) is None


def test_extract_row_dte_buckets():
    row = extract_row("SPY", "2026-01-01", _synthetic_payload())
    assert row["dte_0dte_sentiment"] == 0.1
    assert row["dte_0dte_share_pct"] == 0.5
    assert row["dte_60d_plus_sentiment"] is None  # no 60D+ trades -> genuinely missing, not 0.0


def test_extract_row_call_put_sentiment():
    row = extract_row("SPY", "2026-01-01", _synthetic_payload())
    # call: (300-100)/(300+100) = 0.5 ; put: (50-150)/(50+150) = -0.5
    assert row["call_sentiment"] == 0.5
    assert row["put_sentiment"] == -0.5


def test_extract_row_large_trades():
    row = extract_row("SPY", "2026-01-01", _synthetic_payload())
    assert row["large_trade_count"] == 3
    assert row["large_trade_gross_premium"] == 1750.0
    # directional: +1000 (call, agg=1) + 500 (put, agg=-1 -> premium*-1*-1=+500) + 250 (call, agg=1)
    assert row["large_trade_directional_premium"] == 1750.0
    assert row["large_trade_sentiment"] == 1.0
    # 0DTE premium share: (1000 + 250) / 1750
    assert abs(row["large_trade_0dte_share"] - (1250.0 / 1750.0)) < 1e-9


def test_extract_row_time_of_day():
    row = extract_row("SPY", "2026-01-01", _synthetic_payload())
    assert row["opening_hour_net_premium"] == 40.0  # 4 buckets * 10.0
    assert row["opening_hour_gross_premium"] == 400.0
    assert row["opening_hour_sentiment"] == 0.1
    assert row["closing_hour_net_premium"] == -80.0  # 4 buckets * -20.0
    assert row["closing_hour_sentiment"] == -0.1
    assert row["midday_net_premium"] == 25.0  # 5 buckets * 5.0


def test_extract_row_no_large_trades_is_none_not_zero():
    payload = _synthetic_payload()
    payload["largeTrades"] = []
    row = extract_row("SPY", "2026-01-01", payload)
    assert row["large_trade_count"] == 0
    assert row["large_trade_sentiment"] is None
    assert row["top25_premium_hhi"] is None


def test_extract_row_short_intraday_series():
    payload = _synthetic_payload()
    payload["intraday"] = [{"t": "x", "netPremium": 1.0, "grossPremium": 10.0}] * 3  # < 8 buckets
    row = extract_row("SPY", "2026-01-01", payload)
    assert row["opening_hour_sentiment"] == 0.1  # all 3 buckets counted as "opening"
    assert row["closing_hour_sentiment"] is None
    assert row["midday_sentiment"] is None
