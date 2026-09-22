"""
tests/test_options_flow_qa.py  (Phase 1 QA/reporting math -- pure, synthetic data)

No network, no Postgres. Exercises api/services/options_flow_qa.py against hand-built
rows shaped like what scripts/generate_options_flow_qa_report.py assembles from Postgres.

    python tests/test_options_flow_qa.py
    pytest tests/test_options_flow_qa.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_qa as qa  # noqa: E402

D0 = date(2026, 6, 26)


def full_row(ticker="SPY", d=D0, status="success", sentiment=0.1, ratio=0.05, net_dollar=1e8,
            gross=2e8, atm=0.15, skew=0.02, zero_dte=0.3, classified=0.95, greek=0.9, oi=0.8,
            trades=100_000, eligible=99_000, call_bought=5e7, call_sold=4e7, put_bought=3e7, put_sold=3e7,
            dte_gross=None, pct20=50.0, pct60=None, pct126=None, pct252=None,
            theta_req=30, runtime=120.0):
    buckets = dte_gross or [
        {"bucket": "0DTE", "grossPremium": gross * 0.3}, {"bucket": "1-7D", "grossPremium": gross * 0.3},
        {"bucket": "8-30D", "grossPremium": gross * 0.2}, {"bucket": "31-60D", "grossPremium": gross * 0.15},
        {"bucket": "60D+", "grossPremium": gross * 0.05},
    ]
    payload = {
        "mode": "full_flow", "ticker": ticker, "marketDate": d.isoformat(),
        "sentiment": {"value": sentiment}, "premium": {"gross": gross, "netDirectional": sentiment * gross},
        "delta": {"ratio": ratio, "netDollar": net_dollar},
        "iv": {"atm": atm, "iv7d": atm - 0.01, "iv30d": atm, "iv60d": atm + 0.01, "skew25d30d": skew,
              "percentiles": {"20d": pct20, "60d": pct60, "126d": pct126, "252d": pct252}},
        "dte": {"zeroDteShare": zero_dte, "buckets": buckets},
        "quality": {"classifiedPctPremium": classified, "deltaMatchedPctPremium": greek,
                    "oiMatchedPctPremium": oi, "trades": trades, "eligibleTrades": eligible},
        "aggression": {"callBought": call_bought, "callSold": call_sold, "putBought": put_bought, "putSold": put_sold},
    }
    return {"ticker": ticker, "marketDate": d, "status": status, "payload": payload,
            "diagnostics": {"thetaRequests": theta_req, "runtimeSec": runtime}}


def warmup_row(ticker="SPY", d=D0, status="success", atm=0.15, skew=0.02, pct252=80.0, theta_req=17, runtime=40.0):
    payload = {
        "mode": "iv_warmup", "ticker": ticker, "marketDate": d.isoformat(),
        "iv": {"atm": atm, "iv7d": atm - 0.01, "iv30d": atm, "iv60d": atm + 0.01, "skew25d30d": skew,
              "percentiles": {"20d": 60.0, "60d": 70.0, "126d": 75.0, "252d": pct252}},
    }
    return {"ticker": ticker, "marketDate": d, "status": status, "payload": payload,
            "diagnostics": {"thetaRequests": theta_req, "runtimeSec": runtime}}


# --------------------------------------------------------------------------
# describe / mode detection
# --------------------------------------------------------------------------

def test_describe_basic_and_empty():
    d = qa.describe([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert d["min"] == 1 and d["max"] == 10 and d["median"] == 5.5 and d["n"] == 10
    assert d["mean"] == 5.5
    empty = qa.describe([None, None])
    assert empty["n"] == 0 and empty["median"] is None and empty["mean"] is None


def test_describe_drops_none_and_nan():
    d = qa.describe([1.0, None, 3.0, float("nan"), 5.0])
    assert d["n"] == 3 and d["mean"] == 3.0


def test_mode_detection():
    assert qa.is_full_flow(full_row()) and not qa.is_iv_warmup(full_row())
    assert qa.is_iv_warmup(warmup_row()) and not qa.is_full_flow(warmup_row())


# --------------------------------------------------------------------------
# per-ticker summary (item 5)
# --------------------------------------------------------------------------

def test_summarize_ticker_counts_and_missing():
    full_rows = [qa.extract_full_flow_fields(full_row(d=date(2026, 6, 26 + i), pct252=None if i < 3 else 80.0))
                for i in range(5)]
    warmup_rows = [qa.extract_iv_warmup_fields(warmup_row(d=date(2026, 5, i + 1))) for i in range(3)]
    s = qa.summarize_ticker("SPY", requested_full=5, full_rows=full_rows, full_failed=1,
                            requested_warmup=3, warmup_rows=warmup_rows, warmup_failed=0)
    assert s["fullFlowRequested"] == 5 and s["fullFlowSuccess"] == 5 and s["fullFlowFailed"] == 1
    assert s["warmupSuccess"] == 3 and s["warmupFailed"] == 0
    assert s["totalThetaRequests"] == 5 * 30 + 3 * 17
    assert s["medianRequestsPerTickerDay"] == 30.0        # full-flow (30) and warmup (17) requests differ; median of the pooled list
    assert s["missingPct252d"] == 3           # 3 of the 5 full-flow rows have no 252D percentile
    assert s["missingAtmIv"] == 0
    assert abs(s["missingPct252dPct"] - 3 / 8) < 1e-9      # denominator is ALL rows (full+warmup), per spec


def test_summarize_ticker_handles_all_failed():
    s = qa.summarize_ticker("GLD", requested_full=10, full_rows=[], full_failed=10,
                            requested_warmup=10, warmup_rows=[], warmup_failed=10)
    assert s["fullFlowSuccess"] == 0 and s["fullFlowFailed"] == 10
    assert s["medianGrossPremium"] is None and s["totalThetaRequests"] == 0


# --------------------------------------------------------------------------
# distribution QA (item 6) + extreme-observation flags
# --------------------------------------------------------------------------

def test_distribution_qa_covers_every_required_metric():
    rows = [qa.extract_full_flow_fields(full_row(sentiment=0.1 * i, atm=0.1 + 0.01 * i)) for i in range(20)]
    dist = qa.distribution_qa(rows)
    expected = {"net_trade_sentiment", "delta_imbalance_ratio", "net_dollar_delta_imbalance", "gross_premium",
               "atm_iv", "put_skew_25d", "zero_dte_share", "classification_coverage",
               "greek_match_coverage", "oi_match_coverage"}
    assert set(dist) == expected
    for label, d in dist.items():
        assert d["n"] == 20, label
        assert d["min"] <= d["median"] <= d["max"], label


def test_extreme_observations_are_flagged_not_dropped():
    rows = [qa.extract_full_flow_fields(full_row(d=date(2026, 6, 1 + i), sentiment=0.05)) for i in range(20)]
    rows.append(qa.extract_full_flow_fields(full_row(d=date(2026, 7, 20), sentiment=0.98)))  # one wild outlier
    flagged = qa.flag_extreme_observations(rows, "netTradeSentiment")
    assert len(flagged) == 1 and flagged[0]["value"] == 0.98
    assert len(rows) == 21          # flagging never removes rows


def test_too_few_observations_flags_nothing():
    rows = [qa.extract_full_flow_fields(full_row(sentiment=v)) for v in (0.1, 0.9)]
    assert qa.flag_extreme_observations(rows, "netTradeSentiment") == []


# --------------------------------------------------------------------------
# reconciliation (item 7)
# --------------------------------------------------------------------------

def test_dte_bucket_reconciliation_is_exact_when_buckets_partition_gross():
    rows = [qa.extract_full_flow_fields(full_row(gross=1_000_000.0))]
    r = qa.reconciliation_errors(rows)
    assert abs(r["dteBuckets"]["medianAbsDiff"]) < 1e-6
    assert abs(r["dteBuckets"]["worstAbsDiff"]) < 1e-6


def test_dte_bucket_reconciliation_catches_a_real_gap():
    bad = full_row(gross=1_000_000.0, dte_gross=[{"bucket": "0DTE", "grossPremium": 400_000.0}])  # buckets don't sum to gross
    rows = [qa.extract_full_flow_fields(bad)]
    r = qa.reconciliation_errors(rows)
    assert abs(r["dteBuckets"]["worstAbsDiff"] - 600_000.0) < 1e-6


def test_aggression_reconciliation_gap_is_expected_for_inside_spread_trades():
    # call_bought+call_sold+put_bought+put_sold sums to only 90% of gross premium here,
    # representing inside-spread (fractional aggressor) trades -- documented as expected.
    row = full_row(gross=1_000_000.0, call_bought=300_000.0, call_sold=300_000.0,
                   put_bought=200_000.0, put_sold=100_000.0)   # sums to 900,000
    r = qa.reconciliation_errors([qa.extract_full_flow_fields(row)])
    assert abs(r["aggression"]["medianAbsDiff"] - 100_000.0) < 1e-6
    assert abs(r["aggression"]["medianRelDiff"] - 0.10) < 1e-9
    assert "expected" in r["aggression"]["note"]
    assert "real bug" in r["dteBuckets"]["note"]


def test_reconciliation_skips_rows_with_no_gross_premium():
    row = full_row(gross=0.0)
    row["payload"]["premium"]["gross"] = None
    r = qa.reconciliation_errors([qa.extract_full_flow_fields(row)])
    assert r["aggression"]["n"] == 0 and r["dteBuckets"]["n"] == 0


# --------------------------------------------------------------------------
# cross-ETF comparability (item 8)
# --------------------------------------------------------------------------

def test_cross_etf_comparability_and_verdict_flags_incomparable_raw_magnitudes():
    by_ticker = {
        "SPY": [qa.extract_full_flow_fields(full_row(ticker="SPY", gross=2e9, net_dollar=5e9, trades=1_500_000))],
        "GLD": [qa.extract_full_flow_fields(full_row(ticker="GLD", gross=2e7, net_dollar=1e7, trades=5_000))],
    }
    comp = qa.cross_etf_comparability(by_ticker)
    assert comp[0]["ticker"] == "SPY"        # sorted by descending premium
    assert comp[0]["medianGrossPremium"] > comp[1]["medianGrossPremium"]
    comparable, note = qa.comparability_verdict(comp)
    assert comparable is False
    assert "net_trade_sentiment" in note and "delta_imbalance_ratio" in note


def test_comparability_verdict_with_insufficient_data():
    comparable, note = qa.comparability_verdict([{"ticker": "SPY", "medianGrossPremium": None,
                                                   "medianNetDollarDelta": None, "medianAbsNetDollarDelta": None,
                                                   "medianTrades": None}])
    assert comparable is True and "not enough" in note


# --------------------------------------------------------------------------
# research dataset rows (item 9)
# --------------------------------------------------------------------------

def test_research_row_is_point_in_time_and_has_no_outcome_fields():
    row = full_row(d=date(2026, 7, 1), sentiment=0.22)
    r = qa.research_row(row, methodology_version="1.0.0")
    assert r["date"] == date(2026, 7, 1) and r["ticker"] == "SPY" and r["methodology_version"] == "1.0.0"
    assert r["net_trade_sentiment"] == 0.22
    for forbidden in ("forward_return", "future_return", "return_1d", "return_5d", "label", "target"):
        assert forbidden not in r
    for k in ("atm_iv", "iv_percentile_20d", "iv_percentile_60d", "iv_percentile_126d", "iv_percentile_252d",
              "classification_coverage", "greek_match_coverage", "oi_match_coverage", "trade_count",
              "call_bought_premium", "put_sold_premium", "dte_0dte_gross_premium"):
        assert k in r, k


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if failed else 0)
