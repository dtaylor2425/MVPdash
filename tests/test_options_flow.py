"""
tests/test_options_flow.py  (Macro Options Flow -- backend)

No network, no ThetaData, no Postgres. Frames are synthetic but shaped like
ThetaData's (trade_timestamp/price/size/bid/ask/strike/right, and
timestamp/delta/implied_vol/underlying_price/...), so the real
normalize -> classify -> Greek-join -> aggregate path runs.

What is NOT covered here: the SQL itself. The "latest snapshot" tests check
the pure selection/assembly code and that every read query filters to
published runs; DISTINCT ON behaviour still needs one manual pass against
Railway Postgres (there is no local Postgres -- see tests/test_auth.py).

    python tests/test_options_flow.py
    pytest tests/test_options_flow.py
"""

from __future__ import annotations

import contextlib
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_metrics as m  # noqa: E402
from api.services import options_flow_store as store  # noqa: E402
from api.services.options_flow_calendar import compute_data_status  # noqa: E402
from api.services.options_flow_config import parse_universe  # noqa: E402

D = date(2026, 9, 21)  # a Monday
NY = "America/New_York"


# --------------------------------------------------------------------------
# synthetic ThetaData-shaped frames
# --------------------------------------------------------------------------

def ts(h, mi, s=0):
    return pd.Timestamp(2026, 9, 21, h, mi, s, tz=NY)


def raw_trades(rows, expiration="2026-09-21"):
    """rows: (time, right, strike, price, size, bid, ask)"""
    return pd.DataFrame({
        "trade_timestamp": [r[0] for r in rows],
        "right": [r[1] for r in rows],
        "strike": [r[2] for r in rows],
        "price": [r[3] for r in rows],
        "size": [r[4] for r in rows],
        "bid": [r[5] for r in rows],
        "ask": [r[6] for r in rows],
        "expiration": expiration,
    })


def raw_greeks(rows, expiration="2026-09-21"):
    """rows: (time, right, strike, delta, iv, underlying)"""
    return pd.DataFrame({
        "timestamp": [r[0] for r in rows],
        "right": [r[1] for r in rows],
        "strike": [r[2] for r in rows],
        "delta": [r[3] for r in rows],
        "implied_vol": [r[4] for r in rows],
        "underlying_price": [r[5] for r in rows],
        "bid": 1.0, "ask": 1.05,
        "expiration": expiration,
    })


def one_trade(right, price, bid=1.00, ask=1.20, size=10, delta=None, exp="2026-09-21"):
    """Run a single trade through the full pipeline; return the enriched row."""
    d = delta if delta is not None else (0.50 if right == "CALL" else -0.40)
    trades, _ = m.process_expiration(
        raw_trades([(ts(10, 30), right, 500.0, price, size, bid, ask)], exp),
        raw_greeks([(ts(10, 29), right, 500.0, d, 0.20, 500.0)], exp),
        exp, D,
    )
    return trades.iloc[0]


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------
# directional premium: bought/sold x call/put
# --------------------------------------------------------------------------

def test_ask_side_call_is_bullish():
    r = one_trade("CALL", price=1.20)          # at ask
    assert r.aggressor == 1.0
    assert r.premium == 1.20 * 10 * 100
    assert r.dir_premium == r.premium > 0


def test_bid_side_call_is_bearish():
    r = one_trade("CALL", price=1.00)          # at bid
    assert r.aggressor == -1.0
    assert r.dir_premium == -r.premium < 0


def test_ask_side_put_is_bearish():
    r = one_trade("PUT", price=1.20)
    assert r.aggressor == 1.0
    assert r.dir_premium == -r.premium < 0


def test_bid_side_put_is_bullish():
    r = one_trade("PUT", price=1.00)
    assert r.aggressor == -1.0
    assert r.dir_premium == r.premium > 0      # put sold = bullish


def test_through_the_quote_prices_saturate():
    assert one_trade("CALL", price=1.35).aggressor == 1.0     # above ask
    assert one_trade("CALL", price=0.90).aggressor == -1.0    # below bid


# --------------------------------------------------------------------------
# delta imbalance
# --------------------------------------------------------------------------

def test_call_buyer_delta_positive():
    r = one_trade("CALL", price=1.20, delta=0.50)
    assert r.signed_delta_contracts == 1.0 * 0.50 * 10 * 100 == 500.0
    assert r.signed_dollar_delta == 500.0 * 500.0


def test_call_seller_delta_negative():
    assert one_trade("CALL", price=1.00, delta=0.50).signed_delta_contracts == -500.0


def test_put_buyer_delta_negative_and_put_delta_not_reinverted():
    r = one_trade("PUT", price=1.20, delta=-0.40)
    assert r.delta == -0.40                       # left exactly as Theta reports it
    assert r.signed_delta_contracts == 1.0 * -0.40 * 10 * 100 == -400.0
    assert r.signed_dollar_delta < 0


def test_put_seller_delta_positive():
    r = one_trade("PUT", price=1.00, delta=-0.40)
    assert r.signed_delta_contracts == -1.0 * -0.40 * 10 * 100 == 400.0
    assert r.signed_dollar_delta > 0


def test_delta_block_ratio():
    trades, _ = m.process_expiration(
        raw_trades([
            (ts(10, 30), "CALL", 500.0, 1.20, 10, 1.0, 1.2),   # +500 contracts
            (ts(10, 31), "PUT", 500.0, 1.20, 10, 1.0, 1.2),    # -400
        ]),
        raw_greeks([
            (ts(10, 29), "CALL", 500.0, 0.50, 0.2, 500.0),
            (ts(10, 29), "PUT", 500.0, -0.40, 0.2, 500.0),
        ]), "2026-09-21", D)
    b = m.delta_block(trades)
    assert b["netContracts"] == 100.0
    assert b["grossDollar"] == 900.0 * 500.0
    assert abs(b["ratio"] - (100.0 / 900.0)) < 1e-12


# --------------------------------------------------------------------------
# aggressor score inside the spread
# --------------------------------------------------------------------------

def test_inside_spread_aggressor_score():
    score, valid = m.aggressor_score([1.15, 1.05, 1.10, 1.18], [1.0] * 4, [1.2] * 4)
    assert valid.all()
    assert np.allclose(score, [0.5, -0.5, 0.0, 0.8])


def test_inside_spread_score_scales_directional_premium():
    r = one_trade("CALL", price=1.15)             # halfway between mid and ask
    assert abs(r.aggressor - 0.5) < 1e-12
    assert abs(r.dir_premium - 0.5 * r.premium) < 1e-9


# --------------------------------------------------------------------------
# invalid / crossed quote filtering
# --------------------------------------------------------------------------

def test_invalid_quotes_and_sizes_are_excluded():
    bad = [
        (ts(10, 30), "CALL", 500.0, 1.10, 10, 1.00, 0.0),      # ask <= 0
        (ts(10, 30), "CALL", 500.0, 1.10, 10, -0.05, 1.20),    # bid < 0
        (ts(10, 30), "CALL", 500.0, 1.10, 10, 1.30, 1.20),     # crossed
        (ts(10, 30), "CALL", 500.0, 0.0, 10, 1.00, 1.20),      # price <= 0
        (ts(10, 30), "CALL", 500.0, 1.10, 0, 1.00, 1.20),      # size <= 0
        (ts(10, 30), "CALL", 500.0, 1.10, 10, np.nan, 1.20),   # missing bid
    ]
    good = [(ts(10, 31), "CALL", 500.0, 1.20, 10, 1.00, 1.20)]
    trades, _ = m.process_expiration(raw_trades(bad + good), raw_greeks([]), "2026-09-21", D)
    assert list(trades["valid"]) == [False] * 6 + [True]
    assert (trades.loc[~trades["valid"], "dir_premium"] == 0).all()
    assert (trades.loc[~trades["valid"], "aggressor"] == 0).all()

    t = m.flow_totals(trades)
    assert t["quality"]["trades"] == 7 and t["quality"]["eligibleTrades"] == 1
    assert t["premium"]["gross"] == 1.20 * 10 * 100          # only the eligible trade
    # premium coverage counts unclassified-but-priced trades in the denominator:
    # crossed / bid<0 / ask<=0 / size>0 rows carry premium; price<=0 & size<=0 rows carry none.
    assert 0 < t["quality"]["classifiedPctPremium"] < 1
    assert abs(t["quality"]["classifiedPctTrades"] - 1 / 7) < 1e-12


def test_locked_market_does_not_divide_by_zero():
    score, valid = m.aggressor_score([1.10, 1.00, 1.05], [1.10, 1.10, 1.05], [1.10, 1.10, 1.05])
    assert valid.all() and np.isfinite(score).all()
    assert score[0] == 1.0 and score[1] == -1.0


# --------------------------------------------------------------------------
# sentiment normalisation + labels + low confidence
# --------------------------------------------------------------------------

def test_sentiment_normalised_to_unit_interval():
    assert m.sentiment_value(500.0, 1000.0) == 0.5
    assert m.sentiment_value(-1000.0, 1000.0) == -1.0
    assert m.sentiment_value(1e9, 1000.0) == 1.0          # clamped, never > 1
    assert m.sentiment_value(0.0, 0.0) is None            # no premium -> undefined, not 0


def test_sentiment_pipeline_all_ask_calls_is_plus_one():
    rows = [(ts(10, 30 + i), "CALL", 500.0, 1.20, 50, 1.00, 1.20) for i in range(30)]
    trades, _ = m.process_expiration(raw_trades(rows), raw_greeks([]), "2026-09-21", D)
    tot = m.flow_totals(trades, {"minEligiblePremium": 1000, "minEligibleTrades": 5})
    assert tot["sentiment"]["value"] == 1.0
    assert tot["sentiment"]["label"] == m.LABEL_BULLISH


def test_label_thresholds():
    big = dict(eligible_premium=1e9, eligible_trades=10 ** 6, premium_coverage=1.0)
    lab = lambda v: m.sentiment_label(v, **big)[0]  # noqa: E731
    assert lab(0.21) == m.LABEL_BULLISH
    assert lab(0.20) == m.LABEL_LEAN_BULLISH        # > +0.20 is strict
    assert lab(0.08) == m.LABEL_LEAN_BULLISH
    assert lab(0.0799) == m.LABEL_NEUTRAL
    assert lab(0.0) == m.LABEL_NEUTRAL
    assert lab(-0.0799) == m.LABEL_NEUTRAL
    assert lab(-0.08) == m.LABEL_LEAN_BEARISH
    assert lab(-0.20) == m.LABEL_LEAN_BEARISH
    assert lab(-0.2001) == m.LABEL_BEARISH


def test_low_confidence_on_thin_premium_trades_or_coverage():
    ok = dict(eligible_premium=5e6, eligible_trades=500, premium_coverage=0.95)
    assert m.sentiment_label(0.9, **ok)[0] == m.LABEL_BULLISH

    for override in ({"eligible_premium": 10_000.0}, {"eligible_trades": 3}, {"premium_coverage": 0.2}):
        label, reasons = m.sentiment_label(0.9, **{**ok, **override})
        assert label == m.LABEL_LOW_CONFIDENCE and reasons, override

    assert m.sentiment_label(None)[0] == m.LABEL_LOW_CONFIDENCE
    # thresholds are configurable
    assert m.sentiment_label(0.9, **{**ok, "eligible_premium": 10_000.0}, cfg={"minEligiblePremium": 1000})[0] == m.LABEL_BULLISH


def test_flow_totals_low_confidence_keeps_numeric_value():
    trades, _ = m.process_expiration(
        raw_trades([(ts(10, 30), "CALL", 500.0, 1.20, 1, 1.0, 1.2)]), raw_greeks([]), "2026-09-21", D)
    t = m.flow_totals(trades)      # default thresholds: 1 trade, $120 of premium
    assert t["sentiment"]["label"] == m.LABEL_LOW_CONFIDENCE
    assert t["sentiment"]["lowConfidence"] is True
    assert t["sentiment"]["value"] == 1.0


def test_empty_day_is_low_confidence_not_a_crash():
    trades, snap = m.process_expiration(pd.DataFrame(), pd.DataFrame(), "2026-09-21", D)
    p = m.build_symbol_payload("SPY", "INDEX", D, trades, snap)
    assert p["sentiment"]["label"] == m.LABEL_LOW_CONFIDENCE
    assert p["sentiment"]["value"] is None and p["intraday"] == [] and p["largeTrades"] == []


# --------------------------------------------------------------------------
# Greek join
# --------------------------------------------------------------------------

def test_greek_join_uses_prior_observation_and_tolerance():
    trades = raw_trades([
        (ts(10, 30), "CALL", 500.0, 1.2, 10, 1.0, 1.2),    # prior obs 10:28 (delta .40) exists
        (ts(10, 31), "CALL", 501.0, 1.2, 10, 1.0, 1.2),    # different strike: no obs at all
        (ts(12, 0), "CALL", 500.0, 1.2, 10, 1.0, 1.2),     # nearest obs is 90 min away -> beyond tolerance
    ])
    greeks = raw_greeks([
        (ts(10, 27), "CALL", 500.0, 0.30, 0.2, 500.0),
        (ts(10, 28), "CALL", 500.0, 0.40, 0.2, 500.0),
        (ts(10, 32), "CALL", 500.0, 0.99, 0.2, 500.0),     # FUTURE obs must not be used for the 10:30 trade
    ])
    out, _ = m.process_expiration(trades, greeks, "2026-09-21", D, tolerance_min=15)
    out = out.sort_values("ts").reset_index(drop=True)
    assert abs(out.loc[0, "delta"] - 0.40) < 1e-6
    assert np.isnan(out.loc[1, "delta"]) and np.isnan(out.loc[1, "signed_dollar_delta"])
    assert np.isnan(out.loc[2, "delta"])
    # sentiment still counts the trade whose delta could not be matched
    assert m.flow_totals(out, {"minEligiblePremium": 0, "minEligibleTrades": 0})["quality"]["eligibleTrades"] == 3
    assert m.delta_block(out)["matchedTrades"] == 1


def test_greek_join_falls_back_to_nearest_when_no_prior():
    trades = raw_trades([(ts(9, 45), "PUT", 500.0, 1.2, 10, 1.0, 1.2)])
    greeks = raw_greeks([(ts(9, 47), "PUT", 500.0, -0.35, 0.2, 500.0)])   # only a later obs, 2 min away
    out, _ = m.process_expiration(trades, greeks, "2026-09-21", D, tolerance_min=15)
    assert abs(out.iloc[0]["delta"] + 0.35) < 1e-6


# --------------------------------------------------------------------------
# aggregates
# --------------------------------------------------------------------------

def _mixed_trades():
    rows = [
        (ts(9, 35), "CALL", 500.0, 1.20, 100, 1.0, 1.2),   # call bought   $12,000
        (ts(9, 50), "CALL", 500.0, 1.00, 50, 1.0, 1.2),    # call sold     $5,000
        (ts(10, 5), "PUT", 495.0, 2.00, 20, 1.8, 2.0),     # put bought    $4,000
        (ts(10, 20), "PUT", 495.0, 1.80, 10, 1.8, 2.0),    # put sold      $1,800
    ]
    g = [(ts(9, 30), r, 500.0 if r == "CALL" else 495.0, 0.5 if r == "CALL" else -0.4, 0.2, 500.0)
         for r in ("CALL", "PUT")]
    out, _ = m.process_expiration(raw_trades(rows), raw_greeks(g), "2026-09-21", D)
    return out


def test_aggression_block_and_totals():
    t = _mixed_trades()
    a = m.aggression_block(t)
    assert a == {"callBought": 12000.0, "callSold": 5000.0, "putBought": 4000.0, "putSold": 1800.0}
    tot = m.flow_totals(t, {"minEligiblePremium": 0, "minEligibleTrades": 0})
    p = tot["premium"]
    assert p["gross"] == 22800.0
    assert p["bullish"] == 12000.0 + 1800.0
    assert p["bearish"] == 5000.0 + 4000.0
    assert p["netDirectional"] == 4800.0
    assert abs(tot["sentiment"]["value"] - 4800.0 / 22800.0) < 1e-12
    assert abs(p["top10Concentration"] - 1.0) < 1e-12        # only 4 trades: top-10 is everything


def test_top10_concentration():
    rows = [(ts(10, 30 + i % 20, i), "CALL", 500.0, 1.2, 100 if i == 0 else 1, 1.0, 1.2) for i in range(30)]
    trades, _ = m.process_expiration(raw_trades(rows), raw_greeks([]), "2026-09-21", D)
    c = m.flow_totals(trades)["premium"]["top10Concentration"]
    big, small = 1.2 * 100 * 100, 1.2 * 1 * 100
    assert abs(c - (big + 9 * small) / (big + 29 * small)) < 1e-12


def test_dte_buckets_and_zero_dte_share():
    def leg(exp):
        t, _ = m.process_expiration(
            raw_trades([(ts(10, 30), "CALL", 500.0, 1.2, 10, 1.0, 1.2)], exp), raw_greeks([], exp), exp, D)
        return t
    trades = pd.concat([leg("2026-09-21"), leg("2026-09-25"), leg("2026-10-16"), leg("2026-11-20"), leg("2026-12-18")],
                       ignore_index=True)
    blk = m.dte_block(trades)
    by = {b["bucket"]: b for b in blk["buckets"]}
    assert [b["bucket"] for b in blk["buckets"]] == ["0DTE", "1-7D", "8-30D", "31-60D", "60D+"]
    assert by["0DTE"]["trades"] == by["1-7D"]["trades"] == by["8-30D"]["trades"] == 1
    assert by["31-60D"]["trades"] == 1 and by["60D+"]["trades"] == 1     # 60 DTE and 88 DTE
    assert abs(blk["zeroDteShare"] - 0.2) < 1e-12


def test_open_interest_context():
    t = _mixed_trades()
    oi = m.normalize_open_interest(pd.DataFrame({
        "expiration": ["2026-09-21"] * 2, "strike": [500.0, 495.0], "right": ["CALL", "PUT"],
        "open_interest": [1000, 200],
    }))
    t = m.attach_open_interest(t, oi)
    b = m.oi_block(t)
    assert b["matchedPctPremium"] == 1.0
    assert b["oiContracts"] == 1200.0 and b["tradedContracts"] == 180.0
    assert abs(b["contractsToOi"] - 180.0 / 1200.0) < 1e-12
    assert m.oi_block(m.attach_open_interest(_mixed_trades(), None))["contractsToOi"] is None


def test_intraday_series_cumulates_and_stops_at_last_trade():
    s = m.intraday_series(_mixed_trades(), bucket_min=15, rolling_buckets=2)
    assert s[0]["t"].startswith("2026-09-21T09:30")
    assert s[-1]["t"].startswith("2026-09-21T10:15")             # last trade 10:20 -> bucket 10:15
    assert [x["cumNetPremium"] for x in s][-1] == 4800.0
    assert abs(sum(x["netPremium"] for x in s) - 4800.0) < 1e-9
    assert all(-1.0 <= x["rollingSentiment"] <= 1.0 for x in s if x["rollingSentiment"] is not None)
    cum = [x["cumDollarDelta"] for x in s]
    assert cum[-1] == sum(_mixed_trades()["signed_dollar_delta"].fillna(0))


def test_large_trades_sorted_and_shaped():
    lt = m.large_trades(_mixed_trades(), "SPY", n=2)
    assert [x["premium"] for x in lt] == [12000.0, 5000.0]
    first = lt[0]
    for k in ("time", "ticker", "expiration", "strike", "right", "size", "price", "premium", "bid",
              "ask", "aggressorLabel", "delta", "iv", "dte"):
        assert k in first
    assert first["right"] == "C" and first["aggressorLabel"] == "AT ASK" and first["dte"] == 0
    assert lt[1]["aggressorLabel"] == "AT BID"


# --------------------------------------------------------------------------
# IV analytics
# --------------------------------------------------------------------------

def _surface(iv_fn, expirations=((7, "2026-09-28"), (30, "2026-10-21"), (60, "2026-11-20")), spot=500.0):
    frames = []
    for dte, exp in expirations:
        rows = []
        for k in range(-10, 11):
            strike = spot + k * 5.0
            for right in ("CALL", "PUT"):
                # crude delta: monotone in strike, |delta| spans ~0.05..0.95
                d = float(np.clip(0.5 - k * 0.045, 0.02, 0.98))
                delta = d if right == "CALL" else d - 1.0
                rows.append((ts(15, 55), right, strike, delta, iv_fn(dte, strike, right), spot))
        g = m.normalize_greeks(raw_greeks(rows, exp), exp, D)
        frames.append(m.latest_iv_snapshot(g))
    return pd.concat(frames, ignore_index=True)


def test_flat_surface_atm_term_and_spreads():
    snap = _surface(lambda dte, k, r: 0.20)
    iv = m.iv_block(snap, 500.0)
    assert abs(iv["atm"] - 0.20) < 1e-9
    assert abs(iv["iv7d"] - 0.20) < 1e-9 and abs(iv["iv60d"] - 0.20) < 1e-9
    assert abs(iv["spread7v30"]) < 1e-9 and abs(iv["spread30v60"]) < 1e-9
    assert abs(iv["skew25d30d"]) < 1e-9                    # flat smile -> no skew
    assert len(iv["termStructure"]) == 3


def test_term_structure_spreads_and_put_skew():
    def f(dte, k, r):
        base = 0.30 - 0.0025 * dte if dte <= 30 else 0.225 - 0.0005 * (dte - 30)   # inverted front
        return base + (0.05 if r == "PUT" else 0.0)                                   # puts richer
    snap = _surface(f)
    iv = m.iv_block(snap, 500.0)
    # the ATM strike averages call & put IV
    assert abs(iv["iv7d"] - (f(7, 500, "CALL") + f(7, 500, "PUT")) / 2) < 1e-6
    assert iv["spread7v30"] > 0 > iv["spread30v60"] or iv["spread7v30"] > 0        # front > back (inverted)
    assert abs(iv["skew25d30d"] - 0.05) < 1e-6              # 25d put IV - 25d call IV


def test_iv_changes_and_percentile_from_history():
    snap = _surface(lambda dte, k, r: 0.20)
    prior = [("2026-09-18", 0.19)] + [("2026-09-1%d" % i, 0.18) for i in range(0, 4)] + \
            [("d%d" % i, 0.10 + 0.002 * i) for i in range(30)]     # 35 prior obs, all < 0.20
    iv = m.iv_block(snap, 500.0, {"priorAtmIv": prior}, {"ivPercentileMinObs": 20})
    assert abs(iv["atmChange1d"] - 0.01) < 1e-9
    assert abs(iv["atmChange5d"] - 0.02) < 1e-9            # 5th prior obs = 0.18
    assert iv["percentileObs"] == 35 and iv["percentile"] == 100.0   # 0.20 above every prior value
    thin = m.iv_block(snap, 500.0, {"priorAtmIv": prior[:6]}, {"ivPercentileMinObs": 20})
    assert thin["percentile"] is None and thin["atmChange1d"] is not None and thin["atmChange5d"] is not None
    no_hist = m.iv_block(snap, 500.0)
    assert no_hist["atmChange1d"] is None and no_hist["atmChange5d"] is None and no_hist["percentile"] is None


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------

def test_payload_is_json_safe_and_complete():
    import json
    trades = _mixed_trades()
    snap = _surface(lambda dte, k, r: 0.2)
    p = m.build_symbol_payload("SPY", "INDEX", D, trades, snap, cfg={"minEligiblePremium": 0, "minEligibleTrades": 0},
                               fetched_at=datetime(2026, 9, 21, 19, 0, tzinfo=timezone.utc))
    json.dumps(p, allow_nan=False)          # raises on NaN/inf/numpy leakage
    for k in ("ticker", "group", "marketDate", "asOf", "spot", "sentiment", "premium", "delta", "iv", "dte",
              "aggression", "openInterest", "quality", "intraday", "largeTrades"):
        assert k in p, k
    assert p["spot"] == 500.0 and p["quality"]["thetaFetchedAt"].startswith("2026-09-21T19:00")
    assert p["methodologyVersion"] == m.METHODOLOGY_VERSION == "1.0.0"


def test_payload_rejects_strikes_not_in_dollars():
    snap = _surface(lambda dte, k, r: 0.2)
    snap = snap.assign(strike=snap["strike"] * 1000)          # e.g. strikes reported in 1/1000ths
    try:
        m.build_symbol_payload("SPY", "INDEX", D, _mixed_trades(), snap)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "strike" in str(e)


# --------------------------------------------------------------------------
# universe + config
# --------------------------------------------------------------------------

def test_universe_default_and_env_formats():
    default = parse_universe(None)
    assert list(default) == ["INDEX", "TECH", "CYCLICAL", "DEFENSIVE", "RATES / CREDIT", "REAL ASSETS"]
    assert sum(len(v) for v in default.values()) == 22
    assert parse_universe("SPY, qqq,XLK") == {"INDEX": ["SPY", "QQQ"], "TECH": ["XLK"]}
    assert parse_universe("MINE:AAA,BBB;OTHER:CCC") == {"MINE": ["AAA", "BBB"], "OTHER": ["CCC"]}
    assert parse_universe('{"X": ["a"]}') == {"X": ["A"]}
    assert parse_universe("ZZZZ") == {"CUSTOM": ["ZZZZ"]}


def test_theta_concurrency_is_capped_at_four():
    from api.services.options_flow_config import load_config
    with env(OPTIONS_FLOW_THETA_CONCURRENCY="16"):
        assert load_config()["thetaConcurrency"] == 4
    with env(OPTIONS_FLOW_THETA_CONCURRENCY="2"):
        assert load_config()["thetaConcurrency"] == 2


# --------------------------------------------------------------------------
# data status
# --------------------------------------------------------------------------

def _utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_data_status_rules():
    # Mon 2026-09-21: EDT, session 13:30-20:00 UTC (+15m options tail)
    mid = _utc(2026, 9, 21, 18, 0)
    assert compute_data_status(D, mid - timedelta(minutes=5), mid)["status"] == "LIVE"
    assert compute_data_status(D, mid - timedelta(minutes=40), mid)["status"] == "DELAYED"
    assert compute_data_status(D, mid - timedelta(minutes=120), mid)["status"] == "STALE"
    assert compute_data_status(date(2026, 9, 18), mid - timedelta(minutes=5), mid)["status"] == "STALE"
    after = _utc(2026, 9, 21, 21, 30)
    assert compute_data_status(D, _utc(2026, 9, 21, 20, 20), after)["status"] == "CLOSED"
    assert compute_data_status(D, _utc(2026, 9, 21, 19, 0), after)["status"] == "STALE"     # missed final run
    sat = _utc(2026, 9, 26, 15, 0)
    assert compute_data_status(date(2026, 9, 25), _utc(2026, 9, 25, 20, 20), sat)["status"] == "CLOSED"
    assert compute_data_status(None, None, sat)["status"] == "STALE"


# --------------------------------------------------------------------------
# Postgres "latest snapshot" selection (pure assembly + query guards)
# --------------------------------------------------------------------------

def _row(ticker, run, as_of, market_date=D, sent=0.1):
    return {"ticker": ticker, "group_name": "X", "market_date": market_date, "as_of_timestamp": as_of,
            "created_at": as_of, "run_id": run,
            "payload": {"ticker": ticker, "sentiment": {"value": sent, "label": "NEUTRAL"}}}


def _run(run_id, status="success", created=_utc(2026, 9, 21, 18, 0)):
    return {"id": run_id, "market_date": D, "as_of_timestamp": created, "status": status,
            "created_at": created, "finished_at": created, "diagnostics": {}}


def test_assemble_latest_orders_by_universe_and_flags_carried_forward():
    uni = {"INDEX": ["SPY", "QQQ"], "TECH": ["XLK"]}
    t1, t0 = _utc(2026, 9, 21, 18, 0), _utc(2026, 9, 21, 17, 45)
    rows = [_row("QQQ", "run-2", t1), _row("SPY", "run-2", t1), _row("XLK", "run-1", t0),
            _row("GONE", "run-2", t1)]                         # not in the configured universe
    out = store.assemble_latest(rows, _run("run-2"), uni, {}, now=_utc(2026, 9, 21, 18, 5))
    assert [t["ticker"] for t in out["tickers"]] == ["SPY", "QQQ", "XLK"]
    assert [t["group"] for t in out["tickers"]] == ["INDEX", "INDEX", "TECH"]
    assert [t["carriedForward"] for t in out["tickers"]] == [False, False, True]
    assert out["missing"] == [] and out["run"]["id"] == "run-2" and out["run"]["dataStatus"] == "LIVE"


def test_assemble_latest_reports_missing_and_no_run():
    out = store.assemble_latest([_row("SPY", "r", _utc(2026, 9, 21, 18, 0))], _run("r"),
                                {"INDEX": ["SPY", "QQQ"]}, {}, now=_utc(2026, 9, 21, 18, 1))
    assert out["missing"] == ["QQQ"]
    assert store.assemble_latest([], None, {"INDEX": ["SPY"]}, {})["run"] is None


def test_every_read_query_only_sees_published_runs():
    for sql in (store.LATEST_SUMMARY_SQL, store.LATEST_TICKER_SQL, store.LATEST_RUN_SQL,
                store.HISTORY_SQL, store.IV_HISTORY_SQL):
        assert "IN ('success', 'partial')" in sql, sql
        assert "running" not in sql and "'failed'" not in sql
    assert "DISTINCT ON (s.ticker)" in store.LATEST_SUMMARY_SQL
    # live rows outrank historical_backfill rows for the same market date
    assert "s.market_date DESC, (s.source = 'live') DESC, s.as_of_timestamp DESC" in store.LATEST_SUMMARY_SQL
    assert "heavy" in store.LATEST_SUMMARY_SQL          # intraday/trades stripped in SQL


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self._last = sql

    def fetchall(self):
        return self.conn.rows_for(self._last, many=True)

    def fetchone(self):
        return self.conn.rows_for(self._last, many=False)


class _FakeConn:
    def __init__(self, summary_rows, run):
        self.summary_rows, self.run, self.executed = summary_rows, run, []

    def cursor(self):
        return _FakeCursor(self)

    def rows_for(self, sql, many):
        if sql is store.LATEST_SUMMARY_SQL:
            return self.summary_rows
        return self.run

    def commit(self):
        pass


def test_fetch_latest_passes_heavy_keys_and_uses_published_run():
    t = _utc(2026, 9, 21, 18, 0)
    conn = _FakeConn([_row("SPY", "r1", t)], _run("r1"))
    out = store.fetch_latest(conn, {"INDEX": ["SPY"]}, {}, now=_utc(2026, 9, 21, 18, 2))
    assert out["tickers"][0]["ticker"] == "SPY"
    sql, params = conn.executed[0]
    assert sql is store.LATEST_SUMMARY_SQL and params["heavy"] == ["intraday", "largeTrades"]


def test_ticker_include_trades_flag_strips_large_trades():
    row = _row("SPY", "r1", _utc(2026, 9, 21, 18, 0))
    row["payload"] = {"ticker": "SPY", "largeTrades": [{"premium": 1}], "intraday": [1]}
    conn = _FakeConn([], _run("r1"))
    conn.rows_for = lambda sql, many: row if sql is store.LATEST_TICKER_SQL else conn.run
    with_trades = store.fetch_ticker(conn, "SPY", {"INDEX": ["SPY"]}, {}, include_trades=True)
    without = store.fetch_ticker(conn, "SPY", {"INDEX": ["SPY"]}, {}, include_trades=False)
    assert "largeTrades" in with_trades["ticker"] and "largeTrades" not in without["ticker"]
    assert "intraday" in without["ticker"]


def test_assemble_history_compacts_daily_points():
    rows = [{"market_date": D, "as_of_timestamp": _utc(2026, 9, 21, 20, 0), "payload": {
        "sentiment": {"value": 0.3, "label": "BULLISH"}, "premium": {"netDirectional": 5.0, "gross": 10.0},
        "delta": {"netDollar": 7.0, "ratio": 0.4}, "iv": {"atm": 0.2, "percentile": 55.0},
        "dte": {"zeroDteShare": 0.6}}}]
    h = store.assemble_history(rows, "SPY")
    assert h["days"] == 1 and h["series"][0]["sentiment"] == 0.3 and h["series"][0]["atmIv"] == 0.2
    assert h["series"][0]["zeroDteShare"] == 0.6 and h["series"][0]["date"] == "2026-09-21"


# --------------------------------------------------------------------------
# private API authorisation
# --------------------------------------------------------------------------

def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import private_options_flow as r
    app = FastAPI()
    app.include_router(r.router)
    return TestClient(app), r


class _DummyConn:
    def cursor(self):
        return self

    def execute(self, sql):
        assert "SET TRANSACTION" in sql

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patched(r, monkey):
    monkey.append((r, "get_connection", r.get_connection))
    r.get_connection = lambda: _DummyConn()
    monkey.append((r.publication, "latest", r.publication.latest))
    r.publication.latest = lambda conn, uni, cfg, now=None, session=None: {"run": {"id": "x"}, "tickers": [], "groups": [], "missing": []}
    monkey.append((r.publication, "detail", r.publication.detail))
    r.publication.detail = lambda conn, t, include_trades=True, session=None: {"ticker": {"ticker": t, "include": include_trades}}
    monkey.append((r.store, "fetch_history", r.store.fetch_history))
    r.store.fetch_history = lambda conn, t, days, session=None: {"ticker": t, "days": days, "series": [{"date": "x"}]}
    monkey.append((r.store, "fetch_status", r.store.fetch_status))
    r.store.fetch_status = lambda conn, uni, cfg, now=None: {"latestPublished": None}


def _with_patches(fn):
    def wrapper():
        undo = []
        c, r = _client()
        _patched(r, undo)
        try:
            fn(c)
        finally:
            for obj, name, val in reversed(undo):
                setattr(obj, name, val)
    wrapper.__name__ = fn.__name__
    return wrapper


PATHS = ["/api/private/options-flow/latest", "/api/private/options-flow/status",
         "/api/private/options-flow/ticker/SPY", "/api/private/options-flow/history?ticker=SPY&days=5"]


@_with_patches
def test_private_api_requires_token(c):
    with env(INTERNAL_OPTIONS_API_SECRET="s3cret-value", OPTIONS_FLOW_UNIVERSE=None):
        for p in PATHS:
            assert c.get(p).status_code == 404, p                                        # no header
            assert c.get(p, headers={"X-Internal-Options-Token": "wrong"}).status_code == 404, p
            assert c.get(p, headers={"X-Internal-Options-Token": ""}).status_code == 404, p
            assert c.get(p, headers={"X-Internal-Options-Token": "s3cret-value "}).status_code == 404, p
            ok = c.get(p, headers={"X-Internal-Options-Token": "s3cret-value"})
            assert ok.status_code == 200, (p, ok.text)
            assert ok.headers["cache-control"] == "no-store"
            assert "noindex" in ok.headers["x-robots-tag"]


@_with_patches
def test_private_api_fails_closed_when_secret_unset(c):
    for unset in (None, ""):
        with env(INTERNAL_OPTIONS_API_SECRET=unset):
            for p in PATHS:
                # an empty/absent server secret must never match an empty/absent header
                assert c.get(p).status_code == 503, p
                assert c.get(p, headers={"X-Internal-Options-Token": ""}).status_code == 503, p


@_with_patches
def test_private_api_ticker_validation_and_flags(c):
    h = {"X-Internal-Options-Token": "tok"}
    with env(INTERNAL_OPTIONS_API_SECRET="tok", OPTIONS_FLOW_UNIVERSE=None):
        assert c.get("/api/private/options-flow/ticker/AAPL", headers=h).status_code == 404   # not in universe
        assert c.get("/api/private/options-flow/ticker/..%2F..", headers=h).status_code == 404
        assert c.get("/api/private/options-flow/history?ticker=AAPL", headers=h).status_code == 404
        assert c.get("/api/private/options-flow/history?ticker=SPY&days=0", headers=h).status_code == 422
        assert c.get("/api/private/options-flow/history?ticker=SPY&days=9999", headers=h).status_code == 422
        j = c.get("/api/private/options-flow/ticker/spy?include_trades=false", headers=h).json()
        assert j["ticker"] == {"ticker": "SPY", "include": False}                              # case-insensitive


def test_private_routes_are_hidden_from_openapi_and_registered_in_main():
    from fastapi import FastAPI
    from api.routers import private_options_flow as r
    app = FastAPI()
    app.include_router(r.router)
    assert not [p for p in app.openapi()["paths"] if "options-flow" in p]
    src = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    assert "private_options_flow_router" in src


# --------------------------------------------------------------------------
# worker: orchestration, exit codes, failure preserves last good snapshot
# --------------------------------------------------------------------------

class FakeFetcher:
    """Serves synthetic Theta-shaped frames; optionally fails for chosen tickers."""
    requests = 0

    def __init__(self, fail=(), no_oi=False):
        self.fail, self.no_oi = set(fail), no_oi

    def expirations(self, symbol, market_date, max_dte):
        if symbol in self.fail:
            raise RuntimeError("boom for " + symbol)
        return [market_date, market_date + timedelta(days=30)]

    def trade_quote(self, symbol, exp, d):
        rows = [(pd.Timestamp(d.year, d.month, d.day, 10, 30 + i, tz=NY), "CALL", 500.0, 1.20, 200, 1.0, 1.2)
                for i in range(10)]
        return raw_trades(rows, exp.isoformat())

    def greeks(self, symbol, exp, d):
        g = []
        for k in range(-5, 6):
            strike = 500.0 + 5 * k
            dl = float(np.clip(0.5 - 0.09 * k, 0.05, 0.95))
            for right, delta in (("CALL", dl), ("PUT", dl - 1)):
                g.append((pd.Timestamp(d.year, d.month, d.day, 15, 55, tz=NY), right, strike, delta, 0.2, 500.0))
        g.append((pd.Timestamp(d.year, d.month, d.day, 10, 29, tz=NY), "CALL", 500.0, 0.5, 0.2, 500.0))
        return raw_greeks(g, exp.isoformat())

    def open_interest(self, symbol, d, expirations=None):
        if self.no_oi:
            raise RuntimeError("OI not available on this plan")
        return pd.DataFrame({"expiration": [d.isoformat()], "strike": [500.0], "right": ["CALL"], "open_interest": [5000]})


class _Recorder:
    def __init__(self):
        self.calls = []

    def install(self, monkey):
        def patch(name, fn):
            monkey.append((store, name, getattr(store, name)))
            setattr(store, name, fn)
        patch("ensure_schema", lambda conn: self.calls.append(("ensure_schema",)))
        patch("final_run_exists", lambda conn, d, after: False)
        patch("create_run", lambda conn, d, cfg: self.calls.append(("create_run",)) or "run-123")
        patch("load_iv_history", lambda conn, t, d, n: {"priorAtmIv": []})
        patch("finalize_run", lambda conn, rid, status, as_of, diag, snaps: self.calls.append(("finalize", status, [s["ticker"] for s in snaps], diag)))
        patch("fail_run", lambda conn, rid, diag: self.calls.append(("fail", diag)))
        patch("prune", lambda conn, d, n: self.calls.append(("prune",)) or (0, 0))


class _Conn:
    closed = False

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        assert "pg_try_advisory_lock" in sql

    def fetchone(self):
        return {"acquired": True}

    def close(self):
        self.closed = True

    def rollback(self):
        pass


def _run_job(argv, fetcher):
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    rec, undo = _Recorder(), []
    rec.install(undo)
    try:
        code = job.run(argv, fetcher=fetcher, conn_factory=lambda: _Conn())
    finally:
        for obj, name, val in reversed(undo):
            setattr(obj, name, val)
    return code, rec


def test_worker_success_publishes_and_exits_zero():
    code, rec = _run_job(["--tickers", "SPY,QQQ", "--date", "2026-09-21"], FakeFetcher())
    assert code == 0
    fin = [c for c in rec.calls if c[0] == "finalize"][0]
    assert fin[1] == "success" and fin[2] == ["SPY", "QQQ"]
    assert not [c for c in rec.calls if c[0] == "fail"]


def test_worker_partial_failure_publishes_survivors_and_exits_nonzero():
    code, rec = _run_job(["--tickers", "SPY,QQQ", "--date", "2026-09-21"], FakeFetcher(fail=["QQQ"]))
    assert code == 2
    fin = [c for c in rec.calls if c[0] == "finalize"][0]
    assert fin[1] == "partial" and fin[2] == ["SPY"]
    assert "QQQ" in fin[3]["failed"] and "boom" in fin[3]["failed"]["QQQ"]


def test_worker_total_failure_marks_run_failed_and_never_publishes():
    code, rec = _run_job(["--tickers", "SPY,QQQ", "--date", "2026-09-21"], FakeFetcher(fail=["SPY", "QQQ"]))
    assert code == 1
    assert not [c for c in rec.calls if c[0] == "finalize"]      # previous snapshot untouched
    assert [c for c in rec.calls if c[0] == "fail"]


def test_worker_unexpected_error_exits_nonzero_and_marks_failed():
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    rec, undo = _Recorder(), []
    rec.install(undo)
    undo.append((store, "create_run", store.create_run))
    store.create_run = lambda conn, d, cfg: (_ for _ in ()).throw(RuntimeError("db down"))
    try:
        code = job.run(["--tickers", "SPY", "--date", "2026-09-21"], fetcher=FakeFetcher(), conn_factory=lambda: _Conn())
    finally:
        for obj, name, val in reversed(undo):
            setattr(obj, name, val)
    assert code == 1


def test_worker_open_interest_failure_is_a_warning_not_a_failure():
    code, rec = _run_job(["--tickers", "SPY", "--date", "2026-09-21"], FakeFetcher(no_oi=True))
    assert code == 0
    fin = [c for c in rec.calls if c[0] == "finalize"][0]
    assert any("open interest unavailable" in w for w in fin[3]["warnings"])


def test_worker_dry_run_writes_nothing():
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    rec, undo = _Recorder(), []
    rec.install(undo)

    def no_conn():
        raise AssertionError("dry run must not open a DB connection")
    try:
        code = job.run(["--tickers", "SPY", "--date", "2026-09-21", "--dry-run"], fetcher=FakeFetcher(), conn_factory=no_conn)
    finally:
        for obj, name, val in reversed(undo):
            setattr(obj, name, val)
    assert code == 0 and rec.calls == []


def test_worker_rejects_ticker_outside_universe():
    code, rec = _run_job(["--tickers", "AAPL", "--date", "2026-09-21"], FakeFetcher())
    assert code == 1 and rec.calls == []


def test_worker_skips_when_final_print_exists_unless_forced():
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    rec, undo = _Recorder(), []
    rec.install(undo)
    undo.append((store, "final_run_exists", store.final_run_exists))
    store.final_run_exists = lambda conn, d, after: True
    try:
        skipped = job.run(["--tickers", "SPY", "--date", "2026-09-18"], fetcher=FakeFetcher(), conn_factory=lambda: _Conn())
        assert skipped == 0 and not [c for c in rec.calls if c[0] in ("create_run", "finalize")]
        forced = job.run(["--tickers", "SPY", "--date", "2026-09-18", "--force"], fetcher=FakeFetcher(), conn_factory=lambda: _Conn())
        assert forced == 0 and [c for c in rec.calls if c[0] == "finalize"]
    finally:
        for obj, name, val in reversed(undo):
            setattr(obj, name, val)


def test_theta_key_is_redacted_from_errors():
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    with env(THETADATA_API_KEY="abc123SECRET"):
        assert "abc123SECRET" not in job.redact("auth failed for abc123SECRET here")
    with env(THETADATA_API_KEY=None):
        assert job.redact("nothing") == "nothing"


def test_fetcher_requires_api_key_and_never_falls_back_to_creds_file():
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    with env(THETADATA_API_KEY=None):
        try:
            job.ThetaFetcher._connect()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "THETADATA_API_KEY" in str(e)


def test_fetcher_never_exceeds_four_concurrent_requests():
    import importlib, threading, time
    job = importlib.import_module("jobs.options_flow_refresh")
    active, peak, lock = [0], [0], threading.Lock()

    class Client:
        def option_history_trade_quote(self, **kw):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return pd.DataFrame()

    from api.services.options_flow_config import load_config
    with env(OPTIONS_FLOW_THETA_CONCURRENCY="99"):
        f = job.ThetaFetcher(load_config(), client=Client())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:        # deliberately oversized pool
        list(pool.map(lambda i: f.trade_quote("SPY", D, D), range(64)))
    assert 1 < peak[0] <= 4 and f.requests == 64


def test_fetcher_retries_transient_grpc_and_treats_no_data_as_empty():
    import importlib
    job = importlib.import_module("jobs.options_flow_refresh")
    job.RETRY_BACKOFF_SEC = (0.0, 0.0, 0.0)

    class Code:
        name = "UNAVAILABLE"

    class Transient(Exception):
        def code(self):
            return Code()

    calls = {"n": 0}

    class Client:
        def option_history_trade_quote(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Transient("temporarily down")
            return raw_trades([(ts(10, 30), "CALL", 500.0, 1.2, 1, 1.0, 1.2)])

    from api.services.options_flow_config import load_config
    f = job.ThetaFetcher(load_config(), client=Client())
    assert len(f.trade_quote("SPY", D, D)) == 1 and calls["n"] == 3

    class Perm(Exception):
        def code(self):
            class C:
                name = "PERMISSION_DENIED"
            return C()

    class Denied:
        def option_history_trade_quote(self, **kw):
            raise Perm("nope")

    g = job.ThetaFetcher(load_config(), client=Denied())
    try:
        g.trade_quote("SPY", D, D)
        raise AssertionError("permission errors must not be swallowed")
    except RuntimeError as e:
        assert "nope" in str(e)


def test_local_env_loader_sets_only_theta_keys_and_never_overrides():
    import importlib, tempfile
    job = importlib.import_module("jobs.options_flow_refresh")
    p = Path(tempfile.mkdtemp()) / ".env"
    p.write_text("FRED_API_KEY=nope\nexport THETADATA_API_KEY='abc123'\nDATABASE_URL=postgres://x\n", encoding="utf-8")
    with env(THETADATA_API_KEY=None, DATABASE_URL="already-set", FRED_API_KEY=None):
        assert job.load_local_env(p) == ["THETADATA_API_KEY"]
        assert os.environ["THETADATA_API_KEY"] == "abc123"
        assert os.environ["DATABASE_URL"] == "already-set" and "FRED_API_KEY" not in os.environ
        del os.environ["THETADATA_API_KEY"]
    assert job.load_local_env(p.parent / "missing.env") == []


def test_no_pro_only_endpoints_are_referenced():
    src = (ROOT / "jobs" / "options_flow_refresh.py").read_text(encoding="utf-8")
    assert "trade_greeks" not in src.replace("No trade-greeks", "")


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
