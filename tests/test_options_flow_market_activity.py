"""
tests/test_options_flow_market_activity.py

Unit tests (synthetic data, no DB) for api/services/options_flow_market_activity.py's pure
functions, plus the MANDATORY regression test (item 17) comparing the production calculation
against the frozen research calculation on real historical dates.

The regression test is gated on the REAL production DATABASE_URL (not
OPTIONS_FLOW_TEST_DATABASE_URL / pgserver -- an ephemeral embedded Postgres has none of the
real backfilled history this test needs to compare against) and is strictly READ-ONLY: it never
calls ensure_schema, never drops or writes anything. Skipped when DATABASE_URL is unset.

    OPTIONS_FLOW_TEST_DATABASE_URL is for the rest of this project's pg_-prefixed tests (isolated,
    disposable schema). DATABASE_URL here means "the real thing" -- do not repoint this test at a
    throwaway database, because then it isn't testing anything.
"""

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.options_flow_market_activity import (
    ACTIVITY_TICKERS,
    Z_MIN_PERIODS,
    Z_WINDOW,
    activity_regime,
    breadth,
    compute_market_activity_snapshot,
    historical_percentile,
    load_gross_premium_history,
    market_activity_frame,
    rolling_zscore_20d,
)
from api.services import options_flow_store as store


def test_activity_tickers_matches_phase1_scope():
    assert set(ACTIVITY_TICKERS) == {"SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"}


def test_z_window_constants_frozen():
    # These ARE the methodology (commit 19dc02c / the v3 manifest) -- a change here is a change
    # to the signal itself, not an implementation detail. This test exists so nobody edits them
    # by accident.
    assert Z_WINDOW == 20
    assert Z_MIN_PERIODS == 10


def _synthetic_history(n=40, tickers=("SPY", "QQQ"), seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n)
    rows = []
    for t in tickers:
        base = 1_000_000 + hash(t) % 1000
        for i, d in enumerate(dates):
            rows.append({"ticker": t, "date": str(d.date()), "gross_premium": base + rng.normal(0, 50_000)})
    return pd.DataFrame(rows)


def test_rolling_zscore_matches_manual_calculation():
    df = _synthetic_history(n=35, tickers=("SPY",), seed=2)
    z = rolling_zscore_20d(df, "gross_premium")
    # row 10 (11th obs, index 10): prior 10 observations (indices 0-9) -> hand-computed z
    prior_vals = df["gross_premium"].iloc[0:10].to_numpy()
    expected = (df["gross_premium"].iloc[10] - prior_vals.mean()) / prior_vals.std(ddof=1)
    assert z.iloc[10] == pytest.approx(expected, rel=1e-9)


def test_rolling_zscore_nan_during_warmup():
    df = _synthetic_history(n=15, tickers=("SPY",), seed=3)
    z = rolling_zscore_20d(df, "gross_premium")
    # shift(1) means row 0's "prior" is itself NaN, so row index 10 is the first with a full
    # 10 non-null prior observations (indices 0-9) -- matches the min_periods=10 rolling window.
    assert z.iloc[:10].isna().all()
    assert pd.notna(z.iloc[10])


def test_market_activity_frame_median_and_residual():
    df = pd.DataFrame({
        "ticker": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        "date": [str(d.date()) for d in pd.bdate_range("2026-01-01", periods=15)] * 3,
        "gross_premium": (
            [1_000_000 + i * 1000 for i in range(15)] +
            [2_000_000 + i * 500 for i in range(15)] +
            [500_000 + i * 2000 for i in range(15)]
        ),
    })
    out = market_activity_frame(df, min_tickers=2)
    last_date = sorted(out["date"].unique())[-1]
    rows = out[out["date"] == last_date]
    zs = rows.set_index("ticker")["z_gross_premium_20d"]
    expected_median = float(zs.median())
    assert rows["market_activity_median"].iloc[0] == pytest.approx(expected_median)
    # residual = own z - market median
    for t in ("A", "B", "C"):
        row = rows[rows["ticker"] == t].iloc[0]
        assert row["residual_activity"] == pytest.approx(row["z_gross_premium_20d"] - expected_median)


def test_market_activity_frame_respects_min_tickers():
    # Only 1 ticker with history on a given date -> median/mean should be NaN (min_tickers=3 default)
    df = pd.DataFrame({
        "ticker": ["A"] * 15,
        "date": [str(d.date()) for d in pd.bdate_range("2026-01-01", periods=15)],
        "gross_premium": [1_000_000 + i * 1000 for i in range(15)],
    })
    out = market_activity_frame(df, min_tickers=3)
    assert out["market_activity_median"].isna().all()


def test_historical_percentile_known_value():
    history = list(range(1, 101))  # 1..100, 100 observations
    pct = historical_percentile(50, [float(v) for v in history])
    assert pct == pytest.approx(50.0, abs=0.5)


def test_historical_percentile_insufficient_history():
    pct = historical_percentile(5.0, [1.0, 2.0, 3.0])  # only 3 obs, below MIN_PERCENTILE_OBS
    assert pct is None


def test_historical_percentile_none_current():
    assert historical_percentile(None, [float(v) for v in range(30)]) is None


def test_activity_regime_bands():
    assert activity_regime(10.0) == "QUIET"
    assert activity_regime(24.9) == "QUIET"
    assert activity_regime(25.0) == "NORMAL"
    assert activity_regime(50.0) == "NORMAL"
    assert activity_regime(74.9) == "NORMAL"
    assert activity_regime(75.0) == "ELEVATED"
    assert activity_regime(94.9) == "ELEVATED"
    assert activity_regime(95.0) == "EXTREME"
    assert activity_regime(99.9) == "EXTREME"
    assert activity_regime(None) is None


def test_attach_market_activity_never_raises_on_a_broken_connection():
    """A conn that can't run the real query (e.g. the fake conns other store tests use) must
    leave marketActivity: None and every ticker's residualActivity: None, never propagate the
    exception into the /latest response."""
    class _BrokenConn:
        def cursor(self):
            raise RuntimeError("no real Postgres here")

    response = {"tickers": [{"ticker": "SPY"}, {"ticker": "QQQ"}]}
    store.attach_market_activity(_BrokenConn(), response)
    assert response["marketActivity"] is None
    assert response["tickers"][0]["residualActivity"] is None
    assert response["tickers"][0]["activityZ"] is None
    assert response["tickers"][1]["residualActivity"] is None


def test_attach_market_activity_wires_fields_when_snapshot_available(monkeypatch):
    fake_snapshot = {
        "asOfDate": "2026-09-21", "value": 1.23, "valueMean": 1.1, "percentile": 80.0,
        "regime": "ELEVATED", "breadth": {"aboveNormal": 4, "total": 6, "tracked": 6},
        "history": [{"date": "2026-09-21", "value": 1.23}],
        "zByTicker": {"SPY": 0.5, "QQQ": 2.0}, "residualByTicker": {"SPY": -0.73, "QQQ": 0.77},
        "tickers": ["SPY", "QQQ"],
    }
    import api.services.options_flow_market_activity as activity_module
    monkeypatch.setattr(activity_module, "compute_market_activity_snapshot", lambda conn: fake_snapshot)

    response = {"tickers": [{"ticker": "SPY"}, {"ticker": "QQQ"}]}
    store.attach_market_activity(object(), response)

    assert response["marketActivity"]["value"] == 1.23
    assert response["marketActivity"]["regime"] == "ELEVATED"
    assert response["tickers"][0]["residualActivity"] == -0.73
    assert response["tickers"][1]["activityZ"] == 2.0


def test_breadth_counts_above_normal():
    z = {"SPY": 1.5, "QQQ": -0.5, "IWM": 0.2, "SMH": None, "TLT": -1.0, "GLD": 0.01}
    out = breadth(z)
    assert out["aboveNormal"] == 3  # SPY, IWM, GLD > 0
    assert out["total"] == 5        # 5 non-None
    assert out["tracked"] == 6


# --------------------------------------------------------------------------------------------
# MANDATORY regression test (item 17): production calculation must match the research
# calculation exactly, within floating-point tolerance, on real historical dates.
# --------------------------------------------------------------------------------------------

def _real_pg_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise unittest.SkipTest(
            "DATABASE_URL not set -- the mandatory production-vs-research regression test needs "
            "the REAL database (it has no meaning against an empty/ephemeral one). Skipped, not failed."
        )
    import psycopg
    from psycopg.rows import dict_row
    return psycopg.connect(url, row_factory=dict_row)


def _research_reference_zscores() -> pd.DataFrame:
    """Independent research-side computation: the frozen v3 discovery+validation parquet files
    (gross_premium already computed by the real backfill), z-scored with v3's OWN frozen
    _zscore logic (imported from scripts/run_options_flow_validation_v3.py, never reimplemented
    here) -- a genuinely separate code path from api/services/options_flow_market_activity.py's
    production rolling_zscore_20d, so agreement here is a real check, not a tautology.

    Returns BOTH samples with a `sample` column; see the comment on the regression test itself
    for why only the `validation` rows are directly comparable to a continuously-running
    production system."""
    from scripts.run_options_flow_validation_v3 import load_discovery_frame, load_validation_frame

    discovery = load_discovery_frame()
    validation_full = load_validation_frame()
    validation = validation_full[validation_full["sample"] == "validation"]
    combined = pd.concat([discovery, validation], ignore_index=True, sort=False)
    return combined[["date", "ticker", "z_gross_premium_20d", "sample"]].rename(
        columns={"z_gross_premium_20d": "research_z"})


def test_pg_production_matches_research_calculation_on_historical_dates():
    """
    The mandatory regression test: for real historical dates, the PRODUCTION path (Postgres ->
    load_gross_premium_history -> rolling_zscore_20d, rolling through ONE continuous history)
    must equal the RESEARCH path (frozen v3 parquet -> v3's own _zscore) within floating-point
    tolerance, per ticker.

    Compared ONLY on the 192 VALIDATION-sample dates (2025-09-19 onward), not the original 60
    DISCOVERY-sample dates. This is deliberate, not a loophole: v1/v2 computed the discovery
    sample's z-scores using ONLY discovery-internal lookback, because the extended
    warmup/validation history did not exist yet at that point in the research -- an artifact of
    how the study was staged incrementally, not a property of the z_gross_premium_20d
    methodology itself (which is only well-defined once >=10 sessions of continuous prior
    history exist). A continuously-running production system naturally has that continuous
    history and SHOULD diverge from the discovery-only numbers on those 60 dates -- verified
    below to be exactly where the two paths part ways, and nowhere else.
    """
    conn = _real_pg_conn()
    try:
        prod_raw = load_gross_premium_history(conn)
    finally:
        conn.close()
    if prod_raw.empty:
        raise unittest.SkipTest("no full_flow history in this database -- nothing to regress against")

    prod = market_activity_frame(prod_raw)[["date", "ticker", "z_gross_premium_20d"]]
    research = _research_reference_zscores()
    research_validation = research[research["sample"] == "validation"]

    merged = prod.merge(research_validation, on=["date", "ticker"], how="inner")
    assert len(merged) == 192 * 6, f"expected exactly 192x6 validation-sample rows, got {len(merged)}"

    both_present = merged.dropna(subset=["z_gross_premium_20d", "research_z"])
    assert len(both_present) > 1000, "expected the large majority of validation dates to have a valid z-score"

    max_abs_diff = (both_present["z_gross_premium_20d"] - both_present["research_z"]).abs().max()
    assert max_abs_diff < 1e-6, (
        f"production and research z_gross_premium_20d diverge by up to {max_abs_diff} on the "
        "validation-sample dates -- the production formula in "
        "api/services/options_flow_market_activity.py no longer matches the frozen research "
        "methodology"
    )

    # Spot-check a handful of specific (ticker, date) pairs explicitly, not just the aggregate max.
    sample = both_present.sample(min(20, len(both_present)), random_state=7)
    for _, row in sample.iterrows():
        assert row["z_gross_premium_20d"] == pytest.approx(row["research_z"], abs=1e-6), (
            f"{row['ticker']} {row['date']}: production={row['z_gross_premium_20d']} "
            f"vs research={row['research_z']}"
        )

    # And confirm the discovery-sample divergence is real, expected, and NOT silently masking a
    # bug: production should legitimately differ from the discovery-only-lookback numbers there.
    research_discovery = research[research["sample"] == "discovery"]
    merged_disc = prod.merge(research_discovery, on=["date", "ticker"], how="inner").dropna(
        subset=["z_gross_premium_20d", "research_z"])
    if len(merged_disc):
        disc_diff = (merged_disc["z_gross_premium_20d"] - merged_disc["research_z"]).abs().max()
        assert disc_diff > 1e-6, (
            "expected the discovery-sample dates to differ (continuous vs discovery-only "
            "lookback) -- if this now matches exactly, the two code paths may have silently "
            "converged in a way that deserves a second look, not just a relaxed assertion"
        )


def test_pg_compute_market_activity_snapshot_smoke():
    """Read-only smoke test: the full snapshot function returns a well-formed, self-consistent
    result against the real database (no schema writes, no mutation)."""
    conn = _real_pg_conn()
    try:
        snap = compute_market_activity_snapshot(conn)
    finally:
        conn.close()
    if snap is None:
        raise unittest.SkipTest("no full_flow history available")

    assert set(snap["tickers"]) == set(ACTIVITY_TICKERS)
    assert snap["breadth"]["tracked"] == len(ACTIVITY_TICKERS)
    assert snap["breadth"]["total"] >= snap["breadth"]["aboveNormal"] >= 0
    if snap["percentile"] is not None:
        assert 0.0 <= snap["percentile"] <= 100.0
        assert snap["regime"] in ("QUIET", "NORMAL", "ELEVATED", "EXTREME")
    assert len(snap["history"]) > 0
    assert all(set(t) == {"date", "value"} for t in snap["history"][:5])
