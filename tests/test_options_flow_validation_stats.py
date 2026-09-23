"""
tests/test_options_flow_validation_stats.py

Unit tests for api/services/options_flow_validation_stats.py against synthetic panels with a
known ground-truth answer. No DB/network.
"""

import numpy as np
import pandas as pd
import pytest

from api.services.options_flow_validation_stats import (
    date_block_bootstrap,
    date_level_portfolio_spread,
    date_level_portfolio_test,
    drop_dates_and_recompute,
    leave_one_date_out,
    panel_ols_cluster_by_date,
    panel_ols_two_regressors_cluster_by_date,
)


def _synthetic_panel(n_dates=80, tickers=("AAA", "BBB", "CCC"), true_beta=2.0, seed=7,
                     ticker_effects=None, date_shock_scale=0.0):
    """y = true_beta * x + ticker_effect + date_shock (correlated within a date) + idiosyncratic
    noise. date_shock_scale > 0 induces the exact kind of within-date correlation that makes
    ordinary (non-clustered) SEs too small -- clustering by date should still recover true_beta
    with a sensible SE."""
    rng = np.random.default_rng(seed)
    ticker_effects = ticker_effects or {t: i * 0.5 for i, t in enumerate(tickers)}
    rows = []
    dates = [f"2026-01-{d:02d}" if d <= 28 else f"2026-02-{d - 28:02d}" for d in range(1, n_dates + 1)]
    for d in dates:
        date_shock = rng.normal(0, date_shock_scale) if date_shock_scale else 0.0
        for t in tickers:
            x = rng.normal(0, 1)
            y = true_beta * x + ticker_effects[t] + date_shock + rng.normal(0, 0.1)
            rows.append({"date": d, "ticker": t, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_panel_ols_recovers_true_beta_no_date_shock():
    df = _synthetic_panel(n_dates=100, date_shock_scale=0.0, seed=1)
    out = panel_ols_cluster_by_date(df, y_col="y", x_col="x")
    assert out["beta"] == pytest.approx(2.0, abs=0.05)
    assert out["tStat"] is not None and abs(out["tStat"]) > 20  # very clean signal, tiny noise
    assert out["nClusters"] == 100


def test_panel_ols_recovers_true_beta_with_date_shocks():
    """Even with a strong per-date common shock (the exact pseudo-replication problem v3 exists
    to fix), the coefficient itself should still be ~unbiased; clustering affects the SE, not
    the point estimate. Cross-checked independently against a from-scratch numpy lstsq on the
    same design matrix (see the module's dev notes) -- the point estimate matches exactly, so
    the tolerance here is set by genuine sampling noise (only 3 tickers/date, large date shocks),
    not implementation error; a larger n_dates keeps the test from being seed-flaky."""
    df = _synthetic_panel(n_dates=400, date_shock_scale=5.0, seed=2)
    out = panel_ols_cluster_by_date(df, y_col="y", x_col="x")
    assert out["beta"] == pytest.approx(2.0, abs=0.15)
    assert out["se"] is not None and out["se"] > 0


def test_panel_ols_null_case_beta_near_zero():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(60):
        d = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        for t in ("AAA", "BBB", "CCC"):
            rows.append({"date": d, "ticker": t, "x": rng.normal(), "y": rng.normal()})
    df = pd.DataFrame(rows)
    out = panel_ols_cluster_by_date(df, "y", "x")
    assert abs(out["beta"]) < 0.3
    assert abs(out["tStat"]) < 3  # should not spuriously reject with pure noise


def test_panel_ols_two_regressors_isolates_each_effect():
    rng = np.random.default_rng(4)
    rows = []
    for i in range(80):
        d = f"d{i}"
        for t in ("AAA", "BBB", "CCC"):
            x1 = rng.normal()
            x2 = rng.normal()  # independent of x1
            y = 3.0 * x1 + -1.5 * x2 + rng.normal(0, 0.1)
            rows.append({"date": d, "ticker": t, "x1": x1, "x2": x2, "y": y})
    df = pd.DataFrame(rows)
    out = panel_ols_two_regressors_cluster_by_date(df, "y", "x1", "x2")
    assert out["x1"]["beta"] == pytest.approx(3.0, abs=0.1)
    assert out["x2"]["beta"] == pytest.approx(-1.5, abs=0.1)


def test_panel_ols_insufficient_data_returns_none():
    df = pd.DataFrame({"date": ["d1"], "ticker": ["AAA"], "x": [1.0], "y": [1.0]})
    out = panel_ols_cluster_by_date(df, "y", "x")
    assert out["beta"] is None


def test_date_block_bootstrap_ci_covers_true_positive_effect():
    df = _synthetic_panel(n_dates=100, true_beta=2.0, date_shock_scale=1.0, seed=5)

    def stat_fn(d):
        r = panel_ols_cluster_by_date(d, "y", "x")
        return r["beta"]

    result = date_block_bootstrap(df, stat_fn, n_boot=500, block_length=1, seed=1)
    lo, hi = result["ci95"]
    assert lo < 2.0 < hi
    assert result["shareSameSign"] > 0.9


def test_date_block_bootstrap_null_ci_covers_zero():
    rng = np.random.default_rng(6)
    rows = []
    for i in range(60):
        d = f"d{i}"
        for t in ("AAA", "BBB", "CCC"):
            rows.append({"date": d, "ticker": t, "x": rng.normal(), "y": rng.normal()})
    df = pd.DataFrame(rows)

    def stat_fn(d):
        r = panel_ols_cluster_by_date(d, "y", "x")
        return r["beta"]

    result = date_block_bootstrap(df, stat_fn, n_boot=500, seed=2)
    lo, hi = result["ci95"]
    assert lo < 0 < hi


def test_date_block_bootstrap_too_few_dates():
    df = pd.DataFrame({"date": ["d1", "d2"], "ticker": ["A", "B"], "x": [1, 2], "y": [1, 2]})
    result = date_block_bootstrap(df, lambda d: 1.0, n_boot=10)
    assert result["n_boot"] == 0


def test_date_level_portfolio_spread_basic():
    df = pd.DataFrame({
        "date": ["d1"] * 6,
        "ticker": ["A", "B", "C", "D", "E", "F"],
        "feature": [1, 2, 3, 4, 5, 6],
        "ret": [0.10, 0.05, 0.0, -0.01, 0.20, 0.30],
    })
    spreads = date_level_portfolio_spread(df, "feature", "ret", n_high=3, n_low=3)
    assert len(spreads) == 1
    # low basket = lowest 3 feature values (A,B,C) -> ret mean (0.10+0.05+0.0)/3
    # high basket = highest 3 feature values (D,E,F) -> ret mean (-0.01+0.20+0.30)/3
    row = spreads.iloc[0]
    assert row["lowBasketReturn"] == pytest.approx((0.10 + 0.05 + 0.0) / 3)
    assert row["highBasketReturn"] == pytest.approx((-0.01 + 0.20 + 0.30) / 3)
    assert row["spread"] == pytest.approx(row["highBasketReturn"] - row["lowBasketReturn"])


def test_date_level_portfolio_spread_skips_thin_dates():
    df = pd.DataFrame({
        "date": ["d1", "d1"], "ticker": ["A", "B"], "feature": [1, 2], "ret": [0.1, 0.2],
    })
    spreads = date_level_portfolio_spread(df, "feature", "ret", n_high=3, n_low=3)
    assert len(spreads) == 0  # only 2 tickers, need 6 for n_high=3+n_low=3


def test_date_level_portfolio_test_wraps_hac():
    rng = np.random.default_rng(8)
    rows = []
    for i in range(60):
        d = f"d{i}"
        for j, t in enumerate("ABCDEF"):
            rows.append({"date": d, "ticker": t, "feature": j + rng.normal(0, 0.01),
                        "ret": 0.01 * j + rng.normal(0, 0.001)})
    df = pd.DataFrame(rows)
    out = date_level_portfolio_test(df, "feature", "ret", horizon_days=5)
    assert out["nDates"] == 60
    assert out["mean"] > 0  # high-feature basket (D,E,F) should beat low (A,B,C) by construction


def test_leave_one_date_out_detects_influential_date():
    rows = []
    for i in range(20):
        rows.append({"date": f"d{i}", "value": 1.0})
    rows.append({"date": "OUTLIER", "value": 1000.0})
    df = pd.DataFrame(rows)

    def stat_fn(d):
        return float(d["value"].mean()) if len(d) else None

    out = leave_one_date_out(df, stat_fn)
    assert out["mostInfluentialDates"][0]["date"] == "OUTLIER"
    # removing the outlier should pull the mean down toward 1.0
    assert out["min"] == pytest.approx(1.0, abs=0.01)


def test_drop_dates_and_recompute():
    df = pd.DataFrame({"date": ["d1", "d2", "d3"], "value": [1.0, 2.0, 100.0]})
    result = drop_dates_and_recompute(df, ["d3"], lambda d: float(d["value"].mean()))
    assert result == pytest.approx(1.5)
