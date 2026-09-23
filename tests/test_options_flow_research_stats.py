"""
tests/test_options_flow_research_stats.py

Unit tests for api/services/options_flow_research_stats.py -- pure math, no DB/network.
"""

import math

import numpy as np
import pytest

from api.services.options_flow_research_stats import (
    NEUTRAL_SENTIMENT_BAND,
    benjamini_hochberg,
    classify_agreement,
    describe_returns,
    hac_lag_for_horizon,
    median_split,
    newey_west_mean_variance,
    pd_rank,
    quintile_labels,
    sign_bucket,
    spearman_correlation,
    tercile_split,
)


def test_hac_lag_for_horizon():
    assert hac_lag_for_horizon(1) == 0
    assert hac_lag_for_horizon(5) == 4
    assert hac_lag_for_horizon(20) == 19


def test_describe_returns_basic_stats():
    vals = [0.01, -0.02, 0.03, 0.0, 0.05, -0.01]
    out = describe_returns(vals, horizon_days=1)
    assert out["n"] == 6
    assert out["mean"] == pytest.approx(np.mean(vals))
    assert out["median"] == pytest.approx(np.median(vals))
    assert out["winRate"] == pytest.approx(3 / 6)  # 0.0 is not a "win"
    assert out["std"] == pytest.approx(np.std(vals, ddof=1))
    assert out["tStat"] is not None
    assert out["hacTStat"] is not None
    assert out["hacLag"] == 0


def test_describe_returns_drops_none_and_nan_never_zero_fills():
    vals = [0.01, None, float("nan"), 0.03]
    out = describe_returns(vals, horizon_days=5)
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(0.02)


def test_describe_returns_empty():
    out = describe_returns([], horizon_days=5)
    assert out["n"] == 0
    assert out["mean"] is None
    assert out["tStat"] is None
    assert out["hacTStat"] is None


def test_describe_returns_single_value_no_tstat():
    out = describe_returns([0.02], horizon_days=1)
    assert out["n"] == 1
    assert out["mean"] == 0.02
    assert out["std"] is None
    assert out["tStat"] is None
    assert out["hacTStat"] is None


def test_newey_west_variance_matches_plain_variance_at_lag_zero():
    x = np.array([0.01, -0.02, 0.03, 0.0, 0.05, -0.01, 0.02, -0.03])
    nw_var = newey_west_mean_variance(x, lags=0)
    plain_var = float(np.var(x, ddof=0) / len(x))
    assert nw_var == pytest.approx(plain_var, rel=1e-9)


def test_newey_west_variance_increases_with_positive_autocorrelation():
    # A strongly positively-autocorrelated series should have a LARGER long-run variance
    # of the mean than an i.i.d. series with the same marginal variance.
    rng = np.random.default_rng(42)
    iid = rng.normal(0, 1, 500)
    ar = np.zeros(500)
    ar[0] = iid[0]
    for i in range(1, 500):
        ar[i] = 0.8 * ar[i - 1] + iid[i]
    var_iid = newey_west_mean_variance(iid, lags=4)
    var_ar = newey_west_mean_variance(ar, lags=4)
    assert var_ar > var_iid


def test_quintile_labels_basic_ordering():
    vals = list(range(1, 21))  # 1..20, clean ordering
    labels = quintile_labels(vals)
    assert labels[0] == 1  # smallest value -> lowest quintile
    assert labels[-1] == 5  # largest value -> highest quintile
    assert all(l is not None for l in labels)
    # roughly even bucket sizes
    from collections import Counter
    counts = Counter(labels)
    assert set(counts.keys()) == {1, 2, 3, 4, 5}


def test_quintile_labels_none_preserved_and_too_few_values():
    vals = [1, 2, None, 4, 5]
    labels = quintile_labels(vals)
    assert labels[2] is None
    assert all(l is None for l in quintile_labels([1, 2, 3]))  # < 5 valid -> all None


def test_pd_rank_ties_get_average_rank():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    ranks = pd_rank(x)
    assert ranks[0] == 1.0
    assert ranks[1] == ranks[2] == 2.5
    assert ranks[3] == 4.0


def test_spearman_correlation_perfect_monotonic():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    out = spearman_correlation(x, y)
    assert out["n"] == 5
    assert out["rho"] == pytest.approx(1.0)


def test_spearman_correlation_drops_pairs_with_missing_either_side():
    x = [1, 2, None, 4, 5]
    y = [1, None, 3, 4, 5]
    out = spearman_correlation(x, y)
    assert out["n"] == 3  # only indices 0, 3, 4 have both sides present


def test_spearman_correlation_too_few_pairs():
    out = spearman_correlation([1, 2], [1, 2])
    assert out["n"] == 2
    assert out["rho"] is None


def test_classify_agreement_uses_product_neutral_band():
    assert NEUTRAL_SENTIMENT_BAND == 0.08
    assert classify_agreement(0.08, 5.0) == "approximately_neutral"  # inclusive boundary
    assert classify_agreement(0.081, 5.0) == "agree_bullish"
    assert classify_agreement(-0.081, -5.0) == "agree_bearish"
    assert classify_agreement(0.081, -5.0) == "disagreement"
    assert classify_agreement(-0.081, 5.0) == "disagreement"
    assert classify_agreement(None, 5.0) is None
    assert classify_agreement(0.1, None) is None


def test_median_split():
    med, labels = median_split([1, 2, 3, 4, 5])
    assert med == 3
    assert labels == ["low", "low", "low", "high", "high"]  # equal to median -> low (not > med)


def test_median_split_insufficient_data():
    med, labels = median_split([1])
    assert med is None
    assert labels == [None]


def test_tercile_split():
    edges, labels = tercile_split(list(range(1, 10)))  # 1..9
    assert edges is not None
    assert "low" in labels and "mid" in labels and "high" in labels


def test_benjamini_hochberg_known_example():
    # classic textbook case: m=5, alpha=0.05 -> first 4 (ranked) reject, last does not
    p = [0.01, 0.02, 0.03, 0.04, 0.5]
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["nTested"] == 5
    assert out["qValues"][0] == pytest.approx(0.05)
    assert out["qValues"][3] == pytest.approx(0.05)
    assert out["qValues"][4] == pytest.approx(0.5)
    assert out["reject"] == [True, True, True, True, False]


def test_benjamini_hochberg_preserves_order_and_none():
    p = [0.5, None, 0.01]  # deliberately unsorted, with a missing test
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["nTested"] == 2  # only the 2 non-None p-values are corrected
    assert out["qValues"][1] is None
    assert out["reject"][1] is None
    assert out["qValues"][2] == pytest.approx(0.02)  # 0.01 * 2 / 1
    assert out["reject"][2] is True


def test_benjamini_hochberg_all_none():
    out = benjamini_hochberg([None, None])
    assert out["nTested"] == 0
    assert out["qValues"] == [None, None]


def test_sign_bucket():
    assert sign_bucket(0.01) == "positive"
    assert sign_bucket(-0.01) == "negative"
    assert sign_bucket(0.0) == "neutral"
    assert sign_bucket(None) is None
