"""
api/services/options_flow_validation_stats.py

Statistical machinery specific to Signal Validation v3 -- the date-clustering-robust tests that
v2 explicitly lacked. Pure numpy/pandas (no statsmodels/linearmodels dependency, consistent with
the rest of this codebase's hand-implemented stats), fully unit-testable against synthetic data
with a known, hand-computable answer.

    * panel_ols_cluster_by_date: OLS with ticker fixed effects, one-way cluster-robust (CR1,
      Stata-style small-sample correction) standard errors clustered by market_date -- the
      PRIMARY inference method for v3 because six same-date ETF observations are not six
      independent draws.
    * date_block_bootstrap: resample UNIQUE DATES (optionally in contiguous blocks, to preserve
      serial correlation), not ETF-day rows, so the bootstrap respects the same dependence
      structure the panel regression's clustering respects.
    * date_level_portfolio_test: collapse each date's cross-section into a single high-minus-low
      basket spread, producing ONE observation per date (removes pseudo-replication by
      construction), then applies the existing HAC machinery to that daily spread series.
    * leave_one_date_out: re-run the primary estimate with each date removed in turn.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from api.services.options_flow_research_stats import describe_returns


def _drop_na(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df.dropna(subset=list(cols)).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# A. Panel OLS, ticker fixed effects, cluster-robust (by market_date) standard errors
# --------------------------------------------------------------------------------------------

def panel_ols_cluster_by_date(
    df: pd.DataFrame, y_col: str, x_col: str, ticker_col: str = "ticker", date_col: str = "date",
) -> Dict[str, Any]:
    """
    y = beta * x + ticker_fixed_effects + error, one-way cluster-robust SE clustered by date_col
    (CR1 sandwich estimator with the standard small-sample correction:
        c = (G / (G - 1)) * ((N - 1) / (N - K))
    where G = number of clusters (unique dates), N = observations, K = number of regressors
    including the intercept and all-but-one ticker dummy). t-stat uses a t(G-1) reference.

    Returns beta/se/t/p for x_col only (the fixed effects and intercept are nuisance
    parameters here). n < 2 tickers or < 2 clusters -> None fields, never a fabricated value.
    """
    work = _drop_na(df, [y_col, x_col, ticker_col, date_col])
    n = len(work)
    tickers = sorted(work[ticker_col].unique())
    if n < 10 or len(tickers) < 2:
        return {"n": n, "nClusters": work[date_col].nunique(), "beta": None, "se": None, "tStat": None, "pValue": None}

    y = work[y_col].to_numpy(dtype="float64")
    x = work[x_col].to_numpy(dtype="float64")
    # Design matrix: intercept + x + (len(tickers)-1) ticker dummies (first ticker is reference)
    dummy_tickers = tickers[1:]
    dummies = np.column_stack([
        (work[ticker_col].to_numpy() == t).astype("float64") for t in dummy_tickers
    ]) if dummy_tickers else np.zeros((n, 0))
    X = np.column_stack([np.ones(n), x, dummies])
    k = X.shape[1]

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {"n": n, "nClusters": work[date_col].nunique(), "beta": None, "se": None, "tStat": None, "pValue": None}
    beta_hat = XtX_inv @ X.T @ y
    resid = y - X @ beta_hat

    clusters = work[date_col].to_numpy()
    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    if g < 2:
        return {"n": n, "nClusters": g, "beta": float(beta_hat[1]), "se": None, "tStat": None, "pValue": None}

    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        Xg = X[mask]
        ug = resid[mask]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)

    correction = (g / (g - 1.0)) * ((n - 1.0) / max(n - k, 1))
    vcov = correction * (XtX_inv @ meat @ XtX_inv)
    se_all = np.sqrt(np.clip(np.diag(vcov), 0, None))
    se_x = float(se_all[1])
    beta_x = float(beta_hat[1])
    t_stat = (beta_x / se_x) if se_x > 0 else None
    p_value = None
    if t_stat is not None:
        try:
            from scipy import stats as _stats
            p_value = float(2 * _stats.t.sf(abs(t_stat), df=max(g - 1, 1)))
        except ImportError:
            p_value = None

    return {"n": n, "nClusters": g, "beta": beta_x, "se": se_x, "tStat": t_stat, "pValue": p_value}


def panel_ols_two_regressors_cluster_by_date(
    df: pd.DataFrame, y_col: str, x1_col: str, x2_col: str,
    ticker_col: str = "ticker", date_col: str = "date",
) -> Dict[str, Any]:
    """Same as panel_ols_cluster_by_date but with two continuous regressors (used for the
    market-wide-activity control: x1 = ticker-specific feature, x2 = same-day cross-ETF median).
    Returns beta/se/t for BOTH x1 and x2."""
    work = _drop_na(df, [y_col, x1_col, x2_col, ticker_col, date_col])
    n = len(work)
    tickers = sorted(work[ticker_col].unique())
    empty = {"n": n, "nClusters": work[date_col].nunique() if n else 0,
            "x1": {"beta": None, "se": None, "tStat": None}, "x2": {"beta": None, "se": None, "tStat": None}}
    if n < 10 or len(tickers) < 2:
        return empty

    y = work[y_col].to_numpy(dtype="float64")
    x1 = work[x1_col].to_numpy(dtype="float64")
    x2 = work[x2_col].to_numpy(dtype="float64")
    dummy_tickers = tickers[1:]
    dummies = np.column_stack([
        (work[ticker_col].to_numpy() == t).astype("float64") for t in dummy_tickers
    ]) if dummy_tickers else np.zeros((n, 0))
    X = np.column_stack([np.ones(n), x1, x2, dummies])
    k = X.shape[1]

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return empty
    beta_hat = XtX_inv @ X.T @ y
    resid = y - X @ beta_hat

    clusters = work[date_col].to_numpy()
    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    if g < 2:
        return empty

    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        Xg, ug = X[mask], resid[mask]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    correction = (g / (g - 1.0)) * ((n - 1.0) / max(n - k, 1))
    vcov = correction * (XtX_inv @ meat @ XtX_inv)
    se_all = np.sqrt(np.clip(np.diag(vcov), 0, None))

    def _pack(idx):
        b, s = float(beta_hat[idx]), float(se_all[idx])
        return {"beta": b, "se": s, "tStat": (b / s) if s > 0 else None}

    return {"n": n, "nClusters": g, "x1": _pack(1), "x2": _pack(2)}


# --------------------------------------------------------------------------------------------
# B. Date-block bootstrap (resamples DATES, not ETF-day rows)
# --------------------------------------------------------------------------------------------

def date_block_bootstrap(
    df: pd.DataFrame, statistic_fn: Callable[[pd.DataFrame], Optional[float]],
    date_col: str = "date", n_boot: int = 10_000, block_length: int = 1, seed: int = 42,
) -> Dict[str, Any]:
    """
    Resamples the SEQUENCE of unique dates (sorted) with replacement, in contiguous blocks of
    `block_length` dates (block_length=1 -> ordinary iid date bootstrap; >1 -> a moving/circular
    block bootstrap that preserves short-run serial correlation across consecutive sessions).
    For each resample, pulls every ETF-day row belonging to the drawn dates (rows from a date
    drawn twice are duplicated, which is the correct circular block bootstrap construction) and
    evaluates `statistic_fn` on that resampled panel. Returns the bootstrap distribution's mean,
    2.5th/97.5th percentiles (95% CI), and the share of draws with the same sign as the
    full-sample statistic.
    """
    dates = sorted(df[date_col].unique())
    n_dates = len(dates)
    if n_dates < 5:
        return {"n_boot": 0, "nDates": n_dates, "mean": None, "ci95": (None, None), "shareSameSign": None}

    rng = np.random.default_rng(seed)
    full_stat = statistic_fn(df)
    by_date = {d: df[df[date_col] == d] for d in dates}
    n_blocks_needed = math.ceil(n_dates / block_length)

    stats: List[float] = []
    for _ in range(n_boot):
        starts = rng.integers(0, n_dates, size=n_blocks_needed)
        picked_dates: List[Any] = []
        for s in starts:
            for j in range(block_length):
                picked_dates.append(dates[(s + j) % n_dates])  # circular wrap
        picked_dates = picked_dates[:n_dates]
        sample = pd.concat([by_date[d] for d in picked_dates], ignore_index=True)
        stat = statistic_fn(sample)
        if stat is not None and not (isinstance(stat, float) and math.isnan(stat)):
            stats.append(stat)

    if not stats:
        return {"n_boot": 0, "nDates": n_dates, "mean": None, "ci95": (None, None), "shareSameSign": None}
    arr = np.array(stats, dtype="float64")
    lo, hi = np.percentile(arr, [2.5, 97.5])
    same_sign = None
    if full_stat is not None:
        same_sign = float(np.mean(np.sign(arr) == np.sign(full_stat))) if full_stat != 0 else None
    return {
        "n_boot": len(stats), "nDates": n_dates, "blockLength": block_length,
        "fullSampleStat": full_stat, "mean": float(arr.mean()),
        "ci95": (float(lo), float(hi)), "shareSameSign": same_sign,
    }


# --------------------------------------------------------------------------------------------
# C. Date-level portfolio test (high-minus-low basket spread, one obs per date)
# --------------------------------------------------------------------------------------------

def date_level_portfolio_spread(
    df: pd.DataFrame, feature_col: str, ret_col: str, ticker_col: str = "ticker", date_col: str = "date",
    n_high: int = 3, n_low: int = 3,
) -> pd.DataFrame:
    """
    Per date: rank ETFs by feature_col (predetermined split -- top n_high vs bottom n_low of
    however many ETFs have a non-missing feature value that day, never an optimized threshold),
    equal-weight the high and low baskets' forward returns, and return one row per date with the
    spread. A date needs at least n_high + n_low non-missing (feature, return) pairs to produce a
    row; thinner days are skipped (reported n reflects this).
    """
    rows = []
    for d, day in df.groupby(date_col):
        valid = day.dropna(subset=[feature_col, ret_col])
        if len(valid) < n_high + n_low:
            continue
        ranked = valid.sort_values(feature_col)
        low = ranked.iloc[:n_low]
        high = ranked.iloc[-n_high:]
        rows.append({
            "date": d, "nTickers": len(valid),
            "highBasketReturn": float(high[ret_col].mean()),
            "lowBasketReturn": float(low[ret_col].mean()),
            "spread": float(high[ret_col].mean()) - float(low[ret_col].mean()),
        })
    return pd.DataFrame(rows)


def date_level_portfolio_test(
    df: pd.DataFrame, feature_col: str, ret_col: str, horizon_days: int,
    ticker_col: str = "ticker", date_col: str = "date", n_high: int = 3, n_low: int = 3,
) -> Dict[str, Any]:
    """The date-level portfolio spread series, HAC-tested exactly like any other forward-return
    series (same Bartlett-kernel Newey-West machinery, lag = horizon - 1)."""
    spreads = date_level_portfolio_spread(df, feature_col, ret_col, ticker_col, date_col, n_high, n_low)
    stats = describe_returns(spreads["spread"].tolist() if len(spreads) else [], horizon_days)
    return {"nDates": len(spreads), "nHighBasket": n_high, "nLowBasket": n_low, **stats}


# --------------------------------------------------------------------------------------------
# D. Leave-one-date-out robustness
# --------------------------------------------------------------------------------------------

def leave_one_date_out(
    df: pd.DataFrame, statistic_fn: Callable[[pd.DataFrame], Optional[float]], date_col: str = "date",
) -> Dict[str, Any]:
    """Re-evaluates statistic_fn with each unique date removed in turn. Returns the full-sample
    statistic, the min/max across all leave-one-out runs, the share that keep the full-sample's
    sign, and the 5 most influential dates (largest |leave-one-out estimate - full estimate|)."""
    dates = sorted(df[date_col].unique())
    full_stat = statistic_fn(df)
    results: List[Tuple[Any, Optional[float]]] = []
    for d in dates:
        sub = df[df[date_col] != d]
        results.append((d, statistic_fn(sub)))

    valid = [(d, v) for d, v in results if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid or full_stat is None:
        return {"fullSampleStat": full_stat, "nDates": len(dates), "min": None, "max": None,
               "shareSameSign": None, "mostInfluentialDates": []}
    vals = np.array([v for _, v in valid], dtype="float64")
    same_sign = float(np.mean(np.sign(vals) == np.sign(full_stat))) if full_stat != 0 else None
    influence = sorted(valid, key=lambda dv: -abs(dv[1] - full_stat))[:5]
    return {
        "fullSampleStat": full_stat, "nDates": len(dates),
        "min": float(vals.min()), "max": float(vals.max()), "shareSameSign": same_sign,
        "mostInfluentialDates": [{"date": str(d), "leaveOneOutStat": v, "shift": v - full_stat} for d, v in influence],
    }


def drop_dates_and_recompute(
    df: pd.DataFrame, dates_to_drop: Sequence[Any],
    statistic_fn: Callable[[pd.DataFrame], Optional[float]], date_col: str = "date",
) -> Optional[float]:
    """Drops several dates at once (used for the 'remove the 5 most influential dates
    simultaneously' check) and recomputes the statistic."""
    sub = df[~df[date_col].isin(list(dates_to_drop))]
    return statistic_fn(sub)
