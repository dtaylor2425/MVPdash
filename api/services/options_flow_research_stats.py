"""
api/services/options_flow_research_stats.py

Pure statistics for the Macro Options Flow predictive study (signal discovery only -- no
model fitting, no threshold optimization). No pandas I/O, no file access: everything here
takes plain arrays/sequences and returns plain dicts, so it's fully unit-testable and has no
opinion about where the numbers came from.

Conventions used throughout:
    * Missing values (None / NaN) are ALWAYS dropped before a calculation, never treated as 0.
    * Every result dict carries an explicit "n" -- the actual count used, never assumed.
    * HAC (Newey-West) standard errors use a Bartlett kernel with lag = horizon_days - 1 (the
      standard rule of thumb for daily-sampled, horizon-day overlapping returns), reflecting
      that a 5-day return computed on adjacent days shares 4 days of overlap.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HORIZONS = (1, 3, 5, 10, 20)
NEUTRAL_SENTIMENT_BAND = 0.08  # the product's OWN existing NEUTRAL threshold (options_flow_metrics.py), reused as-is


def _clean(vals: Sequence[Optional[float]]) -> np.ndarray:
    return np.array([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))],
                    dtype="float64")


def hac_lag_for_horizon(horizon_days: int) -> int:
    return max(0, horizon_days - 1)


def newey_west_mean_variance(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of the SAMPLE MEAN of x."""
    n = len(x)
    if n == 0:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(np.sum(xc * xc) / n)
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma_l = float(np.sum(xc[lag:] * xc[:-lag]) / n)
        var += 2.0 * w * gamma_l
    return var / n


def describe_returns(values: Sequence[Optional[float]], horizon_days: int) -> Dict[str, Any]:
    """
    n, mean, median, win_rate (share > 0), std, plain t-stat (vs 0), and the HAC/Newey-West
    t-stat (vs 0) that accounts for the autocorrelation overlapping `horizon_days`-length
    returns necessarily have. All None-safe; n < 2 -> t-stats are None, not a fabricated value.
    """
    x = _clean(values)
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "winRate": None, "std": None,
                "tStat": None, "hacTStat": None, "hacLag": hac_lag_for_horizon(horizon_days)}
    mean = float(x.mean())
    median = float(np.median(x))
    win_rate = float((x > 0).mean())
    std = float(x.std(ddof=1)) if n > 1 else None
    t_stat = (mean / (std / math.sqrt(n))) if (std and std > 0) else None
    lag = hac_lag_for_horizon(horizon_days)
    if n > 1:
        hac_var = newey_west_mean_variance(x, lag)
        hac_se = math.sqrt(hac_var) if hac_var > 0 else None
        hac_t = (mean / hac_se) if hac_se else None
    else:
        hac_t = None
    return {"n": n, "mean": mean, "median": median, "winRate": win_rate, "std": std,
            "tStat": t_stat, "hacTStat": hac_t, "hacLag": lag}


def quintile_labels(values: Sequence[Optional[float]]) -> List[Optional[int]]:
    """1 (lowest) .. 5 (highest) by pooled quantile of the non-missing values, preserving
    input order/length; None stays None. Uses pandas' qcut-equivalent via quantile edges with
    duplicate-edge tolerance (a feature with many ties near 0, e.g. zero_dte_share, can
    collapse quantile bins -- ties are broken by rank, never by inventing separation)."""
    x = np.array([np.nan if v is None else v for v in values], dtype="float64")
    valid_mask = ~np.isnan(x)
    valid = x[valid_mask]
    out: List[Optional[int]] = [None] * len(values)
    if len(valid) < 5:
        return out
    ranks = pd_rank(valid)
    edges = np.quantile(ranks, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(ranks, edges, right=True) + 1
    j = 0
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            out[i] = int(labels[j])
            j += 1
    return out


def pd_rank(x: np.ndarray) -> np.ndarray:
    """Average-rank (1..n), ties split -- avoids a pandas import in a "no pandas I/O" module."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype="float64")
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[Optional[float]], y: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Spearman rank correlation between paired x/y, dropping any pair with a missing value
    on EITHER side (never imputed)."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and not (isinstance(a, float) and math.isnan(a))
             and not (isinstance(b, float) and math.isnan(b))]
    n = len(pairs)
    if n < 3:
        return {"n": n, "rho": None, "pValue": None}
    xs = np.array([p[0] for p in pairs], dtype="float64")
    ys = np.array([p[1] for p in pairs], dtype="float64")
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        return {"n": n, "rho": float(rho), "pValue": float(p)}
    except ImportError:
        rx, ry = pd_rank(xs), pd_rank(ys)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return {"n": n, "rho": rho, "pValue": None}


def classify_agreement(sentiment: Optional[float], delta_ratio: Optional[float],
                       neutral_band: float = NEUTRAL_SENTIMENT_BAND) -> Optional[str]:
    """
    'agree_bullish' | 'agree_bearish' | 'disagreement' | 'approximately_neutral' | None (missing
    input). Uses the sentiment label's OWN existing NEUTRAL band (+-0.08, already defined in
    api/services/options_flow_metrics.py) and plain sign for delta -- no new/optimized cutoff.
    """
    if sentiment is None or delta_ratio is None:
        return None
    if abs(sentiment) <= neutral_band:
        return "approximately_neutral"
    if sentiment > 0 and delta_ratio > 0:
        return "agree_bullish"
    if sentiment < 0 and delta_ratio < 0:
        return "agree_bearish"
    return "disagreement"


def median_split(values: Sequence[Optional[float]]) -> Tuple[Optional[float], List[Optional[str]]]:
    """('high'/'low', per-row) split on the pooled median of the non-missing values."""
    x = _clean(values)
    if len(x) < 2:
        return None, [None] * len(values)
    med = float(np.median(x))
    out = [None if v is None or (isinstance(v, float) and math.isnan(v)) else ("high" if v > med else "low")
          for v in values]
    return med, out


def tercile_split(values: Sequence[Optional[float]]) -> Tuple[Optional[Tuple[float, float]], List[Optional[str]]]:
    """('low'/'mid'/'high', per-row) split on pooled terciles of the non-missing values."""
    x = _clean(values)
    if len(x) < 3:
        return None, [None] * len(values)
    lo, hi = np.quantile(x, [1 / 3, 2 / 3])

    def _lab(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if v <= lo:
            return "low"
        if v >= hi:
            return "high"
        return "mid"
    return (float(lo), float(hi)), [_lab(v) for v in values]


def sign_bucket(v: Optional[float], neutral_band: float = 0.0) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if v > neutral_band:
        return "positive"
    if v < -neutral_band:
        return "negative"
    return "neutral"
