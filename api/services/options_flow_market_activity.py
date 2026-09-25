"""
api/services/options_flow_market_activity.py

THE canonical, single, production implementation of the market-activity indicator that backs
the /internal/options-flow page's hero section. Frontend never reinvents this formula -- it only
ever receives the derived numbers this module computes, server-side.

Mathematically identical to Signal Research v4 (commit 7629c93) / Signal Validation v3's frozen
z_gross_premium_20d (see reports/options-flow-v3-validation-manifest.json):

    z_gross_premium_20d(T, ticker) = (gross_premium(T) - rolling_mean(prior 20)) / rolling_std(prior 20)
    market_activity_z(T) = median(z_gross_premium_20d(T, ticker) for ticker in ACTIVITY_TICKERS)
    residual_activity(T, ticker) = z_gross_premium_20d(T, ticker) - market_activity_z(T)

`tests/test_options_flow_market_activity.py::test_pg_matches_research_calculation_on_historical_dates`
is the mandatory regression test comparing this module's output against the frozen research
scripts' output on real historical dates -- required by design, not optional cleanup. If that
test ever fails after a change here, the change is wrong; do not "fix" the test.

Regime bands (QUIET/NORMAL/ELEVATED/EXTREME) are DESCRIPTIVE activity-level labels derived from
the historical percentile distribution, not trading thresholds and not optimized for returns.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from api.services.options_flow_phase1_scope import PHASE1_TICKERS

# The exact 6-ticker universe Signal Research v1-v4 validated. Deliberately NOT derived from
# OPTIONS_FLOW_UNIVERSE (which has 22 tickers across 6 groups) -- market_activity_z is defined,
# by the research, as the median across precisely these six, and only these six ever received a
# full_flow historical backfill.
ACTIVITY_TICKERS: List[str] = list(PHASE1_TICKERS)

# Verbatim from commit 19dc02c / the v3 manifest. Do not change without a new, separately
# labeled research iteration -- this constant IS the methodology.
Z_WINDOW = 20
Z_MIN_PERIODS = 10

REGIME_BANDS = (
    (25.0, "QUIET"),
    (75.0, "NORMAL"),
    (95.0, "ELEVATED"),
    (None, "EXTREME"),
)

MIN_PERCENTILE_OBS = 20  # below this, percentile/regime are None rather than a noisy guess


def load_gross_premium_history(conn, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    All available (date, ticker, gross_premium) observations from successful/partial full_flow
    runs, for the given tickers (default ACTIVITY_TICKERS). One row per (ticker, market_date) --
    the latest run wins if a date was ever reprocessed. Read-only.
    """
    tickers = tickers or ACTIVITY_TICKERS
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (s.ticker, s.market_date)
                   s.ticker, s.market_date,
                   (s.payload -> 'premium' ->> 'gross')::double precision AS gross_premium
            FROM options_flow_symbol_snapshots s
            JOIN options_flow_runs r ON r.id = s.run_id
            WHERE s.mode = 'full_flow' AND s.source = 'historical_backfill'
              AND r.status IN ('success', 'partial') AND s.ticker = ANY(%(tickers)s)
            ORDER BY s.ticker, s.market_date, s.created_at DESC
            """,
            {"tickers": tickers},
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "gross_premium"])
    df["date"] = df["market_date"].astype(str)
    df = df.drop(columns=["market_date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def rolling_zscore_20d(df: pd.DataFrame, col: str, ticker_col: str = "ticker") -> pd.Series:
    """
    THE canonical rolling z-score -- byte-identical math to commit 19dc02c's `_zscore` closure
    (verified by the mandatory regression test). `df` must be sorted by (ticker, date) with no
    gaps introduced. Current date T is excluded from its own baseline via shift(1) before
    rolling; sample std (ddof=1, pandas default); NaN during warmup or a zero-variance window,
    never zero-filled.
    """
    g = df.groupby(ticker_col, sort=False)
    prior = g[col].shift(1)
    roll_mean = prior.groupby(df[ticker_col]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    roll_std = prior.groupby(df[ticker_col]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
    roll_mean.index = roll_mean.index.droplevel(0)
    roll_std.index = roll_std.index.droplevel(0)
    z = (df[col] - roll_mean) / roll_std
    return z.replace([np.inf, -np.inf], np.nan)


def market_activity_frame(df: pd.DataFrame, min_tickers: int = 3) -> pd.DataFrame:
    """
    df: (ticker, date, gross_premium) history, any order. Returns one row per (ticker, date)
    with z_gross_premium_20d, market_activity_median, market_activity_mean, n_tickers, and
    residual_activity -- everything downstream (percentile, regime, breadth, the chart series)
    is derived from this frame.
    """
    work = df.copy()
    work["_dt"] = pd.to_datetime(work["date"])
    work = work.sort_values(["ticker", "_dt"]).reset_index(drop=True)
    work["z_gross_premium_20d"] = rolling_zscore_20d(work, "gross_premium")

    by_date = work.groupby("date")["z_gross_premium_20d"]
    n_tickers = by_date.apply(lambda s: int(s.notna().sum()))
    median = by_date.median()
    mean = by_date.mean()
    market = pd.DataFrame({"n_tickers": n_tickers, "market_activity_median": median,
                          "market_activity_mean": mean})
    market.loc[market["n_tickers"] < min_tickers, ["market_activity_median", "market_activity_mean"]] = np.nan

    work = work.merge(market, left_on="date", right_index=True, how="left")
    work["residual_activity"] = work["z_gross_premium_20d"] - work["market_activity_median"]
    return work.drop(columns=["_dt"])


def historical_percentile(current: Optional[float], history: List[Optional[float]]) -> Optional[float]:
    """% of `history` (excluding NaN/None) at or below `current`, in [0, 100]. None if fewer
    than MIN_PERCENTILE_OBS valid prior observations or current is None -- never a fabricated
    percentile from a thin sample."""
    if current is None or (isinstance(current, float) and np.isnan(current)):
        return None
    vals = np.array([v for v in history if v is not None and not (isinstance(v, float) and np.isnan(v))],
                    dtype="float64")
    if len(vals) < MIN_PERCENTILE_OBS:
        return None
    return float((vals <= current).mean() * 100.0)


def activity_regime(percentile: Optional[float]) -> Optional[str]:
    """QUIET (<25th) / NORMAL (25-75th) / ELEVATED (75-95th) / EXTREME (>95th) -- descriptive
    activity-regime bands derived from the percentile distribution, not optimized trading
    thresholds. None when percentile itself is None (insufficient history)."""
    if percentile is None:
        return None
    for cutoff, label in REGIME_BANDS:
        if cutoff is None or percentile < cutoff:
            return label
    return "EXTREME"


def breadth(z_by_ticker: Dict[str, Optional[float]]) -> Dict[str, int]:
    """How many of the tracked tickers are currently above THEIR OWN normal activity level
    (z > 0), out of how many have a valid z at all today."""
    valid = [v for v in z_by_ticker.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    above = sum(1 for v in valid if v > 0)
    return {"aboveNormal": above, "total": len(valid), "tracked": len(z_by_ticker)}


def compute_market_activity_snapshot(
    conn, as_of_date: Optional[date] = None, chart_sessions: int = 252,
) -> Optional[Dict[str, Any]]:
    """
    The one function the API layer calls. Returns None if there isn't enough history to say
    anything (never fabricates a snapshot from insufficient data). Otherwise:

        {
          "asOfDate": "YYYY-MM-DD",
          "value": float | None,           # market_activity_z, median (primary)
          "valueMean": float | None,        # secondary, equal-weight mean
          "percentile": float | None,       # of `value` within its own trailing history
          "regime": str | None,             # QUIET | NORMAL | ELEVATED | EXTREME | None
          "breadth": {"aboveNormal": int, "total": int, "tracked": int},
          "history": [{"date": str, "value": float | None}, ...],   # up to chart_sessions
          "zByTicker": {ticker: float | None},
          "residualByTicker": {ticker: float | None},
          "tickers": [...],                 # ACTIVITY_TICKERS, for the frontend to iterate
        }
    """
    raw = load_gross_premium_history(conn)
    if raw.empty:
        return None
    frame = market_activity_frame(raw)

    dates_sorted = sorted(frame["date"].unique())
    target_date = as_of_date.isoformat() if as_of_date else dates_sorted[-1]
    if target_date not in dates_sorted:
        # fall back to the latest date at or before the requested one
        earlier = [d for d in dates_sorted if d <= target_date]
        if not earlier:
            return None
        target_date = earlier[-1]

    market_series = (
        frame.drop_duplicates("date")[["date", "market_activity_median", "market_activity_mean"]]
        .set_index("date")
    )
    current_value = market_series.loc[target_date, "market_activity_median"]
    current_value = None if pd.isna(current_value) else float(current_value)
    current_mean = market_series.loc[target_date, "market_activity_mean"]
    current_mean = None if pd.isna(current_mean) else float(current_mean)

    prior_dates = [d for d in dates_sorted if d < target_date]
    prior_history = [
        (None if pd.isna(v) else float(v))
        for v in market_series.loc[market_series.index.isin(prior_dates), "market_activity_median"]
    ]
    percentile = historical_percentile(current_value, prior_history)
    regime = activity_regime(percentile)

    today_rows = frame[frame["date"] == target_date].set_index("ticker")
    z_by_ticker = {t: (None if t not in today_rows.index or pd.isna(today_rows.loc[t, "z_gross_premium_20d"])
                       else float(today_rows.loc[t, "z_gross_premium_20d"])) for t in ACTIVITY_TICKERS}
    residual_by_ticker = {t: (None if t not in today_rows.index or pd.isna(today_rows.loc[t, "residual_activity"])
                              else float(today_rows.loc[t, "residual_activity"])) for t in ACTIVITY_TICKERS}

    chart_dates = dates_sorted[-chart_sessions:]
    history = [
        {"date": d, "value": (None if pd.isna(market_series.loc[d, "market_activity_median"])
                              else float(market_series.loc[d, "market_activity_median"]))}
        for d in chart_dates if d in market_series.index
    ]

    return {
        "asOfDate": target_date,
        "value": current_value,
        "valueMean": current_mean,
        "percentile": percentile,
        "regime": regime,
        "breadth": breadth(z_by_ticker),
        "history": history,
        "zByTicker": z_by_ticker,
        "residualByTicker": residual_by_ticker,
        "tickers": ACTIVITY_TICKERS,
    }
