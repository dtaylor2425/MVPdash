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

import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from api.services.options_flow_phase1_scope import PHASE1_TICKERS
from api.services.options_flow_profile import profile_identity

ROOT = Path(__file__).resolve().parents[2]
V4_REPORT_PATH = ROOT / "reports" / "options-flow-market-factor-v4.json"

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
_ACTIVITY_CACHE: Dict[Any, Any] = {}


def load_gross_premium_history(conn, tickers: Optional[List[str]] = None, include_live: bool = False) -> pd.DataFrame:
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
                   s.ticker, s.market_date, s.id::text AS snapshot_id,
                   s.methodology_version, r.config AS collection_config,
                   (s.payload -> 'premium' ->> 'gross')::double precision AS gross_premium
            FROM options_flow_symbol_snapshots s
            JOIN options_flow_runs r ON r.id = s.run_id
            WHERE s.mode = 'full_flow' AND (s.source = 'historical_backfill'
              OR (%(include_live)s AND s.source = 'live' AND s.payload->'publication'->>'state' = 'final'))
              AND r.status IN ('success', 'partial') AND s.ticker = ANY(%(tickers)s)
            ORDER BY s.ticker, s.market_date, (s.source='live') DESC, s.as_of_timestamp DESC, s.created_at DESC, s.id DESC
            """,
            {"tickers": tickers, "include_live": include_live},
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


def market_activity_frame(df: pd.DataFrame, min_tickers: int = 3, baseline_group_col: str = "ticker") -> pd.DataFrame:
    """
    df: (ticker, date, gross_premium) history, any order. Returns one row per (ticker, date)
    with z_gross_premium_20d, market_activity_median, market_activity_mean, n_tickers, and
    residual_activity -- everything downstream (percentile, regime, breadth, the chart series)
    is derived from this frame.
    """
    work = df.copy()
    work["_dt"] = pd.to_datetime(work["date"])
    work = work.sort_values(["ticker", "_dt"]).reset_index(drop=True)
    work["z_gross_premium_20d"] = rolling_zscore_20d(work, "gross_premium", ticker_col=baseline_group_col)

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


def compatible_activity_segments(raw: pd.DataFrame) -> pd.DataFrame:
    """Break operational baselines at profile changes, unknown profiles or gaps.

    The same prior-20/minimum-10 calculation then operates on each uninterrupted
    segment. Earlier rows are retained for display, never pooled across profiles.
    """
    from api.services.options_flow_calendar import previous_trading_day
    work = raw.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    profiles, segments = [], []
    last = {}
    for row in work.to_dict("records"):
        ticker, day = row["ticker"], date.fromisoformat(str(row["date"]))
        profile = profile_identity(row.get("methodology_version"), row.get("collection_config"))
        prior = last.get(ticker)
        same = bool(prior and profile and profile == prior[1] and previous_trading_day(day) == prior[0])
        segment = prior[2] if same else (prior[2]+1 if prior else 0)
        profiles.append(profile)
        segments.append(f"{ticker}:{segment}")
        last[ticker] = (day, profile, segment)
    work["activity_profile"] = profiles
    work["activity_segment"] = segments
    return work


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


_V4_SUMMARY_CACHE: Dict[str, Any] = {}


def load_v4_research_summary() -> Optional[Dict[str, Any]]:
    """
    The handful of numbers the page's "Historical Behavior" card quotes, read directly from the
    committed Signal Research v4 report (reports/options-flow-market-factor-v4.json, commit
    7629c93) -- never hand-copied into frontend code, so the copy can't silently drift from the
    actual study. Cached in-process (the file never changes at runtime); returns None if the
    report file isn't present in this deployment rather than raising.
    """
    if "data" in _V4_SUMMARY_CACHE:
        return _V4_SUMMARY_CACHE["data"]
    try:
        raw = json.loads(V4_REPORT_PATH.read_text(encoding="utf-8"))
        spy = raw["marketBattery"]["SPY"]

        def _corr(outcome_key: str) -> Dict[str, Optional[float]]:
            sp = spy.get(outcome_key, {}).get("spearman", {})
            return {"rho": sp.get("rho"), "pValue": sp.get("pValue"), "n": sp.get("n")}

        summary = {
            "interpretation": "Descriptive activity indicator; directional forecasting is not established.",
            "correlationPredictor": "Activity quintile labels",
            "sampleUse": "Combined discovery and validation sample; not independent forward validation",
            "verdict": raw.get("verdict"),
            "spyReturn5d": _corr("ret_5d"),
            "spyReturn10d": _corr("ret_10d"),
            "spyVol10d": _corr("fwd_vol_10d"),
            "vixChange10d": _corr("vix_change_10d"),
            "spyMaxDrawdown10d": _corr("fwd_max_drawdown_10d"),
        }
    except Exception as e:
        print("[options-flow] could not load v4 research summary: {}".format(e))
        summary = None
    _V4_SUMMARY_CACHE["data"] = summary
    return summary


def compute_market_activity_snapshot(
    conn, as_of_date: Optional[date] = None, chart_sessions: int = 252, include_live: bool = False,
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
    cache_key = None
    if include_live:
        # Identity changes for additions, replacements and pruning. Research reads
        # deliberately bypass this operational cache. No unbounded per-poll pandas.
        with conn.cursor() as cur:
            cur.execute("""SELECT md5(string_agg(s.id::text, ',' ORDER BY s.id)) AS identity
                FROM options_flow_symbol_snapshots s JOIN options_flow_runs r ON r.id=s.run_id
                WHERE s.mode='full_flow' AND s.ticker=ANY(%s)
                  AND r.status IN ('success','partial')
                  AND (s.source='historical_backfill' OR
                       (s.source='live' AND s.payload->'publication'->>'state'='final'))""", (ACTIVITY_TICKERS,))
            identity = cur.fetchone()["identity"]
        cache_key = (identity, as_of_date, chart_sessions)
        if cache_key in _ACTIVITY_CACHE:
            return deepcopy(_ACTIVITY_CACHE[cache_key])
    raw = load_gross_premium_history(conn, include_live=True) if include_live else load_gross_premium_history(conn)
    if raw.empty:
        return None
    frame = market_activity_frame(compatible_activity_segments(raw), baseline_group_col="activity_segment") if include_live else market_activity_frame(raw)

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
    profiles_by_date = {}
    if include_live:
        for d, group in frame.groupby("date"):
            profiles_by_date[d] = {r["ticker"]: r["activity_profile"] for r in group.to_dict("records")
                                   if pd.notna(r["z_gross_premium_20d"]) and r["activity_profile"] is not None}
    current_value = market_series.loc[target_date, "market_activity_median"]
    current_value = None if pd.isna(current_value) else float(current_value)
    current_mean = market_series.loc[target_date, "market_activity_mean"]
    current_mean = None if pd.isna(current_mean) else float(current_mean)

    prior_dates = [d for d in dates_sorted if d < target_date and (not include_live or profiles_by_date[d] == profiles_by_date[target_date])]
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

    chart_dates = [d for d in dates_sorted if d <= target_date][-chart_sessions:]
    history = [
        {"date": d, "value": (None if pd.isna(market_series.loc[d, "market_activity_median"])
                              else float(market_series.loc[d, "market_activity_median"]))}
        for d in chart_dates if d in market_series.index
    ]
    from api.services.options_flow_calendar import previous_trading_day
    for point in history if include_live else []:
        d = point["date"]
        point["snapshotIds"] = frame.loc[frame["date"] == d, "snapshot_id"].tolist() if "snapshot_id" in frame.columns else []
        vals = frame[frame["date"] == d].set_index("ticker")["z_gross_premium_20d"]
        point["availableTickers"] = [t for t in ACTIVITY_TICKERS if t in vals.index and pd.notna(vals[t])]
        point["collectionProfiles"] = profiles_by_date[d]
        point["breadth"] = breadth({t: (float(vals[t]) if t in vals.index and pd.notna(vals[t]) else None) for t in ACTIVITY_TICKERS})
        point["percentile"] = historical_percentile(point["value"], [
            None if pd.isna(v) else float(v) for v in market_series.loc[
                market_series.index.isin([prior for prior in dates_sorted if prior < d and profiles_by_date[prior] == profiles_by_date[d]]),
                "market_activity_median"]])
    # Consecutive elevated completed sessions, reset by a missing session.
    by_day = {p["date"]: p for p in history}
    streak, day = 0, date.fromisoformat(target_date)
    while day.isoformat() in by_day:
        p = by_day[day.isoformat()]
        if include_live and p.get("collectionProfiles") != profiles_by_date[target_date]:
            break
        if p.get("percentile") is None or p["percentile"] < 75:
            break
        streak += 1
        day = previous_trading_day(day)

    result = {
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
        "elevatedSessionStreak": streak,
    }
    if cache_key is not None:
        if len(_ACTIVITY_CACHE) >= 8:
            _ACTIVITY_CACHE.clear()
        _ACTIVITY_CACHE[cache_key] = deepcopy(result)
    return result
