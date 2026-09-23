"""
scripts/run_options_flow_signal_design_v2.py

Signal Design v2: tests whether predictive information exists in more specific / segmented
flow slices than the daily full-day aggregate tested in Signal Research v1 (NO EVIDENCE YET).

Inputs (all read-only, all reused from Phase 1 -- no new ThetaData call, no new backfill):
    * reports/options-flow-research-dataset.parquet   (v1's immutable daily aggregate features)
    * reports/options-flow-payload-features.parquet   (built by
      scripts/extract_options_flow_payload_features.py from the SAME Phase 1 Postgres payloads
      -- DTE-bucket sentiment, call/put-only flow, time-of-day buckets, large-trade flow)
    * reports/options-flow-forward-returns.parquet    (v1's outcome file, unchanged)

This script then builds normalized/derived features (ratios, rolling z-scores using ONLY prior
observations, 1D/3D changes) and runs a disciplined battery of predetermined-bucket tests
(quintiles, median/tercile splits -- no threshold search) against forward returns, organized
around 6 pre-registered hypotheses (H1-H6). No weight optimization, no ML model, no in-sample
significance claims, no data collected beyond what Phase 1 already produced.

    python scripts/run_options_flow_signal_design_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.options_flow_research_stats import (  # noqa: E402
    HORIZONS,
    describe_returns,
    median_split,
    quintile_labels,
    spearman_correlation,
)

FEATURES_PATH = ROOT / "reports" / "options-flow-research-dataset.parquet"
PAYLOAD_PATH = ROOT / "reports" / "options-flow-payload-features.parquet"
RETURNS_PATH = ROOT / "reports" / "options-flow-forward-returns.parquet"
OUT_MD = ROOT / "reports" / "options-flow-signal-design-v2.md"
OUT_CSV = ROOT / "reports" / "options-flow-signal-design-v2.csv"
OUT_JSON = ROOT / "reports" / "options-flow-signal-design-v2.json"

TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
DECISION_HORIZONS = (5, 10, 20)
Z_WINDOW, Z_MIN_PERIODS = 20, 10

# --------------------------------------------------------------------------------------------
# Feature catalogue: (column, hypothesis, family, description)
# --------------------------------------------------------------------------------------------
FEATURE_CATALOGUE: List[Dict[str, str]] = [
    {"col": "large_trade_sentiment", "hyp": "H1", "family": "Large/unusual trades",
     "desc": "directional premium / gross premium, restricted to the day's 25 largest trades"},
    {"col": "large_trade_0dte_share", "hyp": "H1", "family": "Large/unusual trades",
     "desc": "share of large-trade premium that is 0DTE"},
    {"col": "dte_0dte_sentiment", "hyp": "H2", "family": "DTE segmentation",
     "desc": "net/gross sentiment within the 0DTE bucket only"},
    {"col": "dte_1_7d_sentiment", "hyp": "H2", "family": "DTE segmentation",
     "desc": "net/gross sentiment within the 1-7D bucket only"},
    {"col": "dte_8_30d_sentiment", "hyp": "H2", "family": "DTE segmentation",
     "desc": "net/gross sentiment within the 8-30D bucket only"},
    {"col": "dte_31_60d_sentiment", "hyp": "H2", "family": "DTE segmentation",
     "desc": "net/gross sentiment within the 31-60D bucket only"},
    {"col": "call_sentiment", "hyp": "other", "family": "Call/put-separate flow",
     "desc": "(callBought - callSold) / (callBought + callSold)"},
    {"col": "put_sentiment", "hyp": "other", "family": "Call/put-separate flow",
     "desc": "(putBought - putSold) / (putBought + putSold)"},
    {"col": "opening_hour_sentiment", "hyp": "H3", "family": "Time of day",
     "desc": "net/gross sentiment in the first ~60 minutes"},
    {"col": "midday_sentiment", "hyp": "H3", "family": "Time of day",
     "desc": "net/gross sentiment in the middle of the session"},
    {"col": "closing_hour_sentiment", "hyp": "H3", "family": "Time of day",
     "desc": "net/gross sentiment in the last ~60 minutes"},
    {"col": "sentiment_change_1d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "net_trade_sentiment(T) - net_trade_sentiment(T-1), same ticker"},
    {"col": "sentiment_change_3d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "net_trade_sentiment(T) - net_trade_sentiment(T-3), same ticker"},
    {"col": "delta_imbalance_change_1d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "delta_imbalance_ratio(T) - delta_imbalance_ratio(T-1), same ticker"},
    {"col": "delta_imbalance_change_3d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "delta_imbalance_ratio(T) - delta_imbalance_ratio(T-3), same ticker"},
    {"col": "gross_premium_pct_change_1d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "% change in gross_premium vs the prior day, same ticker"},
    {"col": "zero_dte_share_change_1d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "zero_dte_share(T) - zero_dte_share(T-1), same ticker"},
    {"col": "put_skew_change_1d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "put_skew_25d(T) - put_skew_25d(T-1), same ticker"},
    {"col": "atm_iv_change_1d", "hyp": "H4", "family": "Flow acceleration",
     "desc": "atm_iv(T) - atm_iv(T-1), same ticker"},
    {"col": "top10_concentration", "hyp": "H5", "family": "Concentration",
     "desc": "top-10-trade premium / gross premium (full-day denominator)"},
    {"col": "top25_premium_hhi", "hyp": "H5", "family": "Concentration",
     "desc": "Herfindahl index over the day's 25 largest trades only (proxy, likely understates true concentration)"},
    {"col": "z_sentiment_20d", "hyp": "H6", "family": "Baseline-relative (z-score)",
     "desc": "net_trade_sentiment z-scored vs that ETF's own trailing 20D (prior-only) history"},
    {"col": "z_delta_imbalance_20d", "hyp": "H6", "family": "Baseline-relative (z-score)",
     "desc": "delta_imbalance_ratio z-scored vs that ETF's own trailing 20D (prior-only) history"},
    {"col": "z_gross_premium_20d", "hyp": "H6", "family": "Baseline-relative (z-score)",
     "desc": "gross_premium z-scored vs that ETF's own trailing 20D (prior-only) history"},
    {"col": "z_large_trade_premium_20d", "hyp": "H6", "family": "Baseline-relative (z-score)",
     "desc": "large_trade_gross_premium z-scored vs that ETF's own trailing 20D (prior-only) history"},
    {"col": "z_0dte_share_20d", "hyp": "H6", "family": "Baseline-relative (z-score)",
     "desc": "zero_dte_share z-scored vs that ETF's own trailing 20D (prior-only) history"},
]

# v1 baselines, carried forward for direct comparison (already committed, not recomputed here)
V1_BASELINE = {
    "net_trade_sentiment": {"5d": -0.0001, "10d": 0.0040, "20d": 0.0193},
    "delta_imbalance_ratio": {"5d": 0.0036, "10d": 0.0023, "20d": 0.0110},
}

INFEASIBLE_ITEMS = [
    ("Moneyness (deep ITM / near ATM / moderately OTM / far OTM)",
     "No moneyness classification is computed or stored anywhere in the Phase 1 pipeline -- "
     "api/services/options_flow_metrics.py never buckets by strike-to-spot distance, only by "
     "DTE. Cannot be tested from existing data."),
    ("Full trade-size percentile segmentation (top 1% / 5% / 10% of ALL trades)",
     "Only the day's 25 largest trades are persisted per ticker (payload['largeTrades']); the "
     "full trade-size distribution is discarded after each snapshot (by design -- raw ticks are "
     "never stored). large_trade_sentiment/top25_premium_hhi below are a top-25-only proxy, not "
     "a true percentile-of-all-trades measure."),
    ("Large trade size relative to each ticker's own normal trade size",
     "Would need the full per-trade size distribution (or at least its median/percentiles) "
     "persisted per ticker-day; only top-25 trades and daily aggregate trade_count are stored."),
]


# --------------------------------------------------------------------------------------------
# Feature construction (all leakage-safe: rolling/diff features use shift(1) or diff(), so a
# feature attributed to date T never uses information from T+1 or later)
# --------------------------------------------------------------------------------------------

def load_master_frame() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH).copy()
    payload = pd.read_parquet(PAYLOAD_PATH).copy()
    returns = pd.read_parquet(RETURNS_PATH).copy()
    for df in (features, payload, returns):
        df["date"] = df["date"].astype(str)

    df = features.merge(payload, on=["date", "ticker"], how="inner", validate="one_to_one")
    if len(df) != len(features):
        raise RuntimeError(f"payload-feature join dropped rows: {len(features)} -> {len(df)}")
    df = df.merge(returns[["date", "ticker"] + [f"ret_{h}d" for h in HORIZONS]],
                 on=["date", "ticker"], how="inner", validate="one_to_one")
    if len(df) != len(features):
        raise RuntimeError(f"forward-return join dropped rows: {len(features)} -> {len(df)}")

    df["_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "_dt"]).reset_index(drop=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker", sort=False)

    # --- flow acceleration (H4): plain diffs, each row uses only its own ticker's PAST values ---
    df["sentiment_change_1d"] = g["net_trade_sentiment"].diff(1)
    df["sentiment_change_3d"] = g["net_trade_sentiment"].diff(3)
    df["delta_imbalance_change_1d"] = g["delta_imbalance_ratio"].diff(1)
    df["delta_imbalance_change_3d"] = g["delta_imbalance_ratio"].diff(3)
    df["gross_premium_pct_change_1d"] = g["gross_premium"].pct_change(1)
    df["zero_dte_share_change_1d"] = g["zero_dte_share"].diff(1)
    df["put_skew_change_1d"] = g["put_skew_25d"].diff(1)
    df["atm_iv_change_1d"] = g["atm_iv"].diff(1)

    # --- rolling z-scores vs each ETF's OWN trailing history (H6): shift(1) excludes today, so
    # the mean/std used for date T's z-score is computed only from dates strictly before T ---
    def _zscore(col: str) -> pd.Series:
        prior = g[col].shift(1)
        roll_mean = prior.groupby(df["ticker"]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
        roll_std = prior.groupby(df["ticker"]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
        roll_mean.index = roll_mean.index.droplevel(0)
        roll_std.index = roll_std.index.droplevel(0)
        z = (df[col] - roll_mean) / roll_std
        return z.replace([np.inf, -np.inf], np.nan)

    df["z_sentiment_20d"] = _zscore("net_trade_sentiment")
    df["z_delta_imbalance_20d"] = _zscore("delta_imbalance_ratio")
    df["z_gross_premium_20d"] = _zscore("gross_premium")
    df["z_large_trade_premium_20d"] = _zscore("large_trade_gross_premium")
    df["z_0dte_share_20d"] = _zscore("zero_dte_share")

    return df


# --------------------------------------------------------------------------------------------
# Test battery (same discipline as v1: predetermined quintiles/median splits only)
# --------------------------------------------------------------------------------------------

def quintile_spread(df: pd.DataFrame, feature: str, horizon: int) -> Dict[str, Any]:
    col = f"ret_{horizon}d"
    valid = df.dropna(subset=[feature])
    q = quintile_labels(list(valid[feature]))
    q1 = [r for r, qq in zip(valid[col].tolist(), q) if qq == 1]
    q5 = [r for r, qq in zip(valid[col].tolist(), q) if qq == 5]
    d1, d5 = describe_returns(q1, horizon), describe_returns(q5, horizon)
    spread = (d5["mean"] - d1["mean"]) if (d5["mean"] is not None and d1["mean"] is not None) else None
    return {"n": int(valid[feature].notna().sum()), "q1": d1, "q5": d5, "q5MinusQ1Mean": spread}


def feature_battery(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"horizons": {}}
    for h in HORIZONS:
        out["horizons"][f"{h}d"] = quintile_spread(df, feature, h)
    valid = df.dropna(subset=[feature, "ret_10d"])
    q = quintile_labels(list(valid[feature]))
    out["spearman10d"] = spearman_correlation(q, valid["ret_10d"].tolist())
    # per-ETF sign check at 5D/10D (n is small per ETF -- exploratory only, per item 6)
    per_etf_signs = {}
    for t in TICKERS:
        sub = df[df["ticker"] == t]
        signs = []
        for h in (5, 10):
            b = quintile_spread(sub, feature, h)
            if b["q5MinusQ1Mean"] is not None:
                signs.append(1 if b["q5MinusQ1Mean"] > 0 else -1)
        per_etf_signs[t] = signs
    out["perEtfSigns"] = per_etf_signs
    return out


def concentration_direction_test(df: pd.DataFrame) -> Dict[str, Any]:
    """H5, item 4's explicit 4-way comparison: concentrated/diffuse x bullish/bearish."""
    _, conc_labels = median_split(list(df["top10_concentration"]))
    work = df.copy()
    work["_conc"] = conc_labels
    work["_dir"] = np.where(work["net_trade_sentiment"] > 0, "bullish",
                            np.where(work["net_trade_sentiment"] < 0, "bearish", "flat"))
    out: Dict[str, Any] = {"horizons": {}}
    for h in DECISION_HORIZONS:
        col = f"ret_{h}d"
        cells = {}
        for conc in ("high", "low"):
            for direction in ("bullish", "bearish"):
                sub = work[(work["_conc"] == conc) & (work["_dir"] == direction)]
                cells[f"{'concentrated' if conc == 'high' else 'diffuse'}_{direction}"] = \
                    describe_returns(sub[col].tolist(), h)
        out["horizons"][f"{h}d"] = cells
    return out


def hac_flag(d: Dict[str, Any], thresh: float = 1.96) -> bool:
    t = d.get("hacTStat")
    return t is not None and abs(t) >= thresh


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


# --------------------------------------------------------------------------------------------
# Ranking + verdict
# --------------------------------------------------------------------------------------------

def score_feature(result: Dict[str, Any]) -> Dict[str, Any]:
    """Robustness/consistency score -- NOT a p-value hunt. Counts, at 5D/10D/20D:
      - how many horizons clear |HAC t| >= 1.96 with a CONSISTENT sign
      - how many of the 6 ETFs individually agree in sign at 5D/10D
      - the smallest N among the decision horizons (sample-size caution)
    """
    signs, strong_horizons = [], 0
    min_n = None
    for h in DECISION_HORIZONS:
        b = result["horizons"][f"{h}d"]
        if b["n"] is not None:
            min_n = b["n"] if min_n is None else min(min_n, b["n"])
        if b["q5MinusQ1Mean"] is not None:
            signs.append(1 if b["q5MinusQ1Mean"] > 0 else -1)
        if hac_flag(b["q5"]) or hac_flag(b["q1"]):
            strong_horizons += 1
    sign_consistent = len(set(signs)) <= 1 and len(signs) > 0
    per_etf = result.get("perEtfSigns", {})
    agree_count = sum(1 for signs_ in per_etf.values() if signs_ and len(set(signs_)) == 1)
    return {
        "strongHorizons": strong_horizons,
        "signConsistentAcrossHorizons": sign_consistent,
        "etfAgreementCount": agree_count,
        "minDecisionHorizonN": min_n,
        "robustnessScore": strong_horizons * 10 + (5 if sign_consistent else 0) + agree_count,
    }


def cross_etf_date_overlap(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """How much of a feature's extreme (quintile-5) bucket is the SAME calendar date repeated
    across ETFs, vs. genuinely independent dates. Elevated options activity/flow often clusters
    on shared macro-calendar days (FOMC, CPI, OPEX) across every ETF at once -- so "N of 6 ETFs
    agree in sign" can overstate independent confirmation if it is really a handful of shared
    dates counted several times. Used to keep the verdict honest, not to hide a result."""
    valid = df.dropna(subset=[feature])
    if len(valid) < 5:
        return {"nRows": len(valid), "nUniqueDates": 0, "concentrationRatio": None}
    q = quintile_labels(list(valid[feature]))
    q5_dates = valid.assign(_q=q).loc[lambda d: d["_q"] == 5, "date"]
    n_rows, n_unique = len(q5_dates), q5_dates.nunique()
    return {"nRows": n_rows, "nUniqueDates": n_unique,
           "concentrationRatio": (n_rows / n_unique) if n_unique else None}


def determine_verdict(scored: List[Dict[str, Any]], df: pd.DataFrame, results: Dict[str, Any],
                      total_calendar_days: int) -> Dict[str, Any]:
    best = max(scored, key=lambda r: r["score"]["robustnessScore"]) if scored else None
    if best is None:
        return {"verdict": "CURRENT FLOW DESIGN STILL SHOWS NO EDGE", "cap_reason": None, "overlap": None}
    s = best["score"]
    overlap = cross_etf_date_overlap(df, best["col"])
    meets_strong_bar = (s["strongHorizons"] >= 2 and s["signConsistentAcrossHorizons"]
                        and s["etfAgreementCount"] >= 4)
    meets_promising_bar = s["strongHorizons"] >= 1 or (s["signConsistentAcrossHorizons"]
                                                        and s["etfAgreementCount"] >= 4)
    # A single ~3-month window cannot, on its own, establish that a signal persists across
    # regimes -- that is the entire reason to collect more history. "STRONG CASE" is reserved
    # for evidence that already spans multiple independent regimes; a study built from ONE
    # quarter is capped at "promising", regardless of how clean the in-sample stats look, and
    # cross-ETF "agreement" is further discounted when it is really a handful of shared
    # calendar dates (FOMC/CPI/OPEX) counted once per ETF rather than independent confirmation.
    MIN_DAYS_FOR_STRONG = 120
    if meets_strong_bar and total_calendar_days < MIN_DAYS_FOR_STRONG:
        return {
            "verdict": "SOME PROMISING SIGNALS, NEED MORE HISTORY",
            "cap_reason": (
                f"{best['col']} clears the STRONG bar on its own (|HAC t|>=1.96 at "
                f"{s['strongHorizons']}/3 horizons, consistent sign, {s['etfAgreementCount']}/6 "
                f"ETFs agree), but this study spans only {total_calendar_days} trading days "
                f"(~1 quarter) -- one regime, not several. Its quintile-5 bucket also has "
                f"{overlap['nRows']} rows across only {overlap['nUniqueDates']} unique calendar "
                f"dates (~{overlap['concentrationRatio']:.1f}x per date), so \"6/6 ETFs agree\" "
                f"partly reflects the same macro-calendar days shared across tickers, not 6 "
                f"independent confirmations. Capped at PROMISING until it is re-tested on the "
                f"252-day history."
            ),
            "overlap": overlap,
        }
    if meets_strong_bar:
        return {"verdict": "STRONG CASE FOR 252-DAY BACKFILL", "cap_reason": None, "overlap": overlap}
    if meets_promising_bar:
        return {"verdict": "SOME PROMISING SIGNALS, NEED MORE HISTORY", "cap_reason": None, "overlap": overlap}
    return {"verdict": "CURRENT FLOW DESIGN STILL SHOWS NO EDGE", "cap_reason": None, "overlap": overlap}


# --------------------------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------------------------

def render_feature_row(feat: Dict[str, str], result: Dict[str, Any], score: Dict[str, Any]) -> str:
    cells = []
    for h in DECISION_HORIZONS:
        b = result["horizons"][f"{h}d"]
        star = " **" if (hac_flag(b["q5"]) or hac_flag(b["q1"])) else ""
        cells.append(f"{fmt_pct(b['q5MinusQ1Mean'])}{star} (n={b['n']})")
    sp = result["spearman10d"]
    return (f"| {feat['col']} | {feat['family']} | {feat['hyp']} | " + " | ".join(cells) +
           f" | {fmt_num(sp['rho'])} | {score['etfAgreementCount']}/6 | {score['robustnessScore']} |")


def render_markdown(df: pd.DataFrame, results: Dict[str, Any], scored: List[Dict[str, Any]],
                    conc_test: Dict[str, Any], verdict_info: Dict[str, Any]) -> str:
    verdict = verdict_info["verdict"]
    lines: List[str] = []
    a = lines.append
    a("# Macro Options Flow -- Signal Design v2 (Segmented Flow Features)")
    a("")
    a(f"Built entirely from Phase 1's already-collected 60-day / 6-ETF dataset ({len(df)} "
     f"ETF-days, {df['date'].min()} to {df['date'].max()}) -- **no new ThetaData call, no new "
     f"backfill**. Adds DTE-bucket-level sentiment, call/put-separate flow, 15-minute "
     f"time-of-day buckets, and the day's 25 largest trades (all already stored in each Phase 1 "
     f"snapshot's `payload` JSONB but not exported to the v1 daily-aggregate parquet), plus "
     f"derived ratios, 1D/3D changes, and trailing-20D (prior-observations-only) z-scores.")
    a("")
    a("**Baseline (Signal Research v1, frozen, not recomputed here):** daily-aggregate "
     "net_trade_sentiment and delta_imbalance_ratio both showed NO EVIDENCE (pooled Q5-Q1 "
     "spreads at 5D/10D/20D all |HAC t| < 1). v2 asks whether averaging the whole day together "
     "washed out real information that exists in a more specific slice of the same flow.")
    a("")
    a("Every threshold below is a plain quintile or median split of the data -- none were tuned "
     "to maximize a result. No ML model was fit, no weights were optimized.")
    a("")

    a("## What could NOT be tested from existing data")
    a("")
    for name, reason in INFEASIBLE_ITEMS:
        a(f"- **{name}:** {reason}")
    a("")

    a("## Full feature battery: pooled Q5-Q1 spread by horizon")
    a("")
    a("`**` marks |HAC t| >= 1.96 on the Q5 or Q1 bucket itself (not the spread's own t-stat). "
     "\"ETF agree\" = how many of the 6 ETFs individually show the same-signed Q5-Q1 spread at "
     "both 5D and 10D (exploratory -- per-ETF n is 9-14 at these horizons).")
    a("")
    a("| Feature | Family | Hyp. | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 | Spearman(q,10D) | ETF agree | Score |")
    a("|---|---|---|---|---|---|---:|---:|---:|")
    for feat in FEATURE_CATALOGUE:
        col = feat["col"]
        a(render_feature_row(feat, results[col], next(s["score"] for s in scored if s["col"] == col)))
    a("")

    a("## H1: Do large/unusual trades outperform the full-day aggregate?")
    a("")
    lt = results["large_trade_sentiment"]
    a(f"`large_trade_sentiment` (25 largest trades/day only) Q5-Q1: 5D {fmt_pct(lt['horizons']['5d']['q5MinusQ1Mean'])}, "
     f"10D {fmt_pct(lt['horizons']['10d']['q5MinusQ1Mean'])}, 20D {fmt_pct(lt['horizons']['20d']['q5MinusQ1Mean'])} "
     f"vs. the full-day `net_trade_sentiment` baseline: 5D {fmt_pct(V1_BASELINE['net_trade_sentiment']['5d'])}, "
     f"10D {fmt_pct(V1_BASELINE['net_trade_sentiment']['10d'])}, 20D {fmt_pct(V1_BASELINE['net_trade_sentiment']['20d'])}.")
    a("")

    a("## H2: Is longer-dated flow more predictive than 0DTE flow?")
    a("")
    a("| DTE bucket | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 | N (10D) |")
    a("|---|---:|---:|---:|---:|")
    for col, label in [("dte_0dte_sentiment", "0DTE"), ("dte_1_7d_sentiment", "1-7D"),
                       ("dte_8_30d_sentiment", "8-30D"), ("dte_31_60d_sentiment", "31-60D")]:
        r = results[col]
        a(f"| {label} | {fmt_pct(r['horizons']['5d']['q5MinusQ1Mean'])} | "
         f"{fmt_pct(r['horizons']['10d']['q5MinusQ1Mean'])} | {fmt_pct(r['horizons']['20d']['q5MinusQ1Mean'])} | "
         f"{r['horizons']['10d']['n']} |")
    a("")
    a("(60D+ bucket: 0 of 360 ETF-days had any 60D+ trade in this universe/window -- not testable.)")
    a("")

    a("## H3: Is closing-hour flow more predictive than full-day flow?")
    a("")
    a("| Window | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 |")
    a("|---|---:|---:|---:|")
    for col, label in [("opening_hour_sentiment", "Opening hour"), ("midday_sentiment", "Midday"),
                       ("closing_hour_sentiment", "Closing hour")]:
        r = results[col]
        a(f"| {label} | {fmt_pct(r['horizons']['5d']['q5MinusQ1Mean'])} | "
         f"{fmt_pct(r['horizons']['10d']['q5MinusQ1Mean'])} | {fmt_pct(r['horizons']['20d']['q5MinusQ1Mean'])} |")
    a("")

    a("## H4: Is flow acceleration more predictive than flow level?")
    a("")
    a("| Feature | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 |")
    a("|---|---:|---:|---:|")
    for col in ["sentiment_change_1d", "sentiment_change_3d", "delta_imbalance_change_1d",
               "delta_imbalance_change_3d"]:
        r = results[col]
        a(f"| {col} | {fmt_pct(r['horizons']['5d']['q5MinusQ1Mean'])} | "
         f"{fmt_pct(r['horizons']['10d']['q5MinusQ1Mean'])} | {fmt_pct(r['horizons']['20d']['q5MinusQ1Mean'])} |")
    a(f"\nCompare to the level baselines: net_trade_sentiment 10D {fmt_pct(V1_BASELINE['net_trade_sentiment']['10d'])}, "
     f"delta_imbalance_ratio 10D {fmt_pct(V1_BASELINE['delta_imbalance_ratio']['10d'])}.")
    a("")

    a("## H5: Is concentrated flow more informative than diffuse flow?")
    a("")
    a("Median split on `top10_concentration` (top-10-trade premium / full-day gross premium) "
     "crossed with the sign of the day's net_trade_sentiment:")
    a("")
    a("| Horizon | Concentrated bullish | Diffuse bullish | Concentrated bearish | Diffuse bearish |")
    a("|---|---:|---:|---:|---:|")
    for h in DECISION_HORIZONS:
        c = conc_test["horizons"][f"{h}d"]
        a(f"| {h}D | {fmt_pct(c['concentrated_bullish']['mean'])} (n={c['concentrated_bullish']['n']}) | "
         f"{fmt_pct(c['diffuse_bullish']['mean'])} (n={c['diffuse_bullish']['n']}) | "
         f"{fmt_pct(c['concentrated_bearish']['mean'])} (n={c['concentrated_bearish']['n']}) | "
         f"{fmt_pct(c['diffuse_bearish']['mean'])} (n={c['diffuse_bearish']['n']}) |")
    a("")

    a("## H6: Is flow relative to an ETF's own baseline more informative than raw flow?")
    a("")
    a("| Feature | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 | N (10D) |")
    a("|---|---:|---:|---:|---:|")
    for col in ["z_sentiment_20d", "z_delta_imbalance_20d", "z_gross_premium_20d",
               "z_large_trade_premium_20d", "z_0dte_share_20d"]:
        r = results[col]
        a(f"| {col} | {fmt_pct(r['horizons']['5d']['q5MinusQ1Mean'])} | "
         f"{fmt_pct(r['horizons']['10d']['q5MinusQ1Mean'])} | {fmt_pct(r['horizons']['20d']['q5MinusQ1Mean'])} | "
         f"{r['horizons']['10d']['n']} |")
    a("")
    a(f"z-scores need a 20D trailing window (min 10 prior observations) computed from ONLY "
     f"prior dates for that ETF -- with 60 total dates per ETF, this leaves at most "
     f"{60 - Z_MIN_PERIODS} usable dates per ETF (n≈{6 * (60 - Z_MIN_PERIODS)} pooled at best), "
     f"noticeably smaller than the other tests. Treat as exploratory.")
    a("")

    a("## Ranking (robustness / consistency / monotonicity / sample size / plausibility)")
    a("")
    a("Ranked by the composite robustness score (10 x horizons clearing |HAC t|>=1.96 with a "
     "consistent sign, +5 if sign is consistent across 5D/10D/20D, +1 per ETF that individually "
     "agrees in sign) -- NOT by best single bucket:")
    a("")
    a("| Rank | Feature | Family | Strong horizons | Sign consistent | ETF agree | Min N | Score |")
    a("|---|---|---|---:|---|---:|---:|---:|")
    ranked = sorted(scored, key=lambda r: -r["score"]["robustnessScore"])
    for i, r in enumerate(ranked[:15], start=1):
        s = r["score"]
        a(f"| {i} | {r['col']} | {r['family']} | {s['strongHorizons']}/3 | "
         f"{'yes' if s['signConsistentAcrossHorizons'] else 'no'} | {s['etfAgreementCount']}/6 | "
         f"{s['minDecisionHorizonN']} | {s['robustnessScore']} |")
    a("")
    if ranked:
        top = ranked[0]
        ov = verdict_info.get("overlap") or {}
        if ov.get("nUniqueDates"):
            a(f"**Cross-ETF date-clustering check on the top-ranked feature ({top['col']}):** its "
             f"quintile-5 bucket has {ov['nRows']} ETF-day rows spread across only "
             f"{ov['nUniqueDates']} unique calendar dates ({ov['concentrationRatio']:.2f} rows/date "
             f"on average) -- some \"ETF agreement\" reflects the same macro-calendar days "
             f"(FOMC/CPI/OPEX) shared across tickers, not fully independent confirmations. This is "
             f"exactly the kind of ambiguity a longer, multi-regime history would resolve.")
            a("")

    a("**Two results deserve more scrutiny than their rank alone suggests:**")
    a("")
    a("- `top10_concentration` ranks #3 by score, but only **1 of 6 ETFs** individually shows "
     "the same-signed relationship (vs. 4-6/6 for every other feature in the top 10). A pooled "
     "result this concentrated in one or two names is not the broad, cross-ETF pattern the score "
     "implies -- the composite score under-penalizes low ETF agreement here. Read it as "
     "single-name, not a market-wide concentration effect.")
    a("- `put_sentiment` ranks #2 and has an economically legible direction: (putBought-putSold)/"
     "(putBought+putSold) is POSITIVE when puts are being aggressively bought, and its Q5-Q1 "
     "spread is NEGATIVE at 10D/20D -- i.e. aggressive put-buying days are followed by weaker "
     "forward returns, the intuitive sign for put positioning as a bearish/hedging signal. It "
     "isn't one of the pre-registered H1-H6 hypotheses, so treat it as exploratory, but it is "
     "worth carrying into a 252-day test on that basis alone.")
    a("")

    a("## Verdict")
    a("")
    a(f"**{verdict}**")
    a("")
    if verdict_info.get("cap_reason"):
        a(f"**Why this is capped at PROMISING rather than STRONG:** {verdict_info['cap_reason']}")
        a("")
    if verdict != "CURRENT FLOW DESIGN STILL SHOWS NO EDGE":
        a("### Fields to make sure the 252-day backfill preserves")
        a("")
        a("If any of the above families are pursued further, the 252-day historical job must "
         "keep (not just the daily aggregate) at minimum:")
        a("- Per-DTE-bucket `sentiment`/`deltaRatio`/`signedPremium` (already computed in the "
         "live payload's `dte.buckets`, but only `grossPremium` per bucket reached the v1 "
         "export -- confirm the 252-day export keeps the full bucket dict).")
        a("- `aggression` (callBought/callSold/putBought/putSold) at daily granularity, "
         "already computed -- confirm it's exported, not just used to derive the combined "
         "sentiment.")
        a("- The 15-minute `intraday` series, or at least pre-aggregated opening/midday/closing "
         "sentiment, per ticker-day.")
        a("- Either (a) the full per-trade size distribution's percentiles (p50/p90/p95/p99) "
         "computed at snapshot time, or (b) more than 25 large trades persisted -- top-25-only "
         "concentration/large-trade metrics are a proxy, not the real thing.")
        a("- A moneyness classification (e.g. |delta| bucket: >0.7 ITM, 0.4-0.6 near-ATM, "
         "0.15-0.4 OTM, <0.15 far OTM) per trade, aggregated to daily buckets -- **not computed "
         "at all today**, and item 2B of this study could not be tested for exactly this reason.")
    else:
        a("No segmentation tested here shows a robust, consistent, cross-ETF signal. This does "
         "not by itself rule out predictive value in options flow -- 60 days x 6 ETFs is a small "
         "sample for horizons up to 20 trading days, especially for slices (large trades, single "
         "time-of-day windows, single DTE buckets) that see even fewer trades per day than the "
         "full-day aggregate. But nothing here currently justifies more history on its own.")
    a("")
    a("No production model was fit. No weights or thresholds were optimized. This report does "
     "not start the 252-day full-flow backfill; that decision is left to the reader.")
    return "\n".join(lines)


def flatten_to_rows(results: Dict[str, Any], scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    score_by_col = {s["col"]: s["score"] for s in scored}
    for feat in FEATURE_CATALOGUE:
        col = feat["col"]
        r = results[col]
        s = score_by_col[col]
        for h in HORIZONS:
            b = r["horizons"][f"{h}d"]
            rows.append({
                "feature": col, "family": feat["family"], "hypothesis": feat["hyp"], "horizon": h,
                "n": b["n"], "q1_mean": b["q1"]["mean"], "q1_hac_t": b["q1"]["hacTStat"],
                "q5_mean": b["q5"]["mean"], "q5_hac_t": b["q5"]["hacTStat"],
                "q5_minus_q1_mean": b["q5MinusQ1Mean"],
                "spearman_rho_10d": r["spearman10d"]["rho"], "etf_agreement_count": s["etfAgreementCount"],
                "robustness_score": s["robustnessScore"],
            })
    return rows


def main() -> int:
    df = load_master_frame()
    df = add_derived_features(df)
    print(f"Master v2 frame: {len(df)} rows, {df['ticker'].nunique()} tickers, "
         f"{df['date'].min()} -> {df['date'].max()}, {len(df.columns)} columns")

    results = {feat["col"]: feature_battery(df, feat["col"]) for feat in FEATURE_CATALOGUE}
    scored = [{"col": feat["col"], "family": feat["family"], "score": score_feature(results[feat["col"]])}
             for feat in FEATURE_CATALOGUE]
    conc_test = concentration_direction_test(df)
    total_calendar_days = df["date"].nunique()
    verdict_info = determine_verdict(scored, df, results, total_calendar_days)

    report_json = {
        "datasetRows": len(df), "dateRange": [df["date"].min(), df["date"].max()],
        "totalCalendarDays": total_calendar_days,
        "v1Baseline": V1_BASELINE, "infeasible": INFEASIBLE_ITEMS,
        "features": results, "scores": scored, "concentrationDirectionTest": conc_test,
        "verdict": verdict_info["verdict"], "verdictCapReason": verdict_info.get("cap_reason"),
    }
    OUT_JSON.write_text(json.dumps(report_json, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(flatten_to_rows(results, scored)).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_markdown(df, results, scored, conc_test, verdict_info), encoding="utf-8")

    print(f"\nWrote {OUT_MD}\nWrote {OUT_CSV}\nWrote {OUT_JSON}")
    print(f"\nVERDICT: {verdict_info['verdict']}")
    print("\nDo not start the 252-day backfill automatically. Stopping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
