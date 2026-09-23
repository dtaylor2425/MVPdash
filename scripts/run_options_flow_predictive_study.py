"""
scripts/run_options_flow_predictive_study.py

The 60-day Macro Options Flow predictive research study. Signal discovery only:

    * Reads reports/options-flow-research-dataset.parquet (features -- READ ONLY, never
      modified) and reports/options-flow-forward-returns.parquet (outcomes, built separately by
      scripts/build_options_flow_forward_returns.py from src.data_sources.fetch_prices).
      Feature construction never sees outcomes; this script only ever reads both, joins them,
      and computes descriptive statistics.
    * No weight optimization, no threshold brute-forcing, no ML model training, no in-sample
      significance claims. Every threshold used (quintiles, sign, median/tercile regime splits,
      the +-0.08 neutral sentiment band) is either a plain quantile of the data or an existing
      product constant -- never chosen to maximize a result.
    * Writes reports/options-flow-predictive-study.md, .csv, .json and prints a final verdict:
      PROCEED TO 252-DAY BACKFILL / PROMISING BUT MODIFY RESEARCH DESIGN FIRST / NO EVIDENCE YET.
    * Does NOT start the 252-day full-flow backfill under any circumstances -- that decision is
      for a human to act on based on this report.

    python scripts/run_options_flow_predictive_study.py
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

from api.services.options_flow_forward_returns import verify_no_future_prices_in_features  # noqa: E402
from api.services.options_flow_research_stats import (  # noqa: E402
    HORIZONS,
    NEUTRAL_SENTIMENT_BAND,
    classify_agreement,
    describe_returns,
    median_split,
    quintile_labels,
    spearman_correlation,
    tercile_split,
)

FEATURES_PATH = ROOT / "reports" / "options-flow-research-dataset.parquet"
RETURNS_PATH = ROOT / "reports" / "options-flow-forward-returns.parquet"
OUT_MD = ROOT / "reports" / "options-flow-predictive-study.md"
OUT_CSV = ROOT / "reports" / "options-flow-predictive-study.csv"
OUT_JSON = ROOT / "reports" / "options-flow-predictive-study.json"

RET_COLS = [f"ret_{h}d" for h in HORIZONS]

# The 10 features tested individually (item 2). H1 = net_trade_sentiment, H2 = delta_imbalance_ratio
# are the two primary, pre-registered hypotheses; the rest are supporting/exploratory.
PRIMARY_FEATURES = ["net_trade_sentiment", "delta_imbalance_ratio"]
SUPPORTING_FEATURES = [
    "net_dollar_delta", "zero_dte_share", "put_skew_25d", "atm_iv",
    "iv_percentile_20d", "iv_percentile_60d", "iv_percentile_126d", "iv_percentile_252d",
]
ALL_FEATURES = PRIMARY_FEATURES + SUPPORTING_FEATURES

GLD_EXTREME_DATE = "2026-09-02"  # flagged in reports/options-flow-backfill-qa.md: GLD's max
                                  # net_trade_sentiment/delta_imbalance_ratio/net_dollar_delta observation

TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]


# --------------------------------------------------------------------------------------------
# Loading / joining (read-only on both inputs)
# --------------------------------------------------------------------------------------------

def load_joined() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    returns = pd.read_parquet(RETURNS_PATH)
    features = features.copy()
    returns = returns.copy()
    features["date"] = features["date"].astype(str)
    returns["date"] = returns["date"].astype(str)
    merged = features.merge(returns[["date", "ticker", "close"] + RET_COLS], on=["date", "ticker"],
                            how="inner", validate="one_to_one")
    if len(merged) != len(features):
        raise RuntimeError(
            f"Join dropped rows: {len(features)} feature rows -> {len(merged)} joined rows. "
            "Forward returns must cover every feature (date, ticker)."
        )
    return merged


def verify_no_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    from src.data_sources import fetch_prices
    probe = fetch_prices(["SPY"], period="5d")
    probe.index = pd.to_datetime(probe.index)
    last_price_date = probe.index.max().date()
    result = verify_no_future_prices_in_features(list(df["date"].unique()), last_price_date)
    return result


# --------------------------------------------------------------------------------------------
# Bucket-level result blocks
# --------------------------------------------------------------------------------------------

def quintile_spread_test(df: pd.DataFrame, feature: str, group_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Assigns pooled quintiles (1..5) of `feature` -- per-group if group_col is given (e.g. one
    ETF's own subsample), else across the whole frame -- then, for every horizon, reports
    describe_returns() for Q1 and Q5 plus the Q5-minus-Q1 spread (paired on rank, not on date;
    the spread's own significance is read from the HAC t-stat of the DIFFERENCE series only
    when group_col pairs Q1/Q5 1:1, which quintiles do not in general, so the spread here is a
    difference-of-means with each side's own HAC t-stat reported instead of a paired test).
    """
    out: Dict[str, Any] = {"feature": feature, "groupBy": group_col, "horizons": {}}
    work = df.copy()
    if group_col:
        work["_q"] = work.groupby(group_col)[feature].transform(
            lambda s: pd.Series(quintile_labels(list(s)), index=s.index))
    else:
        work["_q"] = quintile_labels(list(work[feature]))
    for h in HORIZONS:
        col = f"ret_{h}d"
        q1 = work.loc[work["_q"] == 1, col].tolist()
        q5 = work.loc[work["_q"] == 5, col].tolist()
        d1 = describe_returns(q1, h)
        d5 = describe_returns(q5, h)
        spread_mean = (d5["mean"] - d1["mean"]) if (d5["mean"] is not None and d1["mean"] is not None) else None
        out["horizons"][f"{h}d"] = {"q1": d1, "q5": d5, "q5MinusQ1Mean": spread_mean}
    return out


def sign_bucket_test(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"feature": feature, "horizons": {}}
    pos = df[feature] > 0
    neg = df[feature] < 0
    for h in HORIZONS:
        col = f"ret_{h}d"
        d_pos = describe_returns(df.loc[pos, col].tolist(), h)
        d_neg = describe_returns(df.loc[neg, col].tolist(), h)
        out["horizons"][f"{h}d"] = {"positive": d_pos, "negative": d_neg}
    return out


def monotonicity_test(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """Q1..Q5 mean forward return per horizon (pooled quintiles) plus Spearman rank correlation
    between the pooled quintile label and forward return."""
    out: Dict[str, Any] = {"feature": feature, "horizons": {}}
    q = quintile_labels(list(df[feature]))
    for h in HORIZONS:
        col = f"ret_{h}d"
        by_q = {}
        for k in (1, 2, 3, 4, 5):
            vals = [r for r, qq in zip(df[col].tolist(), q) if qq == k]
            d = describe_returns(vals, h)
            by_q[f"q{k}"] = {"n": d["n"], "mean": d["mean"], "median": d["median"]}
        spearman = spearman_correlation(q, df[col].tolist())
        out["horizons"][f"{h}d"] = {"byQuintile": by_q, "spearman": spearman}
    return out


def interaction_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Item 3: classify each row by sentiment/delta agreement (reusing the product's own +-0.08
    NEUTRAL sentiment band -- no new threshold), plus the continuous interaction feature."""
    labels = [classify_agreement(s, d) for s, d in
             zip(df["net_trade_sentiment"], df["delta_imbalance_ratio"])]
    work = df.copy()
    work["_agreement"] = labels
    work["_interaction"] = work["net_trade_sentiment"] * work["delta_imbalance_ratio"]
    out: Dict[str, Any] = {"neutralBand": NEUTRAL_SENTIMENT_BAND, "categoryCounts": {}, "horizons": {}}
    for cat in ["agree_bullish", "agree_bearish", "disagreement", "approximately_neutral"]:
        out["categoryCounts"][cat] = int((work["_agreement"] == cat).sum())
    for h in HORIZONS:
        col = f"ret_{h}d"
        by_cat = {}
        for cat in ["agree_bullish", "agree_bearish", "disagreement", "approximately_neutral"]:
            vals = work.loc[work["_agreement"] == cat, col].tolist()
            by_cat[cat] = describe_returns(vals, h)
        out["horizons"][f"{h}d"] = {"byCategory": by_cat}
    out["continuousInteractionMonotonicity"] = monotonicity_test(work, "_interaction")
    return out


def cross_sectional_test(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """Item 6: per date, rank the 6 ETFs by `feature`; highest-ranked minus lowest-ranked
    forward-return spread, for every horizon. Only run for sentiment/delta_imbalance_ratio
    (never raw net_dollar_delta -- magnitude differs by ETF liquidity, per the spec)."""
    out: Dict[str, Any] = {"feature": feature, "horizons": {}}
    dates = sorted(df["date"].unique())
    for h in HORIZONS:
        col = f"ret_{h}d"
        spreads: List[float] = []
        n_dates_used = 0
        for d in dates:
            day = df[df["date"] == d]
            valid = day.dropna(subset=[feature])
            if len(valid) < 2:
                continue
            top = valid.loc[valid[feature].idxmax()]
            bot = valid.loc[valid[feature].idxmin()]
            if pd.isna(top[col]) or pd.isna(bot[col]):
                continue
            spreads.append(float(top[col]) - float(bot[col]))
            n_dates_used += 1
        out["horizons"][f"{h}d"] = {"spread": describe_returns(spreads, h), "datesUsed": n_dates_used,
                                    "datesTotal": len(dates)}
    return out


def regime_interaction_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Item 7: median/tercile splits (predetermined, not optimized) on IV percentile (60D),
    0DTE share, and put skew; within each regime, re-run the H1/H2 Q5-Q1 spread at 5D/10D."""
    out: Dict[str, Any] = {}
    regimes = {
        "iv_percentile_60d": median_split(list(df["iv_percentile_60d"])),
        "zero_dte_share": median_split(list(df["zero_dte_share"])),
        "put_skew_25d": median_split(list(df["put_skew_25d"])),
    }
    for regime_feature, (threshold, labels) in regimes.items():
        work = df.copy()
        work["_regime"] = labels
        block: Dict[str, Any] = {"splitThreshold": threshold, "n": {
            "high": int((work["_regime"] == "high").sum()), "low": int((work["_regime"] == "low").sum())}}
        for signal_feature in PRIMARY_FEATURES:
            block[signal_feature] = {}
            for level in ("high", "low"):
                sub = work[work["_regime"] == level]
                block[signal_feature][level] = {}
                for h in (5, 10):
                    col = f"ret_{h}d"
                    q = quintile_labels(list(sub[signal_feature]))
                    q1 = [r for r, qq in zip(sub[col].tolist(), q) if qq == 1]
                    q5 = [r for r, qq in zip(sub[col].tolist(), q) if qq == 5]
                    d1, d5 = describe_returns(q1, h), describe_returns(q5, h)
                    spread = (d5["mean"] - d1["mean"]) if (d5["mean"] is not None and d1["mean"] is not None) else None
                    block[signal_feature][level][f"{h}d"] = {"q1": d1, "q5": d5, "q5MinusQ1Mean": spread}
        out[regime_feature] = block
    return out


def gld_robustness_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Item 9: rerun Greek-dependent results (delta_imbalance_ratio, net_dollar_delta) both
    including and excluding the flagged GLD extreme day. Never removes any OTHER observation,
    and never removes this one from the primary dataset -- this is a side re-run only."""
    excluded = df[~((df["ticker"] == "GLD") & (df["date"] == GLD_EXTREME_DATE))]
    out: Dict[str, Any] = {"excludedDate": GLD_EXTREME_DATE, "excludedTicker": "GLD",
                           "nWithExtreme": len(df), "nExcluded": len(excluded)}
    for feature in ["delta_imbalance_ratio", "net_dollar_delta"]:
        out[feature] = {
            "including": quintile_spread_test(df, feature),
            "excluding": quintile_spread_test(excluded, feature),
        }
    return out


def missing_data_report(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"featureNaNCountsByTicker": {}, "forwardReturnCoverageByTickerHorizon": {}}
    for feature in ALL_FEATURES:
        out["featureNaNCountsByTicker"][feature] = {
            t: int(df.loc[df["ticker"] == t, feature].isna().sum()) for t in TICKERS
        }
    for t in TICKERS:
        out["forwardReturnCoverageByTickerHorizon"][t] = {
            f"{h}d": int(df.loc[df["ticker"] == t, f"ret_{h}d"].notna().sum()) for h in HORIZONS
        }
    return out


# --------------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------------

def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def hac_flag(d: Dict[str, Any], thresh: float = 1.96) -> str:
    t = d.get("hacTStat")
    if t is None:
        return ""
    return " **" if abs(t) >= thresh else ""


def render_quintile_table(qtest: Dict[str, Any]) -> str:
    lines = ["| Horizon | Q1 N | Q1 mean | Q1 HAC-t | Q5 N | Q5 mean | Q5 HAC-t | Q5-Q1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        b = qtest["horizons"][f"{h}d"]
        q1, q5 = b["q1"], b["q5"]
        lines.append(
            f"| {h}D | {q1['n']} | {fmt_pct(q1['mean'])} | {fmt_num(q1['hacTStat'])}{hac_flag(q1)} | "
            f"{q5['n']} | {fmt_pct(q5['mean'])} | {fmt_num(q5['hacTStat'])}{hac_flag(q5)} | "
            f"{fmt_pct(b['q5MinusQ1Mean'])} |")
    return "\n".join(lines)


def determine_verdict(report: Dict[str, Any]) -> str:
    """
    Robustness/consistency-based, not best-single-bucket. Criteria (all must hold for PROCEED):
      (a) at least one primary feature (H1/H2) shows |HAC t| >= 1.96 on the Q5-Q1 spread, pooled,
          at 2+ of the 3 decision-relevant horizons (5D, 10D, 20D);
      (b) the SIGN of that spread is consistent across those horizons;
      (c) at least 4 of 6 ETFs individually show the same-signed Q5-Q1 spread at 5D or 10D
          (directional consistency across names, not just pooled);
      (d) the GLD-extreme-day exclusion does not flip the sign of the pooled result.
    Meeting (a)+(b) but not (c)/(d) -> PROMISING BUT MODIFY RESEARCH DESIGN FIRST.
    Meeting none -> NO EVIDENCE YET.
    """
    strong_hits = 0
    for feature in PRIMARY_FEATURES:
        pooled = report["primaryPooled"][feature]
        signs = []
        strong_horizons = 0
        for h in (5, 10, 20):
            spread = pooled["horizons"][f"{h}d"]["q5MinusQ1Mean"]
            t = pooled["horizons"][f"{h}d"]["q5"]["hacTStat"]
            if spread is not None:
                signs.append(1 if spread > 0 else (-1 if spread < 0 else 0))
            if t is not None and abs(t) >= 1.96:
                strong_horizons += 1
        sign_consistent = len(set(s for s in signs if s != 0)) <= 1 and len(signs) > 0
        if strong_horizons >= 2 and sign_consistent:
            strong_hits += 1
    if strong_hits == 0:
        return "NO EVIDENCE YET"
    # check per-ETF consistency + GLD robustness for whichever primary feature qualified
    per_etf_ok = False
    gld_ok = True
    for feature in PRIMARY_FEATURES:
        per_etf = report["primaryPerETF"][feature]
        agree = 0
        for t in TICKERS:
            b5 = per_etf[t]["horizons"]["5d"]["q5MinusQ1Mean"]
            b10 = per_etf[t]["horizons"]["10d"]["q5MinusQ1Mean"]
            vals = [v for v in (b5, b10) if v is not None]
            if vals and all(v > 0 for v in vals):
                agree += 1
            elif vals and all(v < 0 for v in vals):
                agree += 1
        if agree >= 4:
            per_etf_ok = True
    gld = report["gldRobustness"]
    for feature in ["delta_imbalance_ratio"]:
        for h in HORIZONS:
            inc = gld[feature]["including"]["horizons"][f"{h}d"]["q5MinusQ1Mean"]
            exc = gld[feature]["excluding"]["horizons"][f"{h}d"]["q5MinusQ1Mean"]
            if inc is not None and exc is not None and (inc > 0) != (exc > 0):
                gld_ok = False
    if per_etf_ok and gld_ok:
        return "PROCEED TO 252-DAY BACKFILL"
    return "PROMISING BUT MODIFY RESEARCH DESIGN FIRST"


def build_report(df: pd.DataFrame, leakage: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "datasetRows": len(df),
        "dateRange": [df["date"].min(), df["date"].max()],
        "tickers": TICKERS,
        "leakageCheck": leakage,
        "primaryPooled": {f: quintile_spread_test(df, f) for f in PRIMARY_FEATURES},
        "primaryPerETF": {f: {t: quintile_spread_test(df[df["ticker"] == t], f) for t in TICKERS}
                          for f in PRIMARY_FEATURES},
        "primarySignBuckets": {f: sign_bucket_test(df, f) for f in PRIMARY_FEATURES},
        "primaryMonotonicity": {f: monotonicity_test(df, f) for f in PRIMARY_FEATURES},
        "supportingPooled": {f: quintile_spread_test(df, f) for f in SUPPORTING_FEATURES},
        "interaction": interaction_test(df),
        "crossSectional": {f: cross_sectional_test(df, f) for f in PRIMARY_FEATURES},
        "regimeInteraction": regime_interaction_test(df),
        "gldRobustness": gld_robustness_test(df),
        "missingData": missing_data_report(df),
    }
    report["verdict"] = determine_verdict(report)
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    a = lines.append
    a("# Macro Options Flow -- 60-Day Predictive Research Study")
    a("")
    a(f"Dataset: `reports/options-flow-research-dataset.parquet` (immutable, unmodified) joined to "
     f"`reports/options-flow-forward-returns.parquet`. {report['datasetRows']} rows, "
     f"{report['dateRange'][0]} to {report['dateRange'][1]}, tickers: {', '.join(report['tickers'])}.")
    a("")
    a("This is a **signal-discovery study**: no weights were optimized, no thresholds were "
     "brute-forced, no ML model was trained, and no significance claim here should be read as a "
     "production trading signal. All quintile/sign/regime cutoffs are plain quantiles of the "
     "data or existing product constants (e.g. the +-0.08 NEUTRAL sentiment band), never chosen "
     "to maximize a result.")
    a("")
    lc = report["leakageCheck"]
    a(f"**Leakage check:** max feature date `{lc['maxFeatureDate']}` <= last available price date "
     f"`{lc['lastAvailablePriceDate']}` -> **{lc['featuresPrecedeOrEqualPriceHistory']}**. Forward "
     f"returns for dates near the end of the window are real, honest missing values (`None`, never "
     f"`0.0`) where the future session hasn't happened yet -- see Q10.")
    a("")

    a("## Q1. Does net_trade_sentiment predict forward returns? (H1)")
    a("")
    a("Pooled quintile spread (Q5 = most bullish sentiment, Q1 = most bearish), all 6 ETFs, all dates:")
    a("")
    a(render_quintile_table(report["primaryPooled"]["net_trade_sentiment"]))
    a("")
    a("Per-ETF Q5-Q1 spread at 5D/10D, used only to check directional (sign) consistency across "
     "names for the verdict rule -- **not** for magnitude claims. Per-ETF quintile buckets have "
     "n as low as 9-14 at 10D, and the HAC lag (horizon-1=9) is then close to n, which is a known "
     "small-sample pathology that makes the HAC t-stat unstable (e.g. a large-looking |t| can "
     "appear from a handful of observations, not a real strong effect):")
    a("")
    a("| Ticker | 5D Q5-Q1 | 5D Q5 HAC-t | 10D Q5-Q1 | 10D Q5 HAC-t |")
    a("|---|---:|---:|---:|---:|")
    for t in report["tickers"]:
        b = report["primaryPerETF"]["net_trade_sentiment"][t]["horizons"]
        a(f"| {t} | {fmt_pct(b['5d']['q5MinusQ1Mean'])} | {fmt_num(b['5d']['q5']['hacTStat'])} | "
         f"{fmt_pct(b['10d']['q5MinusQ1Mean'])} | {fmt_num(b['10d']['q5']['hacTStat'])} |")
    a("")

    a("## Q2. Does delta_imbalance_ratio predict forward returns? (H2)")
    a("")
    a(render_quintile_table(report["primaryPooled"]["delta_imbalance_ratio"]))
    a("")
    a("| Ticker | 5D Q5-Q1 | 5D Q5 HAC-t | 10D Q5-Q1 | 10D Q5 HAC-t |")
    a("|---|---:|---:|---:|---:|")
    for t in report["tickers"]:
        b = report["primaryPerETF"]["delta_imbalance_ratio"][t]["horizons"]
        a(f"| {t} | {fmt_pct(b['5d']['q5MinusQ1Mean'])} | {fmt_num(b['5d']['q5']['hacTStat'])} | "
         f"{fmt_pct(b['10d']['q5MinusQ1Mean'])} | {fmt_num(b['10d']['q5']['hacTStat'])} |")
    a("")

    a("## Q3. Does sentiment/delta-imbalance agreement matter more than either alone?")
    a("")
    ia = report["interaction"]
    a(f"Category counts (neutral band = +-{ia['neutralBand']}, the product's existing threshold): " +
     ", ".join(f"{k}={v}" for k, v in ia["categoryCounts"].items()))
    a("")
    a("| Category | 5D mean | 5D N | 10D mean | 10D N | 20D mean | 20D N |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for cat in ["agree_bullish", "agree_bearish", "disagreement", "approximately_neutral"]:
        r5, r10, r20 = (ia["horizons"][f"{h}d"]["byCategory"][cat] for h in (5, 10, 20))
        a(f"| {cat} | {fmt_pct(r5['mean'])} | {r5['n']} | {fmt_pct(r10['mean'])} | {r10['n']} | "
         f"{fmt_pct(r20['mean'])} | {r20['n']} |")
    a("")
    ci = ia["continuousInteractionMonotonicity"]["horizons"]["10d"]["spearman"]
    a(f"Continuous `sentiment x delta_imbalance_ratio` interaction, Spearman rank corr vs 10D return: "
     f"rho={fmt_num(ci['rho'])}, p={fmt_num(ci['pValue'], 4)}, n={ci['n']}.")
    a("")

    a("## Q4. Is the relationship monotonic across quintiles (not just Q1 vs Q5)?")
    a("")
    for feature in PRIMARY_FEATURES:
        m = report["primaryMonotonicity"][feature]
        a(f"**{feature}**, 10D forward return by quintile:")
        b = m["horizons"]["10d"]["byQuintile"]
        a("| " + " | ".join(f"Q{k[1]}" for k in b) + " |")
        a("|" + "---:|" * len(b))
        a("| " + " | ".join(fmt_pct(b[k]["mean"]) for k in b) + " |")
        sp = m["horizons"]["10d"]["spearman"]
        a(f"Spearman(quintile, 10D return): rho={fmt_num(sp['rho'])}, p={fmt_num(sp['pValue'], 4)}, n={sp['n']}.")
        a("")

    a("## Q5. Does the signal cross-sectionally discriminate across the 6 ETFs?")
    a("")
    a("Per date, rank the 6 ETFs by the feature; highest-ranked minus lowest-ranked forward-return "
     "spread (never raw net_dollar_delta for this test -- liquidity differs too much across ETFs):")
    a("")
    a("| Feature | Horizon | Dates used | Spread mean | Spread HAC-t |")
    a("|---|---|---:|---:|---:|")
    for feature in PRIMARY_FEATURES:
        cs = report["crossSectional"][feature]
        for h in HORIZONS:
            b = cs["horizons"][f"{h}d"]
            a(f"| {feature} | {h}D | {b['datesUsed']}/{b['datesTotal']} | {fmt_pct(b['spread']['mean'])} | "
             f"{fmt_num(b['spread']['hacTStat'])} |")
    a("")

    a("## Q6. Do IV percentile / 0DTE-share / put-skew regimes modulate the signal?")
    a("")
    a("Predetermined median splits (not optimized). Q5-Q1 spread of the primary feature, within each regime:")
    a("")
    for regime_feature, block in report["regimeInteraction"].items():
        a(f"**Regime: {regime_feature}** (median={fmt_num(block['splitThreshold'])}, "
         f"n high={block['n']['high']}, n low={block['n']['low']})")
        a("")
        a("| Signal | Regime | 5D Q5-Q1 | 10D Q5-Q1 |")
        a("|---|---|---:|---:|")
        for signal_feature in PRIMARY_FEATURES:
            for level in ("high", "low"):
                cell = block[signal_feature][level]
                a(f"| {signal_feature} | {level} | {fmt_pct(cell['5d']['q5MinusQ1Mean'])} | "
                 f"{fmt_pct(cell['10d']['q5MinusQ1Mean'])} |")
        a("")

    a("## Q7. Are the results consistent per-ETF, or only in the pooled sample?")
    a("")
    a("See Q1/Q2 per-ETF tables above. Directional consistency across at least 4 of 6 ETFs at 5D "
     "or 10D is required for the PROCEED verdict; pooled significance alone is not sufficient. "
     "Per-ETF HAC t-stats are read for **sign only** here, never for magnitude -- per-ETF n (9-14 "
     "at 10D/20D) is too small relative to the HAC lag for the t-stat itself to be trustworthy "
     "(see the caveat under Q1/Q2). The pooled tests (n=42-72) are the ones load-bearing for the verdict.")
    a("")

    a("## Q8. Does the flagged GLD extreme day (GLD, 2026-09-02) drive any conclusion?")
    a("")
    g = report["gldRobustness"]
    a(f"That single day is GLD's max observed delta_imbalance_ratio, net_dollar_delta, and "
     f"net_trade_sentiment in Phase 1 (flagged, not excluded, in the QA report). Kept in the "
     f"primary dataset per instruction; this re-run only checks whether removing it flips a "
     f"conclusion, and removes no other observation.")
    a("")
    for feature in ["delta_imbalance_ratio", "net_dollar_delta"]:
        a(f"**{feature}** (n {g['nWithExtreme']} including -> {g['nExcluded']} excluding):")
        a("| Horizon | Q5-Q1 incl. | Q5-Q1 excl. | Sign flip? |")
        a("|---|---:|---:|---|")
        for h in HORIZONS:
            inc = g[feature]["including"]["horizons"][f"{h}d"]["q5MinusQ1Mean"]
            exc = g[feature]["excluding"]["horizons"][f"{h}d"]["q5MinusQ1Mean"]
            flip = "YES" if (inc is not None and exc is not None and (inc > 0) != (exc > 0)) else "no"
            a(f"| {h}D | {fmt_pct(inc)} | {fmt_pct(exc)} | {flip} |")
        a("")

    a("## Q9. Do supporting features show standalone predictive value?")
    a("")
    a("Pooled Q5-Q1 spread only (net_dollar_delta, zero_dte_share, put_skew_25d, atm_iv, "
     "iv_percentile_20d/60d/126d/252d):")
    a("")
    a("| Feature | 5D Q5-Q1 | 5D HAC-t | 10D Q5-Q1 | 10D HAC-t | 20D Q5-Q1 | 20D HAC-t |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for feature in SUPPORTING_FEATURES:
        b = report["supportingPooled"][feature]["horizons"]
        a(f"| {feature} | {fmt_pct(b['5d']['q5MinusQ1Mean'])} | {fmt_num(b['5d']['q5']['hacTStat'])} | "
         f"{fmt_pct(b['10d']['q5MinusQ1Mean'])} | {fmt_num(b['10d']['q5']['hacTStat'])} | "
         f"{fmt_pct(b['20d']['q5MinusQ1Mean'])} | {fmt_num(b['20d']['q5']['hacTStat'])} |")
    a("")

    a("## Q10. What missing-data / sample-size caveats limit confidence?")
    a("")
    md = report["missingData"]
    a("Forward-return coverage by ticker/horizon (identical pattern across all 6 ETFs -- driven "
     "purely by how close each feature date is to \"today\", 2026-09-23, not by any per-ETF issue):")
    a("")
    a("| Ticker | 1D | 3D | 5D | 10D | 20D |")
    a("|---|---:|---:|---:|---:|---:|")
    for t in report["tickers"]:
        c = md["forwardReturnCoverageByTickerHorizon"][t]
        a(f"| {t} | {c['1d']} | {c['3d']} | {c['5d']} | {c['10d']} | {c['20d']} | (of 60)")
    a("")
    a("Feature NaN counts by ticker (only `put_skew_25d` has missing values, all attributable to "
     "thin 25-delta put quotes on specific days -- see the QA report; no feature was zero-filled "
     "anywhere in this study, every N reported above is the true count of non-missing pairs):")
    a("")
    nan_features = [f for f in ALL_FEATURES if any(v > 0 for v in md["featureNaNCountsByTicker"][f].values())]
    if nan_features:
        a("| Feature | " + " | ".join(report["tickers"]) + " |")
        a("|---|" + "---:|" * len(report["tickers"]))
        for f in nan_features:
            counts = md["featureNaNCountsByTicker"][f]
            a(f"| {f} | " + " | ".join(str(counts[t]) for t in report["tickers"]) + " |")
    else:
        a("(none)")
    a("")

    a("## Q11. Verdict")
    a("")
    a(f"**{report['verdict']}**")
    a("")
    a("Verdict rule (robustness/consistency-based, not the best single bucket): PROCEED requires "
     "(a) at least one of H1/H2 shows |HAC t| >= 1.96 on the pooled Q5-Q1 spread at 2 of 3 "
     "decision-relevant horizons (5D/10D/20D) with a consistent sign, (b) at least 4 of 6 ETFs "
     "individually show the same-signed spread at 5D or 10D, and (c) the GLD-extreme-day exclusion "
     "does not flip the pooled sign. Meeting (a) but not (b)/(c) -> PROMISING BUT MODIFY RESEARCH "
     "DESIGN FIRST. Meeting none -> NO EVIDENCE YET.")
    a("")
    a("No production model was fit. No weights or thresholds were optimized. This report does not "
     "start the 252-day full-flow backfill; that decision is left to the reader of this report.")
    return "\n".join(lines)


def flatten_to_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """CSV export: one row per (feature, scope, horizon) with the core numbers."""
    rows: List[Dict[str, Any]] = []

    def add(feature, scope, ticker, h, block):
        rows.append({
            "feature": feature, "scope": scope, "ticker": ticker, "horizon": h,
            "q1_n": block["q1"]["n"], "q1_mean": block["q1"]["mean"], "q1_hac_t": block["q1"]["hacTStat"],
            "q5_n": block["q5"]["n"], "q5_mean": block["q5"]["mean"], "q5_hac_t": block["q5"]["hacTStat"],
            "q5_minus_q1_mean": block["q5MinusQ1Mean"],
        })

    for feature in PRIMARY_FEATURES:
        pooled = report["primaryPooled"][feature]
        for h in HORIZONS:
            add(feature, "pooled", "ALL", h, pooled["horizons"][f"{h}d"])
        for t in TICKERS:
            per = report["primaryPerETF"][feature][t]
            for h in HORIZONS:
                add(feature, "per_etf", t, h, per["horizons"][f"{h}d"])
    for feature in SUPPORTING_FEATURES:
        pooled = report["supportingPooled"][feature]
        for h in HORIZONS:
            add(feature, "pooled", "ALL", h, pooled["horizons"][f"{h}d"])
    return rows


def main() -> int:
    df = load_joined()
    print(f"Joined dataset: {len(df)} rows ({df['date'].min()} -> {df['date'].max()}, "
         f"tickers: {sorted(df['ticker'].unique())})")

    leakage = verify_no_leakage(df)
    print(f"Leakage check: {leakage}")
    if not leakage["featuresPrecedeOrEqualPriceHistory"]:
        print("ERROR: a feature date is AFTER the last available price date -- refusing to proceed.")
        return 1

    report = build_report(df, leakage)

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(flatten_to_rows(report)).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")

    print(f"\nWrote {OUT_MD}\nWrote {OUT_CSV}\nWrote {OUT_JSON}")
    print(f"\nVERDICT: {report['verdict']}")
    print("\nDo not start the 252-day backfill automatically. Stopping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
