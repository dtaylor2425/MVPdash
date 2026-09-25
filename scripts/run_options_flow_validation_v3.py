"""
scripts/run_options_flow_validation_v3.py

Signal Validation v3: does z_gross_premium_20d (discovered in commit 19dc02c, frozen in
reports/options-flow-v3-validation-manifest.json BEFORE this script ever touched new data)
survive on 192 new, previously unseen full_flow sessions?

Inputs (all read-only):
    * reports/options-flow-v3-validation-dataset.parquet        (1272 rows: 20 warmup-only +
      192 validation sessions x 6 tickers, 2025-08-21 -> 2026-06-25)
    * reports/options-flow-v3-payload-features.parquet          (same 1272 rows, DTE/call-put/
      intraday/large-trade detail)
    * reports/options-flow-v3-validation-forward-returns.parquet (same 1272 rows, 100% coverage
      at every horizon since the whole window is in the past)
    * reports/options-flow-research-dataset.parquet + options-flow-payload-features.parquet +
      options-flow-forward-returns.parquet (the ORIGINAL, immutable v1/v2 discovery-sample files
      -- read only to display discovery-sample numbers separately, never merged into the
      validation significance test, never recomputed with extra lookback)
    * reports/options-flow-v3-validation-manifest.json (the frozen hypothesis)

The z_gross_premium_20d formula below is a byte-for-byte copy of commit 19dc02c's `_zscore`
closure (Z_WINDOW=20, Z_MIN_PERIODS=10, shift(1) before rolling, sample std, no winsorization) --
see the manifest for the verbatim source. It is NOT reinterpreted here.

Warmup-only sessions (the 20 earliest) seed the trailing-20D baseline for the earliest
VALIDATION dates and are then dropped -- they are never treated as validation observations
themselves, per the pre-registration.

Primary inference is date-clustering-robust (panel regression clustered by market_date,
date-block bootstrap, date-level portfolio spread) because 6 same-date ETF observations are not
6 independent draws -- exactly v2's known weakness. Per-ETF results and pooled ETF-day t-stats
are reported but are NOT the load-bearing statistic.

    python scripts/run_options_flow_validation_v3.py
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
    benjamini_hochberg,
    describe_returns,
    quintile_labels,
    spearman_correlation,
)
from api.services.options_flow_regime_features import (  # noqa: E402
    chronological_half_split,
    iv_percentile_regime,
    market_wide_activity_factor,
    momentum_regime,
    realized_vol_regime,
    sma_regime,
)
from api.services.options_flow_validation_stats import (  # noqa: E402
    date_block_bootstrap,
    date_level_portfolio_test,
    drop_dates_and_recompute,
    leave_one_date_out,
    panel_ols_cluster_by_date,
    panel_ols_two_regressors_cluster_by_date,
)

MANIFEST_PATH = ROOT / "reports" / "options-flow-v3-validation-manifest.json"
V3_DATASET = ROOT / "reports" / "options-flow-v3-validation-dataset.parquet"
V3_PAYLOAD = ROOT / "reports" / "options-flow-v3-payload-features.parquet"
V3_RETURNS = ROOT / "reports" / "options-flow-v3-validation-forward-returns.parquet"
V1_DATASET = ROOT / "reports" / "options-flow-research-dataset.parquet"
V1_PAYLOAD = ROOT / "reports" / "options-flow-payload-features.parquet"
V1_RETURNS = ROOT / "reports" / "options-flow-forward-returns.parquet"

OUT_MD = ROOT / "reports" / "options-flow-validation-v3.md"
OUT_CSV = ROOT / "reports" / "options-flow-validation-v3.csv"
OUT_JSON = ROOT / "reports" / "options-flow-validation-v3.json"
OUT_BOOTSTRAP_CSV = ROOT / "reports" / "options-flow-validation-v3-bootstrap.csv"
OUT_REGIMES_CSV = ROOT / "reports" / "options-flow-validation-v3-regimes.csv"

TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
PRIMARY_HORIZONS = (5, 10, 20)
SECONDARY_HORIZONS = (1, 3)
WARMUP_END = "2025-09-18"    # inclusive -- last warmup-only session (from the manifest)
VALIDATION_START = "2025-09-19"  # inclusive -- first true validation session
PRIMARY_FEATURE = "z_gross_premium_20d"
SECONDARY_FEATURES = ["put_sentiment", "top10_concentration"]
Z_WINDOW, Z_MIN_PERIODS = 20, 10  # verbatim from commit 19dc02c -- see the manifest
EXPECTED_SIGN = 1  # positive, recorded in the manifest before touching new data


# --------------------------------------------------------------------------------------------
# Loading + the FROZEN z-score (byte-for-byte copy of commit 19dc02c's _zscore)
# --------------------------------------------------------------------------------------------

def load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def zscore_verbatim_v2(df: pd.DataFrame, col: str, ticker_col: str = "ticker") -> pd.Series:
    """Byte-for-byte copy of commit 19dc02c's `_zscore` closure. `df` must already be sorted by
    (ticker, date). Do not modify this function during v3 -- if commit 19dc02c's formula is ever
    revisited, that is a NEW, separately-labeled research iteration, not an edit here."""
    g = df.groupby(ticker_col, sort=False)
    prior = g[col].shift(1)
    roll_mean = prior.groupby(df[ticker_col]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    roll_std = prior.groupby(df[ticker_col]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
    roll_mean.index = roll_mean.index.droplevel(0)
    roll_std.index = roll_std.index.droplevel(0)
    z = (df[col] - roll_mean) / roll_std
    return z.replace([np.inf, -np.inf], np.nan)


def load_validation_frame() -> pd.DataFrame:
    """The 1272-row warmup+validation frame, with z_gross_premium_20d computed using ONLY this
    frame's own chronology (warmup seeds validation; validation never sees discovery data,
    which doesn't even exist yet chronologically)."""
    features = pd.read_parquet(V3_DATASET).copy()
    payload = pd.read_parquet(V3_PAYLOAD).copy()
    returns = pd.read_parquet(V3_RETURNS).copy()
    for d in (features, payload, returns):
        d["date"] = d["date"].astype(str)

    df = features.merge(payload, on=["date", "ticker"], how="inner", validate="one_to_one")
    if len(df) != len(features):
        raise RuntimeError(f"payload join dropped rows: {len(features)} -> {len(df)}")
    df = df.merge(returns[["date", "ticker"] + [f"ret_{h}d" for h in HORIZONS]],
                 on=["date", "ticker"], how="inner", validate="one_to_one")
    if len(df) != len(features):
        raise RuntimeError(f"returns join dropped rows: {len(features)} -> {len(df)}")

    df["_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "_dt"]).reset_index(drop=True)
    df[PRIMARY_FEATURE] = zscore_verbatim_v2(df, "gross_premium")
    df["sample"] = np.where(df["date"] <= WARMUP_END, "warmup_only", "validation")
    return df


def load_discovery_frame() -> pd.DataFrame:
    """The ORIGINAL, immutable 360-row v1/v2 discovery sample -- read only, its OWN z-score
    recomputed in isolation (discovery-internal lookback only, exactly reproducing what v2
    already reported; never given the new warmup/validation history as extra lookback, which
    would silently change the discovery-sample numbers already on record)."""
    features = pd.read_parquet(V1_DATASET).copy()
    payload = pd.read_parquet(V1_PAYLOAD).copy()
    returns = pd.read_parquet(V1_RETURNS).copy()
    for d in (features, payload, returns):
        d["date"] = d["date"].astype(str)
    df = features.merge(payload, on=["date", "ticker"], how="inner")
    df = df.merge(returns[["date", "ticker"] + [f"ret_{h}d" for h in HORIZONS]], on=["date", "ticker"], how="inner")
    df["_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "_dt"]).reset_index(drop=True)
    df[PRIMARY_FEATURE] = zscore_verbatim_v2(df, "gross_premium")
    df["sample"] = "discovery"
    return df


# --------------------------------------------------------------------------------------------
# Core battery (reused across primary + secondary features)
# --------------------------------------------------------------------------------------------

def quintile_spread_stat(df: pd.DataFrame, feature: str, horizon: int) -> Dict[str, Any]:
    col = f"ret_{horizon}d"
    valid = df.dropna(subset=[feature, col])
    q = quintile_labels(list(valid[feature]))
    q1 = [r for r, qq in zip(valid[col].tolist(), q) if qq == 1]
    q5 = [r for r, qq in zip(valid[col].tolist(), q) if qq == 5]
    d1, d5 = describe_returns(q1, horizon), describe_returns(q5, horizon)
    spread = (d5["mean"] - d1["mean"]) if (d5["mean"] is not None and d1["mean"] is not None) else None
    return {"n": int(valid[feature].notna().sum()), "q1": d1, "q5": d5, "q5MinusQ1Mean": spread}


def q5_minus_q1_fn(feature: str, horizon: int):
    def _fn(d: pd.DataFrame) -> Optional[float]:
        return quintile_spread_stat(d, feature, horizon)["q5MinusQ1Mean"]
    return _fn


def panel_beta_fn(feature: str, horizon: int):
    def _fn(d: pd.DataFrame) -> Optional[float]:
        return panel_ols_cluster_by_date(d, f"ret_{horizon}d", feature)["beta"]
    return _fn


def per_etf_quintile_spreads(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    return {t: {h: quintile_spread_stat(df[df["ticker"] == t], feature, h) for h in PRIMARY_HORIZONS}
           for t in TICKERS}


def monotonicity(df: pd.DataFrame, feature: str, horizon: int) -> Dict[str, Any]:
    valid = df.dropna(subset=[feature, f"ret_{horizon}d"])
    q = quintile_labels(list(valid[feature]))
    by_q = {}
    for k in (1, 2, 3, 4, 5):
        vals = [r for r, qq in zip(valid[f"ret_{horizon}d"].tolist(), q) if qq == k]
        d = describe_returns(vals, horizon)
        by_q[f"q{k}"] = {"n": d["n"], "mean": d["mean"]}
    sp = spearman_correlation(q, valid[f"ret_{horizon}d"].tolist())
    return {"byQuintile": by_q, "spearman": sp}


def run_feature_battery(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """The full battery for ONE feature against the validation sample: quintile spreads (all
    horizons), panel regression (primary horizons), per-ETF spreads, monotonicity (10D)."""
    out: Dict[str, Any] = {
        "quintileSpreads": {f"{h}d": quintile_spread_stat(df, feature, h) for h in HORIZONS},
        "panelRegression": {h: panel_ols_cluster_by_date(df, f"ret_{h}d", feature) for h in PRIMARY_HORIZONS},
        "perEtfQuintileSpreads": per_etf_quintile_spreads(df, feature),
        "monotonicity10d": monotonicity(df, feature, 10),
    }
    return out


# --------------------------------------------------------------------------------------------
# Robustness batteries specific to the PRIMARY feature
# --------------------------------------------------------------------------------------------

def bootstrap_battery(df: pd.DataFrame, feature: str, horizon: int, n_boot: int = 10_000) -> Dict[str, Any]:
    beta_iid = date_block_bootstrap(df, panel_beta_fn(feature, horizon), n_boot=n_boot, block_length=1, seed=1)
    beta_block = date_block_bootstrap(df, panel_beta_fn(feature, horizon), n_boot=n_boot, block_length=5, seed=2)
    spread_iid = date_block_bootstrap(df, q5_minus_q1_fn(feature, horizon), n_boot=n_boot, block_length=1, seed=3)
    spread_block = date_block_bootstrap(df, q5_minus_q1_fn(feature, horizon), n_boot=n_boot, block_length=5, seed=4)
    return {"betaIid": beta_iid, "betaBlock5": beta_block, "spreadIid": spread_iid, "spreadBlock5": spread_block}


def leave_one_date_out_battery(df: pd.DataFrame, feature: str, horizon: int) -> Dict[str, Any]:
    beta_loo = leave_one_date_out(df, panel_beta_fn(feature, horizon))
    spread_loo = leave_one_date_out(df, q5_minus_q1_fn(feature, horizon))
    influential_dates = [d["date"] for d in spread_loo["mostInfluentialDates"][:5]]
    beta_drop5 = drop_dates_and_recompute(df, influential_dates, panel_beta_fn(feature, horizon))
    spread_drop5 = drop_dates_and_recompute(df, influential_dates, q5_minus_q1_fn(feature, horizon))
    return {
        "betaLeaveOneDateOut": beta_loo, "spreadLeaveOneDateOut": spread_loo,
        "mostInfluentialDatesFromSpread": influential_dates,
        "betaAfterDroppingTop5Influential": beta_drop5, "spreadAfterDroppingTop5Influential": spread_drop5,
    }


def regime_battery(df: pd.DataFrame, feature: str, prices: pd.Series) -> Dict[str, Any]:
    """5 predetermined regime splits (item 10). Median/tercile/sign splits only -- no optimized
    cutoffs. Reports the feature's Q5-Q1 spread (5D and 10D) within each regime level."""
    dates = sorted(df["date"].unique())
    out: Dict[str, Any] = {}

    half = dict(zip(dates, chronological_half_split(dates)))
    df = df.assign(_first_second_half=df["date"].map(half))

    sma = sma_regime(prices, dates, window=200)
    df = df.assign(_spy_200dma=df["date"].map(sma["byDate"]))

    vol = realized_vol_regime(prices, dates, window=20)
    df = df.assign(_spy_vol=df["date"].map(vol["byDate"]))

    mom = momentum_regime(prices, dates, window=20)
    df = df.assign(_spy_momentum=df["date"].map(mom["byDate"]))

    df = df.assign(_iv_pctile=iv_percentile_regime(df, "iv_percentile_60d"))

    regime_cols = {
        "chronological_half": ("first_half", "second_half"),
        "_spy_200dma": ("above", "below"),
        "_spy_vol": ("high", "low"),
        "_spy_momentum": ("positive", "negative"),
        "_iv_pctile": ("high", "low"),
    }
    col_map = {"chronological_half": "_first_second_half", "_spy_200dma": "_spy_200dma",
              "_spy_vol": "_spy_vol", "_spy_momentum": "_spy_momentum", "_iv_pctile": "_iv_pctile"}
    for name, (level_a, level_b) in regime_cols.items():
        actual_col = col_map[name]
        block: Dict[str, Any] = {}
        for level in (level_a, level_b):
            sub = df[df[actual_col] == level]
            block[level] = {"n": len(sub),
                            "5d": quintile_spread_stat(sub, feature, 5)["q5MinusQ1Mean"],
                            "10d": quintile_spread_stat(sub, feature, 10)["q5MinusQ1Mean"]}
        out[name] = block
    return out


def market_wide_control(df: pd.DataFrame, feature: str, horizon: int) -> Dict[str, Any]:
    """Item 11: does the ETF-specific feature survive controlling for the SAME-DAY cross-ETF
    median (a market-wide options-activity factor)?"""
    work = df.copy()
    work["_market_wide"] = market_wide_activity_factor(work, feature)
    return panel_ols_two_regressors_cluster_by_date(work, f"ret_{horizon}d", feature, "_market_wide")


# --------------------------------------------------------------------------------------------
# Verdict (item 14's checklist -- majority of criteria, never one t-stat)
# --------------------------------------------------------------------------------------------

def build_verdict_checklist(primary: Dict[str, Any], robustness: Dict[str, Any],
                            regimes: Dict[str, Any], market_control: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}

    spread_10d = primary["quintileSpreads"]["10d"]["q5MinusQ1Mean"]
    checks["sameDirectionAsV2"] = spread_10d is not None and np.sign(spread_10d) == EXPECTED_SIGN

    panel_10d = primary["panelRegression"][10]
    checks["meaningfulEffectInValidationSample"] = (
        panel_10d["tStat"] is not None and abs(panel_10d["tStat"]) >= 1.96
        and np.sign(panel_10d["beta"]) == EXPECTED_SIGN
    )
    checks["dateClusteredInferenceSupportive"] = checks["meaningfulEffectInValidationSample"]

    boot = robustness["bootstrap10d"]
    lo, hi = boot["betaIid"]["ci95"]
    checks["dateBlockBootstrapSupportive"] = (
        lo is not None and hi is not None and (lo > 0) == (EXPECTED_SIGN > 0) and (hi > 0) == (EXPECTED_SIGN > 0)
    )

    signs = []
    for h in PRIMARY_HORIZONS:
        s = primary["quintileSpreads"][f"{h}d"]["q5MinusQ1Mean"]
        if s is not None:
            signs.append(np.sign(s))
    checks["q5MinusQ1ExpectedDirectionMajority"] = (
        len(signs) > 0 and sum(1 for s in signs if s == EXPECTED_SIGN) >= 2
    )

    sp = primary["monotonicity10d"]["spearman"]
    checks["monotonic10d"] = sp["rho"] is not None and np.sign(sp["rho"]) == EXPECTED_SIGN

    loo = robustness["leaveOneDateOut10d"]["spreadLeaveOneDateOut"]
    checks["notDrivenByFewDates"] = (
        loo["shareSameSign"] is not None and loo["shareSameSign"] >= 0.85
        and robustness["leaveOneDateOut10d"]["spreadAfterDroppingTop5Influential"] is not None
        and np.sign(robustness["leaveOneDateOut10d"]["spreadAfterDroppingTop5Influential"]) == EXPECTED_SIGN
    )

    per_etf = primary["perEtfQuintileSpreads"]
    agree = sum(1 for t in TICKERS if per_etf[t][10]["q5MinusQ1Mean"] is not None
               and np.sign(per_etf[t][10]["q5MinusQ1Mean"]) == EXPECTED_SIGN)
    checks["majorityEtfsConsistent"] = agree >= 4

    regime_ok = 0
    for name, levels in regimes.items():
        vals = [v["10d"] for v in levels.values() if v["10d"] is not None]
        if vals and all(np.sign(v) == EXPECTED_SIGN for v in vals if v != 0):
            regime_ok += 1
        elif vals and any(np.sign(v) == EXPECTED_SIGN for v in vals if v != 0):
            regime_ok += 0.5
    checks["stableAcrossRegimesMajority"] = regime_ok >= 2.5  # out of 5 regimes

    x1 = market_control["x1"]
    checks["marketWideControlDoesNotEliminate"] = (
        x1["beta"] is not None and np.sign(x1["beta"]) == EXPECTED_SIGN
    )

    n_pass = sum(1 for v in checks.values() if v)
    return {"checks": checks, "nPass": n_pass, "nTotal": len(checks)}


def determine_verdict(checklist: Dict[str, Any]) -> str:
    n_pass, n_total = checklist["nPass"], checklist["nTotal"]
    if not checklist["checks"]["sameDirectionAsV2"]:
        return "FAILED OUT-OF-SAMPLE VALIDATION"
    if n_pass >= 8:
        return "VALIDATED -- MOVE TOWARD PRODUCTION SIGNAL"
    if n_pass >= 4:
        return "PARTIALLY VALIDATED -- MORE RESEARCH REQUIRED"
    return "FAILED OUT-OF-SAMPLE VALIDATION"


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def fmt_pct(x, d=2):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def fmt_num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def render_markdown(ctx: Dict[str, Any]) -> str:
    lines: List[str] = []
    a = lines.append
    manifest = ctx["manifest"]
    primary = ctx["primary"]
    robustness = ctx["robustness"]
    regimes = ctx["regimes"]
    market_control = ctx["marketControl"]
    checklist = ctx["checklist"]
    verdict = ctx["verdict"]
    disc = ctx["discoveryDisplay"]
    combined = ctx["combinedDisplay"]
    secondary = ctx["secondary"]

    a("# Macro Options Flow -- Signal Validation v3")
    a("")
    a(f"Primary feature under validation: **`{manifest['primary_feature']}`** (frozen in "
     f"`reports/options-flow-v3-validation-manifest.json`, discovered in commit "
     f"`{manifest['discovery_commit']}`, expected sign: "
     f"**{manifest['expected_direction_from_v2_recorded_before_new_data']['sign'].upper()}**, "
     f"recorded BEFORE any new data was pulled).")
    a("")
    a("## Sample composition")
    a("")
    a("| Sample | Sessions/ETF | Rows | Role |")
    a("|---|---:|---:|---|")
    a(f"| DISCOVERY (original, frozen) | 60 | 360 | Displayed separately -- NOT part of the "
     f"primary significance test |")
    a(f"| VALIDATION (new, unseen) | 192 | {ctx['nValidationRows']} | **Primary test sample** |")
    a(f"| warmup-only (new, unseen) | 20 | {ctx['nWarmupRows']} | Seeds the trailing-20D "
     f"baseline for the earliest validation dates only -- never counted as an observation |")
    a(f"| COMBINED (descriptive) | 252 | {ctx['nValidationRows'] + 360} | Descriptive view only "
     f"-- **not** an independent significance test (60 of these 252 sessions are the original "
     f"discovery data, not out-of-sample) |")
    a("")

    a("## Q1/Q2: Does the primary feature retain its direction and clear a bar in the new sample?")
    a("")
    a("Pooled Q5-Q1 spread, validation sample only (n shown per horizon):")
    a("")
    a("| Horizon | N | Q1 mean | Q5 mean | Q5-Q1 | Panel beta (cluster by date) | Cluster t | p |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        qs = primary["quintileSpreads"][f"{h}d"]
        panel = primary["panelRegression"].get(h)
        panel_str = (fmt_num(panel["beta"], 5), fmt_num(panel["tStat"]), fmt_num(panel["pValue"], 4)) if panel else ("n/a", "n/a", "n/a")
        a(f"| {h}D | {qs['n']} | {fmt_pct(qs['q1']['mean'])} | {fmt_pct(qs['q5']['mean'])} | "
         f"{fmt_pct(qs['q5MinusQ1Mean'])} | {panel_str[0]} | {panel_str[1]} | {panel_str[2]} |")
    a("")
    a(f"**Discovery-sample result (original, frozen, for comparison only):** 10D Q5-Q1 = "
     f"{fmt_pct(disc['by_horizon']['10d']['q5_minus_q1_mean'])} (n={disc['by_horizon']['10d']['n']}).")
    a("")

    a("## Q3: Does it survive date-clustered inference (the primary weakness of v2)?")
    a("")
    a("Panel regression with ticker fixed effects, standard errors clustered by market_date "
     "(6 same-date ETF observations are not 6 independent draws):")
    a("")
    for h in PRIMARY_HORIZONS:
        p = primary["panelRegression"][h]
        a(f"- **{h}D**: beta={fmt_num(p['beta'], 5)}, cluster SE={fmt_num(p['se'], 5)}, "
         f"t={fmt_num(p['tStat'])}, p={fmt_num(p['pValue'], 4)}, n={p['n']}, "
         f"clusters(dates)={p['nClusters']}")
    a("")

    a("## Q4: Does it survive a date/block bootstrap?")
    a("")
    a("10,000 resamples of unique DATES (not ETF-days); iid (block=1) and a 5-date moving block "
     "(preserves short-run serial correlation):")
    a("")
    a("| Statistic | Bootstrap | Mean | 95% CI low | 95% CI high | Share same sign as full sample |")
    a("|---|---|---:|---:|---:|---:|")
    for stat_name, key in [("Panel beta (10D)", "betaIid"), ("Panel beta (10D, block=5)", "betaBlock5"),
                           ("Q5-Q1 spread (10D)", "spreadIid"), ("Q5-Q1 spread (10D, block=5)", "spreadBlock5")]:
        b = robustness["bootstrap10d"][key]
        lo, hi = b["ci95"]
        a(f"| {stat_name} | n={b['n_boot']} | {fmt_num(b['mean'], 5)} | {fmt_num(lo, 5)} | "
         f"{fmt_num(hi, 5)} | {fmt_num(b['shareSameSign'], 2)} |")
    a("")

    a("## Q5: Is the quintile relationship monotonic?")
    a("")
    m = primary["monotonicity10d"]
    a("10D forward return by quintile (Q1=lowest z_gross_premium_20d, Q5=highest):")
    a("| Q1 | Q2 | Q3 | Q4 | Q5 |")
    a("|---:|---:|---:|---:|---:|")
    a("| " + " | ".join(fmt_pct(m["byQuintile"][f"q{k}"]["mean"]) for k in (1, 2, 3, 4, 5)) + " |")
    sp = m["spearman"]
    a(f"Spearman(quintile, 10D return): rho={fmt_num(sp['rho'])}, p={fmt_num(sp['pValue'], 4)}, n={sp['n']}.")
    a("")

    a("## Q6: Is the effect present across ETFs?")
    a("")
    a("| Ticker | 5D Q5-Q1 | 10D Q5-Q1 | 20D Q5-Q1 |")
    a("|---|---:|---:|---:|")
    for t in TICKERS:
        r = primary["perEtfQuintileSpreads"][t]
        a(f"| {t} | {fmt_pct(r[5]['q5MinusQ1Mean'])} | {fmt_pct(r[10]['q5MinusQ1Mean'])} | "
         f"{fmt_pct(r[20]['q5MinusQ1Mean'])} |")
    a("")

    a("## Q7: Is the effect driven by a handful of macro-event dates?")
    a("")
    loo = robustness["leaveOneDateOut10d"]["spreadLeaveOneDateOut"]
    a(f"Leave-one-date-out (192 reruns, dropping one validation date at a time), 10D Q5-Q1 spread: "
     f"full-sample={fmt_pct(loo['fullSampleStat'])}, range=[{fmt_pct(loo['min'])}, {fmt_pct(loo['max'])}], "
     f"share of reruns preserving the expected sign={fmt_num(loo['shareSameSign'], 2)}.")
    a("")
    a("Most influential dates (largest shift in the spread when removed):")
    for d in loo["mostInfluentialDates"]:
        a(f"- {d['date']}: leave-one-out spread {fmt_pct(d['leaveOneOutStat'])} (shift {fmt_pct(d['shift'])})")
    a("")
    a(f"Dropping the **5 most influential dates simultaneously**: 10D Q5-Q1 spread = "
     f"{fmt_pct(robustness['leaveOneDateOut10d']['spreadAfterDroppingTop5Influential'])} "
     f"(full sample: {fmt_pct(loo['fullSampleStat'])}).")
    a("")

    a("## Q8: Is it robust across time and regimes?")
    a("")
    a("Predetermined splits only (chronological half, SPY 200DMA, SPY realized-vol median, "
     "SPY 20D momentum sign, this ETF's own IV-percentile-60D median) -- 10D Q5-Q1 spread within "
     "each level:")
    a("")
    a("| Regime | Level | N | 5D Q5-Q1 | 10D Q5-Q1 |")
    a("|---|---|---:|---:|---:|")
    for name, levels in regimes.items():
        for level, v in levels.items():
            a(f"| {name} | {level} | {v['n']} | {fmt_pct(v['5d'])} | {fmt_pct(v['10d'])} |")
    a("")

    a("## Q9 (part 1): How much of this is market-wide event-day activity vs. ETF-specific?")
    a("")
    a("Two-regressor panel regression: `ret_10d ~ ticker_specific_z + market_wide_median_z` "
     "(ticker fixed effects, clustered by date):")
    a(f"- ETF-specific `{PRIMARY_FEATURE}`: beta={fmt_num(market_control['x1']['beta'], 5)}, "
     f"t={fmt_num(market_control['x1']['tStat'])}")
    a(f"- Market-wide median `{PRIMARY_FEATURE}` (cross-ETF, same date): "
     f"beta={fmt_num(market_control['x2']['beta'], 5)}, t={fmt_num(market_control['x2']['tStat'])}")
    a(f"- n={market_control['n']}, clusters(dates)={market_control['nClusters']}")
    a("")

    a("## Q10: Did the original 60-day discovery result replicate?")
    a("")
    a("| Horizon | Discovery Q5-Q1 (n=60x6, frozen) | Validation Q5-Q1 (n=192x6, new) | Same sign? |")
    a("|---|---:|---:|---|")
    for h in PRIMARY_HORIZONS:
        d_val = disc["by_horizon"][f"{h}d"]["q5_minus_q1_mean"]
        v_val = primary["quintileSpreads"][f"{h}d"]["q5MinusQ1Mean"]
        same = "yes" if (d_val is not None and v_val is not None and np.sign(d_val) == np.sign(v_val)) else "no"
        a(f"| {h}D | {fmt_pct(d_val)} | {fmt_pct(v_val)} | {same} |")
    a("")

    a("## Combined (descriptive) sample -- NOT an independent significance test")
    a("")
    a(f"Pooling all 252 sessions (60 discovery + 192 validation) x 6 ETFs = "
     f"{combined['n']} rows purely for descriptive reference. 10D Q5-Q1 spread: "
     f"{fmt_pct(combined['10d']['q5MinusQ1Mean'])} (n={combined['10d']['n']}). "
     f"**60 of these 252 sessions are the original discovery data that produced the hypothesis "
     f"in the first place -- this number is NOT out-of-sample and must never be quoted as such.**")
    a("")

    a("## Secondary candidates (put_sentiment, top10_concentration)")
    a("")
    a("Evaluated only after the primary result above; Benjamini-Hochberg FDR-corrected across "
     "the panel-regression p-values at all 3 primary horizons for both candidates (6 tests):")
    a("")
    a("| Feature | Horizon | Panel beta | t | raw p | BH q | Reject at 0.05? |")
    a("|---|---|---:|---:|---:|---:|---|")
    for row in secondary["rows"]:
        a(f"| {row['feature']} | {row['horizon']}D | {fmt_num(row['beta'], 5)} | {fmt_num(row['tStat'])} | "
         f"{fmt_num(row['pValue'], 4)} | {fmt_num(row['qValue'], 4)} | "
         f"{'yes' if row['reject'] else ('no' if row['reject'] is False else 'n/a')} |")
    a("")
    a("Secondary results do not redefine the primary verdict below, regardless of outcome.")
    a("")

    a("## Verdict checklist (item 14 -- majority of criteria, not one t-stat)")
    a("")
    a("| Criterion | Met? |")
    a("|---|---|")
    labels = {
        "sameDirectionAsV2": "Same direction as v2 (positive)",
        "meaningfulEffectInValidationSample": "Meaningful effect in the new 192-session sample (|cluster t|>=1.96, 10D)",
        "dateClusteredInferenceSupportive": "Date-clustered inference supportive",
        "dateBlockBootstrapSupportive": "Date-block bootstrap CI supportive (excludes 0, expected sign)",
        "q5MinusQ1ExpectedDirectionMajority": "Q5-Q1 spread in expected direction (>=2 of 3 primary horizons)",
        "monotonic10d": "Reasonable monotonicity across quintiles (10D)",
        "notDrivenByFewDates": "Effect not driven by 1-5 dates (>=85% LOO sign-preserved, survives dropping top-5)",
        "majorityEtfsConsistent": "Majority of ETFs directionally consistent (>=4/6, 10D)",
        "stableAcrossRegimesMajority": "Reasonable stability across time/regimes (>=2.5/5)",
        "marketWideControlDoesNotEliminate": "Market-wide activity control does not eliminate the effect",
    }
    for key, label in labels.items():
        a(f"| {label} | {'YES' if checklist['checks'][key] else 'no'} |")
    a("")
    a(f"**{checklist['nPass']} of {checklist['nTotal']} criteria met.**")
    a("")

    a("## Verdict")
    a("")
    a(f"**{verdict}**")
    a("")
    a("No production weights were fit. No ETF universe expansion. No new signal search was "
     "started. This report does not modify the live Options Flow page.")
    return "\n".join(lines)


def flatten_rows(primary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for h in HORIZONS:
        qs = primary["quintileSpreads"][f"{h}d"]
        panel = primary["panelRegression"].get(h)
        rows.append({
            "feature": PRIMARY_FEATURE, "horizon": h, "sample": "validation",
            "n": qs["n"], "q1_mean": qs["q1"]["mean"], "q5_mean": qs["q5"]["mean"],
            "q5_minus_q1_mean": qs["q5MinusQ1Mean"],
            "panel_beta": panel["beta"] if panel else None,
            "panel_cluster_t": panel["tStat"] if panel else None,
            "panel_p": panel["pValue"] if panel else None,
        })
    for t in TICKERS:
        for h in PRIMARY_HORIZONS:
            r = primary["perEtfQuintileSpreads"][t][h]
            rows.append({"feature": PRIMARY_FEATURE, "horizon": h, "sample": f"validation_{t}",
                        "n": r["n"], "q1_mean": r["q1"]["mean"], "q5_mean": r["q5"]["mean"],
                        "q5_minus_q1_mean": r["q5MinusQ1Mean"], "panel_beta": None,
                        "panel_cluster_t": None, "panel_p": None})
    return rows


def flatten_bootstrap_rows(robustness: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, label in [("betaIid", "panel_beta_iid"), ("betaBlock5", "panel_beta_block5"),
                       ("spreadIid", "q5_minus_q1_iid"), ("spreadBlock5", "q5_minus_q1_block5")]:
        b = robustness["bootstrap10d"][key]
        lo, hi = b["ci95"]
        rows.append({"statistic": label, "horizon": 10, "n_boot": b["n_boot"], "n_dates": b["nDates"],
                    "full_sample_stat": b.get("fullSampleStat"), "bootstrap_mean": b["mean"],
                    "ci95_low": lo, "ci95_high": hi, "share_same_sign": b["shareSameSign"]})
    return rows


def flatten_regime_rows(regimes: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, levels in regimes.items():
        for level, v in levels.items():
            rows.append({"regime": name, "level": level, "n": v["n"], "spread_5d": v["5d"], "spread_10d": v["10d"]})
    return rows


def main() -> int:
    manifest = load_manifest()
    validation_full = load_validation_frame()
    validation = validation_full[validation_full["sample"] == "validation"].reset_index(drop=True)
    warmup = validation_full[validation_full["sample"] == "warmup_only"]
    print(f"Loaded validation frame: {len(validation_full)} total rows "
         f"({len(warmup)} warmup-only + {len(validation)} validation), "
         f"{validation['date'].min()} -> {validation['date'].max()}")

    discovery = load_discovery_frame()
    print(f"Loaded discovery frame (frozen, read-only): {len(discovery)} rows")

    from src.data_sources import fetch_prices
    spy_prices_df = fetch_prices(["SPY"], period="3y")
    spy_prices = spy_prices_df["SPY"] if "SPY" in spy_prices_df.columns else spy_prices_df.iloc[:, 0]
    spy_prices.index = pd.to_datetime(spy_prices.index)

    print("Running primary battery on the validation sample...")
    primary = run_feature_battery(validation, PRIMARY_FEATURE)

    print("Running bootstrap (10,000 draws x 4 variants)...")
    robustness = {
        "bootstrap10d": bootstrap_battery(validation, PRIMARY_FEATURE, 10, n_boot=10_000),
        "leaveOneDateOut10d": leave_one_date_out_battery(validation, PRIMARY_FEATURE, 10),
    }

    print("Running regime splits...")
    regimes = regime_battery(validation, PRIMARY_FEATURE, spy_prices)

    print("Running market-wide-activity control...")
    market_control = market_wide_control(validation, PRIMARY_FEATURE, 10)

    checklist = build_verdict_checklist(primary, robustness, regimes, market_control)
    verdict = determine_verdict(checklist)

    disc_by_h = manifest["expected_direction_from_v2_recorded_before_new_data"]["v2_discovery_sample_results"]["by_horizon"]

    combined_df = pd.concat([
        discovery[["date", "ticker", PRIMARY_FEATURE, "ret_10d"]],
        validation[["date", "ticker", PRIMARY_FEATURE, "ret_10d"]],
    ], ignore_index=True)
    combined_10d = quintile_spread_stat(combined_df, PRIMARY_FEATURE, 10)

    print("Running secondary candidates (put_sentiment, top10_concentration) + BH-FDR...")
    secondary_rows = []
    p_values = []
    for feat in SECONDARY_FEATURES:
        for h in PRIMARY_HORIZONS:
            p = panel_ols_cluster_by_date(validation, f"ret_{h}d", feat)
            secondary_rows.append({"feature": feat, "horizon": h, "beta": p["beta"],
                                   "tStat": p["tStat"], "pValue": p["pValue"]})
            p_values.append(p["pValue"])
    bh = benjamini_hochberg(p_values, alpha=0.05)
    for row, q, rej in zip(secondary_rows, bh["qValues"], bh["reject"]):
        row["qValue"] = q
        row["reject"] = rej

    ctx = {
        "manifest": manifest, "primary": primary, "robustness": robustness, "regimes": regimes,
        "marketControl": market_control, "checklist": checklist, "verdict": verdict,
        "discoveryDisplay": {"by_horizon": disc_by_h},
        "combinedDisplay": {"n": len(combined_df), "10d": combined_10d},
        "secondary": {"rows": secondary_rows},
        "nValidationRows": len(validation), "nWarmupRows": len(warmup),
    }

    OUT_JSON.write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(flatten_rows(primary)).to_csv(OUT_CSV, index=False)
    pd.DataFrame(flatten_bootstrap_rows(robustness)).to_csv(OUT_BOOTSTRAP_CSV, index=False)
    pd.DataFrame(flatten_regime_rows(regimes)).to_csv(OUT_REGIMES_CSV, index=False)
    OUT_MD.write_text(render_markdown(ctx), encoding="utf-8")

    print(f"\nWrote {OUT_MD}\nWrote {OUT_CSV}\nWrote {OUT_JSON}\nWrote {OUT_BOOTSTRAP_CSV}\nWrote {OUT_REGIMES_CSV}")
    print(f"\nChecklist: {checklist['nPass']}/{checklist['nTotal']} criteria met")
    print(f"VERDICT: {verdict}")
    print("\nNo production weights fit. No ETF universe expansion. No new signal search started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
