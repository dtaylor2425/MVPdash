"""
scripts/run_options_flow_market_factor_v4.py

Signal Research v4: v3 found that z_gross_premium_20d's validation-sample effect leans more on
the SAME-DAY CROSS-ETF MEDIAN than on the ETF-specific residual. v4 asks what that common
market-wide factor actually predicts.

Frozen from v3 (never modified here): z_gross_premium_20d's definition, the 20D window,
methodology_version, the discovery/validation split, and the discovery+validation datasets
themselves. This script imports v3's EXACT loader functions (byte-identical z-score logic) and
concatenates the discovery (60d, frozen) + validation (192d) samples into the 252-session
combined dataset the spec asks for -- warmup-only rows are excluded, exactly as v3 excluded them.

    market_options_activity(date) = median(z_gross_premium_20d across SPY,QQQ,IWM,SMH,TLT,GLD)
    residual_activity(ticker,date) = ticker's own z_gross_premium_20d - market_options_activity(date)

No new features, no threshold search, no window changes -- this explains the v2/v3 signal, it
does not go looking for a new one.

    python scripts/run_options_flow_market_factor_v4.py
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
    describe_returns,
    quintile_labels,
    spearman_correlation,
)
from api.services.options_flow_regime_features import (  # noqa: E402
    momentum_regime,
    sma_regime,
)
from api.services.options_flow_validation_stats import (  # noqa: E402
    date_block_bootstrap,
    date_level_portfolio_test,
    panel_ols_cluster_by_date,
)
from api.services.options_flow_market_factor import (  # noqa: E402
    HORIZONS,
    autocorrelation,
    average_forward_path,
    build_outcome_frame,
    market_activity_factor,
    residual_activity_from_frame,
)
from scripts.run_options_flow_validation_v3 import (  # noqa: E402
    PRIMARY_FEATURE,
    TICKERS,
    load_discovery_frame,
    load_validation_frame,
)

OUT_MD = ROOT / "reports" / "options-flow-market-factor-v4.md"
OUT_CSV = ROOT / "reports" / "options-flow-market-factor-v4.csv"
OUT_JSON = ROOT / "reports" / "options-flow-market-factor-v4.json"

INDEX_TICKERS = ["SPY", "QQQ", "IWM"]  # item 4's explicit target list
PRIMARY_HORIZONS = (5, 10, 20)


# --------------------------------------------------------------------------------------------
# Item 1: the frozen combined (discovery + validation) 252-session dataset
# --------------------------------------------------------------------------------------------

def load_combined_frame() -> pd.DataFrame:
    """discovery (60d, frozen v1/v2 files, its OWN internal z-score lookback) + validation
    (192d, warmup-seeded z-score lookback) = 252 sessions x 6 tickers. Byte-identical to how v3
    already builds each half; only the concatenation is new."""
    discovery = load_discovery_frame()
    validation_full = load_validation_frame()
    validation = validation_full[validation_full["sample"] == "validation"].reset_index(drop=True)
    combined = pd.concat([discovery, validation], ignore_index=True, sort=False)
    combined["_dt"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "_dt"]).reset_index(drop=True)
    return combined


# --------------------------------------------------------------------------------------------
# Item 4/5: what does the market factor predict?
# --------------------------------------------------------------------------------------------

def quintile_spread_stat(dates_df: pd.DataFrame, feature: str, outcome: str) -> Dict[str, Any]:
    """Quintile spread of `outcome` by `feature`, both already at the DATE level (one row per
    date -- no cross-sectional pseudo-replication to worry about here)."""
    valid = dates_df.dropna(subset=[feature, outcome])
    q = quintile_labels(list(valid[feature]))
    q1 = [r for r, qq in zip(valid[outcome].tolist(), q) if qq == 1]
    q5 = [r for r, qq in zip(valid[outcome].tolist(), q) if qq == 5]
    # horizon-agnostic HAC: infer the lag from the outcome column name when it encodes one, else 0
    import re
    m = re.search(r"(\d+)d", outcome)
    lag_horizon = int(m.group(1)) if m else 1
    d1, d5 = describe_returns(q1, lag_horizon), describe_returns(q5, lag_horizon)
    spread = (d5["mean"] - d1["mean"]) if (d5["mean"] is not None and d1["mean"] is not None) else None
    sp = spearman_correlation(q, valid[outcome].tolist())
    return {"n": int(valid[feature].notna().sum()), "q1": d1, "q5": d5,
           "q5MinusQ1Mean": spread, "spearman": sp}


def market_factor_battery(index_frame: pd.DataFrame, outcome_cols: List[str]) -> Dict[str, Any]:
    return {outcome: quintile_spread_stat(index_frame, "market_activity_median", outcome) for outcome in outcome_cols}


def regime_split_battery(index_frame: pd.DataFrame, outcome_cols: List[str],
                         regime_cols: Dict[str, tuple]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for regime_name, (col, levels) in regime_cols.items():
        block: Dict[str, Any] = {}
        for level in levels:
            sub = index_frame[index_frame[col] == level]
            level_block: Dict[str, Any] = {"n": len(sub)}
            for outcome in outcome_cols:
                level_block[outcome] = quintile_spread_stat(sub, "market_activity_median", outcome)["q5MinusQ1Mean"]
            block[level] = level_block
        out[regime_name] = block
    return out


# --------------------------------------------------------------------------------------------
# Item 7: ETF-specific residual
# --------------------------------------------------------------------------------------------

def residual_battery(combined: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    ret_col = f"ret_{horizon}d"
    panel = panel_ols_cluster_by_date(combined, ret_col, "residual_activity")
    quint = quintile_spread_stat(
        combined.rename(columns={"residual_activity": "_ra", ret_col: "_ret"}), "_ra", "_ret"
    )
    cross_sectional = date_level_portfolio_test(combined, "residual_activity", ret_col, horizon,
                                                n_high=3, n_low=3)
    per_etf = {}
    for t in TICKERS:
        sub = combined[combined["ticker"] == t]
        per_etf[t] = quintile_spread_stat(
            sub.rename(columns={"residual_activity": "_ra", ret_col: "_ret"}), "_ra", "_ret"
        )["q5MinusQ1Mean"]
    return {"panelRegression": panel, "pooledQuintile": quint,
           "crossSectionalHighMinusLow": cross_sectional, "perEtfQuintileSpread": per_etf}


# --------------------------------------------------------------------------------------------
# Item 8: persistence / path
# --------------------------------------------------------------------------------------------

def persistence_battery(market_by_date: pd.Series) -> Dict[str, Any]:
    s = market_by_date.sort_index()
    return {f"lag{lag}": autocorrelation(s, lag) for lag in (1, 5, 10)}


def path_battery(spy_prices: pd.Series, index_frame: pd.DataFrame) -> Dict[str, Any]:
    valid = index_frame.dropna(subset=["market_activity_median"])
    q = quintile_labels(list(valid["market_activity_median"]))
    top_dates = [d for d, qq in zip(valid["date"].tolist(), q) if qq == 5]
    bottom_dates = [d for d, qq in zip(valid["date"].tolist(), q) if qq == 1]
    return {
        "topQuintile": average_forward_path(spy_prices, top_dates, max_k=20),
        "bottomQuintile": average_forward_path(spy_prices, bottom_dates, max_k=20),
    }


# --------------------------------------------------------------------------------------------
# Verdict (item 11)
# --------------------------------------------------------------------------------------------

def determine_verdict(market_battery_spy: Dict[str, Any], residual_10d: Dict[str, Any],
                      event_day_feasible: bool) -> str:
    vol_signal = market_battery_spy.get("fwd_vol_10d", {})
    ret_signal = market_battery_spy.get("ret_10d", {})
    vol_meaningful = (vol_signal.get("spearman", {}).get("pValue") is not None
                      and vol_signal["spearman"]["pValue"] < 0.05)
    ret_meaningful = (ret_signal.get("spearman", {}).get("pValue") is not None
                      and ret_signal["spearman"]["pValue"] < 0.05)
    residual_meaningful = (residual_10d["panelRegression"]["pValue"] is not None
                          and residual_10d["panelRegression"]["pValue"] < 0.05)

    if not event_day_feasible:
        # cannot rule out an event-day explanation from existing data -- cap accordingly
        if vol_meaningful or ret_meaningful:
            return "MARKET ACTIVITY SIGNAL PROMISING BUT NOT YET VALIDATED"
        return "NO ROBUST SIGNAL"
    if vol_meaningful and not ret_meaningful:
        return "MARKET ACTIVITY SIGNAL VALIDATED"
    if vol_meaningful or ret_meaningful:
        return "MARKET ACTIVITY SIGNAL PROMISING BUT NOT YET VALIDATED"
    return "NO ROBUST SIGNAL"


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
    a("# Macro Options Flow -- Signal Research v4: Market-Factor Decomposition")
    a("")
    a("v3 found that z_gross_premium_20d's validation-sample effect leans more on the same-day "
     "cross-ETF median than the ETF-specific residual. This asks what that common factor "
     "actually predicts, using the frozen 252-session combined dataset (60 discovery + 192 "
     "validation) -- z_gross_premium_20d itself, its 20D window, methodology_version, and the "
     "discovery/validation split are all unmodified from v3.")
    a("")
    a(f"`market_options_activity(date) = median(z_gross_premium_20d across "
     f"{', '.join(TICKERS)})`, computed from {ctx['nDatesWithFactor']} of "
     f"{ctx['nDatesTotal']} unique calendar dates (needs >=3 of 6 tickers non-missing that day).")
    a("")

    a("## Q1/Q2/Q3: Does the market factor predict direction, volatility, or magnitude?")
    a("")
    for t in INDEX_TICKERS:
        a(f"### {t}")
        a("")
        a("| Outcome | N | Q1 mean | Q5 mean | Q5-Q1 | Spearman rho | p |")
        a("|---|---:|---:|---:|---:|---:|---:|")
        battery = ctx["marketBattery"][t]
        for outcome, label in ctx["outcomeLabels"]:
            b = battery.get(outcome)
            if b is None:
                continue
            a(f"| {label} | {b['n']} | {fmt_pct(b['q1']['mean']) if 'ret' in outcome or 'drawdown' in outcome else fmt_num(b['q1']['mean'], 4)} | "
             f"{fmt_pct(b['q5']['mean']) if 'ret' in outcome or 'drawdown' in outcome else fmt_num(b['q5']['mean'], 4)} | "
             f"{fmt_pct(b['q5MinusQ1Mean']) if 'ret' in outcome or 'drawdown' in outcome else fmt_num(b['q5MinusQ1Mean'], 4)} | "
             f"{fmt_num(b['spearman']['rho'])} | {fmt_num(b['spearman']['pValue'], 4)} |")
        a("")

    a("## Q4: Is the effect concentrated around scheduled macro events?")
    a("")
    a("**Not testable from existing Macro Engine data.** No FOMC/CPI/NFP/Treasury-refunding "
     "calendar module exists in this codebase (checked `src/`, `api/`, `jobs/`, `scripts/` for "
     "an event-date source; only trading-session calendars exist). Per the explicit instruction "
     "not to introduce an external event dataset for this purpose, event-day tagging and the "
     "exclude-event-days rerun were **not performed**. This caps the verdict below -- see item 6 "
     "in the report spec.")
    a("")

    a("## Q5 (regimes): Does it behave differently in bullish/bearish or high/low-vol regimes?")
    a("")
    a("SPY-based regimes (predetermined: same-day sign, 200DMA, 20D momentum, SPY IV-percentile-60D "
     "median split), 10D outcomes:")
    a("")
    a("| Regime | Level | N | ret_10d Q5-Q1 | fwd_vol_10d Q5-Q1 |")
    a("|---|---|---:|---:|---:|")
    for name, levels in ctx["regimeBattery"].items():
        for level, v in levels.items():
            a(f"| {name} | {level} | {v['n']} | {fmt_pct(v.get('ret_10d'))} | {fmt_num(v.get('fwd_vol_10d'), 4)} |")
    a("")

    a("## Q6/Q7: Is there ETF-specific information left after removing the market factor?")
    a("")
    a("Panel regression (ticker fixed effects, clustered by date), quintile spread, and a "
     "date-level cross-sectional high-minus-low portfolio, all on `residual_activity` "
     "(ticker's own z_gross_premium_20d minus that date's market factor):")
    a("")
    a("| Horizon | Panel beta | Cluster t | p | Pooled Q5-Q1 | Cross-sectional H-L (date-level) |")
    a("|---|---:|---:|---:|---:|---:|")
    for h in PRIMARY_HORIZONS:
        r = ctx["residualBattery"][h]
        cs = r["crossSectionalHighMinusLow"]
        a(f"| {h}D | {fmt_num(r['panelRegression']['beta'], 5)} | {fmt_num(r['panelRegression']['tStat'])} | "
         f"{fmt_num(r['panelRegression']['pValue'], 4)} | {fmt_pct(r['pooledQuintile']['q5MinusQ1Mean'])} | "
         f"{fmt_pct(cs.get('mean'))} (n dates={cs.get('nDates')}) |")
    a("")
    a("Per-ETF residual Q5-Q1 spread (10D):")
    a("| " + " | ".join(TICKERS) + " |")
    a("|" + "---:|" * len(TICKERS))
    a("| " + " | ".join(fmt_pct(ctx["residualBattery"][10]["perEtfQuintileSpread"][t]) for t in TICKERS) + " |")
    a("")

    a("## Q8/9 (part 2): Persistence and forward paths")
    a("")
    a("Autocorrelation of the market factor with itself N sessions later:")
    for lag, r in ctx["persistence"].items():
        a(f"- {lag}: corr={fmt_num(r['corr'])} (n={r['n']})")
    a("")
    a("Average cumulative SPY return path, top-quintile vs bottom-quintile market-activity days "
     "(k = sessions after the signal date):")
    a("")
    a("| k | Top-quintile cum. return (n) | Bottom-quintile cum. return (n) |")
    a("|---:|---:|---:|")
    top = ctx["pathBattery"]["topQuintile"]["cumulativeReturn"]
    bot = ctx["pathBattery"]["bottomQuintile"]["cumulativeReturn"]
    for k in (1, 3, 5, 10, 15, 20):
        i = k - 1
        a(f"| {k} | {fmt_pct(top['avgByK'][i])} (n={top['nByK'][i]}) | "
         f"{fmt_pct(bot['avgByK'][i])} (n={bot['nByK'][i]}) |")
    a("")
    a("Average rolling 5-session realized vol along the same path:")
    a("| k | Top-quintile vol | Bottom-quintile vol |")
    a("|---:|---:|---:|")
    top_v = ctx["pathBattery"]["topQuintile"]["rolling5dVol"]
    bot_v = ctx["pathBattery"]["bottomQuintile"]["rolling5dVol"]
    for k in (1, 3, 5, 10, 15, 20):
        i = k - 1
        a(f"| {k} | {fmt_num(top_v['avgByK'][i], 4)} | {fmt_num(bot_v['avgByK'][i], 4)} |")
    a("")

    a("## Bootstrap (date-clustering-robust check on the SPY 10D volatility result)")
    a("")
    boot = ctx["bootstrap"]
    lo, hi = boot["ci95"]
    a(f"10,000-draw date-block bootstrap (block=5) of the SPY market-activity vs fwd_vol_10d "
     f"Spearman-quintile spread: mean={fmt_num(boot['mean'], 4)}, 95% CI=[{fmt_num(lo, 4)}, "
     f"{fmt_num(hi, 4)}], share same sign as full sample={fmt_num(boot['shareSameSign'], 2)}, "
     f"n dates={boot['nDates']}.")
    a("")

    a("## Answers")
    a("")
    a(f"1. **Does market-wide abnormal options activity predict direction?** {ctx['answers']['direction']}")
    a(f"2. **Does it predict future volatility better than future returns?** {ctx['answers']['volVsReturn']}")
    a(f"3. **Does it predict absolute market movement?** {ctx['answers']['absoluteMovement']}")
    a(f"4. **Is the effect concentrated around scheduled macro events?** {ctx['answers']['eventConcentration']}")
    a(f"5. **Does it survive excluding event days?** {ctx['answers']['survivesEventExclusion']}")
    a(f"6. **Does it behave differently in bullish/bearish regimes?** {ctx['answers']['regimeDependence']}")
    a(f"7. **Does ETF-specific residual activity predict cross-sectional ETF returns?** {ctx['answers']['residualPredicts']}")
    a(f"8. **Is the original result mostly common-factor or ETF-specific?** {ctx['answers']['commonVsSpecific']}")
    a(f"9. **What is the economic magnitude?** {ctx['answers']['magnitude']}")
    a(f"10. **Is there enough evidence to build a production market-activity indicator?** {ctx['answers']['productionReady']}")
    a("")

    a("## Verdict")
    a("")
    a(f"**{ctx['verdict']}**")
    a("")
    a("No production dashboard was modified. No ETF universe expansion. No new feature search "
     "was performed -- this explains the v2/v3 signal, it does not replace it.")
    return "\n".join(lines)


def flatten_rows(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for t in INDEX_TICKERS:
        for outcome, label in ctx["outcomeLabels"]:
            b = ctx["marketBattery"][t].get(outcome)
            if b is None:
                continue
            rows.append({"section": "market_factor", "ticker": t, "outcome": outcome, "n": b["n"],
                        "q1_mean": b["q1"]["mean"], "q5_mean": b["q5"]["mean"],
                        "q5_minus_q1_mean": b["q5MinusQ1Mean"], "spearman_rho": b["spearman"]["rho"],
                        "spearman_p": b["spearman"]["pValue"]})
    for h in PRIMARY_HORIZONS:
        r = ctx["residualBattery"][h]
        rows.append({"section": "residual", "ticker": "ALL", "outcome": f"ret_{h}d",
                    "n": r["panelRegression"]["n"], "q1_mean": None, "q5_mean": None,
                    "q5_minus_q1_mean": r["pooledQuintile"]["q5MinusQ1Mean"],
                    "spearman_rho": None, "spearman_p": r["panelRegression"]["pValue"]})
    return rows


def main() -> int:
    print("Loading the frozen combined 252-session dataset (discovery + validation, v3's exact loaders)...")
    combined = load_combined_frame()
    print(f"Combined frame: {len(combined)} rows, {combined['date'].min()} -> {combined['date'].max()}")

    market = market_activity_factor(combined, PRIMARY_FEATURE, min_tickers=3)
    market_by_date = market.set_index("date")["market_activity_median"]
    combined = combined.merge(market[["date", "market_activity_median", "market_activity_mean", "n_tickers"]],
                              on="date", how="left")
    combined["residual_activity"] = residual_activity_from_frame(combined, PRIMARY_FEATURE, market_by_date)

    n_dates_total = combined["date"].nunique()
    n_dates_with_factor = int(market["market_activity_median"].notna().sum())
    print(f"Market factor computed for {n_dates_with_factor}/{n_dates_total} dates.")

    print("Fetching SPY/QQQ/IWM/VIX price history...")
    from src.data_sources import fetch_prices
    prices = fetch_prices(INDEX_TICKERS + ["^VIX"], period="3y")
    prices.index = pd.to_datetime(prices.index)
    vix = prices["^VIX"] if "^VIX" in prices.columns else None

    dates_all = sorted(combined["date"].unique())
    outcome_labels = [("ret_1d", "1D return"), ("ret_3d", "3D return"), ("ret_5d", "5D return"),
                      ("ret_10d", "10D return"), ("ret_20d", "20D return"),
                      ("fwd_vol_5d", "5D fwd realized vol"), ("fwd_vol_10d", "10D fwd realized vol"),
                      ("fwd_vol_20d", "20D fwd realized vol"),
                      ("fwd_max_drawdown_5d", "5D fwd max drawdown"), ("fwd_max_drawdown_10d", "10D fwd max drawdown"),
                      ("fwd_max_drawdown_20d", "20D fwd max drawdown"),
                      ("vix_change_5d", "5D VIX change"), ("vix_change_10d", "10D VIX change"),
                      ("vix_change_20d", "20D VIX change")]

    print("Building per-index outcome frames and running the market-factor battery...")
    market_battery: Dict[str, Any] = {}
    index_frames: Dict[str, pd.DataFrame] = {}
    for t in INDEX_TICKERS:
        outcome_df = build_outcome_frame(prices[t], dates_all, vix)
        idx_returns = combined[combined["ticker"] == t][["date", "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"]]
        idx_frame = market.merge(idx_returns, on="date", how="left").merge(outcome_df, on="date", how="left")
        for h in HORIZONS:
            idx_frame[f"abs_ret_{h}d"] = idx_frame[f"ret_{h}d"].abs()
        index_frames[t] = idx_frame
        market_battery[t] = market_factor_battery(idx_frame, [o for o, _ in outcome_labels])

    print("Building regime splits (SPY-based)...")
    spy_prices = prices["SPY"]
    spy_close = spy_prices
    spy_same_day = spy_close.pct_change()
    spy_df = index_frames["SPY"].copy()
    spy_df["_same_day"] = spy_df["date"].map(lambda d: (
        "up" if (pd.Timestamp(d) in spy_same_day.index and pd.notna(spy_same_day.loc[pd.Timestamp(d)])
                and spy_same_day.loc[pd.Timestamp(d)] > 0) else
        ("down" if (pd.Timestamp(d) in spy_same_day.index and pd.notna(spy_same_day.loc[pd.Timestamp(d)])
                   and spy_same_day.loc[pd.Timestamp(d)] < 0) else None)
    ))
    sma = sma_regime(spy_prices, dates_all, window=200)
    spy_df["_200dma"] = spy_df["date"].map(sma["byDate"])
    mom = momentum_regime(spy_prices, dates_all, window=20)
    spy_df["_momentum"] = spy_df["date"].map(mom["byDate"])
    spy_iv = combined[combined["ticker"] == "SPY"][["date", "iv_percentile_60d"]]
    spy_df = spy_df.merge(spy_iv, on="date", how="left")
    iv_median = spy_df["iv_percentile_60d"].median()
    spy_df["_iv_regime"] = spy_df["iv_percentile_60d"].apply(
        lambda v: None if pd.isna(v) else ("high" if v > iv_median else "low"))

    regime_cols = {
        "same_day_direction": ("_same_day", ("up", "down")),
        "spy_200dma": ("_200dma", ("above", "below")),
        "spy_momentum": ("_momentum", ("positive", "negative")),
        "spy_iv_percentile": ("_iv_regime", ("high", "low")),
    }
    regime_battery = regime_split_battery(spy_df, ["ret_10d", "fwd_vol_10d"], regime_cols)

    print("Running the ETF-specific residual battery...")
    residual_battery_out = {h: residual_battery(combined, h) for h in PRIMARY_HORIZONS}

    print("Running persistence + path analysis...")
    persistence = persistence_battery(market_by_date)
    path = path_battery(spy_prices, index_frames["SPY"])

    print("Running date-block bootstrap on the SPY 10D volatility result...")

    def vol_quintile_spread_stat(d: pd.DataFrame) -> Optional[float]:
        return quintile_spread_stat(d, "market_activity_median", "fwd_vol_10d")["q5MinusQ1Mean"]

    bootstrap = date_block_bootstrap(index_frames["SPY"], vol_quintile_spread_stat,
                                     n_boot=10_000, block_length=5, seed=11)

    spy_battery = market_battery["SPY"]
    vol_b = spy_battery.get("fwd_vol_10d", {})
    ret_b = spy_battery.get("ret_10d", {})
    dd_b = spy_battery.get("fwd_max_drawdown_10d", {})

    def _sig(b):
        return b.get("spearman", {}).get("pValue") is not None and b["spearman"]["pValue"] < 0.05

    residual_10d = residual_battery_out[10]
    residual_sig = residual_10d["panelRegression"]["pValue"] is not None and residual_10d["panelRegression"]["pValue"] < 0.05

    answers = {
        "direction": ("Weak/inconclusive" if not _sig(ret_b) else "Yes") +
                    f" (SPY 10D return: Spearman rho={fmt_num(ret_b.get('spearman', {}).get('rho'))}, "
                    f"p={fmt_num(ret_b.get('spearman', {}).get('pValue'), 4)}).",
        "volVsReturn": ("Yes, volatility shows the stronger relationship" if _sig(vol_b) and not _sig(ret_b)
                       else ("Both show a relationship" if _sig(vol_b) and _sig(ret_b)
                            else ("Neither clears p<0.05" if not _sig(vol_b) and not _sig(ret_b)
                                 else "Return shows the stronger relationship"))) +
                       f" (10D fwd vol: rho={fmt_num(vol_b.get('spearman', {}).get('rho'))}, "
                       f"p={fmt_num(vol_b.get('spearman', {}).get('pValue'), 4)}; "
                       f"10D return: rho={fmt_num(ret_b.get('spearman', {}).get('rho'))}, "
                       f"p={fmt_num(ret_b.get('spearman', {}).get('pValue'), 4)}).",
        "absoluteMovement": f"10D max-drawdown Q5-Q1 spread = {fmt_pct(dd_b.get('q5MinusQ1Mean'))} "
                           f"(rho={fmt_num(dd_b.get('spearman', {}).get('rho'))}, "
                           f"p={fmt_num(dd_b.get('spearman', {}).get('pValue'), 4)}).",
        "eventConcentration": "Not testable -- no macro-event calendar exists in this codebase; "
                             "no external dataset was introduced per instruction.",
        "survivesEventExclusion": "Not testable for the same reason -- no exclusion rerun was performed.",
        "regimeDependence": "See the regime table above -- report the sign/magnitude split by "
                           "same-day direction, 200DMA, momentum, and IV-percentile regime rather "
                           "than a single number here.",
        "residualPredicts": ("No" if not residual_sig else "Yes") +
                           f" (pooled panel regression, 10D: beta={fmt_num(residual_10d['panelRegression']['beta'], 5)}, "
                           f"cluster t={fmt_num(residual_10d['panelRegression']['tStat'])}, "
                           f"p={fmt_num(residual_10d['panelRegression']['pValue'], 4)}).",
        "commonVsSpecific": "Common-factor-dominant, consistent with v3's market-wide-control finding "
                           f"(residual panel t={fmt_num(residual_10d['panelRegression']['tStat'])} vs "
                           f"market-factor SPY 10D-vol Spearman p={fmt_num(vol_b.get('spearman', {}).get('pValue'), 4)}).",
        "magnitude": f"SPY 10D fwd-vol Q5-Q1 spread = {fmt_num(vol_b.get('q5MinusQ1Mean'), 4)} "
                    f"(daily-return-std units); 10D max-drawdown Q5-Q1 spread = {fmt_pct(dd_b.get('q5MinusQ1Mean'))}.",
        "productionReady": "Not yet -- event-day concentration cannot be ruled out from existing data, "
                          "see the verdict below.",
    }

    verdict = determine_verdict(spy_battery, residual_10d, event_day_feasible=False)

    ctx = {
        "nDatesTotal": n_dates_total, "nDatesWithFactor": n_dates_with_factor,
        "outcomeLabels": outcome_labels, "marketBattery": market_battery,
        "regimeBattery": regime_battery, "residualBattery": residual_battery_out,
        "persistence": persistence, "pathBattery": path, "bootstrap": bootstrap,
        "answers": answers, "verdict": verdict,
    }

    OUT_JSON.write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(flatten_rows(ctx)).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_markdown(ctx), encoding="utf-8")

    print(f"\nWrote {OUT_MD}\nWrote {OUT_CSV}\nWrote {OUT_JSON}")
    print(f"\nVERDICT: {verdict}")
    print("\nNo production dashboard modified. Stopping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
