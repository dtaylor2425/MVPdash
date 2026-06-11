"""
api/routers/portfolio.py
GET /api/portfolio          — full portfolio data
GET /api/portfolio/summary  — lightweight for homepage card

Regime-Adaptive Allocation across 17 assets (matches /dashboard/signals):
  Equity Growth:     QQQ, SMH, XLK
  Equity Cyclical:   XLE, XLF, XLI, XLC
  Equity Defensive:  XLP, XLV, XLU
  Equity Core:       SPY, IWM
  Credit:            HYG
  Real Assets:       GLD, SLV
  Duration:          TLT
  Cash:              SHY
  Crypto:            BTC-USD

Weekly rebalance. Every shift timestamped with rationale.
"""

import time
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Portfolio"])

_cache = {}
_lock = threading.Lock()
_TTL = 30 * 60

# Full universe — matches signals page
ASSETS = ["SPY","QQQ","SMH","XLK","XLE","XLF","XLI","XLC",
          "XLP","XLV","XLU","IWM","HYG","GLD","SLV","TLT","SHY"]
BTC_TICKER = "BTC-USD"
ALL_TICKERS = ASSETS + [BTC_TICKER]
DISPLAY_MAP = {"BTC-USD": "BTC"}  # for display

BENCHMARK_60_40 = {"SPY": 0.60, "TLT": 0.40}

# Asset categories
EQ_GROWTH   = {"QQQ", "SMH", "XLK"}
EQ_CYCLICAL = {"XLE", "XLF", "XLI", "XLC"}
EQ_DEFENSIVE= {"XLP", "XLV", "XLU"}
EQ_CORE     = {"SPY", "IWM"}
CREDIT      = {"HYG"}
REAL_ASSETS = {"GLD", "SLV"}
DURATION    = {"TLT"}
CASH        = {"SHY"}
CRYPTO      = {BTC_TICKER}

ALL_RISK = EQ_GROWTH | EQ_CYCLICAL | EQ_CORE | CREDIT | CRYPTO
ALL_MODERATE = EQ_DEFENSIVE
ALL_DEFENSIVE = REAL_ASSETS | DURATION | CASH


def _zscore_at(series, idx, window=252):
    if idx < 30:
        return 0.0
    start = max(0, idx - window)
    chunk = series.iloc[start:idx + 1].dropna()
    if len(chunk) < 20:
        return 0.0
    val = float(chunk.iloc[-1])
    mean = float(chunk.mean())
    std = float(chunk.std())
    if std == 0:
        return 0.0
    return (val - mean) / std


def _compute_regime_at(macro, date_idx):
    score = 50.0
    signals = []

    def _get_col(name):
        if name not in macro.columns:
            return None
        s = macro[name].iloc[:date_idx + 1].dropna()
        return s if len(s) > 20 else None

    hy = _get_col("hy_oas")
    if hy is not None:
        z = _zscore_at(hy, len(hy) - 1)
        score += max(-15, min(15, -z * 8))
        if z < -0.5:
            signals.append("credit_tight")
        elif z > 0.5:
            signals.append("credit_wide")

    r10 = _get_col("real10")
    r10_val = None
    if r10 is not None:
        r10_val = float(r10.iloc[-1])
        if r10_val < 0.5:
            score += 8
            signals.append("real_yields_low")
        elif r10_val < 1.5:
            score += 3
        elif r10_val > 2.0:
            score -= 8
            signals.append("real_yields_high")
        else:
            score -= 3

    y10 = _get_col("y10")
    y2 = _get_col("y2")
    if y10 is not None and y2 is not None:
        cv = float(y10.iloc[-1]) - float(y2.iloc[-1])
        if cv > 0.5:
            score += 6; signals.append("curve_positive")
        elif cv > 0:
            score += 2
        elif cv > -0.5:
            score -= 4; signals.append("curve_flat")
        else:
            score -= 8; signals.append("curve_inverted")

    dol = _get_col("dollar_broad")
    dol_z = 0.0
    if dol is not None:
        dol_z = _zscore_at(dol, len(dol) - 1)
        score += max(-10, min(10, -dol_z * 5))
        if dol_z < -0.5:
            signals.append("dollar_weak")
        elif dol_z > 0.5:
            signals.append("dollar_strong")

    cl = _get_col("init_claims")
    if cl is not None:
        z = _zscore_at(cl, len(cl) - 1, window=52)
        score += max(-10, min(10, -z * 6))
        if z > 0.5:
            signals.append("claims_rising")
        elif z < -0.3:
            signals.append("claims_healthy")

    # Inflation signal from breakevens
    if y10 is not None and r10 is not None:
        be = float(y10.iloc[-1]) - float(r10.iloc[-1])
        if be > 2.6:
            signals.append("inflation_hot")
        elif be < 2.0:
            signals.append("inflation_cool")

    hy_z = _zscore_at(hy, len(hy) - 1) if hy is not None else 0.0

    return max(5, min(95, int(score))), signals, r10_val, hy_z, dol_z


def _compute_weights(regime_score, signals, r10_val, hy_z, dol_z):
    """
    Full 17-asset allocation.
    
    Step 1: Regime → bucket allocations (risk / moderate / defensive)
    Step 2: Signal tilts → within-bucket weights
    """
    # Step 1: Bucket allocations
    if regime_score >= 75:
        risk_pct, mod_pct, def_pct = 0.55, 0.15, 0.30
    elif regime_score >= 60:
        risk_pct, mod_pct, def_pct = 0.45, 0.18, 0.37
    elif regime_score >= 45:
        risk_pct, mod_pct, def_pct = 0.35, 0.20, 0.45
    elif regime_score >= 30:
        risk_pct, mod_pct, def_pct = 0.20, 0.22, 0.58
    else:
        risk_pct, mod_pct, def_pct = 0.08, 0.15, 0.77

    # Step 2: Within-bucket weights

    # ── RISK BUCKET (QQQ, SMH, XLK, XLE, XLF, XLI, XLC, SPY, IWM, HYG, BTC) ──
    rw = {
        "SPY": 0.20, "QQQ": 0.12, "SMH": 0.10, "XLK": 0.08,
        "XLE": 0.08, "XLF": 0.07, "XLI": 0.06, "XLC": 0.05,
        "IWM": 0.08, "HYG": 0.08, BTC_TICKER: 0.08,
    }

    # Growth/momentum tilt when conditions support it
    if "credit_tight" in signals and "claims_healthy" in signals:
        rw["SMH"] += 0.06; rw["QQQ"] += 0.04; rw["XLK"] += 0.03
        rw["IWM"] += 0.02
        rw["SPY"] -= 0.06; rw["XLE"] -= 0.04; rw[BTC_TICKER] -= 0.02
        rw["XLI"] -= 0.03

    # Stress → underweight high-beta, overweight quality
    if "credit_wide" in signals:
        rw["SMH"] -= 0.05; rw["IWM"] -= 0.04; rw[BTC_TICKER] -= 0.04
        rw["XLF"] -= 0.02
        rw["SPY"] += 0.08; rw["HYG"] -= 0.03; rw["QQQ"] += 0.05
        rw["XLK"] += 0.02; rw["XLC"] += 0.03

    if "claims_rising" in signals:
        rw["IWM"] -= 0.04; rw["XLF"] -= 0.03; rw["XLC"] -= 0.02
        rw["SPY"] += 0.05; rw["QQQ"] += 0.02; rw["HYG"] += 0.02

    # Inflation → energy benefits
    if "inflation_hot" in signals:
        rw["XLE"] += 0.05; rw[BTC_TICKER] += 0.03
        rw["XLF"] += 0.02
        rw["QQQ"] -= 0.04; rw["XLK"] -= 0.03; rw["XLC"] -= 0.03

    # Dollar weak → commodities, international earners
    if "dollar_weak" in signals:
        rw["XLE"] += 0.03; rw[BTC_TICKER] += 0.02
        rw["SPY"] -= 0.03; rw["XLC"] -= 0.02

    # Floor at 0, normalize
    for k in rw:
        rw[k] = max(0.01, rw[k])
    total = sum(rw.values())
    for k in rw:
        rw[k] /= total

    # ── MODERATE BUCKET (XLP, XLV, XLU) ──
    mw = {"XLP": 0.35, "XLV": 0.35, "XLU": 0.30}

    if "real_yields_high" in signals:
        mw["XLU"] += 0.10; mw["XLP"] -= 0.05; mw["XLV"] -= 0.05

    if "claims_rising" in signals:
        mw["XLP"] += 0.08; mw["XLV"] += 0.07; mw["XLU"] -= 0.05
        # Defensive quality in stress

    total = sum(mw.values())
    for k in mw:
        mw[k] /= total

    # ── DEFENSIVE BUCKET (GLD, SLV, TLT, SHY) ──
    dw = {"GLD": 0.30, "SLV": 0.12, "TLT": 0.28, "SHY": 0.30}

    if r10_val is not None and r10_val < 1.0:
        dw["GLD"] += 0.12; dw["SLV"] += 0.10
        dw["TLT"] -= 0.08; dw["SHY"] -= 0.14
    elif r10_val is not None and r10_val > 2.0:
        dw["SHY"] += 0.15; dw["TLT"] -= 0.08
        dw["SLV"] -= 0.04; dw["GLD"] -= 0.03

    if "dollar_weak" in signals:
        dw["GLD"] += 0.08; dw["SLV"] += 0.07
        dw["SHY"] -= 0.10; dw["TLT"] -= 0.05

    if "dollar_strong" in signals:
        dw["SHY"] += 0.10
        dw["GLD"] -= 0.05; dw["SLV"] -= 0.05

    if "inflation_hot" in signals:
        dw["GLD"] += 0.05; dw["SLV"] += 0.05
        dw["TLT"] -= 0.08; dw["SHY"] -= 0.02

    if "inflation_cool" in signals:
        dw["TLT"] += 0.08; dw["SHY"] += 0.04
        dw["GLD"] -= 0.06; dw["SLV"] -= 0.06

    for k in dw:
        dw[k] = max(0.02, dw[k])
    total = sum(dw.values())
    for k in dw:
        dw[k] /= total

    # Combine
    weights = {}
    for k, v in rw.items():
        weights[k] = round(risk_pct * v, 4)
    for k, v in mw.items():
        weights[k] = round(mod_pct * v, 4)
    for k, v in dw.items():
        weights[k] = round(def_pct * v, 4)

    return weights


def _rationale(regime_score, signals, old_score):
    parts = []
    if regime_score >= 65:
        label = "Risk On"
    elif regime_score >= 55:
        label = "Bullish"
    elif regime_score >= 45:
        label = "Neutral"
    elif regime_score >= 35:
        label = "Bearish"
    else:
        label = "Risk Off"
    delta = regime_score - old_score if old_score else 0
    if abs(delta) > 5:
        parts.append("{} ({}{})".format(label, "+" if delta > 0 else "", delta))
    else:
        parts.append(label)
    tag_map = {
        "credit_tight": "credit tight",
        "credit_wide": "spreads widening",
        "real_yields_low": "real yields supportive",
        "real_yields_high": "real yields restrictive",
        "dollar_weak": "weak dollar",
        "dollar_strong": "strong dollar",
        "claims_rising": "claims rising",
        "claims_healthy": "labor healthy",
        "curve_inverted": "curve inverted",
        "curve_positive": "curve positive",
        "inflation_hot": "inflation elevated",
        "inflation_cool": "inflation contained",
    }
    for s in signals[:3]:
        if s in tag_map:
            parts.append(tag_map[s])
    return " \u00B7 ".join(parts)


def _run_simulation(macro, prices):
    common_dates = prices.index.sort_values()
    if len(common_dates) < 60:
        return None

    returns = prices.pct_change().fillna(0)

    # Weekly dates
    weekly_dates = []
    current_week = None
    prev_d = common_dates[0]
    for d in common_dates:
        wk = (d.year, d.isocalendar()[1])
        if wk != current_week:
            if current_week is not None:
                weekly_dates.append(prev_d)
            current_week = wk
        prev_d = d
    weekly_dates.append(common_dates[-1])
    rebal_set = set(weekly_dates)

    all_assets = list(set(ASSETS + [BTC_TICKER]) & set(prices.columns))

    # Init equal weight
    current_weights = {}
    eq_w = 1.0 / len(all_assets) if all_assets else 0
    for a in all_assets:
        current_weights[a] = eq_w

    portfolio_value = [1.0]
    spy_value = [1.0]
    bench_value = [1.0]
    rebalances = []
    prev_regime = 50

    for i in range(1, len(common_dates)):
        date = common_dates[i]

        # Portfolio return
        port_ret = 0.0
        for a in all_assets:
            if a in returns.columns:
                r = float(returns[a].iloc[i])
                if np.isfinite(r):
                    port_ret += current_weights.get(a, 0) * r

        spy_ret = float(returns["SPY"].iloc[i]) if "SPY" in returns.columns else 0
        if not np.isfinite(spy_ret):
            spy_ret = 0

        bench_ret = 0.0
        for a, w in BENCHMARK_60_40.items():
            if a in returns.columns:
                r = float(returns[a].iloc[i])
                if np.isfinite(r):
                    bench_ret += w * r

        portfolio_value.append(portfolio_value[-1] * (1 + port_ret))
        spy_value.append(spy_value[-1] * (1 + spy_ret))
        bench_value.append(bench_value[-1] * (1 + bench_ret))

        # Rebalance
        if date in rebal_set:
            macro_mask = macro.index <= date
            if macro_mask.any():
                midx = macro_mask.sum() - 1
                regime_score, signals, r10_val, hy_z, dol_z = _compute_regime_at(macro, midx)
                new_weights = _compute_weights(regime_score, signals, r10_val, hy_z, dol_z)

                # Only keep weights for assets we have prices for
                final_weights = {}
                for a in all_assets:
                    final_weights[a] = new_weights.get(a, 0)
                # Renormalize
                tw = sum(final_weights.values())
                if tw > 0:
                    for a in final_weights:
                        final_weights[a] /= tw

                weight_change = sum(abs(final_weights.get(a, 0) - current_weights.get(a, 0)) for a in all_assets)
                if weight_change > 0.03:
                    # Display-friendly weights
                    display_w = {}
                    for k, v in final_weights.items():
                        dk = DISPLAY_MAP.get(k, k)
                        display_w[dk] = round(v, 3)
                    rebalances.append({
                        "date": str(date.date()),
                        "regime_score": regime_score,
                        "rationale": _rationale(regime_score, signals, prev_regime),
                        "weights": display_w,
                        "portfolio_value": round(portfolio_value[-1], 4),
                    })

                current_weights = final_weights
                prev_regime = regime_score

    # Equity curve (sampled)
    step = max(1, len(common_dates) // 500)
    curve = []
    for i in range(0, len(common_dates), step):
        curve.append({
            "date": str(common_dates[i].date()),
            "portfolio": round(portfolio_value[i], 4),
            "spy": round(spy_value[i], 4),
            "bench_60_40": round(bench_value[i], 4),
        })
    curve.append({
        "date": str(common_dates[-1].date()),
        "portfolio": round(portfolio_value[-1], 4),
        "spy": round(spy_value[-1], 4),
        "bench_60_40": round(bench_value[-1], 4),
    })

    def _pr(vals, days):
        if len(vals) <= days: return None
        return round((vals[-1] / vals[-days - 1] - 1) * 100, 2)

    def _ytd(vals, dates):
        for i, d in enumerate(dates):
            if d.year == dates[-1].year:
                return round((vals[-1] / vals[i] - 1) * 100, 2) if vals[i] != 0 else None
        return None

    def _mdd(vals):
        peak = vals[0]
        dd = 0
        for v in vals:
            if v > peak: peak = v
            d = (v - peak) / peak if peak > 0 else 0
            if d < dd: dd = d
        return round(dd * 100, 2)

    # Display-friendly current weights
    display_cw = {}
    for k, v in current_weights.items():
        dk = DISPLAY_MAP.get(k, k)
        display_cw[dk] = round(v, 3)

    stats = {
        "current_value": round(portfolio_value[-1], 4),
        "total_return": round((portfolio_value[-1] - 1) * 100, 2),
        "spy_total_return": round((spy_value[-1] - 1) * 100, 2),
        "bench_total_return": round((bench_value[-1] - 1) * 100, 2),
        "returns": {
            "1m": _pr(portfolio_value, 21), "3m": _pr(portfolio_value, 63),
            "6m": _pr(portfolio_value, 126), "ytd": _ytd(portfolio_value, common_dates),
            "1y": _pr(portfolio_value, 252), "2y": _pr(portfolio_value, 504),
        },
        "spy_returns": {
            "1m": _pr(spy_value, 21), "3m": _pr(spy_value, 63),
            "6m": _pr(spy_value, 126), "ytd": _ytd(spy_value, common_dates),
            "1y": _pr(spy_value, 252), "2y": _pr(spy_value, 504),
        },
        "max_drawdown": _mdd(portfolio_value),
        "spy_max_drawdown": _mdd(spy_value),
        "rebalance_count": len(rebalances),
        "current_weights": display_cw,
        "current_regime": prev_regime,
        "asset_count": len(all_assets),
    }

    return {
        "curve": curve,
        "stats": stats,
        "rebalances": rebalances[-20:],
        "last_rebalance": rebalances[-1] if rebalances else None,
    }


def _fetch_and_compute():
    from api.deps import get_macro
    macro = get_macro()

    try:
        df = yf.download(tickers=ALL_TICKERS, period="2y", auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        print("portfolio price fetch error: {}".format(e))
        return None

    if df is None or df.empty:
        return None

    prices = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in ALL_TICKERS:
            try:
                if df.columns.get_level_values(0)[0] in {"Close", "Open", "High", "Low", "Volume"}:
                    s = df.xs(t, axis=1, level=1)
                    if isinstance(s, pd.DataFrame) and "Close" in s.columns:
                        prices[t] = s["Close"].dropna()
                else:
                    sub = df[t]
                    if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                        prices[t] = sub["Close"].dropna()
                    elif isinstance(sub, pd.Series):
                        prices[t] = sub.dropna()
            except Exception:
                pass
    else:
        if "Close" in df.columns and len(ALL_TICKERS) == 1:
            prices[ALL_TICKERS[0]] = df["Close"].dropna()

    if len(prices) < 6:
        return None

    px = pd.DataFrame(prices).dropna(how="all").ffill()
    return _run_simulation(macro, px)


@router.get("/portfolio")
def portfolio_full():
    with _lock:
        if "data" in _cache and (time.time() - _cache.get("ts", 0)) < _TTL:
            return _cache["data"]
    result = _fetch_and_compute()
    if result is None:
        raise HTTPException(status_code=503, detail="Portfolio computation failed")
    with _lock:
        _cache["data"] = result
        _cache["ts"] = time.time()
    return result


@router.get("/portfolio/summary")
def portfolio_summary():
    with _lock:
        if "data" in _cache and (time.time() - _cache.get("ts", 0)) < _TTL:
            data = _cache["data"]
        else:
            data = None
    if data is None:
        data = _fetch_and_compute()
        if data is None:
            raise HTTPException(status_code=503, detail="Portfolio computation failed")
        with _lock:
            _cache["data"] = data
            _cache["ts"] = time.time()
    stats = data.get("stats", {})
    last_reb = data.get("last_rebalance")
    curve = data.get("curve", [])
    return {
        "total_return": stats.get("total_return"),
        "spy_total_return": stats.get("spy_total_return"),
        "bench_total_return": stats.get("bench_total_return"),
        "returns": stats.get("returns", {}),
        "spy_returns": stats.get("spy_returns", {}),
        "max_drawdown": stats.get("max_drawdown"),
        "current_regime": stats.get("current_regime"),
        "current_weights": stats.get("current_weights", {}),
        "asset_count": stats.get("asset_count", 0),
        "last_rebalance": last_reb,
        "curve": curve[-90:] if len(curve) > 90 else curve,
    }