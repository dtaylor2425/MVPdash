"""
api/routers/portfolio.py
GET /api/portfolio — full model portfolio data (both variants)

Macro Momentum Rotation:
  Blended momentum (3m/6m/12m), top N=3 gated vs BIL,
  crash filter (SPY < 200dma), vol-targeted variant.
  Monthly rebalance, 10bps cost.
"""

import time, threading, json
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Portfolio"])

_cache = {}
_lock = threading.Lock()
_TTL = 12 * 60 * 60  # 12 hours

UNIVERSE = ["XLE","SMH","XLP","HYG","XLI","SPY","QQQ","IWM","XLK",
            "TLT","XLV","XLC","BTC-USD","SLV","GLD","XLF","XLU"]
CASH     = "BIL"
DEFENSE  = "TLT"
BENCH    = "SPY"
TOP_N    = 3
COST_BPS = 10
LOOKBACKS = [63, 126, 252]
GATE_LB  = 126
VOL_LB   = 63
VOL_TGT  = 0.15
START    = "2022-06-01"
LIVE_DATE = "2026-06-10"  # backtest/live boundary


def _run_backtest():
    tickers = sorted(set(UNIVERSE + [CASH, BENCH]))
    try:
        raw = yf.download(tickers, start=START, auto_adjust=True, progress=False)
    except Exception as e:
        print("portfolio download error: {}".format(e))
        return None

    if raw is None or raw.empty:
        return None

    # Extract closes
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            px = raw["Close"]
        except KeyError:
            return None
    else:
        px = raw

    px = px.dropna(how="all").ffill()
    if BENCH not in px.columns:
        return None
    px = px.reindex(px[BENCH].dropna().index).ffill()
    daily_ret = px.pct_change().fillna(0.0)
    sma200 = px[BENCH].rolling(200).mean()

    def score_assets(asof):
        p = px[UNIVERSE].loc[:asof]
        moms = []
        for lb in LOOKBACKS:
            if len(p) > lb:
                moms.append(p.iloc[-1] / p.iloc[-1 - lb] - 1)
        if not moms:
            return pd.Series(dtype=float)
        return pd.concat(moms, axis=1).mean(axis=1)

    def gate_returns(asof):
        p = px.loc[:asof]
        if len(p) > GATE_LB:
            return p.iloc[-1] / p.iloc[-1 - GATE_LB] - 1
        return pd.Series(dtype=float)

    month_ends_raw = px.resample("ME").last().index
    month_ends = []
    for d in month_ends_raw:
        if d >= px.index[min(260, len(px.index) - 1)]:
            idx = px.index.get_indexer([d], method="ffill")[0]
            month_ends.append(px.index[idx])
    month_ends = sorted(set(month_ends))

    if len(month_ends) < 2:
        return None

    def build_weights(sig_date, mode):
        mom = score_assets(sig_date).dropna().sort_values(ascending=False)
        gate = gate_returns(sig_date)
        bil_g = float(gate.get(CASH, 0.0)) if CASH in gate.index else 0.0

        picks = [t for t in mom.index if float(gate.get(t, -1)) > bil_g][:TOP_N]
        n_fail = TOP_N - len(picks)
        w = {}

        if mode == "equal" or len(picks) == 0:
            for t in picks:
                w[t] = w.get(t, 0) + 1.0 / TOP_N
        else:
            vols = daily_ret[picks].loc[:sig_date].tail(VOL_LB).std() * np.sqrt(252)
            vols = vols.replace(0, np.nan).fillna(vols.max() if not vols.empty else 1.0)
            iv = (1.0 / vols)
            iv_sum = iv.sum()
            if iv_sum > 0:
                iv = iv / iv_sum
            sleeve = len(picks) / TOP_N
            for t in picks:
                w[t] = w.get(t, 0) + float(iv[t]) * sleeve

        if n_fail:
            fb = DEFENSE if float(gate.get(DEFENSE, -1)) > bil_g else CASH
            w[fb] = w.get(fb, 0) + n_fail / TOP_N

        # crash filter
        spy_px = px[BENCH].loc[sig_date]
        sma_px = sma200.loc[sig_date]
        if isinstance(spy_px, (float, np.floating)) and isinstance(sma_px, (float, np.floating)):
            if spy_px < sma_px:
                risky = {t: v for t, v in w.items() if t != CASH}
                rs = sum(risky.values())
                if rs > 0.5:
                    w = {t: v * 0.5 / rs for t, v in risky.items()}
                    w[CASH] = w.get(CASH, 0) + (1 - sum(w.values()))

        # vol target (variant B)
        if mode == "vol_targeted":
            cols = [c for c in px.columns if c in w]
            wv = pd.Series(0.0, index=px.columns)
            for t, v in w.items():
                if t in wv.index:
                    wv[t] = v
            cov = daily_ret.loc[:sig_date].tail(VOL_LB).cov() * 252
            try:
                pvol = float(np.sqrt(wv @ cov @ wv))
            except Exception:
                pvol = VOL_TGT
            if pvol > VOL_TGT and pvol > 0:
                scale = VOL_TGT / pvol
                w = {t: v * scale for t, v in w.items() if t != CASH}
                w[CASH] = max(0, 1.0 - sum(w.values()))

        return pd.Series(w, index=px.columns).fillna(0.0)

    def run_variant(mode):
        prev_w = pd.Series(0.0, index=px.columns)
        strat = pd.Series(0.0, index=px.index, dtype=float)
        wlog = []
        for i, sig in enumerate(month_ends):
            w = build_weights(sig, mode)
            loc = px.index.get_loc(sig)
            if loc + 1 >= len(px.index):
                wlog.append((sig, w))
                break
            start_d = px.index[loc + 1]
            end_d = month_ends[i + 1] if i + 1 < len(month_ends) else px.index[-1]
            cost = float((w - prev_w).abs().sum()) * COST_BPS / 1e4
            pr = (daily_ret.loc[start_d:end_d] * w).sum(axis=1)
            if len(pr):
                pr.iloc[0] = pr.iloc[0] - cost
            strat.loc[pr.index] = pr
            wlog.append((sig, w))
            prev_w = w
        return strat, wlog

    def compute_stats(r):
        r = r.dropna()
        r = r[r.index >= month_ends[0]]
        if len(r) == 0:
            return {}
        eq = (1 + r).cumprod()
        yrs = len(r) / 252
        vol = float(r.std() * np.sqrt(252))
        return {
            "total_return": round(float(eq.iloc[-1] - 1) * 100, 2),
            "cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else None,
            "vol": round(vol * 100, 2),
            "sharpe": round(float(r.mean() * 252 / vol), 2) if vol > 0 else None,
            "max_dd": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
        }

    def window_stats(r):
        last = r.index[-1]
        return {
            "inception": compute_stats(r),
            "2y": compute_stats(r[r.index >= last - pd.DateOffset(years=2)]),
            "ytd": compute_stats(r[r.index >= pd.Timestamp("{}-01-01".format(last.year))]),
        }

    def series_json(r):
        r = r[r.index >= month_ends[0]]
        eq = (1 + r).cumprod() * 100.0
        dd = eq / eq.cummax() - 1
        # Sample to ~500 points max
        step = max(1, len(eq) // 500)
        points = []
        for i in range(0, len(eq), step):
            points.append({
                "date": eq.index[i].strftime("%Y-%m-%d"),
                "equity": round(float(eq.iloc[i]), 2),
                "drawdown": round(float(dd.iloc[i]), 4),
            })
        if len(eq) > 0:
            points.append({
                "date": eq.index[-1].strftime("%Y-%m-%d"),
                "equity": round(float(eq.iloc[-1]), 2),
                "drawdown": round(float(dd.iloc[-1]), 4),
            })
        return points

    bench_r = daily_ret[BENCH]

    payload = {
        "as_of": px.index[-1].strftime("%Y-%m-%d"),
        "live_date": LIVE_DATE,
        "benchmark": BENCH,
        "variants": {},
    }

    for mode in ["equal", "vol_targeted"]:
        r, wlog = run_variant(mode)
        name = "momentum" if mode == "equal" else "vol_targeted"

        # Build rebalance log with monthly returns
        rebalances = []
        for i, (d, w) in enumerate(wlog):
            weights_dict = {t: round(float(v), 4) for t, v in w.items() if v > 1e-4}
            # Display-friendly names
            display_w = {}
            for t, v in weights_dict.items():
                dt = t.replace("-USD", "") if "-USD" in t else t
                display_w[dt] = v

            reb = {"date": d.strftime("%Y-%m-%d"), "weights": display_w}

            # Monthly return for this period
            if i > 0:
                prev_d = wlog[i - 1][0]
                loc_s = px.index.get_loc(prev_d)
                loc_e = px.index.get_loc(d)
                if loc_s + 1 < loc_e:
                    pr = (daily_ret.iloc[loc_s + 1:loc_e + 1] * wlog[i - 1][1]).sum(axis=1).sum()
                    sr = float(daily_ret[BENCH].iloc[loc_s + 1:loc_e + 1].sum())
                    reb["model_return"] = round(float(pr) * 100, 2)
                    reb["spy_return"] = round(sr * 100, 2)

            rebalances.append(reb)

        # Current allocation display-friendly
        current_w = {}
        if wlog:
            for t, v in wlog[-1][1].items():
                if v > 1e-4:
                    dt = t.replace("-USD", "") if "-USD" in t else t
                    current_w[dt] = round(float(v), 4)

        payload["variants"][name] = {
            "stats": window_stats(r),
            "series": series_json(r),
            "rebalances": rebalances,
            "current_allocation": current_w,
        }

    # Benchmark
    payload["benchmark_stats"] = window_stats(bench_r)
    payload["benchmark_series"] = series_json(bench_r)

    # Allocation history for stacked area
    # Sample monthly weights over time for vol_targeted variant
    _, vt_wlog = run_variant("vol_targeted")
    alloc_history = []
    for d, w in vt_wlog:
        entry = {"date": d.strftime("%Y-%m-%d")}
        for t, v in w.items():
            if v > 1e-4:
                dt = t.replace("-USD", "") if "-USD" in t else t
                entry[dt] = round(float(v), 4)
        alloc_history.append(entry)
    payload["allocation_history"] = alloc_history

    return payload


@router.get("/portfolio")
def portfolio_full():
    with _lock:
        if "data" in _cache and (time.time() - _cache.get("ts", 0)) < _TTL:
            return _cache["data"]

    result = _run_backtest()
    if result is None:
        raise HTTPException(status_code=503, detail="Portfolio computation failed — price data unavailable")

    with _lock:
        _cache["data"] = result
        _cache["ts"] = time.time()

    return result