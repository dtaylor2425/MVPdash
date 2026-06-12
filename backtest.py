"""
Macro Momentum Rotation v2 — equal-weight + vol-targeted variants
Outputs portfolio_data.json for the Macro Engine frontend.

Run: pip install yfinance pandas numpy && python backtest_v2.py
Wire into FastAPI: serve portfolio_data.json from a daily cron (see bottom).

Strategy:
  1. Blended momentum = mean(3m, 6m, 12m return). Production hook: swap in
     Macro Engine composite scores at score_assets().
  2. Top N=3, gated: 6m return must beat BIL's. Failed slots -> TLT if it
     passes the gate, else BIL.
  3. Crash filter: SPY < 200dma at rebalance -> risky sleeve capped at 50%.
  4. Variant A "momentum": equal weight slots.
     Variant B "vol_targeted": inverse-vol weights + 15% ann. portfolio
     vol target (excess scaled to BIL).
  5. Monthly rebalance, signal at month-end close, executed next close.
     10 bps per side on turnover.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- config ----------------
UNIVERSE  = ["XLE","SMH","XLP","HYG","XLI","SPY","QQQ","IWM","XLK",
             "TLT","XLV","XLC","BTC-USD","SLV","GLD","XLF","XLU"]
CASH      = "BIL"
DEFENSE   = "TLT"
BENCH     = "SPY"
TOP_N     = 3
COST_BPS  = 10
LOOKBACKS = [63, 126, 252]
GATE_LB   = 126
VOL_LB    = 63
VOL_TGT   = 0.15
START     = "2022-06-01"
OUT_FILE  = "portfolio_data.json"

# ---------------- data ----------------
tickers = sorted(set(UNIVERSE + [CASH, BENCH]))
px = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
px = px.dropna(how="all").ffill()
px = px.reindex(px[BENCH].dropna().index).ffill()
daily_ret = px.pct_change().fillna(0.0)
sma200 = px[BENCH].rolling(200).mean()

def score_assets(asof: pd.Timestamp) -> pd.Series:
    """Blended price momentum. PRODUCTION HOOK: return your Macro Engine
    composite scores here instead (same index = tickers, higher = better)."""
    p = px[UNIVERSE].loc[:asof]
    moms = [p.iloc[-1] / p.iloc[-1 - lb] - 1 for lb in LOOKBACKS if len(p) > lb]
    return pd.concat(moms, axis=1).mean(axis=1)

def gate_returns(asof: pd.Timestamp) -> pd.Series:
    p = px.loc[:asof]
    return p.iloc[-1] / p.iloc[-1 - GATE_LB] - 1 if len(p) > GATE_LB else pd.Series(dtype=float)

month_ends_raw = px.resample("ME").last().index
month_ends = [px.index[px.index.get_indexer([d], method="ffill")[0]]
              for d in month_ends_raw if d >= px.index[260]]
month_ends = sorted(set(month_ends))

# ---------------- weight construction ----------------
def build_weights(sig_date: pd.Timestamp, mode: str) -> pd.Series:
    mom  = score_assets(sig_date).dropna().sort_values(ascending=False)
    gate = gate_returns(sig_date)
    bil_g = gate.get(CASH, 0.0)

    picks = [t for t in mom.index if gate.get(t, -1) > bil_g][:TOP_N]
    n_fail = TOP_N - len(picks)
    w = {}

    if mode == "equal" or len(picks) == 0:
        for t in picks:
            w[t] = w.get(t, 0) + 1.0 / TOP_N
    else:  # inverse-vol among picks
        vols = daily_ret[picks].loc[:sig_date].tail(VOL_LB).std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).fillna(vols.max() or 1.0)
        iv = (1.0 / vols) / (1.0 / vols).sum()
        sleeve = len(picks) / TOP_N
        for t in picks:
            w[t] = w.get(t, 0) + float(iv[t]) * sleeve

    if n_fail:
        fb = DEFENSE if gate.get(DEFENSE, -1) > bil_g else CASH
        w[fb] = w.get(fb, 0) + n_fail / TOP_N

    # crash filter
    if px[BENCH].loc[sig_date] < sma200.loc[sig_date]:
        risky = {t: v for t, v in w.items() if t != CASH}
        rs = sum(risky.values())
        if rs > 0.5:
            w = {t: v * 0.5 / rs for t, v in risky.items()}
            w[CASH] = w.get(CASH, 0) + (1 - sum(w.values()))

    # vol target (variant B only)
    if mode == "vol_targeted":
        wv = pd.Series(w, index=px.columns).fillna(0.0)
        cov = daily_ret.loc[:sig_date].tail(VOL_LB).cov() * 252
        pvol = float(np.sqrt(wv @ cov @ wv))
        if pvol > VOL_TGT:
            scale = VOL_TGT / pvol
            w = {t: v * scale for t, v in w.items() if t != CASH}
            w[CASH] = 1.0 - sum(w.values())

    return pd.Series(w, index=px.columns).fillna(0.0)

# ---------------- backtest engine ----------------
def run(mode: str):
    prev_w = pd.Series(0.0, index=px.columns)
    strat = pd.Series(0.0, index=px.index)
    wlog = []
    for i, sig in enumerate(month_ends):
        w = build_weights(sig, mode)
        loc = px.index.get_loc(sig)
        if loc + 1 >= len(px.index):
            wlog.append((sig, w))  # current target weights, not yet executed
            break
        start = px.index[loc + 1]
        end = month_ends[i + 1] if i + 1 < len(month_ends) else px.index[-1]
        cost = (w - prev_w).abs().sum() * COST_BPS / 1e4
        pr = (daily_ret.loc[start:end] * w).sum(axis=1)
        if len(pr):
            pr.iloc[0] -= cost
        strat.loc[pr.index] = pr
        wlog.append((sig, w))
        prev_w = w
    return strat, wlog

def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {
        "total_return": round(float(eq.iloc[-1] - 1), 4),
        "cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4) if yrs > 0 else None,
        "vol": round(float(vol), 4),
        "sharpe": round(float(r.mean() * 252 / vol), 2) if vol > 0 else None,
        "max_dd": round(float((eq / eq.cummax() - 1).min()), 4),
    }

def windows(r: pd.Series) -> dict:
    last = r.index[-1]
    return {
        "inception": stats(r[r.index >= month_ends[0]]),
        "2y":  stats(r[r.index >= last - pd.DateOffset(years=2)]),
        "ytd": stats(r[r.index >= pd.Timestamp(f"{last.year}-01-01")]),
    }

def series_json(r: pd.Series, start: pd.Timestamp) -> list:
    r = r[r.index >= start]
    eq = (1 + r).cumprod() * 100.0
    dd = eq / eq.cummax() - 1
    return [{"date": d.strftime("%Y-%m-%d"),
             "equity": round(float(e), 2),
             "drawdown": round(float(x), 4)}
            for d, e, x in zip(eq.index, eq.values, dd.values)]

# ---------------- run + export ----------------
payload = {"as_of": px.index[-1].strftime("%Y-%m-%d"),
           "benchmark": BENCH, "variants": {}}

bench_r = daily_ret[BENCH]
start_plot = month_ends[0]

for mode in ["equal", "vol_targeted"]:
    r, wlog = run(mode)
    name = "momentum" if mode == "equal" else "vol_targeted"
    payload["variants"][name] = {
        "stats": windows(r),
        "series": series_json(r, start_plot),
        "rebalances": [{"date": d.strftime("%Y-%m-%d"),
                        "weights": {t: round(float(v), 4)
                                    for t, v in w.items() if v > 1e-4}}
                       for d, w in wlog],
        "current_allocation": {t: round(float(v), 4)
                               for t, v in wlog[-1][1].items() if v > 1e-4},
    }

payload["benchmark_stats"] = windows(bench_r)
payload["benchmark_series"] = series_json(bench_r, start_plot)

with open(OUT_FILE, "w") as f:
    json.dump(payload, f)
print(f"wrote {OUT_FILE}")

# console summary
for name, v in payload["variants"].items():
    print(f"\n{name}")
    for wname, s in v["stats"].items():
        print(f"  {wname:9s} {s}")
print(f"\nSPY")
for wname, s in payload["benchmark_stats"].items():
    print(f"  {wname:9s} {s}")
print("\ncurrent allocation (vol_targeted):",
      payload["variants"]["vol_targeted"]["current_allocation"])

# ---------------- FastAPI wiring (drop into MVPdash) ----------------
# Run this script on a daily cron after market close, then:
#
#   from fastapi.responses import FileResponse
#
#   @app.get("/api/portfolio")
#   def portfolio():
#       return FileResponse("portfolio_data.json",
#                           headers={"Cache-Control": "public, max-age=3600"})
#
# Frontend fetches once; all figures below render from this one payload.