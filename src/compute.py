import pandas as pd
import numpy as np

HORIZONS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126
}

def returns(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    return prices.pct_change(n)

def rel_strength(prices: pd.DataFrame, base: str = "SPY") -> pd.DataFrame:
    base_ret = prices[base].pct_change()
    rs = prices.pct_change().sub(base_ret, axis=0)
    return rs

def rs_horizon_table(prices: pd.DataFrame, tickers: list[str], base: str = "SPY") -> pd.DataFrame:
    out = []
    for t in tickers:
        if t == base:
            continue
        row = {"ticker": t}
        for label, n in HORIZONS.items():
            r_t = prices[t].pct_change(n).iloc[-1]
            r_b = prices[base].pct_change(n).iloc[-1]
            row[label] = float(r_t - r_b)
        out.append(row)
    df = pd.DataFrame(out).set_index("ticker")
    return df.sort_values("1m", ascending=False)
