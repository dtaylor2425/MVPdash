import numpy as np
import pandas as pd

HORIZONS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}

def rs_horizon_table(prices: pd.DataFrame, tickers: list[str], base: str = "SPY") -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t == base or t not in prices.columns:
            continue
        row = {"ticker": t}
        for label, n in HORIZONS.items():
            r_t = prices[t].pct_change(n).iloc[-1]
            r_b = prices[base].pct_change(n).iloc[-1]
            row[label] = float(r_t - r_b)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("ticker")
    return out.sort_values("1m", ascending=False)

def _trend_up(s: pd.Series, lookback: int = 63) -> int:
    s = s.dropna()
    if len(s) < lookback + 1:
        return 0
    return int(s.iloc[-1] > s.iloc[-lookback])

from src.regime import compute_regime_v3

def compute_regime(macro: pd.DataFrame, prices: pd.DataFrame) -> dict:
    # Backward compatible wrapper
    # prices in your old function is now proxy_prices, pass it as proxies
    res = compute_regime_v3(macro=macro, proxies=prices, lookback_trend=63)
    return {
        "score": res.score,
        "label": res.label,
        "signals": {
            "confidence": res.confidence,
            "summary": res.summary,
        },
    }

def _vol_compression(pr: pd.Series, window: int = 63) -> bool:
    r = pr.pct_change().dropna()
    if len(r) < window * 2:
        return False
    vol_recent = r.iloc[-window:].std()
    vol_prior = r.iloc[-2 * window : -window].std()
    return bool(vol_recent < 0.7 * vol_prior)

def setup_table(stock_prices: pd.DataFrame, base: str = "SPY") -> pd.DataFrame:
    rows = []
    if base not in stock_prices.columns:
        return pd.DataFrame()

    for t in stock_prices.columns:
        if t == base:
            continue
        s = stock_prices[t].dropna()
        if len(s) < 260:
            continue

        rs_3m = float(s.pct_change(63).iloc[-1] - stock_prices[base].pct_change(63).iloc[-1])
        rs_6m = float(s.pct_change(126).iloc[-1] - stock_prices[base].pct_change(126).iloc[-1])

        ma200 = s.rolling(200).mean().iloc[-1]
        trend = int(s.iloc[-1] > ma200)
        comp = int(_vol_compression(s))

        score = 50 + 200 * rs_3m + 150 * rs_6m + 10 * trend + 10 * comp

        rows.append(
            {
                "ticker": t,
                "score": float(score),
                "rs_3m": rs_3m,
                "rs_6m": rs_6m,
                "trend": trend,
                "compression": comp,
            }
        )

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out

def _last_valid(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])

def _value_n_days_ago(series: pd.Series, n: int) -> float | None:
    s = series.dropna()
    if len(s) < n + 1:
        return None
    return float(s.iloc[-(n + 1)])

def weekly_change(series: pd.Series, n: int = 5) -> float | None:
    last = _last_valid(series)
    prev = _value_n_days_ago(series, n)
    if last is None or prev is None:
        return None
    return last - prev

def weekly_pct_change(series: pd.Series, n: int = 5) -> float | None:
    last = _last_valid(series)
    prev = _value_n_days_ago(series, n)
    if last is None or prev is None or prev == 0:
        return None
    return (last / prev) - 1.0

def trend_flip(series: pd.Series, lookback: int = 63) -> str | None:
    s = series.dropna()
    if len(s) < lookback + 2:
        return None
    now_up = int(s.iloc[-1] > s.iloc[-(lookback + 1)])
    prev_up = int(s.iloc[-2] > s.iloc[-(lookback + 2)])
    if now_up == prev_up:
        return None
    return "up" if now_up == 1 else "down"

from typing import List, Dict
import numpy as np

def generate_what_changed(
    macro: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    rotation_rs: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []

    # Credit spreads
    if "hy_oas" in macro.columns:
        ch = weekly_change(macro["hy_oas"], n=5)
        if ch is not None and abs(ch) >= 0.10:
            direction = "widened" if ch > 0 else "tightened"
            bullets.append(f"High yield spreads {direction} by {abs(ch):.2f} percentage points this week.")

    # Curve spread
    if "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        ch = weekly_change(curve, n=5)
        if ch is not None and abs(ch) >= 0.10:
            direction = "steepened" if ch > 0 else "flattened"
            bullets.append(f"The 10y minus 2y curve {direction} by {abs(ch):.2f} percentage points this week.")

        flip = trend_flip(curve, lookback=63)
        if flip:
            bullets.append(f"Curve trend flipped {flip} on a 3 month lookback.")

    # Real yields
    if "real10" in macro.columns:
        ch = weekly_change(macro["real10"], n=5)
        if ch is not None and abs(ch) >= 0.10:
            direction = "rose" if ch > 0 else "fell"
            bullets.append(f"Real yields {direction} by {abs(ch):.2f} percentage points this week.")

    # Dollar broad
    if "dollar_broad" in macro.columns:
        pct = weekly_pct_change(macro["dollar_broad"], n=5)
        if pct is not None and abs(pct) >= 0.005:
            direction = "strengthened" if pct > 0 else "weakened"
            bullets.append(f"The dollar {direction} by {abs(pct)*100:.2f}% this week.")

    # Risk appetite ratio IWM over SPY (from proxy prices)
    if "IWM" in proxy_prices.columns and "SPY" in proxy_prices.columns:
        ratio = (proxy_prices["IWM"] / proxy_prices["SPY"]).dropna()
        pct = weekly_pct_change(ratio, n=5)
        if bullets is not None and pct is not None and abs(pct) >= 0.005:
            direction = "improved" if pct > 0 else "deteriorated"
            bullets.append(f"Risk appetite {direction} as IWM over SPY moved {pct*100:.2f}% this week.")

        flip = trend_flip(ratio, lookback=63)
        if flip:
            bullets.append(f"IWM over SPY trend flipped {flip} on a 3 month lookback.")

    # Rotation winners and losers from the RS table (use 1m by default)
    if rotation_rs is not None and not rotation_rs.empty:
        col = "1m" if "1m" in rotation_rs.columns else rotation_rs.columns[0]
        top = rotation_rs.sort_values(col, ascending=False).head(2)
        bot = rotation_rs.sort_values(col, ascending=True).head(2)

        winners = ", ".join([f"{idx} ({val*100:.1f}%)" for idx, val in top[col].items()])
        losers = ", ".join([f"{idx} ({val*100:.1f}%)" for idx, val in bot[col].items()])

        bullets.append(f"Rotation leaders vs SPY over {col}: {winners}.")
        bullets.append(f"Rotation laggards vs SPY over {col}: {losers}.")

    # Keep it short
    return bullets[:6]
