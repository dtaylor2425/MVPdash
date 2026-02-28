# src/compute.py
import numpy as np
import pandas as pd
from typing import List, Optional

HORIZONS = {
    "1w":  5,
    "1m":  21,
    "3m":  63,
    "6m":  126,
}


# ── Relative strength ─────────────────────────────────────────────────────────

def rs_horizon_table(prices: pd.DataFrame, tickers: list, base: str = "SPY") -> pd.DataFrame:
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


# ── Trend direction (single canonical implementation) ────────────────────────

def trend_dir(series: pd.Series, lookback: int = 63) -> Optional[int]:
    """Returns 1 if series[-1] > series[-lookback-1], else 0. None if insufficient data."""
    s = series.dropna()
    if len(s) < lookback + 1:
        return None
    return int(s.iloc[-1] > s.iloc[-(lookback + 1)])


# ── Volatility compression ────────────────────────────────────────────────────

def vol_compression(series: pd.Series, window: int = 63) -> bool:
    """True if recent volatility is more than 30% below the prior same-length window."""
    r = series.pct_change().dropna()
    if len(r) < window * 2:
        return False
    vol_recent = r.iloc[-window:].std()
    vol_prior  = r.iloc[-2 * window:-window].std()
    if pd.isna(vol_prior) or vol_prior == 0:
        return False
    return bool(vol_recent < 0.7 * vol_prior)


# ── Point-in-time helpers (date-based, safe for FRED gaps) ───────────────────

def _nearest_before(series: pd.Series, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    s   = series.dropna()
    idx = s.index[s.index <= dt]
    return pd.Timestamp(idx.max()) if len(idx) else None


def delta_over_days(series: pd.Series, days: int):
    """Returns (latest, prev, delta) using calendar-day lookback. Safe for gapped series."""
    s = series.dropna()
    if s.empty:
        return None, None, None
    end    = pd.Timestamp(s.index.max())
    end_i  = _nearest_before(s, end)
    prev_i = _nearest_before(s, end - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None, None, None
    latest = float(s.loc[end_i])
    prev   = float(s.loc[prev_i])
    return latest, prev, float(latest - prev)


def pct_return_over_days(series: pd.Series, days: int) -> Optional[float]:
    """Percent return over a calendar-day window. Safe for gapped series."""
    s = series.dropna()
    if len(s) < 2:
        return None
    end    = s.index.max()
    end_i  = _nearest_before(s, end)
    prev_i = _nearest_before(s, end - pd.Timedelta(days=days))
    if end_i is None or prev_i is None:
        return None
    a, b = float(s.loc[end_i]), float(s.loc[prev_i])
    return None if b == 0 else (a / b) - 1.0


def weekly_change(series: pd.Series, n_days: int = 7) -> Optional[float]:
    """Absolute change over n_days calendar days."""
    latest, prev, delta = delta_over_days(series, n_days)
    return delta


def weekly_pct_change(series: pd.Series, n_days: int = 7) -> Optional[float]:
    """Percent change over n_days calendar days."""
    return pct_return_over_days(series, n_days)


def _last_valid(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return None if s.empty else float(s.iloc[-1])


def trend_flip(series: pd.Series, lookback: int = 63) -> Optional[str]:
    s = series.dropna()
    if len(s) < lookback + 2:
        return None
    now_up  = int(s.iloc[-1]  > s.iloc[-(lookback + 1)])
    prev_up = int(s.iloc[-2]  > s.iloc[-(lookback + 2)])
    if now_up == prev_up:
        return None
    return "up" if now_up == 1 else "down"


# ── Component contribution (matches regime.py scoring exactly) ───────────────

def component_contribution(c: dict) -> float:
    """
    Returns the signed contribution of a regime component to the total score.
    Uses the pre-computed 'contribution' field stored by regime._compute_core().
    Falls back to zscore*weight for the dollar component if contribution absent.
    """
    if not isinstance(c, dict):
        return 0.0
    # Primary: use the pre-computed contribution from regime._compute_core()
    contrib = c.get("contribution")
    if isinstance(contrib, (int, float, np.floating)):
        return float(contrib)
    # Legacy fallback: only dollar has zscore; others default to 0
    z = c.get("zscore")
    w = c.get("weight")
    if isinstance(z, (int, float, np.floating)) and isinstance(w, (int, float, np.floating)):
        return float(z) * float(w)
    return 0.0


# ── What changed narrative ────────────────────────────────────────────────────

def generate_what_changed(
    macro: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    rotation_rs: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []

    # Credit spreads (7 calendar days = ~5 trading days)
    if "hy_oas" in macro.columns:
        ch = weekly_change(macro["hy_oas"], n_days=7)
        if ch is not None and abs(ch) >= 0.10:
            direction = "widened" if ch > 0 else "tightened"
            bullets.append(f"High yield spreads {direction} by {abs(ch):.2f} pp this week.")

    # Yield curve
    if "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        ch = weekly_change(curve, n_days=7)
        if ch is not None and abs(ch) >= 0.10:
            direction = "steepened" if ch > 0 else "flattened"
            bullets.append(f"The 10y minus 2y curve {direction} by {abs(ch):.2f} pp this week.")
        flip = trend_flip(curve, lookback=63)
        if flip:
            bullets.append(f"Curve trend flipped {flip} on a 3-month lookback.")

    # Real yields
    if "real10" in macro.columns:
        ch = weekly_change(macro["real10"], n_days=7)
        if ch is not None and abs(ch) >= 0.10:
            direction = "rose" if ch > 0 else "fell"
            bullets.append(f"Real yields {direction} by {abs(ch):.2f} pp this week.")

    # Dollar (weekly FRED series — use 10 calendar days to reliably find prior week)
    if "dollar_broad" in macro.columns:
        pct = weekly_pct_change(macro["dollar_broad"], n_days=10)
        if pct is not None and abs(pct) >= 0.005:
            direction = "strengthened" if pct > 0 else "weakened"
            bullets.append(f"The dollar {direction} by {abs(pct)*100:.2f}% this week.")

    # Risk appetite: IWM / SPY ratio
    if isinstance(proxy_prices, pd.DataFrame) and \
            "IWM" in proxy_prices.columns and "SPY" in proxy_prices.columns:
        ratio = (proxy_prices["IWM"] / proxy_prices["SPY"]).dropna()
        pct   = weekly_pct_change(ratio, n_days=7)
        if pct is not None and abs(pct) >= 0.005:
            direction = "improved" if pct > 0 else "deteriorated"
            bullets.append(f"Risk appetite {direction}: IWM/SPY moved {pct*100:.2f}% this week.")
        flip = trend_flip(ratio, lookback=63)
        if flip:
            bullets.append(f"IWM/SPY trend flipped {flip} on a 3-month lookback.")

    # Rotation leaders / laggards
    if rotation_rs is not None and not rotation_rs.empty:
        col = "1m" if "1m" in rotation_rs.columns else rotation_rs.columns[0]
        top = rotation_rs.sort_values(col, ascending=False).head(2)
        bot = rotation_rs.sort_values(col, ascending=True).head(2)
        winners = ", ".join(f"{idx} ({val*100:.1f}%)" for idx, val in top[col].items())
        losers  = ", ".join(f"{idx} ({val*100:.1f}%)" for idx, val in bot[col].items())
        bullets.append(f"Rotation leaders vs SPY ({col}): {winners}.")
        bullets.append(f"Rotation laggards vs SPY ({col}): {losers}.")

    return bullets[:6]


# ── Backward-compatible regime wrapper ───────────────────────────────────────

def compute_regime(macro: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Thin wrapper kept for any legacy callers. Prefer importing compute_regime_v3 directly."""
    from src.regime import compute_regime_v3
    res = compute_regime_v3(macro=macro, proxies=prices, lookback_trend=63)
    return {
        "score": res.score,
        "label": res.label,
        "signals": {"confidence": res.confidence, "summary": res.summary},
    }