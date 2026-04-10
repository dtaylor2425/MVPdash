"""
src/asset_scoring.py
Four-factor asset alignment scoring.

Each asset is scored 0-100 across four independent dimensions:
  1. Regime Alignment  (0-25) — historical performance in current regime state
  2. Signal Confluence  (0-25) — count of confirming vs opposing macro signals
  3. Momentum & Trend   (0-25) — price-based trend, MA structure, RSI
  4. Relative Value      (0-25) — cross-asset rotation and relative positioning

The total score produces genuinely differentiated readings per asset
because each factor uses asset-specific weights and signal maps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Asset-specific macro signal maps
# Each entry: (signal_key, direction, weight)
#   direction: +1 means rising signal is GOOD for this asset
#              -1 means rising signal is BAD for this asset
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_MAPS = {
    "SPY": [
        ("hy_oas",    -1, 1.2),   # widening credit = bad
        ("real10",    -1, 1.0),   # rising real yields = bad (multiple compression)
        ("curve",     +1, 0.8),   # steepening = good (growth)
        ("breadth",   +1, 0.9),   # improving breadth = good
        ("dollar",    -1, 0.5),   # strong dollar = modest headwind
        ("claims",    -1, 0.7),   # rising claims = bad
        ("vix",       -1, 0.6),   # rising VIX = bad
    ],
    "QQQ": [
        ("hy_oas",    -1, 1.0),
        ("real10",    -1, 1.5),   # growth stocks most sensitive to real yields
        ("curve",     +1, 0.6),
        ("breadth",   +1, 0.5),   # less breadth-dependent (cap-weighted tech)
        ("dollar",    -1, 0.6),
        ("vix",       -1, 0.8),
    ],
    "GLD": [
        ("hy_oas",    +1, 0.6),   # credit stress = gold bid
        ("real10",    -1, 1.5),   # falling real yields = strongest gold driver
        ("dollar",    -1, 1.2),   # weak dollar = strong gold
        ("vix",       +1, 0.5),   # fear = gold bid
        ("breakeven", +1, 0.8),   # rising inflation expectations = gold
    ],
    "SLV": [
        ("hy_oas",    +1, 0.4),
        ("real10",    -1, 1.2),
        ("dollar",    -1, 1.0),
        ("copper_gold", +1, 0.8), # industrial demand component
        ("vix",       +1, 0.3),
    ],
    "TLT": [
        ("hy_oas",    +1, 0.8),   # credit stress = flight to safety
        ("real10",    -1, 0.5),   # already priced in, less direct
        ("curve",     -1, 1.0),   # flattening = good for long bonds
        ("vix",       +1, 0.7),   # fear = duration bid
        ("breakeven", -1, 0.8),   # rising inflation = bad for bonds
        ("claims",    +1, 0.6),   # rising claims = rate cut expectations
    ],
    "HYG": [
        ("hy_oas",    -1, 1.5),   # direct relationship
        ("real10",    -1, 0.5),
        ("curve",     +1, 0.7),
        ("breadth",   +1, 0.6),
        ("vix",       -1, 0.8),
    ],
    "BTC": [
        ("hy_oas",    -1, 0.8),
        ("real10",    -1, 1.0),   # risk asset, sensitive to real yields
        ("dollar",    -1, 1.0),
        ("net_liq",   +1, 1.3),   # most liquidity-sensitive asset
        ("vix",       -1, 0.5),
    ],
    "XLU": [
        ("hy_oas",    +1, 0.5),   # defensive
        ("real10",    -1, 1.0),   # yield proxy, hurt by rising rates
        ("curve",     -1, 0.6),   # late cycle / defensive
        ("vix",       +1, 0.6),   # fear = defensive bid
        ("claims",    +1, 0.5),   # weakness = defensive rotation
    ],
    "XLC": [
        ("hy_oas",    -1, 0.9),
        ("real10",    -1, 0.8),
        ("breadth",   +1, 0.7),
        ("vix",       -1, 0.7),
    ],
}

# Regime return profiles: average z-score of returns in each regime bucket
# Positive = asset tends to do well in this regime
REGIME_PROFILES = {
    "SPY":  {"Risk Off": -1.2, "Bearish": -0.5, "Neutral": 0.2, "Bullish": 0.8, "Risk On": 1.0},
    "QQQ":  {"Risk Off": -1.5, "Bearish": -0.7, "Neutral": 0.3, "Bullish": 1.0, "Risk On": 1.2},
    "GLD":  {"Risk Off": 1.0,  "Bearish": 0.5,  "Neutral": 0.0, "Bullish": -0.3, "Risk On": -0.5},
    "SLV":  {"Risk Off": 0.5,  "Bearish": 0.2,  "Neutral": 0.1, "Bullish": 0.3, "Risk On": 0.0},
    "TLT":  {"Risk Off": 1.2,  "Bearish": 0.7,  "Neutral": 0.0, "Bullish": -0.5, "Risk On": -0.8},
    "HYG":  {"Risk Off": -1.0, "Bearish": -0.4, "Neutral": 0.3, "Bullish": 0.6, "Risk On": 0.5},
    "BTC":  {"Risk Off": -1.5, "Bearish": -0.8, "Neutral": 0.5, "Bullish": 1.2, "Risk On": 1.5},
    "XLU":  {"Risk Off": 0.8,  "Bearish": 0.5,  "Neutral": 0.1, "Bullish": -0.2, "Risk On": -0.5},
    "XLC":  {"Risk Off": -1.0, "Bearish": -0.4, "Neutral": 0.2, "Bullish": 0.7, "Risk On": 0.8},
}

# Asset display names
ASSET_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "GLD": "Gold",
    "SLV": "Silver", "TLT": "Long Bond", "HYG": "High Yield",
    "BTC": "Bitcoin", "XLU": "Utilities", "XLC": "Comms/Media",
}


def _zscore_last(s, window=252):
    s = s.dropna()
    if len(s) < 30:
        return None
    w = min(window, len(s))
    tail = s.iloc[-w:]
    mu = float(tail.mean())
    sd = float(tail.std())
    if sd == 0:
        return 0.0
    return float((tail.iloc[-1] - mu) / sd)


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Factor 1: Regime Alignment (0-25)
# ─────────────────────────────────────────────────────────────────────────────

def _regime_alignment(ticker, regime_label, regime_score):
    profile = REGIME_PROFILES.get(ticker)
    if not profile:
        return 12.5  # neutral default

    # Get the z-score for this regime state
    regime_z = profile.get(regime_label, 0.0)

    # Also blend with continuous score for smoother transitions
    # Map regime_score 0-100 to a blended alignment
    continuous_z = (regime_score - 50) / 50.0  # [-1, +1]

    # Risk assets benefit from high regime, safe havens from low
    is_risk_asset = ticker in ("SPY", "QQQ", "HYG", "BTC", "XLC")
    if not is_risk_asset:
        continuous_z = -continuous_z

    blended = regime_z * 0.6 + continuous_z * 0.4
    # Map [-2, +2] to [0, 25]
    return float(np.clip((blended + 2) / 4 * 25, 0, 25))


# ─────────────────────────────────────────────────────────────────────────────
# Factor 2: Signal Confluence (0-25)
# ─────────────────────────────────────────────────────────────────────────────

def _build_signal_readings(macro, proxies):
    """Extract current z-scores for all signals used in confluence scoring."""
    readings = {}

    if "hy_oas" in macro.columns:
        readings["hy_oas"] = _zscore_last(macro["hy_oas"])

    if "real10" in macro.columns:
        readings["real10"] = _zscore_last(macro["real10"])

    if "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        readings["curve"] = _zscore_last(curve)

    if "RSP" in proxies.columns and "SPY" in proxies.columns:
        breadth = (proxies["RSP"] / proxies["SPY"]).dropna()
        readings["breadth"] = _zscore_last(breadth)

    if "dollar_broad" in macro.columns:
        readings["dollar"] = _zscore_last(macro["dollar_broad"])

    if "init_claims" in macro.columns:
        readings["claims"] = _zscore_last(macro["init_claims"])

    if "^VIX" in proxies.columns:
        readings["vix"] = _zscore_last(proxies["^VIX"])

    if "y10" in macro.columns and "real10" in macro.columns:
        be = (macro["y10"] - macro["real10"]).dropna()
        readings["breakeven"] = _zscore_last(be)

    if "CPER" in proxies.columns and "GLD" in proxies.columns:
        cu_au = (proxies["CPER"] / proxies["GLD"]).dropna()
        readings["copper_gold"] = _zscore_last(cu_au)

    # Net liquidity proxy
    if "fed_assets" in macro.columns:
        readings["net_liq"] = _zscore_last(macro["fed_assets"])

    return readings


def _signal_confluence(ticker, readings):
    sig_map = SIGNAL_MAPS.get(ticker)
    if not sig_map:
        return 12.5, 0, 0, []

    confirm = 0
    against = 0
    total_weight = 0
    weighted_sum = 0
    factors = []

    for signal_key, direction, weight in sig_map:
        z = readings.get(signal_key)
        if z is None:
            continue

        # Signal contribution: z * direction
        # Positive = confirming for this asset, negative = opposing
        contribution = z * direction
        weighted_sum += contribution * weight
        total_weight += weight

        if contribution > 0.3:
            confirm += 1
            factors.append({"name": signal_key, "direction": 1, "z": round(z, 2)})
        elif contribution < -0.3:
            against += 1
            factors.append({"name": signal_key, "direction": -1, "z": round(z, 2)})
        else:
            factors.append({"name": signal_key, "direction": 0, "z": round(z, 2)})

    if total_weight == 0:
        return 12.5, 0, 0, factors

    normalized = weighted_sum / total_weight  # roughly [-2.5, +2.5]
    score = float(np.clip((normalized + 2.5) / 5 * 25, 0, 25))
    return score, confirm, against, factors


# ─────────────────────────────────────────────────────────────────────────────
# Factor 3: Momentum & Trend (0-25)
# ─────────────────────────────────────────────────────────────────────────────

def _momentum_trend(ticker, proxies):
    col = ticker
    # Handle special tickers
    if ticker == "BTC" and "BTC-USD" in proxies.columns:
        col = "BTC-USD"
    elif ticker not in proxies.columns:
        return 12.5, None, None, None, None

    px = proxies[col].dropna()
    if len(px) < 50:
        return 12.5, None, None, None, None

    last = float(px.iloc[-1])
    ma50 = float(px.iloc[-50:].mean())
    ma200 = float(px.iloc[-min(200, len(px)):].mean()) if len(px) >= 200 else None

    # Trend score components
    score = 12.5  # neutral base

    # Above/below 50d MA: +/- 4 pts
    if last > ma50:
        score += 4
    else:
        score -= 4

    # Above/below 200d MA: +/- 4 pts
    if ma200 is not None:
        if last > ma200:
            score += 4
        else:
            score -= 4

    # 50d above 200d (golden cross): +3 pts
    if ma200 is not None and ma50 > ma200:
        score += 3
    elif ma200 is not None:
        score -= 3

    # 1M return z-score: +/- 3 pts
    if len(px) >= 21:
        ret_1m = float(px.iloc[-1] / px.iloc[-21] - 1)
        score += np.clip(ret_1m * 30, -3, 3)

    # 3M return z-score: +/- 2 pts
    if len(px) >= 63:
        ret_3m = float(px.iloc[-1] / px.iloc[-63] - 1)
        score += np.clip(ret_3m * 15, -2, 2)

    # RSI-like: mean reversion pressure at extremes
    if len(px) >= 14:
        changes = px.diff().iloc[-14:]
        gains = changes.clip(lower=0).mean()
        losses = (-changes.clip(upper=0)).mean()
        if losses > 0:
            rsi = 100 - 100 / (1 + gains / losses)
        else:
            rsi = 100
        # Overbought/oversold adjustment
        if rsi > 75:
            score -= 1.5  # slight caution
        elif rsi < 25:
            score += 1.5  # oversold bounce potential

    price = last
    ret_1w = float(px.iloc[-1] / px.iloc[-min(5, len(px))] - 1) if len(px) >= 5 else None
    ret_1m = float(px.iloc[-1] / px.iloc[-min(21, len(px))] - 1) if len(px) >= 21 else None
    ret_3m = float(px.iloc[-1] / px.iloc[-min(63, len(px))] - 1) if len(px) >= 63 else None

    return float(np.clip(score, 0, 25)), price, ret_1w, ret_1m, ret_3m


# ─────────────────────────────────────────────────────────────────────────────
# Factor 4: Relative Value (0-25)
# ─────────────────────────────────────────────────────────────────────────────

def _relative_value(ticker, proxies, macro):
    score = 12.5  # neutral base

    # For equities: RSP/SPY breadth context
    if ticker in ("SPY", "QQQ", "XLC"):
        if "RSP" in proxies.columns and "SPY" in proxies.columns:
            ratio = (proxies["RSP"] / proxies["SPY"]).dropna()
            if len(ratio) >= 63:
                z = _zscore_last(ratio, 252)
                if z is not None:
                    # Improving breadth = positive for broad equity
                    adj = z * 3 if ticker == "SPY" else z * 1.5
                    score += np.clip(adj, -5, 5)

    # For gold: real yield relative value
    if ticker == "GLD" and "real10" in macro.columns:
        real = macro["real10"].dropna()
        if len(real) >= 30:
            z = _zscore_last(real)
            if z is not None:
                # High real yields = gold cheap (relative value opportunity)
                score += np.clip(-z * 2, -5, 5)

    # For bonds: curve steepness
    if ticker == "TLT" and "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        if len(curve) >= 30:
            z = _zscore_last(curve)
            if z is not None:
                # Steep curve = bonds less attractive, flat = more attractive
                score += np.clip(-z * 2.5, -5, 5)

    # For crypto: liquidity relative value
    if ticker == "BTC" and "fed_assets" in macro.columns:
        fa = macro["fed_assets"].dropna()
        if len(fa) >= 60:
            z = _zscore_last(fa.pct_change(63).dropna())
            if z is not None:
                score += np.clip(z * 3, -5, 5)

    return float(np.clip(score, 0, 25))


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def _label_from_score(score):
    if score >= 70: return "Strong Tailwind"
    if score >= 60: return "Bullish"
    if score >= 52: return "Mild Tailwind"
    if score >= 48: return "Neutral"
    if score >= 40: return "Mild Headwind"
    if score >= 30: return "Bearish"
    return "Strong Headwind"


def score_all_assets(
    macro: pd.DataFrame,
    proxies: pd.DataFrame,
    regime_score: int,
    regime_label: str,
) -> Dict:
    """Score all tracked assets using the four-factor model."""
    tickers = list(SIGNAL_MAPS.keys())
    readings = _build_signal_readings(macro, proxies)

    results = {}
    for ticker in tickers:
        f1 = _regime_alignment(ticker, regime_label, regime_score)
        f2_score, n_confirm, n_against, factors = _signal_confluence(ticker, readings)
        f3, price, ret_1w, ret_1m, ret_3m = _momentum_trend(ticker, proxies)
        f4 = _relative_value(ticker, proxies, macro)

        total = f1 + f2_score + f3 + f4
        total = int(round(total))

        results[ticker] = {
            "name": ASSET_NAMES.get(ticker, ticker),
            "price": price,
            "ret_1w": ret_1w,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "alignment": {
                "score": total,
                "label": _label_from_score(total),
                "n_confirm": n_confirm,
                "n_against": n_against,
                "factors": factors,
                "breakdown": {
                    "regime": round(f1, 1),
                    "confluence": round(f2_score, 1),
                    "momentum": round(f3, 1),
                    "relative_value": round(f4, 1),
                },
            },
        }

    return results
#new