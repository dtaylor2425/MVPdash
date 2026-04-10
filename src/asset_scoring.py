"""
src/asset_scoring.py
Four-factor asset alignment scoring.

Each asset scored 0-100 across four dimensions:
  1. Regime Alignment  (0-25)
  2. Signal Confluence  (0-25)
  3. Momentum & Trend   (0-25)
  4. Relative Value      (0-25)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── Asset-specific macro signal maps ─────────────────────────────────────────
# (signal_key, direction, weight)
#   +1 = rising signal is GOOD for this asset
#   -1 = rising signal is BAD for this asset

SIGNAL_MAPS = {
    # ── Broad equity ──────────────────────────────────────────────────────────
    "SPY": [
        ("hy_oas", -1, 1.2), ("real10", -1, 1.0), ("curve", +1, 0.8),
        ("breadth", +1, 0.9), ("dollar", -1, 0.5), ("claims", -1, 0.7), ("vix", -1, 0.6),
    ],
    "QQQ": [
        ("hy_oas", -1, 1.0), ("real10", -1, 1.5), ("curve", +1, 0.6),
        ("breadth", +1, 0.5), ("dollar", -1, 0.6), ("vix", -1, 0.8),
    ],
    "IWM": [
        ("hy_oas", -1, 1.0), ("real10", -1, 0.8), ("curve", +1, 1.2),
        ("breadth", +1, 1.3), ("dollar", -1, 0.7), ("claims", -1, 1.0), ("vix", -1, 0.7),
    ],

    # ── Sectors ───────────────────────────────────────────────────────────────
    "XLE": [
        ("hy_oas", -1, 0.6), ("real10", +1, 0.3), ("curve", +1, 0.9),
        ("dollar", -1, 0.8), ("claims", -1, 0.5), ("breakeven", +1, 1.0),
        ("copper_gold", +1, 0.7),
    ],
    "XLF": [
        ("hy_oas", -1, 1.0), ("real10", +1, 0.6), ("curve", +1, 1.5),
        ("claims", -1, 0.8), ("vix", -1, 0.6),
    ],
    "XLK": [
        ("hy_oas", -1, 0.9), ("real10", -1, 1.4), ("curve", +1, 0.5),
        ("breadth", +1, 0.6), ("dollar", -1, 0.5), ("vix", -1, 0.7),
    ],
    "SMH": [
        ("hy_oas", -1, 0.8), ("real10", -1, 1.3), ("curve", +1, 0.5),
        ("dollar", -1, 0.7), ("vix", -1, 0.9), ("copper_gold", +1, 0.6),
    ],
    "XLI": [
        ("hy_oas", -1, 0.8), ("real10", -1, 0.5), ("curve", +1, 1.2),
        ("breadth", +1, 1.0), ("claims", -1, 0.9), ("copper_gold", +1, 0.8),
    ],
    "XLV": [
        ("hy_oas", -1, 0.5), ("real10", -1, 0.6), ("curve", +1, 0.4),
        ("claims", -1, 0.4), ("vix", +1, 0.3),
    ],
    "XLP": [
        ("hy_oas", +1, 0.4), ("real10", -1, 0.7), ("curve", -1, 0.5),
        ("vix", +1, 0.5), ("claims", +1, 0.4),
    ],
    "XLU": [
        ("hy_oas", +1, 0.5), ("real10", -1, 1.0), ("curve", -1, 0.6),
        ("vix", +1, 0.6), ("claims", +1, 0.5),
    ],
    "XLC": [
        ("hy_oas", -1, 0.9), ("real10", -1, 0.8), ("breadth", +1, 0.7), ("vix", -1, 0.7),
    ],

    # ── Commodities & safe havens ─────────────────────────────────────────────
    "GLD": [
        ("hy_oas", +1, 0.6), ("real10", -1, 1.5), ("dollar", -1, 1.2),
        ("vix", +1, 0.5), ("breakeven", +1, 0.8),
    ],
    "SLV": [
        ("hy_oas", +1, 0.4), ("real10", -1, 1.2), ("dollar", -1, 1.0),
        ("copper_gold", +1, 0.8), ("vix", +1, 0.3),
    ],

    # ── Bonds ─────────────────────────────────────────────────────────────────
    "TLT": [
        ("hy_oas", +1, 0.8), ("real10", -1, 0.5), ("curve", -1, 1.0),
        ("vix", +1, 0.7), ("breakeven", -1, 0.8), ("claims", +1, 0.6),
    ],
    "HYG": [
        ("hy_oas", -1, 1.5), ("real10", -1, 0.5), ("curve", +1, 0.7),
        ("breadth", +1, 0.6), ("vix", -1, 0.8),
    ],

    # ── Crypto ────────────────────────────────────────────────────────────────
    "BTC": [
        ("hy_oas", -1, 0.8), ("real10", -1, 1.0), ("dollar", -1, 1.0),
        ("net_liq", +1, 1.3), ("vix", -1, 0.5),
    ],
}

# Regime return profiles
REGIME_PROFILES = {
    "SPY":  {"Risk Off": -1.2, "Bearish": -0.5, "Neutral": 0.2, "Bullish": 0.8, "Risk On": 1.0},
    "QQQ":  {"Risk Off": -1.5, "Bearish": -0.7, "Neutral": 0.3, "Bullish": 1.0, "Risk On": 1.2},
    "IWM":  {"Risk Off": -1.4, "Bearish": -0.8, "Neutral": 0.1, "Bullish": 1.0, "Risk On": 1.3},
    "XLE":  {"Risk Off": -0.8, "Bearish": -0.3, "Neutral": 0.3, "Bullish": 0.7, "Risk On": 0.5},
    "XLF":  {"Risk Off": -1.3, "Bearish": -0.6, "Neutral": 0.2, "Bullish": 0.9, "Risk On": 1.1},
    "XLK":  {"Risk Off": -1.4, "Bearish": -0.6, "Neutral": 0.3, "Bullish": 1.0, "Risk On": 1.2},
    "SMH":  {"Risk Off": -1.6, "Bearish": -0.8, "Neutral": 0.4, "Bullish": 1.2, "Risk On": 1.4},
    "XLI":  {"Risk Off": -1.1, "Bearish": -0.5, "Neutral": 0.2, "Bullish": 0.8, "Risk On": 1.0},
    "XLV":  {"Risk Off": 0.3,  "Bearish": 0.2,  "Neutral": 0.1, "Bullish": 0.0, "Risk On": -0.2},
    "XLP":  {"Risk Off": 0.7,  "Bearish": 0.4,  "Neutral": 0.0, "Bullish": -0.3, "Risk On": -0.5},
    "XLU":  {"Risk Off": 0.8,  "Bearish": 0.5,  "Neutral": 0.1, "Bullish": -0.2, "Risk On": -0.5},
    "XLC":  {"Risk Off": -1.0, "Bearish": -0.4, "Neutral": 0.2, "Bullish": 0.7, "Risk On": 0.8},
    "GLD":  {"Risk Off": 1.0,  "Bearish": 0.5,  "Neutral": 0.0, "Bullish": -0.3, "Risk On": -0.5},
    "SLV":  {"Risk Off": 0.5,  "Bearish": 0.2,  "Neutral": 0.1, "Bullish": 0.3, "Risk On": 0.0},
    "TLT":  {"Risk Off": 1.2,  "Bearish": 0.7,  "Neutral": 0.0, "Bullish": -0.5, "Risk On": -0.8},
    "HYG":  {"Risk Off": -1.0, "Bearish": -0.4, "Neutral": 0.3, "Bullish": 0.6, "Risk On": 0.5},
    "BTC":  {"Risk Off": -1.5, "Bearish": -0.8, "Neutral": 0.5, "Bullish": 1.2, "Risk On": 1.5},
}

ASSET_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "XLE": "Energy", "XLF": "Financials", "XLK": "Technology",
    "SMH": "Semiconductors", "XLI": "Industrials", "XLV": "Healthcare",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLC": "Comms/Media",
    "GLD": "Gold", "SLV": "Silver", "TLT": "Long Bond",
    "HYG": "High Yield", "BTC": "Bitcoin",
}

# Which assets are risk assets (score benefits from high regime)
RISK_ASSETS = {"SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "SMH", "XLI", "XLC", "HYG", "BTC"}


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


# ── Factor 1: Regime Alignment (0-25) ────────────────────────────────────────

def _regime_alignment(ticker, regime_label, regime_score):
    profile = REGIME_PROFILES.get(ticker)
    if not profile:
        return 12.5

    regime_z = profile.get(regime_label, 0.0)
    continuous_z = (regime_score - 50) / 50.0
    if ticker not in RISK_ASSETS:
        continuous_z = -continuous_z

    blended = regime_z * 0.6 + continuous_z * 0.4
    return float(np.clip((blended + 2) / 4 * 25, 0, 25))


# ── Factor 2: Signal Confluence (0-25) ───────────────────────────────────────

def _build_signal_readings(macro, proxies):
    readings = {}
    if "hy_oas" in macro.columns:
        readings["hy_oas"] = _zscore_last(macro["hy_oas"])
    if "real10" in macro.columns:
        readings["real10"] = _zscore_last(macro["real10"])
    if "y10" in macro.columns and "y2" in macro.columns:
        readings["curve"] = _zscore_last((macro["y10"] - macro["y2"]).dropna())
    if "RSP" in proxies.columns and "SPY" in proxies.columns:
        readings["breadth"] = _zscore_last((proxies["RSP"] / proxies["SPY"]).dropna())
    if "dollar_broad" in macro.columns:
        readings["dollar"] = _zscore_last(macro["dollar_broad"])
    if "init_claims" in macro.columns:
        readings["claims"] = _zscore_last(macro["init_claims"])
    if "^VIX" in proxies.columns:
        readings["vix"] = _zscore_last(proxies["^VIX"])
    if "y10" in macro.columns and "real10" in macro.columns:
        readings["breakeven"] = _zscore_last((macro["y10"] - macro["real10"]).dropna())
    if "CPER" in proxies.columns and "GLD" in proxies.columns:
        readings["copper_gold"] = _zscore_last((proxies["CPER"] / proxies["GLD"]).dropna())
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

    normalized = weighted_sum / total_weight
    score = float(np.clip((normalized + 2.5) / 5 * 25, 0, 25))
    return score, confirm, against, factors


# ── Factor 3: Momentum & Trend (0-25) ────────────────────────────────────────

def _momentum_trend(ticker, proxies):
    col = ticker
    if ticker == "BTC" and "BTC-USD" in proxies.columns:
        col = "BTC-USD"
    elif ticker not in proxies.columns:
        return 12.5, None, None, None, None

    px = proxies[col].dropna()
    if len(px) < 50:
        return 12.5, float(px.iloc[-1]) if len(px) > 0 else None, None, None, None

    last = float(px.iloc[-1])
    ma50 = float(px.iloc[-50:].mean())
    ma200 = float(px.iloc[-min(200, len(px)):].mean()) if len(px) >= 200 else None

    score = 12.5

    # 50d MA: +/- 4
    if last > ma50:
        score += 4
    else:
        score -= 4

    # 200d MA: +/- 4
    if ma200 is not None:
        if last > ma200:
            score += 4
        else:
            score -= 4

    # Golden/death cross: +/- 3
    if ma200 is not None:
        if ma50 > ma200:
            score += 3
        else:
            score -= 3

    # 1M return: +/- 3
    if len(px) >= 21:
        ret_1m = float(px.iloc[-1] / px.iloc[-21] - 1)
        score += float(np.clip(ret_1m * 30, -3, 3))

    # 3M return: +/- 2
    if len(px) >= 63:
        ret_3m = float(px.iloc[-1] / px.iloc[-63] - 1)
        score += float(np.clip(ret_3m * 15, -2, 2))

    # RSI
    if len(px) >= 14:
        changes = px.diff().iloc[-14:]
        gains = changes.clip(lower=0).mean()
        losses = (-changes.clip(upper=0)).mean()
        rsi = 100 if losses == 0 else 100 - 100 / (1 + gains / losses)
        if rsi > 75:
            score -= 1.5
        elif rsi < 25:
            score += 1.5

    ret_1w = float(px.iloc[-1] / px.iloc[-min(5, len(px))] - 1) if len(px) >= 5 else None
    ret_1m = float(px.iloc[-1] / px.iloc[-min(21, len(px))] - 1) if len(px) >= 21 else None
    ret_3m = float(px.iloc[-1] / px.iloc[-min(63, len(px))] - 1) if len(px) >= 63 else None

    return float(np.clip(score, 0, 25)), last, ret_1w, ret_1m, ret_3m


# ── Factor 4: Relative Value (0-25) ──────────────────────────────────────────

def _relative_value(ticker, proxies, macro):
    score = 12.5

    # Breadth context for equities
    if ticker in RISK_ASSETS and "RSP" in proxies.columns and "SPY" in proxies.columns:
        ratio = (proxies["RSP"] / proxies["SPY"]).dropna()
        if len(ratio) >= 63:
            z = _zscore_last(ratio, 252)
            if z is not None:
                mult = 3.0 if ticker == "SPY" else 1.5
                score += float(np.clip(z * mult, -5, 5))

    # Curve steepness for financials
    if ticker == "XLF" and "y10" in macro.columns and "y2" in macro.columns:
        curve = (macro["y10"] - macro["y2"]).dropna()
        if len(curve) >= 30:
            z = _zscore_last(curve)
            if z is not None:
                score += float(np.clip(z * 3, -5, 5))

    # Real yield for gold
    if ticker == "GLD" and "real10" in macro.columns:
        z = _zscore_last(macro["real10"])
        if z is not None:
            score += float(np.clip(-z * 2, -5, 5))

    # Curve for bonds
    if ticker == "TLT" and "y10" in macro.columns and "y2" in macro.columns:
        z = _zscore_last((macro["y10"] - macro["y2"]).dropna())
        if z is not None:
            score += float(np.clip(-z * 2.5, -5, 5))

    # Liquidity for crypto
    if ticker == "BTC" and "fed_assets" in macro.columns:
        fa = macro["fed_assets"].dropna()
        if len(fa) >= 60:
            z = _zscore_last(fa.pct_change(63).dropna())
            if z is not None:
                score += float(np.clip(z * 3, -5, 5))

    # Energy: oil/inflation sensitivity
    if ticker == "XLE" and "y10" in macro.columns and "real10" in macro.columns:
        be = (macro["y10"] - macro["real10"]).dropna()
        if len(be) >= 30:
            z = _zscore_last(be)
            if z is not None:
                score += float(np.clip(z * 2, -4, 4))

    return float(np.clip(score, 0, 25))


# ── Labels ────────────────────────────────────────────────────────────────────

def _label_from_score(score):
    if score >= 70: return "Strong Tailwind"
    if score >= 60: return "Bullish"
    if score >= 52: return "Mild Tailwind"
    if score >= 48: return "Neutral"
    if score >= 40: return "Mild Headwind"
    if score >= 30: return "Bearish"
    return "Strong Headwind"


# ── Main ──────────────────────────────────────────────────────────────────────

def score_all_assets(macro, proxies, regime_score, regime_label):
    tickers = list(SIGNAL_MAPS.keys())
    readings = _build_signal_readings(macro, proxies)

    results = {}
    for ticker in tickers:
        f1 = _regime_alignment(ticker, regime_label, regime_score)
        f2_score, n_confirm, n_against, factors = _signal_confluence(ticker, readings)
        f3, price, ret_1w, ret_1m, ret_3m = _momentum_trend(ticker, proxies)
        f4 = _relative_value(ticker, proxies, macro)

        total = int(round(f1 + f2_score + f3 + f4))

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