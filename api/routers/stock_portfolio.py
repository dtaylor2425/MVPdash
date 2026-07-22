"""
Living Stock Alpha Portfolio API

Routes:
    GET /api/stock-portfolio
    GET /api/stock-portfolio/status

Purpose:
    Make the stock portfolio the main Macro Engine portfolio experience.

The endpoint returns:
    - current stock holdings and target weights
    - model vs S&P 500 performance series
    - performance statistics
    - rebalance log
    - trade queue
    - sector exposure

This is separate from the ETF portfolio. ETF portfolio data should continue to
live behind the existing /api/portfolio route and the frontend toggle.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

try:
    from api.routers.stock_rankings import (
        _get_universe as _rankings_get_universe,
        _scan_rankings as _rankings_scan,
    )
except Exception:
    _rankings_get_universe = None
    _rankings_scan = None


router = APIRouter(
    prefix="/api/stock-portfolio",
    tags=["stock-portfolio"],
)

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

CACHE_TTL_SECONDS = 45 * 60

DEFAULT_TARGET_HOLDINGS = 10
MIN_TARGET_HOLDINGS = 6
MAX_TARGET_HOLDINGS = 15

MAX_SINGLE_POSITION = 0.12
MIN_SINGLE_POSITION = 0.018
MAX_SECTOR_WEIGHT = 0.34
ATR_STOP_MULTIPLE = 2.20

BENCHMARK_TICKER = "SPY"
TRANSACTION_COST_BPS = 10


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = _CACHE.get(key)

    if not item:
        return None

    expires_at, payload = item

    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None

    return payload


def _cache_set(key: str, payload: Dict[str, Any], ttl_seconds: int) -> Dict[str, Any]:
    _CACHE[key] = (time.time() + ttl_seconds, payload)
    return payload


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _clean(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    top = _finite(numerator)
    bottom = _finite(denominator)

    if top is None or bottom is None or abs(bottom) < 1e-12:
        return None

    return top / bottom


def _download_history(tickers: Sequence[str], period: str = "2y") -> pd.DataFrame:
    ticker_list = list(dict.fromkeys([ticker for ticker in tickers if ticker]))

    if not ticker_list:
        return pd.DataFrame()

    try:
        return yf.download(
            tickers=ticker_list,
            period=period,
            interval="1d",
            auto_adjust=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except TypeError:
        return yf.download(
            tickers=ticker_list,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()


def _slice_history(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        fields = {}

        for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if field in frame.columns.get_level_values(0):
                try:
                    fields[field] = frame[field][ticker]
                except Exception:
                    pass

        if not fields:
            return pd.DataFrame()

        output = pd.DataFrame(fields)
    else:
        output = frame.copy()

    output = output.rename(columns={"Adj Close": "AdjClose"})
    output = output.dropna(how="all")

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in output.columns:
            return pd.DataFrame()

    output = output[required].copy()
    output = output.dropna(subset=["Close"])

    return output


def _atr14(data: pd.DataFrame) -> pd.Series:
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(14, min_periods=14).mean()


def _risk_metrics(history: pd.DataFrame, ranking_row: Dict[str, Any]) -> Dict[str, Any]:
    fallback_close = _finite(ranking_row.get("close"))

    if history.empty or len(history) < 70:
        return {
            "close": fallback_close,
            "atr14": None,
            "ema21": None,
            "ema50": None,
            "velocity21": None,
            "annualized_volatility": 0.55,
            "dollar_volume_20": None,
            "below_ema21_count": 0,
            "drawdown_from_63d_high": None,
            "stop_level": None,
            "risk_state": "Data limited",
        }

    data = history.copy()
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)

    returns = close.pct_change()
    annualized_volatility = _finite(returns.tail(60).std() * math.sqrt(252))

    atr = _atr14(data)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    velocity21 = (ema21 - ema21.shift(5)) / atr
    dollar_volume_20 = (close * volume).rolling(20, min_periods=20).mean()

    latest_close = _finite(close.iloc[-1])
    latest_atr = _finite(atr.iloc[-1])
    latest_ema21 = _finite(ema21.iloc[-1])
    latest_ema50 = _finite(ema50.iloc[-1])
    latest_velocity21 = _finite(velocity21.iloc[-1])
    latest_dollar_volume = _finite(dollar_volume_20.iloc[-1])

    below_ema21_count = 0

    for offset in [1, 2, 3]:
        if len(close) <= offset:
            break

        close_value = _finite(close.iloc[-offset])
        ema_value = _finite(ema21.iloc[-offset])

        if close_value is not None and ema_value is not None and close_value < ema_value:
            below_ema21_count += 1
        else:
            break

    recent_high = _finite(close.tail(63).max())
    drawdown = None

    if latest_close is not None and recent_high is not None and recent_high > 0:
        drawdown = (latest_close - recent_high) / recent_high

    stop_level = None

    if latest_close is not None and latest_atr is not None:
        stop_level = latest_close - ATR_STOP_MULTIPLE * latest_atr

    if (
        below_ema21_count >= 3
        and latest_velocity21 is not None
        and latest_velocity21 < 0
    ):
        risk_state = "Trend break"
    elif drawdown is not None and drawdown <= -0.18:
        risk_state = "Drawdown risk"
    elif latest_close is not None and latest_ema21 is not None and latest_close > latest_ema21:
        risk_state = "Trend supported"
    elif latest_close is not None and latest_ema50 is not None and latest_close > latest_ema50:
        risk_state = "Base supported"
    else:
        risk_state = "Neutral"

    return {
        "close": latest_close,
        "atr14": latest_atr,
        "ema21": latest_ema21,
        "ema50": latest_ema50,
        "velocity21": latest_velocity21,
        "annualized_volatility": annualized_volatility or 0.55,
        "dollar_volume_20": latest_dollar_volume,
        "below_ema21_count": below_ema21_count,
        "drawdown_from_63d_high": drawdown,
        "stop_level": stop_level,
        "risk_state": risk_state,
    }


def _calculate_alpha(row: Dict[str, Any], risk: Dict[str, Any]) -> float:
    composite = _finite(row.get("stock_intelligence_score")) or 0.0
    fundamental = _finite(row.get("fundamental_score")) or 0.0
    technical = _finite(row.get("technical_score")) or 0.0
    balance = _finite(row.get("balance_sheet_score")) or 0.0
    valuation = _finite(row.get("valuation_score")) or 0.0

    trend_adjustment = 0.0

    if risk.get("risk_state") == "Trend supported":
        trend_adjustment += 4.0
    elif risk.get("risk_state") == "Base supported":
        trend_adjustment += 2.0
    elif risk.get("risk_state") == "Trend break":
        trend_adjustment -= 8.0
    elif risk.get("risk_state") == "Drawdown risk":
        trend_adjustment -= 5.0

    velocity21 = _finite(risk.get("velocity21"))

    if velocity21 is not None and velocity21 > 0:
        trend_adjustment += 2.0

    alpha = (
        composite * 0.48
        + technical * 0.18
        + fundamental * 0.15
        + balance * 0.10
        + valuation * 0.06
        + trend_adjustment
    )

    return _clamp(alpha)


def _eligible(row: Dict[str, Any], risk: Dict[str, Any], min_score: float) -> Tuple[bool, str]:
    composite = _finite(row.get("stock_intelligence_score")) or 0.0
    technical = _finite(row.get("technical_score")) or 0.0
    balance = _finite(row.get("balance_sheet_score")) or 0.0
    close = _finite(risk.get("close")) or _finite(row.get("close")) or 0.0
    dollar_volume_20 = _finite(risk.get("dollar_volume_20"))

    if composite < min_score:
        return False, "Composite score below threshold"

    if composite < 50:
        return False, "Composite score below hard minimum"

    if technical < 42:
        return False, "Technical score below soft minimum"

    if balance < 35:
        return False, "Balance sheet score below soft minimum"

    if close < 5:
        return False, "Price below hard minimum"

    if dollar_volume_20 is not None and dollar_volume_20 < 8_000_000:
        return False, "20D dollar volume below $8M"

    return True, "Eligible"


def _stock_exposure(holdings: List[Dict[str, Any]]) -> Tuple[float, str]:
    if not holdings:
        return 0.0, "No qualifying stock exposure"

    average_score = sum(
        _finite(item.get("stock_intelligence_score")) or 0.0
        for item in holdings
    ) / len(holdings)

    average_technical = sum(
        _finite(item.get("technical_score")) or 0.0
        for item in holdings
    ) / len(holdings)

    if average_score >= 76 and average_technical >= 62:
        return 0.95, "Full stock risk"

    if average_score >= 70 and average_technical >= 57:
        return 0.88, "Constructive stock risk"

    if average_score >= 63:
        return 0.74, "Moderate stock risk"

    return 0.58, "Reduced stock risk"


def _normalize_with_caps(
    holdings: List[Dict[str, Any]],
    target_exposure: float,
) -> List[Dict[str, Any]]:
    if not holdings:
        return []

    for item in holdings:
        alpha_edge = max(0.01, (_finite(item.get("portfolio_alpha")) or 55.0) - 55.0)
        volatility = max(0.18, _finite(item.get("annualized_volatility")) or 0.55)
        item["raw_weight"] = alpha_edge / (volatility ** 2)

    raw_total = sum(_finite(item.get("raw_weight")) or 0.0 for item in holdings)

    if raw_total <= 0:
        equal_weight = target_exposure / len(holdings)

        for item in holdings:
            item["target_weight"] = equal_weight
    else:
        for item in holdings:
            item["target_weight"] = target_exposure * item["raw_weight"] / raw_total

    # Single-name caps with redistribution.
    for _ in range(8):
        excess = 0.0
        receivers = []

        for item in holdings:
            weight = _finite(item.get("target_weight")) or 0.0

            if weight > MAX_SINGLE_POSITION:
                item["target_weight"] = MAX_SINGLE_POSITION
                excess += weight - MAX_SINGLE_POSITION
            else:
                receivers.append(item)

        if excess <= 1e-8 or not receivers:
            break

        receiver_total = sum(_finite(item.get("target_weight")) or 0.0 for item in receivers)

        if receiver_total <= 0:
            break

        for item in receivers:
            current = _finite(item.get("target_weight")) or 0.0
            item["target_weight"] = current + excess * current / receiver_total

    # Soft sector caps.
    for _ in range(5):
        sector_weights: Dict[str, float] = {}

        for item in holdings:
            sector = item.get("sector") or "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0.0) + (_finite(item.get("target_weight")) or 0.0)

        excess = 0.0

        for sector, weight in sector_weights.items():
            if weight <= MAX_SECTOR_WEIGHT:
                continue

            trim_ratio = MAX_SECTOR_WEIGHT / weight

            for item in holdings:
                if (item.get("sector") or "Unknown") == sector:
                    old_weight = _finite(item.get("target_weight")) or 0.0
                    item["target_weight"] = old_weight * trim_ratio
                    excess += old_weight - item["target_weight"]

        receivers = [
            item
            for item in holdings
            if sector_weights.get(item.get("sector") or "Unknown", 0.0) < MAX_SECTOR_WEIGHT
            and (_finite(item.get("target_weight")) or 0.0) < MAX_SINGLE_POSITION
        ]

        if excess <= 1e-8 or not receivers:
            break

        receiver_total = sum(_finite(item.get("target_weight")) or 0.0 for item in receivers)

        if receiver_total <= 0:
            break

        for item in receivers:
            current = _finite(item.get("target_weight")) or 0.0
            item["target_weight"] = min(
                MAX_SINGLE_POSITION,
                current + excess * current / receiver_total,
            )

    holdings = [
        item
        for item in holdings
        if (_finite(item.get("target_weight")) or 0.0) >= MIN_SINGLE_POSITION
    ]

    weight_total = sum(_finite(item.get("target_weight")) or 0.0 for item in holdings)

    if weight_total > 0:
        scale = min(1.0, target_exposure / weight_total)

        # If cap math leaves unused cash, keep cash rather than forcing caps to break.
        for item in holdings:
            item["target_weight"] = (_finite(item.get("target_weight")) or 0.0) * scale

    return holdings


def _action_for_holding(item: Dict[str, Any]) -> str:
    score = _finite(item.get("stock_intelligence_score")) or 0.0
    technical = _finite(item.get("technical_score")) or 0.0
    risk_state = item.get("risk_state")

    if risk_state == "Trend break" or technical < 45 or score < 58:
        return "Watch"

    if risk_state == "Drawdown risk":
        return "Trim"

    if item.get("portfolio_rank", 999) <= 5 and score >= 72 and technical >= 55:
        return "Buy"

    if score >= 65:
        return "Hold"

    return "Watch"


def _trade_reason(item: Dict[str, Any]) -> str:
    action = item.get("action")

    if action == "Buy":
        return "Top-ranked workbook profile with enough technical support to take stock risk."

    if action == "Hold":
        return "Score remains inside the target stock portfolio range."

    if action == "Trim":
        return "Still qualifies, but drawdown or trend risk has increased."

    if action == "Sell":
        return "Score or trend risk has broken portfolio rules."

    return "Keep on the watchlist until score or trend confirmation improves."


def _current_holdings(
    ranking_rows: List[Dict[str, Any]],
    price_frame: pd.DataFrame,
    target_holdings: int,
    min_score: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    candidates = []
    rejected = []

    for index, row in enumerate(ranking_rows):
        ticker = row.get("ticker")

        if not ticker:
            continue

        history = _slice_history(price_frame, ticker)
        risk = _risk_metrics(history, row)
        eligible, reason = _eligible(row=row, risk=risk, min_score=min_score)
        alpha = _calculate_alpha(row, risk)

        item = {
            **row,
            **risk,
            "rank": index + 1,
            "eligible": eligible,
            "eligibility_reason": reason,
            "portfolio_alpha": alpha,
        }

        if eligible:
            candidates.append(item)
        else:
            rejected.append(item)

    candidates.sort(
        key=lambda item: _finite(item.get("portfolio_alpha")) or 0.0,
        reverse=True,
    )

    used_fallback = False

    if len(candidates) < max(3, min(target_holdings, 6)):
        fallback = [
            item
            for item in rejected
            if (_finite(item.get("stock_intelligence_score")) or 0.0) >= 50
            and (_finite(item.get("close")) or 10.0) >= 5
        ]

        fallback.sort(
            key=lambda item: (
                _finite(item.get("stock_intelligence_score")) or 0.0,
                _finite(item.get("portfolio_alpha")) or 0.0,
            ),
            reverse=True,
        )

        existing = {item.get("ticker") for item in candidates}

        for item in fallback:
            if item.get("ticker") not in existing:
                item["eligible"] = True
                item["eligibility_reason"] = "Fallback inclusion from top workbook rank"
                candidates.append(item)
                existing.add(item.get("ticker"))

            if len(candidates) >= target_holdings:
                break

        used_fallback = True

    selected = candidates[:target_holdings]
    exposure, exposure_regime = _stock_exposure(selected)
    holdings = _normalize_with_caps(selected, exposure)

    holdings.sort(
        key=lambda item: _finite(item.get("target_weight")) or 0.0,
        reverse=True,
    )

    for index, item in enumerate(holdings):
        item["portfolio_rank"] = index + 1
        item["action"] = _action_for_holding(item)
        item["trade_reason"] = _trade_reason(item)
        item["sell_trigger"] = (
            "Review or remove if composite < 60, technical < 45, or price "
            "stays below EMA21 for 3 sessions while Velocity21 is negative."
        )
        item["add_trigger"] = (
            "Add only if the stock remains top ranked and price stays above EMA21."
        )
        item["hold_window"] = "8 to 12 weeks for quality leaders; review weekly."

    return holdings, rejected, used_fallback


def _score_at_date(
    history: pd.DataFrame,
    row: Dict[str, Any],
    position: int,
) -> Optional[Dict[str, Any]]:
    if history.empty or position < 130:
        return None

    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)

    current_close = _finite(close.iloc[position])

    if current_close is None or current_close <= 5:
        return None

    ret_1m = _safe_divide(close.iloc[position] - close.iloc[position - 21], close.iloc[position - 21])
    ret_3m = _safe_divide(close.iloc[position] - close.iloc[position - 63], close.iloc[position - 63])
    ret_6m = _safe_divide(close.iloc[position] - close.iloc[position - 126], close.iloc[position - 126])

    ema21 = close.ewm(span=21, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma100 = close.rolling(100, min_periods=100).mean()

    ema21_value = _finite(ema21.iloc[position])
    sma50_value = _finite(sma50.iloc[position])
    sma100_value = _finite(sma100.iloc[position])

    returns = close.pct_change()
    vol = _finite(returns.iloc[max(0, position - 60):position + 1].std() * math.sqrt(252)) or 0.55
    dollar_volume = _finite((close * volume).rolling(20, min_periods=20).mean().iloc[position])

    if dollar_volume is not None and dollar_volume < 5_000_000:
        return None

    trend_score = 50.0

    if ema21_value is not None:
        trend_score += 10 if current_close > ema21_value else -8

    if sma50_value is not None:
        trend_score += 10 if current_close > sma50_value else -10

    if sma100_value is not None:
        trend_score += 8 if current_close > sma100_value else -8

    momentum_score = 50.0

    if ret_1m is not None:
        momentum_score += max(-12, min(16, ret_1m * 70))

    if ret_3m is not None:
        momentum_score += max(-15, min(22, ret_3m * 55))

    if ret_6m is not None:
        momentum_score += max(-12, min(18, ret_6m * 35))

    if vol > 0.70:
        momentum_score -= 6

    if vol > 1.00:
        momentum_score -= 8

    quality_score = (
        (_finite(row.get("fundamental_score")) or 55.0) * 0.35
        + (_finite(row.get("balance_sheet_score")) or 55.0) * 0.30
        + (_finite(row.get("valuation_score")) or 55.0) * 0.10
        + (_finite(row.get("stock_intelligence_score")) or 55.0) * 0.25
    )

    model_score = _clamp(
        momentum_score * 0.44
        + trend_score * 0.26
        + quality_score * 0.30
    )

    alpha_edge = max(0.01, model_score - 55)
    raw_weight = alpha_edge / (max(0.18, vol) ** 2)

    return {
        "ticker": row.get("ticker"),
        "score": model_score,
        "raw_weight": raw_weight,
        "volatility": vol,
        "return_3m": ret_3m,
        "return_6m": ret_6m,
        "sector": row.get("sector") or "Unknown",
    }


def _weights_for_rebalance(
    date_position: int,
    trading_index: pd.Index,
    histories: Dict[str, pd.DataFrame],
    ranking_lookup: Dict[str, Dict[str, Any]],
    target_holdings: int,
) -> Dict[str, float]:
    scores = []

    for ticker, history in histories.items():
        if ticker == BENCHMARK_TICKER:
            continue

        if history.empty:
            continue

        # Convert rebalance date into this ticker's local position.
        date = trading_index[date_position]

        try:
            local_position = history.index.get_indexer([date], method="pad")[0]
        except Exception:
            continue

        if local_position < 130:
            continue

        row = ranking_lookup.get(ticker, {"ticker": ticker})
        score = _score_at_date(history, row, local_position)

        if score is not None:
            scores.append(score)

    scores.sort(
        key=lambda item: item.get("score") or 0.0,
        reverse=True,
    )

    selected = scores[:target_holdings]

    if not selected:
        return {}

    raw_total = sum(item["raw_weight"] for item in selected)

    if raw_total <= 0:
        return {
            item["ticker"]: 0.90 / len(selected)
            for item in selected
        }

    weights = {
        item["ticker"]: 0.92 * item["raw_weight"] / raw_total
        for item in selected
    }

    for _ in range(8):
        excess = 0.0
        receivers = []

        for ticker, weight in list(weights.items()):
            if weight > MAX_SINGLE_POSITION:
                weights[ticker] = MAX_SINGLE_POSITION
                excess += weight - MAX_SINGLE_POSITION
            else:
                receivers.append(ticker)

        if excess <= 1e-8 or not receivers:
            break

        receiver_total = sum(weights[ticker] for ticker in receivers)

        if receiver_total <= 0:
            break

        for ticker in receivers:
            weights[ticker] += excess * weights[ticker] / receiver_total

    return weights


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    peak = values[0]
    max_dd = 0.0

    for value in values:
        peak = max(peak, value)

        if peak > 0:
            drawdown = (value - peak) / peak
            max_dd = min(max_dd, drawdown)

    return max_dd


def _annualized_volatility(returns: Sequence[float]) -> float:
    if len(returns) < 5:
        return 0.0

    return float(pd.Series(returns).std() * math.sqrt(252))


def _rebalance_actions(old_weights: Dict[str, float], new_weights: Dict[str, float]) -> Dict[str, Any]:
    old_keys = set(old_weights.keys())
    new_keys = set(new_weights.keys())

    buys = sorted(list(new_keys - old_keys))
    sells = sorted(list(old_keys - new_keys))

    adds = []
    trims = []

    for ticker in sorted(list(old_keys & new_keys)):
        old_weight = old_weights.get(ticker, 0.0)
        new_weight = new_weights.get(ticker, 0.0)

        if new_weight - old_weight > 0.015:
            adds.append(ticker)
        elif old_weight - new_weight > 0.015:
            trims.append(ticker)

    headline_parts = []

    if buys:
        headline_parts.append("Bought " + ", ".join(buys[:3]))

    if sells:
        headline_parts.append("Removed " + ", ".join(sells[:3]))

    if not headline_parts and adds:
        headline_parts.append("Added to " + ", ".join(adds[:3]))

    if not headline_parts and trims:
        headline_parts.append("Trimmed " + ", ".join(trims[:3]))

    if not headline_parts:
        headline_parts.append("No major changes")

    return {
        "headline": "; ".join(headline_parts),
        "buys": buys,
        "sells": sells,
        "adds": adds,
        "trims": trims,
        "turnover": 0.5 * sum(
            abs(new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0))
            for ticker in sorted(list(old_keys | new_keys))
        ),
    }


def _simulate_performance(
    ranking_rows: List[Dict[str, Any]],
    price_frame: pd.DataFrame,
    target_holdings: int,
) -> Dict[str, Any]:
    benchmark_history = _slice_history(price_frame, BENCHMARK_TICKER)

    if benchmark_history.empty or len(benchmark_history) < 180:
        return {
            "series": [],
            "rebalance_log": [],
            "stats": {},
            "diagnostics": {
                "reason": "Benchmark history was unavailable.",
            },
        }

    histories: Dict[str, pd.DataFrame] = {}
    ranking_lookup: Dict[str, Dict[str, Any]] = {}

    for row in ranking_rows:
        ticker = row.get("ticker")

        if not ticker:
            continue

        history = _slice_history(price_frame, ticker)

        if not history.empty:
            histories[ticker] = history
            ranking_lookup[ticker] = row

    trading_index = benchmark_history.index
    start_position = max(130, len(trading_index) - 252)

    if start_position >= len(trading_index) - 10:
        start_position = max(0, len(trading_index) - 126)

    benchmark_close = benchmark_history["Close"].astype(float)
    benchmark_returns = benchmark_close.pct_change().fillna(0)

    model_value = 1.0
    benchmark_start = _finite(benchmark_close.iloc[start_position]) or 1.0
    weights: Dict[str, float] = {}
    model_values = []
    benchmark_values = []
    model_returns = []
    benchmark_daily_returns = []
    series = []
    rebalance_log = []

    rebalanced_this_period = False

    for position in range(start_position, len(trading_index)):
        date = trading_index[position]

        if position == start_position or (position - start_position) % 5 == 0:
            new_weights = _weights_for_rebalance(
                date_position=position,
                trading_index=trading_index,
                histories=histories,
                ranking_lookup=ranking_lookup,
                target_holdings=target_holdings,
            )

            if new_weights:
                actions = _rebalance_actions(weights, new_weights)
                turnover = actions["turnover"]

                if weights:
                    model_value *= max(0.0, 1.0 - turnover * TRANSACTION_COST_BPS / 10000.0)

                top_holdings = sorted(
                    [
                        {
                            "ticker": ticker,
                            "weight": weight,
                        }
                        for ticker, weight in new_weights.items()
                    ],
                    key=lambda item: item["weight"],
                    reverse=True,
                )

                rebalance_log.append(
                    {
                        "date": pd.Timestamp(date).date().isoformat(),
                        "headline": actions["headline"],
                        "turnover": turnover,
                        "buys": actions["buys"],
                        "sells": actions["sells"],
                        "adds": actions["adds"],
                        "trims": actions["trims"],
                        "holdings": top_holdings[:12],
                    }
                )

                weights = new_weights
                rebalanced_this_period = True

        daily_return = 0.0

        if weights and position > 0:
            for ticker, weight in weights.items():
                history = histories.get(ticker)

                if history is None or history.empty:
                    continue

                try:
                    local_position = history.index.get_indexer([date], method="pad")[0]
                except Exception:
                    continue

                if local_position <= 0:
                    continue

                close = history["Close"].astype(float)
                today = _finite(close.iloc[local_position])
                yesterday = _finite(close.iloc[local_position - 1])

                if today is None or yesterday is None or yesterday == 0:
                    continue

                daily_return += weight * ((today / yesterday) - 1.0)

        model_value *= (1.0 + daily_return)

        benchmark_value = (_finite(benchmark_close.iloc[position]) or benchmark_start) / benchmark_start
        benchmark_return = _finite(benchmark_returns.iloc[position]) or 0.0

        model_values.append(model_value)
        benchmark_values.append(benchmark_value)
        model_returns.append(daily_return)
        benchmark_daily_returns.append(benchmark_return)

        rebalance_marker = None

        if rebalance_log and rebalance_log[-1]["date"] == pd.Timestamp(date).date().isoformat():
            rebalance_marker = {
                "date": rebalance_log[-1]["date"],
                "headline": rebalance_log[-1]["headline"],
            }

        series.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "model": model_value - 1.0,
                "benchmark": benchmark_value - 1.0,
                "rebalance": rebalance_marker,
            }
        )

    stats = {
        "model_return": model_values[-1] - 1.0 if model_values else 0.0,
        "benchmark_return": benchmark_values[-1] - 1.0 if benchmark_values else 0.0,
        "model_volatility": _annualized_volatility(model_returns),
        "benchmark_volatility": _annualized_volatility(benchmark_daily_returns),
        "model_max_drawdown": _max_drawdown(model_values),
        "benchmark_max_drawdown": _max_drawdown(benchmark_values),
        "rebalance_count": len(rebalance_log),
    }

    return {
        "series": series,
        "rebalance_log": rebalance_log[-14:],
        "stats": stats,
        "diagnostics": {
            "rebalanced": rebalanced_this_period,
            "history_count": len(histories),
            "benchmark": BENCHMARK_TICKER,
        },
    }


def _build_portfolio(
    universe: str,
    tickers: Optional[str],
    max_tickers: int,
    target_holdings: int,
    min_score: float,
) -> Dict[str, Any]:
    if _rankings_get_universe is None or _rankings_scan is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Stock portfolio requires api/routers/stock_rankings.py. "
                "Deploy stock_rankings.py and include it in api/main.py first."
            ),
        )

    target_holdings = max(
        MIN_TARGET_HOLDINGS,
        min(MAX_TARGET_HOLDINGS, int(target_holdings)),
    )

    universe_key, ticker_list = _rankings_get_universe(
        universe=universe,
        tickers=tickers,
        max_tickers=max_tickers,
    )

    ranking_payload = _rankings_scan(
        universe_key=universe_key,
        tickers=ticker_list,
        limit=max(45, target_holdings * 4),
        min_score=0,
    )

    ranking_rows = ranking_payload.get("rows", [])

    if not ranking_rows:
        generated_at = datetime.utcnow().isoformat() + "Z"

        return {
            "generated_at": generated_at,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "portfolio_type": "stock_alpha",
            "universe": universe_key,
            "requested_tickers": len(ticker_list),
            "ranked_candidates": 0,
            "eligible_candidates": 0,
            "target_holdings": target_holdings,
            "stock_exposure": 0.0,
            "cash_weight": 1.0,
            "exposure_regime": "No ranking data",
            "holdings": [],
            "sector_weights": {},
            "trade_queue": [],
            "performance": {
                "series": [],
                "rebalance_log": [],
                "stats": {},
            },
            "diagnostics": {
                "reason": "The stock rankings API returned no rows.",
            },
        }

    history_tickers = [
        row["ticker"]
        for row in ranking_rows
        if row.get("ticker")
    ]
    history_tickers.append(BENCHMARK_TICKER)

    price_frame = _download_history(history_tickers, period="2y")

    holdings, rejected, used_fallback = _current_holdings(
        ranking_rows=ranking_rows,
        price_frame=price_frame,
        target_holdings=target_holdings,
        min_score=min_score,
    )

    exposure, exposure_regime = _stock_exposure(holdings)
    invested_weight = sum(_finite(item.get("target_weight")) or 0.0 for item in holdings)
    cash_weight = max(0.0, 1.0 - invested_weight)

    sector_weights: Dict[str, float] = {}

    for item in holdings:
        sector = item.get("sector") or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0.0) + (_finite(item.get("target_weight")) or 0.0)

    trade_queue = [
        {
            "ticker": item.get("ticker"),
            "action": item.get("action"),
            "target_weight": item.get("target_weight"),
            "portfolio_alpha": item.get("portfolio_alpha"),
            "stock_intelligence_score": item.get("stock_intelligence_score"),
            "reason": item.get("trade_reason"),
        }
        for item in holdings
        if item.get("action") in ["Buy", "Trim", "Watch", "Sell"]
    ]

    performance = _simulate_performance(
        ranking_rows=ranking_rows,
        price_frame=price_frame,
        target_holdings=target_holdings,
    )

    generated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "generated_at": generated_at,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "portfolio_type": "stock_alpha",
        "universe": universe_key,
        "requested_tickers": len(ticker_list),
        "ranked_candidates": len(ranking_rows),
        "eligible_candidates": len(holdings),
        "target_holdings": target_holdings,
        "stock_exposure": exposure,
        "cash_weight": cash_weight,
        "exposure_regime": exposure_regime,
        "used_fallback": used_fallback,
        "holdings": [
            {
                key: _clean(value)
                for key, value in item.items()
                if key != "raw_weight"
            }
            for item in holdings
        ],
        "sector_weights": {
            sector: _clean(weight)
            for sector, weight in sorted(
                sector_weights.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )
        },
        "trade_queue": [
            {
                key: _clean(value)
                for key, value in item.items()
            }
            for item in trade_queue
        ],
        "performance": {
            "series": performance.get("series", []),
            "rebalance_log": performance.get("rebalance_log", []),
            "stats": {
                key: _clean(value)
                for key, value in performance.get("stats", {}).items()
            },
            "benchmark": BENCHMARK_TICKER,
        },
        "diagnostics": {
            "ranking_rows": len(ranking_rows),
            "rejected_candidates": len(rejected),
            "used_fallback": used_fallback,
            "performance": performance.get("diagnostics", {}),
            "note": "Portfolio uses weekly model rebalances and live current holdings.",
        },
        "risk_rules": {
            "max_single_position": MAX_SINGLE_POSITION,
            "max_sector_weight": MAX_SECTOR_WEIGHT,
            "atr_stop_multiple": ATR_STOP_MULTIPLE,
            "hard_sell": [
                "Composite score below 60",
                "Technical score below 45",
                "Close below EMA21 for 3 sessions with negative Velocity21",
                "Trend break risk state",
            ],
            "rebalance": "Weekly target rebalance; daily risk checks.",
        },
        "methodology": {
            "objective": (
                "Maximize expected stock alpha while penalizing volatility, "
                "concentration, turnover, and trend-break risk."
            ),
            "sizing": (
                "Current holdings use workbook alpha divided by volatility squared, "
                "then position and sector caps."
            ),
            "formula": "raw_weight_i = max(alpha_i - 55, 0.01) / vol_i^2",
        },
    }


@router.get("")
def get_stock_portfolio(
    universe: str = Query(default="quality"),
    target_holdings: int = Query(default=DEFAULT_TARGET_HOLDINGS, ge=MIN_TARGET_HOLDINGS, le=MAX_TARGET_HOLDINGS),
    min_score: float = Query(default=60, ge=0, le=100),
    max_tickers: int = Query(default=95, ge=20, le=160),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    cache_key = (
        f"stock-portfolio-living-v1:{universe}:{target_holdings}:{min_score}:"
        f"{max_tickers}:{tickers or ''}"
    )

    if not refresh:
        cached = _cache_get(cache_key)

        if cached is not None:
            return {
                **cached,
                "cached": True,
            }

    payload = _build_portfolio(
        universe=universe,
        tickers=tickers,
        max_tickers=max_tickers,
        target_holdings=target_holdings,
        min_score=min_score,
    )

    payload["cached"] = False

    return _cache_set(cache_key, payload, CACHE_TTL_SECONDS)


@router.get("/status")
def get_stock_portfolio_status() -> Dict[str, Any]:
    rankings_available = _rankings_get_universe is not None and _rankings_scan is not None

    return {
        "status": "ok",
        "route": "/api/stock-portfolio",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "target_holdings_default": DEFAULT_TARGET_HOLDINGS,
        "max_single_position": MAX_SINGLE_POSITION,
        "max_sector_weight": MAX_SECTOR_WEIGHT,
        "benchmark": BENCHMARK_TICKER,
        "requires": "/api/stock-rankings",
        "stock_rankings_imported": rankings_available,
        "returns": [
            "holdings",
            "sector_weights",
            "trade_queue",
            "performance.series",
            "performance.stats",
            "performance.rebalance_log",
        ],
    }
