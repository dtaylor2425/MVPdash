"""
Stock Alpha Portfolio API

FastAPI routes:
    GET /api/stock-portfolio
    GET /api/stock-portfolio/status

Purpose:
    Build a separate stock-only portfolio from Macro Engine's ranked Stock
    Intelligence universe.

This is separate from the ETF portfolio. The stock portfolio uses the workbook
scores as the source of conviction, then translates those scores into position
weights, risk levels, sell triggers, and a trade queue.

Core principles:
    1. The workbook scores decide what deserves capital.
    2. Volatility-adjusted sizing decides how much capital.
    3. EMA/ATR rules decide when risk is failing.
    4. The stock portfolio is separate from the ETF portfolio.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query

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
MAX_TARGET_HOLDINGS = 15
MIN_TARGET_HOLDINGS = 6
MAX_SINGLE_POSITION = 0.12
MIN_SINGLE_POSITION = 0.025
MAX_SECTOR_WEIGHT = 0.32
ATR_STOP_MULTIPLE = 2.20


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

    if pd.isna(value):
        return None

    return value


def _safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    top = _finite(numerator)
    bottom = _finite(denominator)

    if top is None or bottom is None or abs(bottom) < 1e-12:
        return None

    return top / bottom


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _download_price_history(tickers: Sequence[str]) -> pd.DataFrame:
    try:
        return yf.download(
            tickers=list(tickers),
            period="1y",
            interval="1d",
            auto_adjust=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except TypeError:
        return yf.download(
            tickers=list(tickers),
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )


def _slice_ticker_history(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
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

    return output.dropna(subset=required)


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


def _risk_metrics(history: pd.DataFrame) -> Dict[str, Any]:
    if history.empty or len(history) < 80:
        return {
            "close": None,
            "atr14": None,
            "ema21": None,
            "ema50": None,
            "velocity21": None,
            "annualized_volatility": 0.65,
            "dollar_volume_20": None,
            "below_ema21_count": 0,
            "drawdown_from_63d_high": None,
            "stop_level": None,
            "risk_state": "Unknown",
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
    elif drawdown is not None and drawdown <= -0.15:
        risk_state = "Drawdown risk"
    elif latest_close is not None and latest_ema21 is not None and latest_close > latest_ema21:
        risk_state = "Trend supported"
    else:
        risk_state = "Neutral"

    return {
        "close": latest_close,
        "atr14": latest_atr,
        "ema21": latest_ema21,
        "ema50": latest_ema50,
        "velocity21": latest_velocity21,
        "annualized_volatility": annualized_volatility or 0.65,
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

    trend_bonus = 0.0

    if risk.get("risk_state") == "Trend supported":
        trend_bonus += 4

    velocity21 = _finite(risk.get("velocity21"))

    if velocity21 is not None and velocity21 > 0:
        trend_bonus += 3

    if risk.get("risk_state") in ["Trend break", "Drawdown risk"]:
        trend_bonus -= 10

    alpha = (
        composite * 0.45
        + technical * 0.20
        + fundamental * 0.15
        + balance * 0.10
        + valuation * 0.05
        + trend_bonus
    )

    return _clamp(alpha)


def _eligible(row: Dict[str, Any], risk: Dict[str, Any], min_score: float) -> Tuple[bool, str]:
    composite = _finite(row.get("stock_intelligence_score")) or 0.0
    technical = _finite(row.get("technical_score")) or 0.0
    balance = _finite(row.get("balance_sheet_score")) or 0.0
    close = _finite(risk.get("close")) or _finite(row.get("close")) or 0.0
    dollar_volume_20 = _finite(risk.get("dollar_volume_20")) or 0.0

    if composite < min_score:
        return False, "Composite score below threshold"

    if technical < 50:
        return False, "Technical score below minimum"

    if balance < 45:
        return False, "Balance sheet score below minimum"

    if close < 10:
        return False, "Price below liquidity-quality threshold"

    if dollar_volume_20 < 20_000_000:
        return False, "20D dollar volume below $20M"

    if risk.get("risk_state") == "Trend break":
        return False, "Trend break risk"

    return True, "Eligible"


def _normalize_with_caps(
    holdings: List[Dict[str, Any]],
    target_exposure: float,
) -> List[Dict[str, Any]]:
    if not holdings:
        return []

    for item in holdings:
        alpha = max(0.01, item["portfolio_alpha"] - 55)
        volatility = max(0.18, item["annualized_volatility"] or 0.65)
        item["raw_weight"] = alpha / (volatility ** 2)

    total_raw = sum(item["raw_weight"] for item in holdings)

    if total_raw <= 0:
        equal_weight = target_exposure / len(holdings)

        for item in holdings:
            item["target_weight"] = equal_weight

        return holdings

    for item in holdings:
        item["target_weight"] = target_exposure * item["raw_weight"] / total_raw

    # Position caps.
    for _ in range(8):
        excess = 0.0
        uncapped = []

        for item in holdings:
            if item["target_weight"] > MAX_SINGLE_POSITION:
                excess += item["target_weight"] - MAX_SINGLE_POSITION
                item["target_weight"] = MAX_SINGLE_POSITION
            else:
                uncapped.append(item)

        if excess <= 1e-8 or not uncapped:
            break

        uncapped_total = sum(item["target_weight"] for item in uncapped)

        if uncapped_total <= 0:
            break

        for item in uncapped:
            item["target_weight"] += excess * item["target_weight"] / uncapped_total

    # Sector caps.
    for _ in range(6):
        sector_weights: Dict[str, float] = {}

        for item in holdings:
            sector = item.get("sector") or "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0.0) + item["target_weight"]

        excess = 0.0

        for sector, weight in sector_weights.items():
            if weight <= MAX_SECTOR_WEIGHT:
                continue

            trim_ratio = MAX_SECTOR_WEIGHT / weight
            for item in holdings:
                if item.get("sector") == sector:
                    old_weight = item["target_weight"]
                    item["target_weight"] = old_weight * trim_ratio
                    excess += old_weight - item["target_weight"]

        receivers = [
            item
            for item in holdings
            if sector_weights.get(item.get("sector") or "Unknown", 0.0) < MAX_SECTOR_WEIGHT
            and item["target_weight"] < MAX_SINGLE_POSITION
        ]

        if excess <= 1e-8 or not receivers:
            break

        receiver_total = sum(item["target_weight"] for item in receivers)

        if receiver_total <= 0:
            break

        for item in receivers:
            item["target_weight"] += excess * item["target_weight"] / receiver_total
            item["target_weight"] = min(item["target_weight"], MAX_SINGLE_POSITION)

    # Drop tiny weights and renormalize remaining weights to target exposure.
    holdings = [
        item
        for item in holdings
        if item["target_weight"] >= MIN_SINGLE_POSITION
    ]

    total_weight = sum(item["target_weight"] for item in holdings)

    if total_weight > 0:
        scale = target_exposure / total_weight

        for item in holdings:
            item["target_weight"] *= scale
            item["target_weight"] = min(item["target_weight"], MAX_SINGLE_POSITION)

    return holdings


def _stock_exposure(top_rows: List[Dict[str, Any]]) -> Tuple[float, str]:
    if not top_rows:
        return 0.0, "No qualifying stock exposure"

    avg_score = sum(
        _finite(row.get("stock_intelligence_score")) or 0.0
        for row in top_rows
    ) / len(top_rows)

    avg_technical = sum(
        _finite(row.get("technical_score")) or 0.0
        for row in top_rows
    ) / len(top_rows)

    favored_count = sum(
        1
        for row in top_rows
        if (_finite(row.get("stock_intelligence_score")) or 0.0) >= 75
    )

    if avg_score >= 76 and avg_technical >= 65 and favored_count >= 6:
        return 0.95, "Full stock risk"

    if avg_score >= 70 and avg_technical >= 60 and favored_count >= 4:
        return 0.88, "Constructive stock risk"

    if avg_score >= 64 and avg_technical >= 55:
        return 0.72, "Moderate stock risk"

    return 0.55, "Reduced stock risk"


def _action_for_holding(item: Dict[str, Any]) -> str:
    score = _finite(item.get("stock_intelligence_score")) or 0.0
    technical = _finite(item.get("technical_score")) or 0.0
    risk_state = item.get("risk_state")

    if risk_state == "Trend break":
        return "Sell"

    if score < 60 or technical < 45:
        return "Sell"

    if risk_state == "Drawdown risk":
        return "Trim"

    if item.get("rank", 999) <= 5 and score >= 75 and technical >= 60:
        return "Buy"

    if score >= 68 and technical >= 55:
        return "Hold"

    return "Watch"


def _trade_reason(item: Dict[str, Any]) -> str:
    action = item.get("action")

    if action == "Buy":
        return "Top-ranked workbook profile with enough technical support to take risk."

    if action == "Hold":
        return "Score remains inside the target portfolio range."

    if action == "Trim":
        return "Ranking still qualifies, but price risk has increased."

    if action == "Sell":
        return "Score or trend risk has broken the portfolio rules."

    return "Needs either a stronger score or cleaner price confirmation."


def _build_portfolio(
    universe: str,
    tickers: Optional[str],
    max_tickers: int,
    target_holdings: int,
    min_score: float,
) -> Dict[str, Any]:
    if _rankings_get_universe is None or _rankings_scan is None:
        raise RuntimeError(
            "api.routers.stock_rankings is required before stock_portfolio can run."
        )

    target_holdings = max(
        MIN_TARGET_HOLDINGS,
        min(MAX_TARGET_HOLDINGS, target_holdings),
    )

    universe_key, ticker_list = _rankings_get_universe(
        universe=universe,
        tickers=tickers,
        max_tickers=max_tickers,
    )

    ranking_payload = _rankings_scan(
        universe_key=universe_key,
        tickers=ticker_list,
        limit=max(35, target_holdings * 3),
        min_score=0,
    )

    ranking_rows = ranking_payload.get("rows", [])

    tickers_for_history = [
        row["ticker"]
        for row in ranking_rows
        if row.get("ticker")
    ]

    price_frame = _download_price_history(tickers_for_history)

    candidates = []

    for index, row in enumerate(ranking_rows):
        ticker = row.get("ticker")
        history = _slice_ticker_history(price_frame, ticker)
        risk = _risk_metrics(history)

        is_eligible, eligibility_reason = _eligible(
            row=row,
            risk=risk,
            min_score=min_score,
        )

        alpha = _calculate_alpha(row, risk)

        item = {
            **row,
            **risk,
            "rank": index + 1,
            "eligible": is_eligible,
            "eligibility_reason": eligibility_reason,
            "portfolio_alpha": alpha,
        }

        if is_eligible:
            candidates.append(item)

    candidates.sort(
        key=lambda item: item.get("portfolio_alpha") or 0.0,
        reverse=True,
    )

    selected = candidates[:target_holdings]
    stock_exposure, exposure_regime = _stock_exposure(selected)
    holdings = _normalize_with_caps(selected, stock_exposure)

    holdings.sort(
        key=lambda item: item.get("target_weight") or 0.0,
        reverse=True,
    )

    for index, item in enumerate(holdings):
        item["portfolio_rank"] = index + 1
        item["action"] = _action_for_holding(item)
        item["trade_reason"] = _trade_reason(item)
        item["sell_trigger"] = (
            "Sell if composite < 60, technical < 45, or close stays below EMA21 "
            "for 3 days while Velocity21 is negative."
        )
        item["add_trigger"] = (
            "Add only if the stock remains top ranked and price stays above EMA21."
        )
        item["hold_window"] = (
            "8 to 12 weeks for quality leaders; review weekly."
        )

    cash_weight = max(
        0.0,
        1.0 - sum(item.get("target_weight") or 0.0 for item in holdings),
    )

    sector_weights: Dict[str, float] = {}

    for item in holdings:
        sector = item.get("sector") or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0.0) + (item.get("target_weight") or 0.0)

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
        if item.get("action") in ["Buy", "Trim", "Sell", "Watch"]
    ]

    generated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "generated_at": generated_at,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "portfolio_type": "stock_alpha",
        "universe": universe_key,
        "requested_tickers": len(ticker_list),
        "ranked_candidates": len(ranking_rows),
        "eligible_candidates": len(candidates),
        "target_holdings": target_holdings,
        "stock_exposure": stock_exposure,
        "cash_weight": cash_weight,
        "exposure_regime": exposure_regime,
        "holdings": [
            {
                key: _clean(value)
                for key, value in item.items()
                if key not in ["raw_weight"]
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
            "trim": [
                "Drawdown risk state",
                "Position exceeds max weight",
                "Score falls while price keeps rising",
            ],
            "rebalance": "Weekly target rebalance; daily risk checks.",
        },
        "methodology": {
            "objective": (
                "Maximize expected stock alpha while penalizing volatility, "
                "concentration, and trend-break risk."
            ),
            "ranking_signal": (
                "Stock Intelligence Score, technical support, fundamentals, "
                "balance sheet strength, valuation, and price risk."
            ),
            "sizing": (
                "weight_i is proportional to portfolio_alpha_edge divided by "
                "annualized_volatility squared, then capped by position and sector."
            ),
            "formula": "raw_weight_i = max(alpha_i - 55, 0.01) / vol_i^2",
        },
    }


@router.get("")
def get_stock_portfolio(
    universe: str = Query(default="quality"),
    target_holdings: int = Query(default=DEFAULT_TARGET_HOLDINGS, ge=MIN_TARGET_HOLDINGS, le=MAX_TARGET_HOLDINGS),
    min_score: float = Query(default=65, ge=0, le=100),
    max_tickers: int = Query(default=95, ge=20, le=160),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    cache_key = (
        f"stock-portfolio:{universe}:{target_holdings}:{min_score}:"
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
    return {
        "status": "ok",
        "route": "/api/stock-portfolio",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "target_holdings_default": DEFAULT_TARGET_HOLDINGS,
        "max_single_position": MAX_SINGLE_POSITION,
        "max_sector_weight": MAX_SECTOR_WEIGHT,
        "requires": "/api/stock-rankings",
    }
