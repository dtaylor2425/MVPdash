"""
Stock Alpha Portfolio API

Routes:
    GET /api/stock-portfolio
    GET /api/stock-portfolio/status

This version fixes the two common failure modes from the first stock-portfolio build:

1. Empty holdings when the risk filters are too strict.
   - The portfolio now has a soft fallback path.
   - It will still prefer liquid, high-score, trend-supported names.
   - But it will not return a blank portfolio just because Yahoo data is missing
     one risk field or because the filter set is temporarily too restrictive.

2. Fragile dependency behavior.
   - This router still uses the stock_rankings model as the source of truth.
   - It returns clearer errors if stock_rankings is not installed.
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
MIN_SINGLE_POSITION = 0.02
MAX_SECTOR_WEIGHT = 0.34
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

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _download_price_history(tickers: Sequence[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

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
    except Exception:
        return pd.DataFrame()


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

    return output.dropna(subset=["Close"])


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

    if history.empty or len(history) < 60:
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


def _hard_exclusion(row: Dict[str, Any], risk: Dict[str, Any]) -> Optional[str]:
    close = _finite(risk.get("close")) or _finite(row.get("close"))
    composite = _finite(row.get("stock_intelligence_score")) or 0.0

    if composite < 50:
        return "Composite score below hard minimum"

    if close is not None and close < 5:
        return "Price below hard minimum"

    return None


def _eligible(row: Dict[str, Any], risk: Dict[str, Any], min_score: float) -> Tuple[bool, str]:
    hard_exclusion = _hard_exclusion(row, risk)

    if hard_exclusion:
        return False, hard_exclusion

    composite = _finite(row.get("stock_intelligence_score")) or 0.0
    technical = _finite(row.get("technical_score")) or 0.0
    balance = _finite(row.get("balance_sheet_score")) or 0.0
    dollar_volume_20 = _finite(risk.get("dollar_volume_20"))

    if composite < min_score:
        return False, "Composite score below threshold"

    if technical < 42:
        return False, "Technical score below soft minimum"

    if balance < 35:
        return False, "Balance sheet score below soft minimum"

    if dollar_volume_20 is not None and dollar_volume_20 < 8_000_000:
        return False, "20D dollar volume below $8M"

    return True, "Eligible"


def _build_candidate_items(
    ranking_rows: List[Dict[str, Any]],
    price_frame: pd.DataFrame,
    min_score: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = []
    rejected = []

    for index, row in enumerate(ranking_rows):
        ticker = row.get("ticker")

        if not ticker:
            continue

        history = _slice_ticker_history(price_frame, ticker)
        risk = _risk_metrics(history, row)

        eligible, reason = _eligible(
            row=row,
            risk=risk,
            min_score=min_score,
        )

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

    return candidates, rejected


def _fallback_candidates(
    rejected: List[Dict[str, Any]],
    target_holdings: int,
) -> List[Dict[str, Any]]:
    fallback = []

    for item in rejected:
        hard_exclusion = _hard_exclusion(item, item)

        if hard_exclusion:
            continue

        item = {
            **item,
            "eligible": True,
            "eligibility_reason": (
                "Fallback inclusion: top-ranked workbook profile, but one "
                "portfolio filter was incomplete or temporarily failed."
            ),
        }

        fallback.append(item)

    fallback.sort(
        key=lambda item: (
            _finite(item.get("stock_intelligence_score")) or 0.0,
            _finite(item.get("portfolio_alpha")) or 0.0,
        ),
        reverse=True,
    )

    return fallback[:target_holdings]


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

    # Sector caps. This is intentionally soft. It reduces concentration without
    # forcing the portfolio blank.
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
        scale = target_exposure / weight_total

        for item in holdings:
            item["target_weight"] = min(
                MAX_SINGLE_POSITION,
                (_finite(item.get("target_weight")) or 0.0) * scale,
            )

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
        limit=max(40, target_holdings * 4),
        min_score=0,
    )

    ranking_rows = ranking_payload.get("rows", [])

    if not ranking_rows:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
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
            "diagnostics": {
                "reason": (
                    "The stock rankings API returned no rows. Test "
                    "/api/stock-rankings?refresh=true directly."
                ),
            },
        }

    history_tickers = [
        row["ticker"]
        for row in ranking_rows
        if row.get("ticker")
    ]

    price_frame = _download_price_history(history_tickers)

    candidates, rejected = _build_candidate_items(
        ranking_rows=ranking_rows,
        price_frame=price_frame,
        min_score=min_score,
    )

    candidates.sort(
        key=lambda item: _finite(item.get("portfolio_alpha")) or 0.0,
        reverse=True,
    )

    used_fallback = False

    if len(candidates) < max(3, min(target_holdings, 6)):
        fallback = _fallback_candidates(
            rejected=rejected,
            target_holdings=target_holdings,
        )

        existing = {item.get("ticker") for item in candidates}

        for item in fallback:
            if item.get("ticker") not in existing:
                candidates.append(item)

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

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "portfolio_type": "stock_alpha",
        "universe": universe_key,
        "requested_tickers": len(ticker_list),
        "ranked_candidates": len(ranking_rows),
        "eligible_candidates": len(candidates),
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
        "diagnostics": {
            "ranking_rows": len(ranking_rows),
            "initial_eligible_candidates": len([
                item for item in candidates if item.get("eligibility_reason") == "Eligible"
            ]),
            "rejected_candidates": len(rejected),
            "used_fallback": used_fallback,
            "note": (
                "Fallback is used when strict filters leave too few holdings. "
                "This prevents blank portfolios while still using workbook rank as the source of truth."
            ),
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
                "concentration, and trend-break risk."
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
    min_score: float = Query(default=60, ge=0, le=100),
    max_tickers: int = Query(default=95, ge=20, le=160),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    cache_key = (
        f"stock-portfolio-v2:{universe}:{target_holdings}:{min_score}:"
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
        "requires": "/api/stock-rankings",
        "stock_rankings_imported": rankings_available,
        "fixes": [
            "Soft fallback prevents blank holdings",
            "Lower default min_score from 65 to 60",
            "Missing Yahoo risk data no longer rejects a top-ranked workbook stock",
        ],
    }
