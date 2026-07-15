"""
Macro Engine Stock Intelligence API

Drop this file into your FastAPI backend and include `router` on the main app.

Endpoints
---------
GET /api/stocks/movers
GET /api/stocks/{ticker}

Data source
-----------
yfinance / Yahoo Finance. This module is intentionally defensive because Yahoo
fields vary by security and yfinance releases can add or remove fields.
"""

from __future__ import annotations

import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/stocks", tags=["stock-intelligence"])

# In-memory cache. Railway instances may restart, so this is an acceleration
# layer rather than durable storage.
_CACHE: Dict[str, Tuple[float, Any]] = {}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

MOVER_TTL_SECONDS = 300
STOCK_TTL_SECONDS = 900
MAX_MOVER_CANDIDATES = 48
MAX_WORKERS = 6


def _cache_get(key: str) -> Any:
    item = _CACHE.get(key)
    if not item:
        return None
    expires_at, value = item
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any, ttl_seconds: int) -> Any:
    _CACHE[key] = (time.time() + ttl_seconds, value)
    return value


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _pct(value: Optional[float], digits: int = 1) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100.0, digits)


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    n = _finite(numerator)
    d = _finite(denominator)
    if n is None or d is None or abs(d) < 1e-12:
        return None
    return n / d


def _safe_growth(current: Any, previous: Any) -> Optional[float]:
    c = _finite(current)
    p = _finite(previous)
    if c is None or p is None or abs(p) < 1e-12:
        return None
    return c / p - 1.0


def _clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value)
    if number is None:
        return low
    return max(low, min(high, number))


def _normalized_label(value: Any) -> str:
    return " ".join(
        str(value or "")
        .lower()
        .replace("&", "and")
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
        if callable(value):
            value = value()
        return default if value is None else value
    except Exception:
        return default


def _safe_info(ticker: Any) -> Dict[str, Any]:
    info = _safe_attr(ticker, "info", {})
    return info if isinstance(info, dict) else {}


def _safe_history(
    ticker: Any,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    try:
        frame = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            repair=True,
        )
    except TypeError:
        try:
            frame = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=False,
            )
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()

    return frame.dropna(how="all")


def _safe_dataframe(value: Any) -> pd.DataFrame:
    try:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame()
    except Exception:
        pass
    return pd.DataFrame()


def _frame_attr(ticker: Any, *names: str) -> pd.DataFrame:
    for name in names:
        frame = _safe_dataframe(_safe_attr(ticker, name, pd.DataFrame()))
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _safe_download(
    tickers: Sequence[str],
    period: str = "1y",
) -> pd.DataFrame:
    clean = [ticker for ticker in tickers if ticker]
    if not clean:
        return pd.DataFrame()

    try:
        frame = yf.download(
            clean,
            period=period,
            interval="1d",
            auto_adjust=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except TypeError:
        try:
            frame = yf.download(
                clean,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _screen_quotes(name: str, count: int) -> List[Dict[str, Any]]:
    try:
        result = yf.screen(name, count=count)
    except TypeError:
        try:
            result = yf.screen(name, size=count)
        except Exception:
            return []
    except Exception:
        return []

    if isinstance(result, dict):
        quotes = result.get("quotes")
        if isinstance(quotes, list):
            return quotes

        finance = result.get("finance")
        if isinstance(finance, dict):
            results = finance.get("result")
            if isinstance(results, list) and results:
                quotes = results[0].get("quotes")
                if isinstance(quotes, list):
                    return quotes

    return []


def _ticker_symbols_from_screens(
    universe: str,
    limit: int,
) -> List[Tuple[str, str]]:
    requested = universe.lower().strip()

    if requested == "all":
        screens = [
            "day_gainers",
            "day_losers",
            "most_actives",
        ]
    elif requested in {
        "day_gainers",
        "day_losers",
        "most_actives",
        "small_cap_gainers",
        "most_shorted_stocks",
    }:
        screens = [requested]
    else:
        screens = ["day_gainers"]

    seen = set()
    output: List[Tuple[str, str]] = []
    per_screen = min(
        MAX_MOVER_CANDIDATES,
        max(limit * 2, 20),
    )

    for screen in screens:
        for quote in _screen_quotes(screen, per_screen):
            symbol = str(
                quote.get("symbol")
                or quote.get("ticker")
                or ""
            ).upper().strip()

            quote_type = str(
                quote.get("quoteType")
                or quote.get("typeDisp")
                or ""
            ).upper()

            if (
                not symbol
                or symbol in seen
                or quote_type
                and quote_type not in {"EQUITY", "STOCK"}
            ):
                continue

            seen.add(symbol)
            output.append((symbol, screen))

            if len(output) >= MAX_MOVER_CANDIDATES:
                return output

    return output


def _latest_price_stats(history: pd.DataFrame) -> Dict[str, Optional[float]]:
    if history.empty or "Close" not in history.columns:
        return {}

    frame = history.dropna(subset=["Close"]).copy()
    if len(frame) < 2:
        return {}

    close = frame["Close"].astype(float)
    returns = close.pct_change()

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]

    close_value = _finite(latest.get("Close"))
    previous_close = _finite(previous.get("Close"))
    open_value = _finite(latest.get("Open"))
    high_value = _finite(latest.get("High"))
    low_value = _finite(latest.get("Low"))
    volume_value = _finite(latest.get("Volume"))

    return_1d = _safe_growth(close_value, previous_close)
    gap = _safe_growth(open_value, previous_close)

    daily_returns = [
        value
        for value in returns.iloc[-61:-1].tolist()
        if _finite(value) is not None
    ]

    z_score = None
    if return_1d is not None and len(daily_returns) >= 20:
        mean = statistics.fmean(daily_returns)
        std = statistics.pstdev(daily_returns)
        if std > 1e-12:
            z_score = (return_1d - mean) / std

    relative_volume = None
    if "Volume" in frame.columns and volume_value is not None:
        prior_volumes = [
            _finite(value)
            for value in frame["Volume"].iloc[-21:-1].tolist()
        ]
        prior_volumes = [
            value for value in prior_volumes
            if value is not None and value > 0
        ]
        if prior_volumes:
            relative_volume = volume_value / statistics.fmean(prior_volumes)

    range_position = None
    if (
        close_value is not None
        and high_value is not None
        and low_value is not None
        and high_value > low_value
    ):
        range_position = (close_value - low_value) / (high_value - low_value)

    return {
        "price": close_value,
        "previous_close": previous_close,
        "return_1d": return_1d,
        "gap": gap,
        "z_score": z_score,
        "relative_volume": relative_volume,
        "range_position": range_position,
        "volume": volume_value,
    }


def _history_return(history: pd.DataFrame, sessions: int) -> Optional[float]:
    if history.empty or "Close" not in history.columns:
        return None

    closes = history["Close"].dropna().astype(float)
    if len(closes) <= sessions:
        return None

    return _safe_growth(
        closes.iloc[-1],
        closes.iloc[-1 - sessions],
    )


def _light_fundamental_score(info: Dict[str, Any]) -> float:
    revenue_growth = _finite(info.get("revenueGrowth"))
    earnings_growth = _finite(info.get("earningsGrowth"))
    gross_margin = _finite(info.get("grossMargins"))
    operating_margin = _finite(info.get("operatingMargins"))
    return_on_equity = _finite(info.get("returnOnEquity"))
    debt_to_equity = _finite(info.get("debtToEquity"))

    scores: List[Tuple[float, float]] = []

    if revenue_growth is not None:
        scores.append(
            (
                _clamp((revenue_growth + 0.05) / 0.55 * 100),
                0.25,
            )
        )

    if earnings_growth is not None:
        scores.append(
            (
                _clamp((earnings_growth + 0.05) / 0.80 * 100),
                0.25,
            )
        )

    if gross_margin is not None:
        scores.append((_clamp(gross_margin / 0.70 * 100), 0.15))

    if operating_margin is not None:
        scores.append(
            (
                _clamp((operating_margin + 0.05) / 0.40 * 100),
                0.15,
            )
        )

    if return_on_equity is not None:
        scores.append((_clamp(return_on_equity / 0.40 * 100), 0.10))

    if debt_to_equity is not None:
        scores.append(
            (
                _clamp(100 - debt_to_equity / 4.0),
                0.10,
            )
        )

    if not scores:
        return 50.0

    weighted = sum(score * weight for score, weight in scores)
    weight_sum = sum(weight for _, weight in scores)

    return round(weighted / weight_sum, 1)


def _benchmark_one_day_return(symbol: str) -> Optional[float]:
    cache_key = f"benchmark-1d:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    history = _safe_history(yf.Ticker(symbol), period="5d")
    result = _history_return(history, 1)

    _cache_set(cache_key, result, 120)
    return result


def _mover_score(
    stats: Dict[str, Optional[float]],
    sector_return: Optional[float],
    fundamental_score: float,
) -> Dict[str, float]:
    move = stats.get("return_1d") or 0.0
    z_score = abs(stats.get("z_score") or 0.0)
    relative_volume = stats.get("relative_volume") or 1.0
    gap = abs(stats.get("gap") or 0.0)
    range_position = stats.get("range_position")

    z_component = _clamp(z_score / 4.0 * 100)
    volume_component = _clamp((relative_volume - 1.0) / 4.0 * 100)

    if sector_return is None:
        relative_component = _clamp(abs(move) / 0.12 * 100)
    else:
        relative_component = _clamp(
            abs(move - sector_return) / 0.12 * 100
        )

    gap_component = _clamp(gap / 0.08 * 100)

    if range_position is None:
        continuation_component = 50.0
    elif move >= 0:
        continuation_component = _clamp(range_position * 100)
    else:
        continuation_component = _clamp((1.0 - range_position) * 100)

    total = (
        z_component * 0.27
        + volume_component * 0.22
        + relative_component * 0.16
        + gap_component * 0.10
        + continuation_component * 0.10
        + fundamental_score * 0.15
    )

    return {
        "total": round(_clamp(total), 1),
        "price_shock": round(z_component, 1),
        "volume": round(volume_component, 1),
        "relative_strength": round(relative_component, 1),
        "gap": round(gap_component, 1),
        "continuation": round(continuation_component, 1),
        "fundamental": round(fundamental_score, 1),
    }


def _mover_classification(
    stats: Dict[str, Optional[float]],
    info: Dict[str, Any],
    fundamental_score: float,
) -> str:
    relative_volume = stats.get("relative_volume") or 1.0
    return_1d = abs(stats.get("return_1d") or 0.0)
    short_float = _finite(
        info.get("shortPercentOfFloat")
        or info.get("sharesShortPercentOfFloat")
    )
    revenue_growth = _finite(info.get("revenueGrowth"))
    earnings_growth = _finite(info.get("earningsGrowth"))

    if (
        fundamental_score >= 70
        and (
            (revenue_growth is not None and revenue_growth >= 0.18)
            or (earnings_growth is not None and earnings_growth >= 0.20)
        )
    ):
        return "Fundamental Breakout"

    if (
        short_float is not None
        and short_float >= 0.12
        and relative_volume >= 2.0
        and return_1d >= 0.06
    ):
        return "Speculative Squeeze"

    if relative_volume >= 3.0 and return_1d >= 0.08:
        return "Momentum Shock"

    return "Unusual Move"


def _human_mover_summary(
    symbol: str,
    stats: Dict[str, Optional[float]],
    sector_return: Optional[float],
    fundamental_score: float,
    classification: str,
) -> str:
    move = stats.get("return_1d")
    z_score = stats.get("z_score")
    relative_volume = stats.get("relative_volume")
    fragments: List[str] = []

    if move is not None:
        direction = "up" if move >= 0 else "down"
        fragments.append(
            f"{symbol} is {direction} {abs(move) * 100:.1f}% on the latest session"
        )

    if relative_volume is not None and relative_volume >= 1.3:
        fragments.append(f"volume is running at {relative_volume:.1f}x normal")

    if z_score is not None and abs(z_score) >= 2:
        fragments.append(
            f"the move is {abs(z_score):.1f} standard deviations from its recent daily norm"
        )

    if move is not None and sector_return is not None:
        relative = move - sector_return
        if abs(relative) >= 0.02:
            verb = "outperforming" if relative > 0 else "underperforming"
            fragments.append(
                f"it is {verb} its sector proxy by {abs(relative) * 100:.1f} points"
            )

    if fundamental_score >= 70:
        fragments.append(
            "the underlying fundamental profile is strong enough to add credibility"
        )
    elif fundamental_score <= 35:
        fragments.append(
            "the fundamental backdrop is weak, so price action deserves extra skepticism"
        )

    if not fragments:
        return f"{symbol} is showing an unusual market move worth monitoring."

    sentence = ". ".join(
        fragment[0].upper() + fragment[1:]
        for fragment in fragments[:4]
    )

    return sentence + f". The current classification is {classification}."


def _analyze_mover(symbol: str, source_screen: str) -> Optional[Dict[str, Any]]:
    ticker = yf.Ticker(symbol)
    history = _safe_history(ticker, period="6mo")

    if history.empty:
        return None

    stats = _latest_price_stats(history)
    if not stats or stats.get("return_1d") is None:
        return None

    info = _safe_info(ticker)
    sector = str(info.get("sector") or "Unknown")
    sector_etf = SECTOR_ETFS.get(sector)

    sector_return = (
        _benchmark_one_day_return(sector_etf)
        if sector_etf
        else _benchmark_one_day_return("SPY")
    )

    fundamental_score = _light_fundamental_score(info)
    score = _mover_score(stats, sector_return, fundamental_score)
    classification = _mover_classification(
        stats,
        info,
        fundamental_score,
    )

    return {
        "ticker": symbol,
        "name": (
            info.get("shortName")
            or info.get("longName")
            or symbol
        ),
        "sector": sector,
        "industry": info.get("industry"),
        "sector_etf": sector_etf,
        "source_screen": source_screen,
        "market_cap": _clean_scalar(info.get("marketCap")),
        "price": _clean_scalar(stats.get("price")),
        "return_1d": _pct(stats.get("return_1d"), 2),
        "gap": _pct(stats.get("gap"), 2),
        "z_score": (
            round(stats["z_score"], 2)
            if stats.get("z_score") is not None
            else None
        ),
        "relative_volume": (
            round(stats["relative_volume"], 2)
            if stats.get("relative_volume") is not None
            else None
        ),
        "range_position": _pct(stats.get("range_position"), 1),
        "sector_return_1d": _pct(sector_return, 2),
        "relative_to_sector": _pct(
            (
                stats.get("return_1d") - sector_return
                if stats.get("return_1d") is not None
                and sector_return is not None
                else None
            ),
            2,
        ),
        "score": score,
        "classification": classification,
        "summary": _human_mover_summary(
            symbol,
            stats,
            sector_return,
            fundamental_score,
            classification,
        ),
        "fundamental_snapshot": {
            "revenue_growth": _pct(_finite(info.get("revenueGrowth")), 1),
            "earnings_growth": _pct(_finite(info.get("earningsGrowth")), 1),
            "gross_margin": _pct(_finite(info.get("grossMargins")), 1),
            "operating_margin": _pct(_finite(info.get("operatingMargins")), 1),
            "short_percent_float": _pct(
                _finite(
                    info.get("shortPercentOfFloat")
                    or info.get("sharesShortPercentOfFloat")
                ),
                1,
            ),
        },
    }


def _statement_lookup(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    column: Any,
) -> Optional[float]:
    if frame.empty or column not in frame.columns:
        return None

    normalized = {
        _normalized_label(index): index
        for index in frame.index
    }

    for alias in aliases:
        key = _normalized_label(alias)

        if key in normalized:
            return _finite(frame.at[normalized[key], column])

        for normalized_index, original_index in normalized.items():
            if key == normalized_index or key in normalized_index:
                return _finite(frame.at[original_index, column])

    return None


def _last_columns(frame: pd.DataFrame, limit: int = 4) -> List[Any]:
    if frame.empty:
        return []

    try:
        return sorted(
            list(frame.columns),
            key=lambda value: pd.Timestamp(value),
        )[-limit:]
    except Exception:
        return list(frame.columns)[-limit:]


def _sum_last_four_quarters(
    frame: pd.DataFrame,
    aliases: Iterable[str],
) -> Optional[float]:
    columns = _last_columns(frame, 4)
    if len(columns) < 4:
        return None

    values = [
        _statement_lookup(frame, aliases, column)
        for column in columns
    ]

    if any(value is None for value in values):
        return None

    return float(sum(value for value in values if value is not None))


def _latest_quarter_value(
    frame: pd.DataFrame,
    aliases: Iterable[str],
) -> Optional[float]:
    columns = _last_columns(frame, 1)
    if not columns:
        return None
    return _statement_lookup(frame, aliases, columns[-1])


STATEMENT_ALIASES = {
    "revenue": [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
    ],
    "gross_profit": ["Gross Profit"],
    "operating_income": [
        "Operating Income",
        "Total Operating Income As Reported",
    ],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
    ],
    "eps": [
        "Diluted EPS",
        "Basic EPS",
    ],
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
    ],
    "free_cash_flow": ["Free Cash Flow"],
    "stock_comp": [
        "Stock Based Compensation",
        "Stock Based Compensation And Other",
    ],
    "cash": [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash",
    ],
    "debt": [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
    ],
    "equity": [
        "Stockholders Equity",
        "Total Stockholder Equity",
    ],
}


def _financial_period_record(
    label: str,
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    balance: pd.DataFrame,
    column: Any,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "label": label,
    }

    for key in [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps",
    ]:
        record[key] = _statement_lookup(
            income,
            STATEMENT_ALIASES[key],
            column,
        )

    for key in [
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "stock_comp",
    ]:
        record[key] = _statement_lookup(
            cashflow,
            STATEMENT_ALIASES[key],
            column,
        )

    for key in [
        "cash",
        "debt",
        "equity",
    ]:
        record[key] = _statement_lookup(
            balance,
            STATEMENT_ALIASES[key],
            column,
        )

    if record["free_cash_flow"] is None:
        operating_cash_flow = record["operating_cash_flow"]
        capex = record["capex"]
        if operating_cash_flow is not None and capex is not None:
            record["free_cash_flow"] = operating_cash_flow + capex

    revenue = record["revenue"]

    record["gross_margin"] = _safe_div(
        record["gross_profit"],
        revenue,
    )
    record["operating_margin"] = _safe_div(
        record["operating_income"],
        revenue,
    )
    record["net_margin"] = _safe_div(
        record["net_income"],
        revenue,
    )
    record["fcf_margin"] = _safe_div(
        record["free_cash_flow"],
        revenue,
    )
    record["stock_comp_pct_revenue"] = _safe_div(
        record["stock_comp"],
        revenue,
    )
    record["net_cash"] = (
        record["cash"] - record["debt"]
        if record["cash"] is not None
        and record["debt"] is not None
        else None
    )

    return record


def _build_financial_model(ticker: Any) -> Dict[str, Any]:
    annual_income = _frame_attr(
        ticker,
        "income_stmt",
        "financials",
    )
    annual_cash = _frame_attr(
        ticker,
        "cashflow",
        "cash_flow",
    )
    annual_balance = _frame_attr(
        ticker,
        "balance_sheet",
        "balancesheet",
    )

    quarterly_income = _frame_attr(
        ticker,
        "quarterly_income_stmt",
        "quarterly_financials",
    )
    quarterly_cash = _frame_attr(
        ticker,
        "quarterly_cashflow",
        "quarterly_cash_flow",
    )
    quarterly_balance = _frame_attr(
        ticker,
        "quarterly_balance_sheet",
        "quarterly_balancesheet",
    )

    annual_columns = _last_columns(annual_income, 4)
    annual_records: List[Dict[str, Any]] = []

    for column in annual_columns:
        label = pd.Timestamp(column).strftime("FY%y")
        record = _financial_period_record(
            label,
            annual_income,
            annual_cash,
            annual_balance,
            column,
        )
        annual_records.append(record)

    ttm_record: Dict[str, Any] = {"label": "TTM"}

    for key in [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps",
    ]:
        ttm_record[key] = _sum_last_four_quarters(
            quarterly_income,
            STATEMENT_ALIASES[key],
        )

    for key in [
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "stock_comp",
    ]:
        ttm_record[key] = _sum_last_four_quarters(
            quarterly_cash,
            STATEMENT_ALIASES[key],
        )

    for key in ["cash", "debt", "equity"]:
        ttm_record[key] = _latest_quarter_value(
            quarterly_balance,
            STATEMENT_ALIASES[key],
        )

    if ttm_record["free_cash_flow"] is None:
        if (
            ttm_record["operating_cash_flow"] is not None
            and ttm_record["capex"] is not None
        ):
            ttm_record["free_cash_flow"] = (
                ttm_record["operating_cash_flow"]
                + ttm_record["capex"]
            )

    ttm_revenue = ttm_record.get("revenue")

    ttm_record["gross_margin"] = _safe_div(
        ttm_record.get("gross_profit"),
        ttm_revenue,
    )
    ttm_record["operating_margin"] = _safe_div(
        ttm_record.get("operating_income"),
        ttm_revenue,
    )
    ttm_record["net_margin"] = _safe_div(
        ttm_record.get("net_income"),
        ttm_revenue,
    )
    ttm_record["fcf_margin"] = _safe_div(
        ttm_record.get("free_cash_flow"),
        ttm_revenue,
    )
    ttm_record["stock_comp_pct_revenue"] = _safe_div(
        ttm_record.get("stock_comp"),
        ttm_revenue,
    )
    ttm_record["net_cash"] = (
        ttm_record["cash"] - ttm_record["debt"]
        if ttm_record.get("cash") is not None
        and ttm_record.get("debt") is not None
        else None
    )

    records = annual_records[:]

    if any(
        ttm_record.get(key) is not None
        for key in [
            "revenue",
            "net_income",
            "free_cash_flow",
        ]
    ):
        records.append(ttm_record)

    for index, record in enumerate(records):
        previous = records[index - 1] if index > 0 else None
        record["revenue_growth"] = (
            _safe_growth(
                record.get("revenue"),
                previous.get("revenue"),
            )
            if previous
            else None
        )
        record["eps_growth"] = (
            _safe_growth(
                record.get("eps"),
                previous.get("eps"),
            )
            if previous
            else None
        )

    rows = [
        ("revenue", "Revenue", "currency"),
        ("revenue_growth", "YoY Growth", "percent"),
        ("gross_profit", "Gross Profit", "currency"),
        ("gross_margin", "Gross Margin", "percent"),
        ("operating_income", "Operating Income", "currency"),
        ("operating_margin", "Operating Margin", "percent"),
        ("net_income", "Net Income", "currency"),
        ("net_margin", "Net Margin", "percent"),
        ("eps", "Diluted EPS", "number"),
        ("eps_growth", "EPS Growth", "percent"),
        ("operating_cash_flow", "Operating Cash Flow", "currency"),
        ("capex", "Capital Expenditure", "currency"),
        ("free_cash_flow", "Free Cash Flow", "currency"),
        ("fcf_margin", "FCF Margin", "percent"),
        ("stock_comp", "Stock-Based Compensation", "currency"),
        ("stock_comp_pct_revenue", "SBC / Revenue", "percent"),
        ("cash", "Cash & Short-Term Investments", "currency"),
        ("debt", "Total Debt", "currency"),
        ("net_cash", "Net Cash / (Debt)", "currency"),
    ]

    return {
        "periods": [record["label"] for record in records],
        "records": [
            {
                key: _clean_scalar(value)
                for key, value in record.items()
            }
            for record in records
        ],
        "rows": [
            {
                "key": key,
                "label": label,
                "format": format_type,
            }
            for key, label, format_type in rows
        ],
    }


def _quarterly_series(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    columns = _last_columns(frame, limit)
    return [
        {
            "date": pd.Timestamp(column).date().isoformat(),
            "value": _statement_lookup(frame, aliases, column),
        }
        for column in columns
    ]


def _yoy_series(values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for index, item in enumerate(values):
        prior_index = index - 4
        growth = (
            _safe_growth(
                item.get("value"),
                values[prior_index].get("value"),
            )
            if prior_index >= 0
            else None
        )
        output.append(
            {
                "date": item["date"],
                "value": _clean_scalar(item.get("value")),
                "yoy_growth": _clean_scalar(growth),
            }
        )

    return output


def _fundamental_velocity(ticker: Any) -> Dict[str, Any]:
    income = _frame_attr(
        ticker,
        "quarterly_income_stmt",
        "quarterly_financials",
    )
    cashflow = _frame_attr(
        ticker,
        "quarterly_cashflow",
        "quarterly_cash_flow",
    )

    revenue = _yoy_series(
        _quarterly_series(
            income,
            STATEMENT_ALIASES["revenue"],
            8,
        )
    )
    gross_profit = _yoy_series(
        _quarterly_series(
            income,
            STATEMENT_ALIASES["gross_profit"],
            8,
        )
    )
    operating_income = _yoy_series(
        _quarterly_series(
            income,
            STATEMENT_ALIASES["operating_income"],
            8,
        )
    )
    eps = _yoy_series(
        _quarterly_series(
            income,
            STATEMENT_ALIASES["eps"],
            8,
        )
    )
    fcf = _yoy_series(
        _quarterly_series(
            cashflow,
            STATEMENT_ALIASES["free_cash_flow"],
            8,
        )
    )

    revenue_acceleration = None
    revenue_growth_points = [
        item["yoy_growth"]
        for item in revenue
        if item.get("yoy_growth") is not None
    ]
    if len(revenue_growth_points) >= 2:
        revenue_acceleration = (
            revenue_growth_points[-1]
            - revenue_growth_points[-2]
        )

    eps_acceleration = None
    eps_growth_points = [
        item["yoy_growth"]
        for item in eps
        if item.get("yoy_growth") is not None
    ]
    if len(eps_growth_points) >= 2:
        eps_acceleration = (
            eps_growth_points[-1]
            - eps_growth_points[-2]
        )

    gross_margin_series: List[Dict[str, Any]] = []
    revenue_values = _quarterly_series(
        income,
        STATEMENT_ALIASES["revenue"],
        8,
    )
    gross_values = _quarterly_series(
        income,
        STATEMENT_ALIASES["gross_profit"],
        8,
    )

    for revenue_item, gross_item in zip(revenue_values, gross_values):
        gross_margin_series.append(
            {
                "date": revenue_item["date"],
                "value": _safe_div(
                    gross_item.get("value"),
                    revenue_item.get("value"),
                ),
            }
        )

    margin_expansion = None
    if len(gross_margin_series) >= 5:
        latest = gross_margin_series[-1].get("value")
        prior_year = gross_margin_series[-5].get("value")
        if latest is not None and prior_year is not None:
            margin_expansion = latest - prior_year

    components: List[Tuple[float, float]] = []

    if revenue_growth_points:
        components.append(
            (
                _clamp(
                    (revenue_growth_points[-1] + 0.10)
                    / 0.70
                    * 100
                ),
                0.30,
            )
        )

    if revenue_acceleration is not None:
        components.append(
            (
                _clamp(
                    (revenue_acceleration + 0.10)
                    / 0.30
                    * 100
                ),
                0.25,
            )
        )

    if eps_growth_points:
        components.append(
            (
                _clamp(
                    (eps_growth_points[-1] + 0.10)
                    / 1.00
                    * 100
                ),
                0.20,
            )
        )

    if margin_expansion is not None:
        components.append(
            (
                _clamp(
                    (margin_expansion + 0.05)
                    / 0.15
                    * 100
                ),
                0.15,
            )
        )

    fcf_growth_points = [
        item["yoy_growth"]
        for item in fcf
        if item.get("yoy_growth") is not None
    ]
    if fcf_growth_points:
        components.append(
            (
                _clamp(
                    (fcf_growth_points[-1] + 0.10)
                    / 1.00
                    * 100
                ),
                0.10,
            )
        )

    if components:
        total = sum(score * weight for score, weight in components)
        weight_sum = sum(weight for _, weight in components)
        score = round(total / weight_sum, 1)
    else:
        score = 50.0

    if score >= 80:
        label = "Rapidly Accelerating"
    elif score >= 65:
        label = "Accelerating"
    elif score >= 45:
        label = "Stable"
    elif score >= 30:
        label = "Decelerating"
    else:
        label = "Rapidly Decelerating"

    return {
        "score": score,
        "label": label,
        "revenue_acceleration": _pct(revenue_acceleration, 1),
        "eps_acceleration": _pct(eps_acceleration, 1),
        "gross_margin_expansion": _pct(margin_expansion, 1),
        "series": {
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "eps": eps,
            "free_cash_flow": fcf,
            "gross_margin": [
                {
                    "date": item["date"],
                    "value": _clean_scalar(item["value"]),
                }
                for item in gross_margin_series
            ],
        },
    }


def _serialize_dataframe(
    frame: pd.DataFrame,
    max_rows: int = 12,
) -> Dict[str, Any]:
    if frame.empty:
        return {
            "columns": [],
            "rows": [],
        }

    clean = frame.copy().head(max_rows)
    clean = clean.reset_index()

    columns = [str(column) for column in clean.columns]
    rows: List[Dict[str, Any]] = []

    for _, row in clean.iterrows():
        rows.append(
            {
                str(column): _clean_scalar(row[column])
                for column in clean.columns
            }
        )

    return {
        "columns": columns,
        "rows": rows,
    }


def _revision_summary(ticker: Any) -> Dict[str, Any]:
    eps_trend = _frame_attr(ticker, "eps_trend")
    eps_revisions = _frame_attr(ticker, "eps_revisions")
    earnings_estimate = _frame_attr(ticker, "earnings_estimate")
    revenue_estimate = _frame_attr(ticker, "revenue_estimate")
    growth_estimates = _frame_attr(ticker, "growth_estimates")

    score_components: List[Tuple[float, float]] = []
    summary: Dict[str, Any] = {
        "score": 50.0,
        "label": "Neutral",
        "current_eps": None,
        "eps_30d_ago": None,
        "eps_90d_ago": None,
        "eps_revision_30d": None,
        "eps_revision_90d": None,
        "up_30d": None,
        "down_30d": None,
    }

    if not eps_trend.empty:
        first_row = eps_trend.iloc[0]
        columns = {
            _normalized_label(column): column
            for column in eps_trend.columns
        }

        def get_column(*aliases: str) -> Optional[Any]:
            for alias in aliases:
                key = _normalized_label(alias)
                if key in columns:
                    return columns[key]
            return None

        current_column = get_column("current")
        ago_30_column = get_column("30daysAgo", "30 days ago")
        ago_90_column = get_column("90daysAgo", "90 days ago")

        current_eps = (
            _finite(first_row[current_column])
            if current_column is not None
            else None
        )
        eps_30 = (
            _finite(first_row[ago_30_column])
            if ago_30_column is not None
            else None
        )
        eps_90 = (
            _finite(first_row[ago_90_column])
            if ago_90_column is not None
            else None
        )

        revision_30 = _safe_growth(current_eps, eps_30)
        revision_90 = _safe_growth(current_eps, eps_90)

        summary.update(
            {
                "current_eps": current_eps,
                "eps_30d_ago": eps_30,
                "eps_90d_ago": eps_90,
                "eps_revision_30d": _pct(revision_30, 1),
                "eps_revision_90d": _pct(revision_90, 1),
            }
        )

        if revision_30 is not None:
            score_components.append(
                (
                    _clamp((revision_30 + 0.10) / 0.30 * 100),
                    0.55,
                )
            )

        if revision_90 is not None:
            score_components.append(
                (
                    _clamp((revision_90 + 0.15) / 0.45 * 100),
                    0.25,
                )
            )

    if not eps_revisions.empty:
        first_row = eps_revisions.iloc[0]
        columns = {
            _normalized_label(column): column
            for column in eps_revisions.columns
        }

        up_column = next(
            (
                original
                for normalized, original in columns.items()
                if "up" in normalized and "30" in normalized
            ),
            None,
        )
        down_column = next(
            (
                original
                for normalized, original in columns.items()
                if "down" in normalized and "30" in normalized
            ),
            None,
        )

        up_30 = (
            _finite(first_row[up_column])
            if up_column is not None
            else None
        )
        down_30 = (
            _finite(first_row[down_column])
            if down_column is not None
            else None
        )

        summary["up_30d"] = up_30
        summary["down_30d"] = down_30

        if up_30 is not None or down_30 is not None:
            up_value = up_30 or 0.0
            down_value = down_30 or 0.0
            breadth = _safe_div(
                up_value,
                up_value + down_value,
            )
            if breadth is not None:
                score_components.append((breadth * 100, 0.20))

    if score_components:
        total = sum(score * weight for score, weight in score_components)
        weight_sum = sum(weight for _, weight in score_components)
        score = round(total / weight_sum, 1)
    else:
        score = 50.0

    if score >= 80:
        label = "Strongly Rising"
    elif score >= 65:
        label = "Rising"
    elif score >= 45:
        label = "Neutral"
    elif score >= 30:
        label = "Falling"
    else:
        label = "Strongly Falling"

    summary["score"] = score
    summary["label"] = label
    summary["tables"] = {
        "eps_trend": _serialize_dataframe(eps_trend),
        "eps_revisions": _serialize_dataframe(eps_revisions),
        "earnings_estimate": _serialize_dataframe(earnings_estimate),
        "revenue_estimate": _serialize_dataframe(revenue_estimate),
        "growth_estimates": _serialize_dataframe(growth_estimates),
    }

    return summary


def _price_series(history: pd.DataFrame) -> List[Dict[str, Any]]:
    if history.empty or "Close" not in history.columns:
        return []

    output: List[Dict[str, Any]] = []

    for index, row in history.iterrows():
        close = _finite(row.get("Close"))
        if close is None:
            continue

        output.append(
            {
                "date": pd.Timestamp(index).date().isoformat(),
                "close": round(close, 4),
                "volume": _clean_scalar(row.get("Volume")),
            }
        )

    return output


def _relative_performance(
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    windows = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    output: Dict[str, Optional[float]] = {}

    for label, sessions in windows.items():
        stock_return = _history_return(stock_history, sessions)
        benchmark_return = _history_return(benchmark_history, sessions)

        if stock_return is None or benchmark_return is None:
            output[label] = None
        else:
            output[label] = round(
                (stock_return - benchmark_return) * 100,
                2,
            )

    return output


def _absolute_performance(history: pd.DataFrame) -> Dict[str, Optional[float]]:
    windows = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    return {
        label: (
            round(value * 100, 2)
            if (value := _history_return(history, sessions)) is not None
            else None
        )
        for label, sessions in windows.items()
    }


def _earnings_reactions(
    ticker: Any,
    history: pd.DataFrame,
) -> Dict[str, Any]:
    try:
        earnings_dates = ticker.get_earnings_dates(limit=12)
    except Exception:
        earnings_dates = pd.DataFrame()

    earnings_dates = _safe_dataframe(earnings_dates)

    if earnings_dates.empty or history.empty:
        return {
            "events": [],
            "summary": {},
        }

    prices = history.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    events: List[Dict[str, Any]] = []

    for event_date, row in earnings_dates.iterrows():
        try:
            event_timestamp = pd.Timestamp(event_date).tz_localize(None)
        except Exception:
            continue

        on_or_after = prices.index[prices.index >= event_timestamp.normalize()]
        before = prices.index[prices.index < event_timestamp.normalize()]

        if len(on_or_after) == 0 or len(before) == 0:
            continue

        reaction_index = prices.index.get_loc(on_or_after[0])
        previous_index = reaction_index - 1

        if previous_index < 0:
            continue

        previous_close = _finite(prices.iloc[previous_index].get("Close"))
        day_one_close = _finite(prices.iloc[reaction_index].get("Close"))

        def forward_return(offset: int) -> Optional[float]:
            target = reaction_index + offset
            if target >= len(prices):
                return None
            return _safe_growth(
                prices.iloc[target].get("Close"),
                previous_close,
            )

        estimate = None
        reported = None
        surprise = None

        for column in earnings_dates.columns:
            normalized = _normalized_label(column)
            if "eps estimate" in normalized:
                estimate = _finite(row[column])
            elif "reported eps" in normalized:
                reported = _finite(row[column])
            elif "surprise" in normalized:
                surprise = _finite(row[column])

        events.append(
            {
                "date": event_timestamp.date().isoformat(),
                "eps_estimate": estimate,
                "reported_eps": reported,
                "surprise_pct": (
                    round(surprise * 100, 2)
                    if surprise is not None and abs(surprise) < 5
                    else _clean_scalar(surprise)
                ),
                "day_1": _pct(
                    _safe_growth(day_one_close, previous_close),
                    2,
                ),
                "day_5": _pct(forward_return(4), 2),
                "day_20": _pct(forward_return(19), 2),
            }
        )

    events.sort(key=lambda item: item["date"], reverse=True)

    def average(key: str) -> Optional[float]:
        values = [
            _finite(event.get(key))
            for event in events
        ]
        clean = [value for value in values if value is not None]
        return round(statistics.fmean(clean), 2) if clean else None

    beat_count = len(
        [
            event
            for event in events
            if event.get("surprise_pct") is not None
            and event["surprise_pct"] > 0
        ]
    )

    return {
        "events": events[:8],
        "summary": {
            "beat_count": beat_count,
            "event_count": len(events[:8]),
            "average_surprise_pct": average("surprise_pct"),
            "average_day_1": average("day_1"),
            "average_day_5": average("day_5"),
            "average_day_20": average("day_20"),
            "methodology": (
                "Price reactions use the first market session on or after "
                "the reported earnings date and compare against the prior close."
            ),
        },
    }


def _valuation_snapshot(
    info: Dict[str, Any],
    model: Dict[str, Any],
) -> Dict[str, Any]:
    records = model.get("records") or []
    latest = records[-1] if records else {}

    market_cap = _finite(info.get("marketCap"))
    enterprise_value = _finite(info.get("enterpriseValue"))
    revenue = _finite(latest.get("revenue"))
    net_income = _finite(latest.get("net_income"))
    free_cash_flow = _finite(latest.get("free_cash_flow"))
    net_margin = _finite(latest.get("net_margin"))
    shares = _finite(
        info.get("sharesOutstanding")
        or info.get("impliedSharesOutstanding")
    )

    metrics = [
        {
            "key": "forward_pe",
            "label": "Forward P/E",
            "value": _finite(info.get("forwardPE")),
            "format": "multiple",
        },
        {
            "key": "trailing_pe",
            "label": "Trailing P/E",
            "value": _finite(info.get("trailingPE")),
            "format": "multiple",
        },
        {
            "key": "price_sales",
            "label": "Price / Sales",
            "value": (
                _safe_div(market_cap, revenue)
                if revenue is not None
                else _finite(info.get("priceToSalesTrailing12Months"))
            ),
            "format": "multiple",
        },
        {
            "key": "ev_ebitda",
            "label": "EV / EBITDA",
            "value": _finite(info.get("enterpriseToEbitda")),
            "format": "multiple",
        },
        {
            "key": "ev_revenue",
            "label": "EV / Revenue",
            "value": (
                _safe_div(enterprise_value, revenue)
                if revenue is not None
                else _finite(info.get("enterpriseToRevenue"))
            ),
            "format": "multiple",
        },
        {
            "key": "fcf_yield",
            "label": "FCF Yield",
            "value": _safe_div(free_cash_flow, market_cap),
            "format": "percent",
        },
        {
            "key": "earnings_yield",
            "label": "Earnings Yield",
            "value": _safe_div(net_income, market_cap),
            "format": "percent",
        },
    ]

    # Reverse expectations. This is deliberately transparent and simple:
    # equity value / terminal P/E => required year-five earnings, then divide by
    # current net margin to get required year-five revenue.
    terminal_pe = 25.0
    current_revenue = revenue
    current_margin = net_margin

    required_cagr = None
    required_year_5_revenue = None
    required_year_5_net_income = None

    if (
        market_cap is not None
        and current_revenue is not None
        and current_revenue > 0
        and current_margin is not None
        and current_margin > 0.01
    ):
        required_year_5_net_income = market_cap / terminal_pe
        required_year_5_revenue = (
            required_year_5_net_income / current_margin
        )
        if required_year_5_revenue > 0:
            required_cagr = (
                required_year_5_revenue / current_revenue
            ) ** (1.0 / 5.0) - 1.0

    return {
        "metrics": [
            {
                **metric,
                "value": _clean_scalar(metric["value"]),
            }
            for metric in metrics
        ],
        "reverse_expectations": {
            "terminal_pe": terminal_pe,
            "current_revenue": current_revenue,
            "current_net_margin": current_margin,
            "required_year_5_revenue": required_year_5_revenue,
            "required_year_5_net_income": required_year_5_net_income,
            "required_revenue_cagr": required_cagr,
            "shares_outstanding": shares,
            "methodology": (
                "Simple five-year reverse expectations model. It divides "
                "current equity value by a 25x terminal P/E, then uses the "
                "current net margin to estimate required year-five revenue."
            ),
        },
    }


def _scenario_valuation(
    info: Dict[str, Any],
    model: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records = model.get("records") or []
    latest = records[-1] if records else {}

    revenue = _finite(latest.get("revenue"))
    current_margin = _finite(latest.get("net_margin"))
    shares = _finite(
        info.get("sharesOutstanding")
        or info.get("impliedSharesOutstanding")
    )
    current_price = _finite(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
    )
    net_cash = _finite(latest.get("net_cash")) or 0.0

    if (
        revenue is None
        or revenue <= 0
        or shares is None
        or shares <= 0
    ):
        return []

    base_margin = current_margin if current_margin and current_margin > 0 else 0.12

    revenue_growth = _finite(info.get("revenueGrowth"))
    base_growth = _clamp(
        (revenue_growth if revenue_growth is not None else 0.12) * 100,
        5,
        35,
    ) / 100

    cases = [
        {
            "name": "Bear",
            "growth": max(base_growth - 0.10, 0.02),
            "margin": max(base_margin - 0.05, 0.03),
            "pe": 18.0,
        },
        {
            "name": "Base",
            "growth": base_growth,
            "margin": base_margin,
            "pe": 25.0,
        },
        {
            "name": "Bull",
            "growth": min(base_growth + 0.10, 0.50),
            "margin": min(base_margin + 0.05, 0.55),
            "pe": 32.0,
        },
    ]

    output: List[Dict[str, Any]] = []

    for case in cases:
        year_5_revenue = revenue * ((1.0 + case["growth"]) ** 5)
        year_5_net_income = year_5_revenue * case["margin"]
        terminal_equity_value = year_5_net_income * case["pe"] + net_cash
        implied_price = terminal_equity_value / shares
        upside = _safe_growth(implied_price, current_price)

        output.append(
            {
                "name": case["name"],
                "revenue_cagr": case["growth"],
                "net_margin": case["margin"],
                "exit_pe": case["pe"],
                "year_5_revenue": year_5_revenue,
                "year_5_net_income": year_5_net_income,
                "implied_price": implied_price,
                "upside": upside,
            }
        )

    return output


def _options_snapshot(
    ticker: Any,
    spot_price: Optional[float],
) -> Dict[str, Any]:
    options = _safe_attr(ticker, "options", ())

    if not options:
        return {}

    expiration = options[0]

    try:
        chain = ticker.option_chain(expiration)
    except Exception:
        return {}

    calls = _safe_dataframe(getattr(chain, "calls", pd.DataFrame()))
    puts = _safe_dataframe(getattr(chain, "puts", pd.DataFrame()))

    def sum_column(frame: pd.DataFrame, column: str) -> Optional[float]:
        if frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.sum()) if not values.empty else None

    call_oi = sum_column(calls, "openInterest")
    put_oi = sum_column(puts, "openInterest")
    call_volume = sum_column(calls, "volume")
    put_volume = sum_column(puts, "volume")

    def max_oi_strike(frame: pd.DataFrame) -> Optional[float]:
        if (
            frame.empty
            or "openInterest" not in frame.columns
            or "strike" not in frame.columns
        ):
            return None
        temp = frame.copy()
        temp["openInterest"] = pd.to_numeric(
            temp["openInterest"],
            errors="coerce",
        )
        temp = temp.dropna(subset=["openInterest", "strike"])
        if temp.empty:
            return None
        row = temp.loc[temp["openInterest"].idxmax()]
        return _finite(row.get("strike"))

    atm_iv = None
    if spot_price is not None:
        iv_values: List[float] = []

        for frame in [calls, puts]:
            if (
                not frame.empty
                and "strike" in frame.columns
                and "impliedVolatility" in frame.columns
            ):
                temp = frame.copy()
                temp["distance"] = (
                    pd.to_numeric(temp["strike"], errors="coerce")
                    - spot_price
                ).abs()
                temp = temp.dropna(subset=["distance", "impliedVolatility"])
                if not temp.empty:
                    row = temp.loc[temp["distance"].idxmin()]
                    iv = _finite(row.get("impliedVolatility"))
                    if iv is not None:
                        iv_values.append(iv)

        if iv_values:
            atm_iv = statistics.fmean(iv_values)

    expected_move = None
    if atm_iv is not None:
        try:
            expiry_date = datetime.fromisoformat(expiration)
            days = max((expiry_date - datetime.utcnow()).days, 1)
            expected_move = atm_iv * math.sqrt(days / 365.0)
        except Exception:
            expected_move = None

    return {
        "expiration": expiration,
        "call_open_interest": call_oi,
        "put_open_interest": put_oi,
        "put_call_oi_ratio": _safe_div(put_oi, call_oi),
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_wall": max_oi_strike(calls),
        "put_wall": max_oi_strike(puts),
        "atm_implied_volatility": atm_iv,
        "expected_move": expected_move,
    }


def _industry_peers(
    info: Dict[str, Any],
    ticker_symbol: str,
) -> List[str]:
    industry_key = (
        info.get("industryKey")
        or info.get("industryDisp")
    )

    if not industry_key:
        return []

    try:
        industry = yf.Industry(industry_key)
        top_companies = _safe_dataframe(
            _safe_attr(industry, "top_companies", pd.DataFrame())
        )
    except Exception:
        return []

    if top_companies.empty:
        return []

    symbols: List[str] = []

    index_candidates = [str(value).upper() for value in top_companies.index]

    for symbol in index_candidates:
        if (
            symbol
            and symbol != ticker_symbol
            and symbol not in symbols
        ):
            symbols.append(symbol)

    if not symbols:
        for column in top_companies.columns:
            if "symbol" in _normalized_label(column) or "ticker" in _normalized_label(column):
                for value in top_companies[column].tolist():
                    symbol = str(value).upper().strip()
                    if (
                        symbol
                        and symbol != ticker_symbol
                        and symbol not in symbols
                    ):
                        symbols.append(symbol)

    return symbols[:5]


def _peer_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    ticker = yf.Ticker(symbol)
    info = _safe_info(ticker)

    if not info:
        return None

    return {
        "ticker": symbol,
        "name": (
            info.get("shortName")
            or info.get("longName")
            or symbol
        ),
        "market_cap": _clean_scalar(info.get("marketCap")),
        "revenue_growth": _pct(_finite(info.get("revenueGrowth")), 1),
        "earnings_growth": _pct(_finite(info.get("earningsGrowth")), 1),
        "gross_margin": _pct(_finite(info.get("grossMargins")), 1),
        "operating_margin": _pct(_finite(info.get("operatingMargins")), 1),
        "forward_pe": _clean_scalar(info.get("forwardPE")),
        "ev_ebitda": _clean_scalar(info.get("enterpriseToEbitda")),
    }


def _build_peers(info: Dict[str, Any], ticker_symbol: str) -> List[Dict[str, Any]]:
    symbols = _industry_peers(info, ticker_symbol)

    if not symbols:
        return []

    peers: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as executor:
        future_map = {
            executor.submit(_peer_snapshot, symbol): symbol
            for symbol in symbols
        }

        for future in as_completed(future_map):
            try:
                peer = future.result()
            except Exception:
                peer = None

            if peer:
                peers.append(peer)

    order = {symbol: index for index, symbol in enumerate(symbols)}
    peers.sort(key=lambda peer: order.get(peer["ticker"], 999))

    return peers


def _stock_intelligence_score(
    fundamental_velocity: Dict[str, Any],
    revisions: Dict[str, Any],
    price_stats: Dict[str, Optional[float]],
    valuation: Dict[str, Any],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    velocity_score = _finite(fundamental_velocity.get("score")) or 50.0
    revision_score = _finite(revisions.get("score")) or 50.0

    stock_return_6m = price_stats.get("return_6m")
    price_score = (
        _clamp((stock_return_6m + 0.20) / 0.80 * 100)
        if stock_return_6m is not None
        else 50.0
    )

    forward_pe = _finite(info.get("forwardPE"))
    fcf_yield = None

    for metric in valuation.get("metrics", []):
        if metric.get("key") == "fcf_yield":
            fcf_yield = _finite(metric.get("value"))

    valuation_score = 50.0
    parts = []

    if forward_pe is not None and forward_pe > 0:
        parts.append(_clamp(100 - (forward_pe - 10) / 50 * 100))

    if fcf_yield is not None:
        parts.append(_clamp((fcf_yield + 0.02) / 0.10 * 100))

    if parts:
        valuation_score = statistics.fmean(parts)

    current_ratio = _finite(info.get("currentRatio"))
    debt_to_equity = _finite(info.get("debtToEquity"))
    balance_parts = []

    if current_ratio is not None:
        balance_parts.append(_clamp(current_ratio / 2.5 * 100))

    if debt_to_equity is not None:
        balance_parts.append(_clamp(100 - debt_to_equity / 4.0))

    balance_score = (
        statistics.fmean(balance_parts)
        if balance_parts
        else 50.0
    )

    total = (
        velocity_score * 0.28
        + revision_score * 0.22
        + price_score * 0.20
        + valuation_score * 0.15
        + balance_score * 0.15
    )

    return {
        "total": round(_clamp(total), 1),
        "fundamental_velocity": round(_clamp(velocity_score), 1),
        "estimate_revisions": round(_clamp(revision_score), 1),
        "price_leadership": round(_clamp(price_score), 1),
        "valuation": round(_clamp(valuation_score), 1),
        "balance_sheet": round(_clamp(balance_score), 1),
    }


def _why_it_matters(
    symbol: str,
    price_stats: Dict[str, Optional[float]],
    velocity: Dict[str, Any],
    revisions: Dict[str, Any],
    relative_to_sector_1d: Optional[float],
) -> Dict[str, Any]:
    bullets: List[Dict[str, str]] = []

    move = price_stats.get("return_1d")
    z_score = price_stats.get("z_score")
    relative_volume = price_stats.get("relative_volume")

    if move is not None:
        bullets.append(
            {
                "tone": "positive" if move > 0 else "negative",
                "text": (
                    f"Price moved {abs(move) * 100:.1f}% "
                    f"{'higher' if move > 0 else 'lower'} in the latest session."
                ),
            }
        )

    if z_score is not None:
        bullets.append(
            {
                "tone": "positive" if abs(z_score) >= 2 else "neutral",
                "text": (
                    f"The move is {abs(z_score):.1f} standard deviations "
                    "from the stock's recent daily norm."
                ),
            }
        )

    if relative_volume is not None:
        bullets.append(
            {
                "tone": "positive" if relative_volume >= 2 else "neutral",
                "text": f"Volume is running at {relative_volume:.1f}x the 20-day average.",
            }
        )

    if relative_to_sector_1d is not None:
        bullets.append(
            {
                "tone": "positive" if relative_to_sector_1d > 0 else "negative",
                "text": (
                    f"The stock is {'outperforming' if relative_to_sector_1d > 0 else 'underperforming'} "
                    f"its sector proxy by {abs(relative_to_sector_1d) * 100:.1f} percentage points."
                ),
            }
        )

    velocity_score = _finite(velocity.get("score")) or 50
    revision_score = _finite(revisions.get("score")) or 50

    if velocity_score >= 65:
        bullets.append(
            {
                "tone": "positive",
                "text": (
                    f"Fundamental velocity is {velocity.get('label', 'accelerating')} "
                    f"with a score of {velocity_score:.0f}/100."
                ),
            }
        )
    elif velocity_score <= 35:
        bullets.append(
            {
                "tone": "negative",
                "text": (
                    f"Fundamental velocity is weakening with a score of "
                    f"{velocity_score:.0f}/100."
                ),
            }
        )

    if revision_score >= 65:
        bullets.append(
            {
                "tone": "positive",
                "text": (
                    f"Analyst expectations are moving higher. The revision score is "
                    f"{revision_score:.0f}/100."
                ),
            }
        )
    elif revision_score <= 35:
        bullets.append(
            {
                "tone": "negative",
                "text": (
                    f"Analyst expectations are being revised lower. The revision score is "
                    f"{revision_score:.0f}/100."
                ),
            }
        )

    positive = len([item for item in bullets if item["tone"] == "positive"])
    negative = len([item for item in bullets if item["tone"] == "negative"])

    if positive >= negative + 2:
        verdict = (
            f"{symbol}'s move has meaningful confirmation beneath the surface. "
            "Price action, fundamentals, and expectations are pointing in the same direction."
        )
    elif negative >= positive + 2:
        verdict = (
            f"{symbol}'s move is not being confirmed by the broader evidence. "
            "That raises the risk that price is running ahead of fundamentals."
        )
    else:
        verdict = (
            f"{symbol}'s setup is mixed. The move is real, but the evidence is not "
            "strong enough to treat price action alone as confirmation."
        )

    return {
        "verdict": verdict,
        "bullets": bullets[:7],
    }


def _company_profile(info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    return {
        "ticker": symbol,
        "name": (
            info.get("longName")
            or info.get("shortName")
            or symbol
        ),
        "short_name": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "market_cap": _clean_scalar(info.get("marketCap")),
        "enterprise_value": _clean_scalar(info.get("enterpriseValue")),
        "currency": info.get("currency") or "USD",
        "exchange": info.get("exchange"),
    }


def _full_stock_payload(symbol: str) -> Dict[str, Any]:
    ticker = yf.Ticker(symbol)
    info = _safe_info(ticker)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"No Yahoo Finance data found for {symbol}.",
        )

    history = _safe_history(ticker, period="2y")
    if history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price history found for {symbol}.",
        )

    price_stats = _latest_price_stats(history)
    price_stats["return_1w"] = _history_return(history, 5)
    price_stats["return_1m"] = _history_return(history, 21)
    price_stats["return_3m"] = _history_return(history, 63)
    price_stats["return_6m"] = _history_return(history, 126)
    price_stats["return_1y"] = _history_return(history, 252)

    sector = str(info.get("sector") or "")
    sector_etf = SECTOR_ETFS.get(sector)
    benchmark_symbols = ["SPY"]
    if sector_etf:
        benchmark_symbols.append(sector_etf)

    benchmark_data = {
        benchmark: _safe_history(yf.Ticker(benchmark), period="2y")
        for benchmark in benchmark_symbols
    }

    sector_return_1d = (
        _history_return(benchmark_data.get(sector_etf, pd.DataFrame()), 1)
        if sector_etf
        else None
    )

    model = _build_financial_model(ticker)
    velocity = _fundamental_velocity(ticker)
    revisions = _revision_summary(ticker)
    valuation = _valuation_snapshot(info, model)
    scenarios = _scenario_valuation(info, model)
    earnings = _earnings_reactions(ticker, history)
    options = _options_snapshot(ticker, price_stats.get("price"))
    peers = _build_peers(info, symbol)

    score = _stock_intelligence_score(
        velocity,
        revisions,
        price_stats,
        valuation,
        info,
    )

    why = _why_it_matters(
        symbol,
        price_stats,
        velocity,
        revisions,
        (
            price_stats["return_1d"] - sector_return_1d
            if price_stats.get("return_1d") is not None
            and sector_return_1d is not None
            else None
        ),
    )

    relative = {
        "stock": _absolute_performance(history),
        "vs_spy": _relative_performance(
            history,
            benchmark_data.get("SPY", pd.DataFrame()),
        ),
        "sector_etf": sector_etf,
        "vs_sector": (
            _relative_performance(
                history,
                benchmark_data.get(sector_etf, pd.DataFrame()),
            )
            if sector_etf
            else {}
        ),
    }

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profile": _company_profile(info, symbol),
        "price": {
            "current": _clean_scalar(price_stats.get("price")),
            "previous_close": _clean_scalar(price_stats.get("previous_close")),
            "return_1d": _pct(price_stats.get("return_1d"), 2),
            "return_1w": _pct(price_stats.get("return_1w"), 2),
            "return_1m": _pct(price_stats.get("return_1m"), 2),
            "return_3m": _pct(price_stats.get("return_3m"), 2),
            "return_6m": _pct(price_stats.get("return_6m"), 2),
            "return_1y": _pct(price_stats.get("return_1y"), 2),
            "gap": _pct(price_stats.get("gap"), 2),
            "z_score": _clean_scalar(price_stats.get("z_score")),
            "relative_volume": _clean_scalar(price_stats.get("relative_volume")),
            "range_position": _pct(price_stats.get("range_position"), 1),
        },
        "price_history": _price_series(history.iloc[-260:]),
        "stock_intelligence_score": score,
        "why_it_matters": why,
        "financial_model": model,
        "fundamental_velocity": velocity,
        "revisions": revisions,
        "earnings": earnings,
        "valuation": valuation,
        "scenarios": scenarios,
        "relative_performance": relative,
        "options": {
            key: _clean_scalar(value)
            for key, value in options.items()
        },
        "peers": peers,
        "methodology": {
            "price": (
                "Latest available daily Yahoo Finance bars. "
                "Mover statistics compare the latest return and volume "
                "with recent trading history."
            ),
            "fundamental_velocity": (
                "Combines quarterly revenue growth, revenue acceleration, "
                "EPS growth, gross-margin expansion, and free-cash-flow growth."
            ),
            "revisions": (
                "Uses Yahoo Finance analyst estimate trend and revision tables "
                "when available."
            ),
            "scenario_valuation": (
                "Illustrative five-year bear/base/bull framework. It is not a "
                "price target or individualized investment advice."
            ),
        },
    }


@router.get("/movers")
def get_stock_movers(
    limit: int = Query(default=24, ge=5, le=40),
    universe: str = Query(
        default="all",
        description=(
            "all, day_gainers, day_losers, most_actives, "
            "small_cap_gainers, or most_shorted_stocks"
        ),
    ),
) -> Dict[str, Any]:
    cache_key = f"stock-movers:{universe}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    candidates = _ticker_symbols_from_screens(universe, limit)

    if not candidates:
        raise HTTPException(
            status_code=503,
            detail="Yahoo Finance returned no mover candidates.",
        )

    movers: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_analyze_mover, symbol, source_screen): symbol
            for symbol, source_screen in candidates
        }

        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception:
                result = None

            if result:
                movers.append(result)

    movers.sort(
        key=lambda item: item.get("score", {}).get("total", 0),
        reverse=True,
    )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe": universe,
        "candidate_count": len(candidates),
        "count": min(len(movers), limit),
        "movers": movers[:limit],
        "methodology": {
            "mover_score": {
                "price_shock": 27,
                "volume": 22,
                "relative_strength": 16,
                "gap": 10,
                "continuation": 10,
                "fundamental": 15,
            },
            "note": (
                "Mover Score ranks how unusual the move is relative to the "
                "stock's own history and current market context. It is not a "
                "buy or sell signal."
            ),
        },
    }

    return _cache_set(
        cache_key,
        payload,
        MOVER_TTL_SECONDS,
    )


@router.get("/{ticker_symbol}")
def get_stock_intelligence(
    ticker_symbol: str,
) -> Dict[str, Any]:
    symbol = ticker_symbol.upper().strip()

    if not symbol or len(symbol) > 15:
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker symbol.",
        )

    cache_key = f"stock-intelligence:{symbol}"
    cached = _cache_get(cache_key)

    if cached is not None:
        return cached

    payload = _full_stock_payload(symbol)

    return _cache_set(
        cache_key,
        payload,
        STOCK_TTL_SECONDS,
    )
