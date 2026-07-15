"""
EMA Inflection Scanner API

FastAPI route:
    GET /api/ema-inflection

Purpose:
    Find stocks where normalized EMA velocity and acceleration are shifting
    from weak/negative to improving, with volume confirmation.

This route is intentionally independent from Macro Engine's macro regime data.
It only uses Yahoo Finance daily OHLCV data through yfinance.
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

router = APIRouter(
    prefix="/api/ema-inflection",
    tags=["ema-inflection"],
)

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

CACHE_TTL_SECONDS = 30 * 60
MAX_LIMIT = 100

CORE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "AMD", "NFLX", "PLTR", "ORCL", "CRM", "ADBE", "NOW", "SNOW", "SHOP",
    "UBER", "ABNB", "APP", "ARM", "SMCI", "MU", "MRVL", "QCOM", "INTC",
    "TSM", "ASML", "LRCX", "KLAC", "AMAT", "ON", "MPWR", "ALAB", "CRDO",
    "VRT", "ANET", "DELL", "HPE", "PANW", "CRWD", "NET", "DDOG", "MDB",
    "ZS", "OKTA", "S", "PATH", "U", "AI", "SOUN", "BBAI", "RBLX", "RDDT",
    "COIN", "HOOD", "MSTR", "SOFI", "AFRM", "UPST", "PYPL", "SQ", "ROKU",
    "SPOT", "PINS", "TTD", "CART", "DASH", "CAVA", "CMG", "BROS", "ELF",
    "CELH", "DECK", "LULU", "NKE", "HD", "LOW", "TGT", "WMT", "COST",
    "JPM", "BAC", "C", "GS", "MS", "SCHW", "AXP", "V", "MA", "COF",
    "LLY", "UNH", "VRTX", "REGN", "ISRG", "TMDX", "VKTX", "HIMS", "TEM",
    "RXRX", "MRNA", "PFE", "ABBV", "JNJ", "TMO", "DHR", "XOM", "CVX",
    "OXY", "SLB", "COP", "LNG", "FCX", "SCCO", "NEM", "GOLD", "AA", "CLF",
    "MP", "ALB", "LAC", "CCJ", "LEU", "SMR", "OKLO", "NNE", "CEG", "GEV",
    "ETN", "PWR", "EME", "URI", "CAT", "DE", "BA", "GE", "RTX", "LMT",
    "NOC", "RKLB", "ASTS", "LUNR", "ACHR", "JOBY", "IONQ", "RGTI", "QBTS",
    "QUBT", "MARA", "RIOT", "CLSK", "HUT", "BTDR", "CVNA", "GME", "AMC",
]

LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "BRK-B", "LLY", "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG",
    "NFLX", "JNJ", "ABBV", "BAC", "KO", "PLTR", "PM", "ORCL", "CRM", "WMT",
    "CVX", "CSCO", "ABT", "IBM", "GE", "MRK", "TMO", "MCD", "NOW", "ISRG",
    "ACN", "LIN", "AMD", "DIS", "AXP", "MS", "GS", "RTX", "QCOM", "CAT",
    "INTU", "UBER", "VZ", "BKNG", "TXN", "AMAT", "SPGI", "PGR", "BSX",
    "NEE", "BLK", "LOW", "TJX", "C", "SYK", "UNP", "HON", "DHR", "ADBE",
]

GROWTH_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "ARM", "PLTR", "APP", "CRDO", "ALAB", "SMCI",
    "VRT", "ANET", "MU", "MRVL", "TSM", "ASML", "LRCX", "KLAC", "AMAT",
    "NET", "DDOG", "MDB", "SNOW", "CRWD", "PANW", "ZS", "OKTA", "HIMS",
    "TEM", "RXRX", "VKTX", "TMDX", "SHOP", "UBER", "ABNB", "DASH", "RDDT",
    "RBLX", "COIN", "HOOD", "MSTR", "SOFI", "AFRM", "CAVA", "ELF", "CELH",
    "SPOT", "TTD", "PINS", "RKLB", "ASTS", "IONQ", "RGTI", "QBTS", "SOUN",
]

THEME_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "SMCI", "VRT", "CEG", "GEV", "ETN", "PWR", "EME",
    "CRDO", "ALAB", "ARM", "PLTR", "APP", "NET", "DDOG", "CRWD", "PANW",
    "MP", "ALB", "LAC", "FCX", "SCCO", "CCJ", "LEU", "SMR", "OKLO", "NNE",
    "RKLB", "ASTS", "LUNR", "ACHR", "JOBY", "IONQ", "RGTI", "QBTS", "QUBT",
    "COIN", "MSTR", "HOOD", "MARA", "RIOT", "CLSK", "HUT", "BTDR",
]

UNIVERSES = {
    "core": CORE_UNIVERSE,
    "large_cap": LARGE_CAP_UNIVERSE,
    "growth": GROWTH_UNIVERSE,
    "themes": THEME_UNIVERSE,
}


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

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)

    if pd.isna(value):
        return None

    return value


def _clean_ticker(ticker: str) -> str:
    return (
        str(ticker or "")
        .upper()
        .strip()
        .replace(".", "-")
    )


def _dedupe_tickers(tickers: Sequence[str]) -> List[str]:
    seen = set()
    output = []

    for ticker in tickers:
        symbol = _clean_ticker(ticker)

        if not symbol or symbol in seen:
            continue

        seen.add(symbol)
        output.append(symbol)

    return output


def _parse_custom_tickers(value: Optional[str]) -> List[str]:
    if not value:
        return []

    return _dedupe_tickers(
        value
        .replace("\n", ",")
        .replace(" ", ",")
        .split(",")
    )


def _get_universe(universe: str, tickers: Optional[str]) -> Tuple[str, List[str]]:
    custom = _parse_custom_tickers(tickers)

    if custom:
        return "custom", custom[:250]

    key = str(universe or "core").lower().strip()

    if key not in UNIVERSES:
        key = "core"

    return key, _dedupe_tickers(UNIVERSES[key])


def _download_daily_data(tickers: Sequence[str], period: str) -> pd.DataFrame:
    try:
        frame = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            auto_adjust=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except TypeError:
        frame = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )

    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()

    return frame


def _slice_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
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
    output = output.dropna(subset=["High", "Low", "Close", "Volume"])

    return output


def _atr14(data: pd.DataFrame) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

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


def _percentile_rank(values: List[Optional[float]], higher_is_better: bool = True) -> List[float]:
    clean = [value for value in values if value is not None and math.isfinite(value)]

    if not clean:
        return [0.0 for _ in values]

    series = pd.Series(clean)
    ranks = series.rank(pct=True).tolist()

    lookup = {}
    for value, rank in zip(clean, ranks):
        lookup.setdefault(value, rank)

    output = []

    for value in values:
        if value is None or not math.isfinite(value):
            output.append(0.0)
        else:
            rank = lookup.get(value, 0.0)
            output.append(rank if higher_is_better else 1.0 - rank)

    return output


def _calculate_ticker(ticker: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if data.empty or len(data) < 70:
        return None

    data = data.copy()

    data["ATR14"] = _atr14(data)

    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()

    data["Velocity9"] = (data["EMA9"] - data["EMA9"].shift(3)) / data["ATR14"]
    data["Velocity21"] = (data["EMA21"] - data["EMA21"].shift(5)) / data["ATR14"]

    data["Acceleration9"] = data["Velocity9"] - data["Velocity9"].shift(3)
    data["Acceleration21"] = data["Velocity21"] - data["Velocity21"].shift(5)

    data["RVOL"] = data["Volume"] / data["Volume"].rolling(20, min_periods=20).mean()
    data["VolumeMomentum"] = (
        data["Volume"].ewm(span=10, adjust=False).mean()
        / data["Volume"].ewm(span=30, adjust=False).mean()
    )

    range_size = data["High"] - data["Low"]
    data["ClosePosition"] = np.where(
        range_size.abs() > 1e-12,
        (data["Close"] - data["Low"]) / range_size,
        0.5,
    )

    data["DistanceToEMA21"] = (data["Close"] - data["EMA21"]) / data["ATR14"]
    data["DollarVolume20"] = (data["Close"] * data["Volume"]).rolling(
        20,
        min_periods=20,
    ).mean()

    latest = data.iloc[-1]

    close = _finite(latest.get("Close"))
    dollar_volume_20 = _finite(latest.get("DollarVolume20"))
    velocity9 = _finite(latest.get("Velocity9"))
    velocity21 = _finite(latest.get("Velocity21"))
    acceleration9 = _finite(latest.get("Acceleration9"))
    acceleration21 = _finite(latest.get("Acceleration21"))
    rvol = _finite(latest.get("RVOL"))
    volume_momentum = _finite(latest.get("VolumeMomentum"))
    close_position = _finite(latest.get("ClosePosition"))
    distance_to_ema21 = _finite(latest.get("DistanceToEMA21"))
    ema9 = _finite(latest.get("EMA9"))
    ema21 = _finite(latest.get("EMA21"))

    if None in [
        close,
        dollar_volume_20,
        velocity9,
        velocity21,
        acceleration9,
        acceleration21,
        rvol,
        volume_momentum,
        close_position,
        distance_to_ema21,
        ema9,
        ema21,
    ]:
        return None

    setup = (
        close >= 10
        and dollar_volume_20 >= 20_000_000
        and velocity21 < 0
        and acceleration21 > 0
        and velocity9 > velocity21
        and acceleration9 > 0
        and rvol > 1.2
        and close > ema9
        and close_position > 0.6
    )

    watch = (
        close >= 10
        and dollar_volume_20 >= 20_000_000
        and acceleration21 > 0
        and acceleration9 > 0
        and velocity9 > velocity21
        and rvol > 1.0
    )

    if setup:
        signal = "Setup"
    elif watch:
        signal = "Watch"
    else:
        signal = "Developing"

    return {
        "ticker": ticker,
        "date": data.index[-1].date().isoformat(),
        "signal": signal,
        "setup": bool(setup),
        "watch": bool(watch),
        "close": close,
        "ema9": ema9,
        "ema21": ema21,
        "velocity9": velocity9,
        "velocity21": velocity21,
        "acceleration9": acceleration9,
        "acceleration21": acceleration21,
        "rvol": rvol,
        "volume_momentum": volume_momentum,
        "close_position": close_position,
        "distance_to_ema21": distance_to_ema21,
        "dollar_volume_20": dollar_volume_20,
    }


def _score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acceleration21_ranks = _percentile_rank([row["acceleration21"] for row in rows])
    acceleration9_ranks = _percentile_rank([row["acceleration9"] for row in rows])
    rvol_ranks = _percentile_rank([row["rvol"] for row in rows])
    volume_momentum_ranks = _percentile_rank([row["volume_momentum"] for row in rows])
    close_position_ranks = _percentile_rank([row["close_position"] for row in rows])

    # Slightly prefer stocks close to EMA21 from below/nearby rather than wildly
    # extended. This keeps the 5-point component from rewarding exhaustion moves.
    distance_values = []
    for row in rows:
        distance = row["distance_to_ema21"]
        distance_values.append(-abs(distance - 0.25))

    distance_ranks = _percentile_rank(distance_values)

    for index, row in enumerate(rows):
        score = (
            30 * acceleration21_ranks[index]
            + 25 * acceleration9_ranks[index]
            + 20 * rvol_ranks[index]
            + 10 * volume_momentum_ranks[index]
            + 10 * close_position_ranks[index]
            + 5 * distance_ranks[index]
        )

        if row["setup"]:
            score += 7
        elif row["watch"]:
            score += 3

        row["score"] = round(min(score, 100.0), 2)
        row["score_components"] = {
            "acceleration21_rank": round(acceleration21_ranks[index], 4),
            "acceleration9_rank": round(acceleration9_ranks[index], 4),
            "rvol_rank": round(rvol_ranks[index], 4),
            "volume_momentum_rank": round(volume_momentum_ranks[index], 4),
            "close_position_rank": round(close_position_ranks[index], 4),
            "distance_to_ema21_rank": round(distance_ranks[index], 4),
        }

    rows.sort(
        key=lambda item: (
            item["setup"],
            item["watch"],
            item["score"],
        ),
        reverse=True,
    )

    return rows


def _scan(
    universe_key: str,
    tickers: Sequence[str],
    period: str,
    limit: int,
    setups_only: bool,
    min_score: float,
) -> Dict[str, Any]:
    frame = _download_daily_data(tickers, period=period)

    rows = []
    failed = []

    for ticker in tickers:
        ticker_frame = _slice_ticker_frame(frame, ticker)

        try:
            row = _calculate_ticker(ticker, ticker_frame)
        except Exception:
            row = None

        if row is None:
            failed.append(ticker)
        else:
            rows.append(row)

    rows = _score_rows(rows)

    if setups_only:
        rows = [row for row in rows if row["setup"]]

    rows = [row for row in rows if row["score"] >= min_score]
    rows = rows[:limit]

    generated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "generated_at": generated_at,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "universe": universe_key,
        "period": period,
        "requested_tickers": len(tickers),
        "scanned_tickers": len(rows),
        "failed_tickers": failed[:50],
        "rows": [
            {
                key: _clean(value)
                for key, value in row.items()
            }
            for row in rows
        ],
        "methodology": {
            "purpose": (
                "Find stocks where normalized EMA velocity and acceleration "
                "are shifting from weak or negative to improving, with volume confirmation."
            ),
            "setup_condition": {
                "close": ">= 10",
                "dollar_volume_20": ">= 20,000,000",
                "velocity21": "< 0",
                "acceleration21": "> 0",
                "velocity9": "> velocity21",
                "acceleration9": "> 0",
                "rvol": "> 1.2",
                "close": "> ema9",
                "close_position": "> 0.6",
            },
            "score": {
                "acceleration21_rank": 30,
                "acceleration9_rank": 25,
                "rvol_rank": 20,
                "volume_momentum_rank": 10,
                "close_position_rank": 10,
                "distance_to_ema21_rank": 5,
                "setup_bonus": 7,
                "watch_bonus": 3,
            },
        },
    }


@router.get("")
def get_ema_inflection_scan(
    universe: str = Query(default="core"),
    period: str = Query(default="1y", pattern="^(6mo|9mo|1y)$"),
    limit: int = Query(default=75, ge=10, le=MAX_LIMIT),
    setups_only: bool = Query(default=False),
    min_score: float = Query(default=0, ge=0, le=100),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    universe_key, ticker_list = _get_universe(universe, tickers)

    cache_key = (
        f"ema-inflection:{universe_key}:{','.join(ticker_list)}:"
        f"{period}:{limit}:{setups_only}:{min_score}"
    )

    if not refresh:
        cached = _cache_get(cache_key)

        if cached is not None:
            return {
                **cached,
                "cached": True,
            }

    payload = _scan(
        universe_key=universe_key,
        tickers=ticker_list,
        period=period,
        limit=limit,
        setups_only=setups_only,
        min_score=min_score,
    )

    payload["cached"] = False

    return _cache_set(cache_key, payload, CACHE_TTL_SECONDS)


@router.get("/status")
def get_ema_inflection_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "route": "/api/ema-inflection",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "universes": sorted(UNIVERSES.keys()),
        "default_universe": "core",
    }
