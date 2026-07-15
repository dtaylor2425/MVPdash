"""
EMA Inflection Scanner API

FastAPI routes:
    GET /api/ema-inflection
    GET /api/ema-inflection/status

Purpose:
    Catch the curl before the trend is obvious.

This scanner is independent from Macro Engine macro regime data. It only uses
daily OHLCV data from Yahoo Finance through yfinance, then calculates EMA
velocity, EMA acceleration, volume confirmation, close strength, and liquidity.
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
    "CLSK", "MARA", "RIOT", "HUT", "BTDR",
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


def _clean_ticker(ticker: str) -> str:
    return str(ticker or "").upper().strip().replace(".", "-")


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


def _trend_phase(
    velocity9: float,
    velocity21: float,
    acceleration9: float,
    acceleration21: float,
    close: float,
    ema9: float,
    ema21: float,
    ema9_curling_up: bool,
    short_term_reclaim: bool,
) -> str:
    if (
        velocity21 < 0
        and acceleration21 > 0
        and acceleration9 > 0
        and close > ema9
    ):
        return "Falling to Improving"

    if velocity21 < 0 and acceleration21 > 0 and ema9_curling_up:
        return "Early Curl"

    if velocity21 < 0 and acceleration21 > 0:
        return "Downtrend Decelerating"

    if short_term_reclaim and close < ema21:
        return "Short-Term Reclaim"

    if velocity9 > 0 and velocity21 > 0 and close > ema9 and close > ema21:
        return "Rising Trend"

    if velocity9 < 0 and velocity21 < 0 and acceleration21 <= 0:
        return "Still Deteriorating"

    if close > ema9 and close > ema21:
        return "Reclaiming Trend"

    return "Mixed"


def _inflection_quality(
    setup: bool,
    early_curl: bool,
    ema9_curling_up: bool,
    ema21_decelerating_downtrend: bool,
    short_term_reclaim: bool,
    rvol: float,
    close_position: float,
) -> str:
    if setup and short_term_reclaim and rvol >= 1.5:
        return "High Conviction Curl"

    if setup:
        return "Inflection Setup"

    if early_curl and rvol >= 1.2:
        return "Volume-Confirmed Curl"

    if early_curl:
        return "Early Curl"

    if ema21_decelerating_downtrend:
        return "Downtrend Losing Force"

    if ema9_curling_up or short_term_reclaim:
        return "Needs Confirmation"

    return "Low Quality"


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
    data["DistanceToEMA9"] = (data["Close"] - data["EMA9"]) / data["ATR14"]

    data["DollarVolume20"] = (data["Close"] * data["Volume"]).rolling(
        20,
        min_periods=20,
    ).mean()

    latest = data.iloc[-1]
    previous = data.iloc[-2]
    two_back = data.iloc[-3]

    close = _finite(latest.get("Close"))
    previous_close = _finite(previous.get("Close"))
    dollar_volume_20 = _finite(latest.get("DollarVolume20"))
    velocity9 = _finite(latest.get("Velocity9"))
    velocity21 = _finite(latest.get("Velocity21"))
    previous_velocity9 = _finite(previous.get("Velocity9"))
    two_back_velocity9 = _finite(two_back.get("Velocity9"))
    acceleration9 = _finite(latest.get("Acceleration9"))
    acceleration21 = _finite(latest.get("Acceleration21"))
    rvol = _finite(latest.get("RVOL"))
    volume_momentum = _finite(latest.get("VolumeMomentum"))
    close_position = _finite(latest.get("ClosePosition"))
    distance_to_ema21 = _finite(latest.get("DistanceToEMA21"))
    distance_to_ema9 = _finite(latest.get("DistanceToEMA9"))
    ema9 = _finite(latest.get("EMA9"))
    previous_ema9 = _finite(previous.get("EMA9"))
    ema21 = _finite(latest.get("EMA21"))

    if None in [
        close,
        previous_close,
        dollar_volume_20,
        velocity9,
        velocity21,
        previous_velocity9,
        two_back_velocity9,
        acceleration9,
        acceleration21,
        rvol,
        volume_momentum,
        close_position,
        distance_to_ema21,
        distance_to_ema9,
        ema9,
        previous_ema9,
        ema21,
    ]:
        return None

    ema9_curling_up = (
        velocity9 > previous_velocity9
        and velocity9 > two_back_velocity9
    )

    ema21_decelerating_downtrend = (
        velocity21 < 0
        and acceleration21 > 0
    )

    short_term_reclaim = (
        close > ema9
        and previous_close <= previous_ema9
    )

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

    early_curl = (
        close >= 10
        and dollar_volume_20 >= 20_000_000
        and ema21_decelerating_downtrend
        and acceleration9 > 0
        and velocity9 > velocity21
        and ema9_curling_up
        and close_position > 0.45
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
        signal = "Inflection Setup"
    elif early_curl:
        signal = "Early Curl"
    elif watch:
        signal = "Watch"
    else:
        signal = "Developing"

    trend_phase = _trend_phase(
        velocity9=velocity9,
        velocity21=velocity21,
        acceleration9=acceleration9,
        acceleration21=acceleration21,
        close=close,
        ema9=ema9,
        ema21=ema21,
        ema9_curling_up=ema9_curling_up,
        short_term_reclaim=short_term_reclaim,
    )

    inflection_quality = _inflection_quality(
        setup=setup,
        early_curl=early_curl,
        ema9_curling_up=ema9_curling_up,
        ema21_decelerating_downtrend=ema21_decelerating_downtrend,
        short_term_reclaim=short_term_reclaim,
        rvol=rvol,
        close_position=close_position,
    )

    return {
        "ticker": ticker,
        "date": data.index[-1].date().isoformat(),
        "signal": signal,
        "trend_phase": trend_phase,
        "inflection_quality": inflection_quality,
        "setup": bool(setup),
        "early_curl": bool(early_curl),
        "watch": bool(watch),
        "ema9_curling_up": bool(ema9_curling_up),
        "ema21_decelerating_downtrend": bool(ema21_decelerating_downtrend),
        "short_term_reclaim": bool(short_term_reclaim),
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
        "distance_to_ema9": distance_to_ema9,
        "dollar_volume_20": dollar_volume_20,
    }


def _score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acceleration21_ranks = _percentile_rank([row["acceleration21"] for row in rows])
    acceleration9_ranks = _percentile_rank([row["acceleration9"] for row in rows])
    rvol_ranks = _percentile_rank([row["rvol"] for row in rows])
    volume_momentum_ranks = _percentile_rank([row["volume_momentum"] for row in rows])
    close_position_ranks = _percentile_rank([row["close_position"] for row in rows])

    distance_values = []

    for row in rows:
        distance = row["distance_to_ema21"]
        distance_values.append(-abs(distance - 0.25))

    distance_ranks = _percentile_rank(distance_values)

    for index, row in enumerate(rows):
        base_score = (
            30 * acceleration21_ranks[index]
            + 25 * acceleration9_ranks[index]
            + 20 * rvol_ranks[index]
            + 10 * volume_momentum_ranks[index]
            + 10 * close_position_ranks[index]
            + 5 * distance_ranks[index]
        )

        bonus = 0.0

        if row["ema21_decelerating_downtrend"]:
            bonus += 5

        if row["acceleration9"] > 0 and row["velocity9"] > row["velocity21"]:
            bonus += 5

        if row["short_term_reclaim"]:
            bonus += 5

        if row["ema9_curling_up"]:
            bonus += 3

        if row["rvol"] > 1.5:
            bonus += 3

        if row["setup"]:
            bonus += 7
        elif row["early_curl"]:
            bonus += 4
        elif row["watch"]:
            bonus += 2

        row["score"] = round(min(base_score + bonus, 100.0), 2)
        row["score_components"] = {
            "acceleration21_rank": round(acceleration21_ranks[index], 4),
            "acceleration9_rank": round(acceleration9_ranks[index], 4),
            "rvol_rank": round(rvol_ranks[index], 4),
            "volume_momentum_rank": round(volume_momentum_ranks[index], 4),
            "close_position_rank": round(close_position_ranks[index], 4),
            "distance_to_ema21_rank": round(distance_ranks[index], 4),
            "inflection_bonus": round(bonus, 2),
        }

    signal_order = {
        "Inflection Setup": 4,
        "Early Curl": 3,
        "Watch": 2,
        "Developing": 1,
    }

    phase_order = {
        "Falling to Improving": 6,
        "Early Curl": 5,
        "Downtrend Decelerating": 4,
        "Short-Term Reclaim": 3,
        "Reclaiming Trend": 2,
        "Rising Trend": 1,
        "Mixed": 0,
        "Still Deteriorating": -1,
    }

    rows.sort(
        key=lambda item: (
            signal_order.get(item["signal"], 0),
            phase_order.get(item["trend_phase"], 0),
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
        rows = [
            row
            for row in rows
            if row["signal"] == "Inflection Setup"
        ]

    rows = [row for row in rows if row["score"] >= min_score]
    rows = rows[:limit]

    generated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "generated_at": generated_at,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "universe": universe_key,
        "period": period,
        "requested_tickers": len(tickers),
        "returned_rows": len(rows),
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
                "Catch stocks at the moment the EMA structure stops deteriorating "
                "and begins to curl higher, before the broader trend fully confirms."
            ),
            "plain_english": (
                "The stock was falling or weak, but the moving-average structure "
                "is starting to lose downside force. EMA21 velocity can still be "
                "negative, but acceleration must be improving. EMA9 should curl "
                "higher first, price should reclaim short-term trend, and volume "
                "should confirm the shift."
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
                "ema21_decelerating_downtrend_bonus": 5,
                "short_term_acceleration_bonus": 5,
                "short_term_reclaim_bonus": 5,
                "ema9_curl_bonus": 3,
                "rvol_above_1_5_bonus": 3,
                "inflection_setup_bonus": 7,
                "early_curl_bonus": 4,
                "watch_bonus": 2,
            },
        },
    }


@router.get("")
def get_ema_inflection_scan(
    universe: str = Query(default="core"),
    period: str = Query(default="1y", pattern="^(6mo|9mo|1y)$"),
    limit: int = Query(default=75, ge=10, le=MAX_LIMIT),
    setups_only: bool = Query(default=False),
    min_score: float = Query(default=45, ge=0, le=100),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    universe_key, ticker_list = _get_universe(universe, tickers)

    cache_key = (
        f"ema-inflection-curl:{universe_key}:{','.join(ticker_list)}:"
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
        "signal_labels": [
            "Inflection Setup",
            "Early Curl",
            "Watch",
            "Developing",
        ],
        "trend_phases": [
            "Falling to Improving",
            "Early Curl",
            "Downtrend Decelerating",
            "Short-Term Reclaim",
            "Rising Trend",
            "Reclaiming Trend",
            "Mixed",
            "Still Deteriorating",
        ],
    }
