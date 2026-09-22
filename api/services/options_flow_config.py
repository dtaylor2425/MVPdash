"""
api/services/options_flow_config.py

Universe + tunables for Macro Options Flow. Everything is env-overridable so
the worker and the API agree without code changes.

    OPTIONS_FLOW_UNIVERSE   one of:
        JSON     {"INDEX": ["SPY", "QQQ"], "TECH": ["XLK"]}
        grouped  INDEX:SPY,QQQ;TECH:XLK,SMH
        flat     SPY,QQQ,XLK           (group taken from the default map, else "CUSTOM")

No ThetaData import here -- this module is safe on the Python 3.11 API service.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

DEFAULT_UNIVERSE: Dict[str, List[str]] = {
    "INDEX": ["SPY", "QQQ", "IWM", "DIA"],
    "TECH": ["XLK", "SMH", "IGV", "XLC"],
    "CYCLICAL": ["XLF", "XLI", "XLE", "XLY"],
    "DEFENSIVE": ["XLV", "XLP", "XLU"],
    "RATES / CREDIT": ["TLT", "IEF", "HYG", "LQD"],
    "REAL ASSETS": ["GLD", "SLV", "USO"],
}

# ThetaData Options Standard allows 4 concurrent requests. Hard ceiling --
# the env var can lower it but never raise it.
THETA_HARD_MAX_CONCURRENCY = 4


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _group_lookup() -> Dict[str, str]:
    return {t: g for g, ts in DEFAULT_UNIVERSE.items() for t in ts}


def parse_universe(raw: Optional[str]) -> Dict[str, List[str]]:
    if raw is None or not raw.strip():
        return {g: list(ts) for g, ts in DEFAULT_UNIVERSE.items()}
    raw = raw.strip()

    if raw.startswith("{"):
        parsed = json.loads(raw)
        return {str(g): [str(t).strip().upper() for t in ts] for g, ts in parsed.items()}

    if ":" in raw:
        out: Dict[str, List[str]] = {}
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            group, _, tickers = chunk.partition(":")
            out[group.strip().upper()] = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        return out

    lookup = _group_lookup()
    flat: Dict[str, List[str]] = {}
    for t in (x.strip().upper() for x in raw.split(",")):
        if t:
            flat.setdefault(lookup.get(t, "CUSTOM"), []).append(t)
    return flat


def load_universe() -> Dict[str, List[str]]:
    return parse_universe(os.getenv("OPTIONS_FLOW_UNIVERSE"))


def ticker_groups(universe: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    universe = universe or load_universe()
    return {t: g for g, ts in universe.items() for t in ts}


def load_config() -> Dict[str, object]:
    """Numeric tunables, recorded verbatim in options_flow_runs.config."""
    return {
        "maxDte": _env_int("OPTIONS_FLOW_MAX_DTE", 60),
        "strikeRange": _env_int("OPTIONS_FLOW_STRIKE_RANGE", 25),
        "greekInterval": os.getenv("OPTIONS_FLOW_GREEK_INTERVAL", "1m"),
        "greekToleranceMin": _env_int("OPTIONS_FLOW_GREEK_TOLERANCE_MIN", 15),
        "bucketMinutes": _env_int("OPTIONS_FLOW_BUCKET_MIN", 15),
        "rollingBuckets": _env_int("OPTIONS_FLOW_ROLLING_BUCKETS", 4),
        "largeTradeCount": _env_int("OPTIONS_FLOW_LARGE_TRADES", 25),
        "minEligiblePremium": _env_float("OPTIONS_FLOW_MIN_ELIGIBLE_PREMIUM", 250_000.0),
        "minEligibleTrades": _env_int("OPTIONS_FLOW_MIN_ELIGIBLE_TRADES", 25),
        "minPremiumCoverage": _env_float("OPTIONS_FLOW_MIN_PREMIUM_COVERAGE", 0.5),
        "ivPercentileMinObs": _env_int("OPTIONS_FLOW_IV_PCTL_MIN_OBS", 20),
        "ivHistoryDays": _env_int("OPTIONS_FLOW_IV_HISTORY_DAYS", 252),
        "liveMaxAgeMin": _env_int("OPTIONS_FLOW_LIVE_MAX_AGE_MIN", 20),
        "staleAfterMin": _env_int("OPTIONS_FLOW_STALE_AFTER_MIN", 60),
        "intradayRetentionDays": _env_int("OPTIONS_FLOW_INTRADAY_RETENTION_DAYS", 5),
        "thetaConcurrency": max(
            1,
            min(
                _env_int("OPTIONS_FLOW_THETA_CONCURRENCY", THETA_HARD_MAX_CONCURRENCY),
                THETA_HARD_MAX_CONCURRENCY,
            ),
        ),
        # ETF options trade until 16:15 ET.
        "sessionStart": "09:30:00",
        "sessionEnd": "16:15:00",
    }
