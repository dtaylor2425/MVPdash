from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.db import get_connection

NY_TZ = ZoneInfo("America/New_York")

STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stock_alpha": {
        "label": "Stock Alpha Portfolio",
        "min_holdings": 6,
        "max_ticker_change_ratio": 0.80,
        "max_turnover": 0.85,
        "max_cash_weight": 0.95,
        "config": {
            "universe": "quality",
            "target_holdings": 10,
            "min_score": 60,
            "max_tickers": 95,
        },
    },
    "smid_growth": {
        "label": "High-Growth SMID Portfolio",
        "min_holdings": 8,
        "max_ticker_change_ratio": 0.85,
        "max_turnover": 0.90,
        "max_cash_weight": 0.95,
        "config": {
            "target_holdings": 15,
            "min_score": 50,
            "max_tickers": 60,
        },
    },
}


def _now_et() -> datetime:
    return datetime.now(tz=NY_TZ)


def _market_is_open(run_date: date) -> bool:
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=run_date.isoformat(), end_date=run_date.isoformat())
        return not schedule.empty
    except Exception:
        return run_date.weekday() < 5


def _json_sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _get_ticker(row: Dict[str, Any]) -> str:
    return str(row.get("ticker") or row.get("symbol") or row.get("asset") or "").upper().strip()


def _get_score(row: Dict[str, Any]) -> Optional[float]:
    for key in ["stock_intelligence_score", "confluence_score", "composite_score", "score", "total_score"]:
        if row.get(key) is not None:
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                return None
    return None


def _positions_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    holdings = payload.get("holdings") or []
    if not isinstance(holdings, list):
        return []
    return [row for row in holdings if isinstance(row, dict) and _get_ticker(row)]


def _performance_series_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    performance = payload.get("performance") or {}
    if not isinstance(performance, dict):
        return []
    series = performance.get("series") or []
    if not isinstance(series, list):
        return []
    return [row for row in series if isinstance(row, dict) and row.get("date")]


def _position_weight(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("target_weight") or row.get("weight") or row.get("allocation") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ticker_weights(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    return {_get_ticker(row): _position_weight(row) for row in positions if _get_ticker(row)}


def _load_latest_published(strategy: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, strategy, run_date, as_of_timestamp, payload, diagnostics
                FROM portfolio_runs
                WHERE strategy = %s
                  AND status = 'published'
                  AND is_published = TRUE
                ORDER BY run_date DESC, as_of_timestamp DESC, created_at DESC
                LIMIT 1
                """,
                (strategy,),
            )
            return cur.fetchone()


def _official_rebalance_diff(
    old_payload: Optional[Dict[str, Any]],
    new_payload: Dict[str, Any],
    rebalance_date: date,
) -> Dict[str, Any]:
    old_weights = _ticker_weights(_positions_from_payload(old_payload or {}))
    new_weights = _ticker_weights(_positions_from_payload(new_payload))
    old_tickers = set(old_weights)
    new_tickers = set(new_weights)
    buys = sorted(new_tickers - old_tickers)
    sells = sorted(old_tickers - new_tickers)
    adds: List[str] = []
    trims: List[str] = []

    for ticker in sorted(old_tickers & new_tickers):
        delta = new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)
        if delta > 0.015:
            adds.append(ticker)
        elif delta < -0.015:
            trims.append(ticker)

    turnover = 0.5 * sum(
        abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
        for t in sorted(old_tickers | new_tickers)
    )

    parts: List[str] = []
    if buys:
        parts.append("Bought " + ", ".join(buys[:4]))
    if sells:
        parts.append("Removed " + ", ".join(sells[:4]))
    if not parts and adds:
        parts.append("Added to " + ", ".join(adds[:4]))
    if not parts and trims:
        parts.append("Trimmed " + ", ".join(trims[:4]))
    if not parts:
        parts.append("No major changes")

    return {
        "rebalance_date": rebalance_date.isoformat(),
        "headline": "; ".join(parts),
        "buys": buys,
        "sells": sells,
        "adds": adds,
        "trims": trims,
        "turnover": turnover,
        "old_holdings": sorted(
            [{"ticker": t, "weight": w} for t, w in old_weights.items()],
            key=lambda x: x["weight"],
            reverse=True,
        ),
        "new_holdings": sorted(
            [{"ticker": t, "weight": w} for t, w in new_weights.items()],
            key=lambda x: x["weight"],
            reverse=True,
        ),
        "ticker_change_ratio": (len(buys) + len(sells)) / max(1, len(old_tickers | new_tickers)),
    }


def _validate_payload(
    strategy: str,
    payload: Dict[str, Any],
    previous_payload: Optional[Dict[str, Any]],
    official_rebalance: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    cfg = STRATEGY_CONFIGS[strategy]
    errors: List[str] = []
    positions = _positions_from_payload(payload)
    series = _performance_series_from_payload(payload)

    if len(positions) < int(cfg["min_holdings"]):
        errors.append(f"Too few holdings: {len(positions)} < {cfg['min_holdings']}")
    if not series or len(series) < 40:
        errors.append(f"Performance series missing or too short: {len(series)} rows")

    try:
        cash = float(payload.get("cash_weight") or 0.0)
    except (TypeError, ValueError):
        cash = 0.0

    if cash > float(cfg["max_cash_weight"]):
        errors.append(f"Cash weight too high: {cash:.2%}")

    if previous_payload:
        turnover = float(official_rebalance.get("turnover") or 0.0)
        ticker_change_ratio = float(official_rebalance.get("ticker_change_ratio") or 0.0)
        if turnover > float(cfg["max_turnover"]):
            errors.append(f"Turnover guardrail tripped: {turnover:.2%}")
        if ticker_change_ratio > float(cfg["max_ticker_change_ratio"]):
            errors.append(f"Ticker-change guardrail tripped: {ticker_change_ratio:.2%}")

    return len(errors) == 0, errors


def _build_strategy_payload(strategy: str) -> Dict[str, Any]:
    cfg = STRATEGY_CONFIGS[strategy]["config"]
    if strategy == "stock_alpha":
        from api.routers.stock_portfolio import _build_portfolio
        return _build_portfolio(
            universe=cfg["universe"],
            tickers=None,
            max_tickers=int(cfg["max_tickers"]),
            target_holdings=int(cfg["target_holdings"]),
            min_score=float(cfg["min_score"]),
        )
    if strategy == "smid_growth":
        from api.routers.smid_growth_portfolio import _build_payload
        return _build_payload(
            target_holdings=int(cfg["target_holdings"]),
            max_tickers=int(cfg["max_tickers"]),
            tickers=None,
            min_score=float(cfg["min_score"]),
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def _insert_run(
    strategy: str,
    run_date: date,
    as_of_timestamp: datetime,
    status: str,
    is_published: bool,
    config: Dict[str, Any],
    diagnostics: Dict[str, Any],
    payload: Dict[str, Any],
    failure_reason: Optional[str],
) -> str:
    run_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO portfolio_runs (
                  id, strategy, run_date, as_of_timestamp, status, is_published,
                  config, diagnostics, payload, failure_reason, published_at
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  CASE WHEN %s = TRUE THEN now() ELSE NULL END
                )
                """,
                (
                    run_id,
                    strategy,
                    run_date,
                    as_of_timestamp,
                    status,
                    is_published,
                    Jsonb(config),
                    Jsonb(diagnostics),
                    Jsonb(payload),
                    failure_reason,
                    is_published,
                ),
            )
        conn.commit()
    return run_id


def _store_child_rows(run_id: str, payload: Dict[str, Any], official_rebalance: Dict[str, Any]) -> None:
    positions = _positions_from_payload(payload)
    series = _performance_series_from_payload(payload)

    with get_connection() as conn:
        with conn.cursor() as cur:
            for row in positions:
                scores = {
                    k: v
                    for k, v in row.items()
                    if k.endswith("_score") or k in ["stock_intelligence_score", "confluence_score", "portfolio_alpha", "momentum_score"]
                }
                cur.execute(
                    """
                    INSERT INTO portfolio_positions (
                      run_id, ticker, name, sector, theme, target_weight, score,
                      action, reason, stop_level, risk_state, scores, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        _get_ticker(row),
                        row.get("name"),
                        row.get("sector"),
                        row.get("theme"),
                        _position_weight(row),
                        _get_score(row),
                        row.get("action"),
                        row.get("trade_reason") or row.get("reason"),
                        row.get("stop_level"),
                        row.get("risk_state"),
                        Jsonb(_json_sanitize(scores)),
                        Jsonb(_json_sanitize(row)),
                    ),
                )

            for row in series:
                cur.execute(
                    """
                    INSERT INTO portfolio_performance (
                      run_id, performance_date, model_return, benchmark_return, rebalance_marker
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        row.get("date"),
                        row.get("model"),
                        row.get("benchmark"),
                        Jsonb(_json_sanitize(row.get("rebalance"))),
                    ),
                )

            cur.execute(
                """
                INSERT INTO portfolio_rebalances (
                  run_id, rebalance_date, headline, buys, sells, adds, trims,
                  turnover, old_holdings, new_holdings
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    official_rebalance["rebalance_date"],
                    official_rebalance["headline"],
                    Jsonb(official_rebalance["buys"]),
                    Jsonb(official_rebalance["sells"]),
                    Jsonb(official_rebalance["adds"]),
                    Jsonb(official_rebalance["trims"]),
                    official_rebalance["turnover"],
                    Jsonb(official_rebalance["old_holdings"]),
                    Jsonb(official_rebalance["new_holdings"]),
                ),
            )
        conn.commit()


def _mark_failed(run_id: str, reason: str, diagnostics: Dict[str, Any]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE portfolio_runs
                SET status = 'failed', is_published = FALSE,
                    failure_reason = %s, diagnostics = %s
                WHERE id = %s
                """,
                (reason, Jsonb(_json_sanitize(diagnostics)), run_id),
            )
        conn.commit()


def _publish_run(run_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE portfolio_runs SET status = 'published', is_published = TRUE, published_at = now() WHERE id = %s",
                (run_id,),
            )
        conn.commit()


def _run_strategy(strategy: str, run_date: date, dry_run: bool) -> None:
    print(f"[{strategy}] Building portfolio snapshot for {run_date.isoformat()}")
    previous = _load_latest_published(strategy)
    previous_payload = previous["payload"] if previous else None
    payload = _json_sanitize(_build_strategy_payload(strategy))
    official = _official_rebalance_diff(previous_payload, payload, run_date)
    payload["official_rebalance"] = official

    valid, errors = _validate_payload(strategy, payload, previous_payload, official)
    diagnostics = {
        "strategy": strategy,
        "run_date": run_date.isoformat(),
        "valid": valid,
        "errors": errors,
        "official_rebalance": official,
        "previous_run_id": str(previous["id"]) if previous else None,
        "holding_count": len(_positions_from_payload(payload)),
        "performance_rows": len(_performance_series_from_payload(payload)),
    }

    if dry_run:
        print(json.dumps(diagnostics, indent=2))
        return

    config = {
        **STRATEGY_CONFIGS[strategy]["config"],
        "strategy": strategy,
        "label": STRATEGY_CONFIGS[strategy]["label"],
    }
    run_id = _insert_run(
        strategy,
        run_date,
        _now_et(),
        "draft",
        False,
        _json_sanitize(config),
        _json_sanitize(diagnostics),
        payload,
        None,
    )

    if not valid:
        reason = "; ".join(errors)
        _mark_failed(run_id, reason, diagnostics)
        print(f"[{strategy}] FAILED guardrails: {reason}")
        return

    _store_child_rows(run_id, payload, official)
    _publish_run(run_id)
    print(f"[{strategy}] Published {run_id}. Holdings={diagnostics['holding_count']} Turnover={official['turnover']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=sorted(STRATEGY_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-date")
    args = parser.parse_args()

    if not args.all and not args.strategy:
        raise SystemExit("Use --all or --strategy <name>")

    run_date = date.fromisoformat(args.run_date) if args.run_date else _now_et().date()

    if not args.force and not _market_is_open(run_date):
        print(f"NYSE is not open on {run_date.isoformat()}. Skipping. Use --force to override.")
        return

    strategies = list(STRATEGY_CONFIGS.keys()) if args.all else [args.strategy]

    for strategy in strategies:
        try:
            _run_strategy(strategy, run_date, args.dry_run)
        except Exception as exc:
            print(f"[{strategy}] ERROR: {exc}")
            if not args.dry_run:
                _insert_run(
                    strategy,
                    run_date,
                    _now_et(),
                    "failed",
                    False,
                    STRATEGY_CONFIGS[strategy]["config"],
                    {"strategy": strategy, "run_date": run_date.isoformat(), "error": str(exc)},
                    {"error": str(exc)},
                    str(exc),
                )


if __name__ == "__main__":
    main()
