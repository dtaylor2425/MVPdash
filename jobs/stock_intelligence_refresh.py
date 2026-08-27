from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
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


def _ensure_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_intelligence_runs (
                  id UUID PRIMARY KEY,
                  run_date DATE NOT NULL,
                  as_of_timestamp TIMESTAMPTZ NOT NULL,
                  status TEXT NOT NULL CHECK (status IN ('draft', 'published', 'failed')),
                  is_published BOOLEAN NOT NULL DEFAULT FALSE,
                  config JSONB NOT NULL DEFAULT '{}'::jsonb,
                  diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
                  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                  failure_reason TEXT,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  published_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stock_intelligence_runs_latest
                ON stock_intelligence_runs(is_published, status, run_date DESC, as_of_timestamp DESC, created_at DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stock_intelligence_runs_date
                ON stock_intelligence_runs(run_date DESC, as_of_timestamp DESC)
                """
            )
        conn.commit()


def _first(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row.get(key)
    return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _score(row: Dict[str, Any]) -> Optional[float]:
    return _safe_float(
        _first(
            row,
            [
                "stock_intelligence_score",
                "confluence_score",
                "composite_score",
                "alpha_score",
                "total_score",
                "score",
            ],
        )
    )


def _ticker(row: Dict[str, Any]) -> str:
    return str(_first(row, ["ticker", "symbol", "asset"], "") or "").upper().strip()


def _call_dynamic(func: Any, candidates: List[Dict[str, Any]]) -> Any:
    last_error: Optional[Exception] = None
    for kwargs in candidates:
        try:
            signature = inspect.signature(func)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            if accepts_kwargs:
                result = func(**kwargs)
            else:
                allowed = {k: v for k, v in kwargs.items() if k in signature.parameters}
                result = func(**allowed)

            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error

    return func()


def _load_universe(rankings_module: Any, universe: str, max_tickers: int, tickers: Optional[List[str]]) -> Optional[List[str]]:
    if tickers:
        return [t.upper().strip() for t in tickers if t.strip()]

    get_universe = getattr(rankings_module, "_get_universe", None)
    if not get_universe:
        return None

    candidates = [
        {"universe": universe, "max_tickers": max_tickers},
        {"name": universe, "max_tickers": max_tickers},
        {"universe": universe},
        {"name": universe},
        {},
    ]

    result = _call_dynamic(get_universe, candidates)

    if result is None:
        return None
    if isinstance(result, dict):
        for key in ["tickers", "symbols", "universe", "results"]:
            value = result.get(key)
            if isinstance(value, list):
                return [str(x).upper().strip() for x in value if str(x).strip()]
        return None
    if isinstance(result, (list, tuple, set)):
        return [str(x).upper().strip() for x in result if str(x).strip()]

    return None


def _extract_rows(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []

    if isinstance(result, dict):
        for key in ["rows", "rankings", "results", "stocks", "data", "items"]:
            value = result.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if "ticker" in result or "symbol" in result:
            return [result]
        return []

    if isinstance(result, (list, tuple)):
        return [x for x in result if isinstance(x, dict)]

    return []


def _scan_rankings(
    rankings_module: Any,
    universe: str,
    max_tickers: int,
    limit: int,
    min_score: float,
    tickers: Optional[List[str]],
) -> List[Dict[str, Any]]:
    scan = getattr(rankings_module, "_scan_rankings", None)

    if not scan:
        raise RuntimeError("api.routers.stock_rankings._scan_rankings was not found.")

    universe_tickers = _load_universe(rankings_module, universe, max_tickers, tickers)

    candidates = [
        {
            "universe": universe,
            "tickers": universe_tickers,
            "max_tickers": max_tickers,
            "limit": limit,
            "min_score": min_score,
            "refresh": True,
        },
        {
            "universe": universe,
            "tickers": universe_tickers,
            "max_tickers": max_tickers,
            "min_score": min_score,
            "refresh": True,
        },
        {
            "tickers": universe_tickers,
            "max_tickers": max_tickers,
            "limit": limit,
            "min_score": min_score,
            "refresh": True,
        },
        {
            "tickers": universe_tickers,
            "max_tickers": max_tickers,
            "min_score": min_score,
        },
        {
            "universe": universe,
            "max_tickers": max_tickers,
            "limit": limit,
            "min_score": min_score,
        },
        {
            "universe": universe,
            "max_tickers": max_tickers,
        },
        {},
    ]

    result = _call_dynamic(scan, candidates)
    rows = _extract_rows(result)

    return rows


def _normalize_row(row: Dict[str, Any], rank: int) -> Dict[str, Any]:
    ticker = _ticker(row)
    score = _score(row)

    normalized = {
        "rank": rank,
        "ticker": ticker,
        "name": _first(row, ["name", "company", "company_name", "short_name", "long_name"], ticker),
        "sector": _first(row, ["sector", "industry", "group"], "Unclassified"),
        "theme": _first(row, ["theme", "thematic_fit", "category"], None),
        "price": _safe_float(_first(row, ["price", "last_price", "close", "current_price"])),
        "change_1d": _safe_float(_first(row, ["change_1d", "one_day_change", "pct_change", "daily_change"])),
        "stock_intelligence_score": score,
        "fundamental_score": _safe_float(_first(row, ["fundamental_score", "fundamentals_score"])),
        "technical_score": _safe_float(_first(row, ["technical_score", "technicals_score"])),
        "balance_sheet_score": _safe_float(_first(row, ["balance_sheet_score", "balance_score"])),
        "valuation_score": _safe_float(_first(row, ["valuation_score", "value_score"])),
        "momentum_quality_score": _safe_float(_first(row, ["momentum_quality_score", "momentum_score", "quality_momentum_score"])),
        "signal": _first(row, ["signal", "rating", "setup", "verdict"], None),
        "reason": _first(row, ["reason", "trade_reason", "summary", "thesis", "why"], None),
        "workbook_href": f"/stocks/{ticker}" if ticker else None,
    }

    for key in [
        "market_cap",
        "revenue_growth",
        "earnings_growth",
        "gross_margin",
        "operating_margin",
        "relative_strength",
        "rvol",
        "risk_state",
        "catalyst",
        "updated_at",
    ]:
        if row.get(key) is not None and key not in normalized:
            normalized[key] = _json_sanitize(row.get(key))

    return normalized


def _build_payload(
    universe: str,
    max_tickers: int,
    limit: int,
    min_score: float,
    tickers: Optional[List[str]],
    run_date: date,
) -> Dict[str, Any]:
    rankings_module = importlib.import_module("api.routers.stock_rankings")
    raw_rows = _scan_rankings(rankings_module, universe, max_tickers, limit, min_score, tickers)

    cleaned: List[Dict[str, Any]] = []
    seen = set()
    rows_with_scores = []

    for row in raw_rows:
        ticker = _ticker(row)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        rows_with_scores.append(row)

    rows_with_scores.sort(key=lambda r: (_score(r) is not None, _score(r) or -999.0), reverse=True)

    for idx, row in enumerate(rows_with_scores[:limit], start=1):
        cleaned.append(_normalize_row(row, idx))

    top_score = cleaned[0].get("stock_intelligence_score") if cleaned else None
    average_score = None
    score_values = [r.get("stock_intelligence_score") for r in cleaned if r.get("stock_intelligence_score") is not None]
    if score_values:
        average_score = sum(score_values) / len(score_values)

    as_of = _now_et()

    return {
        "source": "postgres_snapshot",
        "kind": "stock_intelligence",
        "run_date": run_date.isoformat(),
        "as_of_timestamp": as_of.isoformat(),
        "universe": universe,
        "rows": cleaned,
        "rankings": cleaned,
        "top_ranked": cleaned[0] if cleaned else None,
        "summary": {
            "count": len(cleaned),
            "top_score": top_score,
            "average_score": average_score,
            "max_tickers_scanned": max_tickers,
            "min_score": min_score,
        },
    }


def _insert_run(
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
                INSERT INTO stock_intelligence_runs (
                  id, run_date, as_of_timestamp, status, is_published,
                  config, diagnostics, payload, failure_reason, published_at
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  CASE WHEN %s = TRUE THEN now() ELSE NULL END
                )
                """,
                (
                    run_id,
                    run_date,
                    as_of_timestamp,
                    status,
                    is_published,
                    Jsonb(_json_sanitize(config)),
                    Jsonb(_json_sanitize(diagnostics)),
                    Jsonb(_json_sanitize(payload)),
                    failure_reason,
                    is_published,
                ),
            )
        conn.commit()
    return run_id


def _publish_run(run_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT run_date FROM stock_intelligence_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"Cannot publish missing stock intelligence run_id={run_id}")

            run_date = row["run_date"]

            cur.execute(
                """
                UPDATE stock_intelligence_runs
                SET is_published = FALSE
                WHERE run_date = %s
                  AND id <> %s
                  AND status = 'published'
                  AND is_published = TRUE
                """,
                (run_date, run_id),
            )

            cur.execute(
                """
                UPDATE stock_intelligence_runs
                SET status = 'published',
                    is_published = TRUE,
                    published_at = now()
                WHERE id = %s
                """,
                (run_id,),
            )
        conn.commit()


def _mark_failed(run_id: str, reason: str, diagnostics: Dict[str, Any]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE stock_intelligence_runs
                SET status = 'failed',
                    is_published = FALSE,
                    failure_reason = %s,
                    diagnostics = %s
                WHERE id = %s
                """,
                (reason, Jsonb(_json_sanitize(diagnostics)), run_id),
            )
        conn.commit()


def _validate_payload(payload: Dict[str, Any], min_rows: int) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    rows = payload.get("rows") or []

    if not isinstance(rows, list):
        errors.append("Payload rows is not a list")
        return False, errors

    if len(rows) < min_rows:
        errors.append(f"Too few ranked stocks: {len(rows)} < {min_rows}")

    tickers = [str(r.get("ticker") or "") for r in rows if isinstance(r, dict)]
    if len(set(tickers)) != len(tickers):
        errors.append("Duplicate tickers in ranking payload")

    scored = [r for r in rows if isinstance(r, dict) and r.get("stock_intelligence_score") is not None]
    if len(scored) < max(1, min_rows // 2):
        errors.append(f"Too few scored rows: {len(scored)}")

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-date")
    parser.add_argument("--universe", default="quality")
    parser.add_argument("--max-tickers", type=int, default=95)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--tickers", help="Optional comma-separated tickers for a targeted rebuild")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.run_date) if args.run_date else _now_et().date()

    if not args.force and not _market_is_open(run_date):
        print(f"NYSE is not open on {run_date.isoformat()}. Skipping. Use --force to override.")
        return

    _ensure_schema()

    config = {
        "universe": args.universe,
        "max_tickers": args.max_tickers,
        "limit": args.limit,
        "min_score": args.min_score,
        "min_rows": args.min_rows,
        "tickers": args.tickers,
    }

    tickers = None
    if args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]

    try:
        print(f"[stock_intelligence] Building rankings snapshot for {run_date.isoformat()}")
        payload = _json_sanitize(
            _build_payload(
                universe=args.universe,
                max_tickers=args.max_tickers,
                limit=args.limit,
                min_score=args.min_score,
                tickers=tickers,
                run_date=run_date,
            )
        )

        valid, errors = _validate_payload(payload, args.min_rows)

        diagnostics = {
            "valid": valid,
            "errors": errors,
            "run_date": run_date.isoformat(),
            "row_count": len(payload.get("rows") or []),
            "config": config,
        }

        if args.dry_run:
            print(json.dumps({"diagnostics": diagnostics, "sample": (payload.get("rows") or [])[:5]}, indent=2))
            if not valid:
                raise SystemExit(1)
            return

        run_id = _insert_run(
            run_date=run_date,
            as_of_timestamp=_now_et(),
            status="draft",
            is_published=False,
            config=config,
            diagnostics=diagnostics,
            payload=payload,
            failure_reason=None,
        )

        if not valid:
            reason = "; ".join(errors)
            _mark_failed(run_id, reason, diagnostics)
            print(f"[stock_intelligence] FAILED guardrails: {reason}")
            raise SystemExit(1)

        _publish_run(run_id)

        print(f"[stock_intelligence] Published {run_id}. Rows={diagnostics['row_count']} Universe={args.universe}")
        print("Stock intelligence snapshot job completed successfully.")

    except Exception as exc:
        print(f"[stock_intelligence] ERROR: {exc}")
        if not args.dry_run:
            try:
                _insert_run(
                    run_date=run_date,
                    as_of_timestamp=_now_et(),
                    status="failed",
                    is_published=False,
                    config=config,
                    diagnostics={"error": str(exc), "run_date": run_date.isoformat(), "config": config},
                    payload={"error": str(exc)},
                    failure_reason=str(exc),
                )
            except Exception as insert_exc:
                print(f"[stock_intelligence] FAILED to record failure row: {insert_exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
