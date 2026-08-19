from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from api.services.volatility_engine import (
    build_natr_ladder,
    build_volatility_snapshot,
    fetch_spx_history,
)

router = APIRouter(prefix="/api/volatility", tags=["volatility"])


def _database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def _decode_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _latest_postgres_snapshot() -> Optional[Dict[str, Any]]:
    url = _database_url()
    if not url:
        return None

    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception:
        return None

    query = """
        SELECT ts, spx, source_symbol, is_spy_fallback, regime, payload
        FROM volatility_snapshots
        ORDER BY ts DESC
        LIMIT 1;
    """

    try:
        with psycopg.connect(url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()
    except Exception:
        return None

    if not row:
        return None

    payload = _decode_payload(row["payload"])
    payload["source"] = "postgres_snapshot"
    payload["snapshot"] = {
        "ts": row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else str(row["ts"]),
        "spx": row.get("spx"),
        "source_symbol": row.get("source_symbol"),
        "is_spy_fallback": row.get("is_spy_fallback"),
        "regime": row.get("regime"),
    }
    return payload


@router.get("/snapshot")
def volatility_snapshot(live: bool = Query(False)):
    """
    Default behavior is fast: read the latest published Postgres snapshot.
    Add ?live=true only for admin/manual rebuilds from Yahoo.
    """
    if not live:
        cached = _latest_postgres_snapshot()
        if cached is not None:
            return cached

    try:
        payload = build_volatility_snapshot()
        payload["source"] = "live_yahoo"
        return payload
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/history")
def volatility_history(limit: int = Query(100, ge=1, le=500)):
    url = _database_url()
    if not url:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:
        raise HTTPException(status_code=503, detail="psycopg is not installed: " + str(exc))

    query = """
        SELECT
            ts,
            spx,
            source_symbol,
            is_spy_fallback,
            iv_7d,
            iv_14d,
            iv_30d,
            iv_60d,
            spread_7d_14d,
            spread_14d_30d,
            spread_30d_60d,
            regime
        FROM volatility_snapshots
        ORDER BY ts DESC
        LIMIT %(limit)s;
    """

    try:
        with psycopg.connect(url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"limit": limit})
                rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "source": "postgres_snapshot",
        "rows": [
            {
                **dict(row),
                "ts": row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else str(row["ts"]),
            }
            for row in rows
        ],
    }


@router.get("/natr")
def natr_only(map_mode: str = Query("Both", pattern="^(Both|ATR length|Norm length)$")):
    try:
        history = fetch_spx_history()
        return {
            "underlying": "^GSPC",
            "spx": round(float(history["Close"].iloc[-1]), 2),
            "map_mode": map_mode,
            "natr": build_natr_ladder(history, map_mode=map_mode),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
