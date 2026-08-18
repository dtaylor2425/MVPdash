from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.services.volatility_engine import (
    build_natr_ladder,
    build_volatility_snapshot,
    fetch_spx_history,
)

router = APIRouter(prefix="/api/volatility", tags=["volatility"])


def _get_connection():
    try:
        from api.db import get_connection
    except Exception as exc:
        raise RuntimeError(
            "api.db.get_connection is required for Postgres snapshot reads. "
            "Install the portfolio Postgres helper first or use live=true."
        ) from exc

    return get_connection()


def _latest_snapshot_from_db() -> Optional[Dict[str, Any]]:
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  ts,
                  spx,
                  source_symbol,
                  is_spy_fallback,
                  regime,
                  payload
                FROM volatility_snapshots
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()

    if not row:
        return None

    payload = row.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"payload": payload}

    payload["source"] = "postgres_snapshot"
    payload["snapshot"] = {
        "ts": row["ts"].isoformat() if row.get("ts") else None,
        "spx": row.get("spx"),
        "source_symbol": row.get("source_symbol"),
        "is_spy_fallback": row.get("is_spy_fallback"),
        "regime": row.get("regime"),
    }
    return payload


@router.get("/snapshot")
def volatility_snapshot(
    live: bool = Query(
        default=False,
        description="When false, returns latest Postgres snapshot. When true, rebuilds live from Yahoo.",
    )
) -> Dict[str, Any]:
    if not live:
        try:
            cached = _latest_snapshot_from_db()
            if cached is not None:
                return cached
        except Exception:
            # Fall through to live calculation if DB is not ready yet.
            pass

    try:
        payload = build_volatility_snapshot()
        payload["source"] = "live_yahoo"
        return payload
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/snapshot/live")
def volatility_snapshot_live() -> Dict[str, Any]:
    try:
        payload = build_volatility_snapshot()
        payload["source"] = "live_yahoo"
        return payload
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/history")
def volatility_history(
    limit: int = Query(default=50, ge=1, le=500)
) -> Dict[str, Any]:
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
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
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "source": "postgres_snapshot",
        "count": len(rows),
        "items": [
            {
                "ts": row["ts"].isoformat() if row.get("ts") else None,
                "spx": row.get("spx"),
                "source_symbol": row.get("source_symbol"),
                "is_spy_fallback": row.get("is_spy_fallback"),
                "iv_7d": row.get("iv_7d"),
                "iv_14d": row.get("iv_14d"),
                "iv_30d": row.get("iv_30d"),
                "iv_60d": row.get("iv_60d"),
                "spread_7d_14d": row.get("spread_7d_14d"),
                "spread_14d_30d": row.get("spread_14d_30d"),
                "spread_30d_60d": row.get("spread_30d_60d"),
                "regime": row.get("regime"),
            }
            for row in rows
        ],
    }


@router.get("/natr")
def natr_only(
    map_mode: str = Query("Both", pattern="^(Both|ATR length|Norm length)$")
) -> Dict[str, Any]:
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
