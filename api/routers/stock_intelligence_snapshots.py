from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from api.db import get_connection

router = APIRouter(prefix="/api/stock-intelligence-snapshots", tags=["stock-intelligence-snapshots"])


def _row_to_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    snapshot = {
        "run_id": str(row.get("id")),
        "run_date": row.get("run_date").isoformat() if row.get("run_date") else None,
        "as_of_timestamp": row.get("as_of_timestamp").isoformat() if row.get("as_of_timestamp") else None,
        "published_at": row.get("published_at").isoformat() if row.get("published_at") else None,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "status": row.get("status"),
        "is_published": row.get("is_published"),
    }

    payload["source"] = "postgres_snapshot"
    payload["snapshot"] = snapshot
    payload["run_id"] = snapshot["run_id"]
    payload["run_date"] = payload.get("run_date") or snapshot["run_date"]
    payload["as_of_timestamp"] = payload.get("as_of_timestamp") or snapshot["as_of_timestamp"]

    return payload


@router.get("/latest")
def latest_stock_intelligence_snapshot() -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, run_date, as_of_timestamp, status, is_published,
                           payload, diagnostics, failure_reason, created_at, published_at
                    FROM stock_intelligence_runs
                    WHERE status = 'published'
                      AND is_published = TRUE
                    ORDER BY run_date DESC, as_of_timestamp DESC, created_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stock intelligence snapshot lookup failed: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="No published stock intelligence snapshot found.")

    return _row_to_payload(row)


@router.get("/history")
def stock_intelligence_snapshot_history(
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, run_date, as_of_timestamp, status, is_published,
                           diagnostics, failure_reason, created_at, published_at
                    FROM stock_intelligence_runs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stock intelligence history lookup failed: {exc}")

    items: List[Dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "run_id": str(row.get("id")),
                "run_date": row.get("run_date").isoformat() if row.get("run_date") else None,
                "as_of_timestamp": row.get("as_of_timestamp").isoformat() if row.get("as_of_timestamp") else None,
                "published_at": row.get("published_at").isoformat() if row.get("published_at") else None,
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "status": row.get("status"),
                "is_published": row.get("is_published"),
                "failure_reason": row.get("failure_reason"),
                "diagnostics": row.get("diagnostics") or {},
            }
        )

    return {"items": items}


@router.get("/status")
def stock_intelligence_snapshot_status() -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COUNT(*) AS total_runs,
                      COUNT(*) FILTER (WHERE status = 'published') AS published_runs,
                      COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs,
                      MAX(created_at) AS last_created_at,
                      MAX(published_at) AS last_published_at
                    FROM stock_intelligence_runs
                    """
                )
                stats = cur.fetchone() or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stock intelligence status lookup failed: {exc}")

    return {
        "ok": True,
        "total_runs": stats.get("total_runs") or 0,
        "published_runs": stats.get("published_runs") or 0,
        "failed_runs": stats.get("failed_runs") or 0,
        "last_created_at": stats.get("last_created_at").isoformat() if stats.get("last_created_at") else None,
        "last_published_at": stats.get("last_published_at").isoformat() if stats.get("last_published_at") else None,
    }
