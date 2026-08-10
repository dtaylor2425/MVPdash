from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from api.db import get_connection

router = APIRouter(prefix="/api/portfolio-snapshots", tags=["portfolio-snapshots"])
VALID_STRATEGIES = {"stock_alpha", "smid_growth", "etf_macro"}


def _validate_strategy(strategy: str) -> str:
    strategy = (strategy or "").strip()
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail="Unknown strategy. Use stock_alpha, smid_growth, or etf_macro.",
        )
    return strategy


def _official_rebalance_log(strategy: str, limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pr.id AS run_id, pr.strategy, pr.run_date, r.rebalance_date,
                       r.headline, r.buys, r.sells, r.adds, r.trims, r.turnover,
                       r.old_holdings, r.new_holdings
                FROM portfolio_rebalances r
                JOIN portfolio_runs pr ON pr.id = r.run_id
                WHERE pr.strategy = %s
                  AND pr.status = 'published'
                  AND pr.is_published = TRUE
                ORDER BY r.rebalance_date DESC, pr.as_of_timestamp DESC
                LIMIT %s
                """,
                (strategy, limit),
            )
            rows = cur.fetchall()

    return [
        {
            "run_id": str(row["run_id"]),
            "strategy": row["strategy"],
            "run_date": row["run_date"].isoformat() if row.get("run_date") else None,
            "rebalance_date": row["rebalance_date"].isoformat() if row.get("rebalance_date") else None,
            "headline": row["headline"],
            "buys": row["buys"] or [],
            "sells": row["sells"] or [],
            "adds": row["adds"] or [],
            "trims": row["trims"] or [],
            "turnover": float(row["turnover"]) if row.get("turnover") is not None else None,
            "old_holdings": row["old_holdings"] or [],
            "new_holdings": row["new_holdings"] or [],
        }
        for row in rows
    ]


def _row_to_payload(row: Dict[str, Any], official_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = row.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"payload": payload}

    payload["source"] = "postgres_snapshot"
    payload["snapshot"] = {
        "run_id": str(row["id"]),
        "strategy": row["strategy"],
        "run_date": row["run_date"].isoformat() if row.get("run_date") else None,
        "as_of_timestamp": row["as_of_timestamp"].isoformat() if row.get("as_of_timestamp") else None,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "published_at": row["published_at"].isoformat() if row.get("published_at") else None,
        "status": row["status"],
        "is_published": row["is_published"],
    }
    payload["official_rebalance_log"] = official_log
    return payload


@router.get("/latest")
def get_latest_portfolio_snapshot(strategy: str = Query(default="stock_alpha")) -> Dict[str, Any]:
    strategy = _validate_strategy(strategy)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, strategy, run_date, as_of_timestamp, status, is_published,
                       config, diagnostics, payload, created_at, published_at
                FROM portfolio_runs
                WHERE strategy = %s
                  AND status = 'published'
                  AND is_published = TRUE
                ORDER BY run_date DESC, as_of_timestamp DESC, created_at DESC
                LIMIT 1
                """,
                (strategy,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No published snapshot for {strategy}. Run jobs/nightly_portfolio_refresh.py first.",
        )
    return _row_to_payload(row, _official_rebalance_log(strategy))


@router.get("/history")
def get_portfolio_snapshot_history(
    strategy: str = Query(default="stock_alpha"),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    strategy = _validate_strategy(strategy)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, strategy, run_date, as_of_timestamp, status, is_published,
                       created_at, published_at, diagnostics, payload -> 'holdings' AS holdings
                FROM portfolio_runs
                WHERE strategy = %s
                  AND status = 'published'
                  AND is_published = TRUE
                ORDER BY run_date DESC, as_of_timestamp DESC, created_at DESC
                LIMIT %s
                """,
                (strategy, limit),
            )
            rows = cur.fetchall()

    return {
        "strategy": strategy,
        "count": len(rows),
        "runs": [
            {
                "run_id": str(row["id"]),
                "run_date": row["run_date"].isoformat() if row.get("run_date") else None,
                "as_of_timestamp": row["as_of_timestamp"].isoformat() if row.get("as_of_timestamp") else None,
                "published_at": row["published_at"].isoformat() if row.get("published_at") else None,
                "holding_count": len(row.get("holdings") or []),
                "diagnostics": row.get("diagnostics") or {},
            }
            for row in rows
        ],
    }


@router.get("/run/{run_id}")
def get_portfolio_snapshot_run(run_id: str) -> Dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, strategy, run_date, as_of_timestamp, status, is_published,
                       config, diagnostics, payload, created_at, published_at
                FROM portfolio_runs
                WHERE id = %s
                LIMIT 1
                """,
                (run_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Portfolio snapshot run not found.")
    return _row_to_payload(row, _official_rebalance_log(row["strategy"]))


@router.get("/status")
def get_portfolio_snapshots_status() -> Dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT strategy, status, COUNT(*) AS count,
                       MAX(run_date) AS latest_run_date,
                       MAX(published_at) AS latest_published_at
                FROM portfolio_runs
                GROUP BY strategy, status
                ORDER BY strategy, status
                """
            )
            rows = cur.fetchall()

    return {
        "status": "ok",
        "route": "/api/portfolio-snapshots",
        "strategies": sorted(VALID_STRATEGIES),
        "runs": [
            {
                "strategy": row["strategy"],
                "status": row["status"],
                "count": int(row["count"]),
                "latest_run_date": row["latest_run_date"].isoformat() if row.get("latest_run_date") else None,
                "latest_published_at": row["latest_published_at"].isoformat() if row.get("latest_published_at") else None,
            }
            for row in rows
        ],
    }
