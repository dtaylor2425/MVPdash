from __future__ import annotations

from typing import Any, Dict, List, Optional
from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth_deps import optional_account
from api.db import get_connection
from src.portfolio_ledger import first_entries, number

router = APIRouter(prefix="/api/portfolio-snapshots", tags=["portfolio-snapshots"])
VALID_STRATEGIES = {"stock_alpha", "smid_growth", "etf_macro"}


def _truncate_for_anon(payload: Dict[str, Any]) -> Dict[str, Any]:
    """holdings + exposure, no rebalance log (matrix: /api/portfolio/*).
    This is the payload actually served to /dashboard/portfolio, so it needs
    the same treatment as the live /api/stock-portfolio and /api/portfolio
    compute endpoints — see platform-backend-accounts-build memory."""
    out = dict(payload)
    performance = dict(payload.get("performance") or {})
    if "rebalance_log" in performance:
        performance["rebalance_log"] = []
        out["performance"] = performance
    out["official_rebalance_log"] = []
    out.pop("official_rebalance", None)
    out["trade_queue"] = []
    out["truncated"] = True
    return out


def _validate_strategy(strategy: str) -> str:
    strategy = (strategy or "").strip()
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail="Unknown strategy. Use stock_alpha, smid_growth, or etf_macro.",
        )
    return strategy


def _official_rebalance_log(strategy: str, limit: int = 20, through=None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pr.id AS run_id, pr.strategy, pr.run_date, r.rebalance_date,
                       r.headline, r.buys, r.sells, r.adds, r.trims, r.turnover,
                       r.old_holdings, r.new_holdings,
                       pr.payload -> 'official_rebalance' -> 'exits' AS exits
                FROM portfolio_rebalances r
                JOIN portfolio_runs pr ON pr.id = r.run_id
                WHERE pr.strategy = %s
                  AND pr.status = 'published'
                  AND pr.is_published = TRUE
                  AND (%s::timestamptz IS NULL OR pr.published_at <= %s::timestamptz)
                ORDER BY r.rebalance_date DESC, pr.as_of_timestamp DESC
                LIMIT %s
                """,
                (strategy, through, through, limit),
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
            "exits": row.get("exits") or [],
        }
        for row in rows
    ]


def _row_to_payload(row: Dict[str, Any], official_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = deepcopy(row.get("payload") or {})
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
    # Internal quantities and full trade history are not a public API surface.
    payload.pop("ledger", None)
    return payload


def _entry_references(row, payload):
    """Read-through enrichment only; never rewrite a historical database row."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT id, run_date, published_at, payload -> 'holdings' AS holdings
                           FROM portfolio_runs WHERE strategy = %s AND status = 'published'
                           AND published_at <= %s ORDER BY run_date, published_at""",
                        (row["strategy"], row["published_at"]))
            entries = first_entries(cur.fetchall())
    for holding in payload.get("holdings") or []:
        if "entry_price_status" not in holding:
            holding.update(entries.get(holding["ticker"], {}))
        if holding.get("current_price") is None:
            holding["current_price"] = number(holding.get("close"))
        entry = number(holding.get("entry_price"), 0)
        current = number(holding.get("current_price"), 0)
        holding["unrealized_return_pct"] = (current / entry - 1) * 100 if entry > 0 and current > 0 else None
        holding["cost_basis_complete"] = entry > 0
    if not payload.get("valuation_as_of"):
        series = (payload.get("performance") or {}).get("series") or []
        payload["valuation_as_of"] = series[-1]["date"] if series else str(row["run_date"])
    if not (row.get("payload") or {}).get("ledger"):
        invested = sum(number(h.get("target_weight"), 0) for h in payload.get("holdings") or [])
        payload["stock_exposure"] = invested
        payload["cash_weight"] = max(0, 1-invested)
        payload["valuation_basis"] = "published_target_weights"
        payload["trade_queue"] = []  # Legacy suggestions were not recorded executions.
        payload["exposure_regime"] = "Published allocation; prospective accounting begins with the ledger migration"
    else:
        payload["valuation_basis"] = "marked_positions"
    return payload


@router.get("/latest")
def get_latest_portfolio_snapshot(
    strategy: str = Query(default="stock_alpha"),
    user: Optional[Dict[str, Any]] = Depends(optional_account),
) -> Dict[str, Any]:
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
    payload = _entry_references(row, _row_to_payload(row, _official_rebalance_log(strategy, through=row["published_at"])))
    if user is None:
        return _truncate_for_anon(payload)
    payload["truncated"] = False
    return payload


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
                       created_at, published_at, payload -> 'holdings' AS holdings
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
                "diagnostics": {},  # Internal diagnostics contain gated trade history.
            }
            for row in rows
        ],
    }


@router.get("/run/{run_id}")
def get_portfolio_snapshot_run(
    run_id: str,
    user: Optional[Dict[str, Any]] = Depends(optional_account),
) -> Dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, strategy, run_date, as_of_timestamp, status, is_published,
                       config, diagnostics, payload, created_at, published_at
                FROM portfolio_runs
                WHERE id = %s AND status = 'published'
                LIMIT 1
                """,
                (run_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Portfolio snapshot run not found.")
    payload = _entry_references(row, _row_to_payload(row, _official_rebalance_log(row["strategy"], through=row["published_at"])))
    if user is None:
        return _truncate_for_anon(payload)
    payload["truncated"] = False
    return payload


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
