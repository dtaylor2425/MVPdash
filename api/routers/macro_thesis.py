"""
api/routers/macro_thesis.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 9)

Read-only routes serving the snapshot jobs/macro_thesis_job.py publishes.
Never recomputes on request -- mirrors api/routers/fx.py's DB-first,
disk-fallback pattern.

    GET  /api/macro/thesis
    GET  /api/macro/quadrant/history?months=300
    GET  /api/macro/assets-by-quadrant?quadrant=STAGFLATION

    POST /api/admin/house-view   (admin only; separate prefix, same file
                                  since it's tightly coupled to this engine)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from api.auth_deps import require_admin
from api.db import get_connection

router = APIRouter(prefix="/api/macro", tags=["macro-thesis"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

_DISK_SNAPSHOT = Path("data/cache/macro_thesis_snapshot_latest.json")
QUADRANTS = ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]


def _load_from_db() -> Optional[Dict[str, Any]]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT as_of, payload FROM macro_thesis_snapshots ORDER BY as_of DESC LIMIT 1"
                )
                row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _load_from_disk() -> Optional[Dict[str, Any]]:
    if not _DISK_SNAPSHOT.exists():
        return None
    try:
        return json.loads(_DISK_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_snapshot() -> Dict[str, Any]:
    payload = _load_from_db() or _load_from_disk()
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="No macro thesis snapshot published yet. Run jobs/macro_thesis_job.py first.",
        )
    return payload


@router.get("/thesis")
def macro_thesis() -> Dict[str, Any]:
    return _latest_snapshot()


@router.get("/quadrant/history")
def quadrant_history(months: int = Query(default=300, ge=1, le=1200)) -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT as_of, growth_level, growth_momentum, inflation_level,
                           inflation_momentum, quadrant, strength, phase
                    FROM macro_quadrant_history
                    ORDER BY as_of DESC
                    LIMIT %s
                    """,
                    (months,),
                )
                rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Quadrant history unavailable: {}".format(e))

    if not rows:
        raise HTTPException(status_code=404, detail="No quadrant history published yet.")

    rows = list(reversed(rows))  # chronological order for charting
    return {
        "months": len(rows),
        "series": [
            {
                "date": r["as_of"].isoformat(),
                "growthLevel": r["growth_level"],
                "growthMomentum": r["growth_momentum"],
                "inflationLevel": r["inflation_level"],
                "inflationMomentum": r["inflation_momentum"],
                "quadrant": r["quadrant"],
                "strength": r["strength"],
                "phase": r["phase"],
            }
            for r in rows
        ],
    }


@router.get("/assets-by-quadrant")
def assets_by_quadrant(quadrant: str = Query(...)) -> Dict[str, Any]:
    quadrant = quadrant.upper().strip()
    if quadrant not in QUADRANTS:
        raise HTTPException(status_code=404, detail="Unknown quadrant '{}'. Known: {}".format(quadrant, QUADRANTS))

    snapshot = _latest_snapshot()
    all_by_quadrant = snapshot.get("assetsByQuadrantAll", {})
    rows: List[Dict[str, Any]] = all_by_quadrant.get(quadrant, [])
    return {
        "asOf": snapshot.get("asOf"),
        "quadrant": quadrant,
        "assets": rows,
        "suppressedNote": "Assets with fewer than 24 historical episodes in this quadrant are omitted.",
    }


# ---------------------------------------------------------------------------
# Admin: house view override (spec section 8)
# ---------------------------------------------------------------------------
@admin_router.post("/house-view")
def upsert_house_view(
    body: Dict[str, Any] = Body(...),
    admin: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    agrees = body.get("agreesWithModel")
    if not isinstance(agrees, bool):
        raise HTTPException(status_code=400, detail="agreesWithModel (bool) is required")

    headline = body.get("headline")
    view_body = body.get("body")
    conviction = body.get("conviction")
    author_note = body.get("authorNote")
    if conviction is not None and conviction not in ("high", "medium", "low"):
        raise HTTPException(status_code=400, detail="conviction must be 'high', 'medium', or 'low'")

    as_of = date.today()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO house_view (as_of, agrees_with_model, headline, body, conviction, author_note, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (as_of) DO UPDATE SET
                        agrees_with_model = EXCLUDED.agrees_with_model,
                        headline = EXCLUDED.headline,
                        body = EXCLUDED.body,
                        conviction = EXCLUDED.conviction,
                        author_note = EXCLUDED.author_note,
                        created_by = EXCLUDED.created_by,
                        created_at = now()
                    RETURNING *
                    """,
                    (as_of, agrees, headline, view_body, conviction, author_note, admin.get("email")),
                )
                row = cur.fetchone()
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save house view: {}".format(e))

    row["as_of"] = row["as_of"].isoformat()
    row["created_at"] = row["created_at"].isoformat() if row.get("created_at") else None
    return row
