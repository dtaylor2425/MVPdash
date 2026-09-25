"""
api/routers/private_options_flow.py

PRIVATE read-only API over the derived snapshots that jobs/options_flow_refresh.py
publishes. Never talks to ThetaData, never computes on request, never returns
credentials.

    GET /api/private/options-flow/latest
    GET /api/private/options-flow/history?ticker=SPY&days=20
    GET /api/private/options-flow/ticker/SPY[?include_trades=false]
    GET /api/private/options-flow/status

Every route requires header  X-Internal-Options-Token == $INTERNAL_OPTIONS_API_SECRET.
The only intended caller is the Next.js server route (which also enforces the
PRIVATE_OPTIONS_EMAILS allow-list on the logged-in Macro Engine user); the
browser never sees the secret.

Failure modes are deliberately opaque:
    wrong / missing token   -> 404 (the route looks like it doesn't exist)
    secret not configured   -> 503 (fail closed)
Routes are excluded from the OpenAPI schema so /docs doesn't advertise them.
"""

from __future__ import annotations

import os
import secrets
from datetime import date
from uuid import UUID
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from api.db import get_connection
from api.services import options_flow_store as store
from api.services import options_flow_publication as publication
from api.services import options_flow_issues as issues
from api.services.options_flow_config import load_config, load_universe

SECRET_ENV = "INTERNAL_OPTIONS_API_SECRET"
TOKEN_HEADER = "X-Internal-Options-Token"


def require_internal_token(
    response: Response,
    x_internal_options_token: Optional[str] = Header(default=None, alias=TOKEN_HEADER),
) -> None:
    # Applies to successful responses; the 404/503 paths carry no body worth caching.
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"

    expected = os.getenv(SECRET_ENV, "")
    if not expected:
        print("[options-flow] {} is not set; refusing all requests".format(SECRET_ENV))
        raise HTTPException(status_code=503, detail="Not configured")
    supplied = x_internal_options_token or ""
    # compare_digest on bytes: constant-time and safe for non-ASCII input
    if not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(
    prefix="/api/private/options-flow",
    tags=["private-options-flow"],
    dependencies=[Depends(require_internal_token)],
    include_in_schema=False,
)


def _valid_ticker(raw: str, universe: Dict[str, Any]) -> str:
    t = (raw or "").strip().upper()
    if not store.TICKER_RE.match(t) or t not in {x for ts in universe.values() for x in ts}:
        raise HTTPException(status_code=404, detail="Unknown ticker")
    return t


def _db_unavailable(e: Exception) -> HTTPException:
    # Log the cause server-side; return nothing that could leak connection details.
    print("[options-flow] database error: {}".format(e))
    return HTTPException(status_code=503, detail="Options flow data unavailable")


def _is_missing_table(e: Exception) -> bool:
    return e.__class__.__name__ == "UndefinedTable"


@router.get("/latest")
def latest(session: Optional[date] = Query(default=None)) -> Dict[str, Any]:
    universe, cfg = load_universe(), load_config()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            out = publication.latest(conn, universe, cfg, session=session)
    except Exception as e:
        if _is_missing_table(e):
            raise HTTPException(status_code=404, detail="No options flow published yet")
        raise _db_unavailable(e)
    if out["run"] is None:
        raise HTTPException(status_code=404, detail="No options flow published yet")
    return out


@router.get("/history")
def history(
    ticker: str = Query(...),
    days: int = Query(default=20, ge=1, le=260),
    session: Optional[date] = Query(default=None),
) -> Dict[str, Any]:
    universe = load_universe()
    t = _valid_ticker(ticker, universe)
    try:
        with get_connection() as conn:
            out = store.fetch_history(conn, t, days, session=session)
    except Exception as e:
        if _is_missing_table(e):
            raise HTTPException(status_code=404, detail="No options flow published yet")
        raise _db_unavailable(e)
    if not out["series"]:
        raise HTTPException(status_code=404, detail="No history for ticker")
    return out


@router.get("/ticker/{ticker}")
def ticker_detail(ticker: str, include_trades: bool = Query(default=True), session: Optional[date] = Query(default=None)) -> Dict[str, Any]:
    universe, cfg = load_universe(), load_config()
    t = _valid_ticker(ticker, universe)
    try:
        with get_connection() as conn:
            out = publication.detail(conn, t, include_trades=include_trades, session=session)
    except Exception as e:
        if _is_missing_table(e):
            raise HTTPException(status_code=404, detail="No options flow published yet")
        raise _db_unavailable(e)
    if out is None:
        raise HTTPException(status_code=404, detail="No snapshot for ticker")
    return out


@router.get("/status")
def status() -> Dict[str, Any]:
    universe, cfg = load_universe(), load_config()
    try:
        with get_connection() as conn:
            return store.fetch_status(conn, universe, cfg)
    except Exception as e:
        if _is_missing_table(e):
            raise HTTPException(status_code=404, detail="No options flow published yet")
        raise _db_unavailable(e)


@router.get("/sessions")
def available_sessions() -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            return {"sessions": publication.sessions(conn)}
    except Exception as e:
        raise _db_unavailable(e)


class IssueRequest(BaseModel):
    session: date
    title: str = Field(min_length=1, max_length=300)
    body: str = Field(max_length=20000)
    chartSpec: Dict[str, Any] = Field(default_factory=dict)
    snapshotIds: list[UUID] = Field(min_length=1, max_length=100)
    expectedEvidenceSha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correctionOf: Optional[UUID] = None
    correctionReason: Optional[str] = Field(default=None, max_length=2000)


@router.post("/issues")
def freeze_issue(body: IssueRequest):
    try:
        with get_connection() as conn:
            # One consistent database view for all summary/history evidence.
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            snapshot = publication.latest(conn, load_universe(), load_config(), body.session)
            return issues.create(conn, body, snapshot)
    except issues.SnapshotConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _db_unavailable(e)


@router.get("/issues")
def issue_list(session: Optional[date] = Query(default=None)):
    try:
        with get_connection() as conn:
            return {"issues": issues.list_issues(conn, session)}
    except Exception as e:
        raise _db_unavailable(e)


@router.get("/issues/{identity}")
def issue_detail(identity: UUID):
    try:
        with get_connection() as conn:
            row = issues.get_issue(conn, identity)
    except Exception as e:
        raise _db_unavailable(e)
    if row is None:
        raise HTTPException(status_code=404, detail="Issue revision not found")
    return row
