"""Bounded, identity-free usage counts; never a subscription/payment ledger."""
from datetime import datetime, timedelta, timezone
from threading import Lock
from time import monotonic
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict

from api.auth_deps import require_admin
from api.db import get_connection

router = APIRouter(prefix="/api/research-usage", tags=["research-usage"])
ORIGINS = {"https://www.macro-engine.com", "https://macro-engine.com"}
_lock = Lock()
_window = 0.0
_received = 0


class UsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: Literal["view", "details_open", "newsletter_click", "login_start"]
    page: Literal["home", "report", "portfolio", "smid", "workbook", "macro"]
    source: Literal["direct", "substack", "email", "search", "social", "internal", "other"]


def _admit():
    """A process-wide bound without storing client addresses or identifiers."""
    global _window, _received
    with _lock:
        now = monotonic()
        if now - _window >= 60:
            _window, _received = now, 0
        if _received >= 240:
            raise HTTPException(429, "Usage collection busy")
        _received += 1


@router.post("/event", status_code=204)
def record_event(event: UsageEvent, request: Request):
    if request.headers.get("origin") not in ORIGINS:
        raise HTTPException(403, "Unsupported origin")
    if request.headers.get("sec-gpc") == "1" or request.headers.get("dnt") == "1":
        return Response(status_code=204)
    _admit()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '2s'")
                cur.execute("""INSERT INTO research_usage_daily (day,event,page,source,count)
                    VALUES ((CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date,%s,%s,%s,1)
                    ON CONFLICT (day,event,page,source)
                    DO UPDATE SET count = research_usage_daily.count + 1""",
                    (event.event, event.page, event.source))
            conn.commit()
    except Exception:
        # Report collection failure without leaking connection details.
        raise HTTPException(503, "Usage collection unavailable") from None
    return Response(status_code=204)


@router.get("/summary")
def usage_summary(response: Response, days: int = Query(30, ge=1, le=90), user=Depends(require_admin)):
    response.headers["Cache-Control"] = "private, no-store"
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=days - 1)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT day,event,page,source,count FROM research_usage_daily
                WHERE day BETWEEN %s AND %s ORDER BY day DESC,event,page,source""", (since, today))
            rows = cur.fetchall()
            cur.execute("""SELECT (created_at AT TIME ZONE 'UTC')::date AS day, count(*) AS count
                FROM users WHERE created_at >= %s::date AT TIME ZONE 'UTC'
                AND created_at < (%s::date + 1) AT TIME ZONE 'UTC'
                GROUP BY 1 ORDER BY 1 DESC""", (since, today))
            accounts = cur.fetchall()
    return {"since": since, "through": today, "timezone": "UTC", "interactions": rows,
            "website_accounts_created": accounts,
            "confirmed_substack_signups": None, "confirmed_paid_conversions": None,
            "unique_readers": None, "returning_readers": None,
            "limitations": ["Browser-reported interactions, not unique people; bots, blockers and outages affect counts.",
                "A click to Substack is not a confirmed subscription or payment.",
                "Account totals count website database rows, without source attribution.",
                "No visitor identifiers are stored, so cross-week reader retention is not measured."]}
