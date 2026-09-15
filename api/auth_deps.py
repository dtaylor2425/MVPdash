"""
api/auth_deps.py  (docs/PLATFORM-BACKEND-PLAN-V2.md section 3)

Three dependencies used on routes:

    optional_account()   -> dict | None   -- never raises, drives truncation
    require_account()    -> dict          -- 401 when anonymous
    require_admin()      -> dict          -- 403 unless is_admin

`optional_account` degrades to "anonymous" (returns None) on any DB error
rather than raising, so routes stay reachable if Postgres is briefly
unavailable or (as in local dev) DATABASE_URL isn't set at all.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request

from api.db import get_connection
from api.services.security import utcnow

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "me_session")
SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "30"))


def _load_session_user(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.id AS session_id, s.expires_at, s.revoked_at,
                           u.id AS user_id, u.email, u.email_raw,
                           u.created_at, u.is_admin
                    FROM sessions s
                    JOIN users u ON u.id = s.user_id
                    WHERE s.id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                if row["revoked_at"] is not None:
                    return None
                if row["expires_at"] <= utcnow():
                    return None

                # Sliding expiry: every authenticated request pushes the
                # session's expiry out another SESSION_TTL_DAYS.
                new_expiry = utcnow() + timedelta(days=SESSION_TTL_DAYS)
                cur.execute(
                    "UPDATE sessions SET last_seen_at = now(), expires_at = %s WHERE id = %s",
                    (new_expiry, session_id),
                )
            conn.commit()
    except Exception as e:
        print("[auth] session lookup failed: {}".format(e))
        return None
    return row


def optional_account(request: Request) -> Optional[Dict[str, Any]]:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        return None
    return _load_session_user(session_id)


def require_account(user: Optional[Dict[str, Any]] = Depends(optional_account)) -> Dict[str, Any]:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in required")
    return user


def require_admin(user: Dict[str, Any] = Depends(require_account)) -> Dict[str, Any]:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
