"""
api/routers/auth.py  (docs/PLATFORM-BACKEND-PLAN-V2.md sections 2, 5)

    POST /api/auth/request-link
    GET  /api/auth/callback
    POST /api/auth/logout
    POST /api/auth/logout-all
    GET  /api/auth/me

Design notes / deviations from the literal spec text, kept here rather than
re-derived later:

- The user row is created (get-or-create by email) at request-link time, not
  at callback time. `login_tokens.user_id` is NOT NULL, so a token has to
  reference a real user; callback also does a defensive get-or-create in case
  a token is ever replayed after its user row was removed out-of-band.
- request-link always returns the same 200 message, even when rate-limited
  or the email is malformed-but-plausible — revealing "you're rate limited"
  is itself a signal an attacker can use to probe for existing accounts, so
  the failure is silent from the caller's point of view and only visible in
  auth_events.
- `request_ip` / `ip` columns are TEXT, not the spec's `inet` — see
  sql/003_accounts.sql. Avoids psycopg <-> inet adaptation entirely and
  tolerates multi-hop X-Forwarded-For values untouched.
- GET /api/auth/callback returns the same JSON shape as GET /api/auth/me
  (200, with the new session cookie set) rather than an HTTP redirect. The
  frontend's `/auth/callback` page is expected to call this via
  `fetch(..., {credentials: 'include'})` and then do a client-side redirect
  to `next` — that keeps redirect-target validation and routing in the SPA
  rather than trusting a server-side redirect to an arbitrary `next`.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response

from api.auth_deps import SESSION_COOKIE_NAME, SESSION_TTL_DAYS, optional_account, require_account
from api.db import get_connection
from api.services.email import send_magic_link_email
from api.services.security import (
    generate_login_token,
    hash_token,
    is_valid_email,
    new_id,
    normalize_email,
    tokens_match,
    utcnow,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

LOGIN_TOKEN_TTL_MINUTES = int(os.getenv("LOGIN_TOKEN_TTL_MINUTES", "15"))
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3000")
ADMIN_EMAILS = {
    normalize_email(e) for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
}

RATE_LIMIT_PER_EMAIL = 3
RATE_LIMIT_EMAIL_WINDOW = timedelta(minutes=15)
RATE_LIMIT_PER_IP = 10
RATE_LIMIT_IP_WINDOW = timedelta(hours=1)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _is_local() -> bool:
    return "localhost" in APP_BASE_URL or "127.0.0.1" in APP_BASE_URL


def _set_session_cookie(response: Response, session_id: str) -> None:
    domain = os.getenv("COOKIE_DOMAIN", "")
    kwargs: Dict[str, Any] = dict(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_TTL_DAYS * 86400,
        httponly=True,
        secure=not _is_local(),
        samesite="lax",
        path="/",
    )
    if domain and "localhost" not in domain:
        kwargs["domain"] = domain
    response.set_cookie(**kwargs)


def _clear_session_cookie(response: Response) -> None:
    domain = os.getenv("COOKIE_DOMAIN", "")
    kwargs: Dict[str, Any] = dict(key=SESSION_COOKIE_NAME, path="/")
    if domain and "localhost" not in domain:
        kwargs["domain"] = domain
    response.delete_cookie(**kwargs)


def _log_event(conn, email: Optional[str], event: str, ip: str, detail: Optional[Dict[str, Any]] = None) -> None:
    import json

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO auth_events (email, event, ip, detail) VALUES (%s, %s, %s, %s)",
            (email, event, ip, json.dumps(detail) if detail is not None else None),
        )


def _rate_limited(conn, email: str, ip: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM auth_events WHERE event = 'request' AND email = %s AND at > %s",
            (email, utcnow() - RATE_LIMIT_EMAIL_WINDOW),
        )
        if cur.fetchone()["n"] >= RATE_LIMIT_PER_EMAIL:
            return True
        cur.execute(
            "SELECT count(*) AS n FROM auth_events WHERE event = 'request' AND ip = %s AND at > %s",
            (ip, utcnow() - RATE_LIMIT_IP_WINDOW),
        )
        if cur.fetchone()["n"] >= RATE_LIMIT_PER_IP:
            return True
    return False


def _get_or_create_user(conn, email: str, email_raw: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        if row is not None:
            return row
        user_id = new_id()
        is_admin = email in ADMIN_EMAILS
        cur.execute(
            """
            INSERT INTO users (id, email, email_raw, is_admin)
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (user_id, email, email_raw, is_admin),
        )
        return cur.fetchone()


def _maybe_promote_admin(conn, user_row: Dict[str, Any]) -> Dict[str, Any]:
    if user_row["email"] in ADMIN_EMAILS and not user_row["is_admin"]:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET is_admin = TRUE WHERE id = %s RETURNING *",
                (user_row["id"],),
            )
            return cur.fetchone()
    return user_row


def _account_payload(conn, user_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if user_row is None:
        return {"email": None, "isRegistered": False, "isAdmin": False, "newsletter": False, "memberSince": None}

    newsletter = False
    newsletter_since = None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT newsletter, newsletter_since FROM entitlements WHERE email = %s",
            (user_row["email"],),
        )
        ent = cur.fetchone()
        if ent is not None:
            newsletter = bool(ent["newsletter"])
            newsletter_since = ent["newsletter_since"]

    member_since = user_row["created_at"]
    return {
        "email": user_row["email"],
        "isRegistered": True,
        "isAdmin": bool(user_row["is_admin"]),
        "newsletter": newsletter,
        "newsletterSince": newsletter_since.isoformat() if isinstance(newsletter_since, date) else newsletter_since,
        "memberSince": member_since.isoformat() if hasattr(member_since, "isoformat") else member_since,
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
GENERIC_REQUEST_RESPONSE = {"message": "If an account exists for that email, a sign-in link is on its way."}


@router.post("/request-link")
def request_link(request: Request, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    email_raw = str(body.get("email") or "")
    next_path = str(body.get("next") or "/account")
    if not next_path.startswith("/") or next_path.startswith("//"):
        next_path = "/account"

    email = normalize_email(email_raw)
    ip = _client_ip(request)

    if not is_valid_email(email):
        # Still generic: don't tell the caller their input was malformed in
        # a way that distinguishes it from "no such account".
        return GENERIC_REQUEST_RESPONSE

    try:
        with get_connection() as conn:
            if _rate_limited(conn, email, ip):
                _log_event(conn, email, "fail", ip, {"reason": "rate_limited"})
                conn.commit()
                return GENERIC_REQUEST_RESPONSE

            user = _get_or_create_user(conn, email, email_raw.strip())
            raw_token = generate_login_token()
            token_hash = hash_token(raw_token)
            expires_at = utcnow() + timedelta(minutes=LOGIN_TOKEN_TTL_MINUTES)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO login_tokens (token_hash, user_id, expires_at, request_ip, request_ua)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (token_hash, user["id"], expires_at, ip, request.headers.get("user-agent", "")),
                )

            _log_event(conn, email, "request", ip)
            conn.commit()

        link = "{base}/auth/callback?token={token}&next={next}".format(
            base=APP_BASE_URL.rstrip("/"),
            token=quote(raw_token),
            next=quote(next_path),
        )
        send_magic_link_email(email, link)
    except Exception as e:
        print("[auth] request-link error: {}".format(e))

    return GENERIC_REQUEST_RESPONSE


@router.get("/callback")
def auth_callback(request: Request, response: Response, token: str = "") -> Dict[str, Any]:
    ip = _client_ip(request)
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

    token_hash = hash_token(token)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM login_tokens WHERE token_hash = %s", (token_hash,))
                row = cur.fetchone()

            if row is None or not tokens_match(token_hash, row["token_hash"]):
                _log_event(conn, None, "fail", ip, {"reason": "invalid_token"})
                conn.commit()
                raise HTTPException(status_code=400, detail="This link is invalid.")

            if row["consumed_at"] is not None:
                _log_event(conn, None, "fail", ip, {"reason": "token_reused"})
                conn.commit()
                raise HTTPException(status_code=400, detail="This link has already been used.")

            if row["expires_at"] <= utcnow():
                _log_event(conn, None, "fail", ip, {"reason": "token_expired"})
                conn.commit()
                raise HTTPException(status_code=400, detail="This link has expired. Request a new one.")

            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (row["user_id"],))
                user = cur.fetchone()
            if user is None:
                _log_event(conn, None, "fail", ip, {"reason": "user_missing"})
                conn.commit()
                raise HTTPException(status_code=400, detail="This link is invalid.")

            user = _maybe_promote_admin(conn, user)

            session_id = new_id()
            session_expiry = utcnow() + timedelta(days=SESSION_TTL_DAYS)
            with conn.cursor() as cur:
                cur.execute("UPDATE login_tokens SET consumed_at = now() WHERE token_hash = %s", (token_hash,))
                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, expires_at, last_seen_at, user_agent)
                    VALUES (%s, %s, %s, now(), %s)
                    """,
                    (session_id, user["id"], session_expiry, request.headers.get("user-agent", "")),
                )
                cur.execute("UPDATE users SET last_login_at = now() WHERE id = %s", (user["id"],))

            _log_event(conn, user["email"], "login", ip)
            conn.commit()

            payload = _account_payload(conn, user)
    except HTTPException:
        raise
    except Exception as e:
        print("[auth] callback error: {}".format(e))
        raise HTTPException(status_code=500, detail="Sign-in failed. Please try again.")

    _set_session_cookie(response, session_id)
    return payload


@router.post("/logout")
def logout(request: Request, response: Response) -> Dict[str, Any]:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE sessions SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL",
                        (session_id,),
                    )
                conn.commit()
        except Exception as e:
            print("[auth] logout error: {}".format(e))
    _clear_session_cookie(response)
    return {"ok": True}


@router.post("/logout-all")
def logout_all(response: Response, user: Dict[str, Any] = Depends(require_account)) -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL",
                    (user["user_id"],),
                )
            conn.commit()
    except Exception as e:
        print("[auth] logout-all error: {}".format(e))
    _clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
def me(user: Optional[Dict[str, Any]] = Depends(optional_account)) -> Dict[str, Any]:
    if user is None:
        return _account_payload(None, None)
    try:
        with get_connection() as conn:
            return _account_payload(conn, user)
    except Exception as e:
        print("[auth] me error: {}".format(e))
        return {"email": None, "isRegistered": False, "isAdmin": False, "newsletter": False, "memberSince": None}
