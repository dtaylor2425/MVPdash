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

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import date, timedelta
from typing import Any, Dict, Optional
from urllib.parse import quote, urlencode

import requests
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

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
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ADMIN_EMAILS = {
    normalize_email(e) for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
}

RATE_LIMIT_PER_EMAIL = 3
RATE_LIMIT_EMAIL_WINDOW = timedelta(minutes=15)
RATE_LIMIT_PER_IP = 10
RATE_LIMIT_IP_WINDOW = timedelta(hours=1)

# --- Google OAuth (added on top of the original plan; magic-link is unaffected) ---
# Not in docs/PLATFORM-BACKEND-PLAN-V2.md or the frontend spec (which explicitly
# said "no social buttons") -- added per direct user request 2026-09-15 as a
# second sign-in path alongside magic-link, not a replacement.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI") or (API_BASE_URL.rstrip("/") + "/api/auth/google/callback")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
OAUTH_STATE_TTL_SECONDS = 600  # 10 minutes; only needs to survive the consent screen


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


def _sign_oauth_state(next_path: str) -> str:
    """Binds `next` to a signed, expiring token so the Google redirect chain
    can't be used for CSRF (a forged callback hit) or an open redirect (a
    tampered `next`) — same threat model as request-link's `next` guard, but
    this has to survive a real browser round-trip through Google instead of
    living in one fetch call, so it's carried in `state` instead of a cookie
    or server-side session."""
    payload = json.dumps({"n": generate_login_token(), "next": next_path, "exp": int(time.time()) + OAUTH_STATE_TTL_SECONDS})
    payload_b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return "{}.{}".format(payload_b64, sig)


def _verify_oauth_state(state: str) -> Optional[str]:
    """Returns the validated `next` path, or None if missing/tampered/expired."""
    if not state or "." not in state:
        return None
    payload_b64, _, sig = state.partition(".")
    expected_sig = hmac.new(SESSION_SECRET.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    next_path = payload.get("next") or "/account"
    if not next_path.startswith("/") or next_path.startswith("//"):
        next_path = "/account"
    return next_path


def _log_event(conn, email: Optional[str], event: str, ip: str, detail: Optional[Dict[str, Any]] = None) -> None:
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


@router.get("/google")
def google_login(next: str = Query(default="/account")) -> RedirectResponse:
    login_url = APP_BASE_URL.rstrip("/") + "/login"

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not SESSION_SECRET:
        return RedirectResponse("{}?error=google_unconfigured".format(login_url), status_code=302)

    next_path = next if (next.startswith("/") and not next.startswith("//")) else "/account"
    state = _sign_oauth_state(next_path)
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return RedirectResponse(
        "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params),
        status_code=302,
    )


@router.get("/google/callback")
def google_callback(request: Request, code: str = "", state: str = "", error: str = "") -> RedirectResponse:
    """Full-page redirect chain, not a fetch target — Google redirects the
    browser here directly, so failures redirect to /login?error=... (for the
    frontend to render) rather than returning JSON, and success redirects
    straight to `next` with the session cookie already set. Contrast with
    GET /callback (magic-link), which returns JSON because the frontend
    calls that one via fetch()."""
    ip = _client_ip(request)
    login_url = APP_BASE_URL.rstrip("/") + "/login"

    if error:
        return RedirectResponse("{}?error=google_denied".format(login_url), status_code=302)

    next_path = _verify_oauth_state(state)
    if next_path is None:
        return RedirectResponse("{}?error=state_invalid".format(login_url), status_code=302)

    if not code:
        return RedirectResponse("{}?error=google_failed".format(login_url), status_code=302)

    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json().get("access_token")
        if not access_token:
            raise ValueError("no access_token in Google's response")

        # Bearer call to Google's own endpoint -- Google authenticates the
        # token for us, so there's no JWT/JWKS verification to reimplement.
        profile_resp = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": "Bearer {}".format(access_token)},
            timeout=10,
        )
        profile_resp.raise_for_status()
        profile = profile_resp.json()
    except Exception as e:
        print("[auth] google token exchange failed: {}".format(e))
        return RedirectResponse("{}?error=google_failed".format(login_url), status_code=302)

    email_raw = str(profile.get("email") or "")
    email = normalize_email(email_raw)
    if not profile.get("email_verified") or not is_valid_email(email):
        return RedirectResponse("{}?error=email_unverified".format(login_url), status_code=302)

    try:
        with get_connection() as conn:
            # Same users table, keyed by email, as magic-link -- a Google
            # sign-in with an email that already has a magic-link account
            # lands on that same account automatically. No linking step.
            user = _get_or_create_user(conn, email, email_raw)
            user = _maybe_promote_admin(conn, user)

            session_id = new_id()
            session_expiry = utcnow() + timedelta(days=SESSION_TTL_DAYS)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, expires_at, last_seen_at, user_agent)
                    VALUES (%s, %s, %s, now(), %s)
                    """,
                    (session_id, user["id"], session_expiry, request.headers.get("user-agent", "")),
                )
                cur.execute("UPDATE users SET last_login_at = now() WHERE id = %s", (user["id"],))

            _log_event(conn, email, "login", ip, {"method": "google"})
            conn.commit()
    except Exception as e:
        print("[auth] google callback error: {}".format(e))
        return RedirectResponse("{}?error=server_error".format(login_url), status_code=302)

    redirect = RedirectResponse(APP_BASE_URL.rstrip("/") + next_path, status_code=302)
    _set_session_cookie(redirect, session_id)
    return redirect


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
