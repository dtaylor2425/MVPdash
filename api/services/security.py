"""
api/services/security.py

Token generation/hashing and small helpers shared by the auth router and
auth dependencies. No DB or FastAPI imports here so this stays unit-testable
without a Postgres connection.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timezone


def generate_login_token() -> str:
    """256 bits of entropy, URL-safe. The raw value is only ever emailed,
    never stored — only its hash is persisted."""
    return secrets.token_urlsafe(32)


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def tokens_match(computed_hash: str, stored_hash: str) -> bool:
    return hmac.compare_digest(computed_hash, stored_hash)


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def is_valid_email(email: str) -> bool:
    if not email or " " in email or "@" not in email:
        return False
    local, _, domain = email.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")
