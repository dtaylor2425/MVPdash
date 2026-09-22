"""
api/services/options_flow_errors.py

Classifies an exception raised while processing one ticker-day as either a normal,
recoverable per-ticker-day failure (record it, continue, retry on a future --resume -- the
existing, validated behaviour) or a RUN-LEVEL FATAL infrastructure error (invalid/duplicate
ThetaData session, authentication/API-key rejection, or a database that has become
unreachable) that must stop the entire process immediately rather than burn through the rest
of the ticker-days hitting the same wall. This is orchestration only: it inspects exception
types and messages, never any analytics value.
"""

from __future__ import annotations

from typing import Optional

# Case-insensitive substrings that, anywhere in an exception's message, mean "this is not
# about today's data, this is the whole session/connection" -- matched against the exact
# text seen in real ThetaData/psycopg failures (e.g. "Invalid session ID. This can occur if
# more than one terminal is running.").
_FATAL_MESSAGE_PATTERNS = (
    "invalid session id",
    "more than one terminal",
    "duplicate terminal",
    "authentication failed",
    "unauthenticated",
    "permission_denied",
    "invalid api key",
    "api key",
    "apikey",
)

CATEGORY_SESSION_OR_AUTH = "thetadata_session_or_auth"
CATEGORY_DATABASE_UNAVAILABLE = "database_unavailable"
CATEGORY_CLIENT_UNAVAILABLE = "thetadata_client_unavailable"


class FatalBackfillError(RuntimeError):
    """Raised to unwind out of the per-ticker-day loop and stop the whole run() invocation.
    Carries the classification category and the original exception."""

    def __init__(self, category: str, original: BaseException):
        super().__init__("{}: {}".format(category, original))
        self.category = category
        self.original = original


def _is_operational_db_error(exc: BaseException) -> bool:
    try:
        import psycopg
    except ImportError:
        return False
    return isinstance(exc, psycopg.OperationalError)


def classify_error(exc: BaseException) -> Optional[str]:
    """None if `exc` is an ordinary per-ticker-day failure. A category string if it is
    run-level fatal and the caller must stop instead of continuing."""
    if _is_operational_db_error(exc):
        return CATEGORY_DATABASE_UNAVAILABLE
    if type(exc).__name__ in ("AuthenticationError",):
        # thetadata.errors.AuthenticationError (or our own THETADATA_API_KEY-not-set
        # RuntimeError, matched below): the client never connected at all -- there is no
        # such thing as "retry this one ticker-day" for that.
        return CATEGORY_CLIENT_UNAVAILABLE
    msg = str(exc).lower()
    if "thetadata_api_key is not set" in msg:
        return CATEGORY_CLIENT_UNAVAILABLE
    for pat in _FATAL_MESSAGE_PATTERNS:
        if pat in msg:
            return CATEGORY_SESSION_OR_AUTH
    return None
