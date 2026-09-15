"""
api/routers/admin.py  (docs/PLATFORM-BACKEND-PLAN-V2.md section 4)

    POST /api/admin/substack-import   -- multipart CSV from the Substack export
    POST /api/admin/entitlements      -- manual entitlement edits / comps

Both routes require require_admin(). This import is informational only —
nothing in the product is gated on newsletter status, so a stale or failed
import breaks nothing (per spec section 4).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File

from api.auth_deps import require_admin
from api.db import get_connection
from api.services.security import is_valid_email, normalize_email

router = APIRouter(prefix="/api/admin", tags=["admin"])

EMAIL_HEADER_CANDIDATES = {
    "email", "email address", "subscriber email", "subscriber_email",
    "subscriberemail", "e-mail", "e mail",
}
DATE_HEADER_CANDIDATES = {
    "created_at", "created at", "subscribed_at", "subscribed at",
    "subscription date", "subscription_date", "date subscribed",
    "signed up", "signed_up", "created", "date",
}

# Substack's own export uses a handful of shapes across free/paid publications;
# these are the header spellings actually seen in exports, not a guess.


def _detect_columns(fieldnames: Sequence[str]) -> Tuple[str, Optional[str]]:
    """Match headers case-insensitively. Raises ValueError (caller turns this
    into a loud 400) rather than silently importing zero rows on an
    unrecognised shape."""
    normalized = {str(f).strip().lower(): f for f in fieldnames or []}

    email_field = None
    for candidate in EMAIL_HEADER_CANDIDATES:
        if candidate in normalized:
            email_field = normalized[candidate]
            break
    if email_field is None:
        raise ValueError(
            "No recognisable email column. Found headers: {}. Expected one of: {}".format(
                sorted(normalized.keys()), sorted(EMAIL_HEADER_CANDIDATES)
            )
        )

    date_field = None
    for candidate in DATE_HEADER_CANDIDATES:
        if candidate in normalized:
            date_field = normalized[candidate]
            break

    return email_field, date_field


def _parse_date(value: Optional[str]):
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value[:26], fmt).date()
        except ValueError:
            continue
    # Common ISO-with-timezone shape, e.g. "2026-01-05T12:00:00.000Z"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_csv_rows(raw: bytes) -> Tuple[Dict[str, Optional[str]], int]:
    """Returns (email -> newsletter_since_str, skipped_row_count)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")

    email_field, date_field = _detect_columns(reader.fieldnames)

    out: Dict[str, Optional[str]] = {}
    skipped = 0
    for row in reader:
        email = normalize_email(row.get(email_field, ""))
        if not is_valid_email(email):
            skipped += 1
            continue
        out[email] = row.get(date_field) if date_field else None
    return out, skipped


@router.post("/substack-import")
def substack_import(
    admin: Dict[str, Any] = Depends(require_admin),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    raw = file.file.read()
    try:
        csv_emails, skipped = _parse_csv_rows(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not csv_emails:
        raise HTTPException(status_code=400, detail="No valid email rows found in the CSV.")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT email FROM entitlements WHERE newsletter = TRUE")
                existing = {row["email"] for row in cur.fetchall()}

            csv_email_set = set(csv_emails.keys())
            added = csv_email_set - existing
            removed = existing - csv_email_set
            unchanged = csv_email_set & existing

            with conn.cursor() as cur:
                for email, since_raw in csv_emails.items():
                    since = _parse_date(since_raw)
                    cur.execute(
                        """
                        INSERT INTO entitlements (email, tier, newsletter, newsletter_since, source, last_seen_in_import)
                        VALUES (%s, 'registered', TRUE, %s, 'substack_csv', now())
                        ON CONFLICT (email) DO UPDATE SET
                            newsletter = TRUE,
                            newsletter_since = COALESCE(entitlements.newsletter_since, EXCLUDED.newsletter_since),
                            source = 'substack_csv',
                            last_seen_in_import = now()
                        """,
                        (email, since),
                    )

                if removed:
                    cur.execute(
                        "UPDATE entitlements SET newsletter = FALSE WHERE email = ANY(%s) AND newsletter = TRUE",
                        (list(removed),),
                    )

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO auth_events (email, event, ip, detail) VALUES (%s, %s, %s, %s)",
                    (
                        admin.get("email"),
                        "substack_import",
                        None,
                        json.dumps({
                            "added": len(added),
                            "removed": len(removed),
                            "unchanged": len(unchanged),
                            "skipped_rows": skipped,
                        }),
                    ),
                )
            conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Import failed: {}".format(e))

    return {
        "added": sorted(added)[:200],
        "removed": sorted(removed)[:200],
        "added_count": len(added),
        "removed_count": len(removed),
        "unchanged_count": len(unchanged),
        "skipped_rows": skipped,
        "total_rows_processed": len(csv_emails),
    }


@router.post("/entitlements")
def upsert_entitlement(
    body: Dict[str, Any] = Body(...),
    admin: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    email = normalize_email(str(body.get("email") or ""))
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Valid email required")

    tier = body.get("tier") if "tier" in body else None
    newsletter = body.get("newsletter") if "newsletter" in body else None
    note = body.get("note") if "note" in body else None

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO entitlements (email, tier, newsletter, source, note)
                    VALUES (%s, COALESCE(%s, 'registered'), COALESCE(%s, FALSE), 'manual', %s)
                    ON CONFLICT (email) DO UPDATE SET
                        tier = COALESCE(%s, entitlements.tier),
                        newsletter = COALESCE(%s, entitlements.newsletter),
                        note = COALESCE(%s, entitlements.note),
                        source = 'manual'
                    RETURNING *
                    """,
                    (email, tier, newsletter, note, tier, newsletter, note),
                )
                row = cur.fetchone()
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Update failed: {}".format(e))

    if row.get("newsletter_since") is not None and hasattr(row["newsletter_since"], "isoformat"):
        row["newsletter_since"] = row["newsletter_since"].isoformat()
    return row
