"""
tests/test_auth.py  (docs/PLATFORM-BACKEND-PLAN-V2.md)

Covers the DB-independent pieces of the accounts/auth build: token
hashing, email normalisation, and the Substack CSV column-detection and
diff logic. Anything that needs Postgres (sessions, entitlements upserts)
is exercised by hand against Railway per the plan's build-order step 3 —
there's no local Postgres in this environment (see memory: fx-backend-build).

Runs standalone or under pytest:

    python tests/test_auth.py
    pytest tests/test_auth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routers.admin import _detect_columns, _parse_csv_rows, _parse_date
from api.services.security import (
    generate_login_token,
    hash_token,
    is_valid_email,
    normalize_email,
    tokens_match,
)


def test_token_roundtrip():
    raw = generate_login_token()
    h = hash_token(raw)
    assert len(raw) >= 32
    assert tokens_match(h, hash_token(raw))
    assert not tokens_match(h, hash_token("something-else"))


def test_email_normalize_and_validate():
    assert normalize_email("  Foo@Bar.COM ") == "foo@bar.com"
    assert is_valid_email("foo@bar.com")
    assert not is_valid_email("not-an-email")
    assert not is_valid_email("a@b")
    assert not is_valid_email("")


def test_csv_standard_shape():
    raw = b"email,created_at\nFoo@Bar.com,2026-01-05\nbad-email,2026-01-06\nbaz@qux.com,01/15/2026\n"
    rows, skipped = _parse_csv_rows(raw)
    assert rows == {"foo@bar.com": "2026-01-05", "baz@qux.com": "01/15/2026"}
    assert skipped == 1


def test_csv_alternate_header_spelling():
    raw = b"Email Address,Subscription Date\nA@B.com,2026-02-01\n"
    rows, skipped = _parse_csv_rows(raw)
    assert rows == {"a@b.com": "2026-02-01"}
    assert skipped == 0


def test_csv_unrecognised_shape_fails_loudly():
    raw = b"subscriber_id,plan\n123,paid\n"
    try:
        _parse_csv_rows(raw)
        raise AssertionError("expected ValueError for unrecognised CSV shape")
    except ValueError as e:
        assert "email column" in str(e)


def test_detect_columns_case_insensitive():
    email_field, date_field = _detect_columns(["EMAIL", "Created_At"])
    assert email_field == "EMAIL"
    assert date_field == "Created_At"


def test_parse_date_formats():
    assert str(_parse_date("2026-01-05")) == "2026-01-05"
    assert str(_parse_date("01/15/2026")) == "2026-01-15"
    assert _parse_date(None) is None
    assert _parse_date("garbage") is None


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
