# Macro Engine — Accounts and Preview Gating
## Backend plan (revised — supersedes PLATFORM-BACKEND-PLAN.md)

**Removed from scope:** the daily PDF, object storage, signed URLs, per-user
watermarking, and all paid-tier logic.

**Retained:** accounts, sessions, entitlement records, and truncated payloads
for anonymous users. Nothing is paywalled. There are two states — anonymous and
registered — and the entitlement table keeps a `tier` column so a paid tier can
be introduced later without reshaping anything.

Stack unchanged: FastAPI + Postgres on Railway.

---

## 0. The Substack constraint

Substack's Developer API exposes public creator profile data only. There is no
subscriber-verification endpoint, no OAuth, and no webhook for subscription
events. The community `substack_api` wrapper uses exported browser cookies and
must never sit in a login path.

So: **we own authentication, Substack owns the newsletter, and the link between
them is the email address.** A user who signs in with the same email they
subscribed with is matched automatically. That is the whole mechanism, and
because nothing is paywalled, a mismatch is cosmetic rather than a support
emergency — it only affects the newsletter status shown on `/account`.

---

## 1. Schema

```sql
users (
  id              uuid primary key,
  email           text unique not null,      -- normalised: trim + lowercase
  email_raw       text not null,
  created_at      timestamptz not null,
  last_login_at   timestamptz,
  is_admin        boolean not null default false
)

login_tokens (
  token_hash      text primary key,          -- sha256; never store the raw token
  user_id         uuid not null references users(id),
  created_at      timestamptz not null,
  expires_at      timestamptz not null,      -- now() + 15 minutes
  consumed_at     timestamptz,
  request_ip      inet,
  request_ua      text
)

sessions (
  id              uuid primary key,          -- opaque; NOT a JWT
  user_id         uuid not null references users(id),
  created_at      timestamptz not null,
  expires_at      timestamptz not null,      -- 30 days, sliding
  revoked_at      timestamptz,
  last_seen_at    timestamptz,
  user_agent      text
)

entitlements (
  email           text primary key,          -- may exist before the user does
  tier            text not null default 'registered',  -- future: 'paid'
  newsletter      boolean not null default false,
  newsletter_since date,
  source          text not null,             -- 'substack_csv' | 'manual'
  last_seen_in_import timestamptz,
  note            text
)

auth_events (
  id              bigserial primary key,
  at              timestamptz not null,
  email           text,
  event           text not null,             -- 'request' | 'login' | 'fail' | 'logout'
  ip              inet,
  detail          jsonb
)
```

Entitlements are keyed by email rather than `user_id` so a newsletter subscriber
who has never logged in still has a row waiting.

---

## 2. Magic-link authentication

1. `POST /api/auth/request-link` — normalise the email, rate limit (3 per email
   per 15 min, 10 per IP per hour), generate 32 bytes via
   `secrets.token_urlsafe`, store **only the SHA-256 hash**, email the raw token
   as a link. Return an identical response whether or not the account exists.
2. `GET /api/auth/callback?token=…` — hash, look up, constant-time compare,
   check expiry and `consumed_at`. Mark consumed and create the session in one
   transaction. Create the `users` row if absent.
   Cookie: `HttpOnly; Secure; SameSite=Lax; Domain=.macro-engine.com; Max-Age=2592000`.
3. `POST /api/auth/logout` and `POST /api/auth/logout-all`.
4. `GET /api/auth/me` → `{ email, isRegistered, isAdmin, newsletter, memberSince }`.

Sessions are opaque uuids, not JWTs, so revocation is a database update rather
than waiting out a token expiry.

---

## 3. Gating

One dependency used on protected routes:

```python
require_account()     # 401 when anonymous
require_admin()       # 403 unless is_admin
optional_account()    # returns user or None — drives truncation
```

Most routes use `optional_account()` and return **different payload sizes**, not
a permission error:

```python
themes = all_themes if user else all_themes[:3]
return {"items": themes, "total": len(all_themes), "truncated": user is None}
```

`total` and `truncated` let the frontend say "see all 10" accurately without
ever receiving the hidden rows.

**Truncation happens server-side, always.** Never return the full payload with a
flag and rely on the client to hide it.

### Truncation matrix

| Endpoint | Anonymous | Registered |
|---|---|---|
| `/api/regime` | full | full |
| `/api/macro/{pillar}` | hero + first chart series only | all series |
| `/api/themes`, `/api/geopolitics` | first 3 | all 10 |
| `/api/fx/snapshot` | first 5 currencies, no `pairs` | full |
| `/api/fx/breadth` | 401 | full |
| `/api/stock-rankings` | first 10 | full |
| `/api/portfolio/*` | holdings + exposure, no rebalance log | full |
| `/api/report/archive` | latest only | all |

---

## 4. Newsletter sync

Admin-only `POST /api/admin/substack-import`, multipart CSV from the Substack
dashboard export.

- Parse defensively by header name, case-insensitive. Substack's export columns
  vary by publication — **fail loudly on an unrecognised shape** rather than
  importing zero rows silently.
- Upsert `entitlements.newsletter = true` and `newsletter_since` for each email.
- Absent from the import → set `newsletter = false`, but **never** delete the
  row or touch `tier`. Losing a newsletter subscription must not affect account
  access, because access is not paid.
- Write an audit row and return a diff: added / removed / unchanged.

Also `POST /api/admin/entitlements` for manual notes and future comps.

Because nothing is gated on newsletter status, this import is informational
only — cadence is whatever suits you, and a stale import breaks nothing.

---

## 5. Security requirements

- Tokens 256-bit, hashed at rest, single use, 15-minute TTL
- Constant-time comparison on lookup
- Rate limits on every auth endpoint, keyed by both email and IP
- Log every auth event
- CORS: explicit origin list, `allow_credentials=true`. A wildcard origin is
  invalid with credentials and will silently break every authenticated fetch
- No gating logic in the frontend

---

## 6. Build order

1. Schema and migrations
2. Magic link, sessions, `/api/auth/me`
3. `optional_account()` plus truncation on one endpoint — verify by hitting it
   with and without a cookie
4. Roll truncation across the matrix
5. Substack CSV import and admin endpoints

Confirm step 3 by hand before building anything on top of it: the same endpoint
must return 3 items anonymous and 10 items authenticated.

## 7. Environment variables

```
SESSION_COOKIE_NAME=me_session
SESSION_TTL_DAYS=30
LOGIN_TOKEN_TTL_MINUTES=15
COOKIE_DOMAIN=.macro-engine.com
APP_BASE_URL=https://macro-engine.com
API_BASE_URL=https://api.macro-engine.com
CORS_ALLOW_ORIGINS=https://macro-engine.com,https://www.macro-engine.com
ADMIN_EMAILS=you@example.com
EMAIL_PROVIDER_API_KEY=
EMAIL_FROM=Macro Engine <hello@macro-engine.com>
SESSION_SECRET=
```

No storage bucket, no signed-URL config, no billing keys.
