-- Accounts, magic-link auth, sessions, entitlements (docs/PLATFORM-BACKEND-PLAN-V2.md).
-- IDs are generated in application code (uuid4), not DB defaults, so this
-- migration has no dependency on the pgcrypto extension being enabled.

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY,
    email           TEXT UNIQUE NOT NULL,      -- normalised: trim + lowercase
    email_raw       TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at   TIMESTAMPTZ,
    is_admin        BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS login_tokens (
    token_hash      TEXT PRIMARY KEY,          -- sha256 hex; raw token never stored
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL,
    consumed_at     TIMESTAMPTZ,
    request_ip      TEXT,
    request_ua      TEXT
);

CREATE INDEX IF NOT EXISTS idx_login_tokens_user_id ON login_tokens (user_id);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY,          -- opaque; NOT a JWT
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL,      -- 30 days, sliding
    revoked_at      TIMESTAMPTZ,
    last_seen_at    TIMESTAMPTZ,
    user_agent      TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at);

CREATE TABLE IF NOT EXISTS entitlements (
    email                TEXT PRIMARY KEY,     -- may exist before the user does
    tier                 TEXT NOT NULL DEFAULT 'registered',  -- future: 'paid'
    newsletter           BOOLEAN NOT NULL DEFAULT FALSE,
    newsletter_since     DATE,
    source               TEXT NOT NULL,        -- 'substack_csv' | 'manual'
    last_seen_in_import  TIMESTAMPTZ,
    note                 TEXT
);

CREATE TABLE IF NOT EXISTS auth_events (
    id      BIGSERIAL PRIMARY KEY,
    at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    email   TEXT,
    event   TEXT NOT NULL,             -- 'request' | 'login' | 'fail' | 'logout' | 'substack_import'
    ip      TEXT,
    detail  JSONB
);

CREATE INDEX IF NOT EXISTS idx_auth_events_email_at ON auth_events (email, at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_events_ip_at ON auth_events (ip, at DESC);
