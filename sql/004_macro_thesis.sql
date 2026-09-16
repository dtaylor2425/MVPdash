-- Macro thesis engine (docs/MACRO-THESIS-BACKEND-SPEC.md sections 3, 8, 9).
-- Compute in the nightly job, publish a snapshot, never recompute on request
-- (same pattern as fx_snapshots / sql/002_fx_snapshots.sql).

CREATE TABLE IF NOT EXISTS macro_quadrant_history (
    as_of               DATE PRIMARY KEY,
    growth_level        DOUBLE PRECISION,
    growth_momentum     DOUBLE PRECISION,
    inflation_level     DOUBLE PRECISION,
    inflation_momentum  DOUBLE PRECISION,
    quadrant            TEXT,
    strength            DOUBLE PRECISION,
    phase               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_macro_quadrant_history_as_of_desc
    ON macro_quadrant_history (as_of DESC);

CREATE TABLE IF NOT EXISTS macro_thesis_snapshots (
    as_of       DATE PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    quadrant    TEXT,
    payload     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_macro_thesis_snapshots_as_of_desc
    ON macro_thesis_snapshots (as_of DESC);

-- Admin-editable layer (spec section 8). The model's mechanical read and
-- this table are never merged -- the API returns both under separate keys
-- so a human/model disagreement is visible, not silently resolved.
CREATE TABLE IF NOT EXISTS house_view (
    as_of               DATE PRIMARY KEY,
    agrees_with_model   BOOLEAN NOT NULL,
    headline            TEXT,
    body                TEXT,
    conviction          TEXT,          -- 'high' | 'medium' | 'low'
    author_note         TEXT,
    created_by          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_house_view_as_of_desc
    ON house_view (as_of DESC);
