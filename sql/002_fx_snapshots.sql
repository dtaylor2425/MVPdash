-- FX currency-strength snapshots (spec section 5).
-- Compute on a schedule, publish a snapshot, never recompute on page load.
-- Every snapshot is retained so published scores stay auditable.

CREATE TABLE IF NOT EXISTS fx_snapshots (
    as_of               DATE PRIMARY KEY,
    observation_date    DATE NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    stale               BOOLEAN NOT NULL DEFAULT FALSE,
    incomplete_count    INT NOT NULL DEFAULT 0,
    dropped_components  TEXT[] NOT NULL DEFAULT '{}',
    usd_score           INT,
    reconciliation      JSONB,
    series_health       JSONB,
    payload             JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fx_snapshots_as_of_desc
    ON fx_snapshots (as_of DESC);

CREATE INDEX IF NOT EXISTS idx_fx_snapshots_created_at_desc
    ON fx_snapshots (created_at DESC);
