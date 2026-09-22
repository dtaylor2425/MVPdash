-- Historical backfill support for Macro Options Flow.
-- Distinguishes live intraday snapshots from reconstructed daily ones and
-- makes a duplicate (ticker, market_date) backfill row impossible.
-- Idempotent; existing rows become source = 'live'. Applied by the workers
-- (store.ensure_schema) or scripts/run_options_flow_migration.py.

ALTER TABLE options_flow_runs
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'live'
    CHECK (source IN ('live', 'historical_backfill'));

ALTER TABLE options_flow_symbol_snapshots
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'live'
    CHECK (source IN ('live', 'historical_backfill'));

-- At most ONE backfilled snapshot per ticker per day. Live intraday snapshots
-- (many per day) are unaffected because the index is partial.
CREATE UNIQUE INDEX IF NOT EXISTS uq_options_flow_snap_backfill
    ON options_flow_symbol_snapshots (ticker, market_date)
    WHERE source = 'historical_backfill';

CREATE INDEX IF NOT EXISTS idx_options_flow_runs_source_date
    ON options_flow_runs (source, market_date DESC, created_at DESC);
