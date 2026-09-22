-- Calculation/schema version tag for Macro Options Flow (live, full-flow backfill and
-- iv-warmup rows alike). DEFAULT '1.0.0' is semantically correct for every row that exists
-- as of this migration: nothing about trade classification, the Greek as-of join, IV/skew/
-- term-structure math, DTE buckets, OI matching, or the sentiment/delta formulas has changed
-- since api/services/options_flow_metrics.METHODOLOGY_VERSION was introduced at "1.0.0" --
-- this column backfills existing rows with the version that in fact produced them, it does
-- not merely stamp a placeholder. A later methodology change must bump METHODOLOGY_VERSION
-- in code (store.py then writes the new value explicitly); it must NOT edit this column's
-- default, which would misrepresent already-written rows.

ALTER TABLE options_flow_runs
    ADD COLUMN IF NOT EXISTS methodology_version TEXT NOT NULL DEFAULT '1.0.0';

ALTER TABLE options_flow_symbol_snapshots
    ADD COLUMN IF NOT EXISTS methodology_version TEXT NOT NULL DEFAULT '1.0.0';

CREATE INDEX IF NOT EXISTS idx_options_flow_snap_methodology_version
    ON options_flow_symbol_snapshots (methodology_version);
