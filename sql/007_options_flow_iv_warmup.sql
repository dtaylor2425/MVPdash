-- Lightweight IV-warmup backfill mode for Macro Options Flow.
-- `mode` distinguishes a full trade-tape day from an IV-only day fetched with
-- jobs/options_flow_backfill.py --mode iv-warmup, independent of `source`
-- (an iv-warmup row is always source = 'historical_backfill'). Idempotent;
-- existing rows become mode = 'full_flow'.

ALTER TABLE options_flow_runs
    ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'full_flow'
    CHECK (mode IN ('full_flow', 'iv_warmup'));

ALTER TABLE options_flow_symbol_snapshots
    ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'full_flow'
    CHECK (mode IN ('full_flow', 'iv_warmup'));

CREATE INDEX IF NOT EXISTS idx_options_flow_runs_mode_date
    ON options_flow_runs (mode, market_date DESC);
