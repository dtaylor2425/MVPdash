-- Macro Options Flow (private page backed by ThetaData Options Standard).
-- The theta-options-worker computes DERIVED analytics and writes them here;
-- the API only ever reads these tables. No raw OPRA ticks are stored.
-- Idempotent: safe to re-run (the worker also runs it on start).

-- gen_random_uuid() is built in on PG >= 13; older servers need pgcrypto. Tolerate a server
-- that doesn't ship the extension -- on PG < 13 the CREATE TABLE below then fails loudly instead.
DO $$ BEGIN
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
EXCEPTION WHEN OTHERS THEN
    NULL;
END $$;

CREATE TABLE IF NOT EXISTS options_flow_runs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_date       DATE NOT NULL,
    as_of_timestamp   TIMESTAMPTZ NOT NULL,      -- newest trade/Greek data included in the run
    status            TEXT NOT NULL,             -- running | success | partial | failed
    config            JSONB NOT NULL DEFAULT '{}'::jsonb,
    diagnostics       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    CONSTRAINT options_flow_runs_status_chk
        CHECK (status IN ('running', 'success', 'partial', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_options_flow_runs_date_created
    ON options_flow_runs (market_date DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_options_flow_runs_status_created
    ON options_flow_runs (status, created_at DESC);

CREATE TABLE IF NOT EXISTS options_flow_symbol_snapshots (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id            UUID NOT NULL REFERENCES options_flow_runs(id) ON DELETE CASCADE,
    ticker            TEXT NOT NULL,
    group_name        TEXT NOT NULL,
    -- Denormalised from the run / payload so "latest per ticker" and the
    -- 1D/5D/percentile history queries never have to unpack JSONB.
    market_date       DATE NOT NULL,
    as_of_timestamp   TIMESTAMPTZ NOT NULL,
    sentiment         DOUBLE PRECISION,          -- net_directional / gross, [-1, 1]
    sentiment_label   TEXT,
    atm_iv            DOUBLE PRECISION,          -- 30D constant-maturity ATM IV, decimal
    payload           JSONB NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT options_flow_symbol_snapshots_run_ticker_uniq UNIQUE (run_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_options_flow_snap_ticker_asof
    ON options_flow_symbol_snapshots (ticker, as_of_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_options_flow_snap_ticker_date
    ON options_flow_symbol_snapshots (ticker, market_date DESC, as_of_timestamp DESC);
