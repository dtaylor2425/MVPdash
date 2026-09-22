-- Control-plane only for Macro Options Flow: a single-row heartbeat showing what the active
-- backfill process is currently doing (jobs/options_flow_backfill.py --status). Whether a
-- process may start is decided entirely by the Postgres advisory lock
-- (api/services/options_flow_lock.py) -- this table is observability only and is never
-- consulted to authorize starting a new process. It never touches options_flow_runs /
-- options_flow_symbol_snapshots, so it cannot affect any analytics snapshot.

CREATE TABLE IF NOT EXISTS options_flow_backfill_heartbeat (
    id                    INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),   -- singleton row
    run_process_id        TEXT NOT NULL,
    phase                 TEXT,               -- 'full_flow' | 'iv_warmup' | NULL
    ticker                TEXT,
    market_date           DATE,
    hostname              TEXT,
    pid                   INT,
    started_at            TIMESTAMPTZ NOT NULL,
    last_heartbeat        TIMESTAMPTZ NOT NULL,
    methodology_version   TEXT
);
