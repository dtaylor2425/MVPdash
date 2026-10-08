CREATE TABLE IF NOT EXISTS options_scanner_runs (
 id TEXT PRIMARY KEY,
 market_date DATE NOT NULL,
 profile TEXT NOT NULL,
 universe_hash TEXT NOT NULL,
 universe JSONB NOT NULL,
 metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
 status TEXT NOT NULL DEFAULT 'running',
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(market_date,profile,universe_hash)
);
CREATE INDEX IF NOT EXISTS options_scanner_latest ON options_scanner_runs(market_date DESC,updated_at DESC);
CREATE TABLE IF NOT EXISTS options_scanner_symbols (
 run_id TEXT NOT NULL REFERENCES options_scanner_runs(id),
 ticker TEXT NOT NULL,
 status TEXT NOT NULL,
 payload JSONB NOT NULL DEFAULT '{}'::jsonb,
 diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(run_id,ticker)
);
CREATE INDEX IF NOT EXISTS options_scanner_history ON options_scanner_symbols(ticker,status);
