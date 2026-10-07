CREATE TABLE IF NOT EXISTS theta_stock_snapshots (
 ticker TEXT NOT NULL,
 market_date DATE NOT NULL,
 payload JSONB NOT NULL,
 daily_bars JSONB NOT NULL DEFAULT '[]'::jsonb,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY (ticker, market_date)
);
CREATE INDEX IF NOT EXISTS theta_stock_latest ON theta_stock_snapshots (ticker, market_date DESC);

-- Shared input cache also covers SPY/sector benchmarks without public snapshots.
CREATE TABLE IF NOT EXISTS theta_stock_daily_cache (
 ticker TEXT PRIMARY KEY,
 market_date DATE NOT NULL,
 daily_bars JSONB NOT NULL,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
