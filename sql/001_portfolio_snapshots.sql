CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS portfolio_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  strategy TEXT NOT NULL,
  run_date DATE NOT NULL,
  as_of_timestamp TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('draft', 'published', 'failed')),
  is_published BOOLEAN NOT NULL DEFAULT FALSE,
  config JSONB NOT NULL DEFAULT '{}'::jsonb,
  diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  failure_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  published_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_portfolio_runs_strategy_date
  ON portfolio_runs(strategy, run_date DESC, as_of_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_portfolio_runs_published
  ON portfolio_runs(strategy, is_published, run_date DESC, as_of_timestamp DESC)
  WHERE status = 'published';

CREATE UNIQUE INDEX IF NOT EXISTS ux_portfolio_runs_one_published_per_day
  ON portfolio_runs(strategy, run_date)
  WHERE status = 'published' AND is_published = TRUE;

CREATE TABLE IF NOT EXISTS portfolio_positions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES portfolio_runs(id) ON DELETE CASCADE,
  ticker TEXT NOT NULL,
  name TEXT,
  sector TEXT,
  theme TEXT,
  target_weight NUMERIC,
  score NUMERIC,
  action TEXT,
  reason TEXT,
  stop_level NUMERIC,
  risk_state TEXT,
  scores JSONB NOT NULL DEFAULT '{}'::jsonb,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_run ON portfolio_positions(run_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker ON portfolio_positions(ticker);

CREATE TABLE IF NOT EXISTS portfolio_performance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES portfolio_runs(id) ON DELETE CASCADE,
  performance_date DATE NOT NULL,
  model_return NUMERIC,
  benchmark_return NUMERIC,
  rebalance_marker JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_run_date ON portfolio_performance(run_id, performance_date);

CREATE TABLE IF NOT EXISTS portfolio_rebalances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES portfolio_runs(id) ON DELETE CASCADE,
  rebalance_date DATE NOT NULL,
  headline TEXT,
  buys JSONB NOT NULL DEFAULT '[]'::jsonb,
  sells JSONB NOT NULL DEFAULT '[]'::jsonb,
  adds JSONB NOT NULL DEFAULT '[]'::jsonb,
  trims JSONB NOT NULL DEFAULT '[]'::jsonb,
  turnover NUMERIC,
  old_holdings JSONB NOT NULL DEFAULT '[]'::jsonb,
  new_holdings JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_portfolio_rebalances_run_date ON portfolio_rebalances(run_id, rebalance_date DESC);

CREATE OR REPLACE VIEW latest_published_portfolio_runs AS
SELECT DISTINCT ON (strategy)
  id, strategy, run_date, as_of_timestamp, status, is_published,
  config, diagnostics, payload, created_at, published_at
FROM portfolio_runs
WHERE status = 'published' AND is_published = TRUE
ORDER BY strategy, run_date DESC, as_of_timestamp DESC, created_at DESC;
