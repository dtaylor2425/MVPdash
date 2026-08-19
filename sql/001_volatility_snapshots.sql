CREATE TABLE IF NOT EXISTS volatility_snapshots (
    ts              TIMESTAMPTZ PRIMARY KEY,
    spx             DOUBLE PRECISION NOT NULL,
    source_symbol   TEXT,
    is_spy_fallback BOOLEAN NOT NULL DEFAULT FALSE,
    iv_7d           DOUBLE PRECISION,
    iv_14d          DOUBLE PRECISION,
    iv_30d          DOUBLE PRECISION,
    iv_60d          DOUBLE PRECISION,
    spread_7d_14d   DOUBLE PRECISION,
    spread_14d_30d  DOUBLE PRECISION,
    spread_30d_60d  DOUBLE PRECISION,
    natr_json       JSONB NOT NULL,
    regime          TEXT,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_volatility_snapshots_ts_desc
    ON volatility_snapshots (ts DESC);

CREATE INDEX IF NOT EXISTS idx_volatility_snapshots_regime
    ON volatility_snapshots (regime);
