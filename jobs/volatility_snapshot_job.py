"""
Persist one Macro Engine volatility snapshot to Postgres.
Recommended Railway cron: */15 13-21 * * 1-5
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from api.services.volatility_engine import build_volatility_snapshot
except Exception:
    from volatility_engine import build_volatility_snapshot

DDL = """
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
    payload         JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_volatility_snapshots_ts_desc ON volatility_snapshots (ts DESC);
"""

INSERT = """
INSERT INTO volatility_snapshots (
    ts, spx, source_symbol, is_spy_fallback,
    iv_7d, iv_14d, iv_30d, iv_60d,
    spread_7d_14d, spread_14d_30d, spread_30d_60d,
    natr_json, regime, payload
) VALUES (
    %(ts)s, %(spx)s, %(source_symbol)s, %(is_spy_fallback)s,
    %(iv_7d)s, %(iv_14d)s, %(iv_30d)s, %(iv_60d)s,
    %(spread_7d_14d)s, %(spread_14d_30d)s, %(spread_30d_60d)s,
    %(natr_json)s::jsonb, %(regime)s, %(payload)s::jsonb
)
ON CONFLICT (ts) DO NOTHING;
"""


def _database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required")
    return url


def main() -> None:
    import psycopg

    snap = build_volatility_snapshot()
    iv = snap["implied_volatility"]
    curve = iv["curve"]
    spreads = iv["spreads"]
    params = {
        "ts": snap["timestamp"],
        "spx": snap["spx"],
        "source_symbol": iv["source_symbol"],
        "is_spy_fallback": iv["is_spy_fallback"],
        "iv_7d": curve["7d"],
        "iv_14d": curve["14d"],
        "iv_30d": curve["30d"],
        "iv_60d": curve["60d"],
        "spread_7d_14d": spreads["7d_14d"],
        "spread_14d_30d": spreads["14d_30d"],
        "spread_30d_60d": spreads["30d_60d"],
        "natr_json": json.dumps(snap["natr"]),
        "regime": snap["divergence"]["regime"],
        "payload": json.dumps(snap),
    }
    with psycopg.connect(_database_url()) as conn:
        conn.execute(DDL)
        conn.execute(INSERT, params)
        conn.commit()
    print(json.dumps({
        "timestamp": snap["timestamp"],
        "spx": snap["spx"],
        "regime": snap["divergence"]["regime"],
        "source_symbol": iv["source_symbol"],
        "is_spy_fallback": iv["is_spy_fallback"],
        "options_ok": snap["data_quality"]["options_ok"],
    }, indent=2))


if __name__ == "__main__":
    main()
