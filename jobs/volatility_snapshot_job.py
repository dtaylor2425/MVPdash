"""
Build and persist a volatility snapshot.

Recommended Railway cron:
    */15 13-21 * * 1-5

That runs every 15 minutes during broad US market hours in UTC. The job exits
when complete, so it is safe for Railway cron.

Required env:
    DATABASE_URL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.db import get_connection
from api.services.volatility_engine import build_volatility_snapshot

DDL_PATH = ROOT / "sql" / "001_volatility_snapshots.sql"

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
    %(natr_json)s, %(regime)s, %(payload)s
)
ON CONFLICT (ts) DO NOTHING;
"""


def main() -> None:
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
        "natr_json": Jsonb(snap["natr"]),
        "regime": snap["divergence"]["regime"],
        "payload": Jsonb(snap),
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_PATH.read_text(encoding="utf-8"))
            cur.execute(INSERT, params)
        conn.commit()

    print(
        json.dumps(
            {
                "timestamp": snap["timestamp"],
                "spx": snap["spx"],
                "regime": snap["divergence"]["regime"],
                "source_symbol": iv["source_symbol"],
                "source": "volatility_snapshot_job",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
