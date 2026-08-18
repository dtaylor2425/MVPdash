"""Run volatility snapshot table migration.

Usage:
    python scripts/run_volatility_snapshot_migration.py

Required env:
    DATABASE_URL
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.db import get_connection


def main() -> None:
    sql = (ROOT / "sql" / "001_volatility_snapshots.sql").read_text(encoding="utf-8")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Volatility snapshot migration applied.")


if __name__ == "__main__":
    main()
