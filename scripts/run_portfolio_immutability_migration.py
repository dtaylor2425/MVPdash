"""Back up portfolio tables and install publication guards atomically."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.db import get_connection

TABLES = ("portfolio_runs", "portfolio_positions", "portfolio_performance", "portfolio_rebalances")


def migrate(backup: Path) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '15s'")
            # Same strategy locks as the publisher, then prevent all table writes
            # while backing up and installing guards. Public reads remain available.
            for strategy in ("stock_alpha", "smid_growth"):
                cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ("portfolio:" + strategy,))
            cur.execute("LOCK TABLE " + ", ".join(TABLES) + " IN SHARE ROW EXCLUSIVE MODE")
            before = {}
            for table in TABLES:
                cur.execute(f"SELECT row_to_json(t) AS record FROM {table} t ORDER BY id")
                before[table] = [row["record"] for row in cur.fetchall()]
            backup.parent.mkdir(parents=True, exist_ok=True)
            with backup.open("x", encoding="utf-8") as handle:
                json.dump(before, handle, ensure_ascii=False)
            sql = (ROOT / "sql/002_portfolio_publication_immutability.sql").read_text(encoding="utf-8")
            # Let the connection own the transaction, including backup verification.
            sql = sql.replace("BEGIN;\nCREATE", "CREATE", 1)
            sql = sql.removesuffix("COMMIT;\n").removesuffix("COMMIT;")
            cur.execute(sql)
            for table in TABLES:
                cur.execute(f"SELECT row_to_json(t) AS record FROM {table} t ORDER BY id")
                if [row["record"] for row in cur.fetchall()] != before[table]:
                    raise RuntimeError(f"Migration changed {table}; rolling back")
        conn.commit()
    print("Publication guards installed; all portfolio rows match the backup.")
    print("Backed up rows: " + json.dumps({table: len(rows) for table, rows in before.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True, help="New JSON backup path (never overwritten)")
    migrate(parser.parse_args().backup)
