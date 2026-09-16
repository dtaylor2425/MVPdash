from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def main() -> None:
    import psycopg

    sql_path = ROOT / "sql" / "004_macro_thesis.sql"
    sql = sql_path.read_text(encoding="utf-8")

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

    print("Macro thesis migration applied (macro_quadrant_history, macro_thesis_snapshots, house_view).")


if __name__ == "__main__":
    main()
