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

    from api.services.options_flow_store import DDL_PATHS  # single source of truth, kept in sync with store.ensure_schema

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            for path in DDL_PATHS:
                cur.execute(path.read_text(encoding="utf-8"))
        conn.commit()

    print("Options flow migrations applied ({}).".format(", ".join(p.name for p in DDL_PATHS)))


if __name__ == "__main__":
    main()
