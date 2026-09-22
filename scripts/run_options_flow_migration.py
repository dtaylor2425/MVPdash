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

    files = ["005_options_flow.sql", "006_options_flow_backfill.sql"]

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            for name in files:
                cur.execute((ROOT / "sql" / name).read_text(encoding="utf-8"))
        conn.commit()

    print("Options flow migrations applied ({}).".format(", ".join(files)))


if __name__ == "__main__":
    main()
