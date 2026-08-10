from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.db import get_connection


def main() -> None:
    sql = (ROOT / "sql" / "001_portfolio_snapshots.sql").read_text(encoding="utf-8")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Portfolio snapshot migration applied.")


if __name__ == "__main__":
    main()
