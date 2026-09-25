from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.db import get_connection

if __name__ == "__main__":
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute((ROOT / "sql/010_research_usage.sql").read_text())
        conn.commit()
    print("Research usage aggregate table is ready.")
