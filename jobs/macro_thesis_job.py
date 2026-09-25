"""
jobs/macro_thesis_job.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 9)

Computes the full macro thesis snapshot and publishes it. The API router
(api/routers/macro_thesis.py) only ever reads what this job wrote -- it
never recomputes on request.

Recommended Railway cron: daily, e.g. 0 7 * * * (after the overnight FRED
update). The heavy pieces (transition matrix, asset-by-quadrant backtest,
long-term gauge) are cheap enough to recompute every run rather than gating
by calendar cadence -- correctness from always using the latest quadrant
history outweighs the (small) cost saving of a monthly/quarterly skip-list,
which is a real but non-trivial bit of extra state to get right. Simpler:
recompute fresh every run; the *history* table row for the current month
just gets upserted in place until the month rolls over.

    python jobs/macro_thesis_job.py                # build + publish
    python jobs/macro_thesis_job.py --dry-run       # build + print, no writes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.db import get_connection  # noqa: E402
from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES  # noqa: E402
from src.data_sources import get_fred_cached  # noqa: E402
from src.macro_thesis.bis_dsr import load_dsr_us  # noqa: E402
from src.macro_thesis.series_map import ALL_PRICE_TICKERS, EXTRA_FRED_SERIES  # noqa: E402
from src.macro_thesis.snapshot import build_thesis_snapshot  # noqa: E402

DISK_SNAPSHOT = Path(CACHE_DIR) / "macro_thesis_snapshot_latest.json"
DDL = (ROOT / "sql" / "004_macro_thesis.sql").read_text(encoding="utf-8")

PRICE_HISTORY_START = "1999-01-01"  # as far back as available; asset backtest suppresses thin series itself


def _fetch_full_prices() -> pd.DataFrame:
    raw = yf.download(ALL_PRICE_TICKERS, start=PRICE_HISTORY_START, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return px.dropna(how="all")


def _load_latest_house_view(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM house_view ORDER BY as_of DESC LIMIT 1"
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "asOf": row["as_of"].isoformat(),
        "agreesWithModel": row["agrees_with_model"],
        "headline": row["headline"],
        "body": row["body"],
        "conviction": row["conviction"],
        "authorNote": row["author_note"],
    }


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def _publish(conn, payload: dict, quadrant_history_row: dict) -> None:
    as_of = payload["asOf"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO macro_quadrant_history (
                as_of, growth_level, growth_momentum, inflation_level,
                inflation_momentum, quadrant, strength, phase
            ) VALUES (%(as_of)s, %(growth_level)s, %(growth_momentum)s, %(inflation_level)s,
                      %(inflation_momentum)s, %(quadrant)s, %(strength)s, %(phase)s)
            ON CONFLICT (as_of) DO UPDATE SET
                growth_level = EXCLUDED.growth_level,
                growth_momentum = EXCLUDED.growth_momentum,
                inflation_level = EXCLUDED.inflation_level,
                inflation_momentum = EXCLUDED.inflation_momentum,
                quadrant = EXCLUDED.quadrant,
                strength = EXCLUDED.strength,
                phase = EXCLUDED.phase
            """,
            quadrant_history_row,
        )
        cur.execute(
            """
            INSERT INTO macro_thesis_snapshots (as_of, quadrant, payload)
            VALUES (%(as_of)s, %(quadrant)s, %(payload)s)
            ON CONFLICT (as_of) DO UPDATE SET
                created_at = now(),
                quadrant = EXCLUDED.quadrant,
                payload = EXCLUDED.payload
            """,
            {"as_of": as_of, "quadrant": payload["quadrant"]["name"], "payload": Jsonb(payload)},
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is required; existing snapshot retained")

    print("[macro_thesis] fetching FRED series...")
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro")
    extra = get_fred_cached(EXTRA_FRED_SERIES, FRED_API_KEY, CACHE_DIR, cache_name="fred_macro_thesis")

    print("[macro_thesis] fetching BIS debt-service-ratio...")
    dsr = load_dsr_us()

    print("[macro_thesis] fetching full price history for {} tickers...".format(len(ALL_PRICE_TICKERS)))
    prices = _fetch_full_prices()
    print("[macro_thesis] prices shape:", prices.shape)
    if macro.empty or extra.empty or prices.empty:
        raise RuntimeError("Required macro or market inputs are empty; existing snapshot retained")

    house_view = None
    if not args.dry_run:
        with get_connection() as conn:
            _ensure_schema(conn)
            house_view = _load_latest_house_view(conn)

    payload = build_thesis_snapshot(macro, extra, prices, dsr, house_view=house_view)

    quadrant_row = {
        "as_of": payload["asOf"],
        "growth_level": payload["growth"]["level"],
        "growth_momentum": payload["growth"]["momentum"],
        "inflation_level": payload["inflation"]["level"],
        "inflation_momentum": payload["inflation"]["momentum"],
        "quadrant": payload["quadrant"]["name"],
        "strength": payload["quadrant"]["quadrantStrength"],
        "phase": payload["cycleMap"]["shortTerm"]["phase"],
    }

    if args.dry_run:
        print(json.dumps({"asOf": payload["asOf"], "quadrant": quadrant_row}, indent=2, default=str))
        return

    with get_connection() as conn:
        _publish(conn, payload, quadrant_row)

    DISK_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = DISK_SNAPSHOT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, default=str), encoding="utf-8")
    temporary.replace(DISK_SNAPSHOT)

    print("[macro_thesis] published {} (quadrant={})".format(payload["asOf"], payload["quadrant"]["name"]))


if __name__ == "__main__":
    main()
