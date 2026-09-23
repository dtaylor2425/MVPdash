"""
scripts/archive_preexisting_iv_warmup.py

Signal Validation v3, pre-flight step: before converting the pre-existing iv_warmup rows in the
v3 validation+warmup date range (2025-08-21 -> 2026-06-25, all 6 Phase 1 tickers) into full_flow
rows via `--overwrite`, archive the existing options_flow_runs + options_flow_symbol_snapshots
rows for those dates to a committed JSON file. The Postgres schema has exactly one physical row
slot per (ticker, market_date) regardless of mode (see options_flow_store.publish_backfill's
docstring) -- overwriting an iv_warmup row with a full_flow row deletes the old run+snapshot row.
No IV/skew/percentile VALUE is lost (full_flow calls the exact same iv_block()/iv_history_stats()
as iv_warmup, given the same underlying Greek observations), but the old row's audit metadata
(run id, created_at, diagnostics) is. This archive preserves that metadata durably before it's
gone, per the user's explicit "archive then overwrite" decision.

Read-only against Postgres (SELECT only, no writes, no deletes).

    python scripts/archive_preexisting_iv_warmup.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "reports" / "options-flow-v3-preexisting-iv-warmup-archive.json"

TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
RANGE_START = date(2025, 8, 21)   # v3 warmup-window start (20 sessions before validation)
RANGE_END = date(2026, 6, 25)     # v3 validation-window end (session immediately before discovery)


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    out = subprocess.run(
        ["railway", "variables", "--service", "Postgres", "--kv"],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("DATABASE_PUBLIC_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Could not resolve DATABASE_URL from env or `railway variables`.")


def main() -> int:
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(_get_database_url(), row_factory=dict_row)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT r.id AS run_id, r.market_date, r.as_of_timestamp, r.status AS run_status,
                   r.config, r.diagnostics, r.created_at AS run_created_at, r.finished_at,
                   r.source AS run_source, r.mode AS run_mode, r.methodology_version,
                   s.id AS snapshot_id, s.ticker, s.group_name, s.sentiment, s.sentiment_label,
                   s.atm_iv, s.payload, s.created_at AS snapshot_created_at
            FROM options_flow_runs r
            JOIN options_flow_symbol_snapshots s ON s.run_id = r.id
            WHERE r.source = 'historical_backfill' AND s.source = 'historical_backfill'
              AND r.mode = 'iv_warmup' AND s.ticker = ANY(%(tickers)s)
              AND r.market_date BETWEEN %(start)s AND %(end)s
            ORDER BY s.ticker, r.market_date
            """,
            {"tickers": TICKERS, "start": RANGE_START, "end": RANGE_END},
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Found {len(rows)} pre-existing iv_warmup run+snapshot rows in "
         f"{RANGE_START} -> {RANGE_END} for {TICKERS} (expected up to {len(TICKERS) * 212}).")

    by_ticker = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], 0)
        by_ticker[r["ticker"]] += 1
    print("By ticker:", by_ticker)

    archive = {
        "archived_at": "2026-09-23",
        "reason": "Signal Validation v3 converts these (ticker, market_date) slots from "
                 "mode=iv_warmup to mode=full_flow via --overwrite (the schema has one physical "
                 "row per (ticker, market_date), not one per mode) -- this file preserves the "
                 "pre-overwrite run+snapshot content. No IV/skew/percentile VALUE is lost by the "
                 "overwrite itself (full_flow recomputes them identically from the same Greek "
                 "observations via the same iv_block()/iv_history_stats() code); this archive "
                 "additionally preserves the exact pre-overwrite audit metadata and payload.",
        "range": {"start": str(RANGE_START), "end": str(RANGE_END)}, "tickers": TICKERS,
        "row_count": len(rows), "row_count_by_ticker": by_ticker,
        "rows": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(archive, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
