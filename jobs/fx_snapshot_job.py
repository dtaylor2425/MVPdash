"""
jobs/fx_snapshot_job.py  (spec sections 5 & 8)

Ingest FRED + Frankfurter, score the 10 G10 currencies, derive 45 pairs, and
publish one dated JSON snapshot. The API routes only read snapshots — they
never call FRED or Frankfurter.

Recommended Railway cron (weekly, after ECB fixing): 0 6 * * 1

    python jobs/fx_snapshot_job.py                 # validate, build, publish
    python jobs/fx_snapshot_job.py --dry-run       # build + print, no writes
    python jobs/fx_snapshot_job.py --no-db         # disk snapshot only
    python jobs/fx_snapshot_job.py --skip-validation

Failure policy:
  * partial data  -> publish with dataIncomplete flags set
  * >3 of 10 incomplete -> abort, do not publish, exit 3
  * total source failure -> do not publish; previous snapshot keeps serving
    (the API marks it stale by age)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR  # noqa: E402
from src.fx.snapshot import build_bundle, build_fx_snapshot, snapshot_is_publishable  # noqa: E402

DISK_SNAPSHOT = Path(CACHE_DIR) / "fx_snapshot_latest.json"

DDL = (ROOT / "sql" / "002_fx_snapshots.sql").read_text(encoding="utf-8")

UPSERT = """
INSERT INTO fx_snapshots (
    as_of, observation_date, stale, incomplete_count, dropped_components,
    usd_score, reconciliation, series_health, payload
) VALUES (
    %(as_of)s, %(observation_date)s, FALSE, %(incomplete_count)s, %(dropped_components)s,
    %(usd_score)s, %(reconciliation)s::jsonb, %(series_health)s::jsonb, %(payload)s::jsonb
)
ON CONFLICT (as_of) DO UPDATE SET
    observation_date  = EXCLUDED.observation_date,
    created_at        = now(),
    stale             = FALSE,
    incomplete_count  = EXCLUDED.incomplete_count,
    dropped_components = EXCLUDED.dropped_components,
    usd_score         = EXCLUDED.usd_score,
    reconciliation    = EXCLUDED.reconciliation,
    series_health     = EXCLUDED.series_health,
    payload           = EXCLUDED.payload;
"""


def _database_url():
    import os
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def _run_validation(skip: bool) -> dict:
    if skip:
        return {"skipped": True}
    try:
        from src.fx.fred_client import health_summary, validate_series
        checks = validate_series()
        summary = health_summary(checks)
        for item in summary["failing"]:
            print(f"[series] FAIL  {item['logical']:<22} {item['seriesId']:<20} {item['detail']}")
        for item in summary["stale"]:
            print(f"[series] STALE {item['logical']:<22} {item['seriesId']:<20} {item['detail']}")
        print(f"[series] {summary['counts']}")
        return summary
    except Exception as exc:  # never let validation itself abort the job
        print(f"[series] validation could not run: {exc}")
        return {"error": str(exc)}


def _write_disk(payload: dict) -> None:
    DISK_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    DISK_SNAPSHOT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[disk] wrote {DISK_SNAPSHOT}")


def _write_db(payload: dict, series_health: dict) -> bool:
    url = _database_url()
    if not url:
        print("[db] DATABASE_URL not set — skipping Postgres write")
        return False
    import psycopg
    from psycopg.types.json import Jsonb

    usd = next((c["score"] for c in payload["currencies"] if c["code"] == "USD"), None)
    params = {
        "as_of": payload["asOf"],
        "observation_date": payload["observationDate"],
        "incomplete_count": len(payload["meta"]["incompleteCurrencies"]),
        "dropped_components": payload["meta"]["droppedComponents"],
        "usd_score": usd,
        "reconciliation": json.dumps(payload["reconciliation"]),
        "series_health": json.dumps(series_health),
        "payload": json.dumps(payload),
    }
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(UPSERT, params)
        conn.commit()
    print(f"[db] published fx_snapshots as_of={payload['asOf']}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-db", action="store_true", help="write disk snapshot only")
    ap.add_argument("--no-cache", action="store_true", help="force fresh FRED/Frankfurter fetch")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--history-months", type=int, default=27)
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    series_health = _run_validation(args.skip_validation)

    try:
        bundle = build_bundle(use_cache=not args.no_cache)
        payload = build_fx_snapshot(bundle=bundle, history_months=args.history_months)
    except Exception as exc:
        # Total source failure: do not publish. Previous snapshot keeps serving.
        print(f"[fx] BUILD FAILED — not publishing. Previous snapshot remains live. {exc}")
        return 2

    payload["meta"]["seriesHealth"] = series_health
    payload["meta"]["jobRanAt"] = started.isoformat()

    ok, why = snapshot_is_publishable(payload)
    incomplete = payload["meta"]["incompleteCurrencies"]
    dropped = payload["meta"]["droppedComponents"]

    # Cross-section size per component, every run (fix-list item 4) -- the
    # minimum-coverage guard (scoring.MIN_CURRENCIES_PER_COMPONENT) drops a
    # component below 6, but a component sitting at 6-8 is worth watching
    # even when it isn't dropped.
    availability = payload["meta"]["componentAvailability"]
    for name, ccys in availability.items():
        flag = " *** DROPPED (below minimum)" if name in dropped else ""
        print(f"[fx] cross-section {name:<14} {len(ccys)}/10 currencies{flag}")

    print(json.dumps({
        "asOf": payload["asOf"],
        "observationDate": payload["observationDate"],
        "usdScore": next((c["score"] for c in payload["currencies"] if c["code"] == "USD"), None),
        "ranking": [(c["code"], c["score"]) for c in payload["currencies"] if c.get("scored")],
        "incompleteCurrencies": incomplete,
        "droppedComponents": dropped,
        "componentCrossSectionSizes": {k: len(v) for k, v in availability.items()},
        "pairCount": len(payload["pairs"]),
        "reconciliation": payload["reconciliation"]["agreement"],
        "publishable": ok,
    }, indent=2))

    if not ok:
        print(f"[fx] ABORT — {why}")
        return 3

    if args.dry_run:
        print("[fx] dry-run — no writes")
        return 0

    _write_disk(payload)
    if not args.no_db:
        _write_db(payload, series_health)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
