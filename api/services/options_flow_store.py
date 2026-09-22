"""
api/services/options_flow_store.py

Postgres access for Macro Options Flow, shared by the worker (writes) and the
private API (reads). Selection rules live here so both sides agree:

    * "published" = run status success | partial. Failed / running runs are
      never served, so a failed Theta job cannot erase the last good board.
    * "latest" is chosen PER TICKER (newest market_date, then as_of, then
      created_at) across published runs. A ticker missing from the newest
      run therefore keeps its previous snapshot, flagged carriedForward.
    * Payload columns that are big (intraday series, large-trade table) are
      stripped in SQL for summary reads.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg.types.json import Jsonb

from api.services.options_flow_calendar import compute_data_status, sessions_before
from api.services.options_flow_metrics import METHODOLOGY_VERSION, _clean, iv_history_stats

ROOT = Path(__file__).resolve().parents[2]
DDL_PATHS = [ROOT / "sql" / "005_options_flow.sql", ROOT / "sql" / "006_options_flow_backfill.sql",
            ROOT / "sql" / "007_options_flow_iv_warmup.sql", ROOT / "sql" / "008_options_flow_methodology_version.sql",
            ROOT / "sql" / "009_options_flow_control_plane.sql"]
SOURCE_LIVE = "live"
SOURCE_BACKFILL = "historical_backfill"
MODE_FULL_FLOW = "full_flow"
MODE_IV_WARMUP = "iv_warmup"

PUBLISHED = ("success", "partial")
HEAVY_KEYS = ["intraday", "largeTrades"]
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

_LATEST_IDS_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (s.ticker) s.id
    FROM options_flow_symbol_snapshots s
    JOIN options_flow_runs r ON r.id = s.run_id
    WHERE r.status IN ('success', 'partial')
    ORDER BY s.ticker, s.market_date DESC, (s.source = 'live') DESC, s.as_of_timestamp DESC, s.created_at DESC
)
"""

LATEST_SUMMARY_SQL = _LATEST_IDS_SQL + """
SELECT s.ticker, s.group_name, s.market_date, s.as_of_timestamp, s.created_at,
       s.run_id, (s.payload - %(heavy)s::text[]) AS payload
FROM latest l
JOIN options_flow_symbol_snapshots s ON s.id = l.id
"""

LATEST_TICKER_SQL = """
SELECT s.ticker, s.group_name, s.market_date, s.as_of_timestamp, s.created_at,
       s.run_id, s.payload
FROM options_flow_symbol_snapshots s
JOIN options_flow_runs r ON r.id = s.run_id
WHERE r.status IN ('success', 'partial') AND s.ticker = %(ticker)s
ORDER BY s.market_date DESC, (s.source = 'live') DESC, s.as_of_timestamp DESC, s.created_at DESC
LIMIT 1
"""

LATEST_RUN_SQL = """
SELECT id, market_date, as_of_timestamp, status, created_at, finished_at, diagnostics
FROM options_flow_runs
WHERE status IN ('success', 'partial')
ORDER BY market_date DESC, (source = 'live') DESC, created_at DESC
LIMIT 1
"""

# The live worker's own last run -- thousands of backfill runs must never shadow it.
LAST_RUN_ANY_SQL = """
SELECT id, market_date, as_of_timestamp, status, created_at, finished_at, diagnostics
FROM options_flow_runs
WHERE source = 'live'
ORDER BY created_at DESC
LIMIT 1
"""

HISTORY_SQL = """
SELECT market_date, as_of_timestamp, payload FROM (
    SELECT DISTINCT ON (s.market_date) s.market_date, s.as_of_timestamp,
           (s.payload - %(heavy)s::text[]) AS payload
    FROM options_flow_symbol_snapshots s
    JOIN options_flow_runs r ON r.id = s.run_id
    WHERE r.status IN ('success', 'partial') AND s.ticker = %(ticker)s
    ORDER BY s.market_date DESC, s.as_of_timestamp DESC, s.created_at DESC
    LIMIT %(days)s
) t
ORDER BY market_date ASC
"""

IV_HISTORY_SQL = """
SELECT market_date, atm_iv FROM (
    SELECT DISTINCT ON (s.market_date) s.market_date, s.atm_iv
    FROM options_flow_symbol_snapshots s
    JOIN options_flow_runs r ON r.id = s.run_id
    WHERE r.status IN ('success', 'partial') AND s.ticker = %(ticker)s
      AND s.market_date < %(before)s AND s.atm_iv IS NOT NULL
    ORDER BY s.market_date DESC, s.as_of_timestamp DESC, s.created_at DESC
    LIMIT %(n)s
) t
ORDER BY market_date DESC
"""


# --------------------------------------------------------------------------
# writes (worker)
# --------------------------------------------------------------------------

def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        for path in DDL_PATHS:
            cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()


def create_run(conn, market_date: date, config: Dict[str, Any], source: str = SOURCE_LIVE) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO options_flow_runs (market_date, as_of_timestamp, status, config, source, methodology_version)
            VALUES (%s, now(), 'running', %s, %s, %s) RETURNING id
            """,
            (market_date, Jsonb(_clean(config)), source, METHODOLOGY_VERSION),
        )
        run_id = cur.fetchone()["id"]
    conn.commit()
    return str(run_id)


def finalize_run(
    conn, run_id: str, status: str, as_of: datetime,
    diagnostics: Dict[str, Any], snapshots: Sequence[Dict[str, Any]],
) -> None:
    """Insert snapshots and flip the run's status in ONE transaction: a run is
    either fully visible (with its snapshots) or not visible at all."""
    if status not in ("success", "partial", "failed"):
        raise ValueError("bad terminal status {!r}".format(status))
    with conn.cursor() as cur:
        for s in snapshots:
            p = s["payload"]
            cur.execute(
                """
                INSERT INTO options_flow_symbol_snapshots
                    (run_id, ticker, group_name, market_date, as_of_timestamp,
                     sentiment, sentiment_label, atm_iv, payload, source, methodology_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'live', %s)
                """,
                (run_id, s["ticker"], s["group"], date.fromisoformat(p["marketDate"]), s["as_of"],
                 (p.get("sentiment") or {}).get("value"),
                 (p.get("sentiment") or {}).get("label"),
                 (p.get("iv") or {}).get("atm"), Jsonb(p), p.get("methodologyVersion", METHODOLOGY_VERSION)),
            )
        cur.execute(
            """
            UPDATE options_flow_runs
               SET status = %s, as_of_timestamp = %s, diagnostics = %s, finished_at = now()
             WHERE id = %s
            """,
            (status, as_of, Jsonb(_clean(diagnostics)), run_id),
        )
    conn.commit()


def fail_run(conn, run_id: str, diagnostics: Dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE options_flow_runs SET status = 'failed', diagnostics = %s, finished_at = now() WHERE id = %s",
            (Jsonb(_clean(diagnostics)), run_id),
        )
    conn.commit()


def final_run_exists(conn, market_date: date, after: datetime) -> bool:
    """True if a published run for market_date was created at/after `after`
    (used to skip redundant runs once the post-close print exists)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM options_flow_runs
            WHERE market_date = %s AND status IN ('success', 'partial') AND created_at >= %s
            LIMIT 1
            """,
            (market_date, after),
        )
        return cur.fetchone() is not None


def load_iv_history(conn, ticker: str, before: date, n: int = 252) -> Dict[str, Any]:
    """Daily ATM IV history (one value per prior market date, newest first)."""
    with conn.cursor() as cur:
        cur.execute(IV_HISTORY_SQL, {"ticker": ticker, "before": before, "n": n})
        rows = cur.fetchall()
    return {"priorAtmIv": [(r["market_date"].isoformat(), r["atm_iv"]) for r in rows]}


def prune(conn, market_date: date, retention_days: int) -> Tuple[int, int]:
    """
    Keep every intraday run for `retention_days`; older than that keep only
    the last snapshot per (ticker, market_date), which is all the 1D/5D /
    IV-percentile / sentiment-history queries need. Returns
    (snapshots_deleted, runs_deleted).
    """
    cutoff = market_date - timedelta(days=retention_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM options_flow_symbol_snapshots
            WHERE market_date < %(cutoff)s
              AND id NOT IN (
                  SELECT DISTINCT ON (ticker, market_date) id
                  FROM options_flow_symbol_snapshots
                  ORDER BY ticker, market_date, as_of_timestamp DESC, created_at DESC
              )
            """,
            {"cutoff": cutoff},
        )
        snaps = cur.rowcount
        cur.execute(
            """
            DELETE FROM options_flow_runs r
            WHERE r.market_date < %(cutoff)s AND r.status <> 'running'
              AND NOT EXISTS (SELECT 1 FROM options_flow_symbol_snapshots s WHERE s.run_id = r.id)
            """,
            {"cutoff": cutoff},
        )
        runs = cur.rowcount
    conn.commit()
    return snaps, runs


# --------------------------------------------------------------------------
# historical backfill (jobs/options_flow_backfill.py)
# --------------------------------------------------------------------------

class DuplicateBackfillError(Exception):
    """A backfill snapshot for this (ticker, market_date) already exists and overwrite was not requested."""


def backfill_existing(conn, tickers: Sequence[str], start: date, end: date) -> Dict[Tuple[str, date], str]:
    """{(ticker, market_date): 'success' | 'partial'} for published backfill snapshots in range."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.ticker, s.market_date, r.status
            FROM options_flow_symbol_snapshots s
            JOIN options_flow_runs r ON r.id = s.run_id
            WHERE s.source = 'historical_backfill' AND r.source = 'historical_backfill'
              AND r.status IN ('success', 'partial')
              AND s.ticker = ANY(%s) AND s.market_date BETWEEN %s AND %s
            """,
            (list(tickers), start, end),
        )
        return {(r["ticker"], r["market_date"]): r["status"] for r in cur.fetchall()}


def publish_backfill(
    conn, ticker: str, group: str, market_date: date, status: str, as_of: datetime,
    diagnostics: Dict[str, Any], payload: Dict[str, Any], config: Dict[str, Any], overwrite: bool = False,
    mode: str = MODE_FULL_FLOW,
) -> str:
    """
    Atomically publish ONE ticker-date: (optionally) drop the existing backfill
    row for it, insert a run + snapshot, commit. Either all of that happens or
    none -- a failure leaves the previous snapshot exactly as it was. Only rows
    with source = 'historical_backfill' can ever be deleted here; live rows are
    untouchable by construction. `mode` is 'full_flow' (trade tape + Greeks) or
    'iv_warmup' (Greeks only, see options_flow_metrics.build_iv_observation) --
    both share the same unique (ticker, market_date) backfill slot, so a later
    full_flow run for a warmed-up date needs overwrite=True, same as any rebuild.
    """
    if status not in ("success", "partial"):
        raise ValueError("bad backfill status {!r}".format(status))
    if mode not in (MODE_FULL_FLOW, MODE_IV_WARMUP):
        raise ValueError("bad mode {!r}".format(mode))
    try:
        with conn.cursor() as cur:
            if overwrite:
                cur.execute(
                    """
                    DELETE FROM options_flow_runs r
                    USING options_flow_symbol_snapshots s
                    WHERE s.run_id = r.id AND r.source = 'historical_backfill'
                      AND s.source = 'historical_backfill' AND s.ticker = %s AND s.market_date = %s
                    """,
                    (ticker, market_date),
                )
            cur.execute(
                """
                INSERT INTO options_flow_runs
                    (market_date, as_of_timestamp, status, config, diagnostics, source, mode,
                     methodology_version, finished_at)
                VALUES (%s, %s, %s, %s, %s, 'historical_backfill', %s, %s, now()) RETURNING id
                """,
                (market_date, as_of, status, Jsonb(_clean(config)), Jsonb(_clean(diagnostics)), mode,
                 payload.get("methodologyVersion", METHODOLOGY_VERSION)),
            )
            run_id = cur.fetchone()["id"]
            cur.execute(
                """
                INSERT INTO options_flow_symbol_snapshots
                    (run_id, ticker, group_name, market_date, as_of_timestamp,
                     sentiment, sentiment_label, atm_iv, payload, source, mode, methodology_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'historical_backfill', %s, %s)
                """,
                (run_id, ticker, group, market_date, as_of,
                 (payload.get("sentiment") or {}).get("value"),
                 (payload.get("sentiment") or {}).get("label"),
                 (payload.get("iv") or {}).get("atm"), Jsonb(payload), mode,
                 payload.get("methodologyVersion", METHODOLOGY_VERSION)),
            )
        conn.commit()
        return str(run_id)
    except Exception as e:
        conn.rollback()
        if e.__class__.__name__ == "UniqueViolation":
            raise DuplicateBackfillError("{} {} already has a backfill snapshot".format(ticker, market_date)) from e
        raise


def record_backfill_failure(
    conn, ticker: str, market_date: date, diagnostics: Dict[str, Any], config: Dict[str, Any],
    mode: str = MODE_FULL_FLOW,
) -> str:
    """Audit row for a ticker-date that could not be built. Publishes nothing, deletes nothing."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO options_flow_runs
                (market_date, as_of_timestamp, status, config, diagnostics, source, mode,
                 methodology_version, finished_at)
            VALUES (%s, now(), 'failed', %s, %s, 'historical_backfill', %s, %s, now()) RETURNING id
            """,
            (market_date, Jsonb(_clean(config)), Jsonb(_clean(diagnostics)), mode, METHODOLOGY_VERSION),
        )
        run_id = cur.fetchone()["id"]
    conn.commit()
    return str(run_id)


def load_atm_series(conn, ticker: str) -> Dict[date, float]:
    """One daily ATM IV observation per market date (all published snapshots, live or
    backfill; the fullest day -- latest as_of -- wins). Deliberately includes dates after
    any given snapshot: point-in-time filtering is build_iv_history's job."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (s.market_date) s.market_date, s.atm_iv
            FROM options_flow_symbol_snapshots s
            JOIN options_flow_runs r ON r.id = s.run_id
            WHERE r.status IN ('success', 'partial') AND s.ticker = %s AND s.atm_iv IS NOT NULL
            ORDER BY s.market_date DESC, s.as_of_timestamp DESC, s.created_at DESC
            """,
            (ticker,),
        )
        return {r["market_date"]: r["atm_iv"] for r in cur.fetchall()}


def build_iv_history(series: Dict[date, float], market_date: date, max_sessions: int = 252) -> Dict[str, Any]:
    """
    Point-in-time IV history for `market_date`: observations from the
    `max_sessions` trading sessions STRICTLY BEFORE it. Built by walking the
    calendar backwards from market_date, so a later observation cannot enter no
    matter what is in `series`.
    """
    sessions = sessions_before(market_date, max_sessions)
    return {
        "priorAtmIv": [(d.isoformat(), series[d]) for d in sessions if series.get(d) is not None],
        "sessionsBack": [d.isoformat() for d in sessions],
    }


def restat_backfill(conn, ticker: str, from_date: date, cfg: Dict[str, Any]) -> int:
    """
    Recompute the rolling IV fields (1D/5D/20D change, percentiles) of this ticker's
    BACKFILL snapshots dated >= from_date from the stored ATM-IV series, using
    only observations before each row's own date. Needed because gap-filling or
    re-running an earlier date changes what later dates' history contains.
    Touches payload->'iv' of source = 'historical_backfill' rows only. Returns rows updated.
    """
    series = load_atm_series(conn, ticker)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, market_date, atm_iv, payload->'iv' AS iv
            FROM options_flow_symbol_snapshots
            WHERE source = 'historical_backfill' AND ticker = %s AND market_date >= %s
            ORDER BY market_date
            """,
            (ticker, from_date),
        )
        rows = cur.fetchall()
    updated = 0
    with conn.cursor() as cur:
        for r in rows:
            iv = r["iv"] if isinstance(r["iv"], dict) else json.loads(r["iv"] or "{}")
            stats = _clean(iv_history_stats(r["atm_iv"], build_iv_history(series, r["market_date"]), cfg))
            new_iv = {**iv, **stats}
            if new_iv != iv:
                cur.execute(
                    "UPDATE options_flow_symbol_snapshots SET payload = jsonb_set(payload, '{iv}', %s) WHERE id = %s",
                    (Jsonb(new_iv), r["id"]),
                )
                updated += 1
    conn.commit()
    return updated


# --------------------------------------------------------------------------
# reads (API)
# --------------------------------------------------------------------------

def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    p = row["payload"]
    return json.loads(p) if isinstance(p, str) else p


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _run_meta(run: Optional[Dict[str, Any]], cfg: Dict[str, Any], now: Optional[datetime]) -> Optional[Dict[str, Any]]:
    if run is None:
        return None
    ds = compute_data_status(
        run["market_date"], run["created_at"], now,
        int(cfg.get("liveMaxAgeMin", 20)), int(cfg.get("staleAfterMin", 60)),
    )
    return {
        "id": str(run["id"]),
        "marketDate": run["market_date"].isoformat(),
        "asOf": _iso(run["as_of_timestamp"]),
        "publishedAt": _iso(run["finished_at"] or run["created_at"]),
        "status": run["status"],
        "dataStatus": ds["status"],
        "dataStatusReason": ds["reason"],
        "ageMinutes": ds["ageMinutes"],
    }


def assemble_latest(
    rows: List[Dict[str, Any]], run: Optional[Dict[str, Any]],
    universe: Dict[str, List[str]], cfg: Dict[str, Any], now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Pure: latest-per-ticker rows + newest published run -> API response.
    Tickers outside the configured universe are dropped; tickers inside it
    but never published are listed in `missing`. A ticker whose snapshot
    comes from an older run than `run` is flagged carriedForward.
    """
    order = [(g, t) for g, ts in universe.items() for t in ts]
    by_ticker = {r["ticker"]: r for r in rows}
    run_id = str(run["id"]) if run else None

    tickers, missing = [], []
    for group, t in order:
        r = by_ticker.get(t)
        if r is None:
            missing.append(t)
            continue
        tickers.append({
            **_payload(r),
            "ticker": t,
            "group": group,
            "asOf": _iso(r["as_of_timestamp"]),
            "publishedAt": _iso(r["created_at"]),
            "carriedForward": run_id is not None and str(r["run_id"]) != run_id,
        })
    return {
        "run": _run_meta(run, cfg, now),
        "groups": [{"name": g, "tickers": list(ts)} for g, ts in universe.items()],
        "tickers": tickers,
        "missing": missing,
    }


def assemble_history(rows: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
    points = []
    for r in rows:
        p = _payload(r)
        iv, prem, delta, dte = p.get("iv") or {}, p.get("premium") or {}, p.get("delta") or {}, p.get("dte") or {}
        sent = p.get("sentiment") or {}
        points.append({
            "date": r["market_date"].isoformat(),
            "asOf": _iso(r["as_of_timestamp"]),
            "sentiment": sent.get("value"),
            "label": sent.get("label"),
            "netDirectionalPremium": prem.get("netDirectional"),
            "grossPremium": prem.get("gross"),
            "netDeltaDollar": delta.get("netDollar"),
            "deltaRatio": delta.get("ratio"),
            "atmIv": iv.get("atm"),
            "iv7d": iv.get("iv7d"),
            "iv30d": iv.get("iv30d"),
            "iv60d": iv.get("iv60d"),
            "skew25d30d": iv.get("skew25d30d"),
            "ivPercentile": iv.get("percentile"),
            "zeroDteShare": dte.get("zeroDteShare"),
        })
    return {"ticker": ticker, "days": len(points), "series": points}


def strip_heavy(payload: Dict[str, Any], include_trades: bool) -> Dict[str, Any]:
    if include_trades:
        return payload
    return {k: v for k, v in payload.items() if k != "largeTrades"}


def fetch_latest(conn, universe: Dict[str, List[str]], cfg: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(LATEST_SUMMARY_SQL, {"heavy": HEAVY_KEYS})
        rows = cur.fetchall()
        cur.execute(LATEST_RUN_SQL)
        run = cur.fetchone()
    return assemble_latest(rows, run, universe, cfg, now)


def fetch_ticker(
    conn, ticker: str, universe: Dict[str, List[str]], cfg: Dict[str, Any],
    include_trades: bool = True, now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(LATEST_TICKER_SQL, {"ticker": ticker})
        row = cur.fetchone()
        cur.execute(LATEST_RUN_SQL)
        run = cur.fetchone()
    if row is None:
        return None
    group = next((g for g, ts in universe.items() if ticker in ts), row["group_name"])
    return {
        "run": _run_meta(run, cfg, now),
        "ticker": {
            **strip_heavy(_payload(row), include_trades),
            "ticker": ticker,
            "group": group,
            "asOf": _iso(row["as_of_timestamp"]),
            "publishedAt": _iso(row["created_at"]),
            "carriedForward": run is not None and str(row["run_id"]) != str(run["id"]),
        },
    }


def fetch_history(conn, ticker: str, days: int) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(HISTORY_SQL, {"ticker": ticker, "days": days, "heavy": HEAVY_KEYS})
        rows = cur.fetchall()
    return assemble_history(rows, ticker)


def fetch_status(conn, universe: Dict[str, List[str]], cfg: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(LATEST_RUN_SQL)
        published = cur.fetchone()
        cur.execute(LAST_RUN_ANY_SQL)
        last = cur.fetchone()

    def _diag(run):
        if run is None:
            return None
        d = run["diagnostics"]
        d = json.loads(d) if isinstance(d, str) else (d or {})
        return {"tickers": d.get("tickers"), "failed": d.get("failed"), "error": d.get("error"),
                "warnings": d.get("warnings"), "durationSec": d.get("durationSec")}

    return {
        "latestPublished": _run_meta(published, cfg, now),
        "lastRun": _run_meta(last, cfg, now),
        "lastRunDiagnostics": _diag(last),
        "universe": {g: list(ts) for g, ts in universe.items()},
    }


# --------------------------------------------------------------------------
# backfill progress queries -- shared by scripts/generate_options_flow_qa_report.py,
# scripts/log_options_flow_phase1_progress.py and jobs/options_flow_backfill.py --status,
# so "what counts as done/failed for a ticker-day" is defined in exactly one place.
# --------------------------------------------------------------------------

def _row_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    p = row["payload"]
    return json.loads(p) if isinstance(p, str) else p


def _row_diagnostics(row: Dict[str, Any]) -> Dict[str, Any]:
    d = row["diagnostics"]
    return json.loads(d) if isinstance(d, str) else (d or {})


def fetch_mode_published(conn, tickers: Sequence[str], mode: str, start: date, end: date) -> List[Dict[str, Any]]:
    """Published (success | partial) backfill rows for one mode within [start, end]."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.ticker, s.market_date, r.status, s.payload, r.diagnostics, s.created_at
            FROM options_flow_symbol_snapshots s
            JOIN options_flow_runs r ON r.id = s.run_id
            WHERE s.source = 'historical_backfill' AND s.mode = %(mode)s
              AND r.status IN ('success', 'partial')
              AND s.ticker = ANY(%(tickers)s) AND s.market_date BETWEEN %(start)s AND %(end)s
            ORDER BY s.ticker, s.market_date
            """,
            {"mode": mode, "tickers": list(tickers), "start": start, "end": end},
        )
        rows = cur.fetchall()
    return [{"ticker": r["ticker"], "marketDate": r["market_date"], "status": r["status"],
             "payload": _row_payload(r), "diagnostics": _row_diagnostics(r), "createdAt": r["created_at"]}
            for r in rows]


def fetch_mode_failed(conn, tickers: Sequence[str], mode: str, start: date, end: date) -> List[Dict[str, Any]]:
    """Latest FAILED run per ticker-day for one mode within [start, end] (config.ticker
    identifies the ticker-day; a failed row has no snapshot). Only the latest attempt per
    ticker-day is kept, matching what --resume would actually retry."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (r.config->>'ticker', r.market_date)
                   r.config->>'ticker' AS ticker, r.market_date, r.diagnostics, r.created_at
            FROM options_flow_runs r
            WHERE r.source = 'historical_backfill' AND r.mode = %(mode)s AND r.status = 'failed'
              AND r.config->>'ticker' = ANY(%(tickers)s) AND r.market_date BETWEEN %(start)s AND %(end)s
            ORDER BY r.config->>'ticker', r.market_date, r.created_at DESC
            """,
            {"mode": mode, "tickers": list(tickers), "start": start, "end": end},
        )
        rows = cur.fetchall()
    return [{"ticker": r["ticker"], "marketDate": r["market_date"], "diagnostics": _row_diagnostics(r),
             "createdAt": r["created_at"]} for r in rows]


def reconcile_failed_days(failed: List[Dict[str, Any]], published: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a 'failed' entry if that ticker-day has SINCE been published (a later --resume
    succeeded) -- it is not actually failed anymore."""
    done = {(r["ticker"], r["marketDate"]) for r in published}
    return [f for f in failed if (f["ticker"], f["marketDate"]) not in done]
