"""
jobs/options_flow_backfill.py  --  historical DAILY Options Flow reconstruction

Rebuilds one derived snapshot per (ticker, trading day) so history (sentiment,
delta imbalance, IV, skew, DTE flow, IV percentiles) exists immediately instead
of accumulating for months. Runs on the same Python >= 3.12 worker image as the
live job and writes the same tables, tagged source = 'historical_backfill'.

    python jobs/options_flow_backfill.py --tickers SPY,QQQ,SMH,IWM,TLT,GLD --start 2026-06-01 --end 2026-09-20
    python jobs/options_flow_backfill.py --tickers SPY --days 60
    python jobs/options_flow_backfill.py --days 60 --resume          # default staged universe, skip done dates
    python jobs/options_flow_backfill.py --ticker SPY --days 5 --dry-run
    python jobs/options_flow_backfill.py --tickers SPY --days 60 --overwrite
    python jobs/options_flow_backfill.py --tickers all --days 60     # full universe, only after phase 1 works
    python jobs/options_flow_backfill.py --status                    # read-only progress report; see below

Design rules
  * SAME MATH AS LIVE. Each ticker-date goes through jobs.options_flow_refresh.process_ticker
    -> api.services.options_flow_metrics (classification, Greek join, sentiment, delta, IV,
    skew, DTE buckets, iv_history_stats). This file contains no analytics of its own.
  * POINT IN TIME. Date t uses only: that day's Theta data (dated contract list, trade_quote,
    Greeks, open interest for t) and prior daily IV observations from the sessions strictly
    before t (store.build_iv_history). Never today's expiration list, never later IV.
  * ONE COMMIT PER TICKER-DAY (run + snapshot in a single transaction), ticker-major and
    chronological, so a kill at day 37 loses nothing: rerun with --resume.
  * NOTHING IS MANUFACTURED. No trade data -> the day is NOT published (recorded as a failed
    audit run with a reason). Missing Greeks / OI -> published as 'partial' with null fields
    and explicit missingReasons; 0 is only ever a measured zero.
  * LIVE DATA IS UNTOUCHABLE. --overwrite deletes only rows with source = 'historical_backfill'.
    Sessions that are not fully finished (incl. the 16:00-16:15 options tail) are never backfilled.
  * Raw frames are discarded after each expiration; only derived metrics are stored.
  * SINGLETON PROCESS. A Postgres session-level advisory lock (api.services.options_flow_lock,
    key "macro_engine_options_flow_backfill") is acquired before anything else touches Postgres
    or ThetaData -- including --dry-run, which still opens a real ThetaData session. A second
    concurrent invocation refuses immediately. The lock is held on its own dedicated connection
    and releases automatically (Postgres-side) the instant that connection closes or drops, for
    any reason including the process dying -- nothing to clean up on crash.
  * FATAL VS RECOVERABLE. A normal per-ticker-day failure (bad/missing data for that one day)
    is recorded and the run continues. An infrastructure-level failure -- invalid/duplicate
    ThetaData session, authentication/API-key rejection, or a database that has gone unreachable
    (api.services.options_flow_errors.classify_error) -- stops the ENTIRE process immediately: no
    more ticker-days, no falling through to another --mode. Re-run with --resume.

Exit codes: 0 every requested ticker-day published (or already present); 1 nothing published,
refused (lock already held / scope guard), or a plain unexpected error; 2 some published, some
failed; 3 run-level fatal infrastructure error (see above); 130 interrupted (Ctrl-C).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_heartbeat as heartbeat  # noqa: E402
from api.services import options_flow_lock as lock  # noqa: E402
from api.services import options_flow_store as store  # noqa: E402
from api.services.options_flow_calendar import (  # noqa: E402
    last_completed_session,
    session_end,
    sessions_before,
    sessions_between,
)
from api.services.options_flow_config import load_config, load_universe, ticker_groups  # noqa: E402
from api.services.options_flow_errors import FatalBackfillError, classify_error  # noqa: E402
from api.services.options_flow_metrics import (  # noqa: E402
    METHODOLOGY_VERSION,
    StageTimer,
    _NullTimer,
    build_iv_observation,
    fmt_money,
    latest_iv_snapshot,
    normalize_greeks,
)
from api.services.options_flow_phase1_scope import (  # noqa: E402
    PHASE1_FULL_FLOW_RANGE,
    PHASE1_TICKERS,
    PHASE1_WARMUP_RANGE,
)
from api.services.options_flow_status import (  # noqa: E402
    format_status,
    median_runtime_by_ticker,
    mode_progress,
    remaining_by_ticker,
    total_eta_seconds,
)
from jobs.options_flow_refresh import ThetaFetcher, log, process_ticker, redact  # noqa: E402

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import pandas as pd  # noqa: E402

_NULL_TIMER = _NullTimer()

SOURCE = store.SOURCE_BACKFILL
MODE_FULL_FLOW = store.MODE_FULL_FLOW
MODE_IV_WARMUP = store.MODE_IV_WARMUP
DEFAULT_DAYS = 60
# Guard rails: the full universe over a long window must be an explicit decision.
MAX_STAGED_TICKERS = len(PHASE1_TICKERS)
MAX_STAGED_DAYS = 126
MAX_TICKER_DAYS = 1500
MIN_GREEK_MATCH = 0.80          # below this share of eligible premium with a matched Greek -> 'partial'
DAILY_EXPIRY_TICKERS = {"SPY", "QQQ", "IWM"}   # list an expiration every trading day
# Stages timed per ticker-day for the runtime QA report (jobs.options_flow_refresh.process_ticker
# and options_flow_metrics.process_expiration record into the same StageTimer).
PROFILE_STAGES = ["contract_lookup", "trade_quote_fetch", "greeks_fetch", "open_interest_fetch",
                  "normalize", "classify", "greek_asof_join", "iv_snapshot", "aggregate", "postgres_write"]


class ScopeError(ValueError):
    pass


class HistoricalFetcher(ThetaFetcher):
    """Live fetcher with point-in-time universe rules switched on (see ThetaFetcher.point_in_time)."""
    point_in_time = True


# --------------------------------------------------------------------------
# planning (pure)
# --------------------------------------------------------------------------

def resolve_tickers(args: argparse.Namespace, universe: Dict[str, List[str]]) -> List[str]:
    groups = ticker_groups(universe)
    if args.tickers and args.ticker:
        raise ScopeError("use --tickers or --ticker, not both")
    raw = args.tickers or args.ticker
    if raw is None:
        tickers = [t for t in PHASE1_TICKERS if t in groups]
    elif raw.strip().lower() == "all":
        tickers = [t for ts in universe.values() for t in ts]
    else:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    unknown = [t for t in tickers if t not in groups]
    if unknown:
        raise ScopeError("not in OPTIONS_FLOW_UNIVERSE: {}".format(", ".join(unknown)))
    seen: set = set()
    return [t for t in tickers if not (t in seen or seen.add(t))]   # de-dupe, keep order


def resolve_dates(args: argparse.Namespace, now: Optional[datetime] = None) -> Tuple[List[date], List[str]]:
    """Trading sessions to build, ascending, never later than the last completed session."""
    notes: List[str] = []
    last = last_completed_session(now)
    if args.days is not None and args.start:
        raise ScopeError("use --days or --start/--end, not both")
    if args.days is not None and args.days < 1:
        raise ScopeError("--days must be >= 1")

    end = date.fromisoformat(args.end) if args.end else last
    if end > last:
        notes.append("end {} is not a completed session; clamped to {}".format(end, last))
        end = last

    if args.start:
        start = date.fromisoformat(args.start)
        dates = list(sessions_between(start, end))
    else:
        n = args.days if args.days is not None else DEFAULT_DAYS
        if args.days is None:
            notes.append("no --days/--start given; defaulting to the last {} sessions".format(DEFAULT_DAYS))
        dates = sorted(sessions_before(end + timedelta(days=1), n))
    if not dates:
        raise ScopeError("no trading sessions in the requested range")
    return dates, notes


def check_scope(n_tickers: int, n_dates: int, allow_large: bool) -> None:
    if allow_large:
        return
    if n_tickers > MAX_STAGED_TICKERS and n_dates > MAX_STAGED_DAYS:
        raise ScopeError(
            "{} tickers x {} sessions is more than a first backfill should be. Do the staged universe "
            "(<= {} tickers) or <= {} sessions first, or pass --allow-large.".format(
                n_tickers, n_dates, MAX_STAGED_TICKERS, MAX_STAGED_DAYS))
    if n_tickers * n_dates > MAX_TICKER_DAYS:
        raise ScopeError("{} ticker-days exceeds the {} guard; pass --allow-large to proceed.".format(
            n_tickers * n_dates, MAX_TICKER_DAYS))


def expected_expirations(ticker: str, max_dte: int) -> int:
    """Expirations that trade inside max_dte (only used for the request estimate). The daily-expiry
    figure is MEASURED: a 5-day SPY dry run at max_dte=60 made 170 requests = ~34/day = ~16
    expirations. Others are an unmeasured guess (weekly + monthly cycle)."""
    if ticker in DAILY_EXPIRY_TICKERS:
        return max(1, round(16 * max_dte / 60))
    return max_dte // 7 + 3


def estimate_requests(ticker: str, max_dte: int, n_dates: int, mode: str = MODE_FULL_FLOW) -> int:
    """full_flow: 1 contract list + 1 open-interest + 2 per expiration (trade_quote + Greeks).
    iv_warmup: 1 contract list + 1 per expiration (Greeks only -- no trade_quote, no open interest)."""
    exps = expected_expirations(ticker, max_dte)
    if mode == MODE_IV_WARMUP:
        return n_dates * (exps + 1)
    return n_dates * (2 * exps + 2)


# --------------------------------------------------------------------------
# quality assessment (pure)
# --------------------------------------------------------------------------

def assess_payload(payload: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    ('failed'|'partial'|'success', reasons). 'failed' means there is no real data for the
    day and it must not be published; 'partial' is published with the reasons attached.
    """
    q = payload["quality"]
    if not q.get("trades"):
        return "failed", ["no_trade_data"]

    reasons: List[str] = []
    if any(str(w).startswith("open interest") for w in q.get("warnings") or []):
        reasons.append("open_interest_unavailable")
    if q["eligibleTrades"] == 0:
        reasons.append("no_eligible_trades")
    if not payload["iv"]["termStructure"] and payload["delta"]["matchedTrades"] == 0:
        reasons.append("greeks_unavailable")
    else:
        if q["eligibleTrades"] and (q.get("deltaMatchedPctPremium") or 0.0) < MIN_GREEK_MATCH:
            reasons.append("low_greek_match({:.0%})".format(q.get("deltaMatchedPctPremium") or 0.0))
        if payload["iv"]["atm"] is None:
            reasons.append("atm_iv_unavailable")
    return ("partial" if reasons else "success"), reasons


def assess_iv_warmup_payload(obs: Dict[str, Any]) -> Tuple[str, List[str]]:
    """('failed'|'partial'|'success', reasons) for a lightweight IV-only observation."""
    if obs.get("contracts", 0) == 0:
        return "failed", ["no_greek_data"]
    reasons: List[str] = []
    if obs.get("atm") is None:
        reasons.append("atm_iv_unavailable")
    if obs.get("skew25d30d") is None:
        reasons.append("skew_unavailable")
    return ("partial" if reasons else "success"), reasons


def wrap_iv_warmup_payload(ticker: str, group: str, market_date: date, obs: Dict[str, Any],
                           fetch_meta: Dict[str, Any], quality_extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same top-level envelope as a full-flow payload (options_flow_metrics.build_symbol_payload), so
    store.restat_backfill / the API's payload readers work unmodified regardless of mode. Every field
    a full-flow day would compute from trade data is explicitly None here -- never a manufactured 0/zero.
    """
    return {
        "source": SOURCE, "mode": MODE_IV_WARMUP, "methodologyVersion": METHODOLOGY_VERSION,
        "ticker": ticker, "group": group,
        "marketDate": market_date.isoformat(), "asOf": obs.get("asOf"), "spot": obs.get("spot"),
        "sentiment": None, "premium": None, "delta": None, "dte": None, "aggression": None, "openInterest": None,
        "iv": {k: v for k, v in obs.items() if k not in ("spot", "asOf", "expirations", "contracts")},
        "quality": {
            "mode": MODE_IV_WARMUP, "trades": None, "eligibleTrades": None, "expirations": obs.get("expirations"),
            "contracts": obs.get("contracts"), "thetaRequests": fetch_meta.get("thetaRequests"),
            "thetaFetchedAt": fetch_meta.get("thetaFetchedAt"), "fetchSeconds": fetch_meta.get("fetchSeconds"),
            "warnings": fetch_meta.get("warnings", []), **quality_extra,
        },
        "intraday": [], "largeTrades": [],
    }


def process_ticker_iv_warmup(
    fetcher: Any, ticker: str, market_date: date, cfg: Dict[str, Any], timer: Optional[StageTimer] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch ONLY first-order Greeks (no trade_quote, no open interest) and return the latest-IV snapshot
    frame -- the caller turns that into an observation with options_flow_metrics.build_iv_observation.
    Reuses the same contract-lookup call as full-flow (never requested twice).
    """
    import time as _time
    t0 = _time.time()
    req0 = getattr(fetcher, "requests", 0)
    tm = timer if timer is not None else _NULL_TIMER

    with tm.stage("contract_lookup"):
        exps = fetcher.expirations(ticker, market_date, int(cfg["maxDte"]))
    if not exps:
        raise RuntimeError("no expirations within {} DTE for {} on {}".format(cfg["maxDte"], ticker, market_date))

    def one(exp):
        with tm.stage("greeks_fetch"):
            raw = fetcher.greeks(ticker, exp, market_date)
        with tm.stage("normalize"):
            g = normalize_greeks(raw, exp, market_date)
        with tm.stage("iv_snapshot"):
            return latest_iv_snapshot(g)

    with ThreadPoolExecutor(max_workers=int(cfg["thetaConcurrency"])) as pool:
        frames = list(pool.map(one, exps))
    frames = [f for f in frames if f is not None and len(f)]
    snap = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    meta = {
        "expirations": len(exps), "thetaRequests": getattr(fetcher, "requests", 0) - req0,
        "fetchSeconds": round(_time.time() - t0, 1), "warnings": [],
        "thetaFetchedAt": datetime.now(timezone.utc).isoformat(),
    }
    return snap, meta


def failure_reason(e: Exception) -> str:
    msg = redact(str(e))
    if msg.startswith("no expirations"):
        return "no_expirations_listed"
    return "error: {}: {}".format(type(e).__name__, msg)[:300]


def build_diagnostics(ticker: str, d: date, status: str, reasons: List[str], payload: Optional[Dict[str, Any]],
                      meta: Optional[Dict[str, Any]], runtime: float, error: Optional[str] = None,
                      mode: str = MODE_FULL_FLOW, stage_seconds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if mode not in (MODE_FULL_FLOW, MODE_IV_WARMUP):
        raise ValueError("bad mode {!r}".format(mode))
    diag: Dict[str, Any] = {"ticker": ticker, "marketDate": d.isoformat(), "source": SOURCE, "mode": mode,
                            "status": status, "missingReasons": reasons, "runtimeSec": round(runtime, 2),
                            "error": error, "stageSeconds": stage_seconds or {}}
    if payload is None:
        if meta:
            diag["thetaRequests"] = meta.get("thetaRequests")
        return diag
    q = payload["quality"]
    if mode == MODE_IV_WARMUP:
        diag.update({"expirations": q.get("expirations"), "contracts": q.get("contracts"),
                    "atmIv": (payload.get("iv") or {}).get("atm"), "skew": (payload.get("iv") or {}).get("skew25d30d"),
                    "thetaRequests": q.get("thetaRequests"), "warnings": q.get("warnings")})
        return diag
    diag.update({
        "tradesDownloaded": q["trades"], "eligibleTrades": q["eligibleTrades"],
        "classifiedPctTrades": q["classifiedPctTrades"], "classifiedPctPremium": q["classifiedPctPremium"],
        "greekMatchedPctPremium": q.get("deltaMatchedPctPremium"), "oiMatchedPctPremium": q.get("oiMatchedPctPremium"),
        "contractsTraded": q.get("contractsTraded"), "grossPremium": payload["premium"]["gross"],
        "expirations": q.get("expirations"), "thetaRequests": q.get("thetaRequests"), "warnings": q.get("warnings"),
    })
    return diag


def progress_line(ticker: str, d: date, payload: Dict[str, Any], runtime: float, status: str, reasons: List[str]) -> str:
    q, s, dl = payload["quality"], payload["sentiment"], payload["delta"]
    sent = "n/a" if s["value"] is None else "{:+.2f}".format(s["value"])
    line = "[{} {}] trades={:,} coverage={:.1f}% sentiment={} delta={} {:.1f}s".format(
        ticker, d, q["trades"], 100.0 * (q["classifiedPctPremium"] or 0.0), sent, fmt_money(dl["netDollar"]), runtime)
    return line + ("  PARTIAL[{}]".format(", ".join(reasons)) if status == "partial" else "")


def progress_line_iv_warmup(ticker: str, d: date, payload: Dict[str, Any], runtime: float,
                            status: str, reasons: List[str]) -> str:
    iv = payload["iv"]
    atm = "n/a" if iv["atm"] is None else "{:.1%}".format(iv["atm"])
    skew = "n/a" if iv["skew25d30d"] is None else "{:+.1%}".format(iv["skew25d30d"])
    line = "[{} {}] IV-WARMUP atm={} skew={} {} expirations {:.1f}s".format(
        ticker, d, atm, skew, payload["quality"]["expirations"], runtime)
    return line + ("  PARTIAL[{}]".format(", ".join(reasons)) if status == "partial" else "")


def _median(vals: List[float]) -> Optional[float]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2.0


def db_size_bytes(conn) -> Optional[int]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_database_size(current_database()) AS n")
            return int(cur.fetchone()["n"])
    except Exception:
        return None


class QaReport:
    """Accumulates the item-9 QA numbers across a run without changing any published data."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.stage_samples: Dict[str, List[float]] = {s: [] for s in PROFILE_STAGES}
        self.requests_per_day: List[int] = []
        self.classified_pct: List[float] = []
        self.greek_matched_pct: List[float] = []
        self.oi_matched_pct: List[float] = []
        self.missing_iv = self.missing_skew = self.missing_percentile = 0
        self.n_success = 0

    def record(self, payload: Dict[str, Any], meta: Optional[Dict[str, Any]], stage_seconds: Dict[str, float]) -> None:
        for stage, secs in stage_seconds.items():
            if stage in self.stage_samples:
                self.stage_samples[stage].append(secs)
        if meta and meta.get("thetaRequests") is not None:
            self.requests_per_day.append(meta["thetaRequests"])
        if payload is None:
            return
        iv = payload.get("iv") or {}
        self.n_success += 1
        if iv.get("atm") is None:
            self.missing_iv += 1
        if iv.get("skew25d30d") is None:
            self.missing_skew += 1
        if all(v is None for v in (iv.get("percentiles") or {}).values()):
            self.missing_percentile += 1
        if self.mode == MODE_FULL_FLOW:
            q = payload["quality"]
            for lst, key in ((self.classified_pct, "classifiedPctPremium"), (self.greek_matched_pct, "deltaMatchedPctPremium"),
                             (self.oi_matched_pct, "oiMatchedPctPremium")):
                if q.get(key) is not None:
                    lst.append(q[key])

    def print_report(self, stats: Dict[str, int], total: int, wall_sec: float, db_bytes_before: Optional[int],
                     db_bytes_after: Optional[int]) -> None:
        log("")
        log("=== QA report ({}) ===".format(self.mode))
        log("  ticker-days: {} successful, {} partial, {} failed, {} of {} total".format(
            stats["success"], stats["partial"], stats["failed"], stats["success"] + stats["partial"] + stats["failed"], total))
        log("  wall-clock runtime: {:.0f}s ({:.1f} min)".format(wall_sec, wall_sec / 60.0))
        if self.requests_per_day:
            log("  ThetaData requests/ticker-day: median {:.0f}, mean {:.1f}, total {}".format(
                _median(self.requests_per_day), sum(self.requests_per_day) / len(self.requests_per_day),
                sum(self.requests_per_day)))
        if db_bytes_before is not None and db_bytes_after is not None:
            log("  Postgres size added: {} ({} -> {})".format(
                fmt_bytes(db_bytes_after - db_bytes_before), fmt_bytes(db_bytes_before), fmt_bytes(db_bytes_after)))
        log("  stage timing profile (median seconds per ticker-day, N samples):")
        for stage in PROFILE_STAGES:
            vals = self.stage_samples[stage]
            if vals:
                log("    {:<20s} median={:6.2f}s  n={}".format(stage, _median(vals), len(vals)))
        if self.mode == MODE_FULL_FLOW:
            for label, lst in (("classification coverage", self.classified_pct),
                               ("Greek-match coverage", self.greek_matched_pct),
                               ("OI-match coverage", self.oi_matched_pct)):
                if lst:
                    log("  {}: median {:.1%}, min {:.1%} ({} obs)".format(label, _median(lst), min(lst), len(lst)))
        n = max(1, self.n_success)
        log("  missing IV observations: {}/{} ({:.0%})".format(self.missing_iv, self.n_success, self.missing_iv / n))
        log("  missing skew observations: {}/{} ({:.0%})".format(self.missing_skew, self.n_success, self.missing_skew / n))
        log("  missing IV-percentile observations (all 4 windows null): {}/{} ({:.0%})".format(
            self.missing_percentile, self.n_success, self.missing_percentile / n))


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    a = abs(n)
    for div, suf in ((1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")):
        if a >= div:
            return "{}{:.1f}{}".format("+" if n >= 0 else "-", a / div, suf)
    return "{}{}B".format("+" if n >= 0 else "-", a)


def print_status(conn_factory: Optional[Callable[[], Any]] = None) -> int:
    """
    `--status`: read-only Phase 1 progress report. Never connects to ThetaData (no fetcher of
    any kind is constructed), never acquires the backfill lock -- it only ever probes it
    (acquire-then-immediately-release, see options_flow_lock.probe_active), so it can never
    block or be blocked by the real backfill.
    """
    if conn_factory is None:
        from api.db import get_connection as conn_factory  # noqa: N813
    try:
        conn = conn_factory()
    except Exception as e:  # noqa: BLE001
        log("Could not reach Postgres: {}".format(redact(str(e))))
        return 1
    try:
        store.ensure_schema(conn)
        active = lock.probe_active(conn)
        hb = heartbeat.read_heartbeat(conn)

        tickers = PHASE1_TICKERS
        full_range, warm_range = PHASE1_FULL_FLOW_RANGE, PHASE1_WARMUP_RANGE
        full_pub = store.fetch_mode_published(conn, tickers, MODE_FULL_FLOW, *full_range)
        full_fail = store.reconcile_failed_days(
            store.fetch_mode_failed(conn, tickers, MODE_FULL_FLOW, *full_range), full_pub)
        warm_pub = store.fetch_mode_published(conn, tickers, MODE_IV_WARMUP, *warm_range)
        warm_fail = store.reconcile_failed_days(
            store.fetch_mode_failed(conn, tickers, MODE_IV_WARMUP, *warm_range), warm_pub)

        n_full = len(sessions_between(*full_range))
        n_warm = len(sessions_between(*warm_range))
        full_progress = mode_progress(n_full * len(tickers),
                                      sum(1 for r in full_pub if r["status"] == "success"),
                                      sum(1 for r in full_pub if r["status"] == "partial"), len(full_fail))
        warm_progress = mode_progress(n_warm * len(tickers),
                                      sum(1 for r in warm_pub if r["status"] == "success"),
                                      sum(1 for r in warm_pub if r["status"] == "partial"), len(warm_fail))

        full_medians = median_runtime_by_ticker(
            [{"ticker": r["ticker"], "runtimeSec": r["diagnostics"].get("runtimeSec")}
             for r in full_pub if r["status"] == "success"])
        warm_medians = median_runtime_by_ticker(
            [{"ticker": r["ticker"], "runtimeSec": r["diagnostics"].get("runtimeSec")}
             for r in warm_pub if r["status"] == "success"])
        full_eta = total_eta_seconds(full_medians, remaining_by_ticker(tickers, n_full, full_pub))
        warm_eta = total_eta_seconds(warm_medians, remaining_by_ticker(tickers, n_warm, warm_pub))

        log(format_status(
            active, hb, METHODOLOGY_VERSION, full_progress, warm_progress,
            full_eta, full_medians.get("__overall__"), warm_eta, warm_medians.get("__overall__")))
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Backfill historical daily Options Flow snapshots from ThetaData.")
    ap.add_argument("--tickers", default=None, help="comma-separated, or 'all' for the full universe "
                                                    "(default: staged phase-1 {})".format(",".join(PHASE1_TICKERS)))
    ap.add_argument("--ticker", default=None, help="single ticker (alias for --tickers X)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD, first session")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD, last session (default: last completed session)")
    ap.add_argument("--days", type=int, default=None, help="last N trading sessions ending at --end (default 60)")
    ap.add_argument("--resume", action="store_true", help="skip ticker-days already published")
    ap.add_argument("--retry-partial", action="store_true", help="with --resume: also rebuild days published as partial")
    ap.add_argument("--overwrite", action="store_true", help="rebuild ticker-days that already have a backfill snapshot")
    ap.add_argument("--dry-run", action="store_true", help="fetch and compute, print progress, write nothing")
    ap.add_argument("--max-dte", type=int, default=None, help="override OPTIONS_FLOW_MAX_DTE (default 60)")
    ap.add_argument("--allow-large", action="store_true", help="bypass the staged-rollout size guards")
    ap.add_argument("--mode", choices=["full-flow", "iv-warmup"], default="full-flow",
                    help="full-flow (default): full trade tape + Greeks + OI. "
                         "iv-warmup: Greeks only (ATM/7D/30D/60D IV, term spreads, 25d skew) -- "
                         "no trade_quote, no open interest -- for populating IV percentile history "
                         "cheaply ahead of a research period (see docs/OPTIONS-FLOW-BACKFILL.md)")
    ap.add_argument("--status", action="store_true",
                    help="print Phase 1 progress (lock state, heartbeat, counts, ETA) and exit. "
                         "Read-only: never connects to ThetaData, never acquires the backfill lock.")
    return ap


def run(argv: Optional[List[str]] = None, fetcher: Any = None,
        conn_factory: Optional[Callable[[], Any]] = None, now: Optional[datetime] = None,
        lock_conn_factory: Optional[Callable[[], Any]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        return print_status(conn_factory)
    mode = MODE_IV_WARMUP if args.mode == "iv-warmup" else MODE_FULL_FLOW
    cfg = load_config()
    if args.max_dte:
        cfg["maxDte"] = args.max_dte
    universe = load_universe()
    groups = ticker_groups(universe)

    try:
        if args.resume and args.overwrite:
            raise ScopeError("--resume and --overwrite are mutually exclusive")
        tickers = resolve_tickers(args, universe)
        dates, notes = resolve_dates(args, now)
        check_scope(len(tickers), len(dates), args.allow_large)
    except ScopeError as e:
        log("ERROR: {}".format(e))
        return 1

    total = len(tickers) * len(dates)
    log("Options flow BACKFILL ({}, mode={}){}".format(SOURCE, mode, " [DRY RUN]" if args.dry_run else ""))
    for n in notes:
        log("  note: {}".format(n))
    log("  tickers : {} ({})".format(", ".join(tickers), len(tickers)))
    log("  sessions: {} -> {} ({})".format(dates[0], dates[-1], len(dates)))
    log("  ticker-days: {}   max_dte={} strike_range={}".format(total, cfg["maxDte"], cfg["strikeRange"]))
    est = sum(estimate_requests(t, int(cfg["maxDte"]), len(dates), mode) for t in tickers)
    per_day_desc = ("1 x expirations + 1" if mode == MODE_IV_WARMUP else "2 x expirations + 2")
    log("  expected ThetaData requests ~{:,} (per day: {}; before retries; nothing is cached)".format(est, per_day_desc))

    started = time.time()
    conn = None
    lock_conn = None
    stats = {"success": 0, "partial": 0, "failed": 0, "skipped": 0}
    partial_days: List[str] = []
    failed_days: List[str] = []
    done = 0
    qa = QaReport(mode)
    db_before = db_after = None
    run_process_id = heartbeat.new_run_process_id()
    run_started_at = datetime.now(timezone.utc)
    try:
        # Singleton-process lock, on its OWN connection, acquired before anything else touches
        # Postgres or ThetaData -- including for --dry-run, which still opens a real ThetaData
        # session. A second concurrent invocation must refuse here, not after doing any work.
        if lock_conn_factory is None:
            from api.db import get_connection as lock_conn_factory  # noqa: N813
        lock_conn = lock_conn_factory()
        store.ensure_schema(lock_conn)
        if not lock.try_acquire(lock_conn):
            log("Another options-flow backfill is already active.")
            log("Refusing to start a second ThetaData session.")
            return 1

        existing: Dict[Tuple[str, date], str] = {}
        if not args.dry_run:
            if conn_factory is None:
                from api.db import get_connection as conn_factory  # noqa: N813
            conn = conn_factory()
            db_before = db_size_bytes(conn)
            existing = store.backfill_existing(conn, tickers, dates[0], dates[-1], mode=mode)
            if existing and not (args.resume or args.overwrite):
                log("ERROR: {} ticker-days in range already have a backfill snapshot. "
                    "Use --resume to skip them or --overwrite to rebuild them.".format(len(existing)))
                return 1

        skip = set()
        if args.resume:
            skip = {k for k, st in existing.items() if st == "success" or (st == "partial" and not args.retry_partial)}
        done = len(skip)

        all_todo = sum(1 for t in tickers for d in dates if (t, d) not in skip)
        if all_todo == 0:
            log("Nothing to do: every requested ticker-day already has a published backfill snapshot.")
        # Lazy: a fully-satisfied --resume must not require ThetaData connectivity at all.
        fetcher = fetcher if fetcher is not None else (HistoricalFetcher(cfg) if all_todo else None)
        run_cfg_base = {**cfg, "source": SOURCE, "mode": mode}

        for t in tickers:
            series: Dict[date, float] = store.load_atm_series(conn, t) if conn is not None else {}
            todo = [d for d in dates if (t, d) not in skip]
            if len(todo) < len(dates):
                log("[{}] resuming: {} of {} sessions already published".format(t, len(dates) - len(todo), len(dates)))
            first_written: Optional[date] = None

            for d in todo:                                   # chronological: prior IV is always built first
                if conn is not None:
                    try:
                        heartbeat.upsert_heartbeat(conn, run_process_id, mode, t, d, METHODOLOGY_VERSION, run_started_at)
                    except Exception as e:  # noqa: BLE001 - heartbeat is observability only, never fatal by itself
                        log("WARNING: heartbeat update failed: {}".format(redact(str(e))))
                t0 = time.time()
                tm = StageTimer()
                payload = meta = None
                try:
                    history = store.build_iv_history(series, d)
                    if mode == MODE_IV_WARMUP:
                        snap, meta = process_ticker_iv_warmup(fetcher, t, d, cfg, timer=tm)
                        with tm.stage("aggregate"):
                            obs = build_iv_observation(snap, cfg, history)
                        payload = wrap_iv_warmup_payload(t, groups[t], d, obs, meta, {})
                        status, reasons = assess_iv_warmup_payload(obs)
                    else:
                        payload, meta = process_ticker(fetcher, t, groups[t], d, cfg, history,
                                                       allow_missing_iv=True, source=SOURCE, verbose=False, timer=tm)
                        status, reasons = assess_payload(payload)
                    error = None
                except Exception as e:  # noqa: BLE001 - classified below; only non-fatal ones are per-ticker-day failures
                    category = classify_error(e)
                    if category is not None:
                        # RUN-LEVEL FATAL: this is not about today's data -- stop the whole process
                        # rather than burn through the rest of the ticker-days hitting the same wall,
                        # and do NOT fall through to another phase. Caught by the outer handler below.
                        raise FatalBackfillError(category, e) from e
                    status, reasons, error = "failed", [failure_reason(e)], redact(traceback.format_exc())
                runtime = time.time() - t0

                if status == "failed":
                    diag = build_diagnostics(t, d, "failed", reasons, payload, meta, runtime, error,
                                             mode=mode, stage_seconds=tm.snapshot())
                    if conn is not None:
                        with tm.stage("postgres_write"):
                            store.record_backfill_failure(conn, t, d, diag, {**run_cfg_base, "ticker": t}, mode=mode)
                    stats["failed"] += 1
                    failed_days.append("{} {}: {}".format(t, d, reasons[0]))
                    log("[{} {}] FAILED {} ({:.1f}s) -- nothing published for this day".format(t, d, reasons[0], runtime))
                    qa.record(None, meta, tm.snapshot())
                else:
                    payload["quality"]["dataStatus"] = "complete" if status == "success" else "partial"
                    payload["quality"]["missingReasons"] = reasons
                    diag = build_diagnostics(t, d, status, reasons, payload, meta, runtime,
                                             mode=mode, stage_seconds=tm.snapshot())
                    as_of = _as_of(payload, d)
                    if conn is not None:
                        try:
                            with tm.stage("postgres_write"):
                                store.publish_backfill(
                                    conn, t, groups[t], d, status, as_of, diag, payload,
                                    {**run_cfg_base, "ticker": t},
                                    overwrite=args.overwrite or (args.retry_partial and existing.get((t, d)) == "partial"),
                                    mode=mode)
                        except store.DuplicateBackfillError:
                            stats["skipped"] += 1
                            done += 1
                            log("[{} {}] already has a backfill snapshot; left as is".format(t, d))
                            continue
                    stats[status] += 1
                    if status == "partial":
                        partial_days.append("{} {}: {}".format(t, d, ", ".join(reasons)))
                    if payload["iv"]["atm"] is not None:
                        series[d] = payload["iv"]["atm"]
                    first_written = d if first_written is None else min(first_written, d)
                    line = progress_line_iv_warmup if mode == MODE_IV_WARMUP else progress_line
                    log(line(t, d, payload, runtime, status, reasons))
                    qa.record(payload, meta, tm.snapshot())
                done += 1
                log("  {}/{} ticker-days complete".format(done, total))

            if conn is not None and first_written is not None:
                n = store.restat_backfill(conn, t, first_written, cfg)
                if n:
                    log("[{}] re-stated rolling IV stats on {} snapshot(s) (earlier days changed their history)".format(t, n))

        if conn is not None:
            db_after = db_size_bytes(conn)

    except FatalBackfillError as e:
        # Infrastructure-level failure (invalid/duplicate ThetaData session, auth/API-key
        # rejection, database unavailable, ...): stop the entire process. Deliberately does
        # NOT continue to the next ticker-day or fall through to another phase -- that
        # fallthrough is exactly what turned one bad connection into two concurrent ThetaData
        # sessions before. --resume picks this ticker-day back up once restarted.
        log("RUN-LEVEL FATAL [{}]: {}".format(e.category, redact(str(e.original))))
        log("Stopping the entire process now. Re-run with --resume once the underlying problem is fixed.")
        return 3
    except KeyboardInterrupt:
        log("\nInterrupted at {}/{} ticker-days. Everything committed so far is safe; re-run with --resume.".format(done, total))
        return 130
    except Exception as e:
        log("FATAL {}".format(redact("{}: {}".format(type(e).__name__, e))))
        log(redact(traceback.format_exc()))
        return 1
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if lock_conn is not None:
            try:
                lock.release(lock_conn)
            except Exception:
                pass
            try:
                lock_conn.close()
            except Exception:
                pass

    published = stats["success"] + stats["partial"]
    log("")
    log("Backfill {}: {} published ({} complete, {} partial), {} failed, {} already present, {} ticker-days, {:.0f}s, {} ThetaData requests".format(
        "dry run" if args.dry_run else "finished", published, stats["success"], stats["partial"], stats["failed"],
        len(skip) + stats["skipped"], total, time.time() - started, getattr(fetcher, "requests", "?")))
    if args.dry_run:
        log("Dry run: nothing was written to Postgres.")
    if partial_days:
        log("PARTIAL days (published, fields null where unavailable; rebuild with --resume --retry-partial or --overwrite):")
        for x in partial_days:
            log("  " + x)
    if failed_days:
        log("FAILED days (not published; re-run with --resume to retry):")
        for x in failed_days:
            log("  " + x)
    if qa.n_success or qa.stage_samples:
        qa.print_report(stats, total, time.time() - started, db_before, db_after)
    if stats["failed"]:
        return 2 if published else 1
    return 0


def _as_of(payload: Dict[str, Any], d: date) -> datetime:
    """Timestamp of the day's last trade; the session close if there is none."""
    if payload.get("asOf"):
        return datetime.fromisoformat(payload["asOf"])
    end = session_end(d)
    return end if end is not None else datetime(d.year, d.month, d.day, 21, 0, tzinfo=timezone.utc)


if __name__ == "__main__":
    sys.exit(run())
