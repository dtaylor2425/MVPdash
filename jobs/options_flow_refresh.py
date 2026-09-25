"""
jobs/options_flow_refresh.py  --  theta-options-worker entrypoint

Fetches ThetaData Options STANDARD data, computes DERIVED analytics and
publishes a snapshot into Railway Postgres. The private API and the page only
ever read what this job wrote; the browser never touches ThetaData.

Runs on Python >= 3.12 (the `thetadata` package requires it) as its own
Railway service -- do NOT install it into the Python 3.11 API service. See
requirements-theta-worker.txt / nixpacks.theta-worker.toml.

    python jobs/options_flow_refresh.py                      # all ETFs, current trading date
    python jobs/options_flow_refresh.py --tickers SPY,QQQ
    python jobs/options_flow_refresh.py --date 2026-09-18
    python jobs/options_flow_refresh.py --force              # ignore "final print already published"
    python jobs/options_flow_refresh.py --dry-run            # fetch + compute + print, write nothing

Exit codes:  0 published cleanly (or nothing to do)
             1 run failed (nothing published) or unexpected error
             2 partial: published, but at least one ticker failed

Environment:
    THETADATA_API_KEY   required, read from the server environment only
    DATABASE_URL        required unless --dry-run
    OPTIONS_FLOW_*      see api/services/options_flow_config.py

Endpoints used (all included in Options Standard): option_list_expirations /
option_list_contracts, option_history_trade_quote (exclusive=True),
option_history_greeks_first_order, option_history_open_interest.
No trade-greeks (Pro) endpoints are called. At most 4 requests are ever in
flight (semaphore-enforced, not just pool-sized).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_local_env(path: Path = ROOT / ".env") -> List[str]:
    """
    Local-dev convenience: read THETADATA_* / DATABASE_* / OPTIONS_FLOW_* from the repo's
    gitignored .env when they are not already in the environment. Never overrides a real
    environment variable (so Railway/CI are unaffected) and ignores every other key.
    Returns the names it set (never their values).
    """
    if not path.is_file():
        return []
    set_names: List[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key.startswith(("THETADATA_", "DATABASE_", "OPTIONS_FLOW_")) and value and key not in os.environ:
            os.environ[key] = value
            set_names.append(key)
    return set_names


load_local_env()

from api.services import options_flow_store as store  # noqa: E402
from api.services.options_flow_calendar import (  # noqa: E402
    NY,
    session_end,
    target_trading_date,
)
from api.services.options_flow_config import load_config, load_universe, ticker_groups  # noqa: E402
from api.services.options_flow_lock import LOCK_KEY  # share vendor-session exclusion with backfill
from api.services.options_flow_metrics import (  # noqa: E402
    StageTimer,
    _NullTimer,
    _pick,
    build_symbol_payload,
    fmt_money,
    normalize_open_interest,
    process_expiration,
)

_NULL_TIMER = _NullTimer()

RETRYABLE_GRPC = {"UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "INTERNAL", "ABORTED"}
RETRY_BACKOFF_SEC = (1.0, 3.0, 8.0)


def log(msg: str) -> None:
    print(msg, flush=True)


def redact(text: str) -> str:
    """Scrub the Theta key from anything that might reach logs / Postgres."""
    key = os.getenv("THETADATA_API_KEY")
    return text.replace(key, "***") if key else text


# --------------------------------------------------------------------------
# ThetaData access
# --------------------------------------------------------------------------

class ThetaFetcher:
    """Thin, rate-limited wrapper over ThetaClient returning pandas frames."""

    # Historical backfill flips this (see jobs/options_flow_backfill.py). When True:
    #   * the contract universe for date t comes ONLY from the dated contract list for t,
    #     never from today's live expiration list;
    #   * max_dte is not sent to list/OI endpoints -- it is filtered client-side relative
    #     to t, because a server-side max_dte might be measured from *today*.
    point_in_time = False

    def __init__(self, cfg: Dict[str, Any], client: Any = None):
        self.cfg = cfg
        self._sem = threading.BoundedSemaphore(int(cfg["thetaConcurrency"]))
        self._lock = threading.Lock()
        self.requests = 0
        self.client = client if client is not None else self._connect()
        self._no_data_exc = self._load_no_data_exc()

    @staticmethod
    def _connect():
        key = os.getenv("THETADATA_API_KEY")
        if not key:
            # ThetaClient would silently fall back to ./creds.txt; we don't want that.
            raise RuntimeError("THETADATA_API_KEY is not set in the worker environment")
        try:
            from thetadata import ThetaClient
        except ImportError as e:
            raise RuntimeError(
                "The `thetadata` package is not installed (needs Python >= 3.12: "
                "pip install -r requirements-theta-worker.txt). Running on Python {}.{}".format(
                    *sys.version_info[:2])
            ) from e
        # The client logs its auth response (account e-mail, subscription tiers) at INFO.
        logging.getLogger("thetadata").setLevel(logging.WARNING)
        return ThetaClient(api_key=key, dataframe_type="pandas")

    @staticmethod
    def _load_no_data_exc():
        try:
            from thetadata.errors import NoDataFoundError
            return NoDataFoundError
        except Exception:
            return None

    def _call(self, label: str, fn: Callable[..., pd.DataFrame], **kwargs) -> pd.DataFrame:
        """One Theta request: semaphore-limited, retried on transient gRPC errors,
        NoDataFound -> empty frame. Anything else propagates."""
        last: Optional[Exception] = None
        for attempt in range(len(RETRY_BACKOFF_SEC) + 1):
            try:
                with self._sem:
                    with self._lock:
                        self.requests += 1
                    return fn(**kwargs)
            except Exception as e:  # noqa: BLE001 - classified below, re-raised if not retryable
                if self._no_data_exc is not None and isinstance(e, self._no_data_exc):
                    return pd.DataFrame()
                code = getattr(e, "code", None)
                code_name = code().name if callable(code) else None
                if code_name in RETRYABLE_GRPC and attempt < len(RETRY_BACKOFF_SEC):
                    last = e
                    time.sleep(RETRY_BACKOFF_SEC[attempt])
                    continue
                raise RuntimeError("{} failed: {}".format(label, redact(str(e)))) from e
        raise RuntimeError("{} failed after retries: {}".format(label, redact(str(last))))

    def expirations(self, symbol: str, market_date: date, max_dte: int) -> List[date]:
        today = datetime.now(timezone.utc).date()
        if not self.point_in_time and market_date >= today - timedelta(days=1):
            df = self._call("option_list_expirations({})".format(symbol),
                            self.client.option_list_expirations, symbol=symbol)
        else:  # past sessions: expirations that have since expired are only in the dated contract list
            df = self._call("option_list_contracts({}, {})".format(symbol, market_date),
                            self.client.option_list_contracts, request_type="trade",
                            date=market_date, symbol=symbol,
                            max_dte=None if self.point_in_time else max_dte)
        if df is None or df.empty:
            return []
        col = _pick(df, ["expiration"], "expiration list")
        exps = sorted({pd.Timestamp(x).date() for x in df[col].dropna().unique()})
        hi = market_date + timedelta(days=max_dte)
        return [e for e in exps if market_date <= e <= hi]

    def trade_quote(self, symbol: str, expiration: date, d: date) -> pd.DataFrame:
        return self._call(
            "option_history_trade_quote({}, {})".format(symbol, expiration),
            self.client.option_history_trade_quote,
            symbol=symbol, expiration=expiration, date=d, strike="*", right="both",
            start_time=self.cfg["sessionStart"], end_time=self.cfg["sessionEnd"],
            exclusive=True, strike_range=int(self.cfg["strikeRange"]),
        )

    def greeks(self, symbol: str, expiration: date, d: date) -> pd.DataFrame:
        # Wildcard expiration is not accepted by this endpoint's client signature -> one call per expiration.
        return self._call(
            "option_history_greeks_first_order({}, {})".format(symbol, expiration),
            self.client.option_history_greeks_first_order,
            symbol=symbol, expiration=expiration, interval=self.cfg["greekInterval"], date=d,
            strike="*", right="both", start_time=self.cfg["sessionStart"],
            end_time=self.cfg["sessionEnd"], strike_range=int(self.cfg["strikeRange"]),
        )

    def open_interest(self, symbol: str, d: date, expirations: Optional[List[date]] = None) -> pd.DataFrame:
        """One wildcard-expiration request for past dates. ThetaData rejects that for the CURRENT
        day ("Cannot fetch current-day data without specifying an expiration"), so for today
        (ET) each expiration is requested individually."""
        today_et = datetime.now(NY).date()
        if d >= today_et and expirations:
            frames = [self._oi_call(symbol, d, e) for e in expirations]
            frames = [f for f in frames if f is not None and len(f)]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._oi_call(symbol, d, "*")

    def _oi_call(self, symbol: str, d: date, expiration: Any) -> pd.DataFrame:
        return self._call(
            "option_history_open_interest({}, {})".format(symbol, expiration),
            self.client.option_history_open_interest,
            symbol=symbol, expiration=expiration, date=d, strike="*", right="both",
            max_dte=None if (self.point_in_time or expiration != "*") else int(self.cfg["maxDte"]),
            strike_range=int(self.cfg["strikeRange"]),
        )


# --------------------------------------------------------------------------
# per-ticker pipeline
# --------------------------------------------------------------------------

def process_ticker(
    fetcher: Any, ticker: str, group: str, market_date: date, cfg: Dict[str, Any],
    history: Optional[Dict[str, Any]],
    allow_missing_iv: bool = False, source: str = "live", verbose: bool = True,
    timer: Optional[StageTimer] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch -> classify -> aggregate one ETF. Returns (payload, run-diagnostics entry).

    Also the code path for historical backfill (jobs/options_flow_backfill.py), which
    passes source="historical_backfill" and allow_missing_iv=True so a date with trades
    but no usable Greeks yields nulls for delta/IV instead of failing. Neither option
    changes any calculation. `timer`, if given, records per-stage wall time (contract
    lookup, trade/Greek/OI fetch, normalize/classify/join, aggregate) for profiling --
    it is pure observation and never changes the result."""
    t0 = time.time()
    req0 = getattr(fetcher, "requests", 0)
    warnings: List[str] = []
    tm = timer if timer is not None else _NULL_TIMER

    with tm.stage("contract_lookup"):
        exps = fetcher.expirations(ticker, market_date, int(cfg["maxDte"]))
    if not exps:
        raise RuntimeError("no expirations within {} DTE for {} on {}".format(cfg["maxDte"], ticker, market_date))

    say = log if verbose else (lambda _msg: None)
    say("[{}] fetching trade_quote ({} expirations, <= {} DTE, exclusive=True)".format(
        ticker, len(exps), cfg["maxDte"]))
    say("[{}] fetching first order greeks ({} interval)".format(ticker, cfg["greekInterval"]))

    def one(exp: date):
        with tm.stage("trade_quote_fetch"):
            trade_raw = fetcher.trade_quote(ticker, exp, market_date)
        with tm.stage("greeks_fetch"):
            greeks_raw = fetcher.greeks(ticker, exp, market_date)
        return process_expiration(trade_raw, greeks_raw, exp, market_date, int(cfg["greekToleranceMin"]), timer=tm)

    workers = int(cfg["thetaConcurrency"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, exps))  # re-raises the first worker exception

    trade_frames = [t for t, _ in results if len(t)]
    iv_frames = [s for _, s in results if s is not None and len(s)]
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else results[0][0]
    iv_snap = pd.concat(iv_frames, ignore_index=True) if iv_frames else results[0][1]
    if iv_snap is None or len(iv_snap) == 0:
        if not allow_missing_iv:
            raise RuntimeError("no valid Greek/IV observations returned for {}".format(ticker))
        warnings.append("no valid Greek/IV observations returned")

    say("[{}] fetching open interest".format(ticker))
    oi = None
    try:
        with tm.stage("open_interest_fetch"):
            oi = normalize_open_interest(fetcher.open_interest(ticker, market_date, exps))
        if len(oi) == 0:
            warnings.append("open interest returned no rows")
    except Exception as e:  # OI is optional context ("where available"); recorded, not fatal
        warnings.append("open interest unavailable: {}".format(redact(str(e))))
        say("[{}] WARNING open interest unavailable: {}".format(ticker, redact(str(e))))

    fetched_at = datetime.now(timezone.utc)
    meta = {
        "expirations": len(exps),
        "thetaRequests": getattr(fetcher, "requests", 0) - req0,
        "fetchSeconds": round(time.time() - t0, 1),
        "warnings": warnings,
    }
    with tm.stage("aggregate"):
        payload = build_symbol_payload(
            ticker, group, market_date, trades, iv_snap, oi=oi, history=history, cfg=cfg,
            fetched_at=fetched_at, fetch_meta=meta, source=source,
        )
    return payload, meta


def log_ticker_summary(ticker: str, p: Dict[str, Any]) -> None:
    q, s, d, iv = p["quality"], p["sentiment"], p["delta"], p["iv"]
    log("[{}] classified {:.1f}% of trade premium ({} of {} trades eligible)".format(
        ticker, 100.0 * (q["classifiedPctPremium"] or 0.0), q["eligibleTrades"], q["trades"]))
    if s["value"] is None:
        log("[{}] sentiment n/a {}".format(ticker, s["label"]))
    else:
        note = " ({})".format("; ".join(s["reasons"])) if s["reasons"] else ""
        log("[{}] sentiment {:+.2f} {}{}".format(ticker, s["value"], s["label"], note))
    log("[{}] delta imbalance {} (ratio {})".format(
        ticker, fmt_money(d["netDollar"]),
        "n/a" if d["ratio"] is None else "{:+.2f}".format(d["ratio"])))
    log("[{}] ATM IV {} | 0DTE share {}".format(
        ticker,
        "n/a" if iv["atm"] is None else "{:.1%}".format(iv["atm"]),
        "n/a" if p["dte"]["zeroDteShare"] is None else "{:.0%}".format(p["dte"]["zeroDteShare"])))


# --------------------------------------------------------------------------
# run orchestration
# --------------------------------------------------------------------------

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


def run(argv: Optional[List[str]] = None, fetcher: Any = None, conn_factory: Optional[Callable[[], Any]] = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh Macro Options Flow snapshots from ThetaData.")
    ap.add_argument("--force", action="store_true", help="run even if the post-close print for the date is already published")
    ap.add_argument("--tickers", default=None, help="comma-separated subset, e.g. SPY,QQQ")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: current trading date)")
    ap.add_argument("--dry-run", action="store_true", help="fetch and compute but write nothing to Postgres")
    args = ap.parse_args(argv)

    cfg = load_config()
    universe = load_universe()
    groups = ticker_groups(universe)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        unknown = [t for t in tickers if t not in groups]
        if unknown:
            log("ERROR: not in OPTIONS_FLOW_UNIVERSE: {}".format(", ".join(unknown)))
            return 1
    else:
        tickers = [t for ts in universe.values() for t in ts]

    market_date = date.fromisoformat(args.date) if args.date else target_trading_date()
    started = time.time()
    log("Options flow refresh: {} tickers, market date {}{}".format(
        len(tickers), market_date, " (dry run)" if args.dry_run else ""))

    if conn_factory is None:
        from api.db import get_connection as conn_factory  # noqa: N813
    conn = None
    run_id: Optional[str] = None
    try:
        if not args.dry_run:
            conn = conn_factory()
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (LOCK_KEY,))
                if not cur.fetchone()["acquired"]:
                    log("Another options-flow collector/backfill is running; skipping overlap.")
                    return 0
            store.ensure_schema(conn)
            end = session_end(market_date)
            if not args.force and end is not None and store.final_run_exists(conn, market_date, end):
                log("Post-close snapshot for {} already published; nothing to do (use --force to re-run).".format(market_date))
                return 0
            run_id = store.create_run(conn, market_date, cfg)

        collection_started = datetime.now(timezone.utc)
        close = session_end(market_date)
        final_collection = close is not None and collection_started >= close + timedelta(minutes=15)
        fetcher = fetcher or ThetaFetcher(cfg)

        snapshots: List[Dict[str, Any]] = []
        per_ticker: Dict[str, Any] = {}
        failed: Dict[str, str] = {}
        all_warnings: List[str] = []

        for t in tickers:
            t_start = time.time()
            try:
                history = None
                if conn is not None:
                    history = store.load_iv_history(conn, t, market_date, int(cfg["ivHistoryDays"]))
                payload, meta = process_ticker(fetcher, t, groups[t], market_date, cfg, history)
                payload["publication"] = {
                    "state": "final" if final_collection else "partial",
                    "collectionBounds": {"start": cfg["sessionStart"], "end": cfg["sessionEnd"],
                                         "timezone": "America/New_York", "maxDte": cfg["maxDte"]},
                    "collectionStartedAt": collection_started.isoformat(),
                    "sessionEnd": close.isoformat() if close else None,
                    "meaning": "Completed collection window, not certainty of trade intent",
                }
                log_ticker_summary(t, payload)
                as_of = _parse_iso(payload["asOf"]) or datetime.now(timezone.utc)
                snapshots.append({"ticker": t, "group": groups[t], "as_of": as_of, "payload": payload})
                per_ticker[t] = {
                    "status": "ok", "seconds": round(time.time() - t_start, 1),
                    "sentiment": payload["sentiment"]["value"], "label": payload["sentiment"]["label"],
                    "classifiedPctPremium": payload["quality"]["classifiedPctPremium"],
                    "eligibleTrades": payload["quality"]["eligibleTrades"],
                    "thetaRequests": meta["thetaRequests"],
                }
                all_warnings += ["[{}] {}".format(t, w) for w in meta["warnings"]]
            except Exception as e:  # recorded + surfaced via exit code; run continues with other tickers
                msg = redact("{}: {}".format(type(e).__name__, e))
                failed[t] = msg
                per_ticker[t] = {"status": "failed", "seconds": round(time.time() - t_start, 1)}
                log("[{}] FAILED {}".format(t, msg))
                log(redact(traceback.format_exc()))

        status = "failed" if not snapshots else ("partial" if failed else "success")
        as_of_run = max((s["as_of"] for s in snapshots), default=datetime.now(timezone.utc))
        diagnostics = {
            "publicationState": "final" if final_collection and not failed and set(tickers) == set(groups) else "partial",
            "requestedTickers": tickers,
            "marketDate": market_date.isoformat(), "dryRun": bool(args.dry_run),
            "tickers": per_ticker, "failed": failed, "warnings": all_warnings,
            "durationSec": round(time.time() - started, 1),
            "thetaRequests": getattr(fetcher, "requests", None),
            "error": None if snapshots else "every ticker failed",
        }

        if args.dry_run:
            log("Dry run complete: {} computed, {} failed. Nothing written.".format(len(snapshots), len(failed)))
            return 1 if status == "failed" else (2 if status == "partial" else 0)

        if status == "failed":
            store.fail_run(conn, run_id, diagnostics)
            log("Options flow run {} FAILED: no ticker produced a snapshot. Previous snapshot left untouched.".format(run_id))
            return 1

        store.finalize_run(conn, run_id, status, as_of_run, diagnostics, snapshots)
        try:
            snaps, runs = store.prune(conn, market_date, int(cfg["intradayRetentionDays"]))
            if snaps or runs:
                log("Pruned {} old snapshots, {} old runs".format(snaps, runs))
        except Exception as e:  # housekeeping must never turn a good publish into a failure
            log("WARNING prune failed: {}".format(redact(str(e))))
        log("Published options flow run {}".format(run_id))
        if failed:
            log("PARTIAL: {} ticker(s) failed: {}".format(len(failed), ", ".join(sorted(failed))))
            return 2
        return 0

    except Exception as e:
        log("FATAL {}".format(redact("{}: {}".format(type(e).__name__, e))))
        log(redact(traceback.format_exc()))
        if conn is not None and run_id is not None:
            try:
                conn.rollback()
                store.fail_run(conn, run_id, {"error": redact(str(e)), "marketDate": market_date.isoformat()})
            except Exception as e2:
                log("could not mark run failed: {}".format(redact(str(e2))))
        return 1
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(run())
