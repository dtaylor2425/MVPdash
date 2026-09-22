"""
scripts/generate_options_flow_qa_report.py

Reads the published Macro Options Flow backfill snapshots from Postgres and writes:

    reports/options-flow-backfill-qa.md        per-ETF + combined QA report (items 5-8)
    reports/options-flow-backfill-qa.csv       the same per-ETF summary table, tabular
    reports/options-flow-backfill-failures.csv every failed ticker-day with its reason

Read-only: issues SELECTs only, never writes to Postgres, never touches
api/services/options_flow_metrics.py's calculations. Safe to run while a backfill is
in progress (the report will simply reflect partial progress) -- pass --require-complete
to instead refuse unless every requested ticker-day for both phases is accounted for
(success + partial + failed == requested), which is what "QA passes" should be gated on
before generating the final report / exporting the research dataset.

    python scripts/generate_options_flow_qa_report.py \
        --tickers SPY,QQQ,IWM,SMH,TLT,GLD \
        --full-flow-start 2026-06-26 --full-flow-end 2026-09-21 \
        --warmup-start 2025-06-25 --warmup-end 2026-06-25
    python scripts/generate_options_flow_qa_report.py --require-complete   # uses the defaults below
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_qa as qa  # noqa: E402
from api.services.options_flow_calendar import sessions_between  # noqa: E402
from api.services.options_flow_metrics import METHODOLOGY_VERSION  # noqa: E402

# Phase 1, as actually launched -- keep these in sync with the invocation used
# (jobs/options_flow_backfill.py --tickers ... --start ... --end ..., run once for
# full-flow and once for --mode iv-warmup). Overridable via CLI for a different scope.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
DEFAULT_FULL_FLOW = (date(2026, 6, 26), date(2026, 9, 21))
DEFAULT_WARMUP = (date(2025, 6, 25), date(2026, 6, 25))

REPORTS_DIR = ROOT / "reports"


# --------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------

def _payload(row) -> Dict[str, Any]:
    p = row["payload"]
    return json.loads(p) if isinstance(p, str) else p


def _diag(row) -> Dict[str, Any]:
    d = row["diagnostics"]
    d = json.loads(d) if isinstance(d, str) else (d or {})
    return d


def fetch_published(conn, tickers: List[str], mode: str, start: date, end: date) -> List[Dict[str, Any]]:
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
            {"mode": mode, "tickers": tickers, "start": start, "end": end},
        )
        rows = cur.fetchall()
    return [{"ticker": r["ticker"], "marketDate": r["market_date"], "status": r["status"],
             "payload": _payload(r), "diagnostics": _diag(r), "createdAt": r["created_at"]} for r in rows]


def fetch_failed(conn, tickers: List[str], mode: str, start: date, end: date) -> List[Dict[str, Any]]:
    """Failed runs are one-per-ticker-day (config.ticker) with no snapshot -- read from
    options_flow_runs directly. Only the LATEST failed attempt per ticker-day is kept,
    matching what --resume would actually retry."""
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
            {"mode": mode, "tickers": tickers, "start": start, "end": end},
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = _diag(r)
        # only count as still-failed if no later success/partial superseded it (resume overwrites in place,
        # so a ticker-day with both a failed AND a later published row is not actually failed anymore)
        out.append({"ticker": r["ticker"], "marketDate": r["market_date"], "diagnostics": d,
                    "createdAt": r["created_at"]})
    return out


def reconcile_failed(failed: List[Dict[str, Any]], published: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    done = {(r["ticker"], r["marketDate"]) for r in published}
    return [f for f in failed if (f["ticker"], f["marketDate"]) not in done]


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------

def build_universe_data(
    conn, tickers: List[str], full_range: Tuple[date, date], warmup_range: Tuple[date, date],
) -> Dict[str, Any]:
    full_sessions = sessions_between(*full_range)
    warmup_sessions = sessions_between(*warmup_range)
    per_ticker: Dict[str, Any] = {}
    for t in tickers:
        full_pub = fetch_published(conn, [t], "full_flow", *full_range)
        full_fail = reconcile_failed(fetch_failed(conn, [t], "full_flow", *full_range), full_pub)
        warm_pub = fetch_published(conn, [t], "iv_warmup", *warmup_range)
        warm_fail = reconcile_failed(fetch_failed(conn, [t], "iv_warmup", *warmup_range), warm_pub)
        per_ticker[t] = {
            "fullRaw": full_pub, "fullFailed": full_fail,
            "warmRaw": warm_pub, "warmFailed": warm_fail,
            "full": [qa.extract_full_flow_fields(r) for r in full_pub],
            "warm": [qa.extract_iv_warmup_fields(r) for r in warm_pub],
        }
    return {"perTicker": per_ticker, "fullSessions": full_sessions, "warmupSessions": warmup_sessions}


def check_complete(data: Dict[str, Any], tickers: List[str]) -> List[str]:
    problems = []
    req_full, req_warm = len(data["fullSessions"]), len(data["warmupSessions"])
    for t in tickers:
        d = data["perTicker"][t]
        got_full = len(d["full"]) + len(d["fullFailed"])
        got_warm = len(d["warm"]) + len(d["warmFailed"])
        if got_full < req_full:
            problems.append("{}: full-flow {} of {} ticker-days accounted for".format(t, got_full, req_full))
        if got_warm < req_warm:
            problems.append("{}: iv-warmup {} of {} ticker-days accounted for".format(t, got_warm, req_warm))
    return problems


def build_report(data: Dict[str, Any], tickers: List[str]) -> Dict[str, Any]:
    req_full, req_warm = len(data["fullSessions"]), len(data["warmupSessions"])
    summaries = []
    all_full: List[Dict[str, Any]] = []
    all_warm: List[Dict[str, Any]] = []
    by_ticker_full: Dict[str, List[Dict[str, Any]]] = {}
    for t in tickers:
        d = data["perTicker"][t]
        summaries.append(qa.summarize_ticker(t, req_full, d["full"], len(d["fullFailed"]),
                                             req_warm, d["warm"], len(d["warmFailed"])))
        all_full += d["full"]
        all_warm += d["warm"]
        by_ticker_full[t] = d["full"]
    combined = qa.summarize_ticker(
        "ALL", req_full * len(tickers), all_full, sum(len(data["perTicker"][t]["fullFailed"]) for t in tickers),
        req_warm * len(tickers), all_warm, sum(len(data["perTicker"][t]["warmFailed"]) for t in tickers))
    distribution = {t: qa.distribution_qa(data["perTicker"][t]["full"]) for t in tickers}
    reconciliation = {t: qa.reconciliation_errors(data["perTicker"][t]["full"]) for t in tickers}
    reconciliation["ALL"] = qa.reconciliation_errors(all_full)
    comparability = qa.cross_etf_comparability(by_ticker_full)
    comparable, comparability_note = qa.comparability_verdict(comparability)
    extreme = {t: qa.flag_extreme_observations(data["perTicker"][t]["full"], "netTradeSentiment") for t in tickers}
    failures = [{"ticker": t, **f, "mode": "full_flow"} for t in tickers for f in data["perTicker"][t]["fullFailed"]] + \
               [{"ticker": t, **f, "mode": "iv_warmup"} for t in tickers for f in data["perTicker"][t]["warmFailed"]]
    return {
        "summaries": summaries, "combined": combined, "distribution": distribution,
        "reconciliation": reconciliation, "comparability": comparability,
        "comparabilityVerdict": (comparable, comparability_note),
        "extreme": extreme, "failures": failures,
        "requestedFullSessions": req_full, "requestedWarmupSessions": req_warm,
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else "{:.1%}".format(x)


def _money(x: Optional[float]) -> str:
    return "n/a" if x is None else "${:,.0f}".format(x)


def _n(x: Optional[float], nd: int = 4) -> str:
    return "n/a" if x is None else "{:.{}f}".format(x, nd)


def render_markdown(report: Dict[str, Any], tickers: List[str], generated_at: datetime, complete: bool) -> str:
    lines = ["# Macro Options Flow -- Phase 1 backfill QA report", "",
            "Generated {} UTC. Methodology version `{}`. Status: **{}**.".format(
                generated_at.isoformat(timespec="seconds"), METHODOLOGY_VERSION,
                "COMPLETE" if complete else "IN PROGRESS -- numbers below are a snapshot, not final"),
            ""]

    lines += ["## Item 5: per-ETF and combined summary", "",
              "| Ticker | Full-flow req | S | P | F | Warmup req | S | P | F | Total Theta req | "
              "Median req/day | Median runtime/day (s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in report["summaries"] + [report["combined"]]:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:,} | {} | {} |".format(
            s["ticker"], s["fullFlowRequested"], s["fullFlowSuccess"], s["fullFlowPartial"], s["fullFlowFailed"],
            s["warmupRequested"], s["warmupSuccess"], s["warmupPartial"], s["warmupFailed"],
            s["totalThetaRequests"], _n(s["medianRequestsPerTickerDay"], 0), _n(s["medianRuntimeSecPerTickerDay"], 1)))
    lines.append("")
    lines += ["Total wall-clock time (sum of runtimeSec across all ticker-days): {:.0f}s ({:.1f}h)".format(
        report["combined"]["totalWallClockSec"], report["combined"]["totalWallClockSec"] / 3600.0), ""]

    lines += ["### Quality medians (full-flow only) and missing counts", "",
              "| Ticker | Trades | Eligible | Gross premium | Classified | Greek match | OI match | "
              "Missing ATM | 7D | 30D | 60D | Skew | Delta | 20D% | 60D% | 126D% | 252D% |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in report["summaries"] + [report["combined"]]:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            s["ticker"], _n(s["medianTradesDownloaded"], 0), _n(s["medianEligibleTrades"], 0),
            _money(s["medianGrossPremium"]), _pct(s["medianClassifiedPctPremium"]), _pct(s["medianGreekMatchedPctPremium"]),
            _pct(s["medianOiMatchedPctPremium"]), s["missingAtmIv"], s["missingIv7d"], s["missingIv30d"],
            s["missingIv60d"], s["missingSkew"], s["missingDeltaImbalance"], s["missingPct20d"],
            s["missingPct60d"], s["missingPct126d"], s["missingPct252d"]))
    lines.append("")

    lines += ["## Item 6: distribution QA (full-flow, per ETF)", ""]
    for t in tickers:
        lines.append("### {}".format(t))
        lines.append("| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, label in qa.DISTRIBUTION_FIELDS:
            d = report["distribution"][t][label]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label, _n(d["min"]), _n(d["p1"]), _n(d["p5"]), _n(d["median"]), _n(d["mean"]),
                _n(d["p95"]), _n(d["p99"]), _n(d["max"]), d["n"]))
        lines.append("")
        if report["extreme"][t]:
            lines.append("Flagged extreme net_trade_sentiment observations (kept in the dataset, not removed):")
            for x in report["extreme"][t][:10]:
                lines.append("  - {} {}: {:.3f} (median {:.3f}, robust z={})".format(
                    x["ticker"], x["marketDate"], x["value"], x["median"], x["robustZ"]))
            lines.append("")

    lines += ["## Item 7: reconciliation", ""]
    for t in tickers + ["ALL"]:
        r = report["reconciliation"][t]
        lines.append("**{}**  ".format(t))
        lines.append("- aggression (call/put bought+sold) vs gross premium: median abs diff {}, "
                     "worst abs diff {}, median rel diff {}, worst rel diff {} (n={}). *{}*".format(
                         _money(r["aggression"]["medianAbsDiff"]), _money(r["aggression"]["worstAbsDiff"]),
                         _pct(r["aggression"]["medianRelDiff"]), _pct(r["aggression"]["worstRelDiff"]),
                         r["aggression"]["n"], r["aggression"]["note"]))
        lines.append("- DTE buckets vs eligible premium: median abs diff {}, worst abs diff {}, "
                     "median rel diff {}, worst rel diff {} (n={}). *{}*".format(
                         _money(r["dteBuckets"]["medianAbsDiff"]), _money(r["dteBuckets"]["worstAbsDiff"]),
                         _pct(r["dteBuckets"]["medianRelDiff"]), _pct(r["dteBuckets"]["worstRelDiff"]),
                         r["dteBuckets"]["n"], r["dteBuckets"]["note"]))
        lines.append("")

    lines += ["## Item 8: cross-ETF comparability", "",
              "| Ticker | Median gross premium | Median net $ delta | Median trades |",
              "|---|---:|---:|---:|"]
    for c in report["comparability"]:
        lines.append("| {} | {} | {} | {} |".format(
            c["ticker"], _money(c["medianGrossPremium"]), _money(c["medianNetDollarDelta"]), _n(c["medianTrades"], 0)))
    lines.append("")
    comparable, note = report["comparabilityVerdict"]
    lines.append("**Verdict:** {}".format(note))
    lines.append("")

    if report["failures"]:
        lines += ["## Failures", "", "See `reports/options-flow-backfill-failures.csv` ({} rows).".format(
            len(report["failures"])), ""]

    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


SUMMARY_CSV_FIELDS = [
    "ticker", "fullFlowRequested", "fullFlowSuccess", "fullFlowPartial", "fullFlowFailed",
    "warmupRequested", "warmupSuccess", "warmupPartial", "warmupFailed",
    "totalThetaRequests", "medianRequestsPerTickerDay", "medianRuntimeSecPerTickerDay", "totalWallClockSec",
    "medianTradesDownloaded", "medianEligibleTrades", "medianGrossPremium",
    "medianClassifiedPctPremium", "medianGreekMatchedPctPremium", "medianOiMatchedPctPremium",
    "missingAtmIv", "missingIv7d", "missingIv30d", "missingIv60d", "missingSkew", "missingDeltaImbalance",
    "missingPct20d", "missingPct60d", "missingPct126d", "missingPct252d",
]
FAILURE_CSV_FIELDS = ["ticker", "marketDate", "mode", "createdAt", "diagnostics"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--full-flow-start", default=DEFAULT_FULL_FLOW[0].isoformat())
    ap.add_argument("--full-flow-end", default=DEFAULT_FULL_FLOW[1].isoformat())
    ap.add_argument("--warmup-start", default=DEFAULT_WARMUP[0].isoformat())
    ap.add_argument("--warmup-end", default=DEFAULT_WARMUP[1].isoformat())
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse to write reports unless every requested ticker-day is accounted for")
    ap.add_argument("--out-dir", default=str(REPORTS_DIR))
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    full_range = (date.fromisoformat(args.full_flow_start), date.fromisoformat(args.full_flow_end))
    warmup_range = (date.fromisoformat(args.warmup_start), date.fromisoformat(args.warmup_end))

    from api.db import get_connection
    with get_connection() as conn:
        data = build_universe_data(conn, tickers, full_range, warmup_range)

    problems = check_complete(data, tickers)
    complete = not problems
    if args.require_complete and problems:
        print("NOT COMPLETE -- refusing to write final reports:")
        for p in problems:
            print("  " + p)
        return 1

    report = build_report(data, tickers)
    out_dir = Path(args.out_dir)
    md_path = out_dir / "options-flow-backfill-qa.md"
    csv_path = out_dir / "options-flow-backfill-qa.csv"
    fail_path = out_dir / "options-flow-backfill-failures.csv"

    md = render_markdown(report, tickers, datetime.now(timezone.utc), complete)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    write_csv(csv_path, report["summaries"] + [report["combined"]], SUMMARY_CSV_FIELDS)
    write_csv(fail_path, report["failures"], FAILURE_CSV_FIELDS)

    print("Wrote {}".format(md_path))
    print("Wrote {}".format(csv_path))
    print("Wrote {} ({} rows)".format(fail_path, len(report["failures"])))
    if not complete:
        print("NOTE: backfill is not yet complete ({} gaps) -- this is a progress snapshot.".format(len(problems)))
        for p in problems[:20]:
            print("  " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
