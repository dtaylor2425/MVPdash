"""
scripts/extract_options_flow_payload_features.py

Signal Design v2, step 0: pull the richer per-day fields that already exist in each Phase 1
full_flow snapshot's stored `payload` JSONB (Postgres, options_flow_symbol_snapshots) but never
made it into the flat 33-column reports/options-flow-research-dataset.parquet export. This is
READ-ONLY against already-computed Phase 1 data -- no ThetaData call, no new backfill, no write
to any options_flow_* table, no change to methodology_version or any stored snapshot.

Extracts, per (market_date, ticker), from the exact payload built by
api/services/options_flow_metrics.py::build_symbol_payload:

    * DTE-bucket-level sentiment/deltaRatio/grossPremium/sharePct/signedPremium (payload["dte"])
      -- the daily aggregate export only kept grossPremium per bucket, not the DIRECTIONAL
      information per bucket, which is what DTE-segmentation hypothesis testing (H2, item 2A)
      actually needs.
    * call-only / put-only aggression sentiment (payload["aggression"])
    * top-10-trade premium concentration (payload["premium"]["top10Concentration"])
    * the day's 25 largest trades (payload["largeTrades"]) -- used to build large/unusual-trade
      features (H1, item 2C). NOTE: only the top 25 trades per ticker-day are persisted (by
      design -- raw ticks are never stored), so any concentration/large-trade metric here is a
      lower bound / proxy over the full trade population, not a true percentile-of-all-trades
      metric. This limitation is carried into the v2 report and the 252-day-backfill field
      recommendations.
    * 15-minute intraday buckets (payload["intraday"]) -- used to build opening/midday/closing
      time-of-day sentiment (H3, item 2D).

Writes reports/options-flow-payload-features.parquet, one row per (date, ticker), matching the
existing 360-row Phase 1 grid.

    python scripts/extract_options_flow_payload_features.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "reports" / "options-flow-payload-features.parquet"

DTE_BUCKET_KEYS = [("0DTE", "0dte"), ("1-7D", "1_7d"), ("8-30D", "8_30d"),
                  ("31-60D", "31_60d"), ("60D+", "60d_plus")]


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # Fall back to the same Railway lookup the rest of this project's scratch tooling uses.
    out = subprocess.run(
        ["railway", "variables", "--service", "Postgres", "--kv"],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("DATABASE_PUBLIC_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Could not resolve DATABASE_URL from env or `railway variables`.")


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _sum_bucket(buckets: List[Dict[str, Any]], key: str) -> float:
    return sum(b.get(key) or 0.0 for b in buckets)


def _hhi(values: List[float]) -> Optional[float]:
    total = sum(values)
    if total <= 0 or not values:
        return None
    return sum((v / total) ** 2 for v in values)


def extract_row(ticker: str, market_date: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"date": str(market_date), "ticker": ticker}

    # --- DTE-bucket sentiment/delta/share (item 2A) ---
    dte_buckets = {b["bucket"]: b for b in (payload.get("dte") or {}).get("buckets", [])}
    for label, suffix in DTE_BUCKET_KEYS:
        b = dte_buckets.get(label, {})
        row[f"dte_{suffix}_sentiment"] = b.get("sentiment")
        row[f"dte_{suffix}_delta_ratio"] = b.get("deltaRatio")
        row[f"dte_{suffix}_share_pct"] = b.get("sharePct")
        row[f"dte_{suffix}_signed_premium"] = b.get("signedPremium")
        row[f"dte_{suffix}_gross_premium"] = b.get("grossPremium")
        row[f"dte_{suffix}_trades"] = b.get("trades")

    # --- call-only / put-only flow sentiment (item 2E) ---
    agg = payload.get("aggression") or {}
    call_bought, call_sold = agg.get("callBought"), agg.get("callSold")
    put_bought, put_sold = agg.get("putBought"), agg.get("putSold")
    row["call_sentiment"] = _ratio(
        (call_bought - call_sold) if (call_bought is not None and call_sold is not None) else None,
        (call_bought + call_sold) if (call_bought is not None and call_sold is not None) else None,
    )
    row["put_sentiment"] = _ratio(
        (put_bought - put_sold) if (put_bought is not None and put_sold is not None) else None,
        (put_bought + put_sold) if (put_bought is not None and put_sold is not None) else None,
    )

    # --- concentration (item 4) ---
    row["top10_concentration"] = (payload.get("premium") or {}).get("top10Concentration")

    # --- large/unusual trades: the day's persisted top-25 (item 2C, H1) ---
    large = payload.get("largeTrades") or []
    row["large_trade_count"] = len(large)
    if large:
        premiums = [float(t["premium"]) for t in large if t.get("premium") is not None]
        directional = [
            float(t["premium"]) * float(t["aggressor"]) * (1.0 if t.get("right") == "C" else -1.0)
            for t in large if t.get("premium") is not None and t.get("aggressor") is not None
        ]
        gross = sum(premiums)
        net = sum(directional)
        row["large_trade_gross_premium"] = gross
        row["large_trade_directional_premium"] = net
        row["large_trade_sentiment"] = _ratio(net, gross)
        zero_dte_premium = sum(float(t["premium"]) for t in large if t.get("dte") == 0 and t.get("premium") is not None)
        row["large_trade_0dte_share"] = _ratio(zero_dte_premium, gross)
        dtes = [t["dte"] for t in large if t.get("dte") is not None]
        row["large_trade_median_dte"] = float(sorted(dtes)[len(dtes) // 2]) if dtes else None
        row["top25_premium_hhi"] = _hhi(premiums)  # proxy only -- see module docstring
    else:
        for c in ("large_trade_gross_premium", "large_trade_directional_premium", "large_trade_sentiment",
                 "large_trade_0dte_share", "large_trade_median_dte", "top25_premium_hhi"):
            row[c] = None

    # --- time-of-day (item 2D, H3): first/last 4 of the 15-minute buckets = ~60 min each ---
    intraday = payload.get("intraday") or []
    n = len(intraday)
    if n >= 8:  # need at least a distinct opening and closing window
        opening = intraday[:4]
        closing = intraday[-4:]
        middle = intraday[4:-4]
    elif n > 0:
        opening, closing, middle = intraday, [], []
    else:
        opening, closing, middle = [], [], []
    for name, bucket_list in (("opening_hour", opening), ("closing_hour", closing), ("midday", middle)):
        net = sum(b.get("netPremium") or 0.0 for b in bucket_list) if bucket_list else None
        gross = sum(b.get("grossPremium") or 0.0 for b in bucket_list) if bucket_list else None
        row[f"{name}_net_premium"] = net
        row[f"{name}_gross_premium"] = gross
        row[f"{name}_sentiment"] = _ratio(net, gross) if bucket_list else None
    row["intraday_buckets_available"] = n

    return row


def main() -> int:
    import pandas as pd
    import psycopg
    from psycopg.rows import dict_row

    database_url = _get_database_url()
    conn = psycopg.connect(database_url, row_factory=dict_row)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ticker, market_date, payload FROM options_flow_symbol_snapshots "
            "WHERE mode = 'full_flow' ORDER BY ticker, market_date"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Fetched {len(rows)} full_flow payload rows from Postgres (read-only).")
    extracted = [extract_row(r["ticker"], r["market_date"], r["payload"]) for r in rows]
    df = pd.DataFrame(extracted)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} rows, {df['ticker'].nunique()} tickers, "
         f"{df['date'].min()} -> {df['date'].max()})")
    print(f"Columns: {list(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
