"""
scripts/export_options_flow_research_dataset.py

Builds reports/options-flow-research-dataset.parquet: one row per (date, ticker), FULL-FLOW
snapshots only (source='historical_backfill', mode='full_flow'), covering only fields knowable
by market close that date (api/services/options_flow_qa.research_row). No forward-looking data
of any kind -- in particular, no future ETF returns are added here; that is a later, separate
step once this dataset has been reviewed.

Deliberately excludes iv-warmup-only rows: they exist solely to seed IV percentile history and
carry no sentiment/premium/delta/aggression/DTE fields, so they are not usable observations for
a predictive-return backtest (same reasoning as "do not include warmup-only dates in the
predictive backtest sample counts").

Refuses to run unless the QA report says the backfill is complete for the requested scope,
unless --force is passed -- this is meant to run only after
scripts/generate_options_flow_qa_report.py has been reviewed and the data judged ready.

    python scripts/export_options_flow_research_dataset.py
    python scripts/export_options_flow_research_dataset.py --tickers SPY,QQQ --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_qa as qa  # noqa: E402
from api.services.options_flow_metrics import METHODOLOGY_VERSION  # noqa: E402
from scripts.generate_options_flow_qa_report import (  # noqa: E402
    DEFAULT_FULL_FLOW,
    DEFAULT_TICKERS,
    DEFAULT_WARMUP,
    build_universe_data,
    check_complete,
)

REPORTS_DIR = ROOT / "reports"
RESEARCH_COLUMNS = [
    "date", "ticker", "methodology_version",
    "net_trade_sentiment", "net_directional_premium", "gross_premium",
    "delta_imbalance_ratio", "net_dollar_delta",
    "call_bought_premium", "call_sold_premium", "put_bought_premium", "put_sold_premium",
    "zero_dte_share", "dte_0dte_gross_premium", "dte_1_7d_gross_premium", "dte_8_30d_gross_premium",
    "dte_31_60d_gross_premium", "dte_60d_plus_gross_premium",
    "atm_iv", "iv_7d", "iv_30d", "iv_60d", "iv_term_slope_7v30", "iv_term_slope_30v60", "put_skew_25d",
    "iv_percentile_20d", "iv_percentile_60d", "iv_percentile_126d", "iv_percentile_252d",
    "classification_coverage", "greek_match_coverage", "oi_match_coverage", "trade_count",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--full-flow-start", default=DEFAULT_FULL_FLOW[0].isoformat())
    ap.add_argument("--full-flow-end", default=DEFAULT_FULL_FLOW[1].isoformat())
    ap.add_argument("--out", default=str(REPORTS_DIR / "options-flow-research-dataset.parquet"))
    ap.add_argument("--force", action="store_true", help="export even if some requested ticker-days are missing")
    args = ap.parse_args()

    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    full_range = (date.fromisoformat(args.full_flow_start), date.fromisoformat(args.full_flow_end))

    from api.db import get_connection
    with get_connection() as conn:
        # warmup range only matters for check_complete's IV-percentile gap accounting, not for
        # which rows get exported (export is full-flow only) -- default scope is fine here.
        data = build_universe_data(conn, tickers, full_range, DEFAULT_WARMUP)

    problems = [p for p in check_complete(data, tickers) if "full-flow" in p]
    if problems and not args.force:
        print("Full-flow backfill is not complete for the requested scope -- refusing to export "
             "(pass --force to override):")
        for p in problems:
            print("  " + p)
        return 1

    import pandas as pd

    rows = []
    for t in tickers:
        for raw in data["perTicker"][t]["fullRaw"]:
            rows.append(qa.research_row(raw, METHODOLOGY_VERSION))
    if not rows:
        print("No full-flow rows found for the requested scope; nothing to export.")
        return 1

    df = pd.DataFrame(rows)[RESEARCH_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print("Wrote {} ({} rows, {} columns, {} tickers, {} -> {})".format(
        out_path, len(df), len(df.columns), df["ticker"].nunique(), df["date"].min(), df["date"].max()))
    counts = df.groupby("ticker").size().to_dict()
    for t in tickers:
        print("  {}: {} rows".format(t, counts.get(t, 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
