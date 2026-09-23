"""
scripts/build_options_flow_forward_returns.py

Builds reports/options-flow-forward-returns.parquet: one row per (date, ticker) with
close-to-close forward returns (ret_1d, ret_3d, ret_5d, ret_10d, ret_20d), counted in TRADING
SESSIONS, using src.data_sources.fetch_prices -- the existing Macro Engine historical ETF
price source (yfinance-backed, already used by jobs/macro_thesis_job.py).

Outcomes only. Never reads or writes anything under api/services/options_flow_metrics.py's
domain, never touches the feature dataset (reports/options-flow-research-dataset.parquet),
and this script's output is joined to features ONLY downstream, in
scripts/run_options_flow_predictive_study.py -- feature construction never sees this file.

    python scripts/build_options_flow_forward_returns.py
    python scripts/build_options_flow_forward_returns.py --dataset reports/options-flow-research-dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.options_flow_forward_returns import HORIZONS, compute_forward_returns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(ROOT / "reports" / "options-flow-research-dataset.parquet"),
                    help="feature parquet to read (date, ticker) pairs from -- read-only, never modified")
    ap.add_argument("--out", default=str(ROOT / "reports" / "options-flow-forward-returns.parquet"))
    ap.add_argument("--period", default="1y", help="src.data_sources.fetch_prices lookback window")
    args = ap.parse_args()

    import pandas as pd
    from src.data_sources import fetch_prices

    features = pd.read_parquet(args.dataset)
    tickers = sorted(features["ticker"].unique())
    dates = sorted(pd.to_datetime(features["date"]).dt.date.unique())
    print("Fetching prices for {} via src.data_sources.fetch_prices(period={!r})...".format(tickers, args.period))

    prices = fetch_prices(tickers, period=args.period)
    if prices.empty:
        print("ERROR: fetch_prices returned no data.")
        return 1
    prices.index = pd.to_datetime(prices.index).date
    prices = prices[~pd.Index(prices.index).duplicated(keep="last")].sort_index()
    prices.index = pd.to_datetime(prices.index)
    last_price_date = prices.index.max().date()
    print("Price history: {} -> {} ({} sessions, tickers: {})".format(
        prices.index.min().date(), last_price_date, len(prices), list(prices.columns)))

    out = compute_forward_returns(prices, dates, tickers)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    n_total = len(out)
    print("Wrote {} ({} rows, {} tickers, {} -> {})".format(out_path, n_total, len(tickers), dates[0], dates[-1]))
    for h in HORIZONS:
        col = "ret_{}d".format(h)
        n_valid = int(out[col].notna().sum())
        print("  ret_{}d: {} of {} rows have a real value ({:.0%}); missing = future session not yet observed".format(
            h, n_valid, n_total, n_valid / n_total if n_total else 0))
    print("  last available price date: {}".format(last_price_date))
    return 0


if __name__ == "__main__":
    sys.exit(main())
