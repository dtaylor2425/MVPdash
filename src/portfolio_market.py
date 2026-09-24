"""Raw completed-session bars for published portfolio accounting."""
from datetime import date, timedelta
import math
import pandas as pd
import yfinance as yf


def historical_share_bars(data, end):
    """Undo Yahoo's split-only OHLC adjustment before booking split events.

    auto_adjust=False disables dividend adjustment, NOT split adjustment.
    Keep the common-share basis separately for volume/direction signals.
    Today's action must be included even though today's price is not valued.
    """
    bars = []
    factor = 1.0
    for timestamp, row in data.sort_index(ascending=False).iterrows():
        split = float(row.get("Stock Splits", 0) or 0)
        if not math.isfinite(split) or split < 0:
            raise ValueError("Invalid corporate-action data")
        values = {key: float(row.get(field, 0) or 0) for key, field in (
            ("open", "Open"), ("close", "Close"), ("volume", "Volume"),
            ("dividend", "Dividends"))}
        if timestamp.date() < end and all(math.isfinite(v) for v in values.values()) and values["open"] > 0 and values["close"] > 0:
            bars.append({"date": timestamp.date().isoformat(), "split": split,
                         "open": values["open"] * factor, "close": values["close"] * factor,
                         "dividend": values["dividend"] * factor, "volume": values["volume"] / factor,
                         "signal_close": values["close"], "signal_volume": values["volume"]})
        if split > 0:
            factor *= split
    return list(reversed(bars))


def load_market(tickers, run_date, anchor=None):
    end = date.fromisoformat(str(run_date)[:10])
    start = end - timedelta(days=90)
    if anchor:
        start = min(start, date.fromisoformat(str(anchor)[:10]) - timedelta(days=45))
    symbols = sorted(set(tickers) | {"SPY"})
    frame = yf.download(symbols, start=start.isoformat(), end=(end+timedelta(days=1)).isoformat(),
                        auto_adjust=False, actions=True, progress=False, threads=True,
                        group_by="ticker", timeout=30)
    result = {}
    for ticker in symbols:
        if isinstance(frame.columns, pd.MultiIndex):
            if ticker not in frame.columns.get_level_values(0):
                raise ValueError(f"Market data unavailable for {ticker}")
            data = frame[ticker]
        else:
            data = frame
        bars = historical_share_bars(data, end)
        if not bars:
            raise ValueError(f"No completed market bars for {ticker}")
        result[ticker] = bars
    import pandas_market_calendars as mcal
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end-timedelta(days=1))
    expected = {d.date().isoformat() for d in schedule.index}
    actual = {b["date"] for b in result["SPY"]}
    if expected - actual:
        raise ValueError("Benchmark sessions missing; refusing an incomplete valuation")
    return result
