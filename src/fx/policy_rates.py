"""
src/fx/policy_rates.py

BIS "Central bank policy rates" bulk dataset -- the actual announced rate for
each central bank (SNB policy rate, Bank of Canada target rate, etc.), not a
money-market proxy (fix-list item 5). Free bulk download, no API key, updates
daily. Third data source alongside FRED and Frankfurter.

Every non-USD/EUR `policy_rate` in src/fx/series_map.py's FRED_SERIES is
actually an OECD money-market rate (call money/interbank, or 3m interbank)
standing in for the announced rate -- that's why, e.g., CHF read -0.04% and
CAD read 2.27% instead of the SNB's and Bank of Canada's stated targets. This
module is the fix: it is used for the *displayed* policy rate and the carry
level. The FRED money-market series are kept regardless (src/fx/components.py
still reads them) -- the spread between a policy rate and its money-market
rate is a funding-stress signal worth having later.

Cached to disk as CSV under data/cache/, same pattern as src/fx/reer.py. Any
failure here degrades to the existing FRED money-market proxy, never a hard
error -- this is a nicer number, not a required one.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.config import CACHE_DIR
from src.fx.series_map import SCORED

# BIS "WS_CBPOL" central bank policy rates. Verified reachable 2026-09-10;
# the BIS bulk path has moved before (see src/fx/reer.py) -- re-verify with
# `curl -I <url>` if load_policy_rates() starts returning {}.
_BIS_URL = "https://data.bis.org/static/bulk/WS_CBPOL_csv_row.zip"
_CACHE = Path(CACHE_DIR) / "bis_policy_rates.csv"
_CACHE_MAX_AGE_DAYS = 1  # this dataset updates daily; keep the cache tight
_TIMEOUT = 60

# BIS reference-area codes -> our ISO currency codes. XM is BIS's code for
# the Euro area aggregate (unlike src/fx/reer.py's REER dataset, this one
# actually publishes a live Euro-area policy rate, so no per-member fallback
# is needed here).
_BIS_AREA_TO_CCY = {
    "US": "USD", "XM": "EUR", "JP": "JPY", "GB": "GBP", "CH": "CHF",
    "CA": "CAD", "AU": "AUD", "NZ": "NZD", "SE": "SEK", "NO": "NOK",
    "CN": "CNY",
}


def _fresh_cache() -> Optional[pd.DataFrame]:
    if not _CACHE.exists():
        return None
    age_days = (datetime.now(timezone.utc).timestamp() - _CACHE.stat().st_mtime) / 86400
    try:
        df = pd.read_csv(_CACHE, parse_dates=["date"]).set_index("date")
        if age_days <= _CACHE_MAX_AGE_DAYS and not df.empty:
            return df
    except Exception:
        return None
    return None


def _parse_bis_zip(content: bytes) -> pd.DataFrame:
    """
    Parse the BIS CBPOL bulk CSV ("...csv_row.zip"). Wide layout: each column
    is one series described by metadata ROWS (Frequency / Reference area /
    ... / Title), then one row per date from the "Time Period" header row
    onward. Several countries publish both a Daily and a Monthly column for
    the same reference area -- the daily one is preferred.

    Verified against the live file on 2026-09-10. Returns an empty frame
    (never raises) if BIS changes this layout again -- see load_policy_rates.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not name:
            return pd.DataFrame()
        raw = zf.read(name).decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if len(rows) < 9:
        return pd.DataFrame()

    header = {r[0].strip().lower(): r for r in rows[:8] if r}
    freq_row = header.get("frequency")
    area_row = header.get("reference area")
    if not (freq_row and area_row):
        return pd.DataFrame()

    n_cols = len(area_row)
    # Prefer the Daily column per currency; fall back to Monthly if that's
    # all a given central bank publishes.
    best: Dict[str, tuple] = {}  # ccy -> (col_index, is_daily)
    for j in range(1, n_cols):
        area_code = area_row[j].split(":")[0].strip().upper()
        ccy = _BIS_AREA_TO_CCY.get(area_code)
        if not ccy:
            continue
        is_daily = j < len(freq_row) and freq_row[j].strip().upper().startswith("D:")
        prev = best.get(ccy)
        if prev is None or (is_daily and not prev[1]):
            best[ccy] = (j, is_daily)

    if not best:
        return pd.DataFrame()
    wanted = {j: ccy for ccy, (j, _) in best.items()}

    data_start = next(
        (i + 1 for i, r in enumerate(rows) if r and r[0].strip().lower() == "time period"),
        8,
    )

    dates: List[pd.Timestamp] = []
    values: Dict[str, List[Optional[float]]] = {ccy: [] for ccy in wanted.values()}
    for r in rows[data_start:]:
        if not r or not r[0].strip():
            continue
        try:
            d = pd.Timestamp(r[0].strip())
        except Exception:
            continue
        dates.append(d)
        for j, ccy in wanted.items():
            raw_val = r[j].strip() if j < len(r) else ""
            try:
                values[ccy].append(float(raw_val))
            except ValueError:
                values[ccy].append(float("nan"))

    if not dates:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(dates))
    return frame.sort_index()


def load_policy_rates(use_cache: bool = True) -> Dict[str, pd.Series]:
    """
    {ISO currency -> daily/monthly policy-rate series}, BIS-announced rates.
    Empty dict on any failure -- callers fall back to the FRED money-market
    proxy, never fabricate a value.
    """
    if use_cache:
        cached = _fresh_cache()
        if cached is not None:
            return {c: cached[c].dropna() for c in cached.columns if c in SCORED or c == "CNY"}

    try:
        resp = requests.get(_BIS_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        frame = _parse_bis_zip(resp.content)
    except Exception:
        cached = _fresh_cache() if not use_cache else None
        if cached is not None:
            return {c: cached[c].dropna() for c in cached.columns if c in SCORED or c == "CNY"}
        return {}

    if frame.empty:
        return {}
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().rename(columns={"index": "date"}).to_csv(_CACHE, index=False)
    except Exception:
        pass
    return {c: frame[c].dropna() for c in frame.columns if c in SCORED or c == "CNY"}
