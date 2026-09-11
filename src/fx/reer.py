"""
src/fx/reer.py

BIS real effective exchange rate indices (broad basket, monthly). Free bulk
download, no API (spec section 3.4). Optional for v1: any failure here must
degrade to an unavailable `valuation` component, never a hard error.

Cached to disk as CSV under data/cache/. If the BIS layout changes, drop a
manually downloaded CSV at data/cache/bis_reer.csv with columns
[date, <ISO currency>...] and this loader will use it.
"""

from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.config import CACHE_DIR
from src.fx.series_map import SCORED

# BIS "WS_EER" real, broad indices. Verified reachable 2026-09-10; BIS has
# moved this bulk path before (it used to live under www.bis.org/statistics/),
# so re-verify with `curl -I <url>` if load_reer() starts returning {}.
_BIS_URL = "https://data.bis.org/static/bulk/WS_EER_csv_row.zip"
_CACHE = Path(CACHE_DIR) / "bis_reer.csv"
_CACHE_MAX_AGE_DAYS = 20
_TIMEOUT = 60

# BIS reference-area codes -> our ISO currency codes.
_BIS_AREA_TO_CCY = {
    "US": "USD", "XM": "EUR", "JP": "JPY", "GB": "GBP", "CH": "CHF",
    "CA": "CAD", "AU": "AUD", "NZ": "NZD", "SE": "SEK", "NO": "NOK",
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
    Parse the BIS EER bulk CSV ("...csv_row.zip"). Despite the name this is a
    wide layout: each column is one series, described by several metadata
    ROWS (Frequency / Type / Basket / Reference area / ... / Title), then one
    row per date from the "Time Period" header row onward. We want the
    Type=Real, Basket=Broad column for each reference area.

    Verified against the live file on 2026-09-10. If BIS changes this layout
    again, this will return {} (see load_reer's docstring) rather than raise.
    """
    import csv

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
    type_row = header.get("type")
    basket_row = header.get("basket")
    area_row = header.get("reference area")
    if not (type_row and basket_row and area_row):
        return pd.DataFrame()

    n_cols = len(area_row)
    wanted: Dict[int, str] = {}
    for j in range(1, n_cols):
        area_code = area_row[j].split(":")[0].strip().upper()
        ccy = _BIS_AREA_TO_CCY.get(area_code)
        if not ccy or ccy in wanted.values():
            continue
        is_real = j < len(type_row) and type_row[j].strip().upper().startswith("R:")
        is_broad = j < len(basket_row) and "BROAD" in basket_row[j].strip().upper()
        if is_real and is_broad:
            wanted[j] = ccy

    if not wanted:
        return pd.DataFrame()

    # Data rows start after the "Time Period" header row.
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
                values[ccy].append(math.nan)

    if not dates:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(dates))
    return frame.sort_index()


def load_reer(use_cache: bool = True) -> Dict[str, pd.Series]:
    """
    {ISO currency -> monthly REER series}. Empty dict on any failure -- the
    caller treats missing currencies as an unavailable valuation component.
    """
    if use_cache:
        cached = _fresh_cache()
        if cached is not None:
            return {c: cached[c].dropna() for c in cached.columns if c in SCORED}

    try:
        resp = requests.get(_BIS_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        frame = _parse_bis_zip(resp.content)
    except Exception:
        cached = _fresh_cache() if not use_cache else None
        if cached is not None:
            return {c: cached[c].dropna() for c in cached.columns if c in SCORED}
        return {}

    if frame.empty:
        return {}
    try:
        frame.reset_index().rename(columns={"index": "date"}).to_csv(_CACHE, index=False)
    except Exception:
        pass
    return {c: frame[c].dropna() for c in frame.columns if c in SCORED}
