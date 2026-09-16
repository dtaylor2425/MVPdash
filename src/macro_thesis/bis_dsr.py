"""
src/macro_thesis/bis_dsr.py  (docs/MACRO-THESIS-BACKEND-SPEC.md section 2)

BIS debt-service-ratio bulk download -- the one "new series required" row
with no FRED mirror (credit-to-GDP, federal debt, GDP, NAIRU, term premium
all resolve directly on FRED; see series_map.py). Mirrors the parsing
approach already used for REER (src/fx/reer.py) and policy rates
(src/fx/policy_rates.py): each column is one series described by several
metadata ROWS, not a normal wide table.

Verified live 2026-09-16: https://data.bis.org/static/bulk/WS_DSR_csv_row.zip
returns a 6-header-row layout (Frequency / Borrowers' country / Borrowers /
Decimals / Title / Time Period), and column index 3 (0-indexed across the
row) is "US:United States" x "P:Private non-financial sector" -- exactly
the series the spec asks for ("Debt service ratio, private non-financial").
If BIS changes this layout, this returns None (see load_dsr_us's docstring)
rather than raising -- the long-term gauge just drops this one component.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import CACHE_DIR

_BIS_URL = "https://data.bis.org/static/bulk/WS_DSR_csv_row.zip"
_CACHE = Path(CACHE_DIR) / "bis_dsr_us.csv"
_CACHE_MAX_AGE_DAYS = 80  # quarterly series; no need to refetch often
_TIMEOUT = 60


def _fresh_cache() -> Optional[pd.Series]:
    if not _CACHE.exists():
        return None
    age_days = (datetime.now(timezone.utc).timestamp() - _CACHE.stat().st_mtime) / 86400
    if age_days > _CACHE_MAX_AGE_DAYS:
        return None
    try:
        df = pd.read_csv(_CACHE, parse_dates=["date"]).set_index("date")
        s = df["dsr"].dropna()
        return s if not s.empty else None
    except Exception:
        return None


def _parse_bis_dsr_zip(content: bytes) -> Optional[pd.Series]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not name:
            return None
        raw = zf.read(name).decode("utf-8", errors="replace")

    rows = list(csv.reader(io.StringIO(raw)))
    if len(rows) < 7:
        return None

    header = {r[0].strip().lower(): r for r in rows[:6] if r}
    country_row = header.get("borrowers' country")
    borrowers_row = header.get("borrowers")
    if not country_row or not borrowers_row:
        return None

    target_col = next(
        (
            j
            for j in range(1, len(country_row))
            if country_row[j].strip().upper().startswith("US:")
            and j < len(borrowers_row)
            and borrowers_row[j].strip().upper().startswith("P:")
        ),
        None,
    )
    if target_col is None:
        return None

    data_start = next(
        (i + 1 for i, r in enumerate(rows) if r and r[0].strip().lower() == "time period"),
        6,
    )

    dates, values = [], []
    for r in rows[data_start:]:
        if not r or not r[0].strip():
            continue
        try:
            d = pd.Period(r[0].strip(), freq="Q").to_timestamp(how="end").normalize()
        except Exception:
            continue
        raw_val = r[target_col].strip() if target_col < len(r) else ""
        try:
            v = float(raw_val)
        except ValueError:
            continue
        dates.append(d)
        values.append(v)

    if not dates:
        return None
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="dsr").sort_index()


def load_dsr_us(use_cache: bool = True) -> Optional[pd.Series]:
    """Quarterly US private-non-financial-sector debt service ratio (% of
    income). None on any failure -- the caller drops this one long-term-gauge
    component rather than failing the whole snapshot."""
    if use_cache:
        cached = _fresh_cache()
        if cached is not None:
            return cached

    try:
        resp = requests.get(_BIS_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        series = _parse_bis_dsr_zip(resp.content)
    except Exception:
        return _fresh_cache() if not use_cache else None

    if series is None or series.empty:
        return _fresh_cache()

    try:
        series.reset_index().rename(columns={"index": "date"}).to_csv(_CACHE, index=False)
    except Exception:
        pass
    return series
