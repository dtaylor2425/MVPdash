"""
src/fx/frankfurter.py

Client for https://api.frankfurter.dev -- ECB (and other central-bank)
reference fixings. No API key, no quota. History to 1999.

IMPORTANT (spec section 3.2): Frankfurter returns daily *reference fixings*,
not live quotes. Every consumer must carry the observation date and label it
a fixing. This client never pretends otherwise -- `FxData.observation_date`
is always populated from the payload.

Results are cached to disk (parquet) so the ingest job is the only thing that
hits the network, and the API routes never do.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from src.config import CACHE_DIR
from src.fx.series_map import FRANKFURTER_QUOTES

BASE_URL = "https://api.frankfurter.dev/v2"
_CACHE_NAME = "fx_frankfurter"
_CACHE_MAX_AGE_SECONDS = 6 * 3600
_TIMEOUT = 30


class FxData:
    """
    Wraps the USD-based fixing table.

    `frame` is a DataFrame indexed by observation date, one column per quote
    currency (upper-case ISO), values = units of quote per 1 USD.
    USD itself is a column of 1.0 for convenience.
    """

    def __init__(self, frame: pd.DataFrame):
        frame = frame.sort_index()
        if "USD" not in frame.columns:
            frame = frame.copy()
            frame["USD"] = 1.0
        self.frame = frame

    # -- metadata ----------------------------------------------------------
    @property
    def observation_date(self) -> Optional[str]:
        if self.frame.empty:
            return None
        return pd.Timestamp(self.frame.index.max()).date().isoformat()

    @property
    def currencies(self) -> List[str]:
        return list(self.frame.columns)

    # -- accessors -------------------------------------------------------- --
    def usd_per(self, ccy: str) -> pd.Series:
        """Units of `ccy` per 1 USD (raw fixing)."""
        ccy = ccy.upper()
        if ccy not in self.frame.columns:
            return pd.Series(dtype="float64")
        return self.frame[ccy].dropna()

    def spot_usd(self, ccy: str) -> pd.Series:
        """
        Value of `ccy` expressed in USD (USD per 1 unit of ccy).
        This is the conventional 'spot vs USD' quote for the detail page.
        """
        s = self.usd_per(ccy)
        if s.empty:
            return s
        return (1.0 / s).rename(ccy)

    def cross(self, base: str, quote: str) -> pd.Series:
        """Units of `quote` per 1 unit of `base`."""
        b = self.usd_per(base)
        q = self.usd_per(quote)
        idx = b.index.intersection(q.index)
        if len(idx) == 0:
            return pd.Series(dtype="float64")
        return (q.loc[idx] / b.loc[idx]).rename(f"{base}{quote}")

    def effective_index(self, ccy: str, partners: Dict[str, float]) -> pd.Series:
        """
        Static trade-weighted nominal index for `ccy`: geometric mean of the
        price of `ccy` measured in each partner currency, partner weights
        normalised to sum to 1. Rising = `ccy` strengthening.
        """
        ccy = ccy.upper()
        weights = {k.upper(): v for k, v in partners.items() if v and v > 0}
        wsum = sum(weights.values())
        if wsum <= 0:
            return pd.Series(dtype="float64")
        acc: Optional[pd.Series] = None
        for partner, w in weights.items():
            price = self.cross(partner, ccy)  # units of ccy per partner -> invert
            if price.empty:
                continue
            leg = (1.0 / price).pow(w / wsum)  # price of ccy in partner terms
            acc = leg if acc is None else acc.mul(leg, fill_value=None).dropna()
        if acc is None:
            return pd.Series(dtype="float64")
        return acc.rename(f"{ccy}_neer")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def _parquet_path() -> Path:
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / f"{_CACHE_NAME}.parquet"


def _parse_rates_payload(payload) -> pd.DataFrame:
    # api.frankfurter.dev/v2 returns a flat list of long-format records --
    # both the "latest" and "from=" forms:
    #   [{"date": "2024-01-02", "base": "USD", "quote": "EUR", "rate": 0.9}, ...]
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        rows: dict = {}
        for rec in payload:
            d, q, r = rec.get("date"), rec.get("quote"), rec.get("rate")
            if d is None or q is None or r is None:
                continue
            rows.setdefault(d, {})[q] = r
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).T
        frame.index = pd.to_datetime(frame.index)
        return frame.sort_index().astype("float64")

    # Defensive fallback for a classic v1-shaped nested payload, in case a
    # future deploy (or a different Frankfurter mirror) reverts the shape.
    if isinstance(payload, dict):
        rates = payload.get("rates") or {}
        if rates and all(isinstance(v, dict) for v in rates.values()):
            frame = pd.DataFrame(rates).T
            frame.index = pd.to_datetime(frame.index)
            return frame.sort_index().astype("float64")
        obs = payload.get("date")
        if obs and rates:
            frame = pd.DataFrame([rates], index=[pd.to_datetime(obs)])
            return frame.astype("float64")
    return pd.DataFrame()


def _request(params: dict, path: str = "/rates") -> pd.DataFrame:
    resp = requests.get(BASE_URL + path, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return _parse_rates_payload(resp.json())


def fetch_history(
    start: date,
    quotes: Optional[Iterable[str]] = None,
    use_cache: bool = True,
) -> FxData:
    """
    Daily USD-based fixings from `start` to latest. Cached to disk; a cache
    younger than 6h is used as-is, otherwise refreshed and merged.
    """
    quotes = [q.lower() for q in (quotes or FRANKFURTER_QUOTES)]
    cache_path = _parquet_path()
    cached: Optional[pd.DataFrame] = None
    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            if age < _CACHE_MAX_AGE_SECONDS and not cached.empty:
                covered = cached.index.min().date() <= start
                if covered and set(c.upper() for c in quotes).issubset(cached.columns):
                    return FxData(cached)
        except Exception:
            cached = None

    params = {"base": "usd", "from": start.isoformat(), "quotes": ",".join(quotes)}
    try:
        fresh = _request(params)
    except Exception:
        if cached is not None and not cached.empty:
            return FxData(cached)
        raise

    fresh.columns = [c.upper() for c in fresh.columns]
    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh.sort_index()

    try:
        merged.to_parquet(cache_path, index=True)
    except Exception:
        pass
    return FxData(merged)


def fetch_latest(quotes: Optional[Iterable[str]] = None) -> FxData:
    quotes = [q.lower() for q in (quotes or FRANKFURTER_QUOTES)]
    params = {"base": "usd", "quotes": ",".join(quotes)}
    frame = _request(params)
    frame.columns = [c.upper() for c in frame.columns]
    return FxData(frame)


def default_history(years: float = 2.6) -> FxData:
    start = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(years * 365))).date()
    return fetch_history(start)


def load_cached() -> Optional[FxData]:
    """
    Read the disk-cached fixing table with no network fallback. Used by
    read-only consumers (the breadth API route) that must never call
    Frankfurter on request -- only jobs/fx_snapshot_job.py fetches live.
    """
    path = _parquet_path()
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        if frame.empty:
            return None
        return FxData(frame)
    except Exception:
        return None
