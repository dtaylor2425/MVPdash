"""
src/fx/fred_client.py

Thin wrapper over the repo's existing FRED plumbing (src.data_sources) for the
FX series set, plus a lightweight validator used by both
scripts/validate_fx_series.py and the ingest job.

A silently dead series must NOT reach the model as a zero -- it must become an
unavailable component that the compositor reweights around (spec 3.3). This
module only reports health; the reweighting lives in src/fx/scoring.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.config import CACHE_DIR, FRED_API_KEY
from src.fx.series_map import all_series_ids, confidence_for

_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
_STALE_DAYS = 60
_TIMEOUT = 30


def build_fx_fred_frame(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch every FRED series the FX model needs, columns keyed by logical name
    (e.g. 'EUR__y10', 'cmdty__brent').

    Deliberately returns the RAW (sparse, non-forward-filled) observations,
    not get_fred_cached's ffilled frame. A component calculator must be able
    to tell "no real observation in N days" from "flat because ffilled" --
    the whole point of spec 3.3 is that a dead series must not silently keep
    reporting its last known value forever. src/fx/components.py enforces
    per-series staleness cutoffs (series_map.max_age_for) against this frame.

    Reuses src.data_sources.get_fred_cached for the actual network fetch,
    disk caching and cache-merge (so behaviour matches the rest of the repo),
    then reads back the pre-ffill parquet it writes to disk.
    """
    from src.data_sources import fetch_fred, get_fred_cached
    from src.storage import read_parquet

    flat = all_series_ids()
    if not use_cache:
        return fetch_fred(flat, FRED_API_KEY).sort_index()

    get_fred_cached(flat, FRED_API_KEY, CACHE_DIR, cache_name="fred_fx")  # fetch + write raw cache
    raw = read_parquet(CACHE_DIR, "fred_fx")
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@dataclass
class SeriesCheck:
    logical: str
    series_id: str
    confidence: str
    status: str            # "ok" | "stale" | "404" | "error"
    last_obs: Optional[str]
    age_days: Optional[int]
    detail: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "logical": self.logical,
            "seriesId": self.series_id,
            "confidence": self.confidence,
            "status": self.status,
            "lastObs": self.last_obs,
            "ageDays": self.age_days,
            "detail": self.detail,
        }


def _check_one(logical: str, series_id: str, api_key: str) -> SeriesCheck:
    conf = confidence_for(series_id)
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    try:
        r = requests.get(_FRED_OBS_URL, params=params, timeout=_TIMEOUT)
    except Exception as exc:  # network
        return SeriesCheck(logical, series_id, conf, "error", None, None, str(exc)[:200])

    if r.status_code == 400 or r.status_code == 404:
        # FRED returns 400 with "series does not exist" for unknown ids.
        return SeriesCheck(logical, series_id, conf, "404", None, None,
                           r.json().get("error_message", r.text[:200]) if _is_json(r) else r.text[:200])
    if r.status_code != 200:
        return SeriesCheck(logical, series_id, conf, "error", None, None,
                           f"HTTP {r.status_code}")

    obs = (r.json().get("observations") or [])
    valid = [o for o in obs if o.get("value") not in (None, ".", "")]
    if not valid:
        return SeriesCheck(logical, series_id, conf, "stale", None, None, "no numeric observations")
    last = valid[0]["date"]
    age = (datetime.now(timezone.utc).date() - datetime.strptime(last, "%Y-%m-%d").date()).days
    status = "stale" if age > _STALE_DAYS else "ok"
    return SeriesCheck(logical, series_id, conf, status, last, age,
                       "" if status == "ok" else f"last obs {last} is {age}d old")


def _is_json(resp: requests.Response) -> bool:
    return "application/json" in resp.headers.get("content-type", "")


def validate_series(api_key: Optional[str] = None) -> List[SeriesCheck]:
    """Check every FRED ID in the series map. Ordered worst-status first."""
    api_key = api_key or FRED_API_KEY or os.getenv("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set; cannot validate series.")
    order = {"404": 0, "error": 1, "stale": 2, "ok": 3}
    checks = [_check_one(name, sid, api_key) for name, sid in sorted(all_series_ids().items())]
    checks.sort(key=lambda c: (order.get(c.status, 9), c.confidence, c.logical))
    return checks


def health_summary(checks: List[SeriesCheck]) -> Dict[str, object]:
    by_status: Dict[str, int] = {}
    for c in checks:
        by_status[c.status] = by_status.get(c.status, 0) + 1
    return {
        "counts": by_status,
        "failing": [c.as_dict() for c in checks if c.status in ("404", "error")],
        "stale": [c.as_dict() for c in checks if c.status == "stale"],
        "ok": by_status.get("ok", 0),
        "total": len(checks),
    }
