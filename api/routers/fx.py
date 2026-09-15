"""
api/routers/fx.py  (spec section 6)

Read-only FX currency-strength routes. These serve the latest published
snapshot and never call FRED or Frankfurter. Response field names are a
contract the frontend spec depends on.

    GET /api/fx/snapshot
    GET /api/fx/currency/{code}
    GET /api/fx/pairs?base=USD
    GET /api/fx/status
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.fx import breadth as breadthmod
from src.fx import frankfurter as fxmod
from api.auth_deps import optional_account, require_account

router = APIRouter(prefix="/api/fx", tags=["fx"])

_DISK_SNAPSHOT = Path("data/cache/fx_snapshot_latest.json")
_STALE_AFTER_DAYS = 8  # weekly refresh + slack
_ANON_CURRENCY_LIMIT = 5  # matrix: anonymous sees first 5 currencies, no pairs


# ---------------------------------------------------------------------------
# snapshot loading
# ---------------------------------------------------------------------------
def _db_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def _load_from_db() -> Optional[Dict[str, Any]]:
    url = _db_url()
    if not url:
        return None
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT as_of, observation_date, created_at, payload
                    FROM fx_snapshots
                    ORDER BY as_of DESC, created_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload.setdefault("asOf", row["as_of"].isoformat())
    payload.setdefault("observationDate", row["observation_date"].isoformat())
    return payload


def _load_from_disk() -> Optional[Dict[str, Any]]:
    if not _DISK_SNAPSHOT.exists():
        return None
    try:
        return json.loads(_DISK_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return None


def _apply_staleness(payload: Dict[str, Any]) -> Dict[str, Any]:
    as_of = payload.get("asOf")
    try:
        as_of_date = date.fromisoformat(as_of)
        age = (datetime.now(timezone.utc).date() - as_of_date).days
    except Exception:
        age = None
    if age is not None and age > _STALE_AFTER_DAYS:
        payload["stale"] = True
        payload["staleSince"] = as_of
    else:
        payload.setdefault("stale", False)
    payload["ageDays"] = age
    return payload


def _latest_snapshot() -> Dict[str, Any]:
    payload = _load_from_db() or _load_from_disk()
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="No FX snapshot published yet. Run jobs/fx_snapshot_job.py first.",
        )
    return _apply_staleness(payload)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@router.get("/snapshot")
def fx_snapshot(user: Optional[Dict[str, Any]] = Depends(optional_account)) -> Dict[str, Any]:
    snap = dict(_latest_snapshot())
    if user is None:
        currencies = snap.get("currencies", [])
        snap["total"] = len(currencies)
        snap["currencies"] = currencies[:_ANON_CURRENCY_LIMIT]
        snap.pop("pairs", None)
        snap["truncated"] = True
    else:
        snap["total"] = len(snap.get("currencies", []))
        snap["truncated"] = False
    return snap


@router.get("/currency/{code}")
def fx_currency(code: str, user: Optional[Dict[str, Any]] = Depends(optional_account)) -> Dict[str, Any]:
    code = (code or "").upper().strip()
    snap = _latest_snapshot()
    entry = next((c for c in snap.get("currencies", []) if c.get("code") == code), None)
    if entry is None:
        known = [c.get("code") for c in snap.get("currencies", [])]
        raise HTTPException(status_code=404, detail=f"Unknown currency '{code}'. Known: {known}")

    out = dict(entry)
    out["asOf"] = snap.get("asOf")
    out["observationDate"] = snap.get("observationDate")
    out["stale"] = snap.get("stale", False)
    out["source"] = snap.get("source")
    if entry.get("scored") and user is not None:
        out["pairs"] = _mirror_for_base(snap.get("pairs", []), code)
        out["reconciliation"] = snap.get("reconciliation") if code == "USD" else None
    out["meta"] = {
        "componentWeights": snap.get("meta", {}).get("componentWeights"),
        "methodology": snap.get("meta", {}).get("methodology"),
    }
    return out


@router.get("/pairs")
def fx_pairs(
    base: str = Query(..., description="Base currency, e.g. USD"),
    user: Dict[str, Any] = Depends(require_account),
) -> Dict[str, Any]:
    base = (base or "").upper().strip()
    snap = _latest_snapshot()
    universe = snap.get("meta", {}).get("universe", [])
    if base not in universe:
        raise HTTPException(status_code=404, detail=f"Base '{base}' is not a scored currency. {universe}")
    return {
        "asOf": snap.get("asOf"),
        "observationDate": snap.get("observationDate"),
        "stale": snap.get("stale", False),
        "base": base,
        "pairs": _mirror_for_base(snap.get("pairs", []), base),
    }


@router.get("/breadth")
def fx_breadth(
    base: str = Query(..., description="Base currency, e.g. USD"),
    horizon: int = Query(breadthmod.DEFAULT_HORIZON, ge=2, le=252),
    universe: str = Query(breadthmod.DEFAULT_UNIVERSE, pattern="^(majors|g10)$"),
    trend_window: int = Query(breadthmod.DEFAULT_TREND_WINDOW, ge=5, le=252, alias="trendWindow"),
    threshold: float = Query(breadthmod.DEFAULT_THRESHOLD, ge=0.0, le=0.1),
    user: Dict[str, Any] = Depends(require_account),
) -> Dict[str, Any]:
    """
    Breadth & attribution (spec 4A) -- an independent model from /snapshot's
    composite score. Recomputed in-process from the Frankfurter fixing table
    the nightly job already cached to disk; this route never calls
    Frankfurter (or FRED) itself, so custom base/horizon/universe combos stay
    cheap without violating the "never fetch on request" rule.
    """
    base = (base or "").upper().strip()
    universe_list = breadthmod.UNIVERSES.get(universe, breadthmod.MAJORS)
    if base not in universe_list:
        raise HTTPException(
            status_code=404,
            detail=f"'{base}' is not in the {universe} universe: {universe_list}",
        )

    fx = fxmod.load_cached()
    if fx is not None:
        snap = breadthmod.compute_breadth_snapshot(
            fx, universe_name=universe, horizon=horizon,
            trend_window=trend_window, threshold=threshold,
        )
        source = "cached_frankfurter_disk"
    else:
        snap = _latest_snapshot().get("breadth") or {}
        source = "embedded_snapshot"

    # A missing/unavailable entry is real information, not an error -- return
    # it with the explicit INSUFFICIENT_DATA verdict rather than a 503, so the
    # caller can render "not enough data" instead of falling through to
    # whatever its error handler does with a failed request (fix-list item 8).
    entry = (snap.get("byBase") or {}).get(base) or {
        "available": False,
        "verdict": breadthmod.INSUFFICIENT_DATA,
    }
    return {
        "universe": snap.get("universe", universe),
        "horizon": snap.get("horizon", horizon),
        "trendWindow": snap.get("trendWindow", trend_window),
        "threshold": snap.get("threshold", threshold),
        "base": base,
        "strengths": snap.get("strengths", {}),
        **entry,
        "history": (snap.get("history") or {}).get(base, {}),
        "source": source,
    }


@router.get("/status")
def fx_status() -> Dict[str, Any]:
    payload = _load_from_db() or _load_from_disk()
    if payload is None:
        return {"status": "empty", "route": "/api/fx", "detail": "no snapshot published"}
    payload = _apply_staleness(payload)
    meta = payload.get("meta", {})
    return {
        "status": "stale" if payload.get("stale") else "ok",
        "route": "/api/fx",
        "asOf": payload.get("asOf"),
        "observationDate": payload.get("observationDate"),
        "ageDays": payload.get("ageDays"),
        "source": "postgres" if _load_from_db() is not None else "disk",
        "incompleteCurrencies": meta.get("incompleteCurrencies", []),
        "droppedComponents": meta.get("droppedComponents", []),
        "reconciliation": payload.get("reconciliation", {}).get("agreement"),
        "seriesHealth": meta.get("seriesHealth", {}).get("counts")
        if isinstance(meta.get("seriesHealth"), dict) else None,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _display(z: float) -> float:
    return round(max(0.0, min(100.0, 50.0 + 12.5 * z)), 1)


def _mirror_for_base(pairs: List[Dict[str, Any]], base: str) -> List[Dict[str, Any]]:
    """The stored triangle holds each pair once; orient every pair with `base` first."""
    out: List[Dict[str, Any]] = []
    for p in pairs:
        if p.get("base") == base:
            out.append(dict(p))
        elif p.get("quote") == base:
            z = -float(p.get("z", 0.0))
            out.append({
                "base": base,
                "quote": p.get("base"),
                "z": round(z, 4),
                "display": _display(z),
                "flags": p.get("flags", {}),
            })
    out.sort(key=lambda x: x["z"], reverse=True)
    return out
