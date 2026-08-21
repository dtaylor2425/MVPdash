from __future__ import annotations

import json
import math
import os
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

try:
    from api.services.volatility_engine import build_natr_ladder, build_volatility_snapshot, fetch_spx_history
except Exception:
    from volatility_engine import build_natr_ladder, build_volatility_snapshot, fetch_spx_history

router = APIRouter(prefix="/api/volatility", tags=["volatility"])
RANGE_DAYS = {"1m": 35, "3m": 110, "6m": 200, "1y": 370}
ALL_HORIZONS = (5, 10, 20, 30, 40, 50)
SHORT_HORIZONS = (5, 10, 20)


def _db_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def _connect():
    url = _db_url()
    if not url:
        return None
    import psycopg
    from psycopg.rows import dict_row
    return psycopg.connect(url, row_factory=dict_row)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _round(value: Any, digits: int = 3) -> Optional[float]:
    x = _float(value)
    return None if x is None else round(x, digits)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _natr_lookup(natr_json: Any) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for row in _as_list(natr_json):
        if isinstance(row, dict) and row.get("horizon") is not None:
            try:
                result[int(row.get("horizon"))] = row
            except Exception:
                pass
    return result


def _fetch_rows(days: Optional[int] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = _connect()
    if conn is None:
        return []
    where = ""
    params: List[Any] = []
    if days is not None:
        where = "WHERE ts >= %s"
        params.append(datetime.now(timezone.utc) - timedelta(days=days))
    lim = ""
    if limit is not None:
        lim = "LIMIT %s"
        params.append(limit)
    sql = f"""
        SELECT ts, spx, source_symbol, is_spy_fallback,
               iv_7d, iv_14d, iv_30d, iv_60d,
               spread_7d_14d, spread_14d_30d, spread_30d_60d,
               natr_json, regime, payload
        FROM volatility_snapshots
        {where}
        ORDER BY ts {'DESC' if limit is not None else 'ASC'}
        {lim}
    """
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = list(cur.fetchall())
    if limit is not None:
        rows.reverse()
    return rows


def _row_to_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _as_dict(row.get("payload")).copy()
    if not payload:
        payload = {
            "timestamp": _iso(row.get("ts")),
            "underlying": "^GSPC",
            "spx": _round(row.get("spx"), 2),
            "natr": _as_list(row.get("natr_json")),
            "implied_volatility": {
                "source_symbol": row.get("source_symbol"),
                "is_spy_fallback": bool(row.get("is_spy_fallback")),
                "rate_proxy": None,
                "curve": {"7d": row.get("iv_7d"), "14d": row.get("iv_14d"), "30d": row.get("iv_30d"), "60d": row.get("iv_60d")},
                "spreads": {"7d_14d": row.get("spread_7d_14d"), "14d_30d": row.get("spread_14d_30d"), "30d_60d": row.get("spread_30d_60d")},
                "expiry_points": [], "warnings": [],
            },
            "divergence": {"implied": "UNAVAILABLE", "realized": "UNAVAILABLE", "regime": row.get("regime")},
            "data_quality": {"options_ok": row.get("iv_7d") is not None, "options_error": None, "note": "Published volatility snapshot loaded from Postgres."},
        }
    payload["timestamp"] = _iso(row.get("ts")) or payload.get("timestamp")
    payload["spx"] = _round(row.get("spx"), 2) if row.get("spx") is not None else payload.get("spx")
    payload["natr"] = _as_list(row.get("natr_json")) or payload.get("natr", [])
    iv = payload.get("implied_volatility") or {}
    curve = iv.get("curve") or {}
    spreads = iv.get("spreads") or {}
    for key, col in (("7d", "iv_7d"), ("14d", "iv_14d"), ("30d", "iv_30d"), ("60d", "iv_60d")):
        if row.get(col) is not None:
            curve[key] = _round(row.get(col), 3)
    for key, col in (("7d_14d", "spread_7d_14d"), ("14d_30d", "spread_14d_30d"), ("30d_60d", "spread_30d_60d")):
        if row.get(col) is not None:
            spreads[key] = _round(row.get(col), 3)
    iv["curve"] = curve
    iv["spreads"] = spreads
    iv["source_symbol"] = row.get("source_symbol") or iv.get("source_symbol")
    iv["is_spy_fallback"] = bool(row.get("is_spy_fallback")) or bool(iv.get("is_spy_fallback"))
    payload["implied_volatility"] = iv
    return payload


def _row_to_obs(row: Dict[str, Any]) -> Dict[str, Any]:
    natr = _natr_lookup(row.get("natr_json"))
    obs = {
        "timestamp": _iso(row.get("ts")),
        "spx": _round(row.get("spx"), 2),
        "iv_7d": _round(row.get("iv_7d"), 3),
        "iv_14d": _round(row.get("iv_14d"), 3),
        "iv_30d": _round(row.get("iv_30d"), 3),
        "iv_60d": _round(row.get("iv_60d"), 3),
        "spread_7d_14d": _round(row.get("spread_7d_14d"), 3),
        "spread_14d_30d": _round(row.get("spread_14d_30d"), 3),
        "spread_30d_60d": _round(row.get("spread_30d_60d"), 3),
        "regime": row.get("regime"),
        "source_symbol": row.get("source_symbol"),
        "is_spy_fallback": bool(row.get("is_spy_fallback")),
    }
    for h in ALL_HORIZONS:
        item = natr.get(h, {})
        obs[f"natr_{h}d"] = _round(item.get("natr"), 4)
        obs[f"natr_{h}d_rising"] = bool(item.get("rising")) if item else None
        obs[f"natr_{h}d_regime"] = item.get("regime") if item else None
    return obs


def _payload_to_obs(payload: Dict[str, Any]) -> Dict[str, Any]:
    iv = payload.get("implied_volatility") or {}
    curve, spreads = iv.get("curve") or {}, iv.get("spreads") or {}
    natr = _natr_lookup(payload.get("natr"))
    obs = {
        "timestamp": payload.get("timestamp"),
        "spx": _round(payload.get("spx"), 2),
        "iv_7d": _round(curve.get("7d"), 3),
        "iv_14d": _round(curve.get("14d"), 3),
        "iv_30d": _round(curve.get("30d"), 3),
        "iv_60d": _round(curve.get("60d"), 3),
        "spread_7d_14d": _round(spreads.get("7d_14d"), 3),
        "spread_14d_30d": _round(spreads.get("14d_30d"), 3),
        "spread_30d_60d": _round(spreads.get("30d_60d"), 3),
        "source_symbol": iv.get("source_symbol"),
        "is_spy_fallback": bool(iv.get("is_spy_fallback")),
    }
    for h in ALL_HORIZONS:
        item = natr.get(h, {})
        obs[f"natr_{h}d"] = _round(item.get("natr"), 4)
        obs[f"natr_{h}d_rising"] = bool(item.get("rising")) if item else None
        obs[f"natr_{h}d_regime"] = item.get("regime") if item else None
    return obs


def _percentile(values: List[float], current: Optional[float]) -> Optional[float]:
    if current is None or not values:
        return None
    return round(100.0 * sum(1 for v in values if v <= current) / len(values), 1)


def _z(values: List[float], current: Optional[float]) -> Optional[float]:
    if current is None or len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return None if std <= 0 else round((current - mean) / std, 2)


def _past_value(obs: List[Dict[str, Any]], current_ts: Optional[datetime], field: str, days: int) -> Optional[float]:
    if current_ts is None:
        return None
    target = current_ts - timedelta(days=days)
    candidates = []
    fallback = []
    for row in obs:
        ts = _dt(row.get("timestamp"))
        val = _float(row.get(field))
        if ts is None or val is None or ts >= current_ts:
            continue
        fallback.append((ts, val))
        if ts <= target:
            candidates.append((ts, val))
    candidates = candidates or fallback
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _context(obs: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    current_obs = obs[-1] if obs else {}
    current = _float(current_obs.get(field))
    current_ts = _dt(current_obs.get("timestamp"))
    values = [_float(x.get(field)) for x in obs if _float(x.get(field)) is not None]
    one = _past_value(obs, current_ts, field, 1)
    five = _past_value(obs, current_ts, field, 5)
    return {
        "current": _round(current, 3),
        "change_1d": _round(current - one, 3) if current is not None and one is not None else None,
        "change_5d": _round(current - five, 3) if current is not None and five is not None else None,
        "percentile": _percentile(values, current),
        "z_score": _z(values, current),
        "sample_size": len(values),
    }


def _storm(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    horizons, expanding, contracting, flips = [], [], [], []
    for h in ALL_HORIZONS:
        rising = current.get(f"natr_{h}d_rising")
        prev_rising = previous.get(f"natr_{h}d_rising") if previous else None
        flipped = bool(rising is True and prev_rising is False)
        if rising is True:
            expanding.append(h)
        elif rising is False:
            contracting.append(h)
        if flipped:
            flips.append(h)
        horizons.append({
            "horizon": h,
            "natr": _round(current.get(f"natr_{h}d"), 4),
            "rising": rising,
            "regime": current.get(f"natr_{h}d_regime"),
            "flipped": flipped,
        })
    short = [x for x in horizons if x["horizon"] in SHORT_HORIZONS]
    exp_short = [x for x in short if x["rising"] is True]
    low_short = [x for x in short if x["natr"] is not None and x["natr"] < 0]
    high_short = [x for x in short if x["natr"] is not None and x["natr"] > 0]
    by_h = {x["horizon"]: x for x in short}
    if len(high_short) >= 2 and len(exp_short) >= 2:
        state = "HIGH VOLATILITY"
    elif len(exp_short) == 3:
        state = "CONFIRMED EXPANSION"
    elif by_h.get(5, {}).get("rising") is True and by_h.get(10, {}).get("rising") is True and len(low_short) >= 2:
        state = "EARLY WARNING"
    elif by_h.get(5, {}).get("rising") is True and by_h.get(5, {}).get("natr") is not None and by_h.get(5, {}).get("natr") < 0:
        state = "WATCH"
    elif len(low_short) >= 2 and len(exp_short) <= 1:
        state = "QUIET"
    else:
        state = "MIXED"
    return {
        "horizons": horizons,
        "expanding_horizons": expanding,
        "contracting_horizons": contracting,
        "short_horizon_expanding": [x["horizon"] for x in exp_short],
        "new_flips": flips,
        "transition_state": state,
        "expanding_count": len(expanding),
        "total_count": len(horizons),
        "short_summary": {"low_short_count": len(low_short), "positive_short_count": len(high_short)},
    }


def _setup(current: Dict[str, Any], ctx: Dict[str, Any], storm: Dict[str, Any]) -> Dict[str, Any]:
    spread = _float(current.get("spread_7d_14d"))
    pct = _float(ctx.get("percentile"))
    state = storm.get("transition_state") or "MIXED"
    short_values = [_float(current.get("natr_5d")), _float(current.get("natr_10d")), _float(current.get("natr_20d"))]
    short_values = [x for x in short_values if x is not None]
    low_count = sum(1 for x in short_values if x < 0)
    high_count = sum(1 for x in short_values if x > 0)
    exp_count = len(storm.get("short_horizon_expanding") or [])
    implied = "COMPRESSED" if (pct is not None and pct <= 20) or (spread is not None and spread < -0.5) else "FRONT-END BID" if (pct is not None and pct >= 80) or (spread is not None and spread > 0.5) else "NEUTRAL"
    realized = "HIGH" if high_count >= 2 else "LOW" if low_count >= 2 else "MIXED"
    direction = "EXPANDING" if exp_count >= 2 else "CONTRACTING" if exp_count <= 1 else "MIXED"
    term = "UNAVAILABLE" if spread is None else "CONTANGO" if spread < -0.25 else "INVERTED" if spread > 0.25 else "FLAT"
    signal = "NO VOL EXPANSION YET"
    body = "Realized volatility remains quiet and the short end of the IV curve has not begun repricing expansion."
    watch = ["5D N-ATR turns higher", "10D confirms expansion", "7D-14D spread begins rising", "20D N-ATR confirms"]
    if state == "WATCH":
        signal = "EARLY VOLATILITY WARNING"
        body = "5D realized volatility has begun expanding while short-end implied volatility remains compressed. Waiting for 10D confirmation."
        watch = ["10D confirms expansion", "7D-14D spread begins rising", "20D N-ATR confirms"]
    elif state == "EARLY WARNING" and implied == "COMPRESSED":
        signal = "IMPLIED / REALIZED DIVERGENCE"
        body = "Short-term realized volatility is expanding while the options market continues to price near-term calm."
        watch = ["7D-14D spread begins rising", "20D N-ATR confirms", "Front-end IV moves from compression to bid"]
    elif state == "EARLY WARNING":
        signal = "EARLY VOLATILITY EXPANSION"
        body = "5D and 10D realized volatility are expanding. Watch for 20D propagation."
        watch = ["20D N-ATR confirms", "Short-end IV follows realized expansion"]
    elif state == "CONFIRMED EXPANSION":
        signal = "VOLATILITY EXPANSION UNDERWAY"
        body = "5D, 10D, and 20D realized volatility are expanding together."
        watch = ["Does IV continue repricing higher?", "Does realized expansion persist?"]
    elif state == "HIGH VOLATILITY":
        signal = "HIGH VOLATILITY REGIME"
        body = "The majority of short-horizon realized volatility is above zero and expanding."
        watch = ["Does 5D begin contracting?", "Does the IV curve remain front-end bid?"]
    return {"implied_volatility": implied, "realized_volatility": realized, "realized_direction": direction, "term_structure": term, "storm_transition": state, "signal": signal, "signal_body": body, "watch_for": watch, "is_divergence": signal == "IMPLIED / REALIZED DIVERGENCE"}


def _data_status(payload: Dict[str, Any], live_error: Optional[str] = None) -> Dict[str, Any]:
    ts = _dt(payload.get("timestamp"))
    iv = payload.get("implied_volatility") or {}
    quality = payload.get("data_quality") or {}
    try:
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo("America/New_York")
    except Exception:
        eastern = timezone(timedelta(hours=-5))
    now_et = datetime.now(timezone.utc).astimezone(eastern)
    age = None if ts is None else max(0, int((datetime.now(timezone.utc) - ts).total_seconds() // 60))
    market_open = now_et.weekday() < 5 and time(9, 30) <= now_et.time() <= time(16, 0)
    if live_error:
        status = "ERROR"
    elif not market_open:
        status = "MARKET CLOSED"
    elif not quality.get("options_ok"):
        status = "ERROR"
    elif age is not None and age <= 20:
        status = "CURRENT"
    else:
        status = "STALE"
    source = "SPY PROXY" if iv.get("is_spy_fallback") else (str(iv.get("source_symbol")) + " Yahoo" if iv.get("source_symbol") else "Unavailable")
    return {"last_options_snapshot": payload.get("timestamp"), "last_options_snapshot_et": _iso(ts.astimezone(eastern)) if ts else None, "age_minutes": age, "source": source, "source_symbol": iv.get("source_symbol"), "is_spy_fallback": bool(iv.get("is_spy_fallback")), "status": status, "market_open": market_open, "live_error": live_error, "options_ok": bool(quality.get("options_ok"))}


def _enrich(payload: Dict[str, Any], history: List[Dict[str, Any]], source: str, live_error: Optional[str] = None) -> Dict[str, Any]:
    current = _payload_to_obs(payload)
    observations = [x for x in history if x.get("timestamp") and x.get("timestamp") != current.get("timestamp")]
    observations.append(current)
    observations.sort(key=lambda x: x.get("timestamp") or "")
    momentum = {k: _context(observations, k) for k in ("spread_7d_14d", "spread_14d_30d", "spread_30d_60d")}
    previous = observations[-2] if len(observations) >= 2 else None
    storm = _storm(current, previous)
    enriched = payload.copy()
    warnings = list(((enriched.get("implied_volatility") or {}).get("warnings") or []))
    if live_error:
        warnings.extend(["Live Yahoo rebuild failed. Showing the latest valid published snapshot.", live_error])
    enriched.update({
        "source": source,
        "spread_momentum": momentum,
        "spread_context": {"spread_7d_14d": momentum.get("spread_7d_14d", {})},
        "storm_transition": storm,
        "volatility_setup": _setup(current, momentum.get("spread_7d_14d", {}), storm),
        "data_status": _data_status(enriched, live_error),
        "warnings": warnings,
    })
    return enriched


@router.get("/snapshot")
def volatility_snapshot(live: bool = Query(False)):
    history = [_row_to_obs(r) for r in _fetch_rows(days=370)]
    if live:
        try:
            return _enrich(build_volatility_snapshot(), history, "live_yahoo")
        except Exception as exc:
            rows = _fetch_rows(limit=1)
            if not rows:
                raise HTTPException(status_code=503, detail=str(exc))
            return _enrich(_row_to_payload(rows[-1]), history, "postgres_snapshot", str(exc))
    rows = _fetch_rows(limit=1)
    if rows:
        return _enrich(_row_to_payload(rows[-1]), history, "postgres_snapshot")
    try:
        return _enrich(build_volatility_snapshot(), history, "live_yahoo_no_published_snapshot")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/history")
def volatility_history(range: str = Query("3m", pattern="^(1m|3m|6m|1y)$")):
    rows = _fetch_rows(days=RANGE_DAYS.get(range, 110))
    if not rows:
        rows = _fetch_rows(limit=500)
    observations = [_row_to_obs(r) for r in rows]
    return {"range": range, "requested_days": RANGE_DAYS.get(range, 110), "available_count": len(observations), "observations": observations}


@router.get("/natr")
def natr_only(map_mode: str = Query("Both", pattern="^(Both|ATR length|Norm length)$")):
    try:
        history = fetch_spx_history()
        return {"underlying": "^GSPC", "spx": round(float(history["Close"].iloc[-1]), 2), "map_mode": map_mode, "natr": build_natr_ladder(history, map_mode=map_mode)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
