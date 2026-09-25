"""Session-aligned editorial reads, independent of frozen research datasets."""
from datetime import date
import json
from api.services import options_flow_store as store
from api.services.options_flow_calendar import previous_trading_day
from api.services.options_flow_profile import collection_profile, profile_identity

ROWS_SQL = """
SELECT DISTINCT ON (s.ticker) s.*, r.status AS run_status,
       r.finished_at, r.config AS run_config
FROM options_flow_symbol_snapshots s JOIN options_flow_runs r ON r.id=s.run_id
WHERE s.mode='full_flow' AND r.status IN ('success','partial')
  AND s.market_date=%(session)s AND (%(ticker)s::text IS NULL OR s.ticker=%(ticker)s)
ORDER BY s.ticker,
 (s.source='historical_backfill' OR s.payload->'publication'->>'state'='final') DESC NULLS LAST,
 (s.source='live') DESC, s.as_of_timestamp DESC, s.created_at DESC, s.id DESC
"""


def sessions(conn):
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT s.market_date FROM options_flow_symbol_snapshots s
            JOIN options_flow_runs r ON r.id=s.run_id
            WHERE s.mode='full_flow' AND r.status IN ('success','partial')
            ORDER BY s.market_date DESC LIMIT 260""")
        return [{"sessionDate": r["market_date"].isoformat()} for r in cur.fetchall()]


def provenance(row):
    payload = store._payload(row)
    publication = payload.get("publication") or {}
    historical = row.get("source") == "historical_backfill"
    cfg = row.get("run_config") or {}
    cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    bounds = publication.get("collectionBounds") or {k: cfg.get(k) for k in ("sessionStart", "sessionEnd", "maxDte", "strikeRange")}
    state = "historical" if historical else publication.get("state", "partial")
    return {"snapshotId": str(row["id"]), "runId": str(row["run_id"]),
            "sessionDate": row["market_date"].isoformat(), "source": row.get("source"),
            "mode": row.get("mode"), "methodologyVersion": row.get("methodology_version") or payload.get("methodologyVersion"),
            "publicationState": state, "comparable": historical or state == "final",
            "collectionBounds": bounds,
            "collectionProfile": collection_profile(cfg),
            "profileIdentity": profile_identity(row.get("methodology_version") or payload.get("methodologyVersion"), cfg),
            "runStatus": row.get("run_status")}


def rows_for(conn, session, ticker=None):
    with conn.cursor() as cur:
        cur.execute(ROWS_SQL, {"session": session, "ticker": ticker})
        return cur.fetchall()


def ticker_view(row, heavy=False):
    payload = dict(store._payload(row))
    if not heavy:
        payload = {k: v for k, v in payload.items() if k not in store.HEAVY_KEYS}
    return {**payload, "ticker": row["ticker"], "group": row["group_name"],
            "asOf": store._iso(row["as_of_timestamp"]),
            "publishedAt": store._iso(row.get("finished_at") or row["created_at"]),
            "carriedForward": False, "provenance": provenance(row)}


METRICS = {"grossPremium": ("premium", "gross", "USD"),
           "directionalPremium": ("premium", "netDirectional", "USD"),
           "atmIv": ("iv", "atm", "fraction"),
           "skew": ("iv", "skew25d30d", "fraction"),
           "directionalImbalance": ("delta", "ratio", "ratio"),
           "zeroDteShare": ("dte", "zeroDteShare", "ratio")}


def comparisons(current, prior):
    previous = {r["ticker"]: r for r in prior}
    output = []
    for row in current:
        old = previous.get(row["ticker"])
        p, q = provenance(row), provenance(old) if old else {}
        comparable = bool(old and p["comparable"] and q.get("comparable")
                          and p["methodologyVersion"] is not None
                          and p["profileIdentity"] is not None
                          and p["methodologyVersion"] == q.get("methodologyVersion")
                          and p["collectionProfile"] == q.get("collectionProfile"))
        changes = {}
        for key, (section, field, unit) in METRICS.items():
            a = (store._payload(row).get(section) or {}).get(field)
            b = (store._payload(old).get(section) or {}).get(field) if old else None
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
            changes[key] = {"current": a, "previous": b, "delta": a-b if comparable and numeric else None, "unit": unit}
        def term(payload):
            iv = payload.get("iv") or {}
            a, b = iv.get("iv60d"), iv.get("iv30d")
            return a-b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        current_payload, old_payload = store._payload(row), store._payload(old) if old else {}
        for key, a, b, unit in [("spot", current_payload.get("spot"), old_payload.get("spot"), "USD underlying snapshot reference"),
                                 ("termSpread", term(current_payload), term(old_payload), "fraction")]:
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
            changes[key] = {"current": a, "previous": b, "delta": a-b if comparable and numeric else None, "unit": unit}
        output.append({"ticker": row["ticker"], "snapshotId": p["snapshotId"], "previousSnapshotId": q.get("snapshotId"), "comparable": comparable,
                       "reason": None if comparable else "Missing, partial, or methodology-incompatible comparison",
                       "metrics": changes})
    return output


def latest(conn, universe, cfg, session=None, now=None):
    available = sessions(conn)
    if session is None:
        session = date.fromisoformat(available[0]["sessionDate"]) if available else None
    if session is None:
        return {"run": None}
    rows = rows_for(conn, session)
    tracked = [t for ts in universe.values() for t in ts]
    rows = [r for r in rows if r["ticker"] in tracked]
    if not rows:
        return {"run": None}
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM options_flow_runs WHERE id=%s", (rows[0]["run_id"],))
        run = cur.fetchone()
    states = [provenance(r) for r in rows]
    complete = len(rows) == len(tracked) and all(p["comparable"] for p in states)
    state = "historical" if all(p["source"] == "historical_backfill" for p in states) else ("final" if complete else "partial")
    prior_date = previous_trading_day(session)
    output = {"run": store._run_meta(run, cfg, now), "tickers": [ticker_view(r) for r in rows],
              "groups": [{"name": g, "tickers": ts} for g, ts in universe.items()],
              "missing": [t for t in tracked if t not in {r["ticker"] for r in rows}],
              "availableSessions": available,
              "session": {"sessionDate": session.isoformat(), "state": state, "comparable": complete,
                          "coverage": {"available": len(rows), "tracked": len(tracked)},
                          "snapshotIds": [p["snapshotId"] for p in states]},
              "collectionStatus": store.fetch_status(conn, universe, cfg, now),
              "dailyBrief": {"previousSession": prior_date.isoformat(),
                             "tickers": comparisons(rows, rows_for(conn, prior_date))}}
    store.attach_market_activity(conn, output, as_of_date=session)
    factor = output.get("marketActivity")
    previous_factor = next((p for p in (factor or {}).get("history", []) if p["date"] == prior_date.isoformat()), None)
    current_factor = next((p for p in (factor or {}).get("history", []) if p["date"] == session.isoformat()), None)
    comparable_factor = bool(factor and previous_factor and current_factor
                             and factor.get("value") is not None and previous_factor.get("value") is not None
                             and current_factor.get("availableTickers") == previous_factor.get("availableTickers")
                             and current_factor.get("collectionProfiles") == previous_factor.get("collectionProfiles"))
    output["dailyBrief"]["marketActivity"] = {
        "comparable": comparable_factor,
        "reason": None if comparable_factor else "Missing activity observation or changed ETF coverage/collection profile",
        "current": factor.get("value") if factor else None,
        "previous": previous_factor.get("value") if previous_factor else None,
        "delta": factor["value"]-previous_factor["value"] if comparable_factor else None,
        "percentile": factor.get("percentile") if factor else None,
        "previousPercentile": previous_factor.get("percentile") if previous_factor else None,
        "breadth": factor.get("breadth") if factor else None,
        "previousBreadth": previous_factor.get("breadth") if previous_factor else None,
        "elevatedSessionStreak": factor.get("elevatedSessionStreak") if factor else None,
    }
    from api.services.options_flow_issues import freeze_evidence
    output["session"]["evidenceSha256"] = freeze_evidence(output, output["session"]["snapshotIds"])[1]
    return output


def detail(conn, ticker, include_trades=True, session=None):
    if session is None:
        available = sessions(conn)
        session = date.fromisoformat(available[0]["sessionDate"]) if available else None
    rows = rows_for(conn, session, ticker) if session else []
    return {"ticker": store.strip_heavy(ticker_view(rows[0], heavy=True), include_trades),
            "sessionDate": session.isoformat()} if rows else None
