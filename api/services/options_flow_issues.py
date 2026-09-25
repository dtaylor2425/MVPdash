"""Immutable editorial evidence, copied independently of mutable analytical tables."""
import hashlib
import json
import uuid
from psycopg.types.json import Jsonb


class SnapshotConflict(ValueError):
    pass


def freeze_evidence(snapshot, expected_ids):
    actual = snapshot.get("session", {}).get("snapshotIds", [])
    if not actual or sorted(map(str, actual)) != sorted(map(str, expected_ids)):
        raise SnapshotConflict("Snapshot selection changed; refresh and review before freezing")
    # Exports must not disclose operational diagnostics or raw prints.
    evidence = {k: snapshot.get(k) for k in ("session", "groups", "missing", "marketActivity", "dailyBrief")}
    evidence["session"] = {k: v for k, v in (snapshot.get("session") or {}).items() if k != "evidenceSha256"}
    evidence["tickers"] = [{k: v for k, v in t.items() if k not in ("largeTrades", "intraday")}
                           for t in snapshot.get("tickers", [])]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(encoded), hashlib.sha256(encoded.encode()).hexdigest()


def create(conn, request, snapshot):
    evidence, digest = freeze_evidence(snapshot, request.snapshotIds)
    if request.expectedEvidenceSha256 != digest:
        raise SnapshotConflict("Analytical evidence changed; refresh and review before freezing")
    identity = str(uuid.uuid4())
    with conn.cursor() as cur:
        if request.correctionOf:
            cur.execute("SELECT session_date FROM options_flow_issue_revisions WHERE id=%s", (request.correctionOf,))
            prior = cur.fetchone()
            if not prior or prior["session_date"] != request.session:
                raise ValueError("Correction must reference an existing issue for this session")
            if not request.correctionReason or not request.correctionReason.strip():
                raise ValueError("A correction reason is required")
        cur.execute("""INSERT INTO options_flow_issue_revisions
            (id,session_date,correction_of,correction_reason,title,body,chart_spec,evidence,evidence_sha256)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id,created_at,evidence_sha256""",
            (identity, request.session, request.correctionOf, request.correctionReason,
             request.title, request.body, Jsonb(request.chartSpec), Jsonb(evidence), digest))
        row = cur.fetchone()
    conn.commit()
    return {"id": str(row["id"]), "createdAt": row["created_at"].isoformat(), "evidenceSha256": digest}


def list_issues(conn, session):
    with conn.cursor() as cur:
        cur.execute("""SELECT id,session_date,created_at,correction_of,correction_reason,title,evidence_sha256
            FROM options_flow_issue_revisions WHERE (%s::date IS NULL OR session_date=%s)
            ORDER BY created_at DESC LIMIT 100""", (session, session))
        return cur.fetchall()


def get_issue(conn, identity):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM options_flow_issue_revisions WHERE id=%s", (identity,))
        return cur.fetchone()
