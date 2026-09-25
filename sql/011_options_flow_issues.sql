-- Apply explicitly before enabling editorial archive endpoints. No vendor rows touched.
CREATE TABLE IF NOT EXISTS options_flow_issue_revisions (
    id UUID PRIMARY KEY,
    session_date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    correction_of UUID REFERENCES options_flow_issue_revisions(id),
    correction_reason TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    chart_spec JSONB NOT NULL,
    evidence JSONB NOT NULL,
    evidence_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS options_flow_issues_session ON options_flow_issue_revisions(session_date, created_at DESC);
CREATE OR REPLACE FUNCTION options_flow_issue_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Issue revisions are immutable; insert a correction revision'; END;
$$;
DROP TRIGGER IF EXISTS options_flow_issue_immutable ON options_flow_issue_revisions;
CREATE TRIGGER options_flow_issue_immutable BEFORE UPDATE OR DELETE ON options_flow_issue_revisions
FOR EACH ROW EXECUTE FUNCTION options_flow_issue_immutable();
