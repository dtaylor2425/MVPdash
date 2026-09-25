-- Aggregate product interactions only. No visitor IDs, emails, URLs or IPs.
CREATE TABLE IF NOT EXISTS research_usage_daily (
    day date NOT NULL,
    event text NOT NULL,
    page text NOT NULL,
    source text NOT NULL,
    count bigint NOT NULL DEFAULT 0 CHECK (count >= 0),
    PRIMARY KEY (day, event, page, source)
);
