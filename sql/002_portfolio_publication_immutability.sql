-- Apply before enabling the ledger publisher. Existing records are not changed.
BEGIN;
CREATE OR REPLACE FUNCTION protect_published_portfolio_run() RETURNS trigger AS $$
BEGIN
  IF OLD.status = 'published' THEN
    RAISE EXCEPTION 'Published portfolio records are immutable; append a new dated publication';
  END IF;
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS portfolio_run_immutable ON portfolio_runs;
CREATE TRIGGER portfolio_run_immutable BEFORE UPDATE OR DELETE ON portfolio_runs
FOR EACH ROW EXECUTE FUNCTION protect_published_portfolio_run();

CREATE OR REPLACE FUNCTION protect_published_portfolio_child() RETURNS trigger AS $$
DECLARE parent_id uuid;
BEGIN
  IF TG_OP <> 'INSERT' THEN
    parent_id := OLD.run_id;
    PERFORM 1 FROM portfolio_runs WHERE id = parent_id AND status = 'published' FOR SHARE;
    IF FOUND THEN RAISE EXCEPTION 'Published portfolio detail is immutable'; END IF;
  END IF;
  IF TG_OP <> 'DELETE' THEN
    parent_id := NEW.run_id;
    PERFORM 1 FROM portfolio_runs WHERE id = parent_id FOR SHARE;
    IF EXISTS (SELECT 1 FROM portfolio_runs WHERE id = parent_id AND status = 'published') THEN
      RAISE EXCEPTION 'Cannot add or move detail into a published portfolio';
    END IF;
    RETURN NEW;
  END IF;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS portfolio_positions_immutable ON portfolio_positions;
CREATE TRIGGER portfolio_positions_immutable BEFORE INSERT OR UPDATE OR DELETE ON portfolio_positions
FOR EACH ROW EXECUTE FUNCTION protect_published_portfolio_child();
DROP TRIGGER IF EXISTS portfolio_performance_immutable ON portfolio_performance;
CREATE TRIGGER portfolio_performance_immutable BEFORE INSERT OR UPDATE OR DELETE ON portfolio_performance
FOR EACH ROW EXECUTE FUNCTION protect_published_portfolio_child();
DROP TRIGGER IF EXISTS portfolio_rebalances_immutable ON portfolio_rebalances;
CREATE TRIGGER portfolio_rebalances_immutable BEFORE INSERT OR UPDATE OR DELETE ON portfolio_rebalances
FOR EACH ROW EXECUTE FUNCTION protect_published_portfolio_child();

-- Older publisher versions must not append a freshly recalculated past curve.
CREATE OR REPLACE FUNCTION require_append_only_portfolio_publication() RETURNS trigger AS $$
DECLARE prior portfolio_runs%ROWTYPE; previous_series jsonb; prefix jsonb; previous_trades jsonb;
BEGIN
  IF NEW.status <> 'published' OR NEW.strategy NOT IN ('stock_alpha', 'smid_growth') THEN RETURN NEW; END IF;
  IF NEW.payload #>> '{ledger,version}' IS DISTINCT FROM 'published-ledger-v1' THEN
    RAISE EXCEPTION 'Publication requires the append-only ledger publisher';
  END IF;
  SELECT * INTO prior FROM portfolio_runs
    WHERE strategy = NEW.strategy AND status = 'published' AND is_published = TRUE AND id <> NEW.id
    ORDER BY run_date DESC, published_at DESC LIMIT 1 FOR SHARE;
  IF NOT FOUND THEN RAISE EXCEPTION 'An audited opening publication is required'; END IF;
  IF NEW.run_date <= prior.run_date THEN RAISE EXCEPTION 'Publication date must advance'; END IF;
  previous_series := COALESCE(prior.payload #> '{performance,series}', '[]'::jsonb);
  SELECT COALESCE(jsonb_agg(value ORDER BY ord), '[]'::jsonb) INTO prefix
    FROM jsonb_array_elements(COALESCE(NEW.payload #> '{performance,series}', '[]'::jsonb)) WITH ORDINALITY AS x(value, ord)
    WHERE ord <= jsonb_array_length(previous_series);
  IF prefix IS DISTINCT FROM previous_series THEN RAISE EXCEPTION 'Published curve history changed'; END IF;
  previous_trades := COALESCE(prior.payload #> '{ledger,trades}', '[]'::jsonb);
  SELECT COALESCE(jsonb_agg(value ORDER BY ord), '[]'::jsonb) INTO prefix
    FROM jsonb_array_elements(COALESCE(NEW.payload #> '{ledger,trades}', '[]'::jsonb)) WITH ORDINALITY AS x(value, ord)
    WHERE ord <= jsonb_array_length(previous_trades);
  IF prefix IS DISTINCT FROM previous_trades THEN RAISE EXCEPTION 'Published trade history changed'; END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS portfolio_publication_append_only ON portfolio_runs;
CREATE TRIGGER portfolio_publication_append_only BEFORE INSERT OR UPDATE ON portfolio_runs
FOR EACH ROW EXECUTE FUNCTION require_append_only_portfolio_publication();
COMMIT;
