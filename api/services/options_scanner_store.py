"""Durable scanner resume state and read-only board/history queries."""
from pathlib import Path
import hashlib
import json
from psycopg.types.json import Jsonb
from api.db import get_connection

DDL = Path(__file__).resolve().parents[2] / 'sql' / '013_options_scanner.sql'

def ensure_schema():
    with get_connection() as conn:
        conn.execute(DDL.read_text(encoding='utf-8-sig'))

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]

def published_universe():
    with get_connection() as conn:
        row=conn.execute("""SELECT run_date,payload FROM stock_intelligence_runs WHERE is_published AND status='published'
          ORDER BY run_date DESC,as_of_timestamp DESC LIMIT 1""").fetchone()
    if not row:
        raise RuntimeError('No published stock-ranking universe')
    rows=row['payload'].get('rows') or row['payload'].get('rankings') or []
    symbols=list(dict.fromkeys(str(r.get('ticker') or r.get('symbol') or '').strip().upper() for r in rows))
    return symbols, str(row['run_date'])

def start_run(day,profile,symbols,metadata):
    uh=fingerprint(sorted(symbols))
    run_id=fingerprint([str(day),profile,uh])
    with get_connection() as conn:
        conn.execute("""INSERT INTO options_scanner_runs(id,market_date,profile,universe_hash,universe,metadata)
          VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET
          status='running',updated_at=clock_timestamp(),metadata=options_scanner_runs.metadata || excluded.metadata""",
          (run_id,day,profile,uh,Jsonb(symbols),Jsonb(metadata)))
    return run_id

def symbol_states(run_id):
    with get_connection() as conn:
        rows=conn.execute('SELECT ticker,status,payload,diagnostics FROM options_scanner_symbols WHERE run_id=%s',(run_id,)).fetchall()
    return {r['ticker']:r for r in rows}

def save_symbol(run_id,ticker,status,payload=None,diagnostics=None):
    with get_connection() as conn:
        conn.execute("""INSERT INTO options_scanner_symbols(run_id,ticker,status,payload,diagnostics)
          VALUES(%s,%s,%s,%s,%s) ON CONFLICT(run_id,ticker) DO UPDATE SET
          status=excluded.status,payload=excluded.payload,diagnostics=excluded.diagnostics,updated_at=clock_timestamp()
          WHERE options_scanner_symbols.status <> 'completed' OR excluded.status='completed'""",
          (run_id,ticker,status,Jsonb(payload or {}),Jsonb(diagnostics or {})))

def finish_run(run_id,status,metadata):
    with get_connection() as conn:
        conn.execute("UPDATE options_scanner_runs SET status=%s,metadata=metadata || %s,updated_at=clock_timestamp() WHERE id=%s",
          (status,Jsonb(metadata),run_id))

def history(ticker,profile,before_date,limit=20):
    with get_connection() as conn:
        rows=conn.execute("""SELECT payload FROM (
          SELECT DISTINCT ON(r.market_date) r.market_date,s.payload,r.updated_at
          FROM options_scanner_symbols s JOIN options_scanner_runs r ON r.id=s.run_id
          WHERE s.ticker=%s AND s.status='completed' AND r.profile=%s AND r.market_date<%s
            AND NOT COALESCE((r.metadata->>'benchmark')::boolean,false)
          ORDER BY r.market_date DESC,r.updated_at DESC
        ) h ORDER BY market_date DESC LIMIT %s""",(ticker,profile,before_date,limit)).fetchall()
    return [r['payload'] for r in rows]

def latest_board(profile=None, session=None):
    with get_connection() as conn:
        row=conn.execute("""SELECT * FROM options_scanner_runs
          WHERE (%s::text IS NULL OR profile=%s) AND (%s::date IS NULL OR market_date=%s::date)
          AND NOT COALESCE((metadata->>'benchmark')::boolean,false)
          ORDER BY market_date DESC,updated_at DESC LIMIT 1""",(profile,profile,session,session)).fetchone()
    if not row:
        return None
    states=symbol_states(row['id'])
    groups={key:[] for key in ['completed','pending','failed','excluded']}
    for symbol in row['universe']:
        state=states.get(symbol,{}).get('status','pending')
        groups[state if state in groups else 'pending'].append(symbol)
    return {'run':{**row,'market_date':str(row['market_date']),'updated_at':row['updated_at'].isoformat()},
      'symbols':[states[s]['payload'] for s in row['universe'] if s in states and states[s]['status']=='completed'],
      **groups,'diagnostics':{s:r['diagnostics'] for s,r in states.items()},
      'coverage':{'total':len(row['universe']),**{k:len(v) for k,v in groups.items()}}}
