"""Publish simple VIX / VIX3M research; no option-chain downloads."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psycopg.types.json import Jsonb
from api.db import get_connection
from api.services.volatility_overview import build_overview

def main():
    payload = build_overview()
    if payload['stale'] or payload['latest']['ratio'] is None:
        raise RuntimeError('Completed VIX / VIX3M session missing; existing snapshot retained')
    with get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS volatility_overview_snapshots (as_of date PRIMARY KEY, payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())")
        conn.execute("INSERT INTO volatility_overview_snapshots(as_of,payload) VALUES (%s,%s) ON CONFLICT(as_of) DO UPDATE SET payload=excluded.payload,updated_at=clock_timestamp()", (payload['as_of'],Jsonb(payload)))
    print('Published VIX overview: '+payload['as_of']+'; source='+payload['source'],flush=True)

if __name__ == '__main__':
    main()
