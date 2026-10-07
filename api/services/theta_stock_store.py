"""Snapshot-only stock participation storage. No vendor imports or calls in API."""
from pathlib import Path
import re
from psycopg.types.json import Jsonb
from api.db import get_connection

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
DDL = Path(__file__).resolve().parents[2] / "sql" / "012_theta_stock_snapshots.sql"

def ensure_schema():
    with get_connection() as conn:
        conn.execute(DDL.read_text(encoding="utf-8-sig"))

def latest(ticker):
    with get_connection() as conn:
        return conn.execute("SELECT market_date, payload, daily_bars, updated_at FROM theta_stock_snapshots WHERE ticker=%s ORDER BY market_date DESC LIMIT 1", (ticker,)).fetchone()

def save(ticker, market_date, payload, daily_bars):
    with get_connection() as conn:
        conn.execute("""INSERT INTO theta_stock_snapshots(ticker,market_date,payload,daily_bars)
            VALUES (%s,%s,%s,%s) ON CONFLICT(ticker,market_date) DO UPDATE
            SET payload=excluded.payload,daily_bars=excluded.daily_bars,updated_at=clock_timestamp()""",
            (ticker,market_date,Jsonb(payload),Jsonb(daily_bars)))

def published_universe():
    """Read existing holdings and rankings, never launch live discovery scans."""
    holdings, ranked = [], []
    with get_connection() as conn:
        if conn.execute("SELECT to_regclass('portfolio_runs') AS t").fetchone()["t"]:
            holdings = conn.execute("""WITH latest AS (
                SELECT DISTINCT ON(strategy) id FROM portfolio_runs
                WHERE is_published AND status='published' ORDER BY strategy,run_date DESC,as_of_timestamp DESC
            ) SELECT p.ticker,p.sector FROM portfolio_positions p JOIN latest l ON p.run_id=l.id
              WHERE p.target_weight>0 ORDER BY p.target_weight DESC""").fetchall()
        if conn.execute("SELECT to_regclass('stock_intelligence_runs') AS t").fetchone()["t"]:
            row = conn.execute("""SELECT payload FROM stock_intelligence_runs
              WHERE is_published AND status='published' ORDER BY run_date DESC,as_of_timestamp DESC LIMIT 1""").fetchone()
            if row:
                ranked = (row["payload"].get("rankings") or row["payload"].get("rows") or [])[:20]
    return holdings, ranked


def load_daily_cache(ticker):
    with get_connection() as conn:
        return conn.execute("SELECT market_date,daily_bars FROM theta_stock_daily_cache WHERE ticker=%s", (ticker,)).fetchone()

def save_daily_cache(ticker, daily_bars):
    if not daily_bars:
        return
    with get_connection() as conn:
        conn.execute("""INSERT INTO theta_stock_daily_cache(ticker,market_date,daily_bars)
          VALUES (%s,%s,%s) ON CONFLICT(ticker) DO UPDATE
          SET market_date=excluded.market_date,daily_bars=excluded.daily_bars,updated_at=clock_timestamp()
          WHERE excluded.market_date >= theta_stock_daily_cache.market_date""",
          (ticker,daily_bars[-1]['date'],Jsonb(daily_bars)))
