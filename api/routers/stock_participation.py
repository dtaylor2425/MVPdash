"""Public read-only access to worker-produced Theta stock research snapshots."""
from datetime import datetime, timezone, time
from fastapi import APIRouter, HTTPException, Response
from psycopg.errors import UndefinedTable
from api.services import theta_stock_store as store
from api.services.options_flow_calendar import sessions_between, NY, is_trading_day, previous_trading_day

router = APIRouter(prefix="/api/stock-participation", tags=["stock-participation"])

@router.get("/{ticker}")
def stock_participation(ticker: str, response: Response):
    symbol = ticker.strip().upper()
    if not store.TICKER_RE.fullmatch(symbol):
        raise HTTPException(400, "Invalid ticker")
    try:
        row = store.latest(symbol)
    except UndefinedTable:
        raise HTTPException(503, "Stock participation snapshots are being prepared")
    if not row:
        raise HTTPException(404, "Theta stock participation is not yet available for this ticker")
    payload = dict(row["payload"])
    now = datetime.now(timezone.utc).astimezone(NY)
    today = now.date()
    expected = today if is_trading_day(today) and now.time() >= time(17,30) else previous_trading_day(today)
    age = max(0, len(sessions_between(row["market_date"], today)) - 1)
    payload["published_at"] = row["updated_at"].isoformat()
    payload["stale"] = row["market_date"] < expected
    payload["session_age"] = age
    response.headers["Cache-Control"] = "public, max-age=300"
    return payload
