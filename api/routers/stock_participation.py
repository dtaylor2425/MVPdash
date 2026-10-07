"""Public read-only access to worker-produced Theta stock research snapshots."""
from datetime import datetime, timezone, time
from fastapi import APIRouter, HTTPException, Response
from psycopg.errors import UndefinedTable
from api.services import theta_stock_store as store
from api.services.stock_participation import clean_bars, enrich_price_bars, session_anatomy, rounded, metric_availability
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
    # Add display analytics to older snapshots using their original stored inputs.
    # This neither fetches vendor data nor rewrites a published score or history.
    if payload.get('analytics_view_version', 1) < 2:
        bars = enrich_price_bars(clean_bars(row['daily_bars']))
        payload['bars'] = bars[-260:]
        metrics = dict(payload.get('metrics', {}))
        metrics['sma200'] = bars[-1]['sma200'] if bars else None
        metrics.update({k: rounded(v) for k,v in session_anatomy(
            clean_bars(payload.get('intraday', []), 'timestamp'), str(row['market_date'])).items()})
        payload['metrics'] = metrics
        payload['analytics_view_version'] = 2
    if not payload.get('sector_symbol') and symbol in store.SECTOR_FALLBACKS:
        sector = store.SECTOR_FALLBACKS[symbol]
        cache = store.load_daily_cache(sector)
        bars = payload.get('bars', [])
        lookup = {b['date']: b['close'] for b in clean_bars(cache['daily_bars'])} if cache else {}
        if len(bars) >= 21 and payload.get('metrics', {}).get('relative_strength_spy_20d_pct') is not None:
            first, last = bars[-21], bars[-1]
            a, b = lookup.get(first['date']), lookup.get(last['date'])
            if a and b:
                payload['metrics'] = dict(payload['metrics'], relative_strength_sector_20d_pct=rounded(
                    ((last['close']/first['close']-1)-(b/a-1))*100))
                payload['sector_symbol'] = sector
    payload['availability'] = metric_availability(payload.get('metrics', {}), payload.get('sector_symbol'))
    now = datetime.now(timezone.utc).astimezone(NY)
    today = now.date()
    expected = today if is_trading_day(today) and now.time() >= time(17,30) else previous_trading_day(today)
    age = max(0, len(sessions_between(row["market_date"], today)) - 1)
    payload["published_at"] = row["updated_at"].isoformat()
    payload["stale"] = row["market_date"] < expected
    payload["session_age"] = age
    response.headers["Cache-Control"] = "public, max-age=300"
    return payload
