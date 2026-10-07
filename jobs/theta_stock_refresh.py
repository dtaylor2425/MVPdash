"""Bounded post-close Theta stock snapshots. Python 3.12+ worker only.

Run after 17:30 Eastern; before then uses the preceding completed session.
python jobs/theta_stock_refresh.py --tickers AAPL,MSFT --dry-run
Uses national EOD + explicit UTP/CTA bars, never Nasdaq Basic comparisons.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, time, timedelta, timezone
import math
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jobs.options_flow_refresh import ThetaFetcher, load_local_env
from api.services.options_flow_calendar import NY, is_trading_day, previous_trading_day, session_bounds
from api.services.options_flow_lock import try_acquire, release
from api.services import theta_stock_store as store
from api.db import get_connection

SECTORS = {'Technology':'XLK','Financial Services':'XLF','Financials':'XLF',
 'Healthcare':'XLV','Consumer Cyclical':'XLY','Consumer Defensive':'XLP',
 'Industrials':'XLI','Energy':'XLE','Basic Materials':'XLB','Real Estate':'XLRE',
 'Utilities':'XLU','Communication Services':'XLC'}
WATCHLIST = 'AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX PLTR JPM XOM LLY NEM TSM ASML ANET SHOP'.split()
SECTOR_HINTS = dict.fromkeys('AAPL MSFT NVDA AVGO AMD PLTR TSM ASML ANET SHOP'.split(),'XLK')
SECTOR_HINTS.update({'AMZN':'XLY','TSLA':'XLY','GOOGL':'XLC','META':'XLC','NFLX':'XLC','JPM':'XLF','XOM':'XLE','LLY':'XLV','NEM':'XLB'})
FEED = {'provider':'ThetaData','daily_feed':'national_eod','intraday_feed':'utp_cta',
 'adjustment':'unadjusted','session':'completed','intraday_interval':'5m','daily_report_time':'17:15 America/New_York',
 'coverage_note':'National EOD reports are generated at 17:15 ET and can include post-market trades. Intraday bars cover the regular session only (consolidated UTP/CTA). Unadjusted prices, not total returns; no Nasdaq Basic volume is mixed into comparisons.'}

def target_date(now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    today = now.date()
    return today if is_trading_day(today) and now.time() >= time(17,30) else previous_trading_day(today)

def normalize(frame, intraday=False):
    if frame is None or frame.empty:
        return []
    records = []
    for raw in frame.to_dict('records'):
        raw = {str(k).lower():v for k,v in raw.items()}
        stamp = raw.get('timestamp') if intraday else raw.get('created',raw.get('date',raw.get('timestamp')))
        if stamp is None or pd.isna(stamp):
            continue
        stamp = pd.Timestamp(stamp)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert('America/New_York')
        row = {'timestamp' if intraday else 'date':stamp.isoformat() if intraday else stamp.date().isoformat()}
        if not intraday and raw.get('last_trade') is not None and pd.notna(raw.get('last_trade')):
            row['last_trade'] = pd.Timestamp(raw['last_trade']).isoformat()
        for field in ['open','high','low','close','volume','vwap']:
            value = raw.get(field)
            row[field] = float(value) if value is not None and pd.notna(value) and math.isfinite(float(value)) else None
        if all(row[k] is not None and row[k]>0 for k in ['open','high','low','close']) and row['volume'] is not None and row['volume']>=0:
            records.append(row)
    key = 'timestamp' if intraday else 'date'
    return sorted({r[key]:r for r in records}.values(),key=lambda r:r[key])

class StockFetcher(ThetaFetcher):
    def daily(self,symbol,start,end):
        rows = []
        while start <= end:
            stop = min(end, start + timedelta(days=364))
            rows.extend(normalize(self._call('stock EOD '+symbol,self.client.stock_history_eod,
                symbol=symbol,start_date=start,end_date=stop)))
            start = stop + timedelta(days=1)
        return rows
    def intraday(self,symbol,day):
        bounds = session_bounds(day)
        if bounds is None:
            return []
        close = bounds[1].astimezone(NY).strftime('%H:%M:%S')
        rows = normalize(self._call('stock OHLC '+symbol,self.client.stock_history_ohlc,
            symbol=symbol,date=day,interval='5m',venue='utp_cta',start_time='09:30:00',end_time=close),True)
        return [r for r in rows if pd.Timestamp(r['timestamp']).date()==day and '09:30:00'<=pd.Timestamp(r['timestamp']).strftime('%H:%M:%S')<close]

def universe(explicit=None):
    if explicit:
        symbols = list(dict.fromkeys(s.strip().upper() for s in explicit.split(',') if s.strip()))
        if len(symbols)>60 or any(not store.TICKER_RE.fullmatch(s) for s in symbols):
            raise ValueError('Provide at most 60 valid tickers')
        return symbols,SECTOR_HINTS.copy()
    holdings,ranked = store.published_universe()
    sectors = SECTOR_HINTS.copy()
    symbols = []
    for row in [*holdings,*ranked]:
        symbol = str(row.get('ticker') or row.get('symbol') or '').upper()
        if store.TICKER_RE.fullmatch(symbol):
            symbols.append(symbol)
            if row.get('sector') in SECTORS:
                sectors[symbol] = SECTORS[row['sector']]
    return list(dict.fromkeys(symbols+WATCHLIST))[:60],sectors

def refresh_daily(symbol, day, fetch, dry_run=False):
    """Reuse stored benchmark inputs as well as stock inputs on subsequent runs."""
    previous = None if dry_run else store.load_daily_cache(symbol)
    if previous is None and not dry_run:
        previous = store.latest(symbol)  # Migrate previously published input history.
    old = previous['daily_bars'] if previous else []
    old = [r for r in old if r['date'] <= day.isoformat()]
    # Five-session overlap incorporates vendor corrections with bounded requests.
    start = date.fromisoformat(old[-5]['date']) if len(old) >= 260 else day-timedelta(days=410)
    new = fetch.daily(symbol, start, day)
    rows = {r['date']:r for r in old+new if r['date'] <= day.isoformat()}
    result = [rows[k] for k in sorted(rows)][-280:]
    if not dry_run and result:
        store.save_daily_cache(symbol,result)
    return result

def run(args, fetcher=None):
    from api.services.stock_participation import build_participation_payload
    day = date.fromisoformat(args.date) if args.date else target_date()
    if day>target_date() or not is_trading_day(day):
        raise ValueError('Date must be a completed, EOD-ready trading session')
    if not args.dry_run:
        store.ensure_schema()
    symbols,sectors = universe(args.tickers)
    # Dedicated vendor-session lock shared with options jobs. Dry-run without DB
    # is deliberately explicit and intended only when no other session is active.
    lock = get_connection() if not args.dry_run else None
    if lock is not None and not try_acquire(lock):
        lock.close()
        print('Theta session busy; stock refresh deferred.',flush=True)
        return 0
    if lock is not None:
        lock.commit()  # Advisory session lock survives; avoid an idle transaction.
    fetch = fetcher
    try:
        fetch = fetch or StockFetcher({'thetaConcurrency':1})
        cache = {}
        def daily(symbol):
            if symbol not in cache:
                cache[symbol] = refresh_daily(symbol,day,fetch,args.dry_run)
            return cache[symbol]
        benchmark = daily('SPY')
        if not benchmark or benchmark[-1]['date']!=day.isoformat():
            raise RuntimeError('SPY EOD not ready; preserve previous stock snapshots')
        failed = 0
        for symbol in symbols:
            try:
                existing = None if args.dry_run else store.latest(symbol)
                if existing and existing['market_date']==day and not args.force:
                    print(symbol+' already published; skipped.',flush=True)
                    continue
                bars = daily(symbol)
                if not bars or bars[-1]['date']!=day.isoformat():
                    raise RuntimeError('Requested EOD session missing; previous snapshot preserved')
                sector = sectors.get(symbol)
                sector_bars = None
                if sector:
                    try:
                        sector_bars = daily(sector)
                    except Exception:
                        print(symbol+': sector comparison unavailable.',flush=True)
                intraday = []
                try:
                    intraday = fetch.intraday(symbol,previous_trading_day(day)) + fetch.intraday(symbol,day)
                except Exception:
                    print(symbol+': consolidated intraday unavailable; publishing dated EOD only.',flush=True)
                metadata = dict(FEED)
                previous_bars = [r for r in intraday if pd.Timestamp(r['timestamp']).date() < day]
                metadata['previous_regular_close'] = previous_bars[-1]['close'] if previous_bars else None
                payload = build_participation_payload(symbol,bars,intraday,benchmark,sector_bars,
                    sector_symbol=sector,feed_metadata=metadata)
                payload['as_of'] = day.isoformat()
                payload['source'] = 'ThetaData'
                payload['feed_metadata'] = metadata
                payload['coverage_note'] = FEED['coverage_note']
                payload['intraday_available'] = bool(intraday)
                if not args.dry_run:
                    store.save(symbol,day,payload,bars)
                print('{} {} daily={} intraday={} {}'.format(symbol,day,len(bars),len(intraday),'dry-run' if args.dry_run else 'published'),flush=True)
            except Exception as exc:
                failed += 1
                # No vendor exception text: it can contain account details.
                print('{} failed ({}); previous snapshot preserved.'.format(symbol,type(exc).__name__),flush=True)
        return 2 if failed else 0
    finally:
        if fetch is not None:
            close = getattr(fetch.client,'close',None)
            if callable(close):
                close()
        if lock is not None:
            release(lock)
            lock.close()

if __name__=='__main__':
    load_local_env()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers')
    parser.add_argument('--date')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        sys.exit(run(args))
    except Exception as exc:
        print('Theta stock refresh failed ({}); existing snapshots preserved.'.format(type(exc).__name__),flush=True)
        sys.exit(1)
