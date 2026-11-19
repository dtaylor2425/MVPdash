"""Bounded post-close Yahoo Finance stock snapshots; no ThetaData subscription required.

Run after 17:30 Eastern; before then uses the preceding completed session.
python jobs/stock_research_refresh.py --tickers AAPL,MSFT --dry-run
Fetches a complete Yahoo history each session to avoid mixing vendor prices or split bases.
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
import yfinance as yf
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
SECTOR_HINTS.update(store.SECTOR_FALLBACKS)
FEED = {'provider':'Yahoo Finance / yfinance','daily_feed':'yahoo_daily','intraday_feed':'yahoo_5m',
 'adjustment':'Yahoo split-adjusted OHLC; auto_adjust=False (no dividend adjustment)',
 'session':'completed','intraday_interval':'5m',
 'coverage_note':'Yahoo Finance completed daily bars and available regular-session five-minute bars. Price returns, not dividend-reinvested returns. Intraday VWAP and turnover are bar-based estimates; missing intraday coverage is withheld.'}


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

class StockFetcher:
    def __init__(self):
        self.client = None
        self._intraday = {}

    def daily(self, symbol, start, end):
        frame = yf.Ticker(symbol.replace('.', '-')).history(
            start=start.isoformat(), end=(end+timedelta(days=1)).isoformat(),
            interval='1d', auto_adjust=False, actions=False, prepost=False, timeout=25)
        return normalize(frame.rename_axis('date').reset_index())

    def intraday(self, symbol, day):
        # One recent request per ticker supplies both sessions. Yahoo's intraday
        # retention is limited; do not claim coverage for unavailable old dates.
        if symbol not in self._intraday:
            frame = yf.Ticker(symbol.replace('.', '-')).history(
                period='5d', interval='5m', auto_adjust=False, actions=False,
                prepost=False, timeout=25)
            self._intraday[symbol] = normalize(frame.rename_axis('timestamp').reset_index(), True)
        bounds = session_bounds(day)
        if not bounds:
            return []
        opening, closing = bounds
        rows = [r for r in self._intraday[symbol]
                if opening <= pd.Timestamp(r['timestamp']).to_pydatetime() < closing]
        # Reject partial sessions rather than treating an early observation as
        # the close or computing distorted volume-share/VWAP statistics.
        expected = int((closing-opening).total_seconds()/300)
        if len(rows) != expected:
            return []
        return rows


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
    """Fetch vendor-consistent daily history; cache only within this run."""
    # Full replacement is deliberate: never splice Theta and Yahoo inputs, or
    # yesterday's pre-split cache into today's split-adjusted Yahoo history.
    rows = fetch.daily(symbol, day-timedelta(days=730), day)
    return [r for r in rows if r['date'] <= day.isoformat()][-520:]


def run(args, fetcher=None):
    from api.services.stock_participation import build_participation_payload
    day = date.fromisoformat(args.date) if args.date else target_date()
    if day>target_date() or not is_trading_day(day):
        raise ValueError('Date must be a completed, EOD-ready trading session')
    if not args.dry_run:
        store.ensure_schema()
    symbols,sectors = universe(args.tickers)
    # Reuse the existing singleton lock to prevent overlapping scheduled runs.
    lock = get_connection() if not args.dry_run else None
    if lock is not None and not try_acquire(lock):
        lock.close()
        print('Stock refresh busy; deferred.',flush=True)
        return 0
    if lock is not None:
        lock.commit()  # Advisory session lock survives; avoid an idle transaction.
    fetch = fetcher
    try:
        fetch = fetch or StockFetcher()
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
                if existing and existing['market_date']==day and existing['payload'].get('source') == FEED['provider'] and not args.force:
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
                    print(symbol+': Yahoo intraday unavailable; publishing dated EOD only.',flush=True)
                metadata = dict(FEED)
                previous_bars = [r for r in intraday if pd.Timestamp(r['timestamp']).date() < day]
                metadata['previous_regular_close'] = previous_bars[-1]['close'] if previous_bars else None
                payload = build_participation_payload(symbol,bars,intraday,benchmark,sector_bars,
                    sector_symbol=sector,feed_metadata=metadata)
                payload['as_of'] = day.isoformat()
                payload['source'] = FEED['provider']
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers')
    parser.add_argument('--date')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        sys.exit(run(args))
    except Exception as exc:
        print('Yahoo stock refresh failed ({}); existing snapshots preserved.'.format(type(exc).__name__),flush=True)
        sys.exit(1)
