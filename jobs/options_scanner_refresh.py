"""Resumable post-close premium-only options scanner for every published stock ranking.

No Greeks or open-interest requests. Completion means every eligible expiry was
collected, not a prediction. --tickers is an explicit benchmark subset; default
boards never serve it. Python 3.12+ using the existing Theta worker environment.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as day_time, timedelta, timezone
import os
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from jobs.options_flow_refresh import ThetaFetcher,load_local_env
from api.db import get_connection
from api.services.options_flow_calendar import NY,is_trading_day,previous_trading_day,session_bounds
from api.services.options_flow_lock import try_acquire,release
from api.services import options_scanner_store as store

SYMBOL=re.compile(r'^[A-Z][A-Z0-9.\-]{0,9}$')
def safe_error(exc):
    message=str(exc)
    for key,value in os.environ.items():
        if key.startswith('THETADATA_') and value:
            message=message.replace(value,'[redacted]')
    message=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[redacted-email]', message)
    return message[:350]

class BudgetReached(Exception):
    pass

def ready_date(now=None):
    now=(now or datetime.now(timezone.utc)).astimezone(NY)
    return now.date() if is_trading_day(now.date()) and now.time()>=day_time(17,30) else previous_trading_day(now.date())

def next_session(day):
    result=day+timedelta(days=1)
    while not is_trading_day(result):
        result+=timedelta(days=1)
    return result

def selection(tickers=None):
    if tickers:
        symbols=list(dict.fromkeys(s.strip().upper() for s in tickers.split(',') if s.strip()))
        if not symbols or any(not SYMBOL.fullmatch(s) for s in symbols):
            raise ValueError('Invalid benchmark ticker list')
        return symbols,None
    symbols,asof=store.published_universe()
    if not symbols:
        raise RuntimeError('Published stock ranking universe is empty')
    return symbols,asof

def configuration():
    from api.services.options_flow_scanner import METHODOLOGY_VERSION
    allowlist=[v.strip() for v in os.getenv('OPTIONS_SCANNER_CONDITION_ALLOWLIST','0,18,95').split(',') if v.strip()]
    config={'version':'near-term-premium-v1','methodologyVersion':METHODOLOGY_VERSION,'maxDte':30,
      'strikeRange':int(os.getenv('OPTIONS_SCANNER_STRIKE_RANGE','25')),
      'conditionAllowlist':allowlist,'session':'regular','excludeExpiredZeroDte':True,
      'windows':['next_session','7d','30d'],'noGreeks':True}
    return config

def collect_expirations(fetch, symbol, day, expirations, budget):
    """At most two requests in flight; never return a partially collected ticker.

    Check the soft budget before each submission. The request counter can race
    with one other submission (at most one extra initial call); vendor retries
    and already-running requests can additionally exceed time/request budgets.
    Drain the current batch on failure before another ticker starts.
    """
    contracts=[]
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='scanner-expiry') as pool:
        for start in range(0,len(expirations),2):
            pending=[]
            for expiry in expirations[start:start+2]:
                budget()
                print('{} requesting trade/quote expiry {}'.format(symbol,expiry),flush=True)
                pending.append((expiry,pool.submit(fetch.trade_quote,symbol,expiry,day)))
            # Stable expiration order regardless of network completion order.
            for expiry,future in pending:
                contracts.append({'expiration':expiry,'trades':future.result()})
    return contracts

def run(args,fetcher=None):
    from api.services.options_flow_scanner import build_scanner_snapshot
    started=time.monotonic()
    day=date.fromisoformat(args.date) if args.date else ready_date()
    if day>ready_date() or not is_trading_day(day):
        raise ValueError('Scanner date must be a completed EOD-ready session')
    symbols,universe_date=selection(args.tickers)
    config=configuration()
    profile=store.fingerprint(config)
    max_seconds=max(1,float(os.getenv('OPTIONS_SCANNER_MAX_SECONDS','1800')))
    max_requests=max(1,int(os.getenv('OPTIONS_SCANNER_MAX_REQUESTS','1000')))
    metadata={'config':config,'benchmark':bool(args.tickers),'universeDate':universe_date,
      'maxSeconds':max_seconds,'maxRequests':max_requests,'noSilentUniverseCap':True,'expiryConcurrency':2}
    lock=None
    run_id=None
    fetch=fetcher
    states={}
    try:
        # Even dry runs share the vendor lock when database credentials exist.
        if os.getenv('DATABASE_URL') or os.getenv('POSTGRES_URL'):
            lock=get_connection()
            if not try_acquire(lock):
                print('Scanner deferred: another Theta process owns the session.',flush=True)
                return 0
            lock.commit()
        elif not args.dry_run:
            raise RuntimeError('Database required for scanner publication')
        if not args.dry_run:
            store.ensure_schema()
            run_id=store.start_run(day,profile,symbols,metadata)
            states=store.symbol_states(run_id)
        if states and all(states.get(s,{}).get('status') in ('completed','excluded') for s in symbols) and not args.force:
            store.finish_run(run_id,'completed',{'requests':0,'seconds':round(time.monotonic()-started,3),'resumedAlreadyComplete':True})
            print('Scanner already complete for this session, profile, and universe.',flush=True)
            return 0
        bounds=session_bounds(day)
        cfg={'thetaConcurrency':2,'strikeRange':config['strikeRange'],
          'sessionStart':'09:30:00','sessionEnd':bounds[1].astimezone(NY).strftime('%H:%M:%S')}
        fetch=fetch or ThetaFetcher(cfg)
        fetch.point_in_time=True
        next_day=next_session(day)
        def budget():
            if time.monotonic()-started>=max_seconds or fetch.requests>=max_requests:
                raise BudgetReached()
        def save(symbol,status,payload=None,diagnostics=None):
            previous=states.get(symbol,{})
            if previous.get('status')=='completed' and status!='completed':
                return
            states[symbol]={'status':status,'payload':payload or {},'diagnostics':diagnostics or {}}
            if run_id:
                store.save_symbol(run_id,symbol,status,payload,diagnostics)
        for symbol in symbols:
            if not SYMBOL.fullmatch(symbol):
                save(symbol,'excluded',diagnostics={'reason':'Unsupported vendor ticker format'})
                continue
            if states.get(symbol,{}).get('status')=='completed' and not args.force:
                continue
            tick_started=time.monotonic()
            requests_before=fetch.requests
            try:
                budget()
                print('{} requesting dated expirations for {}'.format(symbol,day),flush=True)
                expirations=fetch.expirations(symbol,day,30)
                expirations=sorted(set(e for e in expirations if next_day<=e<=day+timedelta(days=30)))
                contracts=collect_expirations(fetch,symbol,day,expirations,budget)
                history=[] if args.dry_run else store.history(symbol,profile,day)
                payload=build_scanner_snapshot(symbol,day,contracts,next_session=next_day,
                  completed=True,condition_allowlist=config['conditionAllowlist'],history=history,profile_identity=profile)
                payload['profile']=profile
                payload['profileIdentity']=profile
                payload['strikeRange']=config['strikeRange']
                diagnostics={'seconds':round(time.monotonic()-tick_started,3),
                  'requests':fetch.requests-requests_before,'expirations':len(expirations),
                  'rows':sum(len(c['trades']) for c in contracts),'empty':not expirations}
                condition_counts=Counter()
                for contract in contracts:
                    raw=contract['trades']
                    condition_col=next((c for c in ('condition','trade_condition') if c in raw.columns),None)
                    if condition_col is None:
                        condition_counts['missing']+=len(raw)
                    else:
                        condition_counts.update(str(v) for v in raw[condition_col].tolist())
                diagnostics['rawConditionCounts']=dict(sorted(condition_counts.items()))
                seven=payload.get('windows',{}).get('7d',{})
                quality=seven.get('quality',{})
                diagnostics['window7d']={'netBullishPremium':seven.get('netBullishPremium'),
                  'classificationCoverage':quality.get('classificationCoverage'),
                  'classifiableTrades':quality.get('classifiableTrades'),'status':seven.get('status')}
                save(symbol,'completed',payload,diagnostics)
                print('{} completed {} expiries {} requests {:.1f}s; 7d {}'.format(symbol,len(expirations),diagnostics['requests'],diagnostics['seconds'],diagnostics['window7d']),flush=True)
                print('{} raw trade conditions {}'.format(symbol,diagnostics['rawConditionCounts']),flush=True)
            except BudgetReached:
                save(symbol,'pending',diagnostics={'reason':'Runtime/request budget reached; resume on next run'})
                print('Scanner budget reached; remaining symbols explicitly pending.',flush=True)
                break
            except Exception as exc:
                save(symbol,'failed',diagnostics={'errorType':type(exc).__name__,'error':safe_error(exc),
                  'seconds':round(time.monotonic()-tick_started,3),'requests':fetch.requests-requests_before})
                print('{} failed ({}): {}; completed records preserved.'.format(symbol,type(exc).__name__,safe_error(exc)),flush=True)
        counts={k:0 for k in ['completed','pending','failed','excluded']}
        for symbol in symbols:
            counts[states.get(symbol,{}).get('status','pending')]+=1
        status='completed' if counts['completed']+counts['excluded']==len(symbols) else 'partial'
        metrics={**counts,'seconds':round(time.monotonic()-started,3),'requests':fetch.requests,'total':len(symbols)}
        if run_id:
            store.finish_run(run_id,status,metrics)
        print('Scanner {}: {}'.format(status,metrics),flush=True)
        return 0 if status=='completed' else 2
    except Exception as exc:
        if run_id:
            store.finish_run(run_id,'partial',{'errorType':type(exc).__name__,'seconds':round(time.monotonic()-started,3)})
        raise
    finally:
        if lock is not None:
            release(lock)
            lock.close()

if __name__=='__main__':
    load_local_env()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date')
    parser.add_argument('--tickers',help='Benchmark subset; excluded from the default board')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    try:
        sys.exit(run(args))
    except Exception as exc:
        print('Options scanner failed ({}); prior completed data preserved.'.format(type(exc).__name__),flush=True)
        sys.exit(1)
