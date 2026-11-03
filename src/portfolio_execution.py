"""Same-publication model executions at validated, observed market quotes."""
from copy import deepcopy
from datetime import datetime, timezone
import math
from src.portfolio_decisions import daily_targets


def execute_at_job_quote(book, candidate, market, day, score, prior_trades):
    from src.portfolio_ledger import _views, target_holdings, volume_signal
    rows, nav = _views(book)
    desired, exposure = target_holdings(candidate, {t:[b for b in bs if not b.get('intraday')] for t,bs in market.items()}, score)
    evidence = {r['ticker']:r for r in candidate['holdings']}
    desired = [r for r in desired if r['ticker']=='SPY' or
               (r.get('stock_intelligence_score',r.get('confluence_score',100)) >= 60 and
                r.get('technical_score',100) >= 45 and r.get('risk_state') != 'Trend break')]
    sessions = [b['date'] for b in market['SPY'] if not b.get('intraday')]
    cooldown = {t:d for t,d in book.get('risk_cooldowns',{}).items() if len([s for s in sessions if s>d]) < 5}
    targets, review = daily_targets(rows, desired, exposure, market, candidate['risk_rules'], cooldown, evidence=evidence)
    trades = list(prior_trades)
    if targets is not None:
        positions = {}
        target_map = {r['ticker']:r for r in targets}
        for ticker in sorted(set(book['positions']) | set(target_map)):
            quote = market[ticker][-1]
            if quote['date'] != day or not quote.get('quote_timestamp'):
                raise ValueError(f'Missing execution quote for {ticker}')
            price = quote['close']
            prior = book['positions'].get(ticker)
            target = target_map.get(ticker)
            quantity = nav*target['target_weight']/price if target else 0
            delta = quantity-(prior['quantity'] if prior else 0)
            reason = review['risk_exits'].get(ticker) or ('Below minimum position size' if not target else review['reason'])
            if abs(delta)>1e-12:
                trade = dict(date=day,ticker=ticker,action='Buy' if delta>0 else 'Sell',quantity=abs(delta),price=price,
                             decision_date=day,weight_traded=abs(delta)*price/nav,price_basis='Observed job-time model quote; not a broker fill',
                             quote_timestamp=quote['quote_timestamp'],decision_reason=reason)
                if delta<0:
                    entry=prior.get('entry_price')
                    trade.update(exit_type='Full exit' if not target else 'Trim',entry_date=prior.get('entry_date'),entry_price=entry,
                                 exit_price=price,exit_return_pct=(price/entry-1)*100 if entry else None,
                                 return_basis='Price change from original model entry; not average-cost realized P/L')
                trades.append(trade)
            if target:
                # Preserve provenance on adds/trims, refresh assessments from this run.
                entry={k:v for k,v in (prior or {}).items() if k.startswith('entry_') or k=='original_entry_price'}
                if not prior:
                    entry=dict(entry_date=day,entry_price=price,entry_source='Observed job-time model quote',entry_price_status='recorded_model_entry')
                positions[ticker]={**deepcopy(target),**entry,'quantity':quantity,'current_price':price,'action':'Buy' if delta>1e-12 else 'Hold'}
        book['positions']=positions
        book['cash']=nav*(1-sum(r['target_weight'] for r in targets))
        review['status']='executed'
    else:
        for ticker,position in book['positions'].items():
            fresh=evidence.get(ticker,{})
            for key in ('stock_intelligence_score','confluence_score','technical_score','risk_state','trade_reason'):
                if key in fresh: position[key]=fresh[key]
            position['action']='Hold'
        exposure=sum(r['current_weight'] for r in rows)
    for ticker in review.get('risk_exits',{}): cooldown[ticker]=day
    book['risk_cooldowns']=cooldown
    book['pending']=None
    book['last_execution_date']=day
    book['valuation_timestamp']=market['SPY'][-1]['quote_timestamp']
    book['execution_policy']='job-quote-v1'
    # build_ledger already appended prior_trades; append only this job's new executions.
    book['trades'].extend(trades[len(prior_trades):])
    holdings,marked_nav=_views(book)
    if not math.isclose(marked_nav,nav,rel_tol=1e-10): raise ValueError('Execution changed NAV without a market move')
    return holdings,marked_nav,trades,review,exposure


def load_job_quotes(tickers, now=None):
    """Regular-session minute quotes only; never backfill a historical execution."""
    import yfinance as yf
    import pandas as pd
    import pandas_market_calendars as mcal
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None: raise ValueError('Timezone-aware quote observation required')
    local=now.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York'))
    schedule=mcal.get_calendar('NYSE').schedule(start_date=local.date(),end_date=local.date())
    if schedule.empty or not (schedule.iloc[0]['market_open'].to_pydatetime() <= now <= schedule.iloc[0]['market_close'].to_pydatetime()):
        raise ValueError('Same-job model execution requires an open NYSE session; no backdated close fills')
    symbols=sorted(set(tickers)|{'SPY'})
    frame=yf.download(symbols,period='5d',interval='1m',auto_adjust=False,actions=True,prepost=False,progress=False,group_by='ticker',threads=True,timeout=30)
    quotes={}
    for ticker in symbols:
        data=frame[ticker] if isinstance(frame.columns,pd.MultiIndex) else frame
        data=data.dropna(subset=['Close'])
        if data.empty: raise ValueError(f'No current quote for {ticker}')
        stamp=data.index[-1].to_pydatetime()
        price=float(data.iloc[-1]['Close'])
        if stamp.tzinfo is None or not 0 <= (now-stamp).total_seconds() <= 900 or not math.isfinite(price) or price<=0:
            raise ValueError(f'Stale or invalid execution quote for {ticker}')
        quotes[ticker]=dict(date=local.date().isoformat(),open=price,close=price,volume=0,split=0,dividend=0,
                            intraday=True,quote_timestamp=stamp.isoformat())
    return quotes
