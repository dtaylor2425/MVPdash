from copy import deepcopy
import pytest
from src.portfolio_ledger import build_ledger
from src.portfolio_decisions import daily_targets


def setup():
    position=dict(ticker='AAA',quantity=.001,current_price=100,entry_price=90,entry_date='2026-09-01',stop_level=95,current_weight=.1,target_weight=.1)
    spy=dict(ticker='SPY',quantity=.00175,current_price=400,entry_price=400,entry_date='2026-09-01',current_weight=.7,target_weight=.7)
    book=dict(version='published-ledger-v1',anchor_date='2026-09-18',as_of='2026-09-18',cash=.2,positions={'AAA':position,'SPY':spy},base_model=0,base_benchmark=0,benchmark_anchor=400,benchmark_units=1,benchmark_cash=0,pending={'decision_date':'2026-09-18','execute_not_before':'2026-09-22','holdings':[position,spy]},trades=[{'decision_date':'2026-09-01','date':'2026-09-01','ticker':'AAA','action':'Buy'}])
    previous={'payload':dict(ledger=book,holdings=[position,spy],performance={'series':[{'date':'2026-09-18','model':0,'benchmark':0}],'stats':{}})}
    candidate=dict(holdings=[dict(ticker='AAA',target_weight=.12,stock_intelligence_score=80,technical_score=80,stop_level=95)],risk_rules=dict(decision_policy='job-execution-v3',min_position_weight=.01,max_single_position=.12,max_sector_weight=1))
    def bar(t,p,intra=False):return dict(date=t,open=p,close=p,volume=100,split=0,dividend=0,intraday=intra,quote_timestamp=t+'T15:00:00-04:00' if intra else None)
    market={'AAA':[bar('2026-09-18',100),bar('2026-09-21',80,True)],'SPY':[bar('2026-09-18',400),bar('2026-09-21',400,True)]}
    return candidate,previous,market


def test_immediate_exit_supersedes_pending_and_preserves_history():
    c,p,m=setup();original=deepcopy(p)
    out=build_ledger(c,p,[],m,'2026-09-21',50)
    assert p==original
    assert out['pending_allocation'] is None
    assert out['decision_review']['status']=='executed'
    assert 'AAA' not in [h['ticker'] for h in out['holdings']]
    exit=next(t for t in out['trade_queue'] if t['ticker']=='AAA')
    assert exit['date']=='2026-09-21' and exit['price']==80
    assert exit['exit_return_pct']==pytest.approx((80/90-1)*100)
    assert out['ledger']['trades'][:1]==original['payload']['ledger']['trades']
    assert out['performance']['series'][:1]==original['payload']['performance']['series']
    assert out['performance']['series'][-1]['model']==pytest.approx(-.02)
    assert sum(h['current_weight'] for h in out['holdings'])+out['cash_weight']==pytest.approx(1)
    assert out['stock_exposure']==pytest.approx(.8)


def test_rerun_cannot_change_same_day_publication():
    c,p,m=setup();first=build_ledger(c,p,[],m,'2026-09-21',50)
    m['SPY'][-1]['close']=500
    assert build_ledger(c,{'payload':first},[],m,'2026-09-21',50)==first


def test_next_day_different_completed_close_does_not_rewrite_intraday_history():
    c,p,m=setup();first=build_ledger(c,p,[],m,'2026-09-21',50)
    for t in m:
        m[t][-1]['intraday']=False
        m[t][-1]['close']*=1.01
        m[t].append(dict(m[t][-1],date='2026-09-22',intraday=True,quote_timestamp='2026-09-22T15:00:00-04:00'))
    second=build_ledger(c,{'payload':first},[],m,'2026-09-22',50)
    assert second['performance']['series'][:len(first['performance']['series'])]==first['performance']['series']
    assert second['ledger']['trades'][:len(first['ledger']['trades'])]==first['ledger']['trades']


@pytest.mark.parametrize('key,value,reason',[('stock_intelligence_score',59,'Composite score below 60'),('technical_score',44,'Technical score below 45'),('risk_state','Trend break','Trend break risk state')])
def test_score_hard_exits(key,value,reason):
    c,p,m=setup();m['AAA'][-1]['close']=100;c['holdings'][0][key]=value
    out=build_ledger(c,p,[],m,'2026-09-21',50)
    assert out['decision_review']['risk_exits']['AAA']==reason
    assert 'AAA' not in [h['ticker'] for h in out['holdings']]


def test_minimum_position_cleanup():
    old=[dict(ticker='A',current_weight=.005,target_weight=.005),dict(ticker='SPY',current_weight=.795,target_weight=.795)]
    rows,_=daily_targets(old,[dict(ticker='B',target_weight=.1)],.8,{'A':[{'close':100}]},{'min_position_weight':.01})
    assert all(r['ticker']=='SPY' or r['target_weight']>=.01 for r in rows)


def test_missing_quote_blocks_execution():
    c,p,m=setup();m['AAA'][-1]['quote_timestamp']=None
    with pytest.raises(ValueError,match='Missing execution quote'):build_ledger(c,p,[],m,'2026-09-21',50)


def test_small_residual_closes_even_inside_deadband():
    old=[dict(ticker='A',current_weight=.005,target_weight=.005),dict(ticker='SPY',current_weight=.795,target_weight=.795)]
    rows,review=daily_targets(old,old,.8,{'A':[{'close':100}]},{'min_position_weight':.01})
    assert [r['ticker'] for r in rows]==['SPY']
    assert rows[0]['target_weight']==pytest.approx(.8)
    assert review['risk_exits']['A']=='Below minimum position size'


def test_retained_position_keeps_original_entry():
    c,p,m=setup();m['AAA'][-1]['close']=100
    out=build_ledger(c,p,[],m,'2026-09-21',50)
    row=next(h for h in out['holdings'] if h['ticker']=='AAA')
    assert row['entry_date']=='2026-09-01' and row['entry_price']==90


def test_quote_loader_refuses_closed_market():
    from datetime import datetime,timezone
    from src.portfolio_execution import load_job_quotes
    with pytest.raises(ValueError,match='open NYSE'):
        load_job_quotes(['AAA'],datetime(2026,9,21,23,tzinfo=timezone.utc))


@pytest.mark.parametrize('minutes',[1,20])
def test_quote_loader_requires_fresh_observations(monkeypatch,minutes):
    import pandas as pd
    import yfinance as yf
    from datetime import datetime,timezone,timedelta
    from src.portfolio_execution import load_job_quotes
    now=datetime(2026,9,21,19,tzinfo=timezone.utc)
    frame=pd.DataFrame({'Close':[100]},index=pd.DatetimeIndex([now-timedelta(minutes=minutes)]))
    monkeypatch.setattr(yf,'download',lambda *a,**k:frame)
    if minutes==20:
        with pytest.raises(ValueError,match='Stale'):load_job_quotes(['SPY'],now)
    else:
        result=load_job_quotes(['SPY'],now)
        assert result['SPY']['close']==100 and result['SPY']['date']=='2026-09-21'
