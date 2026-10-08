import pandas as pd
import pytest
from api.services.options_flow_scanner import build_scanner_symbol,build_scanner_snapshot,rank_scanner


def trades():
    return pd.DataFrame(dict(expiration=['2026-10-09']*4,strike=[100]*4,is_call=[True,False,True,False],
        price=[2,1.5,1.5,2],bid=[1.5]*4,ask=[2]*4,size=[10,10,10,10],condition=[0]*4))


def scan(frame,**kwargs):
    return build_scanner_symbol('ABC',frame,'2026-10-08',condition_allowlist={0},**kwargs)


def test_bullish_bearish_attribution_and_offsets():
    r=scan(trades())
    assert r['callBuyPremium']==2000
    assert r['putSellPremium']==1500
    assert r['callSellPremium']==1500
    assert r['putBuyPremium']==2000
    assert r['netBullishPremium']==0
    assert r['grossPremium']==7000
    assert r['status']=='ok'
    assert r['historicalUnusualness'] is None


def test_unknown_conditions_and_multileg_not_bullish():
    frame=trades().iloc[:1].copy()
    frame['condition']=99
    assert scan(frame)['netBullishPremium']==0
    assert scan(frame)['quality']['excludedConditionTrades']==1
    frame['condition']=0
    frame['is_multileg']=True
    assert scan(frame)['netBullishPremium']==0
    frame=frame.drop(columns=['condition','is_multileg'])
    assert scan(frame)['status']=='insufficient_quality'


@pytest.mark.parametrize('bid,ask,price',[(2,2,2),(3,2,2),(0,2,2),(1.5,2,1.75),(1.5,2,2.1)])
def test_ambiguous_quotes_are_not_directional(bid,ask,price):
    frame=trades().iloc[:1].copy()
    frame[['bid','ask','price']]=[bid,ask,price]
    r=scan(frame)
    assert r['netBullishPremium']==0
    assert r['unclassifiedPremium']==r['grossPremium']
    assert r['confidence']=='low'


def test_fractional_inside_spread_attribution():
    frame=trades().iloc[:1].copy()
    frame['price']=1.875
    r=scan(frame)
    assert r['callBuyPremium']==937.5
    assert r['quality']['classificationCoverage']==.5


def test_next_exchange_session_and_calendar_windows_exclude_same_day():
    frame=pd.concat([trades().iloc[:1]]*4,ignore_index=True)
    frame['expiration']=['2026-10-09','2026-10-12','2026-10-16','2026-11-09']
    r=build_scanner_symbol('ABC',frame,'2026-10-09',window='next_session',next_session='2026-10-12',condition_allowlist={0})
    assert r['grossPremium']==2000
    assert len(r['contracts'])==1
    r=build_scanner_symbol('ABC',frame,'2026-10-09',window='7d',condition_allowlist={0})
    assert r['grossPremium']==4000
    with pytest.raises(ValueError):
        build_scanner_symbol('ABC',frame,'2026-10-09',window='next_session')


def test_failures_are_not_zero_flow():
    r=scan(pd.DataFrame(),collection_error='source failed')
    assert r['status']=='failed'
    assert r['netBullishPremium'] is None
    assert scan(pd.DataFrame())['status']=='no_eligible_trades'


def test_rank_uses_net_not_call_volume_and_deterministic_ties():
    bullish=scan(trades().iloc[:1],completed=True)
    offset=scan(trades())
    offset['ticker']='AAA'
    snapshots=[dict(marketDate='2026-10-08',windows={'7d':x}) for x in [offset,bullish]]
    r=rank_scanner(snapshots)
    assert [x['ticker'] for x in r['shortlist']]==['ABC']
    snapshots.append(dict(marketDate='2026-10-07',windows={'7d':bullish}))
    with pytest.raises(ValueError):rank_scanner(snapshots)


def test_raw_wrapper_preserves_condition_and_requires_no_greeks():
    raw=pd.DataFrame(dict(trade_timestamp=['2026-10-08T15:00:00Z'],strike=[100],right=['C'],price=[2],size=[10],bid=[1.5],ask=[2],condition=[99]))
    r=build_scanner_snapshot('ABC','2026-10-08',[dict(expiration='2026-10-09',trades=raw)],next_session='2026-10-09',condition_allowlist={0},completed=True)
    assert r['windows']['7d']['netBullishPremium']==0
    assert r['windows']['7d']['completed']


def test_partial_rows_are_visible_but_not_shortlisted():
    row=scan(trades().iloc[:1])
    r=rank_scanner([dict(marketDate='2026-10-08',windows={'7d':row})])
    assert len(r['rows'])==1
    assert r['shortlist']==[]



def history_fixture():
    from api.services.options_flow_calendar import sessions_before
    from api.services.options_flow_scanner import METHODOLOGY_VERSION
    from copy import deepcopy
    current=dict(ticker='ABC',marketDate='2026-10-08',profileIdentity='profile-a',methodologyVersion=METHODOLOGY_VERSION,
                 windows={'7d':scan(trades().iloc[:1],completed=True)})
    history=[]
    for i,day in enumerate(sessions_before(pd.Timestamp('2026-10-08').date(),20)):
        prior=deepcopy(current)
        prior['marketDate']=day.isoformat()
        prior['windows']['7d']['marketDate']=day.isoformat()
        prior['windows']['7d']['grossPremium']=1000+i
        history.append(prior)
    return current,history


def test_historical_activity_requires_twenty_same_profile_sessions():
    from api.services.options_flow_scanner import apply_historical_unusualness
    current,history=history_fixture()
    apply_historical_unusualness(current,history)
    result=current['windows']['7d']['historicalUnusualness']
    assert result['sampleSize']==20
    assert result['percentile']==100
    assert result['priorMean']==1009.5


@pytest.mark.parametrize('issue',['gap','partial','profile','duplicate','methodology','future','subset'])
def test_baseline_rejects_noncomparable_history(issue):
    from api.services.options_flow_scanner import apply_historical_unusualness
    current,history=history_fixture()
    if issue=='gap':history.pop()
    elif issue=='partial':history[0]['windows']['7d']['completed']=False
    elif issue=='profile':history[0]['profileIdentity']='other'
    elif issue=='duplicate':history.append(history[0])
    elif issue=='methodology':history[0]['methodologyVersion']='other'
    elif issue=='future':history[0]['marketDate']='2026-10-09'
    elif issue=='subset':history[0]['windows']['7d']['status']='failed'
    apply_historical_unusualness(current,history)
    assert current['windows']['7d']['historicalUnusualness'] is None


def test_official_regular_and_sweep_default_only():
    for condition in [0,18,18.0,"18.0",95,125,130,144,999]:
        frame=trades().iloc[:1].copy()
        frame['condition']=condition
        row=build_scanner_symbol('ABC',frame,'2026-10-08')
        assert (row['netBullishPremium']>0) == (condition in [0,18,18.0,"18.0",95])



def test_condition_histogram_and_classifiable_counts():
    frame=trades().copy()
    frame['condition']=[18.0,130,95,999]
    frame['ts']=pd.to_datetime(['2026-10-08T15:01:00Z']*4)
    result=build_scanner_symbol('ABC',frame,'2026-10-08')
    quality=result['quality']
    assert quality['eligibleTrades']==4
    assert quality['classifiableTrades']==2
    codes={r['condition']:r for r in quality['conditionPremiumHistogram']}
    assert codes['18']['grossPremium']==2000
    assert codes['18']['excludedPremium']==0
    assert codes['130']['excludedPremium']==1500
    assert codes['130']['attributedPremium']==0
    assert sum(r['grossPremium'] for r in codes.values())==result['grossPremium']
    contract=result['contracts'][0]
    assert contract['largestTrade']['size']==10
    assert contract['largestTrade']['time']=='2026-10-08T15:01:00+00:00'
    assert contract['totalContracts']==20



def test_low_directional_coverage_does_not_discard_completed_gross_activity():
    from api.services.options_flow_scanner import apply_historical_unusualness
    current,history=history_fixture()
    current['windows']['7d']['status']='insufficient_quality'
    current['windows']['7d']['confidence']='low'
    current['windows']['7d']['quality']['classificationCoverage']=.1
    for prior in history:
        prior['windows']['7d']['status']='insufficient_quality'
        prior['windows']['7d']['quality']['classificationCoverage']=.1
    apply_historical_unusualness(current,history)
    assert current['windows']['7d']['historicalUnusualness']['sampleSize']==20
    assert rank_scanner([current])['shortlist']==[]


@pytest.mark.parametrize('bad_value',[None,float('nan'),-1])
def test_bad_current_gross_suppresses_baseline(bad_value):
    from api.services.options_flow_scanner import apply_historical_unusualness
    current,history=history_fixture()
    current['windows']['7d']['grossPremium']=bad_value
    apply_historical_unusualness(current,history)
    assert current['windows']['7d']['historicalUnusualness'] is None


def test_confidence_explicitly_identifies_classification_basis():
    assert scan(trades())['confidenceBasis']=='directional_classification_coverage'
