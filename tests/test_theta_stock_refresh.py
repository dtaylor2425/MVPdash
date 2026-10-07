from datetime import datetime, date, timezone
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from jobs.theta_stock_refresh import normalize, target_date, StockFetcher, universe


def test_eod_uses_created_session_and_rejects_invalid_prices():
    frame=pd.DataFrame([
        {'created':'2026-10-06T17:15:00','open':10,'high':12,'low':9,'close':11,'volume':100},
        {'created':'2026-10-05T17:15:00','open':0,'high':12,'low':9,'close':11,'volume':100}])
    assert normalize(frame)==[{'date':'2026-10-06','open':10.,'high':12.,'low':9.,'close':11.,'volume':100.,'vwap':None}]


def test_before_report_ready_uses_previous_session():
    assert target_date(datetime(2026,10,6,21,29,tzinfo=timezone.utc))==date(2026,10,5)
    assert target_date(datetime(2026,10,6,21,30,tzinfo=timezone.utc))==date(2026,10,6)
    assert target_date(datetime(2026,10,4,22,tzinfo=timezone.utc))==date(2026,10,2)


def test_intraday_explicit_consolidated_venue_and_regular_hours():
    calls=[]
    def ohlc(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame([{'timestamp':t,'open':10,'high':12,'low':9,'close':11,'volume':100}
            for t in ['2026-10-06T09:25:00','2026-10-06T09:30:00','2026-10-06T16:00:00']])
    fetch=StockFetcher({'thetaConcurrency':1},client=SimpleNamespace(stock_history_ohlc=ohlc))
    rows=fetch.intraday('AAPL',date(2026,10,6))
    assert len(rows)==1
    assert calls[0]['venue']=='utp_cta'
    assert calls[0]['interval']=='5m'


def test_universe_prioritizes_published_positions_and_bound():
    with patch('jobs.theta_stock_refresh.store.published_universe',return_value=([{'ticker':'XYZ','sector':'Energy'}],[{'ticker':'ABC','sector':'Healthcare'}])):
        symbols,sectors=universe()
    assert symbols[:2]==['XYZ','ABC']
    assert sectors['XYZ']=='XLE'
    assert sectors['ABC']=='XLV'
    assert len(symbols)<=60


def test_explicit_symbols_validated():
    import pytest
    with pytest.raises(ValueError):
        universe('AAPL,../../../x')
    assert universe('aapl,AAPL,MSFT')[0]==['AAPL','MSFT']


def test_eod_requests_respect_vendor_365_day_limit():
    calls=[]
    def eod(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame()
    fetch=StockFetcher({'thetaConcurrency':1},client=SimpleNamespace(stock_history_eod=eod))
    fetch.daily('AAPL',date(2025,8,1),date(2026,10,7))
    assert len(calls)==2
    assert all((r['end_date']-r['start_date']).days<365 for r in calls)
    from datetime import timedelta
    assert calls[1]['start_date']==calls[0]['end_date']+timedelta(days=1)


def test_benchmark_cache_uses_five_session_overlap_without_public_snapshot():
    from jobs.theta_stock_refresh import refresh_daily
    from unittest.mock import Mock
    dates=pd.bdate_range(end='2026-10-06',periods=280)
    bars=[{'date':d.date().isoformat(),'close':100} for d in dates]
    fetch=Mock()
    fetch.daily.return_value=[{'date':'2026-10-07','close':101}]
    with patch('jobs.theta_stock_refresh.store.load_daily_cache',return_value={'daily_bars':bars}), \
         patch('jobs.theta_stock_refresh.store.latest') as latest, \
         patch('jobs.theta_stock_refresh.store.save_daily_cache') as save:
        result=refresh_daily('SPY',date(2026,10,7),fetch)
    fetch.daily.assert_called_once_with('SPY',dates[-5].date(),date(2026,10,7))
    latest.assert_not_called()
    save.assert_called_once_with('SPY',result)
    assert len(result)==280
    assert result[-1]['date']=='2026-10-07'
