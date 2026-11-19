from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pandas as pd
from jobs import stock_research_refresh as job
from api.services.stock_participation import build_participation_payload

def test_yahoo_daily_uses_exclusive_end_and_no_dividend_adjustment():
    ticker=Mock()
    ticker.history.return_value=pd.DataFrame({'Open':[100], 'High':[103], 'Low':[99], 'Close':[102], 'Volume':[1000]},index=pd.DatetimeIndex(['2026-11-17'],tz='America/New_York'))
    with patch.object(job.yf,'Ticker',return_value=ticker) as factory:
        bars=job.StockFetcher().daily('BRK.B',date(2026,11,16),date(2026,11,17))
    factory.assert_called_once_with('BRK-B')
    assert ticker.history.call_args.kwargs['end']=='2026-11-18'
    assert ticker.history.call_args.kwargs['auto_adjust'] is False
    assert bars[0]['date']=='2026-11-17'

def test_partial_intraday_withheld_and_full_day_accepted():
    fetch=job.StockFetcher()
    stamps=pd.date_range('2026-11-17 09:30',periods=78,freq='5min',tz='America/New_York')
    rows=[{'timestamp':t.isoformat(),'open':100,'high':102,'low':99,'close':101,'volume':10} for t in stamps]
    fetch._intraday['AAPL']=rows[:-1]
    assert fetch.intraday('AAPL',date(2026,11,17))==[]
    fetch._intraday['AAPL']=rows
    assert len(fetch.intraday('AAPL',date(2026,11,17)))==78

def test_never_splices_old_provider_history():
    fetch=Mock(); fetch.daily.return_value=[{'date':'2026-11-17','close':101}]
    with patch.object(job.store,'load_daily_cache',side_effect=AssertionError('must not mix providers')):
        rows=job.refresh_daily('AAPL',date(2026,11,17),fetch)
    assert rows==fetch.daily.return_value
    assert fetch.daily.call_args.args[1] < date(2025,1,1)

def test_daily_yahoo_preserves_score_without_fabricated_intraday():
    dates=pd.bdate_range(end='2026-11-17',periods=220)
    rows=[{'date':d.date().isoformat(),'open':100+i,'high':102+i,'low':99+i,'close':101+i,'volume':1000} for i,d in enumerate(dates)]
    payload=build_participation_payload('AAPL',rows,[],rows,feed_metadata=job.FEED)
    assert payload['source']=='Yahoo Finance / yfinance'
    assert payload['score']['coverage_pct']==100
    assert payload['metrics']['sma200'] is not None
    assert payload['metrics']['gap_pct'] is not None
    assert payload['metrics']['above_vwap_pct'] is None
    assert payload['metrics']['opening_30m_volume_share_pct'] is None
    assert payload['intraday']==[]

def test_orchestrator_never_runs_options_even_with_old_flag():
    from jobs import market_data_refresh
    with patch.dict('os.environ',{'OPTIONS_SCANNER_ENABLED':'true'}), patch.object(market_data_refresh.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
        assert market_data_refresh.main()==0
    assert len(run.call_args_list)==1
    assert run.call_args.args[0][-1].endswith('stock_research_refresh.py')


def test_retired_vendor_snapshot_not_served_as_current(monkeypatch):
    import pytest
    from fastapi import HTTPException, Response
    from api.routers.stock_participation import stock_participation
    monkeypatch.setattr(job.store, 'latest', lambda ticker: {'payload': {'source': 'ThetaData'}})
    with pytest.raises(HTTPException) as caught:
        stock_participation('AAPL', Response())
    assert caught.value.status_code == 404
