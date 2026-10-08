from argparse import Namespace
from datetime import date
from unittest.mock import patch,MagicMock
import pandas as pd
from jobs import options_scanner_refresh as worker

class Fetcher:
    def __init__(self):
        self.requests=0
        self.requested=[]
    def expirations(self,symbol,day,max_dte):
        self.requests+=1
        return [date(2026,10,7),date(2026,10,8),date(2026,11,20)]
    def trade_quote(self,symbol,expiry,day):
        self.requests+=1
        self.requested.append(expiry)
        return pd.DataFrame()

def args():
    return Namespace(date='2026-10-07',tickers='AAPL,MSFT',dry_run=True,force=False)

def test_scanner_collects_only_future_through_30_days():
    fetch=Fetcher()
    with patch.dict('os.environ',{'DATABASE_URL':'','POSTGRES_URL':''}),patch.object(worker,'ready_date',return_value=date(2026,10,7)):
        assert worker.run(args(),fetch)==0
    assert fetch.requested==[date(2026,10,8),date(2026,10,8)]

def test_budget_preserves_unattempted_pending():
    fetch=Fetcher()
    with patch.dict('os.environ',{'DATABASE_URL':'','POSTGRES_URL':'','OPTIONS_SCANNER_MAX_REQUESTS':'1'}),patch.object(worker,'ready_date',return_value=date(2026,10,7)):
        assert worker.run(args(),fetch)==2
    assert fetch.requests==1
    assert fetch.requested==[]

def test_full_published_universe_not_capped():
    symbols=['T'+str(i) for i in range(120)]
    with patch.object(worker.store,'published_universe',return_value=(symbols,'2026-10-07')):
        assert worker.selection()[0]==symbols

def test_profile_changes_with_condition_policy():
    with patch.dict('os.environ',{'OPTIONS_SCANNER_CONDITION_ALLOWLIST':'18'}):
        first=worker.store.fingerprint(worker.configuration())
    with patch.dict('os.environ',{'OPTIONS_SCANNER_CONDITION_ALLOWLIST':'18,19'}):
        assert first!=worker.store.fingerprint(worker.configuration())

def test_completed_resume_does_not_reauthenticate_or_fetch():
    a=args();a.dry_run=False
    connection=MagicMock()
    states={s:{'status':'completed'} for s in ['AAPL','MSFT']}
    with patch.dict('os.environ',{'DATABASE_URL':'test'}),patch.object(worker,'ready_date',return_value=date(2026,10,7)), \
      patch.object(worker,'get_connection',return_value=connection),patch.object(worker,'try_acquire',return_value=True),patch.object(worker,'release'), \
      patch.object(worker.store,'ensure_schema'),patch.object(worker.store,'start_run',return_value='run'), \
      patch.object(worker.store,'symbol_states',return_value=states),patch.object(worker.store,'finish_run') as finish,patch.object(worker,'ThetaFetcher') as client:
        assert worker.run(a)==0
    client.assert_not_called()
    assert finish.call_args.args[1]=='completed'


def test_profile_changes_when_methodology_changes():
    with patch('api.services.options_flow_scanner.METHODOLOGY_VERSION','scanner-test-a'):
        first=worker.store.fingerprint(worker.configuration())
    with patch('api.services.options_flow_scanner.METHODOLOGY_VERSION','scanner-test-b'):
        assert first!=worker.store.fingerprint(worker.configuration())

def test_failed_force_refresh_never_overwrites_completed_publication():
    a=args();a.dry_run=False;a.force=True;a.tickers='AAPL'
    connection=MagicMock()
    states={'AAPL':{'status':'completed','payload':{'trusted':'previous'}}}
    fetch=Fetcher()
    def failure(*args):
        raise RuntimeError('Temporary vendor error')
    fetch.expirations=failure
    with patch.dict('os.environ',{'DATABASE_URL':'test'}),patch.object(worker,'ready_date',return_value=date(2026,10,7)), \
      patch.object(worker,'get_connection',return_value=connection),patch.object(worker,'try_acquire',return_value=True),patch.object(worker,'release'), \
      patch.object(worker.store,'ensure_schema'),patch.object(worker.store,'start_run',return_value='run'), \
      patch.object(worker.store,'symbol_states',return_value=states),patch.object(worker.store,'finish_run'),patch.object(worker.store,'save_symbol') as save:
        worker.run(a,fetch)
    save.assert_not_called()
    assert states['AAPL']['payload']=={'trusted':'previous'}


def test_default_condition_policy_includes_verified_auto_execution():
    with patch.dict('os.environ',{},clear=True):
        assert worker.configuration()['conditionAllowlist']==['0','18','95']
