from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers import research_usage as usage
from api.auth_deps import require_admin, require_account


def client():
    app = FastAPI()
    app.include_router(usage.router)
    return TestClient(app)


def test_unknown_payload_fields_and_dimensions_rejected_before_database(monkeypatch):
    def fail():
        raise AssertionError("No database access expected")
    monkeypatch.setattr(usage, "get_connection", fail)
    c = client()
    headers = {"origin":"https://www.macro-engine.com"}
    assert c.post('/api/research-usage/event', headers=headers, json={"event":"paid","page":"report","source":"direct"}).status_code == 422
    assert c.post('/api/research-usage/event', headers=headers, json={"event":"view","page":"report","source":"direct","email":"private@example.com"}).status_code == 422
    assert c.post('/api/research-usage/event', json={"event":"view","page":"report","source":"direct"}).status_code == 403
    headers['sec-gpc'] = '1'
    assert c.post('/api/research-usage/event', headers=headers, json={"event":"view","page":"report","source":"direct"}).status_code == 204


def test_summary_is_admin_only():
    assert client().get('/api/research-usage/summary').status_code == 401
    c=client()
    c.app.dependency_overrides[require_account]=lambda:{'is_admin':False}
    assert c.get('/api/research-usage/summary').status_code == 403


def test_collection_rate_bound_does_not_store_identifiers(monkeypatch):
    monkeypatch.setattr(usage,'monotonic',lambda:10)
    monkeypatch.setattr(usage,'_window',1)
    monkeypatch.setattr(usage,'_received',240)
    response=client().post('/api/research-usage/event',headers={'origin':'https://www.macro-engine.com'},json={'event':'view','page':'report','source':'direct'})
    assert response.status_code == 429


def test_aggregate_write_stores_only_allowed_dimensions(monkeypatch):
    calls = []
    class Cursor:
        def execute(self, sql, params=None): calls.append((sql,params))
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Connection(Cursor):
        def cursor(self): return Cursor()
        def commit(self): calls.append(('commit',None))
    monkeypatch.setattr(usage,'get_connection',lambda:Connection())
    response=client().post('/api/research-usage/event',headers={'origin':'https://www.macro-engine.com'},json={'event':'view','page':'workbook','source':'substack'})
    assert response.status_code == 204
    assert calls[1][1] == ('view','workbook','substack')
    assert 'ON CONFLICT' in calls[1][0]


def test_summary_does_not_invent_conversion_or_retention_counts(monkeypatch):
    class Cursor:
        def execute(self,*args): pass
        def fetchall(self): return []
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def cursor(self): return self
    monkeypatch.setattr(usage,'get_connection',lambda:Cursor())
    c=client()
    c.app.dependency_overrides[require_admin]=lambda:{'is_admin':True}
    response=c.get('/api/research-usage/summary')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'private, no-store'
    for field in ('confirmed_substack_signups','confirmed_paid_conversions','unique_readers','returning_readers'):
        assert response.json()[field] is None
