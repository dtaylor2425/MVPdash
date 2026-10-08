from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.private_options_flow import router


def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_scanner_requires_internal_token():
    with patch.dict('os.environ', {'INTERNAL_OPTIONS_API_SECRET': 'test-secret'}):
        assert client().get('/api/private/options-flow/scanner').status_code == 404


def test_scanner_empty_and_invalid_window():
    with patch.dict('os.environ', {'INTERNAL_OPTIONS_API_SECRET': 'test-secret'}), \
         patch('api.services.options_scanner_store.latest_board', return_value=None):
        c = client()
        headers = {'X-Internal-Options-Token': 'test-secret'}
        response = c.get('/api/private/options-flow/scanner', headers=headers)
        assert response.status_code == 200
        assert response.json()['status'] == 'not_published'
        assert response.headers['cache-control'] == 'no-store'
        assert c.get('/api/private/options-flow/scanner?window=invalid', headers=headers).status_code == 422


def test_scanner_exposes_partial_coverage_without_inventing_rows():
    board = {'run': {'market_date': '2026-10-07'}, 'symbols': [],
             'coverage': {'total': 3, 'completed': 0, 'pending': 2, 'failed': 1, 'excluded': 0},
             'completed': [], 'pending': ['AAPL', 'MSFT'], 'failed': ['NVDA'], 'excluded': []}
    connection = MagicMock()
    connection.__enter__.return_value.execute.return_value.fetchall.return_value = []
    with patch.dict('os.environ', {'INTERNAL_OPTIONS_API_SECRET': 'test-secret'}), \
         patch('api.services.options_scanner_store.latest_board', return_value=board) as read, \
         patch('api.routers.private_options_flow.get_connection', return_value=connection):
        response = client().get('/api/private/options-flow/scanner?session=2026-10-07',
                                headers={'X-Internal-Options-Token': 'test-secret'})
        assert response.status_code == 200
        out = response.json()
        assert out['rows'] == []
        assert out['failedTickers'] == ['NVDA']
        assert out['coverage']['pending'] == 2
        assert str(read.call_args.kwargs['session']) == '2026-10-07'
