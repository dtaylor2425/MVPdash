from datetime import datetime, timezone
import pytest
from api.services.volatility_overview import overview_from_series
NOW = datetime(2026, 11, 20, 23, tzinfo=timezone.utc)
def test_matching_dates_and_change():
    result = overview_from_series({'vix':[('2026-11-19',20),('2026-11-20',22)],'vix3m':[('2026-11-20',25)]}, NOW)
    assert result['latest']['ratio'] == .88
    assert result['latest']['change_points'] == 2
    assert result['points'][0]['ratio'] is None
    assert not result['stale']
def test_incomplete_session_excluded():
    result = overview_from_series({'vix':[('2026-11-19',20),('2026-11-20',22)]}, datetime(2026,11,20,19,tzinfo=timezone.utc))
    assert result['as_of'] == '2026-11-19'
    assert not result['stale']
def test_weekend_and_invalid_values():
    result = overview_from_series({'vix':[('2026-11-19',20),('2026-11-20',float('nan'))]}, datetime(2026,11,22,20,tzinfo=timezone.utc))
    assert result['stale']
    assert result['expected_session'] == '2026-11-20'
def test_empty_fails_instead_of_zero():
    with pytest.raises(ValueError): overview_from_series({'vix':[]}, NOW)
