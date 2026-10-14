from datetime import datetime
from zoneinfo import ZoneInfo
from src.portfolio_health import publication_health

def health(run, valuation, now):
    return publication_health({"snapshot":{"run_date":run},"valuation_as_of":valuation},datetime.fromisoformat(now).replace(tzinfo=ZoneInfo("America/New_York")))

def test_deadline_and_weekend():
    assert health("2026-10-13","2026-10-12","2026-10-14T15:29")["status"] == "current"
    assert health("2026-10-13","2026-10-12","2026-10-14T15:30")["publication_overdue"]
    assert health("2026-10-09","2026-10-08","2026-10-11T18:00")["status"] == "current"

def test_holiday_and_stale_valuation():
    assert health("2026-12-24","2026-12-23","2026-12-25T18:00")["status"] == "current"
    assert health("2026-10-14","2026-10-09","2026-10-14T16:00")["valuation_overdue"]

def test_dst_keeps_new_york_deadline():
    assert health("2026-10-30","2026-10-29","2026-11-02T15:29")["status"] == "current"
    assert health("2026-10-30","2026-10-29","2026-11-02T15:30")["publication_overdue"]
