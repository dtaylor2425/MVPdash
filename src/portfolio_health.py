"""Read-time publication health; never changes the immutable ledger."""
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo
import pandas_market_calendars as mcal

@lru_cache(maxsize=32)
def sessions_through(day):
    schedule = mcal.get_calendar("NYSE").schedule(start_date=day-timedelta(days=20), end_date=day)
    return [stamp.date() for stamp in schedule.index]

def publication_health(payload, now=None):
    now = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    sessions = sessions_through(now.date())
    # Allow the scheduled 15:00 job thirty minutes to finish.
    due = [day for day in sessions if day < now.date() or (now.hour, now.minute) >= (15, 30)]
    expected = due[-1].isoformat() if due else None
    run = str((payload.get("snapshot") or {}).get("run_date") or "")[:10]
    prior = [day for day in sessions if day.isoformat() < run]
    expected_valuation = prior[-1].isoformat() if prior else None
    actual = str(payload.get("valuation_as_of") or "")[:10]
    publication_late = not run or bool(expected and run < expected)
    valuation_late = not actual or bool(expected_valuation and actual < expected_valuation)
    return {"status": "overdue" if publication_late or valuation_late else "current",
            "checked_at": now.isoformat(), "expected_run_date": expected,
            "expected_valuation_date": expected_valuation,
            "publication_overdue": publication_late, "valuation_overdue": valuation_late,
            "schedule": "Weekdays at 3 p.m. New York time on NYSE trading days; 30-minute completion allowance."}
