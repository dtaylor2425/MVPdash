"""
api/services/options_flow_calendar.py

NYSE session helpers + the LIVE / DELAYED / CLOSED / STALE data-status rule.
Used by the worker (which trading date to compute) and the private API
(what status badge to show). Falls back to a weekday/09:30-16:00 approximation
if pandas_market_calendars is not installed -- holidays would then be treated
as trading days, which only costs a harmless empty run.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
# ETF options keep trading 15 minutes after the equity close.
OPTIONS_TAIL = timedelta(minutes=15)

STATUS_LIVE = "LIVE"
STATUS_DELAYED = "DELAYED"
STATUS_CLOSED = "CLOSED"
STATUS_STALE = "STALE"


@lru_cache(maxsize=1)
def _calendar():
    try:
        import pandas_market_calendars as mcal
        return mcal.get_calendar("NYSE")
    except Exception:
        return None


@lru_cache(maxsize=512)
def session_bounds(d: date) -> Optional[Tuple[datetime, datetime]]:
    """(open, close) as aware UTC datetimes for a trading day, else None."""
    cal = _calendar()
    if cal is not None:
        sched = cal.schedule(start_date=d, end_date=d)
        if sched.empty:
            return None
        row = sched.iloc[0]
        return row["market_open"].to_pydatetime(), row["market_close"].to_pydatetime()
    if d.weekday() >= 5:
        return None
    o = datetime(d.year, d.month, d.day, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    c = datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY).astimezone(timezone.utc)
    return o, c


def is_trading_day(d: date) -> bool:
    return session_bounds(d) is not None


def previous_trading_day(d: date) -> date:
    cur = d - timedelta(days=1)
    for _ in range(15):
        if is_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    raise RuntimeError("No trading day found in the 15 days before {}".format(d))


@lru_cache(maxsize=256)
def sessions_between(start: date, end: date) -> Tuple[date, ...]:
    """Trading sessions in [start, end], ascending."""
    if end < start:
        return ()
    cal = _calendar()
    if cal is not None:
        return tuple(ts.date() for ts in cal.schedule(start_date=start, end_date=end).index)
    out, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return tuple(out)


@lru_cache(maxsize=1024)
def sessions_before(d: date, n: int) -> Tuple[date, ...]:
    """The n trading sessions strictly before d, NEWEST FIRST. Point-in-time
    lookback windows are built from this so nothing on/after d can enter."""
    out: list = []
    hi = d - timedelta(days=1)
    span = max(14, int(n * 1.6) + 10)
    while len(out) < n:
        lo = hi - timedelta(days=span)
        out.extend(reversed(sessions_between(lo, hi)))
        hi = lo - timedelta(days=1)
        if hi.year < 2000:  # no data that old; stop rather than loop forever
            break
    return tuple(out[:n])


def last_completed_session(now: Optional[datetime] = None) -> date:
    """Most recent session whose options trading (incl. the 15-minute tail) has finished."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today = now.astimezone(NY).date()
    end = session_end(today)
    if end is not None and now >= end:
        return today
    return previous_trading_day(today)


def target_trading_date(now: Optional[datetime] = None) -> date:
    """Today if it is a trading day and the session has opened, else the last completed session."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today = now.astimezone(NY).date()
    b = session_bounds(today)
    if b is not None and now >= b[0]:
        return today
    return previous_trading_day(today)


def market_is_open(now: Optional[datetime] = None) -> bool:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    b = session_bounds(now.astimezone(NY).date())
    return b is not None and b[0] <= now <= b[1] + OPTIONS_TAIL


def session_end(d: date) -> Optional[datetime]:
    b = session_bounds(d)
    return None if b is None else b[1] + OPTIONS_TAIL


def compute_data_status(
    market_date: Optional[date],
    published_at: Optional[datetime],
    now: Optional[datetime] = None,
    live_max_age_min: int = 20,
    stale_after_min: int = 60,
) -> Dict[str, Any]:
    """
    Status of the newest published snapshot.

      session open   LIVE     snapshot is for today and was published within live_max_age_min
                     DELAYED  ... within stale_after_min
                     STALE    older than that, or not today's date
      session closed CLOSED   snapshot is for the last completed session and was published
                              after that session's close (i.e. it is the final print)
                     STALE    anything else (missed the final run / an older session)

    The page keeps rendering the last good snapshot in every case; this only
    drives the badge.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if market_date is None or published_at is None:
        return {"status": STATUS_STALE, "reason": "no snapshot published", "ageMinutes": None}
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_min = (now - published_at).total_seconds() / 60.0

    today = now.astimezone(NY).date()
    if market_is_open(now):
        if market_date != today:
            return {"status": STATUS_STALE, "reason": "snapshot is for {}, session is live".format(market_date),
                    "ageMinutes": age_min}
        if age_min <= live_max_age_min:
            return {"status": STATUS_LIVE, "reason": None, "ageMinutes": age_min}
        if age_min <= stale_after_min:
            return {"status": STATUS_DELAYED, "reason": "last refresh {:.0f} min ago".format(age_min),
                    "ageMinutes": age_min}
        return {"status": STATUS_STALE, "reason": "last refresh {:.0f} min ago".format(age_min),
                "ageMinutes": age_min}

    last_session = target_trading_date(now)
    if market_date != last_session:
        return {"status": STATUS_STALE, "reason": "snapshot is for {}, last session was {}".format(
            market_date, last_session), "ageMinutes": age_min}
    close = session_bounds(market_date)[1]
    if published_at >= close:
        return {"status": STATUS_CLOSED, "reason": None, "ageMinutes": age_min}
    return {"status": STATUS_STALE, "reason": "final post-close refresh missing", "ageMinutes": age_min}
