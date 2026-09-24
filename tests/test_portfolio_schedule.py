from datetime import datetime, timezone

from jobs.nightly_portfolio_refresh import _scheduled_hour_matches


def test_three_pm_new_york_tracks_daylight_saving():
    assert _scheduled_hour_matches(datetime(2026, 9, 24, 19, tzinfo=timezone.utc), 15)
    assert not _scheduled_hour_matches(datetime(2026, 9, 24, 20, tzinfo=timezone.utc), 15)
    assert not _scheduled_hour_matches(datetime(2026, 12, 1, 19, tzinfo=timezone.utc), 15)
    assert _scheduled_hour_matches(datetime(2026, 12, 1, 20, tzinfo=timezone.utc), 15)


def test_daylight_saving_boundary_and_container_start_delay():
    assert _scheduled_hour_matches(datetime(2026, 3, 6, 20, 7, tzinfo=timezone.utc), 15)
    assert _scheduled_hour_matches(datetime(2026, 3, 9, 19, 7, tzinfo=timezone.utc), 15)
    assert _scheduled_hour_matches(datetime(2026, 10, 30, 19, tzinfo=timezone.utc), 15)
    assert _scheduled_hour_matches(datetime(2026, 11, 2, 20, tzinfo=timezone.utc), 15)
