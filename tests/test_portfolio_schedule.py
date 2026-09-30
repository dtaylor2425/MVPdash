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


def test_publication_log_explains_pending_without_claiming_execution():
    from jobs.nightly_portfolio_refresh import _publication_summary
    text = _publication_summary({"holdings": [{"ticker": "A"}], "valuation_as_of": "2026-09-28",
        "pending_allocation": {"decision_date": "2026-09-28", "execute_not_before": "2026-09-29"},
        "trade_queue": []}, {"turnover": 0})
    assert "Valuation through=2026-09-28" in text
    assert "Executed reference sessions=none" in text
    assert "Pending decision=2026-09-28" in text
    assert "recorded after that session completes" in text


def test_publication_log_keeps_execution_separate_from_publication():
    from jobs.nightly_portfolio_refresh import _publication_summary
    text = _publication_summary({"holdings": [{"ticker": "A"}], "valuation_as_of": "2026-09-29",
        "trade_queue": [{"date": "2026-09-29"}, {"date": "2026-09-29"}]}, {"turnover": .1548})
    assert "Executed reference sessions=2026-09-29" in text
    assert "Turnover=15.48%" in text
    assert "No pending allocation" in text
