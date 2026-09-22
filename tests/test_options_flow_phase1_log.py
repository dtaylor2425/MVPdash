"""
tests/test_options_flow_phase1_log.py  (Phase 1 progress-log formatting -- pure, synthetic)

    pytest tests/test_options_flow_phase1_log.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services import options_flow_phase1_log as plog  # noqa: E402


def test_format_line_full_flow_has_all_fields():
    line = plog.format_line("SPY", date(2026, 6, 26), "full_flow", "success", 151.1, 32, 1741936,
                            1.0, 1.0, 0.981, -0.003, -1.79e9)
    assert "SPY" in line and "2026-06-26" in line and "full_flow" in line and "success" in line
    assert "runtime=151.1s" in line and "requests=32" in line and "trades=1,741,936" in line
    assert "classCov=100.0%" in line and "greekCov=100.0%" in line and "oiCov=98.1%" in line
    assert "sentiment=-0.0030" in line and "deltaImb=-$1.79B" in line


def test_format_line_iv_warmup_has_explicit_na_not_zero():
    line = plog.format_line("SPY", date(2026, 6, 1), "iv_warmup", "success", 43.0, 18, None,
                            None, None, None, None, None)
    assert "trades=n/a" in line and "classCov=n/a" in line and "sentiment=n/a" in line and "deltaImb=n/a" in line
    assert "trades=0" not in line and "sentiment=+0.0000" not in line


def test_format_line_failed_includes_reason():
    line = plog.format_line("QQQ", date(2026, 7, 1), "full_flow", "failed", 5.2, 2, None,
                            None, None, None, None, None, reason="no_trade_data")
    assert "failed" in line and "reason=no_trade_data" in line


def test_parse_logged_keys_roundtrips_format_line():
    now = datetime(2026, 9, 22, tzinfo=timezone.utc)
    lines = [
        plog.format_line("SPY", date(2026, 6, 26), "full_flow", "success", 1.0, 1, 1, 1.0, 1.0, 1.0, 0.1, 1.0, now=now),
        plog.format_line("QQQ", date(2026, 6, 26), "iv_warmup", "success", 1.0, 1, None, None, None, None, None, None, now=now),
    ]
    keys = plog.parse_logged_keys("\n".join(lines))
    assert keys == {("SPY", "2026-06-26", "full_flow"), ("QQQ", "2026-06-26", "iv_warmup")}


def test_parse_logged_keys_ignores_checkpoint_blocks_and_blank_lines():
    text = "\n".join([
        plog.format_line("SPY", date(2026, 6, 26), "full_flow", "success", 1, 1, 1, 1, 1, 1, 0.1, 1),
        "",
        plog.format_checkpoint(10, 0, 0, 360, 0, 0, 0, 1512, datetime(2026, 9, 21, tzinfo=timezone.utc)),
        "",
    ])
    keys = plog.parse_logged_keys(text)
    assert keys == {("SPY", "2026-06-26", "full_flow")}


def test_checkpoint_format_and_elapsed():
    started = datetime(2026, 9, 21, 19, 14, 46, tzinfo=timezone.utc)
    now = datetime(2026, 9, 22, 1, 30, 0, tzinfo=timezone.utc)
    text = plog.format_checkpoint(31, 0, 0, 360, 0, 0, 0, 1512, started, now)
    assert "Full flow: 31 / 360" in text
    assert "Warmup: 0 / 1512" in text
    assert "Failed: 0" in text and "Partial: 0" in text
    assert "Elapsed: 6h15m14s" in text


def test_checkpoint_no_start_time_is_na():
    text = plog.format_checkpoint(5, 0, 0, 360, 0, 0, 0, 1512, None)
    assert "Elapsed: n/a" in text


def test_checkpoint_boundaries_crossed_single_and_batch():
    assert plog.checkpoint_boundaries_crossed(9, 10) == [10]
    assert plog.checkpoint_boundaries_crossed(8, 9) == []
    assert plog.checkpoint_boundaries_crossed(5, 25) == [10, 20]
    assert plog.checkpoint_boundaries_crossed(30, 30) == []
    assert plog.checkpoint_boundaries_crossed(0, 0) == []
    assert plog.checkpoint_boundaries_crossed(10, 10) == []


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if failed else 0)
