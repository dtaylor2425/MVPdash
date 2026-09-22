"""
api/services/options_flow_phase1_scope.py

The actual Phase 1 scope (which tickers, which date ranges) as launched, shared by
jobs/options_flow_backfill.py (--status), scripts/generate_options_flow_qa_report.py and
scripts/export_options_flow_research_dataset.py so they can never drift apart. Control-plane
only -- no analytics.
"""

from __future__ import annotations

from datetime import date

PHASE1_TICKERS = ["SPY", "QQQ", "IWM", "SMH", "TLT", "GLD"]
PHASE1_FULL_FLOW_RANGE = (date(2026, 6, 26), date(2026, 9, 21))     # 60 sessions
PHASE1_WARMUP_RANGE = (date(2025, 6, 25), date(2026, 6, 25))        # 252 sessions immediately prior
