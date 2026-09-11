"""
scripts/validate_fx_series.py  (spec section 3.3)

Request every FRED series ID in src/fx/series_map.py and report OK / 404 / stale
(no real observation within that series' own publication-frequency allowance --
see STALE_DAYS in src/fx/series_map.py; normal OECD publication lag is not
staleness). Run this before shipping a snapshot and as part of the ingest job.

    python scripts/validate_fx_series.py            # table + non-zero exit on any 404
    python scripts/validate_fx_series.py --json     # machine-readable

Exit codes: 0 = all resolve (stale allowed), 1 = at least one 404/error,
2 = FRED_API_KEY missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fx.fred_client import health_summary, validate_series  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    try:
        checks = validate_series()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = health_summary(checks)

    if args.json:
        print(json.dumps({"summary": summary, "checks": [c.as_dict() for c in checks]}, indent=2))
    else:
        width = max(len(c.logical) for c in checks)
        print(f"{'LOGICAL':<{width}}  {'SERIES ID':<20} {'CONF':<7} {'STATUS':<7} {'LAST OBS':<12} DETAIL")
        print("-" * (width + 60))
        for c in checks:
            print(
                f"{c.logical:<{width}}  {c.series_id:<20} {c.confidence:<7} "
                f"{c.status:<7} {str(c.last_obs or '-'):<12} {c.detail}"
            )
        print("-" * (width + 60))
        print("counts:", summary["counts"])
        if summary["failing"]:
            print(f"\n{len(summary['failing'])} series FAILED to resolve — fix src/fx/series_map.py "
                  "and mark affected components unavailable until then.")
        if summary["stale"]:
            print(f"{len(summary['stale'])} series are stale (per their own publication-frequency "
                  "threshold) — the component will be dropped and reweighted for the affected currencies.")

    return 1 if summary["failing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
