"""FX currency-strength backend.

See docs/FX-BACKEND-SPEC.md for the design. The spec is written against a
Next.js/JS layout; this implementation follows the repo's existing Python /
FastAPI conventions instead. Mapping:

    src/lib/fx/seriesMap.js      -> src/fx/series_map.py
    scripts/validateFxSeries.js  -> scripts/validate_fx_series.py
    src/app/api/fx/*             -> api/routers/fx.py
    snapshot writer + cron       -> jobs/fx_snapshot_job.py
"""
