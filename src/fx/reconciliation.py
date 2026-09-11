"""
src/fx/reconciliation.py

Reconcile the FX page's USD score with the existing macro `dollar` signal
(spec section 4.5). Two public models that contradict each other is worse than
one, so the snapshot exposes both and a plain aligned/divergent verdict.

The macro dollar signal here mirrors api/routers/signals.py: last value of the
broad dollar index (DTWEXBGS) and its ~252-observation z-score.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.fx.series_map import DOLLAR_BROAD_SERIES

_FX_HIGH = 55.0
_FX_LOW = 45.0


def _zscore(series: pd.Series, window: int = 252) -> Optional[float]:
    s = series.dropna()
    if len(s) < 30:
        return None
    tail = s.iloc[-min(window, len(s)):]
    sd = float(tail.std())
    if sd == 0 or not math.isfinite(sd):
        return 0.0
    return round(float((tail.iloc[-1] - tail.mean()) / sd), 3)


def macro_dollar_signal(fred: pd.DataFrame, logical: str = "macro__dollar_broad") -> dict:
    col = None
    if logical in fred.columns:
        col = fred[logical].dropna()
    elif DOLLAR_BROAD_SERIES in fred.columns:
        col = fred[DOLLAR_BROAD_SERIES].dropna()
    if col is None or col.empty:
        return {"value": None, "zscore": None, "direction": 0}
    value = round(float(col.iloc[-1]), 1)
    z = _zscore(col)
    direction = 0
    if z is not None:
        direction = 1 if z > 0.5 else -1 if z < -0.5 else 0
    return {"value": value, "zscore": z, "direction": direction}


def build_reconciliation(fx_usd_score: Optional[float], fred: pd.DataFrame) -> dict:
    macro = macro_dollar_signal(fred)
    agreement = "unknown"
    z = macro.get("zscore")
    if fx_usd_score is not None and z is not None:
        divergent = (fx_usd_score > _FX_HIGH and z < 0) or (fx_usd_score < _FX_LOW and z > 0)
        agreement = "divergent" if divergent else "aligned"
    return {
        "fxUsdScore": None if fx_usd_score is None else round(fx_usd_score),
        "macroDollarSignal": macro,
        "agreement": agreement,
    }
