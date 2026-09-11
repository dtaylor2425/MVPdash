"""
tests/test_fx.py  (spec section 7)

Runs standalone (no network, no DB) on a synthetic input bundle, and also
under pytest if it is installed:

    python tests/test_fx.py
    pytest tests/test_fx.py

Covers the failure modes from the spec:
  1. antisymmetry            pair(A,B).z == -pair(B,A).z
  2. self-pair is zero       pair(A,A).z == 0
  3. rank consistency        top currency wins every pair it bases
  4. weights sum to 1.0      including after reweighting for dropped components
  5. no NaN reaches payload
  6. sanity anchor           highest real policy rate + positive momentum not bottom-3
  7. series validation       (skipped without FRED_API_KEY / network)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fx.components import InputBundle
from src.fx.frankfurter import FxData
from src.fx.scoring import (
    COMPONENT_WEIGHTS,
    composite_scores,
    cross_sectional_z,
    derive_pairs,
    pair_lookup,
    weights_sum_ok,
)
from src.fx.series_map import COMMODITY_SERIES, FRED_SERIES, SCORED
from src.fx.snapshot import build_fx_snapshot

# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------
# Deterministic per-currency knobs. USD is engineered to have the highest real
# policy rate AND rising yields -> it must not rank in the bottom three (check 6).
_POLICY = {"USD": 5.25, "EUR": 3.75, "JPY": 0.10, "GBP": 4.75, "CHF": 1.25,
           "CAD": 4.50, "AUD": 4.10, "NZD": 5.00, "SEK": 3.60, "NOK": 4.25}
_CPI_G = {"USD": 0.031, "EUR": 0.026, "JPY": 0.028, "GBP": 0.034, "CHF": 0.014,
          "CAD": 0.029, "AUD": 0.036, "NZD": 0.033, "SEK": 0.030, "NOK": 0.032}
_YSLOPE = {"USD": 0.9, "EUR": 0.2, "JPY": 0.05, "GBP": 0.1, "CHF": -0.1,
           "CAD": 0.0, "AUD": 0.1, "NZD": 0.15, "SEK": 0.05, "NOK": 0.0}
_FX_DRIFT = {"USD": 0.0, "EUR": 0.04, "JPY": 0.16, "GBP": 0.05, "CHF": -0.02,
             "CAD": 0.03, "AUD": 0.08, "NZD": 0.06, "SEK": 0.10, "NOK": 0.09}


def _synthetic_bundle() -> InputBundle:
    idx = pd.date_range(end=pd.Timestamp("2026-09-01"), periods=1200, freq="D")
    t = np.arange(len(idx)) / 365.0

    fred = {}
    for ccy in SCORED:
        fred[f"{ccy}__policy_rate"] = np.full(len(idx), _POLICY[ccy])
        y10_base = _POLICY[ccy] - 0.5
        fred[f"{ccy}__y10"] = y10_base + _YSLOPE[ccy] * t
        fred[f"{ccy}__cpi"] = 100.0 * np.exp(_CPI_G[ccy] * t)
    # USD alone has a real 2y series
    fred["USD__y2"] = (_POLICY["USD"] - 0.3) + _YSLOPE["USD"] * t

    for i, name in enumerate(COMMODITY_SERIES):
        fred[f"cmdty__{name}"] = 80.0 + 5.0 * i + 8.0 * np.sin(t + i) + 2.0 * t
    fred["macro__dollar_broad"] = 118.0 + 1.5 * np.sin(t) + 0.4 * t

    fred_df = pd.DataFrame(fred, index=idx)

    fx_cols = {"USD": np.ones(len(idx))}
    for ccy in [c for c in SCORED if c != "USD"] + ["CNY"]:
        base = {"EUR": 0.92, "JPY": 145.0, "GBP": 0.79, "CHF": 0.88, "CAD": 1.36,
                "AUD": 1.52, "NZD": 1.64, "SEK": 10.6, "NOK": 10.7, "CNY": 7.2}[ccy]
        drift = _FX_DRIFT.get(ccy, 0.05)
        fx_cols[ccy] = base * np.exp(drift * t / t[-1] * 0.12 + 0.01 * np.sin(t))
    fx_frame = pd.DataFrame(fx_cols, index=idx)

    # REER for all 10 so `valuation` is available in the main payload.
    reer = {}
    for k, ccy in enumerate(SCORED):
        m_idx = pd.date_range(end=idx[-1], periods=160, freq="ME")
        mt = np.arange(len(m_idx)) / 12.0
        reer[ccy] = pd.Series(100.0 + (k - 4) * 1.5 + 3.0 * np.sin(mt / 2 + k), index=m_idx)

    return InputBundle(fred=fred_df, fx=FxData(fx_frame), reer=reer)


# ---------------------------------------------------------------------------
# build once
# ---------------------------------------------------------------------------
_BUNDLE = _synthetic_bundle()
_PAYLOAD = build_fx_snapshot(bundle=_BUNDLE, history_months=8)
_SCORED_ENTRIES = [c for c in _PAYLOAD["currencies"] if c.get("scored")]


def _walk_no_nan(obj, path="root"):
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_no_nan(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_no_nan(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_antisymmetry():
    m = pair_lookup(_PAYLOAD["pairs"])
    for a in SCORED:
        for b in SCORED:
            if a == b:
                continue
            assert abs(m[(a, b)] + m[(b, a)]) < 1e-9, f"{a}/{b} not antisymmetric"


def test_self_pair_zero():
    m = pair_lookup(_PAYLOAD["pairs"])
    for a in SCORED:
        assert m[(a, a)] == 0.0


def test_triangle_only():
    # 45 unordered pairs stored, not 90
    assert len(_PAYLOAD["pairs"]) == 45
    seen = {(p["base"], p["quote"]) for p in _PAYLOAD["pairs"]}
    for b, q in seen:
        assert (q, b) not in seen


def test_rank_consistency():
    top = min(_SCORED_ENTRIES, key=lambda c: c["rank"])
    m = pair_lookup(_PAYLOAD["pairs"])
    for other in SCORED:
        if other == top["code"]:
            continue
        assert m[(top["code"], other)] > 0, (
            f"rank-1 {top['code']} does not win pair vs {other}"
        )


def test_weights_sum_to_one():
    comps = composite_scores(
        {c: cross_sectional_z({k: 1.0 + hash((c, k)) % 5 for k in SCORED}) for c in COMPONENT_WEIGHTS},
    )
    assert weights_sum_ok(comps)
    # after a real drop: only 3 components survive, weights must still renormalise
    partial = composite_scores({
        "carry": cross_sectional_z({k: k_i for k_i, k in enumerate(SCORED)}),
        "trend": cross_sectional_z({k: -k_i for k_i, k in enumerate(SCORED)}),
        "valuation": cross_sectional_z({k: (k_i % 3) for k_i, k in enumerate(SCORED)}),
    })
    assert weights_sum_ok(partial)
    for ccy, c in partial.items():
        assert abs(sum(c["weights"].values()) - 1.0) < 1e-9


def test_component_drop_below_six():
    # 5 valid currencies -> component dropped (empty dict)
    assert cross_sectional_z({"USD": 1.0, "EUR": 2.0, "JPY": 3.0, "GBP": 4.0, "CHF": 5.0}) == {}
    # 6 valid -> not dropped
    six = cross_sectional_z({"USD": 1, "EUR": 2, "JPY": 3, "GBP": 4, "CHF": 5, "CAD": 6})
    assert len(six) == 6


def test_winsorised():
    raw = {c: 0.0 for c in SCORED}
    raw["USD"] = 1000.0  # policy-crisis outlier
    z = cross_sectional_z(raw)
    assert max(abs(v) for v in z.values()) <= 2.5 + 1e-9


def test_no_nan_in_payload():
    _walk_no_nan(_PAYLOAD)
    for entry in _SCORED_ENTRIES:
        for name, comp in entry["components"].items():
            if not comp["available"]:
                assert comp["z"] is None and comp["display"] is None, name


def test_scores_in_range_and_ranked():
    ranks = sorted(c["rank"] for c in _SCORED_ENTRIES)
    assert ranks == list(range(1, 11))
    for c in _SCORED_ENTRIES:
        assert 0 <= c["score"] <= 100


def test_sanity_anchor():
    # USD has the highest real policy rate and rising yields (see knobs above).
    usd = next(c for c in _SCORED_ENTRIES if c["code"] == "USD")
    assert usd["rank"] <= 7, f"USD ranked {usd['rank']} — sign convention likely inverted"
    assert usd["components"]["policyMomentum"]["z"] > 0


def test_cny_display_only():
    cny = next(c for c in _PAYLOAD["currencies"] if c["code"] == "CNY")
    assert cny["scored"] is False
    assert cny["score"] is None
    assert "managed" in cny["label"].lower()


def test_reconciliation_shape():
    rec = _PAYLOAD["reconciliation"]
    assert set(rec) >= {"fxUsdScore", "macroDollarSignal", "agreement"}
    assert rec["agreement"] in {"aligned", "divergent", "unknown"}


def test_weights_metadata_sums_to_one():
    assert abs(sum(_PAYLOAD["meta"]["componentWeights"].values()) - 1.0) < 1e-9


def test_history_present():
    for c in _SCORED_ENTRIES:
        assert len(c["history"]["score"]) >= 6
        for pt in c["history"]["score"]:
            assert 0 <= pt["value"] <= 100


def test_series_validation_script():
    if not os.getenv("FRED_API_KEY"):
        print("SKIP test_series_validation_script (no FRED_API_KEY)")
        return
    from src.fx.fred_client import health_summary, validate_series

    summary = health_summary(validate_series())
    assert not summary["failing"], f"FRED series failed to resolve: {summary['failing']}"


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
