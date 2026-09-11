"""
tests/test_fx_breadth.py  (spec section 4A.7)

Validates the breadth & attribution module against its own hard invariants,
independent of the composite scoring model. Reuses the synthetic FxData built
in tests/test_fx.py so it exercises the same shape of data the job would see.

    python tests/test_fx_breadth.py
    pytest tests/test_fx_breadth.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fx import breadth as bd
from tests.test_fx import _BUNDLE  # reuse the synthetic FxData, built once

_FX = _BUNDLE.fx
_UNIVERSE = bd.G10
_L = bd.log_prices_usd(_FX, _UNIVERSE)
_S = bd.strengths_frame(_L)
_SNAP = bd.compute_breadth_snapshot(_FX, universe_name="g10")


def test_exact_identity():
    # strength(a) - strength(b) == log(a/b) exactly, for every pair, latest date.
    row_L = _L.iloc[-1]
    row_S = _S.iloc[-1]
    for a in _UNIVERSE:
        for b in _UNIVERSE:
            lhs = row_S[a] - row_S[b]
            rhs = row_L[a] - row_L[b]
            assert abs(lhs - rhs) < 1e-10, f"{a}/{b}: {lhs} vs {rhs}"


def test_strength_sums_to_zero():
    totals = _S.sum(axis=1)
    assert (totals.abs() < 1e-9).all(), f"max |sum| = {totals.abs().max()}"


def test_share_bounded():
    for base, entry in _SNAP["byBase"].items():
        if not entry.get("available"):
            continue
        assert 0.0 <= entry["share"] <= 1.0, f"{base}: share={entry['share']}"
        for cross in entry["crosses"]:
            assert 0.0 <= cross["baseShare"] <= 1.0


def test_breadth_bounds():
    for base, entry in _SNAP["byBase"].items():
        if not entry.get("available"):
            continue
        m = len(_UNIVERSE) - 1
        assert -m <= entry["breadth"] <= m
        assert entry["upN"] + entry["downN"] <= m


def test_base_invariance():
    # Deriving the log-price panel through USD vs through EUR must give
    # identical strengths (the additive base-offset cancels in demeaning).
    l_usd = bd.log_prices_usd(_FX, _UNIVERSE)
    l_eur = bd.log_prices_via_base(_FX, _UNIVERSE, "EUR")
    idx = l_usd.index.intersection(l_eur.index)
    s_usd = bd.strengths_frame(l_usd.loc[idx])
    s_eur = bd.strengths_frame(l_eur.loc[idx])
    diff = (s_usd - s_eur).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9, f"base mismatch, max diff {np.nanmax(diff)}"


def test_move_decomposes_into_from_base_and_from_quote():
    for base, entry in _SNAP["byBase"].items():
        if not entry.get("available"):
            continue
        for cross in entry["crosses"]:
            # each field is independently rounded to 6dp in base_metrics, so allow
            # for that rounding rather than requiring bit-exact equality
            assert abs(cross["move"] - (cross["fromBase"] + cross["fromQuote"])) < 5e-6


def test_verdict_is_one_of_taxonomy():
    allowed = {"BASE_DRIVEN", "BASE_PLUS_STRESS", "OTHERS_MOVING", "QUIET_MIXED"}
    for base, entry in _SNAP["byBase"].items():
        if entry.get("available"):
            assert entry["verdict"] in allowed


def test_no_nan():
    def walk(obj, path="root"):
        if isinstance(obj, float):
            assert math.isfinite(obj), f"non-finite at {path}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
    walk(_SNAP)


def test_majors_vs_g10_both_computable():
    majors_snap = bd.compute_breadth_snapshot(_FX, universe_name="majors")
    assert set(majors_snap["byBase"]) == set(bd.MAJORS)
    assert set(_SNAP["byBase"]) == set(bd.G10)


def test_history_present():
    entry = _SNAP["history"].get("USD", {})
    assert len(entry.get("share", [])) >= 6


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
