"""
api/services/options_flow_metrics.py

Pure derived-analytics for Macro Options Flow. Takes ThetaData trade_quote /
first-order-Greek / open-interest frames in, returns JSON-safe dicts out.
No ThetaData, no Postgres, no network -- imported by the Python 3.12 worker
(jobs/options_flow_refresh.py) and unit-tested on the API's Python 3.11.

Conventions (kept identical everywhere so the frontend can rely on them):
    * implied vols are DECIMALS (0.152 == 15.2%); IV changes/spreads are
      decimal differences (0.008 == +0.8 vol points).
    * premium / delta figures are USD.
    * sentiment and ratios are in [-1, +1].
    * aggressor score: +1 = lifted the ask, -1 = hit the bid, fractional
      inside the spread. directional premium = aggressor * (+1 call / -1 put)
      * premium, so call bought = bullish, put sold = bullish.
    * put deltas are used exactly as ThetaData reports them (negative);
      they are never re-signed here.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"
CONTRACT_MULTIPLIER = 100

# Calculation/schema version for every snapshot this module produces (live, backfill, iv-warmup).
# Bump this whenever trade classification, the Greek as-of join, IV/skew/term-structure math, DTE
# buckets, OI matching, or the sentiment/delta formulas change -- consumers (the research dataset
# export, any backtest) must never silently mix observations computed under different versions.
METHODOLOGY_VERSION = "1.0.0"

LABEL_BULLISH = "BULLISH"
LABEL_LEAN_BULLISH = "LEAN BULLISH"
LABEL_NEUTRAL = "NEUTRAL"
LABEL_LEAN_BEARISH = "LEAN BEARISH"
LABEL_BEARISH = "BEARISH"
LABEL_LOW_CONFIDENCE = "LOW CONFIDENCE"

DTE_BUCKETS: List[Tuple[str, int, Optional[int]]] = [
    ("0DTE", 0, 0),
    ("1-7D", 1, 7),
    ("8-30D", 8, 30),
    ("31-60D", 31, 60),
    ("60D+", 61, None),
]

TRADE_COLUMNS = [
    "ts", "expiration", "exp_i", "dte", "strike", "strike_k", "is_call", "size",
    "price", "bid", "ask", "premium", "valid", "aggressor", "direction",
    "dir_premium", "delta", "iv", "underlying", "signed_delta_contracts",
    "signed_dollar_delta", "oi",
]


class StageTimer:
    """Thread-safe accumulator of seconds per named stage (cumulative thread time, so parallel
    network stages can sum to more than wall time). Pure bookkeeping -- never affects results."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.seconds: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self.seconds[name] = self.seconds.get(name, 0.0) + seconds
            self.calls[name] = self.calls.get(name, 0) + 1

    def stage(self, name: str):
        import contextlib
        import time as _time

        @contextlib.contextmanager
        def _cm():
            t0 = _time.perf_counter()
            try:
                yield
            finally:
                self.add(name, _time.perf_counter() - t0)
        return _cm()

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {k: round(v, 3) for k, v in self.seconds.items()}


class _NullTimer:
    def stage(self, name: str):
        import contextlib
        return contextlib.nullcontext()

    def add(self, name: str, seconds: float) -> None:
        pass


class MissingColumnsError(ValueError):
    """A ThetaData frame lacked columns we cannot compute without."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _pick(df: pd.DataFrame, candidates: Sequence[str], what: str, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise MissingColumnsError(
            "ThetaData frame has none of {} for {}; columns present: {}".format(
                list(candidates), what, list(df.columns)
            )
        )
    return None


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def _is_call(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().str.startswith("C")


def _exp_int(expiration: Any) -> int:
    d = pd.Timestamp(expiration).date()
    return d.year * 10000 + d.month * 100 + d.day


def _strike_key(strike: pd.Series) -> pd.Series:
    return np.rint(strike.to_numpy(dtype="float64") * 1000).astype("int64")


def _clean(obj: Any, ndigits: int = 6) -> Any:
    """Recursively make a structure JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, ndigits) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits) + 0.0     # + 0.0 turns -0.0 into 0.0
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def fmt_money(v: Optional[float]) -> str:
    """+$1.42B / -$310.5M / +$12.0K -- used in worker diagnostics."""
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else "-"
    a = abs(v)
    for div, suffix, nd in ((1e12, "T", 2), (1e9, "B", 2), (1e6, "M", 1), (1e3, "K", 1)):
        if a >= div:
            return "{}${:.{}f}{}".format(sign, a / div, nd, suffix)
    return "{}${:.0f}".format(sign, a)


# --------------------------------------------------------------------------
# raw ThetaData -> normalized frames
# --------------------------------------------------------------------------

def normalize_trade_quote(raw: pd.DataFrame, expiration: Any, market_date: date) -> pd.DataFrame:
    """option_history_trade_quote rows -> compact frame (one expiration)."""
    if raw is None or len(raw) == 0:
        return _empty_trades()
    ts_col = _pick(raw, ["trade_timestamp", "timestamp", "trade_time"], "trade timestamp")
    px_col = _pick(raw, ["price", "trade_price"], "trade price")
    size_col = _pick(raw, ["size", "trade_size"], "trade size")
    bid_col = _pick(raw, ["bid", "bid_price"], "bid")
    ask_col = _pick(raw, ["ask", "ask_price"], "ask")
    strike_col = _pick(raw, ["strike"], "strike")
    right_col = _pick(raw, ["right"], "right")

    exp_ts = pd.Timestamp(expiration)
    strike = _num(raw[strike_col])
    out = pd.DataFrame({
        "ts": _to_utc(raw[ts_col]),
        "strike": strike,
        "is_call": _is_call(raw[right_col]),
        "size": _num(raw[size_col]),
        "price": _num(raw[px_col]),
        "bid": _num(raw[bid_col]),
        "ask": _num(raw[ask_col]),
    })
    out["expiration"] = exp_ts.date().isoformat()
    out["exp_i"] = _exp_int(exp_ts)
    out["dte"] = (exp_ts.date() - market_date).days
    out["strike_k"] = _strike_key(strike)
    return out.reset_index(drop=True)


def normalize_greeks(raw: pd.DataFrame, expiration: Any, market_date: date) -> pd.DataFrame:
    """option_history_greeks_first_order rows -> compact frame (one expiration)."""
    cols = ["ts", "expiration", "exp_i", "dte", "strike", "strike_k", "is_call",
            "delta", "iv", "underlying", "bid", "ask"]
    if raw is None or len(raw) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}).astype(
            {"ts": "datetime64[ns, UTC]"}
        )
    ts_col = _pick(raw, ["timestamp", "greek_timestamp"], "greeks timestamp")
    delta_col = _pick(raw, ["delta"], "delta")
    iv_col = _pick(raw, ["implied_vol", "implied_volatility", "iv"], "implied vol")
    und_col = _pick(raw, ["underlying_price", "underlying"], "underlying price")
    strike_col = _pick(raw, ["strike"], "strike")
    right_col = _pick(raw, ["right"], "right")
    bid_col = _pick(raw, ["bid"], "bid", required=False)
    ask_col = _pick(raw, ["ask"], "ask", required=False)

    exp_ts = pd.Timestamp(expiration)
    strike = _num(raw[strike_col])
    out = pd.DataFrame({
        "ts": _to_utc(raw[ts_col]),
        "strike": strike,
        "is_call": _is_call(raw[right_col]),
        "delta": _num(raw[delta_col]),
        "iv": _num(raw[iv_col]),
        "underlying": _num(raw[und_col]),
        "bid": _num(raw[bid_col]) if bid_col else np.nan,
        "ask": _num(raw[ask_col]) if ask_col else np.nan,
    })
    out["expiration"] = exp_ts.date().isoformat()
    out["exp_i"] = _exp_int(exp_ts)
    out["dte"] = (exp_ts.date() - market_date).days
    out["strike_k"] = _strike_key(strike)
    return out.reset_index(drop=True)


def normalize_open_interest(raw: pd.DataFrame) -> pd.DataFrame:
    """option_history_open_interest rows -> (exp_i, strike_k, is_call, open_interest)."""
    cols = ["exp_i", "strike_k", "is_call", "open_interest"]
    if raw is None or len(raw) == 0:
        return pd.DataFrame({"exp_i": pd.Series(dtype="int64"), "strike_k": pd.Series(dtype="int64"),
                             "is_call": pd.Series(dtype="bool"), "open_interest": pd.Series(dtype="float64")})
    oi_col = _pick(raw, ["open_interest", "oi"], "open interest")
    exp_col = _pick(raw, ["expiration"], "expiration")
    strike_col = _pick(raw, ["strike"], "strike")
    right_col = _pick(raw, ["right"], "right")
    ts_col = _pick(raw, ["timestamp"], "timestamp", required=False)

    df = pd.DataFrame({
        "exp_i": pd.to_datetime(raw[exp_col]).map(_exp_int).astype("int64"),
        "strike_k": _strike_key(_num(raw[strike_col])),
        "is_call": _is_call(raw[right_col]),
        "open_interest": _num(raw[oi_col]),
    })
    if ts_col:
        df["_ts"] = pd.to_datetime(raw[ts_col], utc=True)
        df = df.sort_values("_ts").drop(columns="_ts")
    return df.drop_duplicates(["exp_i", "strike_k", "is_call"], keep="last").reset_index(drop=True)[cols]


def _empty_trades() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in TRADE_COLUMNS})
    return df.astype({
        "ts": "datetime64[ns, UTC]", "expiration": "object", "exp_i": "int64", "dte": "int64",
        "strike_k": "int64", "is_call": "bool", "valid": "bool",
    })


# --------------------------------------------------------------------------
# trade classification
# --------------------------------------------------------------------------

def aggressor_score(price, bid, ask) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised aggressor score + quote-validity mask.

        price >= ask -> +1 ; price <= bid -> -1
        else clip((price - mid) / ((ask - bid) / 2), -1, 1)

    Invalid where ask <= 0, bid < 0, ask < bid or price <= 0 (or non-finite).
    Invalid rows score 0 -- callers must gate on the mask, not the score.
    """
    price = np.asarray(price, dtype="float64")
    bid = np.asarray(bid, dtype="float64")
    ask = np.asarray(ask, dtype="float64")
    finite = np.isfinite(price) & np.isfinite(bid) & np.isfinite(ask)
    valid = finite & (ask > 0) & (bid >= 0) & (ask >= bid) & (price > 0)

    half = (ask - bid) / 2.0
    mid = (bid + ask) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = np.where(half > 0, (price - mid) / half, 0.0)
    inside = np.clip(inside, -1.0, 1.0)
    score = np.where(price >= ask, 1.0, np.where(price <= bid, -1.0, inside))
    score = np.where(valid, score, 0.0)
    return score, valid


def classify_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds premium / valid / aggressor / direction / dir_premium.

    Needs: price, size, bid, ask, is_call. `valid` also requires size > 0.
    `premium` is computed for any row with price > 0 and size > 0 (so the
    classification-coverage denominator includes trades we could not
    classify); dir_premium is 0 for invalid rows.
    """
    out = df.copy()
    size = out["size"].to_numpy(dtype="float64")
    price = out["price"].to_numpy(dtype="float64")
    score, quote_ok = aggressor_score(price, out["bid"], out["ask"])
    size_ok = np.isfinite(size) & (size > 0)
    px_ok = np.isfinite(price) & (price > 0)

    valid = quote_ok & size_ok
    out["premium"] = np.where(px_ok & size_ok, price * size * CONTRACT_MULTIPLIER, 0.0)
    out["valid"] = valid
    out["aggressor"] = np.where(valid, score, 0.0)
    out["direction"] = np.where(out["is_call"].to_numpy(dtype=bool), 1.0, -1.0)
    out["dir_premium"] = out["aggressor"] * out["direction"] * out["premium"]
    return out


def attach_greeks(trades: pd.DataFrame, greeks: pd.DataFrame, tolerance_min: int = 15) -> pd.DataFrame:
    """
    Join the nearest PRIOR first-order Greek observation (same expiration,
    strike, right) to each trade; trades with no prior observation inside
    the tolerance fall back to the nearest observation either side, still
    inside the tolerance. Adds delta / iv / underlying and the signed delta
    columns:

        signed_delta_contracts = aggressor * delta * size * 100
        signed_dollar_delta    = signed_delta_contracts * underlying
    """
    out = trades.copy()
    for c in ("delta", "iv", "underlying"):
        out[c] = np.nan

    if len(out) and greeks is not None and len(greeks):
        g = greeks[np.isfinite(greeks["delta"].astype("float64"))].copy()
        g["iv"] = g["iv"].where(g["iv"] > 0)
        g = g.sort_values("ts")[["ts", "exp_i", "strike_k", "is_call", "delta", "iv", "underlying"]]
        g["is_call"] = g["is_call"].astype("int8")
        left = out.assign(_call=out["is_call"].astype("int8")).sort_values("ts")
        gr = g.rename(columns={"is_call": "_call"})
        by = ["exp_i", "strike_k", "_call"]
        tol = pd.Timedelta(minutes=tolerance_min)

        merged = pd.merge_asof(
            left.drop(columns=["delta", "iv", "underlying"]), gr, on="ts", by=by,
            direction="backward", tolerance=tol,
        )
        missing = merged["delta"].isna()
        if missing.any():
            fill = pd.merge_asof(
                left.loc[left.index[missing.to_numpy()]].drop(columns=["delta", "iv", "underlying"]),
                gr, on="ts", by=by, direction="nearest", tolerance=tol,
            )
            for c in ("delta", "iv", "underlying"):
                merged.loc[missing.to_numpy(), c] = fill[c].to_numpy()
        out = merged.drop(columns="_call").reset_index(drop=True)

    for c in ("delta", "iv", "underlying"):
        out[c] = out[c].astype("float64")
    has_delta = out["valid"] & out["delta"].notna() & out["underlying"].notna()
    out["signed_delta_contracts"] = np.where(
        has_delta, out["aggressor"] * out["delta"] * out["size"] * CONTRACT_MULTIPLIER, np.nan
    )
    out["signed_dollar_delta"] = out["signed_delta_contracts"] * out["underlying"]
    if "oi" not in out.columns:
        out["oi"] = np.nan
    return out.reindex(columns=TRADE_COLUMNS)


def latest_iv_snapshot(greeks: pd.DataFrame, max_stale_min: int = 30) -> pd.DataFrame:
    """
    Last valid (delta + IV) observation per contract, dropping contracts whose
    last observation is more than `max_stale_min` before the newest
    observation in the frame. Feeds the ATM/term/skew analytics so we never
    persist the full 1-minute Greek history.
    """
    if greeks is None or len(greeks) == 0:
        return greeks
    g = greeks[(greeks["iv"].astype("float64") > 0) & np.isfinite(greeks["delta"].astype("float64"))]
    if g.empty:
        return g
    g = g.sort_values("ts").groupby(["exp_i", "strike_k", "is_call"], sort=False).tail(1)
    cutoff = g["ts"].max() - pd.Timedelta(minutes=max_stale_min)
    return g[g["ts"] >= cutoff].reset_index(drop=True)


def process_expiration(
    trade_raw: Optional[pd.DataFrame],
    greeks_raw: Optional[pd.DataFrame],
    expiration: Any,
    market_date: date,
    tolerance_min: int = 15,
    timer: Optional[StageTimer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One expiration's raw ThetaData frames -> (classified+Greek-joined trades,
    latest IV snapshot). The 1-minute Greek frame is discarded after this.
    """
    tm = timer or _NullTimer()
    with tm.stage("normalize"):
        trades = normalize_trade_quote(trade_raw, expiration, market_date) if trade_raw is not None else _empty_trades()
        greeks = normalize_greeks(greeks_raw, expiration, market_date)
    with tm.stage("classify"):
        trades = classify_trades(trades) if len(trades) else classify_trades(_empty_trades())
    with tm.stage("greek_asof_join"):
        trades = attach_greeks(trades, greeks, tolerance_min)
    with tm.stage("iv_snapshot"):
        snap = latest_iv_snapshot(greeks)
    return trades, snap


def attach_open_interest(trades: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    out = trades.drop(columns=["oi"], errors="ignore")
    if len(out) == 0 or oi is None or len(oi) == 0:
        out["oi"] = np.nan
        return out.reindex(columns=TRADE_COLUMNS)
    right = oi.rename(columns={"open_interest": "oi"})
    out = out.merge(right, on=["exp_i", "strike_k", "is_call"], how="left")
    return out.reindex(columns=TRADE_COLUMNS)


# --------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------

def sentiment_value(net_directional: float, gross: float) -> Optional[float]:
    """net_directional_premium / gross_premium, in [-1, +1]; None when gross is 0."""
    if gross is None or gross <= 0:
        return None
    return float(max(-1.0, min(1.0, net_directional / gross)))


def sentiment_label(
    value: Optional[float],
    eligible_premium: float = math.inf,
    eligible_trades: int = 10 ** 9,
    premium_coverage: float = 1.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """
    (label, low_confidence_reasons). A thin or badly-classified sample gets
    LOW CONFIDENCE instead of a strong label -- the numeric value is still
    returned by the caller so the UI can show it dimmed.
    """
    cfg = cfg or {}
    min_prem = float(cfg.get("minEligiblePremium", 250_000.0))
    min_trades = int(cfg.get("minEligibleTrades", 25))
    min_cov = float(cfg.get("minPremiumCoverage", 0.5))

    reasons: List[str] = []
    if value is None:
        reasons.append("no eligible premium")
    else:
        if eligible_premium < min_prem:
            reasons.append("eligible premium ${:,.0f} < ${:,.0f}".format(eligible_premium, min_prem))
        if eligible_trades < min_trades:
            reasons.append("eligible trades {} < {}".format(eligible_trades, min_trades))
        if premium_coverage < min_cov:
            reasons.append("classified premium {:.0%} < {:.0%}".format(premium_coverage, min_cov))
    if reasons:
        return LABEL_LOW_CONFIDENCE, reasons

    if value > 0.20:
        return LABEL_BULLISH, []
    if value >= 0.08:
        return LABEL_LEAN_BULLISH, []
    if value > -0.08:
        return LABEL_NEUTRAL, []
    if value >= -0.20:
        return LABEL_LEAN_BEARISH, []
    return LABEL_BEARISH, []


# --------------------------------------------------------------------------
# flow blocks
# --------------------------------------------------------------------------

def flow_totals(trades: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n_total = int(len(trades))
    elig = trades[trades["valid"]] if n_total else trades
    n_elig = int(len(elig))
    total_premium = float(trades["premium"].sum()) if n_total else 0.0
    gross = float(elig["premium"].sum()) if n_elig else 0.0
    dp = elig["dir_premium"] if n_elig else pd.Series(dtype="float64")
    bullish = float(dp.clip(lower=0).sum())
    bearish = float((-dp.clip(upper=0)).sum())
    net = bullish - bearish

    value = sentiment_value(net, gross)
    prem_cov = (gross / total_premium) if total_premium > 0 else 0.0
    trade_cov = (n_elig / n_total) if n_total else 0.0
    label, reasons = sentiment_label(value, gross, n_elig, prem_cov, cfg)

    top10 = float(elig["premium"].nlargest(10).sum()) if n_elig else 0.0
    return {
        "sentiment": {"value": value, "label": label, "lowConfidence": label == LABEL_LOW_CONFIDENCE,
                      "reasons": reasons},
        "premium": {
            "gross": gross, "bullish": bullish, "bearish": bearish, "netDirectional": net,
            "totalAllTrades": total_premium,
            "top10Concentration": (top10 / gross) if gross > 0 else None,
        },
        "quality": {
            "trades": n_total, "eligibleTrades": n_elig,
            "classifiedPctTrades": trade_cov, "classifiedPctPremium": prem_cov,
        },
    }


def delta_block(trades: pd.DataFrame) -> Dict[str, Any]:
    d = trades[trades["signed_dollar_delta"].notna()] if len(trades) else trades
    elig_prem = float(trades.loc[trades["valid"], "premium"].sum()) if len(trades) else 0.0
    if len(d) == 0:
        return {"netContracts": None, "netDollar": None, "grossDollar": None, "ratio": None,
                "matchedPctPremium": 0.0, "matchedTrades": 0}
    net_c = float(d["signed_delta_contracts"].sum())
    net_d = float(d["signed_dollar_delta"].sum())
    gross_d = float(d["signed_dollar_delta"].abs().sum())
    return {
        "netContracts": net_c,
        "netDollar": net_d,
        "grossDollar": gross_d,
        "ratio": (net_d / gross_d) if gross_d > 0 else None,
        "matchedPctPremium": (float(d["premium"].sum()) / elig_prem) if elig_prem > 0 else 0.0,
        "matchedTrades": int(len(d)),
    }


def _bucket_mask(dte: pd.Series, lo: int, hi: Optional[int]) -> pd.Series:
    return (dte >= lo) if hi is None else ((dte >= lo) & (dte <= hi))


def dte_block(trades: pd.DataFrame) -> Dict[str, Any]:
    elig = trades[trades["valid"]] if len(trades) else trades
    gross_all = float(elig["premium"].sum()) if len(elig) else 0.0
    rows = []
    zero_dte_gross = 0.0
    for name, lo, hi in DTE_BUCKETS:
        b = elig[_bucket_mask(elig["dte"], lo, hi)] if len(elig) else elig
        gross = float(b["premium"].sum()) if len(b) else 0.0
        net_prem = float(b["dir_premium"].sum()) if len(b) else 0.0
        dd = b[b["signed_dollar_delta"].notna()] if len(b) else b
        net_d = float(dd["signed_dollar_delta"].sum()) if len(dd) else None
        gross_d = float(dd["signed_dollar_delta"].abs().sum()) if len(dd) else None
        if name == "0DTE":
            zero_dte_gross = gross
        rows.append({
            "bucket": name, "trades": int(len(b)), "grossPremium": gross,
            "sharePct": (gross / gross_all) if gross_all > 0 else None,
            "signedPremium": net_prem,
            "sentiment": sentiment_value(net_prem, gross),
            "deltaImbalance": net_d,
            "deltaRatio": (net_d / gross_d) if (net_d is not None and gross_d) else None,
        })
    return {"buckets": rows, "zeroDteShare": (zero_dte_gross / gross_all) if gross_all > 0 else None}


def aggression_block(trades: pd.DataFrame) -> Dict[str, float]:
    """Premium bought/sold at the bid/ask, weighted by |aggressor| so inside-spread prints count partially."""
    if len(trades) == 0:
        return {"callBought": 0.0, "callSold": 0.0, "putBought": 0.0, "putSold": 0.0}
    e = trades[trades["valid"]]
    buy = e["premium"] * e["aggressor"].clip(lower=0)
    sell = e["premium"] * (-e["aggressor"].clip(upper=0))
    call = e["is_call"]
    return {
        "callBought": float(buy[call].sum()), "callSold": float(sell[call].sum()),
        "putBought": float(buy[~call].sum()), "putSold": float(sell[~call].sum()),
    }


def oi_block(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Volume-vs-open-interest context over the contracts that traded. High
    turnover is an activity signal only -- it is NOT folded into sentiment
    (call volume alone is not bullish).
    """
    empty = {"matchedPctPremium": 0.0, "contractsToOi": None, "premiumPerOiContract": None,
             "tradedContracts": 0.0, "oiContracts": 0.0}
    if len(trades) == 0:
        return empty
    e = trades[trades["valid"]]
    elig_prem = float(e["premium"].sum())
    m = e[e["oi"].notna() & (e["oi"] > 0)]
    if len(m) == 0 or elig_prem <= 0:
        return empty
    per_contract = m.groupby(["exp_i", "strike_k", "is_call"]).agg(
        contracts=("size", "sum"), premium=("premium", "sum"), oi=("oi", "first")
    )
    oi_total = float(per_contract["oi"].sum())
    return {
        "matchedPctPremium": float(m["premium"].sum()) / elig_prem,
        "contractsToOi": float(per_contract["contracts"].sum()) / oi_total if oi_total > 0 else None,
        "premiumPerOiContract": float(per_contract["premium"].sum()) / oi_total if oi_total > 0 else None,
        "tradedContracts": float(per_contract["contracts"].sum()),
        "oiContracts": oi_total,
    }


def intraday_series(
    trades: pd.DataFrame, bucket_min: int = 15, rolling_buckets: int = 4,
    session_start: str = "09:30:00", session_end: str = "16:15:00",
) -> List[Dict[str, Any]]:
    """
    Per-bucket net directional premium, cumulative directional premium,
    cumulative dollar-delta imbalance and a rolling sentiment ratio
    (rolling net/gross over the last `rolling_buckets` buckets). Runs from
    the open to the last bucket containing a trade -- no future buckets.
    """
    if len(trades) == 0:
        return []
    e = trades[trades["valid"]]
    if len(e) == 0:
        return []
    ts = e["ts"].dt.tz_convert(NY_TZ)
    day = ts.iloc[0].normalize()
    start = pd.Timestamp("{} {}".format(day.date().isoformat(), session_start), tz=NY_TZ)
    end = pd.Timestamp("{} {}".format(day.date().isoformat(), session_end), tz=NY_TZ)
    keep = (ts >= start) & (ts <= end)
    e, ts = e[keep], ts[keep]
    if len(e) == 0:
        return []

    freq = "{}min".format(bucket_min)
    bucket = ts.dt.floor(freq)
    grp = pd.DataFrame({
        "net": e["dir_premium"].to_numpy(), "gross": e["premium"].to_numpy(),
        "dd": e["signed_dollar_delta"].fillna(0.0).to_numpy(),
        "n": 1,
    }).groupby(bucket.to_numpy()).sum()
    idx = pd.date_range(start.floor(freq), bucket.max(), freq=freq, tz=NY_TZ)
    grp = grp.reindex(idx, fill_value=0.0)

    roll_net = grp["net"].rolling(rolling_buckets, min_periods=1).sum()
    roll_gross = grp["gross"].rolling(rolling_buckets, min_periods=1).sum()
    cum_net = grp["net"].cumsum()
    cum_dd = grp["dd"].cumsum()

    out = []
    for t in grp.index:
        rg = float(roll_gross[t])
        out.append({
            "t": t.isoformat(),
            "netPremium": float(grp.at[t, "net"]),
            "grossPremium": float(grp.at[t, "gross"]),
            "cumNetPremium": float(cum_net[t]),
            "cumDollarDelta": float(cum_dd[t]),
            "rollingSentiment": (float(roll_net[t]) / rg) if rg > 0 else None,
            "trades": int(grp.at[t, "n"]),
        })
    return out


def aggressor_label(score: float) -> str:
    if score >= 0.999:
        return "AT ASK"
    if score <= -0.999:
        return "AT BID"
    if abs(score) < 0.1:
        return "MID"
    return "ABOVE MID" if score > 0 else "BELOW MID"


def large_trades(trades: pd.DataFrame, ticker: str, n: int = 25) -> List[Dict[str, Any]]:
    if len(trades) == 0:
        return []
    e = trades[trades["valid"]].nlargest(n, "premium")
    rows = []
    for r in e.itertuples(index=False):
        directional = r.aggressor * r.direction
        rows.append({
            "time": r.ts.tz_convert(NY_TZ).isoformat(),
            "ticker": ticker,
            "expiration": r.expiration,
            "strike": _f(r.strike),
            "right": "C" if r.is_call else "P",
            "size": _f(r.size),
            "price": _f(r.price),
            "premium": _f(r.premium),
            "bid": _f(r.bid),
            "ask": _f(r.ask),
            "aggressor": _f(r.aggressor),
            "aggressorLabel": aggressor_label(r.aggressor),
            "direction": "BULLISH" if directional > 0.1 else "BEARISH" if directional < -0.1 else "NEUTRAL",
            "delta": _f(r.delta),
            "iv": _f(r.iv),
            "dte": int(r.dte),
            "openInterest": _f(r.oi),
        })
    return rows


# --------------------------------------------------------------------------
# implied-vol analytics
# --------------------------------------------------------------------------

def _liquid(df: pd.DataFrame, max_spread_ratio: float = 0.5) -> pd.DataFrame:
    """Drop contracts with a visibly broken/wide market when bid/ask are available."""
    if "bid" not in df.columns or df["bid"].isna().all():
        return df
    bid = df["bid"].astype("float64")
    ask = df["ask"].astype("float64")
    mid = (bid + ask) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (ask > 0) & (bid >= 0) & (ask >= bid) & (((ask - bid) / mid) <= max_spread_ratio)
    # rows lacking a quote (NaN) are kept -- absence of quote data is not evidence of illiquidity
    return df[ok | bid.isna() | ask.isna()]


def _atm_iv_for_expiry(df: pd.DataFrame, spot: float) -> Optional[float]:
    """Average call/put IV per strike, linearly interpolated to spot."""
    g = _liquid(df)
    g = g[g["iv"] > 0]
    if g.empty or not spot or spot <= 0:
        return None
    per_strike = g.groupby("strike")["iv"].mean().sort_index()
    if len(per_strike) == 1:
        return float(per_strike.iloc[0])
    return float(np.interp(spot, per_strike.index.to_numpy(dtype="float64"),
                           per_strike.to_numpy(dtype="float64")))


def _iv_at_abs_delta(df: pd.DataFrame, target: float, tol: float = 0.08) -> Optional[float]:
    """IV at |delta| == target, interpolating between the closest bracketing liquid contracts."""
    g = _liquid(df)
    g = g[(g["iv"] > 0) & np.isfinite(g["delta"].astype("float64"))]
    if g.empty:
        return None
    ad = g["delta"].astype("float64").abs().round(4)
    per = pd.DataFrame({"ad": ad, "iv": g["iv"].astype("float64")}).groupby("ad")["iv"].mean().sort_index()
    xs, ys = per.index.to_numpy(dtype="float64"), per.to_numpy(dtype="float64")
    if xs.min() <= target <= xs.max():
        return float(np.interp(target, xs, ys))
    j = int(np.argmin(np.abs(xs - target)))
    return float(ys[j]) if abs(xs[j] - target) <= tol else None


def _interp_variance(term: List[Tuple[float, float]], target: float) -> Optional[float]:
    """Constant-maturity IV via linear interpolation of total variance (iv^2 * T) across expirations."""
    pts = sorted((d, v) for d, v in term if d >= 1 and v and v > 0)
    if not pts:
        return None
    tol = max(3.0, 0.25 * target)
    ds = np.array([p[0] for p in pts], dtype="float64")
    vs = np.array([p[1] for p in pts], dtype="float64")
    if target < ds.min() or target > ds.max():
        j = int(np.argmin(np.abs(ds - target)))
        return float(vs[j]) if abs(ds[j] - target) <= tol else None
    w = np.interp(target, ds, vs ** 2 * ds)
    return float(math.sqrt(w / target))


def _interp_linear(points: List[Tuple[float, float]], target: float) -> Optional[float]:
    pts = sorted((d, v) for d, v in points if v is not None)
    if not pts:
        return None
    tol = max(3.0, 0.25 * target)
    ds = np.array([p[0] for p in pts], dtype="float64")
    vs = np.array([p[1] for p in pts], dtype="float64")
    if target < ds.min() or target > ds.max():
        j = int(np.argmin(np.abs(ds - target)))
        return float(vs[j]) if abs(ds[j] - target) <= tol else None
    return float(np.interp(target, ds, vs))


def spot_from_snapshot(iv_snap: pd.DataFrame) -> Optional[float]:
    if iv_snap is None or len(iv_snap) == 0:
        return None
    u = iv_snap[iv_snap["underlying"].notna() & (iv_snap["underlying"] > 0)]
    if u.empty:
        return None
    return float(u.sort_values("ts")["underlying"].iloc[-1])


IV_CHANGE_LAGS = {"atmChange1d": 1, "atmChange5d": 5, "atmChange20d": 20}
IV_PERCENTILE_WINDOWS = (20, 60, 126, 252)
# Minimum observations (INCLUDING the current one) for a windowed percentile to be reported at all.
IV_WINDOW_MIN_OBS = {20: 15, 60: 45, 126: 95, 252: 190}


def iv_history_stats(
    iv30: Optional[float], history: Optional[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rolling IV changes and percentiles of `iv30` against PRIOR daily observations
    (percentile windows additionally include `iv30` itself as their last observation).
    Single implementation shared by live snapshots, historical backfill and the
    backfill re-stat pass.

    history = {
        "priorAtmIv":   [(session_iso, atm_iv), ...]   observations strictly before this date
        "sessionsBack": [session_iso, ...]             prior trading sessions, newest first (optional)
    }
    With sessionsBack (backfill), "N days ago" means the N-th prior TRADING SESSION and a
    session with no observation is null -- gaps are never bridged. Without it (live legacy),
    the observation list itself is the session list.

    Nothing here can see a date on or after the snapshot's own: the caller passes only
    prior observations (see options_flow_store.build_iv_history).
    """
    cfg = cfg or {}
    prior = list((history or {}).get("priorAtmIv", []))
    by_session = {d: v for d, v in prior if v is not None}
    sessions = list((history or {}).get("sessionsBack") or [d for d, _ in prior])

    out: Dict[str, Any] = {}
    for key, lag in IV_CHANGE_LAGS.items():
        ref = by_session.get(sessions[lag - 1]) if len(sessions) >= lag else None
        out[key] = (iv30 - ref) if (iv30 is not None and ref is not None) else None

    hist_vals = np.array(list(by_session.values()), dtype="float64")
    min_obs = int(cfg.get("ivPercentileMinObs", 20))
    out["percentile"] = (
        float((hist_vals <= iv30).mean() * 100.0) if (iv30 is not None and len(hist_vals) >= min_obs) else None
    )
    out["percentileObs"] = int(len(hist_vals))

    # Windowed percentile rank of the current observation within the trailing window that ENDS AT
    # and INCLUDES this date: the previous n-1 sessions plus today (n sessions in total).
    windows: Dict[str, Optional[float]] = {}
    window_obs: Dict[str, int] = {}
    for n in IV_PERCENTILE_WINDOWS:
        prior_vals = [by_session[d] for d in sessions[: n - 1] if d in by_session]
        vals = np.array(prior_vals + ([iv30] if iv30 is not None else []), dtype="float64")
        window_obs[f"{n}d"] = int(len(vals))
        ok = iv30 is not None and len(vals) >= IV_WINDOW_MIN_OBS[n]
        windows[f"{n}d"] = float((vals <= iv30).mean() * 100.0) if ok else None
    out["percentiles"] = windows
    out["percentilesObs"] = window_obs
    return out


def iv_block(
    iv_snap: pd.DataFrame, spot: Optional[float],
    history: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    ATM IV is defined as the 30-day constant-maturity ATM IV (total-variance
    interpolation across the bracketing expirations). 25-delta skew is
    25d-put IV minus 25d-call IV, interpolated to 30 DTE.

    history = {"priorAtmIv": [(market_date_iso, atm_iv), ...]}  newest first,
    current market date excluded.
    """
    cfg = cfg or {}
    empty = {
        "atm": None,
        "iv7d": None, "iv30d": None, "iv60d": None,
        "spread7v30": None, "spread30v60": None,
        "skew25d30d": None, "termStructure": [], "definition": "ATM IV = 30D constant-maturity",
        **iv_history_stats(None, history, cfg),
    }
    if iv_snap is None or len(iv_snap) == 0 or not spot:
        return empty

    term_rows: List[Dict[str, Any]] = []
    skew_pts: List[Tuple[float, float]] = []
    for (exp_i, dte), g in iv_snap.groupby(["exp_i", "dte"]):
        atm = _atm_iv_for_expiry(g, spot)
        put_iv = _iv_at_abs_delta(g[~g["is_call"]], 0.25)
        call_iv = _iv_at_abs_delta(g[g["is_call"]], 0.25)
        skew = (put_iv - call_iv) if (put_iv is not None and call_iv is not None) else None
        if atm is not None:
            term_rows.append({"expiration": g["expiration"].iloc[0], "dte": int(dte), "atmIv": atm})
        if skew is not None and dte >= 1:
            skew_pts.append((float(dte), skew))
    term_rows.sort(key=lambda r: r["dte"])
    term = [(r["dte"], r["atmIv"]) for r in term_rows]

    iv7 = _interp_variance(term, 7)
    iv30 = _interp_variance(term, 30)
    iv60 = _interp_variance(term, 60)

    return {
        "atm": iv30,
        "iv7d": iv7, "iv30d": iv30, "iv60d": iv60,
        "spread7v30": (iv7 - iv30) if (iv7 is not None and iv30 is not None) else None,
        "spread30v60": (iv30 - iv60) if (iv30 is not None and iv60 is not None) else None,
        "skew25d30d": _interp_linear(skew_pts, 30),
        "termStructure": term_rows, "definition": empty["definition"],
        **iv_history_stats(iv30, history, cfg),
    }


# --------------------------------------------------------------------------
# per-ticker payload
# --------------------------------------------------------------------------

def build_iv_observation(
    iv_snap: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None, history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One daily IV observation from a latest-IV snapshot frame, WITHOUT any trade data: spot, ATM/7D/30D/60D IV,
    term spreads, 25-delta skew, the term structure, and (with `history`) the rolling changes/percentiles.
    Uses exactly the same iv_block() as full-flow snapshots, so an iv-warmup day and a full-flow day agree
    given the same Greek observations -- this is the entire lightweight-mode payload, deliberately built with
    no trade_quote / open_interest fetch at all.
    """
    spot = spot_from_snapshot(iv_snap)
    _check_strike_units(iv_snap, spot)
    iv = iv_block(iv_snap, spot, history, cfg or {})
    last_ts = iv_snap["ts"].max() if iv_snap is not None and len(iv_snap) else None
    out = dict(iv)
    out.pop("definition", None)
    out.update({
        "spot": spot,
        "asOf": last_ts.tz_convert(NY_TZ).isoformat() if last_ts is not None and not pd.isna(last_ts) else None,
        "expirations": int(iv_snap["exp_i"].nunique()) if iv_snap is not None and len(iv_snap) else 0,
        "contracts": int(len(iv_snap)) if iv_snap is not None else 0,
    })
    return _clean(out)


def _check_strike_units(iv_snap: pd.DataFrame, spot: Optional[float]) -> None:
    """Strikes are assumed to be in dollars like the underlying. If ThetaData ever
    changes units, ATM interpolation would silently return nonsense -- fail loudly."""
    if not spot or iv_snap is None or len(iv_snap) == 0:
        return
    ratio = float(iv_snap["strike"].median()) / spot
    if not 0.3 < ratio < 3.0:
        raise ValueError(
            "median strike / spot = {:.3f}; strikes do not look like dollars (spot {})".format(ratio, spot)
        )


def build_symbol_payload(
    ticker: str, group: str, market_date: date,
    trades: pd.DataFrame, iv_snap: pd.DataFrame,
    oi: Optional[pd.DataFrame] = None,
    history: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    fetched_at: Optional[datetime] = None,
    fetch_meta: Optional[Dict[str, Any]] = None,
    source: str = "live",
) -> Dict[str, Any]:
    """Everything the page needs for one ETF, as one JSON-safe dict.
    `source` is provenance only ("live" | "historical_backfill"); it never changes the math."""
    cfg = cfg or {}
    if oi is not None:
        trades = attach_open_interest(trades, oi)

    totals = flow_totals(trades, cfg)
    delta = delta_block(trades)
    dte = dte_block(trades)
    spot = spot_from_snapshot(iv_snap)
    _check_strike_units(iv_snap, spot)
    iv = iv_block(iv_snap, spot, history, cfg)
    oi_stats = oi_block(trades)

    last_trade = trades["ts"].max() if len(trades) else None
    quality = dict(totals["quality"])
    quality.update({
        "contractsTraded": int(trades[["exp_i", "strike_k", "is_call"]].drop_duplicates().shape[0]) if len(trades) else 0,
        "deltaMatchedPctPremium": delta["matchedPctPremium"],
        "oiMatchedPctPremium": oi_stats["matchedPctPremium"],
        "lastTradeTime": last_trade.tz_convert(NY_TZ).isoformat() if last_trade is not None and not pd.isna(last_trade) else None,
        "thetaFetchedAt": fetched_at.isoformat() if fetched_at else None,
    })
    quality.update(fetch_meta or {})

    payload = {
        "source": source,
        "methodologyVersion": METHODOLOGY_VERSION,
        "ticker": ticker,
        "group": group,
        "marketDate": market_date.isoformat(),
        "asOf": quality["lastTradeTime"],
        "spot": spot,
        "sentiment": totals["sentiment"],
        "premium": totals["premium"],
        "delta": delta,
        "iv": iv,
        "dte": dte,
        "aggression": aggression_block(trades),
        "openInterest": oi_stats,
        "quality": quality,
        "intraday": intraday_series(
            trades, int(cfg.get("bucketMinutes", 15)), int(cfg.get("rollingBuckets", 4)),
            str(cfg.get("sessionStart", "09:30:00")), str(cfg.get("sessionEnd", "16:15:00")),
        ),
        "largeTrades": large_trades(trades, ticker, int(cfg.get("largeTradeCount", 25))),
    }
    return _clean(payload)
