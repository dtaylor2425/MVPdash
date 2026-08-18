from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq


HORIZONS = (5, 10, 20, 30, 40, 50)
TARGET_DTES = (7, 14, 30, 60)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def fetch_spx_history(period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Yahoo uses ^GSPC for the S&P 500 cash index.
    """
    df = yf.download(
        "^GSPC",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned no ^GSPC price history.")
    df = _clean_columns(df)
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Missing OHLC columns. Found: {list(df.columns)}")
    return df.dropna(subset=["High", "Low", "Close"])


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _rma(series: pd.Series, length: int) -> pd.Series:
    # TradingView-style Wilder RMA is equivalent to EMA alpha=1/length.
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _smooth(series: pd.Series, length: int, method: str) -> pd.Series:
    method = method.upper()
    if method == "RMA":
        return _rma(series, length)
    if method == "SMA":
        return series.rolling(length).mean()
    if method == "EMA":
        return series.ewm(span=length, adjust=False, min_periods=length).mean()
    if method == "WMA":
        weights = np.arange(1, length + 1, dtype=float)
        return series.rolling(length).apply(
            lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
        )
    raise ValueError(f"Unsupported smoothing method: {method}")


def natr_series(
    df: pd.DataFrame,
    atr_len: int,
    norm_len: int,
    value_smoothing: int = 5,
    atr_smoothing: str = "RMA",
) -> pd.Series:
    tr = _true_range(df)
    atr = _smooth(tr, atr_len, atr_smoothing)
    hi = atr.rolling(norm_len).max()
    lo = atr.rolling(norm_len).min()
    denom = hi - lo
    normalized = pd.Series(
        np.where(denom != 0, 2.0 * (atr - lo) / denom - 1.0, 0.0),
        index=df.index,
        dtype=float,
    )
    return normalized.ewm(
        span=value_smoothing,
        adjust=False,
        min_periods=value_smoothing,
    ).mean()


def _regime(value: float, rising: bool) -> str:
    if not np.isfinite(value):
        return "UNAVAILABLE"
    if value > 0 and rising:
        return "HIGH & EXPANDING"
    if value > 0 and not rising:
        return "HIGH & CONTRACTING"
    if value < 0 and rising:
        return "LOW & EXPANDING"
    if value < 0 and not rising:
        return "LOW & CONTRACTING"
    return "FLAT"


def build_natr_ladder(
    df: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
    atr_len_base: int = 14,
    norm_len_base: int = 20,
    value_smoothing: int = 5,
    atr_smoothing: str = "RMA",
    map_mode: str = "Both",
) -> List[Dict[str, Any]]:
    """
    Reproduces the Pine logic:
      Both        => ATR len=N, normalization len=N
      ATR length  => ATR len=N, normalization len=norm_len_base
      Norm length => ATR len=atr_len_base, normalization len=N
    """
    result: List[Dict[str, Any]] = []
    for h in horizons:
        if map_mode == "Both":
            atr_len, norm_len = h, h
        elif map_mode == "ATR length":
            atr_len, norm_len = h, norm_len_base
        elif map_mode == "Norm length":
            atr_len, norm_len = atr_len_base, h
        else:
            raise ValueError("map_mode must be Both, ATR length, or Norm length")

        s = natr_series(
            df,
            atr_len=atr_len,
            norm_len=norm_len,
            value_smoothing=value_smoothing,
            atr_smoothing=atr_smoothing,
        ).dropna()

        if len(s) < 2:
            value, rising = float("nan"), False
        else:
            value = float(s.iloc[-1])
            rising = bool(s.iloc[-1] > s.iloc[-2])

        result.append(
            {
                "horizon": int(h),
                "natr": None if not np.isfinite(value) else round(value, 4),
                "rising": rising,
                "regime": _regime(value, rising),
            }
        )
    return result


@dataclass
class OptionQuote:
    strike: float
    call_mid: Optional[float]
    put_mid: Optional[float]
    call_bid: Optional[float]
    call_ask: Optional[float]
    put_bid: Optional[float]
    put_ask: Optional[float]
    call_oi: int
    put_oi: int
    call_vol: int
    put_vol: int


def _mid(bid: Any, ask: Any, last: Any = None) -> Optional[float]:
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        b, a = 0.0, 0.0

    if b > 0 and a > 0 and a >= b:
        return (b + a) / 2.0

    # Last is a fallback only. It is less reliable than a live midpoint.
    try:
        l = float(last)
    except (TypeError, ValueError):
        l = 0.0
    return l if l > 0 else None


def _safe_int(x: Any) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except (TypeError, ValueError):
        return 0


def _yield_rate() -> float:
    """
    Uses ^IRX (13-week T-bill yield) as a lightweight short-rate proxy.
    Falls back to 4% if Yahoo fails.
    """
    try:
        h = yf.Ticker("^IRX").history(period="5d", interval="1d", auto_adjust=False)
        if h is not None and not h.empty:
            val = float(h["Close"].dropna().iloc[-1]) / 100.0
            if 0.0 <= val <= 0.20:
                return val
    except Exception:
        pass
    return 0.04


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76_price(
    option_type: str,
    forward: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
) -> float:
    if t <= 0 or sigma <= 0 or forward <= 0 or strike <= 0:
        return float("nan")
    root_t = math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    disc = math.exp(-rate * t)
    if option_type == "call":
        return disc * (forward * _normal_cdf(d1) - strike * _normal_cdf(d2))
    return disc * (strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1))


def implied_vol_black76(
    option_type: str,
    price: float,
    forward: float,
    strike: float,
    t: float,
    rate: float,
) -> Optional[float]:
    if price <= 0 or forward <= 0 or strike <= 0 or t <= 0:
        return None

    disc = math.exp(-rate * t)
    intrinsic = disc * (
        max(forward - strike, 0.0)
        if option_type == "call"
        else max(strike - forward, 0.0)
    )
    if price <= intrinsic + 1e-8:
        return None

    try:
        return float(
            brentq(
                lambda vol: black76_price(
                    option_type, forward, strike, t, rate, vol
                )
                - price,
                1e-4,
                5.0,
                maxiter=100,
            )
        )
    except (ValueError, RuntimeError):
        return None


def _option_symbol_candidates() -> List[str]:
    # Yahoo coverage can vary. ^SPX is attempted first; SPY is fallback only.
    return ["^SPX", "^GSPC", "SPY"]


def _get_option_ticker() -> Tuple[yf.Ticker, str, Tuple[str, ...]]:
    errors: List[str] = []
    for symbol in _option_symbol_candidates():
        try:
            t = yf.Ticker(symbol)
            expiries = tuple(t.options)
            if expiries:
                return t, symbol, expiries
            errors.append(f"{symbol}: no expirations")
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    raise RuntimeError("No Yahoo option chain available. " + " | ".join(errors))


def _expiration_dte(expiry: str, now: datetime) -> int:
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp_date - now.date()).days


def _select_expiries(
    expiries: Iterable[str],
    target_dtes: Iterable[int],
    now: datetime,
) -> List[str]:
    pairs = [(e, _expiration_dte(e, now)) for e in expiries]
    pairs = [(e, d) for e, d in pairs if d >= 2]
    if not pairs:
        return []

    needed = set()
    dtes = sorted(d for _, d in pairs)

    for target in target_dtes:
        below = [d for d in dtes if d <= target]
        above = [d for d in dtes if d >= target]
        if below:
            d = max(below)
            needed.add(next(e for e, x in pairs if x == d))
        if above:
            d = min(above)
            needed.add(next(e for e, x in pairs if x == d))

    return sorted(needed, key=lambda e: _expiration_dte(e, now))


def _quality_ok(bid: float, ask: float, mid: float, max_rel_spread: float) -> bool:
    if bid <= 0 or ask <= 0 or ask < bid or mid <= 0:
        return False
    return ((ask - bid) / mid) <= max_rel_spread


def _extract_quotes(
    chain: Any,
    spot: float,
    strike_band_pct: float = 0.08,
) -> List[OptionQuote]:
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    if calls.empty or puts.empty:
        return []

    lo, hi = spot * (1.0 - strike_band_pct), spot * (1.0 + strike_band_pct)
    calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
    puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]

    c_by_k = {float(r["strike"]): r for _, r in calls.iterrows()}
    p_by_k = {float(r["strike"]): r for _, r in puts.iterrows()}

    quotes: List[OptionQuote] = []
    for k in sorted(set(c_by_k).intersection(p_by_k)):
        c, p = c_by_k[k], p_by_k[k]
        quotes.append(
            OptionQuote(
                strike=k,
                call_mid=_mid(c.get("bid"), c.get("ask"), c.get("lastPrice")),
                put_mid=_mid(p.get("bid"), p.get("ask"), p.get("lastPrice")),
                call_bid=float(c.get("bid") or 0.0),
                call_ask=float(c.get("ask") or 0.0),
                put_bid=float(p.get("bid") or 0.0),
                put_ask=float(p.get("ask") or 0.0),
                call_oi=_safe_int(c.get("openInterest")),
                put_oi=_safe_int(p.get("openInterest")),
                call_vol=_safe_int(c.get("volume")),
                put_vol=_safe_int(p.get("volume")),
            )
        )
    return quotes


def _estimate_forward(
    quotes: List[OptionQuote],
    spot: float,
    t: float,
    rate: float,
) -> float:
    """
    Estimate the index forward from put-call parity near spot:
        C - P = exp(-rT) * (F - K)
        F = K + exp(rT) * (C - P)

    Median across nearby strikes reduces dependence on one quote.
    """
    vals = []
    for q in sorted(quotes, key=lambda x: abs(x.strike - spot))[:7]:
        if q.call_mid is None or q.put_mid is None:
            continue
        vals.append(q.strike + math.exp(rate * t) * (q.call_mid - q.put_mid))
    if not vals:
        return spot * math.exp(rate * t)
    fwd = float(np.median(vals))
    if not (0.85 * spot <= fwd <= 1.15 * spot):
        return spot * math.exp(rate * t)
    return fwd


def _expiry_atm_iv(
    chain: Any,
    spot: float,
    dte: int,
    rate: float,
    strikes_each_side: int = 2,
    max_rel_spread: float = 0.35,
) -> Dict[str, Any]:
    t = max(dte, 1) / 365.0
    quotes = _extract_quotes(chain, spot)
    if not quotes:
        raise RuntimeError("No paired call/put strikes near ATM.")

    forward = _estimate_forward(quotes, spot, t, rate)
    chosen = sorted(quotes, key=lambda q: abs(q.strike - forward))[
        : 2 * strikes_each_side + 1
    ]

    ivs: List[Tuple[float, float]] = []
    used = 0

    for q in chosen:
        per_strike = []
        if (
            q.call_mid
            and _quality_ok(q.call_bid, q.call_ask, q.call_mid, max_rel_spread)
        ):
            iv = implied_vol_black76(
                "call", q.call_mid, forward, q.strike, t, rate
            )
            if iv and 0.01 < iv < 3.0:
                per_strike.append(iv)

        if (
            q.put_mid
            and _quality_ok(q.put_bid, q.put_ask, q.put_mid, max_rel_spread)
        ):
            iv = implied_vol_black76(
                "put", q.put_mid, forward, q.strike, t, rate
            )
            if iv and 0.01 < iv < 3.0:
                per_strike.append(iv)

        if per_strike:
            # Weight nearest strikes more heavily.
            weight = 1.0 / (1.0 + abs(q.strike - forward) / max(spot * 0.005, 1e-6))
            ivs.append((float(np.median(per_strike)), weight))
            used += 1

    if not ivs:
        # Fallback to Yahoo's supplied IV field around ATM.
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        frames = []
        for frame in (calls, puts):
            if "impliedVolatility" in frame and not frame.empty:
                temp = frame.assign(
                    distance=(frame["strike"].astype(float) - forward).abs()
                ).nsmallest(5, "distance")
                vals = pd.to_numeric(temp["impliedVolatility"], errors="coerce")
                vals = vals[(vals > 0.01) & (vals < 3.0)]
                frames.extend(vals.tolist())
        if not frames:
            raise RuntimeError("Could not calculate ATM implied volatility.")
        atm_iv = float(np.median(frames))
        method = "yahoo_iv_fallback"
    else:
        atm_iv = float(np.average([x[0] for x in ivs], weights=[x[1] for x in ivs]))
        method = "midpoint_black76"

    return {
        "dte": int(dte),
        "atm_iv": atm_iv,
        "forward": round(forward, 4),
        "strikes_used": used,
        "method": method,
    }


def _constant_maturity_iv(
    expiry_points: List[Dict[str, Any]],
    target_dte: int,
) -> Optional[float]:
    """
    Interpolate total variance, not raw IV:
        w(T) = sigma(T)^2 * T
    """
    pts = sorted(expiry_points, key=lambda x: x["dte"])
    if not pts:
        return None

    exact = next((p for p in pts if p["dte"] == target_dte), None)
    if exact:
        return float(exact["atm_iv"])

    lower = [p for p in pts if p["dte"] < target_dte]
    upper = [p for p in pts if p["dte"] > target_dte]

    if not lower or not upper:
        nearest = min(pts, key=lambda p: abs(p["dte"] - target_dte))
        # Do not extrapolate too far.
        if abs(nearest["dte"] - target_dte) > 5:
            return None
        return float(nearest["atm_iv"])

    p1, p2 = max(lower, key=lambda p: p["dte"]), min(upper, key=lambda p: p["dte"])
    t1, t2, tt = p1["dte"] / 365.0, p2["dte"] / 365.0, target_dte / 365.0
    w1 = p1["atm_iv"] ** 2 * t1
    w2 = p2["atm_iv"] ** 2 * t2
    weight = (tt - t1) / (t2 - t1)
    wt = w1 + weight * (w2 - w1)
    return math.sqrt(max(wt / tt, 0.0))


def fetch_iv_term_structure(
    spot: float,
    target_dtes: Iterable[int] = TARGET_DTES,
) -> Dict[str, Any]:
    now = _utc_now()
    ticker, source_symbol, expiries = _get_option_ticker()
    selected = _select_expiries(expiries, target_dtes, now)
    if not selected:
        raise RuntimeError("No suitable Yahoo option expirations found.")

    rate = _yield_rate()
    expiry_points: List[Dict[str, Any]] = []
    errors: List[str] = []

    for expiry in selected:
        dte = _expiration_dte(expiry, now)
        try:
            chain = ticker.option_chain(expiry)
            pt = _expiry_atm_iv(chain, spot, dte, rate)
            pt["expiry"] = expiry
            expiry_points.append(pt)
        except Exception as exc:
            errors.append(f"{expiry}: {exc}")

    if not expiry_points:
        raise RuntimeError("All Yahoo option-chain calculations failed: " + " | ".join(errors))

    curve: Dict[str, Optional[float]] = {}
    for target in target_dtes:
        iv = _constant_maturity_iv(expiry_points, int(target))
        curve[f"{int(target)}d"] = None if iv is None else round(iv * 100.0, 3)

    def spread(a: str, b: str) -> Optional[float]:
        va, vb = curve.get(a), curve.get(b)
        if va is None or vb is None:
            return None
        return round(va - vb, 3)

    spreads = {
        "7d_14d": spread("7d", "14d"),
        "14d_30d": spread("14d", "30d"),
        "30d_60d": spread("30d", "60d"),
    }

    return {
        "source_symbol": source_symbol,
        "is_spy_fallback": source_symbol == "SPY",
        "rate_proxy": round(rate * 100.0, 3),
        "curve": curve,
        "spreads": spreads,
        "expiry_points": expiry_points,
        "warnings": errors,
    }


def _classify_divergence(
    ladder: List[Dict[str, Any]],
    spreads: Dict[str, Optional[float]],
) -> Dict[str, str]:
    short = [x for x in ladder if x["horizon"] in (5, 10, 20)]
    expanding = sum(1 for x in short if x["rising"])
    low = sum(1 for x in short if x["natr"] is not None and x["natr"] < 0)

    s714 = spreads.get("7d_14d")
    if s714 is None:
        implied = "UNAVAILABLE"
    elif s714 <= -1.5:
        implied = "VERY COMPRESSED"
    elif s714 < -0.5:
        implied = "COMPRESSED"
    elif s714 <= 0.5:
        implied = "NEUTRAL"
    else:
        implied = "FRONT-END BID"

    if expanding >= 2 and low >= 2:
        realized = "LOW & BEGINNING TO EXPAND"
    elif expanding >= 2:
        realized = "EXPANDING"
    elif expanding <= 1 and low >= 2:
        realized = "LOW & CONTRACTING"
    else:
        realized = "MIXED"

    if implied in {"VERY COMPRESSED", "COMPRESSED"} and expanding >= 2:
        regime = "EARLY VOLATILITY EXPANSION"
    elif implied == "FRONT-END BID" and expanding >= 2:
        regime = "VOLATILITY EXPANSION"
    elif implied in {"VERY COMPRESSED", "COMPRESSED"} and expanding <= 1:
        regime = "COMPLACENT / QUIET"
    else:
        regime = "MIXED"

    return {"implied": implied, "realized": realized, "regime": regime}


def build_volatility_snapshot() -> Dict[str, Any]:
    history = fetch_spx_history()
    close = float(history["Close"].iloc[-1])

    ladder = build_natr_ladder(history)

    option_error = None
    try:
        iv = fetch_iv_term_structure(close)
    except Exception as exc:
        option_error = str(exc)
        iv = {
            "source_symbol": None,
            "is_spy_fallback": False,
            "rate_proxy": None,
            "curve": {f"{d}d": None for d in TARGET_DTES},
            "spreads": {
                "7d_14d": None,
                "14d_30d": None,
                "30d_60d": None,
            },
            "expiry_points": [],
            "warnings": [option_error],
        }

    divergence = _classify_divergence(ladder, iv["spreads"])

    return {
        "timestamp": _utc_now().isoformat(),
        "underlying": "^GSPC",
        "spx": round(close, 2),
        "natr": ladder,
        "implied_volatility": iv,
        "divergence": divergence,
        "data_quality": {
            "options_ok": option_error is None,
            "options_error": option_error,
            "note": (
                "Yahoo option data is suitable for a regime dashboard, not execution-grade "
                "or tick-level trading. If SPX index options are unavailable, the service "
                "falls back to SPY and flags that explicitly."
            ),
        },
    }
