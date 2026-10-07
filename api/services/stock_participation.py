"""Deterministic research analytics from completed, same-feed stock sessions.

Scores describe observed evidence, not return probabilities or portfolio instructions.
All percent fields use percentage points, not decimal fractions.
"""
from __future__ import annotations

import math
import statistics
from typing import Any


def finite(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def rounded(value):
    return round(value, 4) if value is not None and math.isfinite(value) else None


def clamp(value):
    return max(0.0, min(100.0, value))


def clean_bars(rows, field="date"):
    out = {}
    for row in rows or []:
        stamp = str(row.get(field) or "")
        values = {k: finite(row.get(k)) for k in ("open", "high", "low", "close", "volume")}
        if not stamp or any(values[k] is None or values[k] <= 0 for k in ("open", "high", "low", "close")):
            continue
        if values["high"] < max(values["open"], values["close"], values["low"]) or values["low"] > min(values["open"], values["close"]):
            continue
        if values["volume"] is not None and values["volume"] < 0:
            values["volume"] = None
        out[stamp] = {field: stamp, **values}
    return [out[k] for k in sorted(out)]


def build_participation_payload(symbol, daily_bars, intraday_bars, benchmark_bars,
                                sector_bars=None, *, sector_symbol=None, feed_metadata=None):
    bars = clean_bars(daily_bars)
    if not bars:
        raise ValueError("No valid completed stock sessions")
    now = bars[-1]
    prior = bars[-2] if len(bars) > 1 else None
    closes = [r["close"] for r in bars]
    # Raw histories are not silently interpreted as adjusted total returns.
    discontinuity = any(abs(b["close"] / a["close"] - 1) > .45 for a, b in zip(bars[-64:-1], bars[-63:])) if len(bars) >= 64 else any(abs(b["close"] / a["close"] - 1) > .45 for a,b in zip(bars,bars[1:]))
    daily_return = (now["close"] / prior["close"] - 1) * 100 if prior else None
    sessions = clean_bars(intraday_bars, "timestamp")
    regular = [r for r in sessions if r["timestamp"][:10] == now["date"]]
    previous_regular = [r for r in sessions if prior and r["timestamp"][:10] == prior["date"]]
    previous_close = previous_regular[-1]["close"] if previous_regular else None
    regular_close = regular[-1]["close"] if regular else None
    regular_open = regular[0]["open"] if regular else None
    gap = (regular_open / previous_close - 1) * 100 if regular_open and previous_close else None
    gap_retained = (regular_close - previous_close) / (regular_open - previous_close) * 100 if previous_close and regular_close and abs(gap or 0) >= .1 else None
    regular_return = (regular_close / previous_close - 1) * 100 if regular_close and previous_close else None
    prior_vol = [r["volume"] for r in bars[-21:-1] if r["volume"] is not None and r["volume"] > 0]
    relative_volume = now["volume"] / statistics.median(prior_vol) if len(prior_vol) == 20 and now["volume"] is not None else None
    regular_high = max(r["high"] for r in regular) if regular else None
    regular_low = min(r["low"] for r in regular) if regular else None
    close_location = (regular_close - regular_low) / (regular_high - regular_low) * 100 if regular_high and regular_low and regular_high > regular_low else None
    sma20 = statistics.mean(closes[-20:]) if len(closes) >= 20 and not discontinuity else None
    sma50 = statistics.mean(closes[-50:]) if len(closes) >= 50 and not discontinuity else None

    def excess(other):
        if len(bars) < 21 or discontinuity:
            return None
        lookup = {r["date"]: r["close"] for r in clean_bars(other)}
        first, last = bars[-21], bars[-1]
        a, b = lookup.get(first["date"]), lookup.get(last["date"])
        if not a or not b:
            return None
        return ((last["close"] / first["close"] - 1) - (b / a - 1)) * 100

    spy_excess = excess(benchmark_bars)
    sector_excess = excess(sector_bars)
    intraday = regular
    cumulative_value = cumulative_volume = 0.0
    for row in intraday:
        volume = row["volume"] or 0
        cumulative_value += (row["high"] + row["low"] + row["close"]) / 3 * volume
        cumulative_volume += volume
        row["vwap"] = rounded(cumulative_value / cumulative_volume) if cumulative_volume else None
    session_vwap = intraday[-1]["vwap"] if intraday else None
    above_vwap = (intraday[-1]["close"] / session_vwap - 1) * 100 if session_vwap else None
    logs = [math.log(b/a) for a,b in zip(closes[-21:-1],closes[-20:])] if len(closes) >= 21 else []
    realized = statistics.stdev(logs) * math.sqrt(252) * 100 if len(logs) == 20 and not discontinuity else None
    drawdown = (closes[-1] / max(closes[-63:]) - 1) * 100 if len(closes) >= 63 and not discontinuity else None
    trend = statistics.mean([clamp(50 + 500 * (now["close"] / v - 1)) for v in (sma20,sma50) if v]) if sma20 and sma50 else None
    leadership = clamp(50 + 5 * spy_excess) if spy_excess is not None else None
    volume_evidence = clamp(50 + (1 if daily_return > 0 else -1 if daily_return < 0 else 0) * 25 * min(relative_volume, 2)) if relative_volume is not None and daily_return is not None else None
    components = [
        {"key":"trend","label":"Trend support","value":rounded(trend),"weight":35,"explanation":"Average of price distance from 20- and 50-session means. At each mean = 50; 10% above = 100; 10% below = 0."},
        {"key":"leadership","label":"Market leadership","value":rounded(leadership),"weight":35,"explanation":"20-session return minus SPY return. Equal performance = 50; +10 percentage points = 100; -10 = 0."},
        {"key":"close","label":"Closing strength","value":rounded(close_location),"weight":20,"explanation":"Close within the daily high-low range: low = 0, high = 100. A flat range is unavailable."},
        {"key":"volume","label":"Directional participation","value":rounded(volume_evidence),"weight":10,"explanation":"50 plus or minus 25 times relative volume (capped at 2x), using the daily price direction. Volume does not identify buyers or institutions."},
    ]
    coverage = sum(c["weight"] for c in components if c["value"] is not None)
    total = sum(c["value"] * c["weight"] for c in components if c["value"] is not None) / coverage if coverage >= 70 and not discontinuity else None
    label = "Insufficient evidence" if total is None else "Supportive" if total >= 65 else "Mixed" if total >= 40 else "Weak"
    observations = []
    if relative_volume is not None:
        observations.append(f"Volume was {relative_volume:.2f}x the median of the previous 20 sessions.")
    if spy_excess is not None:
        observations.append(f"20-session performance was {abs(spy_excess):.1f} percentage points {'ahead of' if spy_excess >= 0 else 'behind'} SPY.")
    if close_location is not None:
        observations.append(f"The close finished {close_location:.0f}% of the way from the session low to its high.")
    if discontinuity:
        observations.append("A large price discontinuity requires corporate-action review; multi-session signals and the composite are withheld.")
    metrics = dict(relative_volume=relative_volume,return_1d_pct=daily_return,regular_return_1d_pct=regular_return,regular_close=regular_close,relative_strength_spy_20d_pct=spy_excess,
        relative_strength_sector_20d_pct=sector_excess,close_location_pct=close_location,gap_pct=gap,
        gap_retained_pct=gap_retained,above_vwap_pct=above_vwap,realized_volatility_20d_pct=realized,
        sma20=sma20,sma50=sma50,drawdown_63d_pct=drawdown)
    return {"ticker":symbol,"as_of":now["date"],"source":"ThetaData","feed":feed_metadata or {},
        "coverage_note":f"{len(bars)} daily EOD reports (17:15 ET, may include after-hours); {len(intraday)} regular-session intraday bars. Unadjusted prices, not total returns.",
        "bars":bars[-260:],"intraday":intraday,"sector_symbol":sector_symbol,
        "metrics":{k:rounded(v) for k,v in metrics.items()},
        "score":{"total":rounded(total),"coverage_pct":coverage,"components":components,"version":"participation-v1","label":label},
        "observations":observations,"methodology":[
            "Participation score is a fixed descriptive rubric, not a probability, forecast, or portfolio allocation signal.",
            "Missing measures remain unavailable. A composite requires at least 70% of the intended weight; available weights are rescaled.",
            "Relative volume excludes the current session from its 20-session median baseline. No partial-session/full-day comparison is used.",
            "Gap and closing-strength metrics use regular-session bars only; daily return and relative volume use national EOD reports, which can include after-hours trades.",
            "Intraday VWAP is an approximation from bar typical price weighted by bar volume, not tick-level execution VWAP.",
            "Relative returns use matching session dates and price returns, not dividend-reinvested returns. Corporate actions can affect raw histories."]}
