"""Prospective daily decisions. Published fills and curves are never recomputed."""
from copy import deepcopy


def daily_targets(current, desired, exposure, market, rules, blocked=()):
    """Exit breached published stops; stage ordinary rotation toward today's targets.

    Routine changes: 1 percentage point deadband, 2 points/name/decision and
    10% one-way turnover including cash. SPY balances the allocation. Safety
    exits and restoring hard exposure/concentration limits override pacing.
    """
    old = {r["ticker"]: r for r in current}
    target = {r["ticker"]: deepcopy(r) for r in desired}
    for ticker in blocked:
        target.pop(ticker, None)
    risk = {}
    for ticker, row in old.items():
        if ticker == "SPY":
            continue
        bars = market.get(ticker, [])
        if not bars:
            raise ValueError(f"Daily risk data missing for {ticker}")
        close = bars[-1]["close"]
        stop = row.get("stop_level")
        closes = [bar.get("signal_close", bar["close"]) for bar in bars]
        reason = None
        if stop is not None and close < float(stop):
            reason = "Completed close below published stop"
        elif len(closes) >= 23:
            ema = closes[0]
            values = []
            for price in closes:
                ema += (price-ema)*2/22
                values.append(ema)
            if all(closes[i] < values[i] for i in (-1,-2,-3)) and values[-1] < values[-2]:
                reason = "Three closes below falling EMA21"
        if reason:
            risk[ticker] = reason
            target.pop(ticker, None)
    weights = {t: r["current_weight"] for t,r in old.items()}
    desired_weights = {t:r["target_weight"] for t,r in target.items()}
    desired_weights["SPY"] = max(0, exposure-sum(w for t,w in desired_weights.items() if t != "SPY"))
    significant = any(abs(desired_weights.get(t,0)-weights.get(t,0)) >= .01 for t in set(weights)|set(target))
    exposure_gap = exposure-sum(weights.values())
    if not risk and not significant and abs(exposure_gap) < .01 and .65 <= sum(weights.values()) <= .95:
        return None, {"status":"hold", "reason":"All target differences below 1 percentage point; no completed-session risk trigger", "risk_exits":{}}
    # Move non-risk positions together toward their desired allocations, preserving
    # relative changes while limiting turnover. Never use today's partial bar.
    delta = {t: desired_weights.get(t,0)-weights.get(t,0) for t in set(weights)|set(target) if t != "SPY" and t not in risk}
    largest = max([abs(v) for v in delta.values()] or [0])
    scale = min(1, .02/largest) if largest else 1
    gross = sum(abs(v*scale) for v in delta.values())
    # SPY can take the other leg, so cap non-SPY gross at 10%.
    if gross > .10:
        scale *= .10/gross
    result = {}
    for ticker in sorted(set(old)|set(target)):
        if ticker == "SPY" or ticker in risk:
            continue
        weight = max(0, weights.get(ticker,0)+delta.get(ticker,0)*scale)
        if weight <= 1e-8:
            continue
        row = deepcopy(target.get(ticker) or old[ticker])
        row["target_weight"] = min(weight, rules.get("max_single_position", .12))
        # An existing stop cannot drift down simply because price fell or a trim
        # executed; entry references and protective levels survive ordinary adds.
        prior_stop = old.get(ticker, {}).get("stop_level")
        if prior_stop is not None:
            row["stop_level"] = max(prior_stop, row.get("stop_level") or prior_stop)
        result[ticker] = row
    for key,limit in (("sector",rules.get("max_sector_weight",1)),("theme",rules.get("max_theme_weight",1))):
        groups = {}
        for row in result.values():
            groups.setdefault(row.get(key) or "Unknown",[]).append(row)
        for group in groups.values():
            total=sum(r["target_weight"] for r in group)
            if total > limit:
                for row in group: row["target_weight"] *= limit/total
    total=sum(r["target_weight"] for r in result.values())
    if total > exposure:
        for row in result.values(): row["target_weight"] *= exposure/total
        total=exposure
    reserve=exposure-total
    if reserve > 1e-8:
        row=deepcopy(target.get("SPY") or old.get("SPY") or {"ticker":"SPY","name":"S&P 500 long-exposure sleeve","sector":"Diversified ETF","theme":"Macro long-exposure reserve"})
        row["target_weight"]=reserve
        result["SPY"]=row
    return list(result.values()), {"status":"queued", "reason":"Daily risk exit" if risk else "Gradual daily target adjustment", "risk_exits":risk,
        "routine_name_step":.02,"routine_turnover_budget":.10,"target_deadband":.01,
        "safety_override":"Risk exits and exposure/concentration limits take priority over routine pacing"}
