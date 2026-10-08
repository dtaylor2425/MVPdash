"""Pure near-term options premium scanner; no network, Greeks or prediction claims."""
from datetime import date
from typing import Any
import numpy as np
import pandas as pd
from api.services.options_flow_metrics import classify_trades, normalize_trade_quote, _clean

METHODOLOGY_VERSION = "scanner-1.1.0"
WINDOWS = ("next_session", "7d", "30d")


def _condition_code(value):
    """Theta numeric codes may arrive as ints, floats or numeric strings."""
    try:
        number = float(value)
        return str(int(number)) if np.isfinite(number) and number.is_integer() else "unknown"
    except (TypeError, ValueError, OverflowError):
        return "unknown"


def build_scanner_symbol(ticker, trades, market_date, *, window="7d", next_session=None,
                         completed=False, collection_error=None, condition_allowlist=(0,18,95),
                         min_premium=0, max_spread_ratio=.5):
    if window not in WINDOWS:
        raise ValueError("Unknown expiry window")
    market_date = pd.Timestamp(market_date).date()
    if window == "next_session" and next_session is None:
        raise ValueError("Explicit next exchange session is required")
    target = pd.Timestamp(next_session).date() if next_session else None
    if target and target <= market_date:
        raise ValueError("Next session must follow market date")
    out = dict(ticker=str(ticker).upper(), marketDate=market_date.isoformat(), window=window,
               completed=bool(completed), status="failed" if collection_error else "no_eligible_trades",
               methodologyVersion=METHODOLOGY_VERSION, confidence="unavailable", confidenceBasis="directional_classification_coverage", contracts=[],
               historicalUnusualness=None, summary="Collection failed." if collection_error else "No eligible future-expiry trades in this window.")
    for key in ("netBullishPremium", "bullishPremium", "bearishPremium", "callBuyPremium", "putSellPremium",
                "callSellPremium", "putBuyPremium", "grossPremium", "unclassifiedPremium"):
        out[key] = None if collection_error else 0.0
    quality = dict(classificationCoverage=None, excludedConditionTrades=0, ambiguousQuoteTrades=0,
                   eligibleTrades=0, classifiableTrades=0, conditionPremiumHistogram=[], warnings=["Trade direction is inferred from quotes; opening positions and strategy intent are unknown."])
    out["quality"] = quality
    if collection_error:
        quality["warnings"].append("Source collection was incomplete; ranking suppressed.")
        return out
    if trades is None or trades.empty:
        return out
    frame = trades.copy()
    expiry = pd.to_datetime(frame["expiration"], errors="coerce").dt.date
    dte = expiry.map(lambda d: (d-market_date).days if isinstance(d,date) else -1)
    mask = expiry.eq(target) if window == "next_session" else dte.between(1, int(window[:-1]))
    frame = frame.loc[mask].copy()
    if frame.empty:
        return out
    # Keep questionable prints in gross coverage, never attribute them as bullish.
    frame = classify_trades(frame)
    frame = frame.loc[(frame.premium > 0) & np.isfinite(frame.premium)].copy()
    if frame.empty:
        return out
    condition_codes = frame.condition.map(_condition_code) if "condition" in frame else pd.Series("unknown", index=frame.index)
    if "condition_eligible" in frame:
        condition_ok = frame.condition_eligible.eq(True)
    elif condition_allowlist is not None and "condition" in frame:
        allowed = {_condition_code(x) for x in condition_allowlist} - {"unknown"}
        condition_ok = condition_codes.isin(allowed)
    else:
        condition_ok = pd.Series(False,index=frame.index)
        quality["warnings"].append("Standard trade conditions were not verified; direction is suppressed.")
    if "is_multileg" in frame:
        condition_ok &= ~frame.is_multileg.fillna(False).astype(bool)
    spread = frame.ask-frame.bid
    mid = (frame.ask+frame.bid)/2
    quote_ok = frame.valid & (spread>0) & (spread/mid<=max_spread_ratio)
    # Prints beyond the contemporaneous quote cannot be trusted as aggressive orders.
    quote_ok &= (frame.price>=frame.bid) & (frame.price<=frame.ask)
    contract_ok = pd.to_numeric(frame.strike, errors="coerce").gt(0) & np.isfinite(pd.to_numeric(frame.strike, errors="coerce"))
    if "right_valid" in frame:
        contract_ok &= frame.right_valid.eq(True)
    valid = condition_ok & quote_ok & contract_ok
    frame["weighted"] = frame.premium * frame.aggressor.abs() * valid.astype(float)
    buy = frame.aggressor > 0
    for key, side in {"callBuyPremium":frame.is_call & buy,
                      "putSellPremium":~frame.is_call & ~buy,
                      "callSellPremium":frame.is_call & ~buy,
                      "putBuyPremium":~frame.is_call & buy}.items():
        frame[key] = frame.weighted.where(side,0)
        out[key] = float(frame[key].sum())
    out["bullishPremium"] = out["callBuyPremium"]+out["putSellPremium"]
    out["bearishPremium"] = out["callSellPremium"]+out["putBuyPremium"]
    out["netBullishPremium"] = out["bullishPremium"]-out["bearishPremium"]
    out["grossPremium"] = float(frame.premium.sum())
    attributed = out["bullishPremium"]+out["bearishPremium"]
    out["unclassifiedPremium"] = max(0.,out["grossPremium"]-attributed)
    coverage = attributed/out["grossPremium"]
    quality.update(classificationCoverage=coverage,excludedConditionTrades=int((~condition_ok).sum()),
                   ambiguousQuoteTrades=int((~quote_ok | frame.aggressor.eq(0)).sum()),eligibleTrades=len(frame),
                   classifiableTrades=int((valid & frame.aggressor.ne(0)).sum()))
    for code in sorted(condition_codes.unique()):
        mask=condition_codes.eq(code)
        quality["conditionPremiumHistogram"].append(dict(condition=code,tradeCount=int(mask.sum()),
            grossPremium=float(frame.loc[mask,"premium"].sum()),
            attributedPremium=float(frame.loc[mask,"weighted"].sum()),
            excludedPremium=float(frame.loc[mask & ~condition_ok,"premium"].sum())))
    out["confidence"] = "high" if coverage>=.8 else "medium" if coverage>=.5 else "low"
    out["status"] = "ok" if attributed>0 and coverage>=.5 and out["grossPremium"]>=min_premium else "insufficient_quality"
    if not completed:
        quality["warnings"].append("Partial session totals are not comparable with completed sessions.")
    for (expiration,strike,is_call), part in frame.groupby(["expiration","strike","is_call"],sort=True):
        record=dict(expiration=str(expiration),strike=float(strike),right="C" if is_call else "P",
                    grossPremium=float(part.premium.sum()),tradeCount=len(part),
                    totalContracts=float(part["size"].sum()))
        largest=part.sort_values("premium",ascending=False,kind="stable").iloc[0]
        trade_time=pd.to_datetime(largest.get("ts"),utc=True,errors="coerce")
        record["largestTrade"]={"size":float(largest["size"]),"premium":float(largest["premium"]),
            "time":None if pd.isna(trade_time) else trade_time.isoformat(),
            "attributedPremium":float(largest["weighted"])}
        for key in ("callBuyPremium","putSellPremium","callSellPremium","putBuyPremium"):
            record[key]=float(part[key].sum())
        record["netBullishPremium"]=record["callBuyPremium"]+record["putSellPremium"]-record["callSellPremium"]-record["putBuyPremium"]
        out["contracts"].append(record)
    out["contracts"].sort(key=lambda x:(-x["netBullishPremium"],x["expiration"],x["strike"],x["right"]))
    out["summary"] = ("Quote-inferred bullish premium exceeds bearish offsets." if out["netBullishPremium"]>0
                      else "Bullish premium does not exceed bearish offsets.") if out["status"]=="ok" else "Insufficient classified premium for a directional shortlist."
    return _clean(out)


def build_scanner_snapshot(ticker, market_date, contracts, *, next_session, completed=False,
                           collection_error=None, condition_allowlist=(0,18,95), history=None, profile_identity=None):
    frames=[]
    if not collection_error:
        for contract in contracts:
            raw=contract["trades"]
            if raw is None or raw.empty:
                continue
            normalized=normalize_trade_quote(raw,contract["expiration"],pd.Timestamp(market_date).date())
            normalized["right_valid"] = raw["right"].astype(str).str.upper().str.strip().isin(["C", "CALL", "P", "PUT"]).reset_index(drop=True)
            for key in ("condition","condition_eligible","is_multileg"):
                if key in raw:
                    normalized[key]=raw[key].reset_index(drop=True)
            frames.append(normalized)
    trades=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    snapshot = dict(ticker=str(ticker).upper(),marketDate=pd.Timestamp(market_date).date().isoformat(),
                methodologyVersion=METHODOLOGY_VERSION, profileIdentity=profile_identity,
                windows={w:build_scanner_symbol(ticker,trades,market_date,window=w,next_session=next_session,
                    completed=completed,collection_error=collection_error,condition_allowlist=condition_allowlist) for w in WINDOWS})
    apply_historical_unusualness(snapshot, history or [])
    return snapshot


def apply_historical_unusualness(snapshot, history):
    """Prior 20 completed exchange sessions, identical collection profile/window.

    Percentiles describe gross observed activity, including prints excluded from
    directional inference. Low classification coverage does not invalidate a
    completed same-profile gross total. These are not return forecasts or tests
    of statistical significance.
    Duplicate dates, changed profiles, gaps and unknown coverage fail closed.
    """
    from api.services.options_flow_calendar import _calendar, sessions_before
    if not snapshot.get("profileIdentity") or _calendar() is None:
        return snapshot
    expected = [d.isoformat() for d in sessions_before(pd.Timestamp(snapshot["marketDate"]).date(),20)]
    by_date = {}
    duplicates = set()
    for prior in history:
        if (prior.get("marketDate") not in expected or prior.get("profileIdentity") != snapshot["profileIdentity"]
                or prior.get("methodologyVersion") != METHODOLOGY_VERSION
                or prior.get("ticker") != snapshot["ticker"]):
            continue
        day=prior["marketDate"]
        if day in by_date:
            duplicates.add(day)
        by_date[day]=prior
    if duplicates or any(day not in by_date for day in expected):
        return snapshot
    for window, current in snapshot["windows"].items():
        if not current.get("completed") or current.get("status") not in {"ok", "insufficient_quality", "no_eligible_trades"}:
            continue
        rows=[by_date[day].get("windows",{}).get(window,{}) for day in expected]
        if any(not row.get("completed") or row.get("status") not in {"ok","insufficient_quality","no_eligible_trades"}
               or row.get("window") != window or row.get("marketDate") != day
               for row,day in zip(rows,expected)):
            continue
        values=[row.get("grossPremium") for row in rows]
        if any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in values):
            continue
        value=current.get("grossPremium")
        if not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
            continue
        mean=float(np.mean(values))
        current["historicalUnusualness"]=_clean(dict(metric="grossPremium",sampleSize=20,
            percentile=100.0*sum(v<=value for v in values)/20,
            priorMean=mean,relativeToMean=value/mean if mean>0 else None,
            firstSession=min(expected),lastSession=max(expected),
            interpretation="Gross observed premium, including unclassified trades, versus the prior 20 comparable completed sessions; not directional conviction or a return forecast."))
    return snapshot


def rank_scanner(snapshots, window="7d", limit=20):
    if window not in WINDOWS:
        raise ValueError("Unknown expiry window")
    dates={s["marketDate"] for s in snapshots}
    if len(dates)>1:
        raise ValueError("Cannot rank mixed market sessions")
    rows=[s["windows"][window] for s in snapshots]
    rows.sort(key=lambda r:(r["status"]!="ok",-(r.get("netBullishPremium") or 0),r["ticker"]))
    shortlist=[r for r in rows if r["status"]=="ok" and r.get("completed") and r["netBullishPremium"]>0][:limit]
    return dict(window=window,marketDate=next(iter(dates),None),rows=rows,shortlist=shortlist,
                status="ok" if shortlist else "no_eligible_candidates",
                failedTickers=[r["ticker"] for r in rows if r["status"]=="failed"],methodologyVersion=METHODOLOGY_VERSION)
