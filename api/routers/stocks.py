"""
api/routers/stocks.py
═══════════════════════════════════════════════════════════════════════════════
Stock Research endpoints

GET /api/stocks/search?q=AAPL        — ticker autocomplete (yfinance)
GET /api/stocks/screener             — most undervalued S&P 500 vs DCF
GET /api/stocks/{ticker}             — price + fundamentals snapshot
GET /api/stocks/{ticker}/chart       — OHLCV + moving averages
GET /api/stocks/{ticker}/pe          — historical P/E ratio (FMP)
GET /api/stocks/{ticker}/dcf         — DCF valuation (fixed assumptions)
GET /api/stocks/{ticker}/patterns    — pattern matching similar historical periods

FMP free tier: 250 calls/day — used for historical P/E and screener only.
yfinance: used for price data, fundamentals, financials.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["Stocks"])

FMP_KEY = "CJQNAEmAHwczmW95TzmdyIUfghkWOcw8"
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ── DCF fixed assumptions ─────────────────────────────────────────────────────
DCF_WACC          = 0.10   # 10% discount rate
DCF_TERMINAL_RATE = 0.03   # 3% terminal growth
DCF_YEARS         = 5      # 5-year explicit forecast
DCF_FCF_GROWTH    = 0.08   # 8% assumed FCF growth (conservative)

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()

def _get(key: str, ttl: int = 3600):
    with _cache_lock:
        if key in _cache:
            if time.time() - _cache[key]["ts"] < ttl:
                return _cache[key]["data"]
    return None

def _set(key: str, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}

def _fmp(path: str, params: dict = {}):
    """Call FMP API with caching."""
    cache_key = f"fmp:{path}:{sorted(params.items())}"
    cached = _get(cache_key, ttl=3600 * 6)
    if cached is not None:
        return cached
    try:
        r = requests.get(
            f"{FMP_BASE}{path}",
            params={**params, "apikey": FMP_KEY},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            _set(cache_key, data)
            return data
    except Exception:
        pass
    return None

def _yf_info(ticker: str) -> dict:
    cached = _get(f"yf:info:{ticker}", ttl=1800)
    if cached is not None:
        return cached
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        _set(f"yf:info:{ticker}", info)
        return info
    except Exception:
        return {}

def _yf_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    cache_key = f"yf:hist:{ticker}:{period}"
    cached = _get(cache_key, ttl=900)
    if cached is not None:
        return cached
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        _set(cache_key, df)
        return df
    except Exception:
        return pd.DataFrame()


# ── DCF computation ───────────────────────────────────────────────────────────

def _compute_dcf(ticker: str) -> dict:
    """
    Compute intrinsic value using DCF with fixed assumptions.
    Uses trailing free cash flow from yfinance financials.
    """
    try:
        t = yf.Ticker(ticker)
        info = _yf_info(ticker)

        # Get trailing FCF from cash flow statement
        cf = t.cashflow
        fcf = None
        if cf is not None and not cf.empty:
            for row in ["Free Cash Flow", "Total Cash From Operating Activities"]:
                if row in cf.index:
                    vals = cf.loc[row].dropna()
                    if not vals.empty:
                        fcf = float(vals.iloc[0])
                        break

        # Fallback: operatingCashflow - capex from info
        if fcf is None:
            opcf = info.get("operatingCashflow")
            capex = info.get("capitalExpenditures", 0)
            if opcf:
                fcf = float(opcf) - abs(float(capex or 0))

        if not fcf or fcf <= 0:
            return {"error": "Insufficient FCF data", "ticker": ticker}

        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares:
            return {"error": "Shares outstanding unavailable", "ticker": ticker}

        shares = float(shares)
        price  = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # Project FCF for DCF_YEARS years
        projected = []
        cf_t = fcf
        for yr in range(1, DCF_YEARS + 1):
            cf_t *= (1 + DCF_FCF_GROWTH)
            pv = cf_t / ((1 + DCF_WACC) ** yr)
            projected.append({"year": yr, "fcf": round(cf_t), "pv": round(pv)})

        # Terminal value
        terminal_fcf  = cf_t * (1 + DCF_TERMINAL_RATE)
        terminal_val  = terminal_fcf / (DCF_WACC - DCF_TERMINAL_RATE)
        terminal_pv   = terminal_val / ((1 + DCF_WACC) ** DCF_YEARS)

        total_pv    = sum(p["pv"] for p in projected) + terminal_pv
        fair_value  = total_pv / shares

        margin      = ((fair_value - price) / fair_value * 100) if fair_value > 0 else None
        upside      = ((fair_value / price - 1) * 100) if price > 0 else None

        return {
            "ticker":        ticker,
            "fair_value":    round(fair_value, 2),
            "current_price": round(float(price), 2),
            "upside_pct":    round(upside, 1) if upside is not None else None,
            "margin_of_safety": round(margin, 1) if margin is not None else None,
            "trailing_fcf":  round(fcf),
            "shares_out":    round(shares),
            "projected_fcf": projected,
            "terminal_value": round(terminal_pv),
            "total_pv":      round(total_pv),
            "assumptions": {
                "wacc":          DCF_WACC,
                "terminal_rate": DCF_TERMINAL_RATE,
                "fcf_growth":    DCF_FCF_GROWTH,
                "years":         DCF_YEARS,
            }
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# ── Pattern matching ──────────────────────────────────────────────────────────

def _pattern_match(prices: pd.Series, window: int = 30, n_matches: int = 5) -> list:
    """
    Find historical periods with similar price shapes using DTW distance.
    Returns the n_matches most similar periods and what happened next.
    """
    if len(prices) < window * 3:
        return []

    prices = prices.dropna()
    # Normalise recent window
    recent = prices.iloc[-window:].values
    recent_norm = (recent - recent.mean()) / (recent.std() + 1e-9)

    matches = []
    step = max(1, window // 5)

    for i in range(0, len(prices) - window * 2, step):
        segment = prices.iloc[i:i + window].values
        seg_norm = (segment - segment.mean()) / (segment.std() + 1e-9)

        # Simple Euclidean distance on normalised series (fast DTW approximation)
        dist = float(np.sqrt(np.mean((recent_norm - seg_norm) ** 2)))

        # Forward return after this pattern
        fwd_end = min(i + window + 20, len(prices) - 1)
        fwd_ret = float((prices.iloc[fwd_end] / prices.iloc[i + window - 1]) - 1) * 100

        matches.append({
            "start_date": str(prices.index[i].date()),
            "end_date":   str(prices.index[i + window - 1].date()),
            "dist":       dist,
            "fwd_20d_ret": round(fwd_ret, 2),
        })

    matches.sort(key=lambda x: x["dist"])
    top = matches[:n_matches]

    avg_fwd = np.mean([m["fwd_20d_ret"] for m in top])
    pct_positive = sum(1 for m in top if m["fwd_20d_ret"] > 0) / len(top) * 100

    return {
        "matches": top,
        "avg_fwd_20d_ret": round(float(avg_fwd), 2),
        "pct_positive": round(float(pct_positive), 1),
        "window_days": window,
        "summary": f"{pct_positive:.0f}% of similar patterns led to gains over the next 20 days "
                   f"(avg {avg_fwd:+.1f}%)",
    }


# ── S&P 500 tickers ───────────────────────────────────────────────────────────

SP500_SAMPLE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","LLY","AVGO","TSLA",
    "JPM","V","UNH","XOM","MA","HD","PG","COST","JNJ","ABBV",
    "MRK","CRM","BAC","CVX","NFLX","AMD","TMO","KO","PEP","WMT",
    "ADBE","MCD","ACN","LIN","CSCO","ABT","TXN","DHR","NKE","PM",
    "NEE","INTC","QCOM","INTU","DIS","AMGN","BMY","UNP","HON","CAT",
    "GE","LOW","SPGI","MS","AXP","ISRG","SBUX","BLK","GS","SYK",
    "GILD","MDT","CB","ADI","REGN","PLD","VRTX","C","DE","MO",
    "ZTS","BSX","ETN","MMC","PGR","LRCX","SO","DUK","CI","HCA",
    "TJX","EOG","SHW","ELV","ITW","AON","FDX","COP","MCK","GD",
]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stocks/search")
def stock_search(q: str = Query(..., min_length=1)):
    """Ticker autocomplete using FMP search."""
    cached = _get(f"search:{q.upper()}", ttl=86400)
    if cached:
        return {"results": cached}

    results = []
    try:
        data = _fmp(f"/search", {"query": q, "limit": 8, "exchange": "NASDAQ,NYSE"})
        if data:
            results = [
                {
                    "ticker":   r.get("symbol", ""),
                    "name":     r.get("name", ""),
                    "exchange": r.get("exchangeShortName", ""),
                }
                for r in data
                if r.get("symbol") and r.get("name")
            ][:8]
    except Exception:
        pass

    # Fallback: filter SP500_SAMPLE
    if not results:
        q_up = q.upper()
        results = [
            {"ticker": t, "name": t, "exchange": "US"}
            for t in SP500_SAMPLE
            if t.startswith(q_up)
        ][:6]

    _set(f"search:{q.upper()}", results)
    return {"results": results}


@router.get("/stocks/screener")
def stock_screener():
    """
    Run DCF on S&P 500 sample and return most undervalued tickers.
    Cached for 24 hours to preserve FMP API calls.
    """
    cached = _get("screener:sp500", ttl=86400)
    if cached:
        return cached

    results = []
    for ticker in SP500_SAMPLE[:40]:  # limit to 40 for free tier
        dcf = _compute_dcf(ticker)
        if "error" not in dcf and dcf.get("upside_pct") is not None:
            info = _yf_info(ticker)
            results.append({
                "ticker":      ticker,
                "name":        info.get("shortName", ticker),
                "price":       dcf["current_price"],
                "fair_value":  dcf["fair_value"],
                "upside_pct":  dcf["upside_pct"],
                "margin":      dcf["margin_of_safety"],
                "sector":      info.get("sector", ""),
                "pe":          info.get("trailingPE"),
            })

    results.sort(key=lambda x: x["upside_pct"], reverse=True)
    out = {"screener": results[:20], "updated": datetime.now().date().isoformat()}
    _set("screener:sp500", out)
    return out


@router.get("/stocks/{ticker}")
def stock_snapshot(ticker: str):
    """Full snapshot: price, fundamentals, quick DCF."""
    ticker = ticker.upper()
    cached = _get(f"snap:{ticker}", ttl=900)
    if cached:
        return cached

    info = _yf_info(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"{ticker} not found")

    price     = info.get("currentPrice") or info.get("regularMarketPrice")
    prev      = info.get("previousClose")
    chg_pct   = ((price / prev - 1) * 100) if price and prev else None

    snap = {
        "ticker":       ticker,
        "name":         info.get("longName") or info.get("shortName", ticker),
        "sector":       info.get("sector"),
        "industry":     info.get("industry"),
        "price":        round(float(price), 2) if price else None,
        "change_pct":   round(float(chg_pct), 2) if chg_pct else None,
        "market_cap":   info.get("marketCap"),
        "pe_trailing":  info.get("trailingPE"),
        "pe_forward":   info.get("forwardPE"),
        "ps_ratio":     info.get("priceToSalesTrailing12Months"),
        "pb_ratio":     info.get("priceToBook"),
        "ev_ebitda":    info.get("enterpriseToEbitda"),
        "roe":          info.get("returnOnEquity"),
        "roa":          info.get("returnOnAssets"),
        "profit_margin":info.get("profitMargins"),
        "revenue_growth":info.get("revenueGrowth"),
        "earnings_growth":info.get("earningsGrowth"),
        "debt_equity":  info.get("debtToEquity"),
        "current_ratio":info.get("currentRatio"),
        "free_cashflow":info.get("freeCashflow"),
        "dividend_yield":info.get("dividendYield"),
        "52w_high":     info.get("fiftyTwoWeekHigh"),
        "52w_low":      info.get("fiftyTwoWeekLow"),
        "avg_volume":   info.get("averageVolume"),
        "beta":         info.get("beta"),
        "description":  (info.get("longBusinessSummary") or "")[:400],
    }

    _set(f"snap:{ticker}", snap)
    return snap


@router.get("/stocks/{ticker}/chart")
def stock_chart(
    ticker: str,
    period: str = Query("1y", description="1mo|6mo|1y|2y|5y"),
):
    """
    OHLCV + 50MA + 100MA + 200MA for the given period.
    """
    ticker = ticker.upper()
    cache_key = f"chart:{ticker}:{period}"
    cached = _get(cache_key, ttl=900)
    if cached:
        return cached

    df = _yf_history(ticker, period=period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    close = df["Close"].dropna()

    ma50  = close.rolling(50,  min_periods=20).mean()
    ma100 = close.rolling(100, min_periods=40).mean()
    ma200 = close.rolling(200, min_periods=80).mean()

    points = []
    for idx in close.index:
        d = str(idx.date())
        points.append({
            "date":   d,
            "open":   round(float(df["Open"].loc[idx]), 2)   if idx in df.index else None,
            "high":   round(float(df["High"].loc[idx]), 2)   if idx in df.index else None,
            "low":    round(float(df["Low"].loc[idx]), 2)    if idx in df.index else None,
            "close":  round(float(close.loc[idx]), 2),
            "volume": int(df["Volume"].loc[idx]) if idx in df.index and not np.isnan(df["Volume"].loc[idx]) else None,
            "ma50":   round(float(ma50.loc[idx]), 2)  if not np.isnan(ma50.loc[idx])  else None,
            "ma100":  round(float(ma100.loc[idx]), 2) if not np.isnan(ma100.loc[idx]) else None,
            "ma200":  round(float(ma200.loc[idx]), 2) if not np.isnan(ma200.loc[idx]) else None,
        })

    out = {
        "ticker": ticker,
        "period": period,
        "points": points,
        "meta": {
            "last":     round(float(close.iloc[-1]), 2),
            "high_52w": round(float(close.rolling(252).max().iloc[-1]), 2),
            "low_52w":  round(float(close.rolling(252).min().iloc[-1]), 2),
            "count":    len(points),
        }
    }
    _set(cache_key, out)
    return out


@router.get("/stocks/{ticker}/pe")
def stock_pe_history(ticker: str):
    """
    Historical P/E ratio from FMP.
    Falls back to computing trailing P/E from price / EPS if FMP fails.
    """
    ticker = ticker.upper()
    cached = _get(f"pe:{ticker}", ttl=3600 * 12)
    if cached:
        return cached

    points = []

    # Try FMP historical P/E
    data = _fmp(f"/historical-price-full/{ticker}", {"serietype": "line"})
    earnings = _fmp(f"/income-statement/{ticker}", {"limit": 20, "period": "quarter"})

    if earnings:
        # Build quarterly EPS history
        eps_map = {}
        for e in earnings:
            date = e.get("date", "")[:10]
            eps  = e.get("eps")
            if date and eps:
                eps_map[date] = float(eps)

    # Get price history and compute trailing P/E
    df = _yf_history(ticker, period="5y")
    if not df.empty and earnings:
        close = df["Close"].dropna()
        # Use most recent 4 quarters EPS for trailing
        if eps_map:
            sorted_eps = sorted(eps_map.items(), reverse=True)
            # Compute trailing 12m EPS per quarter
            for i in range(len(sorted_eps)):
                date_str = sorted_eps[i][0]
                trailing_eps = sum(v for _, v in sorted_eps[i:i+4])
                if trailing_eps > 0:
                    try:
                        dt = pd.Timestamp(date_str)
                        price_at = close.asof(dt) if dt in close.index or dt >= close.index[0] else None
                        if price_at and not np.isnan(price_at):
                            pe = round(float(price_at) / trailing_eps, 2)
                            if 0 < pe < 500:  # sanity filter
                                points.append({"date": date_str, "pe": pe})
                    except Exception:
                        pass

    points.sort(key=lambda x: x["date"])

    out = {
        "ticker": ticker,
        "points": points,
        "current_pe": points[-1]["pe"] if points else None,
        "avg_pe": round(np.mean([p["pe"] for p in points]), 1) if points else None,
        "min_pe": round(min(p["pe"] for p in points), 1) if points else None,
        "max_pe": round(max(p["pe"] for p in points), 1) if points else None,
    }
    _set(f"pe:{ticker}", out)
    return out


@router.get("/stocks/{ticker}/dcf")
def stock_dcf(ticker: str):
    """Full DCF valuation with fixed assumptions."""
    ticker = ticker.upper()
    cached = _get(f"dcf:{ticker}", ttl=3600 * 12)
    if cached:
        return cached
    result = _compute_dcf(ticker)
    if "error" not in result:
        _set(f"dcf:{ticker}", result)
    return result


@router.get("/stocks/{ticker}/patterns")
def stock_patterns(
    ticker: str,
    window: int = Query(30, description="Pattern window in trading days"),
    period: str = Query("5y", description="History to search"),
):
    """
    Find historical periods with similar price shapes.
    Returns top matches and forward return statistics.
    """
    ticker = ticker.upper()
    cache_key = f"patterns:{ticker}:{window}:{period}"
    cached = _get(cache_key, ttl=3600 * 6)
    if cached:
        return cached

    df = _yf_history(ticker, period=period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    result = _pattern_match(df["Close"], window=window)
    result["ticker"] = ticker
    result["period"] = period

    _set(cache_key, result)
    return result