
"""
Stock Intelligence Rankings API

Routes:
    GET /api/stock-rankings
    GET /api/stock-rankings/status

Ranks stocks by workbook-style quality instead of recent price movement.
Uses yfinance data only. This is analytical ranking, not investment advice.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/stock-rankings", tags=["stock-rankings"])
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 45 * 60
MAX_WORKERS = 8
MAX_LIMIT = 50

QUALITY_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","LLY","JPM","V","MA","COST","NFLX","ORCL","CRM","NOW","AMD","ADBE","INTU",
    "ISRG","ACN","LIN","UBER","BKNG","SPGI","GS","MS","AXP","CAT","AMAT","KLAC","LRCX","ASML","TSM","QCOM","ANET","VRT",
    "PANW","CRWD","NET","DDOG","PLTR","APP","ARM","CRDO","ALAB","SMCI","MU","MRVL","DELL","GEV","ETN","PWR","EME","CEG",
    "GE","RTX","LMT","NOC","TMO","DHR","VRTX","REGN","TMDX","HIMS","TEM","SHOP","SPOT","TTD","DASH","CAVA","ELF","DECK",
    "LULU","COIN","HOOD","MSTR","SOFI","RKLB","ASTS","IONQ","RGTI","CLSK","MARA","CCJ","LEU","OKLO","SMR","MP","FCX","SCCO","NEM","XOM","CVX",
]
GROWTH_UNIVERSE = [
    "NVDA","AMD","AVGO","ARM","PLTR","APP","CRDO","ALAB","SMCI","VRT","ANET","MU","MRVL","TSM","ASML","LRCX","KLAC","AMAT",
    "NET","DDOG","MDB","SNOW","CRWD","PANW","ZS","OKTA","HIMS","TEM","RXRX","VKTX","TMDX","SHOP","UBER","ABNB","DASH","RDDT",
    "RBLX","COIN","HOOD","MSTR","SOFI","AFRM","CAVA","ELF","CELH","SPOT","TTD","PINS","RKLB","ASTS","IONQ","RGTI","QBTS","SOUN","CLSK","MARA","RIOT","HUT","BTDR",
]
LARGE_CAP_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","BRK-B","LLY","JPM","V","XOM","UNH","MA","COST","HD","PG","NFLX","JNJ",
    "ABBV","BAC","KO","PLTR","PM","ORCL","CRM","WMT","CVX","CSCO","ABT","IBM","GE","MRK","TMO","MCD","NOW","ISRG","ACN",
    "LIN","AMD","DIS","AXP","MS","GS","RTX","QCOM","CAT","INTU","UBER","VZ","BKNG","TXN","AMAT","SPGI","PGR","BSX","NEE","BLK","LOW","TJX","C","SYK","UNP","HON","DHR","ADBE",
]
THEME_UNIVERSE = [
    "NVDA","AMD","AVGO","SMCI","VRT","CEG","GEV","ETN","PWR","EME","CRDO","ALAB","ARM","PLTR","APP","NET","DDOG","CRWD","PANW",
    "MP","ALB","LAC","FCX","SCCO","CCJ","LEU","SMR","OKLO","NNE","RKLB","ASTS","LUNR","ACHR","JOBY","IONQ","RGTI","QBTS","QUBT","COIN","MSTR","HOOD","MARA","RIOT","CLSK","HUT","BTDR",
]
UNIVERSES = {"quality": QUALITY_UNIVERSE, "growth": GROWTH_UNIVERSE, "large_cap": LARGE_CAP_UNIVERSE, "themes": THEME_UNIVERSE}

def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item: return None
    expires_at, payload = item
    if time.time() >= expires_at:
        _CACHE.pop(key, None); return None
    return payload

def _cache_set(key: str, payload: Dict[str, Any], ttl: int):
    _CACHE[key] = (time.time() + ttl, payload)
    return payload

def _finite(value: Any) -> Optional[float]:
    try: number = float(value)
    except (TypeError, ValueError): return None
    return number if math.isfinite(number) else None

def _clean(value: Any, digits: int = 4):
    if value is None: return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.floating): value = float(value)
    if isinstance(value, float): return round(value, digits) if math.isfinite(value) else None
    try:
        if pd.isna(value): return None
    except Exception: pass
    return value

def _safe_divide(a: Any, b: Any) -> Optional[float]:
    x, y = _finite(a), _finite(b)
    if x is None or y is None or abs(y) < 1e-12: return None
    return x / y

def _clamp(v: float, lo: float=0, hi: float=100) -> float:
    return max(lo, min(hi, v))

def _clean_ticker(t: str) -> str:
    return str(t or '').upper().strip().replace('.', '-')

def _dedupe(tickers: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for ticker in tickers:
        s = _clean_ticker(ticker)
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _parse_custom(v: Optional[str]) -> List[str]:
    if not v: return []
    return _dedupe(v.replace('\n', ',').replace(' ', ',').split(','))

def _get_universe(universe: str, tickers: Optional[str], max_tickers: int):
    custom = _parse_custom(tickers)
    if custom: return 'custom', custom[:max_tickers]
    key = str(universe or 'quality').lower().strip()
    if key not in UNIVERSES: key = 'quality'
    return key, _dedupe(UNIVERSES[key])[:max_tickers]

def _series_get(statement: pd.DataFrame, names: Sequence[str]) -> List[Optional[float]]:
    if statement is None or statement.empty: return []
    lookup = {str(label).lower().strip(): label for label in statement.index}
    for name in names:
        key = str(name).lower().strip()
        if key in lookup:
            return [_finite(v) for v in statement.loc[lookup[key]].tolist()]
    return []

def _latest(values):
    for v in values:
        if v is not None: return v
    return None

def _prior(values):
    first = False
    for v in values:
        if v is None: continue
        if not first: first = True; continue
        return v
    return None

def _growth(latest, prior):
    if latest is None or prior is None or abs(prior) < 1e-12: return None
    return (latest - prior) / abs(prior)

def _score_growth(v): return 45.0 if v is None else _clamp(50 + v * 125)
def _score_margin(v): return 45.0 if v is None else _clamp(v * 180)
def _score_fcf(v): return 45.0 if v is None else _clamp(50 + v * 160)

def _score_balance(cash, debt, op_income):
    cash, debt = cash or 0.0, debt or 0.0
    if debt <= 0 and cash > 0: base = 90.0
    elif debt <= 0: base = 70.0
    else: base = _clamp(55 + ((cash - debt) / debt) * 25)
    if op_income is not None and op_income > 0 and debt > 0:
        ratio = debt / op_income
        if ratio < 1: base += 10
        elif ratio > 5: base -= 18
        elif ratio > 3: base -= 8
    return _clamp(base)

def _score_valuation(info, revenue_growth, net_margin):
    fpe = _finite(info.get('forwardPE')); tpe = _finite(info.get('trailingPE'))
    ps = _finite(info.get('priceToSalesTrailing12Months')); ev = _finite(info.get('enterpriseToEbitda'))
    pe = fpe if fpe and fpe > 0 else tpe
    if pe is None or pe <= 0: pe_score = 50
    elif pe <= 15: pe_score = 85
    elif pe <= 25: pe_score = 72
    elif pe <= 40: pe_score = 55
    elif pe <= 70: pe_score = 38
    else: pe_score = 22
    if ps is None or ps <= 0: ps_score = 50
    elif ps <= 3: ps_score = 80
    elif ps <= 7: ps_score = 63
    elif ps <= 12: ps_score = 45
    else: ps_score = 25
    if ev is None or ev <= 0: ev_score = 50
    elif ev <= 12: ev_score = 78
    elif ev <= 20: ev_score = 62
    elif ev <= 35: ev_score = 42
    else: ev_score = 25
    adj = 0
    if revenue_growth is not None and revenue_growth > .2: adj += 8
    if net_margin is not None and net_margin > .15: adj += 7
    return _clamp(pe_score*.45 + ps_score*.30 + ev_score*.25 + adj)

def _download_price_history(tickers):
    try:
        return yf.download(list(tickers), period='1y', interval='1d', auto_adjust=False, repair=True, progress=False, group_by='column', threads=True)
    except TypeError:
        return yf.download(list(tickers), period='1y', interval='1d', auto_adjust=False, progress=False, group_by='column', threads=True)

def _slice_history(frame, ticker):
    if frame is None or frame.empty: return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        fields = {}
        for field in ['Open','High','Low','Close','Adj Close','Volume']:
            if field in frame.columns.get_level_values(0):
                try: fields[field] = frame[field][ticker]
                except Exception: pass
        if not fields: return pd.DataFrame()
        out = pd.DataFrame(fields)
    else: out = frame.copy()
    out = out.rename(columns={'Adj Close': 'AdjClose'}).dropna(how='all')
    for col in ['Close','Volume']:
        if col not in out.columns: return pd.DataFrame()
    return out.dropna(subset=['Close','Volume'])

def _score_technical(history):
    if history.empty or len(history) < 80:
        return {'technical_score':45, 'return_1m':None, 'return_3m':None, 'return_6m':None, 'close':None, 'rvol':None}
    close = history['Close'].astype(float); volume = history['Volume'].astype(float)
    latest = _finite(close.iloc[-1])
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50 = close.rolling(50, min_periods=50).mean().iloc[-1]
    sma200 = close.rolling(200, min_periods=120).mean().iloc[-1]
    r1 = _safe_divide(close.iloc[-1]-close.iloc[-21], close.iloc[-21]) if len(close)>22 else None
    r3 = _safe_divide(close.iloc[-1]-close.iloc[-63], close.iloc[-63]) if len(close)>64 else None
    r6 = _safe_divide(close.iloc[-1]-close.iloc[-126], close.iloc[-126]) if len(close)>127 else None
    rvol = _safe_divide(volume.iloc[-1], volume.rolling(20, min_periods=20).mean().iloc[-1])
    score = 50.0
    for level, bonus, penalty in [(ema21,10,-8),(sma50,12,-10),(sma200,14,-12)]:
        val = _finite(level)
        if latest is not None and val is not None: score += bonus if latest > val else penalty
    if r3 is not None: score += max(-12, min(18, r3*55))
    if r6 is not None: score += max(-10, min(16, r6*35))
    if rvol is not None and rvol > 1.4 and r1 is not None and r1 > 0: score += 5
    return {'technical_score':_clamp(score), 'return_1m':r1, 'return_3m':r3, 'return_6m':r6, 'close':latest, 'rvol':rvol}

def _analyze_ticker(ticker, history):
    t = yf.Ticker(ticker)
    try: info = t.info or {}
    except Exception: info = {}
    try: financials = t.financials
    except Exception: financials = pd.DataFrame()
    try: bs = t.balance_sheet
    except Exception: bs = pd.DataFrame()
    try: cf = t.cashflow
    except Exception: cf = pd.DataFrame()
    name = info.get('shortName') or info.get('longName') or ticker
    sector = info.get('sector') or 'Unknown'; industry = info.get('industry') or ''
    market_cap = _finite(info.get('marketCap'))
    revenue = _series_get(financials, ['Total Revenue','Operating Revenue'])
    gross = _series_get(financials, ['Gross Profit'])
    op = _series_get(financials, ['Operating Income','EBIT'])
    net = _series_get(financials, ['Net Income','Net Income Common Stockholders'])
    ocf = _series_get(cf, ['Operating Cash Flow','Total Cash From Operating Activities'])
    capex = _series_get(cf, ['Capital Expenditure','Capital Expenditures'])
    cash = _series_get(bs, ['Cash And Cash Equivalents','Cash Cash Equivalents And Short Term Investments','Cash And Short Term Investments'])
    debt = _series_get(bs, ['Total Debt','Long Term Debt'])
    rev_latest, rev_prior = _latest(revenue), _prior(revenue)
    gp, opi, ni = _latest(gross), _latest(op), _latest(net)
    ocf_latest, capex_latest = _latest(ocf), _latest(capex)
    cash_latest, debt_latest = _latest(cash), _latest(debt)
    revenue_growth = _growth(rev_latest, rev_prior)
    gross_margin = _safe_divide(gp, rev_latest)
    operating_margin = _safe_divide(opi, rev_latest)
    net_margin = _safe_divide(ni, rev_latest)
    fcf = (ocf_latest + (capex_latest or 0)) if ocf_latest is not None else None
    fcf_margin = _safe_divide(fcf, rev_latest)
    fundamental = _clamp(_score_growth(revenue_growth)*.34 + _score_margin(gross_margin)*.18 + _score_margin(operating_margin)*.20 + _score_margin(net_margin)*.14 + _score_fcf(fcf_margin)*.14)
    balance = _score_balance(cash_latest, debt_latest, opi)
    technical = _score_technical(history); technical_score = technical['technical_score']
    valuation = _score_valuation(info, revenue_growth, net_margin)
    momentum_quality = _clamp(technical_score*.55 + fundamental*.30 + balance*.15)
    composite = _clamp(fundamental*.35 + technical_score*.30 + balance*.20 + valuation*.10 + momentum_quality*.05)
    reasons = []
    if fundamental >= 75: reasons.append('Strong fundamental profile')
    elif fundamental >= 60: reasons.append('Improving fundamental profile')
    elif fundamental < 45: reasons.append('Weak fundamental profile')
    if technical_score >= 75: reasons.append('Strong technical leadership')
    elif technical_score >= 60: reasons.append('Constructive technical setup')
    elif technical_score < 45: reasons.append('Weak technical profile')
    if balance >= 75: reasons.append('Healthy balance sheet')
    elif balance < 45: reasons.append('Balance sheet risk')
    if valuation >= 65: reasons.append('Valuation is not demanding versus quality')
    elif valuation < 40: reasons.append('Valuation requires stronger execution')
    if not reasons: reasons.append('Mixed profile with enough evidence to monitor')
    verdict = 'Favored' if composite >= 78 else 'High Quality Watch' if composite >= 65 else 'Watch' if composite >= 55 else 'Mixed'
    return {
        'ticker':ticker, 'name':name, 'sector':sector, 'industry':industry, 'verdict':verdict,
        'stock_intelligence_score':composite, 'fundamental_score':fundamental, 'technical_score':technical_score,
        'balance_sheet_score':balance, 'valuation_score':valuation, 'momentum_quality_score':momentum_quality,
        'market_cap':market_cap, 'close':technical['close'], 'return_1m':technical['return_1m'], 'return_3m':technical['return_3m'], 'return_6m':technical['return_6m'], 'rvol':technical['rvol'],
        'revenue_growth':revenue_growth, 'gross_margin':gross_margin, 'operating_margin':operating_margin, 'net_margin':net_margin, 'fcf_margin':fcf_margin,
        'cash':cash_latest, 'debt':debt_latest, 'forward_pe':_finite(info.get('forwardPE')), 'price_to_sales':_finite(info.get('priceToSalesTrailing12Months')), 'reasons':reasons[:4],
    }

def _scan_rankings(universe_key, tickers, limit, min_score):
    price_frame = _download_price_history(tickers)
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_analyze_ticker, ticker, _slice_history(price_frame, ticker)): ticker for ticker in tickers}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try: row = fut.result()
            except Exception: row = None
            if row is None: failed.append(ticker)
            else: rows.append(row)
    rows = [r for r in rows if (r.get('stock_intelligence_score') or 0) >= min_score]
    rows.sort(key=lambda r: r.get('stock_intelligence_score') or 0, reverse=True)
    rows = rows[:limit]
    return {
        'generated_at': datetime.utcnow().isoformat()+'Z', 'cache_ttl_seconds':CACHE_TTL_SECONDS, 'universe':universe_key,
        'requested_tickers':len(tickers), 'returned_rows':len(rows), 'failed_tickers':failed[:50],
        'rows':[{k:_clean(v) for k,v in r.items()} for r in rows],
        'methodology': {'purpose':'Rank stocks by workbook-quality scores rather than recent price movement.', 'weights':{'fundamental_score':35, 'technical_score':30, 'balance_sheet_score':20, 'valuation_score':10, 'momentum_quality_score':5}, 'note':'Analytical ranking only. Not investment advice.'},
    }

@router.get('')
def get_stock_rankings(
    universe: str = Query(default='quality'),
    limit: int = Query(default=25, ge=10, le=MAX_LIMIT),
    min_score: float = Query(default=0, ge=0, le=100),
    max_tickers: int = Query(default=90, ge=10, le=160),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    key, ticker_list = _get_universe(universe, tickers, max_tickers)
    cache_key = f"stock-rankings:{key}:{','.join(ticker_list)}:{limit}:{min_score}:{max_tickers}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None: return {**cached, 'cached': True}
    payload = _scan_rankings(key, ticker_list, limit, min_score)
    payload['cached'] = False
    return _cache_set(cache_key, payload, CACHE_TTL_SECONDS)

@router.get('/status')
def get_stock_rankings_status() -> Dict[str, Any]:
    return {'status':'ok', 'route':'/api/stock-rankings', 'cache_ttl_seconds':CACHE_TTL_SECONDS, 'universes':sorted(UNIVERSES.keys()), 'default_universe':'quality', 'weights':{'fundamental_score':35, 'technical_score':30, 'balance_sheet_score':20, 'valuation_score':10, 'momentum_quality_score':5}}


@router.get("/{ticker}")
def get_single_stock_ranking(
    ticker: str,
    refresh: bool = Query(default=False),
) -> Dict[str, Any]:
    symbol = _clean_ticker(ticker)

    if not symbol:
        raise HTTPException(status_code=400, detail="Ticker is required.")

    payload = get_stock_rankings(
        universe="quality",
        limit=10,
        min_score=0,
        max_tickers=10,
        refresh=refresh,
        tickers=symbol,
    )

    rows = payload.get("rows") or []

    for row in rows:
        if row.get("ticker") == symbol:
            return {
                "generated_at": payload.get("generated_at"),
                "cached": payload.get("cached", False),
                "ranking": row,
                "methodology": payload.get("methodology"),
            }

    raise HTTPException(
        status_code=404,
        detail=f"No ranking data available for {symbol}.",
    )
