"""
api/routers/momentum.py
═══════════════════════════════════════════════════════════════════════════════
Momentum screener endpoints — Tao-of-Trading strategy

GET /api/momentum/historical/{symbol}   — 300d OHLCV + pre-computed indicators
POST /api/momentum/batch                — multi-symbol OHLCV + indicators
GET /api/momentum/screener              — run full screener pipeline

All FMP calls are server-side only. FMP key never touches the browser.
Caching: OHLCV 6h, screener 4h (data doesn't change intraday until close).
"""

import os, time, threading, math
from typing import Optional, List
from pydantic import BaseModel
import numpy as np
import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["Momentum"])

FMP_KEY  = os.getenv("FMP_KEY", "CJQNAEmAHwczmW95TzmdyIUfghkWOcw8")
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
_lock = threading.Lock()

def _get(key: str, ttl: int):
    with _lock:
        if key in _cache and (time.time() - _cache[key]["ts"]) < ttl:
            return _cache[key]["data"]
    return None

def _set(key: str, data):
    with _lock:
        _cache[key] = {"data": data, "ts": time.time()}

# ── FMP helpers ───────────────────────────────────────────────────────────────

def _fmp_get(path: str, params: dict = {}, timeout: int = 15):
    try:
        r = requests.get(
            f"{FMP_BASE}{path}",
            params={**params, "apikey": FMP_KEY},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def _fetch_ohlcv_single(symbol: str) -> list:
    """300 bars for a single symbol, oldest-first."""
    key = f"ohlcv:{symbol}"
    cached = _get(key, 6 * 3600)
    if cached is not None:
        return cached
    data = _fmp_get(f"/historical-price-full/{symbol}", {"timeseries": 300})
    if not data or not data.get("historical"):
        return []
    bars = [
        {
            "date":   d["date"],
            "open":   float(d["open"]),
            "high":   float(d["high"]),
            "low":    float(d["low"]),
            "close":  float(d["close"]),
            "volume": int(d.get("volume") or 0),
        }
        for d in reversed(data["historical"])
    ]
    _set(key, bars)
    return bars

def _fetch_ohlcv_batch(symbols: list) -> dict:
    """
    Fetch up to 5 symbols per FMP batch call.
    Returns dict: { symbol -> [bars] }
    """
    result = {}
    # Check cache first
    to_fetch = []
    for s in symbols:
        cached = _get(f"ohlcv:{s}", 6 * 3600)
        if cached is not None:
            result[s] = cached
        else:
            to_fetch.append(s)

    # Batch fetch in groups of 5
    BATCH_SIZE = 5
    for i in range(0, len(to_fetch), BATCH_SIZE):
        group = to_fetch[i:i + BATCH_SIZE]
        joined = ",".join(group)
        data = _fmp_get(f"/historical-price-full/{joined}", {"timeseries": 300})
        if not data:
            continue
        # FMP returns either a single object or {"historicalStockList": [...]}
        if "historicalStockList" in data:
            items = data["historicalStockList"]
        elif "historical" in data:
            items = [{"symbol": group[0], "historical": data["historical"]}]
        else:
            continue
        for item in items:
            sym = item.get("symbol", "").upper()
            hist = item.get("historical", [])
            if not hist:
                continue
            bars = [
                {
                    "date":   d["date"],
                    "open":   float(d["open"]),
                    "high":   float(d["high"]),
                    "low":    float(d["low"]),
                    "close":  float(d["close"]),
                    "volume": int(d.get("volume") or 0),
                }
                for d in reversed(hist)
            ]
            result[sym] = bars
            _set(f"ohlcv:{sym}", bars)

    return result

# ── Indicator math ────────────────────────────────────────────────────────────

def _ema(values: list, period: int) -> list:
    out = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    s = sum(values[:period]) / period
    out[period - 1] = s
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out

def _sma(values: list, period: int) -> list:
    out = [None] * len(values)
    if len(values) < period:
        return out
    s = sum(values[:period])
    out[period - 1] = s / period
    for i in range(period, len(values)):
        s += values[i] - values[i - period]
        out[i] = s / period
    return out

def _true_range(high, low, prev_close):
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

def _atr(bars: list, period: int = 14) -> list:
    trs = [_true_range(b["high"], b["low"], bars[i-1]["close"] if i > 0 else None)
           for i, b in enumerate(bars)]
    out = [None] * len(bars)
    if len(trs) < period:
        return out
    s = sum(trs[:period])
    out[period - 1] = s / period
    for i in range(period, len(bars)):
        out[i] = (out[i-1] * (period - 1) + trs[i]) / period
    return out

def _adx(bars: list, period: int = 13) -> list:
    n = len(bars)
    out = [None] * n
    if n < period + 1:
        return out
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up   = bars[i]["high"] - bars[i-1]["high"]
        down = bars[i-1]["low"]  - bars[i]["low"]
        plus_dm[i]  = up   if up > down and up > 0   else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0
        tr[i] = _true_range(bars[i]["high"], bars[i]["low"], bars[i-1]["close"])

    def wilder(arr):
        r = [None] * n
        s = sum(arr[1:period+1])
        r[period] = s
        for i in range(period + 1, n):
            r[i] = r[i-1] - r[i-1] / period + arr[i]
        return r

    sp = wilder(plus_dm)
    sm = wilder(minus_dm)
    st = wilder(tr)

    dx = [None] * n
    for i in range(period, n):
        if st[i] is None or st[i] == 0:
            continue
        pdi = (sp[i] / st[i]) * 100
        mdi = (sm[i] / st[i]) * 100
        denom = pdi + mdi
        dx[i] = 0.0 if denom == 0 else abs(pdi - mdi) / denom * 100

    first = period * 2
    if first >= n:
        return out
    s = sum(x for x in dx[period:first] if x is not None)
    out[first - 1] = s / period
    for i in range(first, n):
        if dx[i] is not None and out[i-1] is not None:
            out[i] = (out[i-1] * (period - 1) + dx[i]) / period
    return out

def _slow_stoch(bars: list, k_period: int = 8, k_slow: int = 3, d_period: int = 3):
    n = len(bars)
    fast_k = [None] * n
    for i in range(k_period - 1, n):
        hh = max(b["high"] for b in bars[i - k_period + 1:i + 1])
        ll = min(b["low"]  for b in bars[i - k_period + 1:i + 1])
        fast_k[i] = 50.0 if hh == ll else (bars[i]["close"] - ll) / (hh - ll) * 100

    def sma_nullable(arr, p):
        r = [None] * len(arr)
        vals = [v if v is not None else 0.0 for v in arr]
        s = sum(vals[:p])
        r[p-1] = s / p if arr[p-1] is not None else None
        for i in range(p, len(arr)):
            s += vals[i] - vals[i-p]
            r[i] = s / p if arr[i] is not None else None
        return r

    slow_k = sma_nullable(fast_k, k_slow)
    slow_d = sma_nullable(slow_k, d_period)
    return {"k": slow_k, "d": slow_d}

def _compute_indicators(bars: list) -> dict:
    """Compute all indicators for a bar list. Returns dict of named arrays."""
    closes = [b["close"] for b in bars]
    e8   = _ema(closes, 8)
    e21  = _ema(closes, 21)
    e34  = _ema(closes, 34)
    e55  = _ema(closes, 55)
    e89  = _ema(closes, 89)
    s50  = _sma(closes, 50)
    s100 = _sma(closes, 100)
    s200 = _sma(closes, 200)
    atr14 = _atr(bars, 14)
    adx13 = _adx(bars, 13)
    stoch = _slow_stoch(bars, 8, 3, 3)

    # Keltner channels: 21 EMA ± 1/2/3 × ATR(14)
    def kc_band(mult):
        return [
            (e21[i] + mult * atr14[i]) if (e21[i] is not None and atr14[i] is not None) else None
            for i in range(len(bars))
        ]

    return {
        "ema8":   e8,  "ema21": e21, "ema34": e34, "ema55": e55, "ema89": e89,
        "sma50":  s50, "sma100": s100, "sma200": s200,
        "atr14":  atr14, "adx13": adx13,
        "stoch_k": stoch["k"], "stoch_d": stoch["d"],
        "kc_mid": e21,
        "kc_u1": kc_band(1),  "kc_u2": kc_band(2),  "kc_u3": kc_band(3),
        "kc_l1": kc_band(-1), "kc_l2": kc_band(-2), "kc_l3": kc_band(-3),
    }

# ── Trend scoring ─────────────────────────────────────────────────────────────

def _score_trend(bars: list, ind: dict) -> dict:
    """
    Returns score 0-5, direction, check details, and screener metrics.
    """
    n = len(bars)
    if n == 0:
        return None
    i = n - 1
    close  = bars[i]["close"]
    e8     = ind["ema8"][i]
    e21    = ind["ema21"][i]
    e34    = ind["ema34"][i]
    e55    = ind["ema55"][i]
    e89    = ind["ema89"][i]
    s50    = ind["sma50"][i]
    adx_v  = ind["adx13"][i]
    atr_v  = ind["atr14"][i]

    if any(v is None for v in [e8, e21, e34, e55, e89, s50]):
        return None

    bull_stack = e8 > e21 > e34 > e55 > e89
    bear_stack = e8 < e21 < e34 < e55 < e89

    checks = []

    # 1. EMAs stacked
    checks.append({
        "label": "EMAs stacked",
        "pass":  bull_stack or bear_stack,
        "detail": "bullish" if bull_stack else "bearish" if bear_stack else "tangled",
    })

    # 2. 21 EMA slope (proxy for higher-highs / lower-lows)
    lookback = min(40, i)
    e21_prev = ind["ema21"][i - lookback]
    slope_pct = 0.0
    if e21_prev and e21_prev != 0:
        slope_pct = ((e21 - e21_prev) / e21_prev) * 100
    checks.append({
        "label": "Trend slope",
        "pass":  abs(slope_pct) > 0.5,
        "detail": f"{slope_pct:+.2f}% over {lookback}b",
    })

    # 3. Correct side of 50 SMA
    correct_50 = (bull_stack and close > s50) or (bear_stack and close < s50)
    checks.append({
        "label": "Correct side of 50 SMA",
        "pass":  correct_50,
        "detail": f"close {'>' if close > s50 else '<'} 50sma",
    })

    # 4. ADX >= 20
    checks.append({
        "label": "ADX(13) ≥ 20",
        "pass":  adx_v is not None and adx_v >= 20,
        "detail": f"{adx_v:.1f}" if adx_v is not None else "—",
    })

    # 5. Trend duration: bars 21 EMA has been monotonic
    duration = 0
    for k in range(i, 0, -1):
        prev = ind["ema21"][k-1]
        curr = ind["ema21"][k]
        if curr is None or prev is None:
            break
        if bull_stack and curr > prev:
            duration += 1
        elif bear_stack and curr < prev:
            duration += 1
        else:
            break
    checks.append({
        "label": "Trend duration ≥ 40 bars",
        "pass":  duration >= 40,
        "detail": f"{duration} bars",
    })

    score     = sum(1 for c in checks if c["pass"])
    direction = "UPTREND" if bull_stack else "DOWNTREND" if bear_stack else "CHOP"

    # Screener metrics
    spread_atr = abs(e8 - e89) / atr_v if atr_v else None
    dist_21    = (close - e21) / atr_v  if atr_v else None

    return {
        "score":       score,
        "max":         5,
        "direction":   direction,
        "checks":      checks,
        "adx":         round(adx_v, 1) if adx_v else None,
        "duration":    duration,
        "slope_pct":   round(slope_pct, 3),
        "spread_atr":  round(spread_atr, 2) if spread_atr is not None else None,
        "dist_21_atr": round(dist_21, 2)    if dist_21 is not None else None,
        "above_50":    bool(close > s50),
    }

def _safe_floats(d: dict) -> dict:
    """Replace NaN/inf with None recursively."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, list):
            out[k] = [None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x for x in v]
        else:
            out[k] = v
    return out

# ── Screener universe ─────────────────────────────────────────────────────────

# Full universe — S&P 500 large caps + mid caps + high-momentum names
# Grouped by category for readability
SCREENER_UNIVERSE = [
    # Mega cap tech
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL",
    "ADBE","CRM","INTU","NOW","SNOW","PANW","CRWD","ZS","DDOG","NET",
    "AMD","INTC","QCOM","TXN","AMAT","LRCX","KLAC","MRVL","MU","SMCI",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","V","MA","PYPL","SQ",
    "SPGI","MCO","ICE","CME","CBOE","HOOD","COIN","MSTR",
    # Healthcare / biotech
    "LLY","UNH","JNJ","ABBV","MRK","PFE","TMO","ABT","DHR","BSX",
    "MDT","SYK","ISRG","VRTX","REGN","AMGN","BIIB","GILD","BMY","CI",
    "HCA","ELV","CVS","MCK","CAH",
    # Consumer
    "AMZN","WMT","COST","HD","LOW","TGT","TJX","SBUX","MCD","YUM",
    "NKE","LULU","DECK","ONON","SKX","CMG","DPZ","WING","SHAK",
    "KO","PEP","PM","MO","STZ","BUD","MNST","CELH",
    # Industrials / energy
    "CAT","DE","HON","GE","ETN","EMR","ROK","PH","ITW","MMM",
    "XOM","CVX","COP","EOG","SLB","HAL","MPC","VLO","PSX","OXY",
    "NEE","SO","DUK","AEP","EXC","PCG","D",
    # Growth / high momentum
    "PLTR","RBLX","DKNG","SOFI","AFRM","UPST","OPEN","CVNA","CARVANA",
    "SHOP","MELI","SE","GRAB","BABA","JD","PDD","BIDU",
    "UBER","LYFT","ABNB","DASH","RDFN",
    "TSLA","RIVN","NIO","LCID","FSR","XPEV","LI",
    "SPOT","NFLX","DIS","PARA","WBD","ROKU",
    "TWLO","OKTA","GTLB","HCP","CFLT","MDB","ESTC",
    # Commodities / materials / metals
    "GLD","SLV","GDX","GDXJ","GOLD","NEM","AEM","WPM","FNV",
    "FCX","SCCO","VALE","RIO","BHP","AA","X","CLF","NUE","STLD",
    "CF","MOS","NTR","ADM","BG",
    # Real estate / REITs
    "PLD","AMT","EQIX","CCI","SPG","O","VICI","WELL","DLR","PSA",
    # Small / mid cap momentum
    "CELH","AXON","PODD","TMDX","HRMY","RXRX","ACHR","JOBY","LILM",
    "APP","CIEN","ONTO","IPGP","COHU","FORM","RMBS","UCTT",
    "DUOL","HIMS","GKOS","INSP","SWAV","NTRA","ARDX","TGTX",
]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/momentum/historical/{symbol}")
def momentum_historical(symbol: str):
    """
    Returns 300d of OHLCV + all pre-computed indicators for a single symbol.
    Frontend renders — no indicator math in the browser.
    """
    sym = symbol.upper()
    cache_key = f"mom_hist:{sym}"
    cached = _get(cache_key, 6 * 3600)
    if cached:
        return cached

    bars = _fetch_ohlcv_single(sym)
    if not bars:
        raise HTTPException(status_code=404, detail=f"No data for {sym}")

    ind   = _compute_indicators(bars)
    trend = _score_trend(bars, ind)

    # Fib auto-anchor on full window
    closes = [b["close"] for b in bars]
    highs  = [b["high"]  for b in bars]
    lows   = [b["low"]   for b in bars]
    hi     = max(highs[-130:])
    lo     = min(lows[-130:])
    hi_idx = highs.index(max(highs[-130:], key=lambda x: x), len(highs) - 130)
    lo_idx = lows.index(min(lows[-130:],   key=lambda x: x), len(lows)  - 130)
    ascending = lo_idx < hi_idx
    rng    = hi - lo
    fib_levels = [
        {"ratio": 0,     "label": "0.000",       "golden": False},
        {"ratio": 0.236, "label": "0.236",        "golden": False},
        {"ratio": 0.382, "label": "0.382",        "golden": False},
        {"ratio": 0.5,   "label": "0.500",        "golden": False},
        {"ratio": 0.618, "label": "0.618 ⌬",      "golden": True},
        {"ratio": 0.786, "label": "0.786",        "golden": False},
        {"ratio": 1,     "label": "1.000",        "golden": False},
    ]
    for f in fib_levels:
        f["price"] = round(hi - rng * f["ratio"] if ascending else lo + rng * f["ratio"], 4)

    out = _safe_floats({
        "symbol":     sym,
        "bars":       bars,
        "indicators": ind,
        "trend":      trend,
        "fib_levels": fib_levels,
        "last_price": bars[-1]["close"],
        "prev_close": bars[-2]["close"] if len(bars) > 1 else None,
    })
    _set(cache_key, out)
    return out


class BatchRequest(BaseModel):
    symbols: List[str]

@router.post("/momentum/batch")
def momentum_batch(req: BatchRequest):
    """
    Batch fetch OHLCV + indicators for multiple symbols.
    Uses FMP batch endpoint — ceil(N/5) API calls instead of N.
    """
    symbols = [s.upper() for s in req.symbols[:50]]  # cap at 50
    ohlcv   = _fetch_ohlcv_batch(symbols)
    results = {}
    for sym, bars in ohlcv.items():
        if not bars:
            continue
        ind   = _compute_indicators(bars)
        trend = _score_trend(bars, ind)
        if trend is None:
            continue
        results[sym] = {
            "symbol":     sym,
            "last_price": bars[-1]["close"],
            "trend":      trend,
            # Return last 60 closes for sparkline
            "sparkline":  [b["close"] for b in bars[-60:]],
        }
    return {"results": results}


@router.get("/momentum/screener")
def momentum_screener(
    direction:      str   = Query("ANY"),
    min_score:      int   = Query(3),
    min_adx:        float = Query(20.0),
    min_duration:   int   = Query(40),
    require_50:     bool  = Query(True),
    max_spread_atr: Optional[float] = Query(None),
    max_dist_21:    Optional[float] = Query(None),
    max_tickers:    int   = Query(60),
):
    class _F:
        pass
    filters = _F()
    filters.direction      = direction
    filters.min_score      = min_score
    filters.min_adx        = min_adx
    filters.min_duration   = min_duration
    filters.require_50     = require_50
    filters.max_spread_atr = max_spread_atr
    filters.max_dist_21    = max_dist_21
    filters.max_tickers    = max_tickers
    """
    Full screener pipeline — pre-filter universe, score, rank, return results.
    Cached 4h. POST so filters aren't in the URL.
    """
    cache_key = f"screener:{direction}:{min_score}:{min_adx}:{min_duration}:{require_50}:{max_spread_atr}:{max_dist_21}:{max_tickers}"
    cached = _get(cache_key, 4 * 3600)
    if cached:
        return cached

    universe = SCREENER_UNIVERSE[:filters.max_tickers]
    ohlcv    = _fetch_ohlcv_batch(universe)

    results = []
    for sym, bars in ohlcv.items():
        if len(bars) < 100:
            continue
        ind   = _compute_indicators(bars)
        trend = _score_trend(bars, ind)
        if trend is None:
            continue

        # Direction filter
        if filters.direction == "UP"   and trend["direction"] != "UPTREND":   continue
        if filters.direction == "DOWN" and trend["direction"] != "DOWNTREND": continue
        if trend["direction"] == "CHOP": continue

        # Score filter
        if trend["score"] < filters.min_score: continue

        # ADX filter
        if trend["adx"] is None or trend["adx"] < filters.min_adx: continue

        # Duration filter
        if trend["duration"] < filters.min_duration: continue

        # 50 SMA side
        if filters.require_50 and not trend["above_50"] and trend["direction"] == "UPTREND": continue
        if filters.require_50 and trend["above_50"]    and trend["direction"] == "DOWNTREND": continue

        # EMA compactness
        if filters.max_spread_atr is not None and trend["spread_atr"] is not None:
            if trend["spread_atr"] > filters.max_spread_atr: continue

        # Distance from 21 EMA
        if filters.max_dist_21 is not None and trend["dist_21_atr"] is not None:
            if abs(trend["dist_21_atr"]) > filters.max_dist_21: continue

        results.append({
            "symbol":      sym,
            "score":       trend["score"],
            "direction":   trend["direction"],
            "last_price":  bars[-1]["close"],
            "adx":         trend["adx"],
            "duration":    trend["duration"],
            "slope_pct":   trend["slope_pct"],
            "spread_atr":  trend["spread_atr"],
            "dist_21_atr": trend["dist_21_atr"],
            "above_50":    trend["above_50"],
            "sparkline":   [b["close"] for b in bars[-60:]],
        })

    results.sort(key=lambda x: (-x["score"], -(x["adx"] or 0)))
    out = {"results": results, "scanned": len(ohlcv), "matched": len(results)}
    _set(cache_key, out)
    return out