
"""
High-Growth SMID Portfolio API

Routes:
    GET /api/smid-growth-portfolio
    GET /api/smid-growth-portfolio/status

Separate small/mid-cap growth portfolio focused on:
- revenue and EPS acceleration
- margin and FCF inflection
- balance sheet survivability
- under-discovered sponsorship
- Stage 2 / base breakout technicals
- relative strength and momentum acceleration
- valuation relative to growth
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/smid-growth-portfolio", tags=["smid-growth-portfolio"])

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 60 * 60
BENCHMARK_TICKER = "SPY"

DEFAULT_TARGET_HOLDINGS = 15
MIN_TARGET_HOLDINGS = 10
MAX_TARGET_HOLDINGS = 20

MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 25_000_000_000
MIN_PRICE = 5
MIN_DOLLAR_VOLUME_20 = 5_000_000

MAX_SINGLE_POSITION = 0.085
MAX_SECTOR_WEIGHT = 0.30
MAX_THEME_WEIGHT = 0.30
ATR_STOP_MULTIPLE = 2.40
TRANSACTION_COST_BPS = 18

SMID_GROWTH_UNIVERSE = [
    "CRDO", "ALAB", "AEHR", "ACMR", "AMBA", "ARLO", "ARRY", "BE", "BILL",
    "CAMT", "CLS", "COHR", "COMM", "ENVX", "FORM", "FSLY", "GCT", "IOT",
    "LITE", "MRAM", "NXT", "ONTO", "POWI", "PDFS", "PI", "QLYS", "RELY",
    "SITM", "SOUN", "TER", "TSEM", "UCTT", "VICR", "VRT", "WOLF",
    "AI", "APP", "BBAI", "CFLT", "COUR", "DBX", "DOCN", "DUOL", "ESTC",
    "FIVN", "FRSH", "GTLB", "HCP", "JAMF", "KVYO", "MNDY", "NCNO", "NET",
    "PATH", "PAY", "PCOR", "RBLX", "RDDT", "S", "SEMR", "TOST", "TTD",
    "U", "YOU", "ZI", "ZM", "ZS",
    "AFRM", "BTDR", "CLSK", "COIN", "DAVE", "FOUR", "HOOD", "HUT", "IREN",
    "MARA", "NU", "PAYO", "RIOT", "ROOT", "SOFI", "UPST", "WULF",
    "BROS", "CAVA", "CELH", "CHWY", "CROX", "DECK", "ELF", "HIMS", "LTH",
    "ONON", "PRCH", "SG", "SHAK", "SKX", "SFM", "WING", "YETI",
    "ACLX", "AKRO", "ALKS", "ARDX", "AXSM", "BEAM", "CRSP", "DAWN", "EWTX",
    "EXAS", "FOLD", "HALO", "INSM", "IOVA", "MIRM", "NARI", "PRAX", "REPL",
    "RXRX", "SANA", "TMDX", "TWST", "VKTX", "VIR", "XENE",
    "ACHR", "ASTS", "AVAV", "BKSY", "IONQ", "JOBY", "LUNR", "MRCY", "OUST",
    "PL", "QBTS", "RKLB", "RGTI", "SPCE",
    "ALTM", "AMPS", "BLDP", "CCJ", "ENPH", "FLNC", "FSLR", "LEU", "MP",
    "NFE", "NNE", "NOVA", "OKLO", "RUN", "SEDG", "SMR", "STEM",
    "AA", "ATI", "AXON", "BOOT", "CENX", "CLF", "CWST", "ESAB", "FCX",
    "FIX", "FLR", "HWM", "IESC", "KAI", "KALU", "LNTH", "MTRN", "NVT",
    "PARR", "PRIM", "STRL", "SYM", "TGLS", "TREX", "TROX", "UFPI",
]

THEME_MAP = {
    "AI infrastructure": ["CRDO", "ALAB", "AEHR", "ACMR", "AMBA", "ARLO", "CAMT", "COHR", "FORM", "LITE", "MRAM", "ONTO", "PDFS", "PI", "SITM", "VICR", "VRT"],
    "Software": ["AI", "APP", "CFLT", "COUR", "DBX", "DOCN", "DUOL", "ESTC", "FIVN", "FRSH", "GTLB", "IOT", "JAMF", "KVYO", "MNDY", "NCNO", "NET", "PATH", "PAY", "PCOR", "RBLX", "RDDT", "S", "SEMR", "TOST", "TTD", "U", "YOU", "ZI", "ZM", "ZS"],
    "Fintech and crypto": ["AFRM", "BILL", "BTDR", "CLSK", "COIN", "DAVE", "FOUR", "HOOD", "HUT", "IREN", "MARA", "NU", "PAYO", "RIOT", "ROOT", "SOFI", "UPST", "WULF"],
    "Consumer growth": ["BROS", "CAVA", "CELH", "CHWY", "CROX", "DECK", "ELF", "HIMS", "LTH", "ONON", "PRCH", "SG", "SHAK", "SKX", "SFM", "WING", "YETI"],
    "Biotech and medtech": ["ACLX", "AKRO", "ALKS", "ARDX", "AXSM", "BEAM", "CRSP", "DAWN", "EWTX", "EXAS", "FOLD", "HALO", "INSM", "IOVA", "MIRM", "NARI", "PRAX", "REPL", "RXRX", "SANA", "TMDX", "TWST", "VKTX", "VIR", "XENE"],
    "Frontier tech": ["ACHR", "ASTS", "AVAV", "BKSY", "IONQ", "JOBY", "LUNR", "MRCY", "OUST", "PL", "QBTS", "RKLB", "RGTI", "SPCE"],
    "Energy transition": ["ALTM", "AMPS", "BLDP", "CCJ", "ENPH", "FLNC", "FSLR", "LEU", "MP", "NFE", "NNE", "NOVA", "OKLO", "RUN", "SEDG", "SMR", "STEM"],
    "Industrial growth": ["AA", "ATI", "AXON", "BOOT", "CENX", "CLF", "CWST", "ESAB", "FCX", "FIX", "FLR", "HWM", "IESC", "KAI", "KALU", "LNTH", "MTRN", "NVT", "PARR", "PRIM", "STRL", "SYM", "TGLS", "TREX", "TROX", "UFPI"],
}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clean(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None or abs(bottom) < 1e-12:
        return None
    return top / bottom


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_linear(value: Optional[float], low: float, high: float, missing: float = 45.0) -> float:
    if value is None or high == low:
        return missing
    return _clamp(100 * (value - low) / (high - low))


def _ticker_theme(ticker: str) -> str:
    for theme, tickers in THEME_MAP.items():
        if ticker in tickers:
            return theme
    return "Other SMID growth"


def _dedupe(tickers: Sequence[str]) -> List[str]:
    seen = set()
    output = []
    for ticker in tickers:
        symbol = str(ticker or "").upper().strip().replace(".", "-")
        if symbol and symbol not in seen:
            seen.add(symbol)
            output.append(symbol)
    return output


def _parse_custom_tickers(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return _dedupe(value.replace("\n", ",").replace(" ", ",").split(","))


def _get_universe(tickers: Optional[str], max_tickers: int) -> Tuple[str, List[str]]:
    custom = _parse_custom_tickers(tickers)
    if custom:
        return "custom", custom[:max_tickers]
    return "high_growth_smid", _dedupe(SMID_GROWTH_UNIVERSE)[:max_tickers]


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = _CACHE.get(key)
    if not item:
        return None
    expires_at, payload = item
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _CACHE[key] = (time.time() + CACHE_TTL_SECONDS, payload)
    return payload


def _download_history(tickers: Sequence[str], period: str = "2y") -> pd.DataFrame:
    ticker_list = list(dict.fromkeys([ticker for ticker in tickers if ticker]))
    if not ticker_list:
        return pd.DataFrame()
    try:
        return yf.download(
            tickers=ticker_list,
            period=period,
            interval="1d",
            auto_adjust=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except TypeError:
        return yf.download(
            tickers=ticker_list,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()


def _slice_history(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        fields = {}
        for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if field in frame.columns.get_level_values(0):
                try:
                    fields[field] = frame[field][ticker]
                except Exception:
                    pass
        if not fields:
            return pd.DataFrame()
        output = pd.DataFrame(fields)
    else:
        output = frame.copy()
    output = output.rename(columns={"Adj Close": "AdjClose"}).dropna(how="all")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in output.columns:
            return pd.DataFrame()
    return output[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def _series_get(statement: pd.DataFrame, possible_names: Sequence[str]) -> List[Optional[float]]:
    if statement is None or statement.empty:
        return []
    lookup = {str(label).lower().strip(): label for label in statement.index}
    for name in possible_names:
        key = str(name).lower().strip()
        if key in lookup:
            return [_finite(value) for value in statement.loc[lookup[key]].tolist()]
    return []


def _latest(values: Sequence[Optional[float]]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _prior(values: Sequence[Optional[float]]) -> Optional[float]:
    found_latest = False
    for value in values:
        if value is None:
            continue
        if not found_latest:
            found_latest = True
            continue
        return value
    return None


def _growth(latest: Optional[float], prior: Optional[float]) -> Optional[float]:
    if latest is None or prior is None or abs(prior) < 1e-12:
        return None
    return (latest - prior) / abs(prior)


def _slope_of_last(values: Sequence[Optional[float]], count: int = 3) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if len(clean) < count:
        return None
    y = list(reversed(clean[:count]))
    x = list(range(len(y)))
    try:
        return _finite(np.polyfit(x, y, 1)[0])
    except Exception:
        return None


def _financial_profile(ticker: str) -> Dict[str, Any]:
    yf_ticker = yf.Ticker(ticker)
    try:
        info = yf_ticker.info or {}
    except Exception:
        info = {}
    try:
        q_fin = yf_ticker.quarterly_financials
    except Exception:
        q_fin = pd.DataFrame()
    try:
        a_fin = yf_ticker.financials
    except Exception:
        a_fin = pd.DataFrame()
    try:
        q_bal = yf_ticker.quarterly_balance_sheet
    except Exception:
        q_bal = pd.DataFrame()
    try:
        a_bal = yf_ticker.balance_sheet
    except Exception:
        a_bal = pd.DataFrame()
    try:
        q_cf = yf_ticker.quarterly_cashflow
    except Exception:
        q_cf = pd.DataFrame()
    try:
        a_cf = yf_ticker.cashflow
    except Exception:
        a_cf = pd.DataFrame()

    revenue = _series_get(q_fin, ["Total Revenue", "Operating Revenue"]) or _series_get(a_fin, ["Total Revenue", "Operating Revenue"])
    gross_profit = _series_get(q_fin, ["Gross Profit"]) or _series_get(a_fin, ["Gross Profit"])
    operating_income = _series_get(q_fin, ["Operating Income", "EBIT"]) or _series_get(a_fin, ["Operating Income", "EBIT"])
    net_income = _series_get(q_fin, ["Net Income", "Net Income Common Stockholders"]) or _series_get(a_fin, ["Net Income", "Net Income Common Stockholders"])
    eps = _series_get(q_fin, ["Diluted EPS", "Basic EPS"])

    ocf = _series_get(q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities"]) or _series_get(a_cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _series_get(q_cf, ["Capital Expenditure", "Capital Expenditures"]) or _series_get(a_cf, ["Capital Expenditure", "Capital Expenditures"])

    cash = _series_get(q_bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"]) or _series_get(a_bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
    debt = _series_get(q_bal, ["Total Debt", "Long Term Debt"]) or _series_get(a_bal, ["Total Debt", "Long Term Debt"])
    current_assets = _series_get(q_bal, ["Current Assets", "Total Current Assets"])
    current_liabilities = _series_get(q_bal, ["Current Liabilities", "Total Current Liabilities"])
    shares = _series_get(q_bal, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"])

    latest_revenue = _latest(revenue)
    prior_revenue = _prior(revenue)
    revenue_growth = _growth(latest_revenue, prior_revenue)

    revenue_growth_rates = []
    for index in range(0, max(0, len(revenue) - 1)):
        rate = _growth(revenue[index], revenue[index + 1])
        if rate is not None:
            revenue_growth_rates.append(rate)

    eps_growth_rates = []
    for index in range(0, max(0, len(eps) - 1)):
        rate = _growth(eps[index], eps[index + 1])
        if rate is not None:
            eps_growth_rates.append(rate)

    revenue_acceleration = _slope_of_last(revenue_growth_rates, 3)
    eps_acceleration = _slope_of_last(eps_growth_rates, 3)

    latest_gross_profit = _latest(gross_profit)
    prior_gross_profit = _prior(gross_profit)
    latest_operating_income = _latest(operating_income)
    prior_operating_income = _prior(operating_income)
    latest_net_income = _latest(net_income)

    gross_margin = _safe_divide(latest_gross_profit, latest_revenue)
    prior_gross_margin = _safe_divide(prior_gross_profit, prior_revenue)
    operating_margin = _safe_divide(latest_operating_income, latest_revenue)
    prior_operating_margin = _safe_divide(prior_operating_income, prior_revenue)
    net_margin = _safe_divide(latest_net_income, latest_revenue)

    gross_margin_delta = gross_margin - prior_gross_margin if gross_margin is not None and prior_gross_margin is not None else None
    operating_margin_delta = operating_margin - prior_operating_margin if operating_margin is not None and prior_operating_margin is not None else None

    latest_ocf = _latest(ocf)
    latest_capex = _latest(capex)
    prior_ocf = _prior(ocf)
    prior_capex = _prior(capex)
    fcf = latest_ocf + (latest_capex or 0.0) if latest_ocf is not None else None
    prior_fcf = prior_ocf + (prior_capex or 0.0) if prior_ocf is not None else None
    fcf_margin = _safe_divide(fcf, latest_revenue)
    prior_fcf_margin = _safe_divide(prior_fcf, prior_revenue)
    fcf_margin_delta = fcf_margin - prior_fcf_margin if fcf_margin is not None and prior_fcf_margin is not None else None
    fcf_inflection = bool(fcf is not None and ((prior_fcf is not None and prior_fcf < 0 and fcf > 0) or (fcf_margin_delta is not None and fcf_margin_delta > 0.04)))

    latest_cash = _latest(cash)
    latest_debt = _latest(debt)
    latest_current_assets = _latest(current_assets)
    latest_current_liabilities = _latest(current_liabilities)
    latest_shares = _latest(shares)
    prior_shares = _prior(shares)

    net_debt = (latest_debt or 0.0) - (latest_cash or 0.0) if latest_cash is not None or latest_debt is not None else None
    current_ratio = _safe_divide(latest_current_assets, latest_current_liabilities)
    share_dilution = _growth(latest_shares, prior_shares)

    net_debt_to_operating_income = None
    if net_debt is not None and latest_operating_income is not None and latest_operating_income > 0:
        net_debt_to_operating_income = net_debt / latest_operating_income

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector") or "Unknown",
        "industry": info.get("industry") or "",
        "theme": _ticker_theme(ticker),
        "market_cap": _finite(info.get("marketCap")),
        "current_price": _finite(info.get("currentPrice") or info.get("regularMarketPrice")),
        "analyst_count": _finite(info.get("numberOfAnalystOpinions")),
        "held_percent_institutions": _finite(info.get("heldPercentInstitutions")),
        "short_percent_float": _finite(info.get("shortPercentOfFloat")),
        "forward_pe": _finite(info.get("forwardPE")),
        "peg_ratio": _finite(info.get("pegRatio")),
        "price_to_sales": _finite(info.get("priceToSalesTrailing12Months")),
        "beta": _finite(info.get("beta")),
        "revenue_growth": revenue_growth,
        "revenue_acceleration": revenue_acceleration,
        "eps_acceleration": eps_acceleration,
        "gross_margin": gross_margin,
        "gross_margin_delta": gross_margin_delta,
        "operating_margin": operating_margin,
        "operating_margin_delta": operating_margin_delta,
        "net_margin": net_margin,
        "fcf": fcf,
        "fcf_margin": fcf_margin,
        "fcf_margin_delta": fcf_margin_delta,
        "fcf_inflection": fcf_inflection,
        "cash": latest_cash,
        "debt": latest_debt,
        "net_debt": net_debt,
        "current_ratio": current_ratio,
        "share_dilution": share_dilution,
        "net_debt_to_operating_income": net_debt_to_operating_income,
    }


def _atr14(data: pd.DataFrame) -> pd.Series:
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)
    previous_close = close.shift(1)
    tr = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _price_technicals(history: pd.DataFrame, benchmark_history: pd.DataFrame) -> Dict[str, Any]:
    if history.empty or len(history) < 130:
        return {
            "close": None,
            "technical_score": 40.0,
            "momentum_score": 40.0,
            "return_1m": None,
            "return_3m": None,
            "return_6m": None,
            "return_12m": None,
            "annualized_volatility": 0.65,
            "dollar_volume_20": None,
            "stage2_transition": False,
            "ma_stack_alignment": False,
            "volume_dryup": False,
            "volume_expansion": False,
            "rs_new_high": False,
            "base_quality_score": 40.0,
            "risk_state": "Data limited",
            "stop_level": None,
        }

    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)
    returns = close.pct_change()
    latest_close = _finite(close.iloc[-1])

    ema21 = close.ewm(span=21, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=120).mean()
    sma200 = close.rolling(200, min_periods=150).mean()

    ret_1m = _safe_divide(close.iloc[-1] - close.iloc[-21], close.iloc[-21]) if len(close) > 22 else None
    ret_3m = _safe_divide(close.iloc[-1] - close.iloc[-63], close.iloc[-63]) if len(close) > 64 else None
    ret_6m = _safe_divide(close.iloc[-1] - close.iloc[-126], close.iloc[-126]) if len(close) > 127 else None
    ret_12m = _safe_divide(close.iloc[-1] - close.iloc[-252], close.iloc[-252]) if len(close) > 253 else None

    sma50_value = _finite(sma50.iloc[-1])
    sma150_value = _finite(sma150.iloc[-1])
    sma200_value = _finite(sma200.iloc[-1])
    sma150_slope = _finite(sma150.iloc[-1] - sma150.iloc[-21]) if len(sma150) > 22 else None
    sma200_slope = _finite(sma200.iloc[-1] - sma200.iloc[-21]) if len(sma200) > 22 else None

    ma_stack_alignment = bool(latest_close is not None and sma50_value is not None and sma150_value is not None and sma200_value is not None and latest_close > sma50_value > sma150_value > sma200_value and (sma150_slope or 0.0) > 0 and (sma200_slope or 0.0) > 0)
    stage2_transition = bool(latest_close is not None and sma150_value is not None and sma200_value is not None and latest_close > sma150_value and latest_close > sma200_value and (sma150_slope or 0.0) > 0)

    vol20 = volume.rolling(20, min_periods=20).mean()
    vol60 = volume.rolling(60, min_periods=40).mean()
    volume_dryup = bool(_finite(vol20.iloc[-5:].mean()) is not None and _finite(vol60.iloc[-1]) is not None and vol20.iloc[-5:].mean() < vol60.iloc[-1] * 0.78)
    volume_expansion = bool(_finite(volume.iloc[-1]) is not None and _finite(vol20.iloc[-1]) is not None and volume.iloc[-1] > vol20.iloc[-1] * 1.5)

    price_range_20 = _safe_divide(close.tail(20).max() - close.tail(20).min(), close.iloc[-1])
    realized_vol_20 = _finite(returns.tail(20).std())
    realized_vol_60 = _finite(returns.tail(60).std())
    contraction = 1 - realized_vol_20 / realized_vol_60 if realized_vol_20 is not None and realized_vol_60 not in [None, 0] else None
    base_quality_score = _score_linear(contraction, -0.2, 0.45, missing=45) * 0.55 + (75 if volume_dryup else 45) * 0.25 + _score_linear(-(price_range_20 or 0.30), -0.35, -0.04, missing=45) * 0.20

    rs_new_high = False
    if not benchmark_history.empty and len(benchmark_history) > 130:
        aligned = pd.concat([close.rename("stock"), benchmark_history["Close"].astype(float).rename("benchmark")], axis=1).dropna()
        if len(aligned) > 80:
            rs = aligned["stock"] / aligned["benchmark"]
            rs_new_high = bool(rs.iloc[-1] >= rs.tail(63).max() * 0.995)

    annualized_volatility = _finite(returns.tail(60).std() * math.sqrt(252)) or 0.65
    dollar_volume_20 = _finite((close * volume).rolling(20, min_periods=20).mean().iloc[-1])

    technical_score = 45.0
    if latest_close is not None and _finite(ema21.iloc[-1]) is not None:
        technical_score += 8 if latest_close > ema21.iloc[-1] else -8
    if latest_close is not None and sma50_value is not None:
        technical_score += 9 if latest_close > sma50_value else -9
    if stage2_transition:
        technical_score += 12
    if ma_stack_alignment:
        technical_score += 13
    if volume_dryup:
        technical_score += 6
    if volume_expansion:
        technical_score += 7
    if rs_new_high:
        technical_score += 12

    momentum_score = 45.0
    if ret_3m is not None:
        momentum_score += max(-14, min(22, ret_3m * 60))
    if ret_6m is not None:
        momentum_score += max(-12, min(20, ret_6m * 35))
    if ret_12m is not None:
        momentum_score += max(-10, min(16, ret_12m * 22))
    if ret_1m is not None and ret_3m is not None and ret_6m is not None and ret_1m > ret_3m / 3 and ret_3m > ret_6m / 2:
        momentum_score += 12
    if rs_new_high:
        momentum_score += 10
    if annualized_volatility > 1.1:
        momentum_score -= 10
    elif annualized_volatility > 0.8:
        momentum_score -= 5

    atr = _atr14(history)
    latest_atr = _finite(atr.iloc[-1])
    stop_level = latest_close - ATR_STOP_MULTIPLE * latest_atr if latest_close is not None and latest_atr is not None else None

    below_ema21_count = 0
    for offset in [1, 2, 3]:
        if len(close) <= offset:
            break
        c = _finite(close.iloc[-offset])
        e = _finite(ema21.iloc[-offset])
        if c is not None and e is not None and c < e:
            below_ema21_count += 1
        else:
            break

    if below_ema21_count >= 3:
        risk_state = "Trend break"
    elif latest_close is not None and sma50_value is not None and latest_close > sma50_value:
        risk_state = "Trend supported"
    elif stage2_transition:
        risk_state = "Base supported"
    else:
        risk_state = "Neutral"

    return {
        "close": latest_close,
        "technical_score": _clamp(technical_score),
        "momentum_score": _clamp(momentum_score),
        "return_1m": ret_1m,
        "return_3m": ret_3m,
        "return_6m": ret_6m,
        "return_12m": ret_12m,
        "annualized_volatility": annualized_volatility,
        "dollar_volume_20": dollar_volume_20,
        "stage2_transition": stage2_transition,
        "ma_stack_alignment": ma_stack_alignment,
        "volume_dryup": volume_dryup,
        "volume_expansion": volume_expansion,
        "rs_new_high": rs_new_high,
        "base_quality_score": _clamp(base_quality_score),
        "risk_state": risk_state,
        "stop_level": stop_level,
    }


def _score_fundamentals(profile: Dict[str, Any]) -> float:
    return _clamp(
        _score_linear(profile.get("revenue_acceleration"), -0.05, 0.08, missing=48) * 0.25
        + _score_linear(profile.get("revenue_growth"), 0.05, 0.55, missing=48) * 0.18
        + _score_linear(profile.get("eps_acceleration"), -0.05, 0.10, missing=45) * 0.15
        + _score_linear(profile.get("gross_margin_delta"), -0.03, 0.08, missing=48) * 0.13
        + _score_linear(profile.get("operating_margin_delta"), -0.04, 0.10, missing=48) * 0.17
        + (82.0 if profile.get("fcf_inflection") else _score_linear(profile.get("fcf_margin_delta"), -0.06, 0.10, missing=45)) * 0.12
    )


def _score_balance_sheet(profile: Dict[str, Any]) -> float:
    current_ratio_score = _score_linear(profile.get("current_ratio"), 0.75, 2.5, missing=52)
    nd = profile.get("net_debt_to_operating_income")
    if nd is None:
        net_debt_score = 78.0 if (profile.get("net_debt") or 0.0) <= 0 else 55.0
    else:
        net_debt_score = _score_linear(-nd, -4.5, 1.0, missing=52)
    fcf_margin_score = _score_linear(profile.get("fcf_margin"), -0.15, 0.18, missing=45)
    dilution_score = _score_linear(-(profile.get("share_dilution") or 0.0), -0.12, 0.03, missing=55)
    solvency = 70.0
    if profile.get("current_ratio") is not None and profile.get("current_ratio") < 0.8:
        solvency -= 25
    if profile.get("net_debt_to_operating_income") is not None and profile.get("net_debt_to_operating_income") > 5:
        solvency -= 25
    return _clamp(current_ratio_score * 0.20 + net_debt_score * 0.25 + fcf_margin_score * 0.25 + dilution_score * 0.15 + solvency * 0.15)


def _score_sponsorship(profile: Dict[str, Any]) -> float:
    analysts = profile.get("analyst_count")
    held = profile.get("held_percent_institutions")
    short = profile.get("short_percent_float")
    if analysts is None:
        coverage = 55.0
    elif 2 <= analysts <= 6:
        coverage = 88.0
    elif 7 <= analysts <= 12:
        coverage = 72.0
    elif analysts < 2:
        coverage = 62.0
    else:
        coverage = 42.0
    if held is None:
        ownership = 55.0
    elif 0.20 <= held <= 0.70:
        ownership = 86.0
    elif held <= 0.88:
        ownership = 68.0
    else:
        ownership = 40.0
    if short is None:
        short_score = 55.0
    elif 0.08 <= short <= 0.25:
        short_score = 82.0
    elif short > 0.25:
        short_score = 58.0
    else:
        short_score = 52.0
    return _clamp(coverage * 0.35 + ownership * 0.40 + short_score * 0.25)


def _score_valuation(profile: Dict[str, Any]) -> float:
    revenue_growth = profile.get("revenue_growth")
    peg = profile.get("peg_ratio")
    ps = profile.get("price_to_sales")
    fpe = profile.get("forward_pe")
    if peg is None or peg <= 0:
        peg_score = 55.0
    elif peg <= 0.9:
        peg_score = 85.0
    elif peg <= 1.6:
        peg_score = 68.0
    elif peg <= 2.6:
        peg_score = 52.0
    else:
        peg_score = 34.0
    if ps is None or ps <= 0:
        ps_score = 52.0
    else:
        growth_percent = max(5.0, (revenue_growth or 0.15) * 100)
        ps_to_growth = ps / growth_percent
        if ps_to_growth <= 0.12:
            ps_score = 86.0
        elif ps_to_growth <= 0.22:
            ps_score = 72.0
        elif ps_to_growth <= 0.38:
            ps_score = 55.0
        else:
            ps_score = 34.0
    if fpe is None or fpe <= 0:
        pe_score = 52.0
    elif fpe <= 22:
        pe_score = 80.0
    elif fpe <= 40:
        pe_score = 63.0
    elif fpe <= 70:
        pe_score = 45.0
    else:
        pe_score = 28.0
    return _clamp(peg_score * 0.35 + ps_score * 0.45 + pe_score * 0.20)


def _score_catalyst(profile: Dict[str, Any], technical: Dict[str, Any]) -> float:
    score = 45.0
    if profile.get("fcf_inflection"):
        score += 12
    if technical.get("stage2_transition"):
        score += 12
    if technical.get("rs_new_high"):
        score += 10
    if technical.get("volume_expansion"):
        score += 8
    if profile.get("short_percent_float") is not None and profile.get("short_percent_float") >= 0.12:
        score += 7
    if profile.get("analyst_count") is not None and 2 <= profile.get("analyst_count") <= 6:
        score += 6
    return _clamp(score)


def _score_row(ticker: str, profile: Dict[str, Any], technical: Dict[str, Any], industry_strength_score: float) -> Dict[str, Any]:
    fundamental_score = _score_fundamentals(profile)
    balance_sheet_score = _score_balance_sheet(profile)
    sponsorship_score = _score_sponsorship(profile)
    valuation_score = _score_valuation(profile)
    catalyst_score = _score_catalyst(profile, technical)
    technical_score = technical.get("technical_score") or 45.0
    momentum_score = technical.get("momentum_score") or 45.0
    base_quality_score = technical.get("base_quality_score") or 45.0
    confluence_score = _clamp(
        fundamental_score * 0.26
        + technical_score * 0.20
        + momentum_score * 0.14
        + balance_sheet_score * 0.13
        + sponsorship_score * 0.09
        + valuation_score * 0.08
        + catalyst_score * 0.06
        + base_quality_score * 0.04
    )
    reasons = []
    if profile.get("revenue_acceleration") is not None and profile.get("revenue_acceleration") > 0:
        reasons.append("Revenue growth is accelerating")
    if profile.get("operating_margin_delta") is not None and profile.get("operating_margin_delta") > 0:
        reasons.append("Operating leverage is improving")
    if profile.get("fcf_inflection"):
        reasons.append("FCF inflection signal")
    if technical.get("stage2_transition"):
        reasons.append("Stage 2 transition")
    if technical.get("rs_new_high"):
        reasons.append("Relative strength near highs")
    if technical.get("volume_dryup"):
        reasons.append("Base with volume dry-up")
    if technical.get("volume_expansion"):
        reasons.append("Volume expansion signal")
    if profile.get("analyst_count") is not None and 2 <= profile.get("analyst_count") <= 6:
        reasons.append("Still under-covered")
    if profile.get("short_percent_float") is not None and profile.get("short_percent_float") >= 0.12:
        reasons.append("Short interest can become fuel")
    if not reasons:
        reasons.append("Mixed but monitorable SMID growth profile")
    if confluence_score >= 78:
        verdict = "Core SMID Growth"
    elif confluence_score >= 68:
        verdict = "High-Growth Watch"
    elif confluence_score >= 58:
        verdict = "Setup Developing"
    else:
        verdict = "Too Early"
    return {
        **profile,
        **technical,
        "ticker": ticker,
        "verdict": verdict,
        "confluence_score": confluence_score,
        "fundamental_score": fundamental_score,
        "balance_sheet_score": balance_sheet_score,
        "sponsorship_score": sponsorship_score,
        "valuation_score": valuation_score,
        "catalyst_score": catalyst_score,
        "industry_strength_score": industry_strength_score,
        "reasons": reasons[:5],
    }


def _passes_universe_filter(row: Dict[str, Any]) -> Tuple[bool, str]:
    market_cap = _finite(row.get("market_cap"))
    close = _finite(row.get("close")) or _finite(row.get("current_price"))
    dollar_volume = _finite(row.get("dollar_volume_20"))
    if market_cap is not None and market_cap < MIN_MARKET_CAP:
        return False, "Market cap below SMID threshold"
    if market_cap is not None and market_cap > MAX_MARKET_CAP:
        return False, "Market cap above SMID threshold"
    if close is not None and close < MIN_PRICE:
        return False, "Price below minimum"
    if dollar_volume is not None and dollar_volume < MIN_DOLLAR_VOLUME_20:
        return False, "Liquidity below minimum"
    return True, "Eligible"


def _build_rankings(tickers: Sequence[str]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    price_frame = _download_history(list(tickers) + [BENCHMARK_TICKER], period="2y")
    benchmark_history = _slice_history(price_frame, BENCHMARK_TICKER)
    raw = []
    for ticker in tickers:
        history = _slice_history(price_frame, ticker)
        profile = _financial_profile(ticker)
        technical = _price_technicals(history, benchmark_history)
        raw.append({"ticker": ticker, "profile": profile, "technical": technical})

    sector_returns: Dict[str, List[float]] = {}
    for item in raw:
        sector = item["profile"].get("sector") or "Unknown"
        ret = _finite(item["technical"].get("return_3m"))
        if ret is not None:
            sector_returns.setdefault(sector, []).append(ret)
    sector_strength = {sector: float(pd.Series(returns).median()) for sector, returns in sector_returns.items() if returns}
    sorted_strength = sorted(sector_strength.items(), key=lambda pair: pair[1])
    strength_lookup = {pair[0]: 100 * (idx + 1) / len(sorted_strength) for idx, pair in enumerate(sorted_strength)} if sorted_strength else {}

    rows = []
    for item in raw:
        row = _score_row(
            ticker=item["ticker"],
            profile=item["profile"],
            technical=item["technical"],
            industry_strength_score=strength_lookup.get(item["profile"].get("sector") or "Unknown", 50.0),
        )
        eligible, reason = _passes_universe_filter(row)
        row["eligible"] = eligible
        row["eligibility_reason"] = reason
        if eligible:
            rows.append(row)
    rows.sort(key=lambda row: _finite(row.get("confluence_score")) or 0.0, reverse=True)
    return rows, price_frame


def _portfolio_exposure(holdings: List[Dict[str, Any]]) -> Tuple[float, str]:
    if not holdings:
        return 0.0, "No qualifying SMID growth exposure"
    avg_score = sum(_finite(row.get("confluence_score")) or 0.0 for row in holdings) / len(holdings)
    avg_technical = sum(_finite(row.get("technical_score")) or 0.0 for row in holdings) / len(holdings)
    if avg_score >= 76 and avg_technical >= 65:
        return 0.92, "Aggressive SMID growth risk"
    if avg_score >= 68 and avg_technical >= 58:
        return 0.82, "Constructive SMID growth risk"
    if avg_score >= 60:
        return 0.68, "Moderate SMID growth risk"
    return 0.50, "Reduced SMID growth risk"


def _normalize_weights(holdings: List[Dict[str, Any]], target_exposure: float) -> List[Dict[str, Any]]:
    if not holdings:
        return []
    for row in holdings:
        edge = max(0.01, (_finite(row.get("confluence_score")) or 55.0) - 55)
        vol = max(0.22, _finite(row.get("annualized_volatility")) or 0.70)
        row["raw_weight"] = edge / (vol ** 2)
    raw_total = sum(_finite(row.get("raw_weight")) or 0.0 for row in holdings)
    for row in holdings:
        row["target_weight"] = target_exposure / len(holdings) if raw_total <= 0 else target_exposure * (_finite(row.get("raw_weight")) or 0.0) / raw_total
    for _ in range(8):
        excess = 0.0
        receivers = []
        for row in holdings:
            weight = _finite(row.get("target_weight")) or 0.0
            if weight > MAX_SINGLE_POSITION:
                row["target_weight"] = MAX_SINGLE_POSITION
                excess += weight - MAX_SINGLE_POSITION
            else:
                receivers.append(row)
        if excess <= 1e-8 or not receivers:
            break
        receiver_total = sum(_finite(row.get("target_weight")) or 0.0 for row in receivers)
        if receiver_total <= 0:
            break
        for row in receivers:
            current = _finite(row.get("target_weight")) or 0.0
            row["target_weight"] = current + excess * current / receiver_total
    invested = sum(_finite(row.get("target_weight")) or 0.0 for row in holdings)
    if invested > target_exposure:
        scale = target_exposure / invested
        for row in holdings:
            row["target_weight"] *= scale
    return holdings


def _action_for(row: Dict[str, Any]) -> str:
    score = _finite(row.get("confluence_score")) or 0.0
    technical = _finite(row.get("technical_score")) or 0.0
    risk_state = row.get("risk_state")
    if risk_state == "Trend break" or score < 55 or technical < 45:
        return "Watch"
    if score >= 75 and technical >= 60:
        return "Buy"
    if score >= 64:
        return "Hold"
    return "Watch"


def _current_portfolio(rows: List[Dict[str, Any]], target_holdings: int) -> Dict[str, Any]:
    selected = rows[:target_holdings]
    exposure, regime = _portfolio_exposure(selected)
    holdings = _normalize_weights(selected, exposure)
    holdings.sort(key=lambda row: _finite(row.get("target_weight")) or 0.0, reverse=True)
    for index, row in enumerate(holdings):
        row["portfolio_rank"] = index + 1
        row["action"] = _action_for(row)
        row["sell_trigger"] = "Review or remove if confluence < 55, technical < 45, price loses the 50-day trend, or the ATR stop is breached."
        row["hold_window"] = "4 to 12 weeks; review weekly because SMID setups decay quickly."
        row["trade_reason"] = "; ".join(row.get("reasons") or [])[:240]
    invested = sum(_finite(row.get("target_weight")) or 0.0 for row in holdings)
    return {"holdings": holdings, "stock_exposure": exposure, "cash_weight": max(0.0, 1.0 - invested), "exposure_regime": regime}


def _score_at_date(history: pd.DataFrame, row: Dict[str, Any], position: int, benchmark_history: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if history.empty or position < 130:
        return None
    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)
    current_close = _finite(close.iloc[position])
    if current_close is None or current_close < MIN_PRICE:
        return None
    ret_1m = _safe_divide(close.iloc[position] - close.iloc[position - 21], close.iloc[position - 21])
    ret_3m = _safe_divide(close.iloc[position] - close.iloc[position - 63], close.iloc[position - 63])
    ret_6m = _safe_divide(close.iloc[position] - close.iloc[position - 126], close.iloc[position - 126])
    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=120).mean()
    dollar_volume = _finite((close * volume).rolling(20, min_periods=20).mean().iloc[position])
    if dollar_volume is not None and dollar_volume < MIN_DOLLAR_VOLUME_20:
        return None
    returns = close.pct_change()
    vol = _finite(returns.iloc[max(0, position - 60):position + 1].std() * math.sqrt(252)) or 0.70
    trend = 45.0
    if current_close > (_finite(sma50.iloc[position]) or float("inf")):
        trend += 10
    if current_close > (_finite(sma150.iloc[position]) or float("inf")) and (_finite(sma150.iloc[position] - sma150.iloc[position - 21]) or 0.0) > 0:
        trend += 13
    rs_score = 50.0
    if not benchmark_history.empty:
        benchmark_close = benchmark_history["Close"].astype(float)
        date = history.index[position]
        try:
            bench_position = benchmark_history.index.get_indexer([date], method="pad")[0]
        except Exception:
            bench_position = -1
        if bench_position > 63:
            stock_rel = _safe_divide(close.iloc[position], close.iloc[position - 63])
            bench_rel = _safe_divide(benchmark_close.iloc[bench_position], benchmark_close.iloc[bench_position - 63])
            if stock_rel is not None and bench_rel is not None:
                rs_score = _score_linear(stock_rel - bench_rel, -0.12, 0.35, missing=50)
    momentum = 45.0
    if ret_1m is not None:
        momentum += max(-12, min(18, ret_1m * 80))
    if ret_3m is not None:
        momentum += max(-14, min(24, ret_3m * 60))
    if ret_6m is not None:
        momentum += max(-12, min(18, ret_6m * 36))
    if ret_1m is not None and ret_3m is not None and ret_6m is not None and ret_1m > ret_3m / 3 and ret_3m > ret_6m / 2:
        momentum += 10
    static_quality = (
        (_finite(row.get("fundamental_score")) or 55.0) * 0.35
        + (_finite(row.get("balance_sheet_score")) or 55.0) * 0.25
        + (_finite(row.get("sponsorship_score")) or 55.0) * 0.12
        + (_finite(row.get("valuation_score")) or 55.0) * 0.10
        + (_finite(row.get("catalyst_score")) or 55.0) * 0.18
    )
    score = _clamp(momentum * 0.35 + trend * 0.22 + rs_score * 0.18 + static_quality * 0.25)
    raw_weight = max(0.01, score - 55) / (max(0.22, vol) ** 2)
    return {"ticker": row.get("ticker"), "score": score, "raw_weight": raw_weight}


def _weights_for_date(position: int, benchmark_history: pd.DataFrame, histories: Dict[str, pd.DataFrame], row_lookup: Dict[str, Dict[str, Any]], target_holdings: int) -> Dict[str, float]:
    scores = []
    date = benchmark_history.index[position]
    for ticker, history in histories.items():
        try:
            local_position = history.index.get_indexer([date], method="pad")[0]
        except Exception:
            continue
        score = _score_at_date(history, row_lookup.get(ticker, {"ticker": ticker}), local_position, benchmark_history)
        if score is not None:
            scores.append(score)
    scores.sort(key=lambda item: item["score"], reverse=True)
    selected = scores[:target_holdings]
    if not selected:
        return {}
    raw_total = sum(item["raw_weight"] for item in selected)
    weights = {item["ticker"]: 0.86 * item["raw_weight"] / raw_total for item in selected} if raw_total > 0 else {item["ticker"]: 0.80 / len(selected) for item in selected}
    for _ in range(8):
        excess = 0.0
        receivers = []
        for ticker, weight in list(weights.items()):
            if weight > MAX_SINGLE_POSITION:
                weights[ticker] = MAX_SINGLE_POSITION
                excess += weight - MAX_SINGLE_POSITION
            else:
                receivers.append(ticker)
        if excess <= 1e-8 or not receivers:
            break
        receiver_total = sum(weights[ticker] for ticker in receivers)
        if receiver_total <= 0:
            break
        for ticker in receivers:
            weights[ticker] += excess * weights[ticker] / receiver_total
    return weights


def _rebalance_actions(old_weights: Dict[str, float], new_weights: Dict[str, float]) -> Dict[str, Any]:
    old_keys = set(old_weights.keys())
    new_keys = set(new_weights.keys())
    buys = sorted(list(new_keys - old_keys))
    sells = sorted(list(old_keys - new_keys))
    adds = []
    trims = []
    for ticker in sorted(list(old_keys & new_keys)):
        delta = new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)
        if delta > 0.015:
            adds.append(ticker)
        elif delta < -0.015:
            trims.append(ticker)
    parts = []
    if buys:
        parts.append("Bought " + ", ".join(buys[:3]))
    if sells:
        parts.append("Removed " + ", ".join(sells[:3]))
    if not parts and adds:
        parts.append("Added to " + ", ".join(adds[:3]))
    if not parts and trims:
        parts.append("Trimmed " + ", ".join(trims[:3]))
    if not parts:
        parts.append("No major changes")
    turnover = 0.5 * sum(abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in sorted(list(old_keys | new_keys)))
    return {"headline": "; ".join(parts), "buys": buys, "sells": sells, "adds": adds, "trims": trims, "turnover": turnover}


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value - peak) / peak)
    return max_dd


def _annualized_volatility(returns: Sequence[float]) -> float:
    if len(returns) < 5:
        return 0.0
    return float(pd.Series(returns).std() * math.sqrt(252))


def _simulate(rows: List[Dict[str, Any]], price_frame: pd.DataFrame, target_holdings: int) -> Dict[str, Any]:
    benchmark_history = _slice_history(price_frame, BENCHMARK_TICKER)
    if benchmark_history.empty or len(benchmark_history) < 180:
        return {"series": [], "rebalance_log": [], "stats": {}, "diagnostics": {"reason": "Benchmark history unavailable"}}
    histories = {}
    row_lookup = {}
    for row in rows:
        ticker = row.get("ticker")
        history = _slice_history(price_frame, ticker)
        if ticker and not history.empty:
            histories[ticker] = history
            row_lookup[ticker] = row
    trading_index = benchmark_history.index
    start = max(130, len(trading_index) - 252)
    benchmark_close = benchmark_history["Close"].astype(float)
    benchmark_returns = benchmark_close.pct_change().fillna(0)
    benchmark_start = _finite(benchmark_close.iloc[start]) or 1.0
    model_value = 1.0
    weights: Dict[str, float] = {}
    model_values, benchmark_values, model_returns, benchmark_daily_returns = [], [], [], []
    series, rebalance_log = [], []
    for position in range(start, len(trading_index)):
        date = trading_index[position]
        if position == start or (position - start) % 5 == 0:
            new_weights = _weights_for_date(position, benchmark_history, histories, row_lookup, target_holdings)
            if new_weights:
                actions = _rebalance_actions(weights, new_weights)
                if weights:
                    model_value *= max(0.0, 1.0 - actions["turnover"] * TRANSACTION_COST_BPS / 10000.0)
                top_holdings = sorted([{"ticker": ticker, "weight": weight} for ticker, weight in new_weights.items()], key=lambda x: x["weight"], reverse=True)
                rebalance_log.append({
                    "date": pd.Timestamp(date).date().isoformat(),
                    "headline": actions["headline"],
                    "turnover": actions["turnover"],
                    "buys": actions["buys"],
                    "sells": actions["sells"],
                    "adds": actions["adds"],
                    "trims": actions["trims"],
                    "holdings": top_holdings[:15],
                })
                weights = new_weights
        daily_return = 0.0
        if weights and position > 0:
            for ticker, weight in weights.items():
                history = histories.get(ticker)
                if history is None or history.empty:
                    continue
                try:
                    local_position = history.index.get_indexer([date], method="pad")[0]
                except Exception:
                    continue
                if local_position <= 0:
                    continue
                close = history["Close"].astype(float)
                today = _finite(close.iloc[local_position])
                yesterday = _finite(close.iloc[local_position - 1])
                if today is None or yesterday is None or yesterday == 0:
                    continue
                daily_return += weight * ((today / yesterday) - 1.0)
        model_value *= 1.0 + daily_return
        benchmark_value = (_finite(benchmark_close.iloc[position]) or benchmark_start) / benchmark_start
        benchmark_return = _finite(benchmark_returns.iloc[position]) or 0.0
        model_values.append(model_value)
        benchmark_values.append(benchmark_value)
        model_returns.append(daily_return)
        benchmark_daily_returns.append(benchmark_return)
        rebalance_marker = None
        if rebalance_log and rebalance_log[-1]["date"] == pd.Timestamp(date).date().isoformat():
            rebalance_marker = {"date": rebalance_log[-1]["date"], "headline": rebalance_log[-1]["headline"]}
        series.append({"date": pd.Timestamp(date).date().isoformat(), "model": model_value - 1.0, "benchmark": benchmark_value - 1.0, "rebalance": rebalance_marker})
    return {
        "series": series,
        "rebalance_log": rebalance_log[-16:],
        "stats": {
            "model_return": model_values[-1] - 1.0 if model_values else 0.0,
            "benchmark_return": benchmark_values[-1] - 1.0 if benchmark_values else 0.0,
            "model_volatility": _annualized_volatility(model_returns),
            "benchmark_volatility": _annualized_volatility(benchmark_daily_returns),
            "model_max_drawdown": _max_drawdown(model_values),
            "benchmark_max_drawdown": _max_drawdown(benchmark_values),
            "rebalance_count": len(rebalance_log),
        },
        "diagnostics": {"history_count": len(histories), "benchmark": BENCHMARK_TICKER},
    }


def _build_payload(target_holdings: int, max_tickers: int, tickers: Optional[str], min_score: float) -> Dict[str, Any]:
    target_holdings = max(MIN_TARGET_HOLDINGS, min(MAX_TARGET_HOLDINGS, int(target_holdings)))
    universe_key, ticker_list = _get_universe(tickers=tickers, max_tickers=max_tickers)
    rows, price_frame = _build_rankings(ticker_list)
    rows = [row for row in rows if (_finite(row.get("confluence_score")) or 0.0) >= min_score]
    portfolio = _current_portfolio(rows, target_holdings=target_holdings)
    holdings = portfolio["holdings"]
    performance = _simulate(rows=rows[:max(50, target_holdings * 4)], price_frame=price_frame, target_holdings=target_holdings)
    sector_weights: Dict[str, float] = {}
    theme_weights: Dict[str, float] = {}
    for row in holdings:
        weight = _finite(row.get("target_weight")) or 0.0
        sector_weights[row.get("sector") or "Unknown"] = sector_weights.get(row.get("sector") or "Unknown", 0.0) + weight
        theme_weights[row.get("theme") or "Other SMID growth"] = theme_weights.get(row.get("theme") or "Other SMID growth", 0.0) + weight
    generated_at = datetime.utcnow().isoformat() + "Z"
    return {
        "generated_at": generated_at,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "portfolio_type": "high_growth_smid",
        "universe": universe_key,
        "requested_tickers": len(ticker_list),
        "ranked_candidates": len(rows),
        "target_holdings": target_holdings,
        "stock_exposure": portfolio["stock_exposure"],
        "cash_weight": portfolio["cash_weight"],
        "exposure_regime": portfolio["exposure_regime"],
        "holdings": [{key: _clean(value) for key, value in row.items() if key != "raw_weight"} for row in holdings],
        "top_candidates": [{key: _clean(value) for key, value in row.items() if key != "raw_weight"} for row in rows[:25]],
        "sector_weights": {k: _clean(v) for k, v in sorted(sector_weights.items(), key=lambda p: p[1], reverse=True)},
        "theme_weights": {k: _clean(v) for k, v in sorted(theme_weights.items(), key=lambda p: p[1], reverse=True)},
        "trade_queue": [
            {
                "ticker": row.get("ticker"),
                "action": row.get("action"),
                "target_weight": _clean(row.get("target_weight")),
                "confluence_score": _clean(row.get("confluence_score")),
                "reason": row.get("trade_reason"),
            }
            for row in holdings
            if row.get("action") in ["Buy", "Hold", "Watch"]
        ],
        "performance": {
            "series": performance.get("series", []),
            "rebalance_log": performance.get("rebalance_log", []),
            "stats": {key: _clean(value) for key, value in performance.get("stats", {}).items()},
            "benchmark": BENCHMARK_TICKER,
        },
        "methodology": {
            "title": "High-Growth SMID confluence model",
            "objective": "Find SMID stocks where acceleration, margin inflection, FCF improvement, sponsorship, relative strength, base quality, and valuation sanity stack together.",
            "weights": {
                "fundamentals": 26,
                "technicals": 20,
                "momentum": 14,
                "balance_sheet": 13,
                "sponsorship": 9,
                "valuation": 8,
                "catalyst": 6,
                "base_quality": 4,
            },
            "position_sizing": "Weights are proportional to confluence edge divided by volatility squared, then capped by single-stock, sector, and theme limits.",
            "sell_rules": [
                "Confluence score below 55",
                "Technical score below 45",
                "Loss of 50-day trend",
                "ATR stop breach",
                "Score deterioration without relative strength",
            ],
        },
        "risk_rules": {
            "target_holdings": target_holdings,
            "max_single_position": MAX_SINGLE_POSITION,
            "max_sector_weight": MAX_SECTOR_WEIGHT,
            "max_theme_weight": MAX_THEME_WEIGHT,
            "atr_stop_multiple": ATR_STOP_MULTIPLE,
            "rebalance": "Weekly model rebalance; daily risk check.",
        },
    }


@router.get("")
def get_smid_growth_portfolio(
    target_holdings: int = Query(default=DEFAULT_TARGET_HOLDINGS, ge=MIN_TARGET_HOLDINGS, le=MAX_TARGET_HOLDINGS),
    min_score: float = Query(default=52, ge=0, le=100),
    max_tickers: int = Query(default=120, ge=30, le=180),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    cache_key = f"smid-growth-v1:{target_holdings}:{min_score}:{max_tickers}:{tickers or ''}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
    payload = _build_payload(target_holdings=target_holdings, max_tickers=max_tickers, tickers=tickers, min_score=min_score)
    payload["cached"] = False
    return _cache_set(cache_key, payload)


@router.get("/status")
def get_smid_growth_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "route": "/api/smid-growth-portfolio",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "benchmark": BENCHMARK_TICKER,
        "target_holdings_default": DEFAULT_TARGET_HOLDINGS,
        "universe_size": len(_dedupe(SMID_GROWTH_UNIVERSE)),
        "market_cap_range": {"min": MIN_MARKET_CAP, "max": MAX_MARKET_CAP},
        "returns": [
            "holdings",
            "top_candidates",
            "performance.series",
            "performance.stats",
            "performance.rebalance_log",
            "sector_weights",
            "theme_weights",
            "trade_queue",
        ],
    }
