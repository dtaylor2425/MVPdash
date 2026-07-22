
"""
Timeout-safe High-Growth SMID Portfolio API

Routes:
    GET /api/smid-growth-portfolio
    GET /api/smid-growth-portfolio/status

This version is designed to avoid 5-minute browser/Railway timeouts.

Main changes:
    1. Fast technical pre-scan first.
    2. Fundamentals are fetched only for the strongest technical candidates.
    3. Fundamentals fetch has a hard time budget.
    4. Missing fundamentals fall back to neutral scores instead of blocking.
    5. Cached responses return instantly.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/smid-growth-portfolio", tags=["smid-growth-portfolio"])

CACHE_TTL_SECONDS = 60 * 60
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

BENCHMARK_TICKER = "SPY"
DEFAULT_TARGET_HOLDINGS = 15
MIN_TARGET_HOLDINGS = 10
MAX_TARGET_HOLDINGS = 20

DEFAULT_MAX_TICKERS = 60
MAX_FUNDAMENTAL_FETCH = 30
FUNDAMENTAL_TIME_BUDGET_SECONDS = 32
MAX_WORKERS = 8

MIN_PRICE = 5
MIN_DOLLAR_VOLUME_20 = 4_000_000
MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 35_000_000_000

MAX_SINGLE_POSITION = 0.085
MAX_SECTOR_WEIGHT = 0.30
MAX_THEME_WEIGHT = 0.30
ATR_STOP_MULTIPLE = 2.40
TRANSACTION_COST_BPS = 18

SMID_GROWTH_UNIVERSE = [
    "CRDO", "ALAB", "AEHR", "ACMR", "AMBA", "ARLO", "ARRY", "BE", "BILL",
    "CAMT", "CLS", "COHR", "COMM", "ENVX", "FORM", "FSLY", "GCT", "IOT",
    "LITE", "MRAM", "NXT", "ONTO", "PDFS", "PI", "QLYS", "RELY", "SITM",
    "SOUN", "TER", "TSEM", "UCTT", "VICR", "VRT", "WOLF", "AI", "APP",
    "BBAI", "CFLT", "COUR", "DBX", "DOCN", "DUOL", "ESTC", "FIVN", "FRSH",
    "GTLB", "HCP", "JAMF", "KVYO", "MNDY", "NCNO", "NET", "PATH", "PAY",
    "PCOR", "RBLX", "RDDT", "S", "SEMR", "TOST", "TTD", "U", "YOU", "ZI",
    "ZM", "ZS", "AFRM", "BTDR", "CLSK", "COIN", "DAVE", "FOUR", "HOOD",
    "HUT", "IREN", "MARA", "NU", "PAYO", "RIOT", "ROOT", "SOFI", "UPST",
    "WULF", "BROS", "CAVA", "CELH", "CHWY", "CROX", "DECK", "ELF", "HIMS",
    "LTH", "ONON", "SG", "SHAK", "SKX", "SFM", "WING", "YETI", "ACLX",
    "AKRO", "ALKS", "ARDX", "AXSM", "BEAM", "CRSP", "EXAS", "HALO", "INSM",
    "IOVA", "NARI", "RXRX", "TMDX", "TWST", "VKTX", "XENE", "ACHR", "ASTS",
    "AVAV", "BKSY", "IONQ", "JOBY", "LUNR", "MRCY", "OUST", "PL", "QBTS",
    "RKLB", "RGTI", "ALTM", "CCJ", "ENPH", "FLNC", "FSLR", "LEU", "MP",
    "NNE", "OKLO", "RUN", "SMR", "STEM", "AA", "ATI", "AXON", "BOOT",
    "CENX", "CLF", "CWST", "ESAB", "FIX", "FLR", "HWM", "IESC", "KAI",
    "KALU", "LNTH", "MTRN", "PARR", "PRIM", "STRL", "SYM", "TGLS", "TREX"
]

THEME_MAP = {
    "AI infrastructure": ["CRDO", "ALAB", "AEHR", "ACMR", "AMBA", "CAMT", "COHR", "FORM", "LITE", "ONTO", "PDFS", "PI", "SITM", "VICR", "VRT"],
    "Software": ["AI", "APP", "CFLT", "DBX", "DOCN", "DUOL", "ESTC", "FRSH", "GTLB", "IOT", "MNDY", "NET", "PATH", "PCOR", "RDDT", "S", "TOST", "TTD", "U", "ZS"],
    "Fintech and crypto": ["AFRM", "BTDR", "CLSK", "COIN", "DAVE", "FOUR", "HOOD", "HUT", "IREN", "MARA", "NU", "PAYO", "RIOT", "ROOT", "SOFI", "UPST", "WULF"],
    "Consumer growth": ["BROS", "CAVA", "CELH", "CHWY", "CROX", "DECK", "ELF", "HIMS", "ONON", "SG", "SHAK", "SFM", "WING"],
    "Biotech and medtech": ["ACLX", "AKRO", "ALKS", "ARDX", "AXSM", "BEAM", "CRSP", "EXAS", "HALO", "INSM", "IOVA", "NARI", "RXRX", "TMDX", "TWST", "VKTX", "XENE"],
    "Frontier tech": ["ACHR", "ASTS", "AVAV", "BKSY", "IONQ", "JOBY", "LUNR", "MRCY", "OUST", "PL", "QBTS", "RKLB", "RGTI"],
    "Energy transition": ["ALTM", "CCJ", "ENPH", "FLNC", "FSLR", "LEU", "MP", "NNE", "OKLO", "RUN", "SMR", "STEM"],
    "Industrial growth": ["AA", "ATI", "AXON", "BOOT", "CENX", "CLF", "CWST", "ESAB", "FIX", "FLR", "HWM", "IESC", "KAI", "KALU", "LNTH", "MTRN", "PARR", "PRIM", "STRL", "SYM", "TGLS", "TREX"],
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


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_linear(value: Optional[float], low: float, high: float, missing: float = 50.0) -> float:
    if value is None or high == low:
        return missing
    return _clamp(100 * (value - low) / (high - low))


def _safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None or abs(bottom) < 1e-12:
        return None
    return top / bottom


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


def _ticker_theme(ticker: str) -> str:
    for theme, tickers in THEME_MAP.items():
        if ticker in tickers:
            return theme
    return "Other SMID growth"


def _get_universe(tickers: Optional[str], max_tickers: int) -> Tuple[str, List[str]]:
    custom = _parse_custom_tickers(tickers)
    if custom:
        return "custom", custom[:max_tickers]
    return "high_growth_smid", _dedupe(SMID_GROWTH_UNIVERSE)[:max_tickers]


def _download_history(tickers: Sequence[str], period: str = "2y") -> pd.DataFrame:
    ticker_list = _dedupe(tickers)
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


def _atr14(data: pd.DataFrame) -> pd.Series:
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(14, min_periods=14).mean()


def _technical_row(ticker: str, history: pd.DataFrame, benchmark_history: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if history.empty or len(history) < 130:
        return None

    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)

    latest_close = _finite(close.iloc[-1])
    if latest_close is None or latest_close < MIN_PRICE:
        return None

    dollar_volume_20 = _finite((close * volume).rolling(20, min_periods=20).mean().iloc[-1])
    if dollar_volume_20 is not None and dollar_volume_20 < MIN_DOLLAR_VOLUME_20:
        return None

    returns = close.pct_change()
    vol = _finite(returns.tail(60).std() * math.sqrt(252)) or 0.70

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

    ma_stack_alignment = bool(
        sma50_value is not None
        and sma150_value is not None
        and sma200_value is not None
        and latest_close > sma50_value > sma150_value > sma200_value
        and (sma150_slope or 0) > 0
        and (sma200_slope or 0) > 0
    )
    stage2_transition = bool(
        sma150_value is not None
        and sma200_value is not None
        and latest_close > sma150_value
        and latest_close > sma200_value
        and (sma150_slope or 0) > 0
    )

    rolling_vol_20 = volume.rolling(20, min_periods=20).mean()
    rolling_vol_60 = volume.rolling(60, min_periods=40).mean()
    volume_dryup = bool(
        _finite(rolling_vol_20.iloc[-5:].mean()) is not None
        and _finite(rolling_vol_60.iloc[-1]) is not None
        and rolling_vol_20.iloc[-5:].mean() < rolling_vol_60.iloc[-1] * 0.80
    )
    volume_expansion = bool(
        _finite(volume.iloc[-1]) is not None
        and _finite(rolling_vol_20.iloc[-1]) is not None
        and volume.iloc[-1] > rolling_vol_20.iloc[-1] * 1.5
    )

    rs_new_high = False
    if not benchmark_history.empty:
        aligned = pd.concat(
            [
                close.rename("stock"),
                benchmark_history["Close"].astype(float).rename("benchmark"),
            ],
            axis=1,
        ).dropna()
        if len(aligned) > 80:
            rs_line = aligned["stock"] / aligned["benchmark"]
            rs_new_high = bool(rs_line.iloc[-1] >= rs_line.tail(63).max() * 0.995)

    price_range_20 = _safe_divide(close.tail(20).max() - close.tail(20).min(), close.iloc[-1])
    realized_vol_20 = _finite(returns.tail(20).std())
    realized_vol_60 = _finite(returns.tail(60).std())
    contraction = None
    if realized_vol_20 is not None and realized_vol_60 is not None and realized_vol_60 > 0:
        contraction = 1 - realized_vol_20 / realized_vol_60

    base_quality_score = (
        _score_linear(contraction, -0.2, 0.45, missing=45) * 0.55
        + _score_linear(-(price_range_20 or 0.30), -0.35, -0.04, missing=45) * 0.20
        + (75 if volume_dryup else 45) * 0.25
    )

    technical_score = 45.0
    if latest_close > (_finite(ema21.iloc[-1]) or float("inf")):
        technical_score += 8
    if sma50_value is not None and latest_close > sma50_value:
        technical_score += 9
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
    if ret_1m is not None:
        momentum_score += max(-12, min(18, ret_1m * 80))
    if ret_3m is not None:
        momentum_score += max(-14, min(24, ret_3m * 60))
    if ret_6m is not None:
        momentum_score += max(-12, min(18, ret_6m * 36))
    if ret_12m is not None:
        momentum_score += max(-8, min(12, ret_12m * 20))
    if ret_1m is not None and ret_3m is not None and ret_6m is not None:
        if ret_1m > ret_3m / 3 and ret_3m > ret_6m / 2:
            momentum_score += 10
    if rs_new_high:
        momentum_score += 10
    if vol > 1.1:
        momentum_score -= 10
    elif vol > 0.8:
        momentum_score -= 5

    latest_atr = _finite(_atr14(history).iloc[-1])
    stop_level = latest_close - ATR_STOP_MULTIPLE * latest_atr if latest_atr is not None else None

    risk_state = "Trend supported"
    if sma50_value is not None and latest_close < sma50_value:
        risk_state = "Neutral"
    if _finite(ema21.iloc[-1]) is not None and latest_close < ema21.iloc[-1]:
        risk_state = "Watch"

    fast_score = _clamp(
        technical_score * 0.46
        + momentum_score * 0.34
        + base_quality_score * 0.20
    )

    return {
        "ticker": ticker,
        "close": latest_close,
        "theme": _ticker_theme(ticker),
        "technical_score": _clamp(technical_score),
        "momentum_score": _clamp(momentum_score),
        "base_quality_score": _clamp(base_quality_score),
        "fast_score": fast_score,
        "return_1m": ret_1m,
        "return_3m": ret_3m,
        "return_6m": ret_6m,
        "return_12m": ret_12m,
        "annualized_volatility": vol,
        "dollar_volume_20": dollar_volume_20,
        "stage2_transition": stage2_transition,
        "ma_stack_alignment": ma_stack_alignment,
        "volume_dryup": volume_dryup,
        "volume_expansion": volume_expansion,
        "rs_new_high": rs_new_high,
        "risk_state": risk_state,
        "stop_level": stop_level,
    }


def _financial_profile(ticker: str) -> Dict[str, Any]:
    output = {
        "ticker": ticker,
        "name": ticker,
        "sector": "Unknown",
        "industry": "",
        "market_cap": None,
        "analyst_count": None,
        "held_percent_institutions": None,
        "short_percent_float": None,
        "forward_pe": None,
        "peg_ratio": None,
        "price_to_sales": None,
        "revenue_growth": None,
        "revenue_acceleration": None,
        "eps_acceleration": None,
        "gross_margin": None,
        "gross_margin_delta": None,
        "operating_margin": None,
        "operating_margin_delta": None,
        "fcf_margin": None,
        "fcf_margin_delta": None,
        "fcf_inflection": False,
        "current_ratio": None,
        "share_dilution": None,
        "net_debt_to_operating_income": None,
        "fundamental_fetch_status": "fallback",
    }

    try:
        stock = yf.Ticker(ticker)

        try:
            info = stock.info or {}
        except Exception:
            info = {}

        output.update(
            {
                "name": info.get("shortName") or info.get("longName") or ticker,
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "",
                "market_cap": _finite(info.get("marketCap")),
                "analyst_count": _finite(info.get("numberOfAnalystOpinions")),
                "held_percent_institutions": _finite(info.get("heldPercentInstitutions")),
                "short_percent_float": _finite(info.get("shortPercentOfFloat")),
                "forward_pe": _finite(info.get("forwardPE")),
                "peg_ratio": _finite(info.get("pegRatio")),
                "price_to_sales": _finite(info.get("priceToSalesTrailing12Months")),
            }
        )

        try:
            q_fin = stock.quarterly_financials
        except Exception:
            q_fin = pd.DataFrame()

        try:
            q_bal = stock.quarterly_balance_sheet
        except Exception:
            q_bal = pd.DataFrame()

        try:
            q_cf = stock.quarterly_cashflow
        except Exception:
            q_cf = pd.DataFrame()

        def series_get(statement: pd.DataFrame, names: Sequence[str]) -> List[Optional[float]]:
            if statement is None or statement.empty:
                return []
            lookup = {str(label).lower().strip(): label for label in statement.index}
            for name in names:
                key = str(name).lower().strip()
                if key in lookup:
                    return [_finite(value) for value in statement.loc[lookup[key]].tolist()]
            return []

        def latest(values: Sequence[Optional[float]]) -> Optional[float]:
            for value in values:
                if value is not None:
                    return value
            return None

        def prior(values: Sequence[Optional[float]]) -> Optional[float]:
            found = False
            for value in values:
                if value is None:
                    continue
                if not found:
                    found = True
                    continue
                return value
            return None

        def growth(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None or abs(b) < 1e-12:
                return None
            return (a - b) / abs(b)

        def slope(values: Sequence[Optional[float]]) -> Optional[float]:
            clean = [v for v in values if v is not None]
            if len(clean) < 3:
                return None
            y = list(reversed(clean[:3]))
            return _finite(np.polyfit(list(range(len(y))), y, 1)[0])

        revenue = series_get(q_fin, ["Total Revenue", "Operating Revenue"])
        gross_profit = series_get(q_fin, ["Gross Profit"])
        operating_income = series_get(q_fin, ["Operating Income", "EBIT"])
        eps = series_get(q_fin, ["Diluted EPS", "Basic EPS"])
        ocf = series_get(q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = series_get(q_cf, ["Capital Expenditure", "Capital Expenditures"])
        cash = series_get(q_bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
        debt = series_get(q_bal, ["Total Debt", "Long Term Debt"])
        current_assets = series_get(q_bal, ["Current Assets", "Total Current Assets"])
        current_liabilities = series_get(q_bal, ["Current Liabilities", "Total Current Liabilities"])
        shares = series_get(q_bal, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"])

        latest_revenue = latest(revenue)
        prior_revenue = prior(revenue)
        revenue_growth = growth(latest_revenue, prior_revenue)

        revenue_growth_rates = []
        for index in range(0, max(0, len(revenue) - 1)):
            rate = growth(revenue[index], revenue[index + 1])
            if rate is not None:
                revenue_growth_rates.append(rate)

        eps_growth_rates = []
        for index in range(0, max(0, len(eps) - 1)):
            rate = growth(eps[index], eps[index + 1])
            if rate is not None:
                eps_growth_rates.append(rate)

        latest_gross_profit = latest(gross_profit)
        prior_gross_profit = prior(gross_profit)
        latest_operating_income = latest(operating_income)
        prior_operating_income = prior(operating_income)

        gross_margin = _safe_divide(latest_gross_profit, latest_revenue)
        prior_gross_margin = _safe_divide(prior_gross_profit, prior_revenue)
        operating_margin = _safe_divide(latest_operating_income, latest_revenue)
        prior_operating_margin = _safe_divide(prior_operating_income, prior_revenue)

        latest_ocf = latest(ocf)
        prior_ocf = prior(ocf)
        latest_capex = latest(capex)
        prior_capex = prior(capex)

        fcf = latest_ocf + (latest_capex or 0.0) if latest_ocf is not None else None
        prior_fcf = prior_ocf + (prior_capex or 0.0) if prior_ocf is not None else None

        fcf_margin = _safe_divide(fcf, latest_revenue)
        prior_fcf_margin = _safe_divide(prior_fcf, prior_revenue)
        fcf_margin_delta = None
        if fcf_margin is not None and prior_fcf_margin is not None:
            fcf_margin_delta = fcf_margin - prior_fcf_margin

        latest_cash = latest(cash)
        latest_debt = latest(debt)
        net_debt = None
        if latest_cash is not None or latest_debt is not None:
            net_debt = (latest_debt or 0.0) - (latest_cash or 0.0)

        current_ratio = _safe_divide(latest(current_assets), latest(current_liabilities))
        share_dilution = growth(latest(shares), prior(shares))

        net_debt_to_operating_income = None
        if net_debt is not None and latest_operating_income is not None and latest_operating_income > 0:
            net_debt_to_operating_income = net_debt / latest_operating_income

        output.update(
            {
                "revenue_growth": revenue_growth,
                "revenue_acceleration": slope(revenue_growth_rates),
                "eps_acceleration": slope(eps_growth_rates),
                "gross_margin": gross_margin,
                "gross_margin_delta": gross_margin - prior_gross_margin if gross_margin is not None and prior_gross_margin is not None else None,
                "operating_margin": operating_margin,
                "operating_margin_delta": operating_margin - prior_operating_margin if operating_margin is not None and prior_operating_margin is not None else None,
                "fcf_margin": fcf_margin,
                "fcf_margin_delta": fcf_margin_delta,
                "fcf_inflection": bool(fcf is not None and ((prior_fcf is not None and prior_fcf < 0 and fcf > 0) or (fcf_margin_delta is not None and fcf_margin_delta > 0.04))),
                "current_ratio": current_ratio,
                "share_dilution": share_dilution,
                "net_debt_to_operating_income": net_debt_to_operating_income,
                "fundamental_fetch_status": "complete",
            }
        )
    except Exception:
        pass

    return output


def _neutral_profile(ticker: str) -> Dict[str, Any]:
    return _financial_profile.__defaults__[0] if False else {
        "ticker": ticker,
        "name": ticker,
        "sector": "Unknown",
        "industry": "",
        "market_cap": None,
        "analyst_count": None,
        "held_percent_institutions": None,
        "short_percent_float": None,
        "forward_pe": None,
        "peg_ratio": None,
        "price_to_sales": None,
        "revenue_growth": None,
        "revenue_acceleration": None,
        "eps_acceleration": None,
        "gross_margin": None,
        "gross_margin_delta": None,
        "operating_margin": None,
        "operating_margin_delta": None,
        "fcf_margin": None,
        "fcf_margin_delta": None,
        "fcf_inflection": False,
        "current_ratio": None,
        "share_dilution": None,
        "net_debt_to_operating_income": None,
        "fundamental_fetch_status": "timeout_fallback",
    }


def _fetch_profiles_bounded(tickers: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    profiles = {}
    ticker_list = list(tickers)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(_financial_profile, ticker): ticker
            for ticker in ticker_list
        }

        done, not_done = wait(
            future_to_ticker.keys(),
            timeout=FUNDAMENTAL_TIME_BUDGET_SECONDS,
        )

        for future in done:
            ticker = future_to_ticker[future]
            try:
                profiles[ticker] = future.result()
            except Exception:
                profiles[ticker] = _neutral_profile(ticker)

        for future in not_done:
            ticker = future_to_ticker[future]
            profiles[ticker] = _neutral_profile(ticker)

    return profiles


def _score_fundamentals(profile: Dict[str, Any]) -> float:
    return _clamp(
        _score_linear(profile.get("revenue_acceleration"), -0.05, 0.08, missing=50) * 0.26
        + _score_linear(profile.get("revenue_growth"), 0.05, 0.55, missing=50) * 0.18
        + _score_linear(profile.get("eps_acceleration"), -0.05, 0.10, missing=48) * 0.14
        + _score_linear(profile.get("gross_margin_delta"), -0.03, 0.08, missing=50) * 0.12
        + _score_linear(profile.get("operating_margin_delta"), -0.04, 0.10, missing=50) * 0.18
        + (82 if profile.get("fcf_inflection") else _score_linear(profile.get("fcf_margin_delta"), -0.06, 0.10, missing=48)) * 0.12
    )


def _score_balance(profile: Dict[str, Any]) -> float:
    current_ratio_score = _score_linear(profile.get("current_ratio"), 0.75, 2.5, missing=55)
    debt_value = profile.get("net_debt_to_operating_income")
    debt_score = 55 if debt_value is None else _score_linear(-debt_value, -4.5, 1.0, missing=55)
    fcf_score = _score_linear(profile.get("fcf_margin"), -0.15, 0.18, missing=50)
    dilution_score = _score_linear(-(profile.get("share_dilution") or 0.0), -0.12, 0.03, missing=55)
    return _clamp(current_ratio_score * 0.20 + debt_score * 0.28 + fcf_score * 0.30 + dilution_score * 0.22)


def _score_sponsorship(profile: Dict[str, Any]) -> float:
    analyst_count = profile.get("analyst_count")
    held = profile.get("held_percent_institutions")
    short_interest = profile.get("short_percent_float")

    if analyst_count is None:
        coverage = 55
    elif 2 <= analyst_count <= 6:
        coverage = 88
    elif 7 <= analyst_count <= 12:
        coverage = 72
    elif analyst_count < 2:
        coverage = 62
    else:
        coverage = 42

    if held is None:
        ownership = 55
    elif 0.20 <= held <= 0.70:
        ownership = 86
    elif 0.70 < held <= 0.88:
        ownership = 68
    elif held > 0.88:
        ownership = 40
    else:
        ownership = 55

    if short_interest is None:
        short_score = 55
    elif 0.08 <= short_interest <= 0.25:
        short_score = 82
    elif short_interest > 0.25:
        short_score = 58
    else:
        short_score = 52

    return _clamp(coverage * 0.35 + ownership * 0.40 + short_score * 0.25)


def _score_valuation(profile: Dict[str, Any]) -> float:
    peg = profile.get("peg_ratio")
    ps = profile.get("price_to_sales")
    growth = profile.get("revenue_growth")
    fpe = profile.get("forward_pe")

    if peg is None or peg <= 0:
        peg_score = 55
    elif peg <= 0.9:
        peg_score = 85
    elif peg <= 1.6:
        peg_score = 68
    elif peg <= 2.6:
        peg_score = 52
    else:
        peg_score = 34

    if ps is None or ps <= 0:
        ps_score = 52
    else:
        growth_pct = max(5, (growth or 0.15) * 100)
        ratio = ps / growth_pct
        if ratio <= 0.12:
            ps_score = 86
        elif ratio <= 0.22:
            ps_score = 72
        elif ratio <= 0.38:
            ps_score = 55
        else:
            ps_score = 34

    if fpe is None or fpe <= 0:
        pe_score = 52
    elif fpe <= 22:
        pe_score = 80
    elif fpe <= 40:
        pe_score = 63
    elif fpe <= 70:
        pe_score = 45
    else:
        pe_score = 28

    return _clamp(peg_score * 0.35 + ps_score * 0.45 + pe_score * 0.20)


def _score_catalyst(profile: Dict[str, Any], technical: Dict[str, Any]) -> float:
    score = 45
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


def _combine_row(technical: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    fundamental_score = _score_fundamentals(profile)
    balance_score = _score_balance(profile)
    sponsorship_score = _score_sponsorship(profile)
    valuation_score = _score_valuation(profile)
    catalyst_score = _score_catalyst(profile, technical)

    confluence = _clamp(
        fundamental_score * 0.26
        + technical["technical_score"] * 0.20
        + technical["momentum_score"] * 0.14
        + balance_score * 0.13
        + sponsorship_score * 0.09
        + valuation_score * 0.08
        + catalyst_score * 0.06
        + technical["base_quality_score"] * 0.04
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
    if not reasons:
        reasons.append("Monitorable SMID growth profile")

    market_cap = profile.get("market_cap")
    if market_cap is not None and (market_cap < MIN_MARKET_CAP or market_cap > MAX_MARKET_CAP):
        confluence -= 8

    if confluence >= 78:
        verdict = "Core SMID Growth"
    elif confluence >= 68:
        verdict = "High-Growth Watch"
    elif confluence >= 58:
        verdict = "Setup Developing"
    else:
        verdict = "Too Early"

    return {
        **profile,
        **technical,
        "ticker": technical["ticker"],
        "theme": technical.get("theme") or _ticker_theme(technical["ticker"]),
        "fundamental_score": _clamp(fundamental_score),
        "balance_sheet_score": _clamp(balance_score),
        "sponsorship_score": _clamp(sponsorship_score),
        "valuation_score": _clamp(valuation_score),
        "catalyst_score": _clamp(catalyst_score),
        "industry_strength_score": 50,
        "confluence_score": _clamp(confluence),
        "verdict": verdict,
        "reasons": reasons[:5],
    }


def _select_holdings(rows: List[Dict[str, Any]], target_holdings: int) -> Dict[str, Any]:
    candidates = [
        row for row in rows
        if (row.get("close") or 0) >= MIN_PRICE
        and ((row.get("dollar_volume_20") is None) or row.get("dollar_volume_20") >= MIN_DOLLAR_VOLUME_20)
        and (row.get("confluence_score") or 0) >= 52
    ]
    selected = candidates[:target_holdings]
    if not selected:
        selected = rows[:target_holdings]

    if not selected:
        return {"holdings": [], "stock_exposure": 0, "cash_weight": 1, "exposure_regime": "No SMID growth exposure"}

    avg_score = sum(row.get("confluence_score") or 0 for row in selected) / len(selected)
    avg_technical = sum(row.get("technical_score") or 0 for row in selected) / len(selected)

    if avg_score >= 76 and avg_technical >= 65:
        exposure, regime = 0.92, "Aggressive SMID growth risk"
    elif avg_score >= 68 and avg_technical >= 58:
        exposure, regime = 0.82, "Constructive SMID growth risk"
    elif avg_score >= 60:
        exposure, regime = 0.68, "Moderate SMID growth risk"
    else:
        exposure, regime = 0.50, "Reduced SMID growth risk"

    for row in selected:
        edge = max(0.01, (row.get("confluence_score") or 55) - 55)
        vol = max(0.22, row.get("annualized_volatility") or 0.70)
        row["raw_weight"] = edge / (vol ** 2)

    raw_total = sum(row["raw_weight"] for row in selected)
    for row in selected:
        row["target_weight"] = exposure * row["raw_weight"] / raw_total if raw_total > 0 else exposure / len(selected)

    for _ in range(8):
        excess = 0
        receivers = []
        for row in selected:
            if row["target_weight"] > MAX_SINGLE_POSITION:
                excess += row["target_weight"] - MAX_SINGLE_POSITION
                row["target_weight"] = MAX_SINGLE_POSITION
            else:
                receivers.append(row)
        if excess <= 1e-8 or not receivers:
            break
        receiver_total = sum(row["target_weight"] for row in receivers)
        if receiver_total <= 0:
            break
        for row in receivers:
            row["target_weight"] += excess * row["target_weight"] / receiver_total

    selected.sort(key=lambda row: row.get("target_weight") or 0, reverse=True)

    for index, row in enumerate(selected):
        row["portfolio_rank"] = index + 1
        score = row.get("confluence_score") or 0
        technical = row.get("technical_score") or 0
        if score >= 75 and technical >= 60:
            row["action"] = "Buy"
        elif score >= 64:
            row["action"] = "Hold"
        else:
            row["action"] = "Watch"
        row["trade_reason"] = "; ".join(row.get("reasons") or [])[:240]
        row["sell_trigger"] = "Review if confluence < 55, technical < 45, price loses the 50-day trend, or ATR stop is breached."
        row["hold_window"] = "4 to 12 weeks; review weekly."

    invested = sum(row.get("target_weight") or 0 for row in selected)
    return {
        "holdings": selected,
        "stock_exposure": exposure,
        "cash_weight": max(0, 1 - invested),
        "exposure_regime": regime,
    }


def _simulate(rows: List[Dict[str, Any]], price_frame: pd.DataFrame, target_holdings: int) -> Dict[str, Any]:
    benchmark = _slice_history(price_frame, BENCHMARK_TICKER)
    if benchmark.empty or len(benchmark) < 180:
        return {"series": [], "rebalance_log": [], "stats": {}, "diagnostics": {"reason": "No benchmark history"}}

    histories = {}
    for row in rows[:max(35, target_holdings * 3)]:
        history = _slice_history(price_frame, row["ticker"])
        if not history.empty:
            histories[row["ticker"]] = history

    trading_index = benchmark.index
    start_position = max(130, len(trading_index) - 252)
    benchmark_close = benchmark["Close"].astype(float)
    benchmark_start = _finite(benchmark_close.iloc[start_position]) or 1

    value = 1.0
    old_weights = {}
    series = []
    log = []
    model_returns = []
    benchmark_returns = []

    for position in range(start_position, len(trading_index)):
        date = trading_index[position]

        if position == start_position or (position - start_position) % 5 == 0:
            scores = []
            for row in rows[:max(35, target_holdings * 3)]:
                ticker = row["ticker"]
                history = histories.get(ticker)
                if history is None:
                    continue
                try:
                    local_position = history.index.get_indexer([date], method="pad")[0]
                except Exception:
                    continue
                if local_position < 130:
                    continue
                partial = history.iloc[: local_position + 1]
                tech = _technical_row(ticker, partial, benchmark.iloc[: position + 1])
                if tech:
                    score = tech["fast_score"] * 0.70 + (row.get("fundamental_score") or 55) * 0.30
                    vol = max(0.22, tech.get("annualized_volatility") or 0.70)
                    scores.append({"ticker": ticker, "score": score, "raw": max(0.01, score - 55) / (vol ** 2)})
            scores.sort(key=lambda item: item["score"], reverse=True)
            selected = scores[:target_holdings]
            raw_total = sum(item["raw"] for item in selected)
            new_weights = {item["ticker"]: 0.86 * item["raw"] / raw_total for item in selected} if raw_total > 0 else {}

            for key, weight in list(new_weights.items()):
                if weight > MAX_SINGLE_POSITION:
                    new_weights[key] = MAX_SINGLE_POSITION

            buys = sorted(list(set(new_weights) - set(old_weights)))
            sells = sorted(list(set(old_weights) - set(new_weights)))
            adds = []
            trims = []
            for key in sorted(list(set(new_weights) & set(old_weights))):
                if new_weights[key] - old_weights[key] > 0.015:
                    adds.append(key)
                elif old_weights[key] - new_weights[key] > 0.015:
                    trims.append(key)

            if new_weights:
                turnover = 0.5 * sum(abs(new_weights.get(k, 0) - old_weights.get(k, 0)) for k in set(new_weights) | set(old_weights))
                if old_weights:
                    value *= max(0, 1 - turnover * TRANSACTION_COST_BPS / 10000)
                headline = "No major changes"
                if buys:
                    headline = "Bought " + ", ".join(buys[:3])
                elif sells:
                    headline = "Removed " + ", ".join(sells[:3])
                elif adds:
                    headline = "Added to " + ", ".join(adds[:3])
                elif trims:
                    headline = "Trimmed " + ", ".join(trims[:3])
                log.append({
                    "date": pd.Timestamp(date).date().isoformat(),
                    "headline": headline,
                    "turnover": turnover,
                    "buys": buys,
                    "sells": sells,
                    "adds": adds,
                    "trims": trims,
                    "holdings": sorted([{"ticker": k, "weight": v} for k, v in new_weights.items()], key=lambda x: x["weight"], reverse=True)[:15],
                })
                old_weights = new_weights

        daily_return = 0
        if old_weights and position > 0:
            for ticker, weight in old_weights.items():
                history = histories.get(ticker)
                if history is None:
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
                if today is not None and yesterday is not None and yesterday != 0:
                    daily_return += weight * ((today / yesterday) - 1)

        value *= 1 + daily_return
        benchmark_value = (_finite(benchmark_close.iloc[position]) or benchmark_start) / benchmark_start
        benchmark_return = benchmark_close.pct_change().fillna(0).iloc[position]

        model_returns.append(daily_return)
        benchmark_returns.append(float(benchmark_return))
        marker = None
        if log and log[-1]["date"] == pd.Timestamp(date).date().isoformat():
            marker = {"date": log[-1]["date"], "headline": log[-1]["headline"]}
        series.append({"date": pd.Timestamp(date).date().isoformat(), "model": value - 1, "benchmark": benchmark_value - 1, "rebalance": marker})

    model_values = [point["model"] + 1 for point in series]
    bench_values = [point["benchmark"] + 1 for point in series]

    def max_dd(values: Sequence[float]) -> float:
        peak = values[0] if values else 1
        dd = 0
        for item in values:
            peak = max(peak, item)
            if peak > 0:
                dd = min(dd, (item - peak) / peak)
        return dd

    stats = {
        "model_return": series[-1]["model"] if series else 0,
        "benchmark_return": series[-1]["benchmark"] if series else 0,
        "model_volatility": float(pd.Series(model_returns).std() * math.sqrt(252)) if len(model_returns) > 5 else 0,
        "benchmark_volatility": float(pd.Series(benchmark_returns).std() * math.sqrt(252)) if len(benchmark_returns) > 5 else 0,
        "model_max_drawdown": max_dd(model_values),
        "benchmark_max_drawdown": max_dd(bench_values),
        "rebalance_count": len(log),
    }
    return {"series": series, "rebalance_log": log[-16:], "stats": stats, "diagnostics": {"history_count": len(histories)}}


def _build_payload(target_holdings: int, max_tickers: int, tickers: Optional[str], min_score: float) -> Dict[str, Any]:
    started_at = time.time()
    target_holdings = max(MIN_TARGET_HOLDINGS, min(MAX_TARGET_HOLDINGS, int(target_holdings)))
    universe_key, ticker_list = _get_universe(tickers, max_tickers)

    price_frame = _download_history(ticker_list + [BENCHMARK_TICKER], period="2y")
    benchmark_history = _slice_history(price_frame, BENCHMARK_TICKER)

    technical_rows = []
    for ticker in ticker_list:
        row = _technical_row(ticker, _slice_history(price_frame, ticker), benchmark_history)
        if row:
            technical_rows.append(row)

    technical_rows.sort(key=lambda row: row.get("fast_score") or 0, reverse=True)
    profile_targets = [row["ticker"] for row in technical_rows[:MAX_FUNDAMENTAL_FETCH]]
    profiles = _fetch_profiles_bounded(profile_targets)

    rows = []
    for row in technical_rows:
        profile = profiles.get(row["ticker"]) or _neutral_profile(row["ticker"])
        combined = _combine_row(row, profile)
        if (combined.get("confluence_score") or 0) >= min_score:
            rows.append(combined)

    rows.sort(key=lambda row: row.get("confluence_score") or 0, reverse=True)

    portfolio = _select_holdings(rows, target_holdings)
    holdings = portfolio["holdings"]

    sector_weights: Dict[str, float] = {}
    theme_weights: Dict[str, float] = {}
    for row in holdings:
        weight = row.get("target_weight") or 0
        sector = row.get("sector") or "Unknown"
        theme = row.get("theme") or "Other SMID growth"
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
        theme_weights[theme] = theme_weights.get(theme, 0) + weight

    performance = _simulate(rows, price_frame, target_holdings)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "portfolio_type": "high_growth_smid",
        "mode": "timeout_safe",
        "universe": universe_key,
        "requested_tickers": len(ticker_list),
        "technical_candidates": len(technical_rows),
        "fundamental_fetch_count": len(profile_targets),
        "ranked_candidates": len(rows),
        "target_holdings": target_holdings,
        "stock_exposure": portfolio["stock_exposure"],
        "cash_weight": portfolio["cash_weight"],
        "exposure_regime": portfolio["exposure_regime"],
        "holdings": [{k: _clean(v) for k, v in row.items() if k != "raw_weight"} for row in holdings],
        "top_candidates": [{k: _clean(v) for k, v in row.items() if k != "raw_weight"} for row in rows[:25]],
        "sector_weights": {k: _clean(v) for k, v in sorted(sector_weights.items(), key=lambda p: p[1], reverse=True)},
        "theme_weights": {k: _clean(v) for k, v in sorted(theme_weights.items(), key=lambda p: p[1], reverse=True)},
        "trade_queue": [
            {
                "ticker": row.get("ticker"),
                "action": row.get("action"),
                "target_weight": row.get("target_weight"),
                "confluence_score": row.get("confluence_score"),
                "reason": row.get("trade_reason"),
            }
            for row in holdings
        ],
        "performance": {
            "series": performance.get("series", []),
            "rebalance_log": performance.get("rebalance_log", []),
            "stats": {k: _clean(v) for k, v in performance.get("stats", {}).items()},
            "benchmark": BENCHMARK_TICKER,
        },
        "methodology": {
            "title": "High-Growth SMID timeout-safe confluence model",
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
            "note": "The live request pre-screens technically, then fetches fundamentals only for top candidates to avoid user-facing timeouts.",
        },
        "risk_rules": {
            "target_holdings": target_holdings,
            "max_single_position": MAX_SINGLE_POSITION,
            "max_sector_weight": MAX_SECTOR_WEIGHT,
            "max_theme_weight": MAX_THEME_WEIGHT,
            "atr_stop_multiple": ATR_STOP_MULTIPLE,
            "rebalance": "Weekly model rebalance; daily risk check.",
        },
        "diagnostics": {
            "runtime_seconds": round(time.time() - started_at, 2),
            "fundamental_time_budget_seconds": FUNDAMENTAL_TIME_BUDGET_SECONDS,
            "max_fundamental_fetch": MAX_FUNDAMENTAL_FETCH,
        },
    }
    return payload


@router.get("")
def get_smid_growth_portfolio(
    target_holdings: int = Query(default=DEFAULT_TARGET_HOLDINGS, ge=MIN_TARGET_HOLDINGS, le=MAX_TARGET_HOLDINGS),
    min_score: float = Query(default=50, ge=0, le=100),
    max_tickers: int = Query(default=DEFAULT_MAX_TICKERS, ge=20, le=100),
    refresh: bool = Query(default=False),
    tickers: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    cache_key = f"smid-growth-timeout-safe-v1:{target_holdings}:{min_score}:{max_tickers}:{tickers or ''}"

    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

    payload = _build_payload(target_holdings, max_tickers, tickers, min_score)
    payload["cached"] = False
    return _cache_set(cache_key, payload)


@router.get("/status")
def get_smid_growth_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "route": "/api/smid-growth-portfolio",
        "mode": "timeout_safe",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "default_max_tickers": DEFAULT_MAX_TICKERS,
        "max_fundamental_fetch": MAX_FUNDAMENTAL_FETCH,
        "fundamental_time_budget_seconds": FUNDAMENTAL_TIME_BUDGET_SECONDS,
        "benchmark": BENCHMARK_TICKER,
        "universe_size": len(_dedupe(SMID_GROWTH_UNIVERSE)),
        "expected_fresh_runtime": "25 to 55 seconds depending on Yahoo response time",
        "expected_cached_runtime": "1 to 5 seconds",
    }
