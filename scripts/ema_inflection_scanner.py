import json
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "AMD", "AVGO", "PLTR", "COIN", "HOOD", "SOFI", "HIMS",
    "CRWD", "NET", "SNOW", "DDOG", "SMCI", "ARM",
    "SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE"
]


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True).fillna(0)


def calculate_for_ticker(ticker: str, df: pd.DataFrame):
    if df.empty or len(df) < 80:
        return None

    df = df.copy().dropna()

    close = df["Close"]
    volume = df["Volume"]

    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["ATR14"] = atr(df, 14)

    df["Velocity9"] = (df["EMA9"] - df["EMA9"].shift(3)) / df["ATR14"]
    df["Velocity21"] = (df["EMA21"] - df["EMA21"].shift(5)) / df["ATR14"]

    df["Acceleration9"] = df["Velocity9"] - df["Velocity9"].shift(3)
    df["Acceleration21"] = df["Velocity21"] - df["Velocity21"].shift(5)

    df["RVOL"] = volume / volume.rolling(20).mean()
    df["VolMomentum"] = volume.ewm(span=10, adjust=False).mean() / volume.ewm(span=30, adjust=False).mean()

    day_range = df["High"] - df["Low"]
    df["ClosePosition"] = np.where(day_range > 0, (df["Close"] - df["Low"]) / day_range, 0.5)

    df["DollarVolume20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["DistanceToEMA21"] = (df["Close"] - df["EMA21"]) / df["ATR14"]

    latest = df.iloc[-1]

    if pd.isna(latest["ATR14"]):
        return None

    liquid = latest["Close"] >= 10 and latest["DollarVolume20"] >= 20_000_000

    setup = (
        liquid
        and latest["Velocity21"] < 0
        and latest["Acceleration21"] > 0
        and latest["Velocity9"] > latest["Velocity21"]
        and latest["Acceleration9"] > 0
        and latest["RVOL"] > 1.2
        and latest["Close"] > latest["EMA9"]
        and latest["ClosePosition"] > 0.6
    )

    return {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "close": round(float(latest["Close"]), 2),
        "ema9": round(float(latest["EMA9"]), 2),
        "ema21": round(float(latest["EMA21"]), 2),
        "velocity9": round(float(latest["Velocity9"]), 4),
        "velocity21": round(float(latest["Velocity21"]), 4),
        "acceleration9": round(float(latest["Acceleration9"]), 4),
        "acceleration21": round(float(latest["Acceleration21"]), 4),
        "rvol": round(float(latest["RVOL"]), 2),
        "volMomentum": round(float(latest["VolMomentum"]), 2),
        "closePosition": round(float(latest["ClosePosition"]), 2),
        "distanceToEma21": round(float(latest["DistanceToEMA21"]), 2),
        "dollarVolume20": round(float(latest["DollarVolume20"]), 0),
        "setup": bool(setup),
    }


def scan(tickers):
    raw = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    results = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw
            else:
                df = raw[ticker]

            result = calculate_for_ticker(ticker, df)

            if result:
                results.append(result)

        except Exception:
            continue

    if not results:
        return []

    result_df = pd.DataFrame(results)

    result_df["a21Rank"] = percentile_rank(result_df["acceleration21"])
    result_df["a9Rank"] = percentile_rank(result_df["acceleration9"])
    result_df["rvolRank"] = percentile_rank(result_df["rvol"])
    result_df["volMomentumRank"] = percentile_rank(result_df["volMomentum"])
    result_df["closePositionRank"] = percentile_rank(result_df["closePosition"])
    result_df["distanceRank"] = percentile_rank(result_df["distanceToEma21"])

    result_df["score"] = (
        30 * result_df["a21Rank"]
        + 25 * result_df["a9Rank"]
        + 20 * result_df["rvolRank"]
        + 10 * result_df["volMomentumRank"]
        + 10 * result_df["closePositionRank"]
        + 5 * result_df["distanceRank"]
    )

    result_df["score"] = result_df["score"].round(1)

    result_df = result_df.sort_values(["setup", "score"], ascending=[False, False])

    keep_cols = [
        "ticker", "date", "score", "setup", "close",
        "velocity9", "velocity21", "acceleration9", "acceleration21",
        "rvol", "volMomentum", "closePosition", "distanceToEma21",
        "ema9", "ema21", "dollarVolume20"
    ]

    return result_df[keep_cols].head(75).to_dict(orient="records")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    payload = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "count": len(tickers),
        "results": scan(tickers),
    }

    print(json.dumps(payload))