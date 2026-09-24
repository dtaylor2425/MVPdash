"""Native monthly transformations before alignment to mixed-frequency frames.

FRED observation dates are not release/vintage timestamps. These helpers avoid
backfilling future observations, but do not make historical data point-in-time.
"""

import pandas as pd

CPI_CACHE_SUFFIX = "_native_cpi_v1"


def align_macro_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry daily inputs forward while preserving native CPI observations.

    CPI must remain sparse: carrying its index forward before a monthly
    calculation destroys the distinction between an observation and a gap.
    """
    frame = frame.sort_index()
    aligned = frame.ffill()
    if "cpi" in frame.columns:
        aligned["cpi"] = frame["cpi"]
    return aligned


def monthly_year_over_year(series: pd.Series) -> pd.Series:
    """Percent change against the same calendar month one year earlier.

    Input must contain native monthly observations (NaNs on intervening days).
    Missing calendar months are not compressed into a twelve-row comparison.
    Results become available at their actual input timestamp and may then be
    carried forward onto the input grid. Invalid new comparisons stay missing
    instead of silently carrying an earlier valid annual change over them.
    """
    series = series.sort_index()
    observations = pd.to_numeric(series, errors="coerce").dropna()
    if observations.empty:
        return pd.Series(float("nan"), index=series.index, name=series.name)

    periods = observations.index.to_period("M")
    # Keep every current timestamp; no later value is moved to month-start.
    # If a prior-year month has multiple observations, use its final value.
    monthly = pd.Series(observations.to_numpy(), index=periods).groupby(level=0).last()
    prior = monthly.reindex(periods - 12).to_numpy()
    denominator = pd.Series(prior, index=observations.index).replace(0, float("nan"))
    annual_change = (observations / denominator - 1.0) * 100.0
    return annual_change.reindex(series.index, method="ffill").rename(series.name)
