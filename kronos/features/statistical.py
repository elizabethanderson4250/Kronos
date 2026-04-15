"""Statistical features for financial time series analysis.

Provides functions to compute statistical indicators such as z-score,
skewness, kurtosis, and rolling correlation/covariance measures.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling z-score of a price series.

    Args:
        series: Price or return series.
        period: Lookback window for mean and std calculation.

    Returns:
        Rolling z-score series.
    """
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=1)
    zscore = (series - mean) / std
    zscore.name = f"zscore_{period}"
    return zscore


def compute_rolling_skewness(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling skewness of a price series.

    Skewness measures the asymmetry of the return distribution.
    Positive skew indicates a longer right tail; negative skew a longer left tail.

    Args:
        series: Price or return series.
        period: Lookback window.

    Returns:
        Rolling skewness series.
    """
    skew = series.rolling(window=period).skew()
    skew.name = f"skewness_{period}"
    return skew


def compute_rolling_kurtosis(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling excess kurtosis of a price series.

    Excess kurtosis measures the heaviness of the tails relative to a normal
    distribution. Values > 0 indicate heavier tails (leptokurtic).

    Args:
        series: Price or return series.
        period: Lookback window.

    Returns:
        Rolling excess kurtosis series.
    """
    kurt = series.rolling(window=period).kurt()
    kurt.name = f"kurtosis_{period}"
    return kurt


def compute_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compute rolling Pearson correlation between two series.

    Args:
        series_a: First series (e.g., close prices).
        series_b: Second series (e.g., volume or another asset).
        period: Lookback window.

    Returns:
        Rolling correlation series in the range [-1, 1].
    """
    corr = series_a.rolling(window=period).corr(series_b)
    corr.name = f"corr_{period}"
    return corr


def compute_statistical_indicators(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: Optional[str] = "volume",
    period: int = 20,
) -> pd.DataFrame:
    """Compute a standard set of statistical indicators and append them to df.

    Computes z-score, rolling skewness, and rolling kurtosis on the close
    price. If a volume column is provided, also computes rolling correlation
    between close price and volume.

    Args:
        df: OHLCV DataFrame.
        close_col: Name of the close price column.
        volume_col: Name of the volume column. Set to None to skip correlation.
        period: Lookback window used for all indicators.

    Returns:
        Copy of df with additional statistical indicator columns appended.
    """
    result = df.copy()
    close = result[close_col]

    result[f"zscore_{period}"] = compute_zscore(close, period)
    result[f"skewness_{period}"] = compute_rolling_skewness(close, period)
    result[f"kurtosis_{period}"] = compute_rolling_kurtosis(close, period)

    if volume_col is not None and volume_col in result.columns:
        result[f"close_vol_corr_{period}"] = compute_rolling_correlation(
            close, result[volume_col], period
        )

    return result
