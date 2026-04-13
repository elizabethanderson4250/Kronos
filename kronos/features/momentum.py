"""Momentum indicators for Kronos."""

import pandas as pd
import numpy as np


def compute_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (ROC) indicator.

    Args:
        prices: Series of closing prices.
        period: Look-back period.

    Returns:
        ROC values as a percentage.
    """
    roc = prices.pct_change(periods=period) * 100
    roc.name = f"ROC_{period}"
    return roc


def compute_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Raw momentum (price difference over period).

    Args:
        prices: Series of closing prices.
        period: Look-back period.

    Returns:
        Momentum values.
    """
    mom = prices.diff(period)
    mom.name = f"MOM_{period}"
    return mom


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D).

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        k_period: Period for %K calculation.
        d_period: Period for %D (signal line) smoothing.

    Returns:
        DataFrame with columns 'stoch_k' and 'stoch_d'.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    stoch_k = pd.Series(
        np.where(denom == 0, 0.0, (close - lowest_low) / denom * 100),
        index=close.index,
        name="stoch_k",
    )
    stoch_d = stoch_k.rolling(window=d_period).mean()
    stoch_d.name = "stoch_d"
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R indicator.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Look-back period.

    Returns:
        Williams %R values (range -100 to 0).
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = highest_high - lowest_low
    wr = pd.Series(
        np.where(denom == 0, -50.0, (highest_high - close) / denom * -100),
        index=close.index,
        name=f"WILLR_{period}",
    )
    return wr
