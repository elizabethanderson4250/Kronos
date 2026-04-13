"""Type stubs for kronos.features.pattern."""

import pandas as pd


def compute_doji(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = ...,
) -> pd.Series:
    """Detect Doji candles.

    Returns a boolean Series named 'doji'.
    """
    ...


def compute_hammer(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    body_ratio: float = ...,
    shadow_ratio: float = ...,
) -> pd.Series:
    """Detect Hammer candles.

    Returns a boolean Series named 'hammer'.
    """
    ...


def compute_engulfing(
    open_: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Detect Bullish and Bearish Engulfing patterns.

    Returns a DataFrame with columns 'bullish_engulfing' and 'bearish_engulfing'.
    """
    ...


def compute_pattern_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all candlestick pattern indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close.

    Returns
    -------
    pd.DataFrame
        DataFrame with doji, hammer, bullish_engulfing, bearish_engulfing columns appended.
    """
    ...
