"""Candlestick pattern recognition indicators."""

import pandas as pd
import numpy as np


def compute_doji(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                threshold: float = 0.1) -> pd.Series:
    """Detect Doji candles (open ≈ close).

    Returns a boolean Series where True indicates a Doji pattern.
    """
    body = (close - open_).abs()
    range_ = high - low
    ratio = body / range_.replace(0, np.nan)
    return (ratio < threshold).rename("doji")


def compute_hammer(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                  body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """Detect Hammer candles (small body, long lower shadow).

    Returns a boolean Series where True indicates a Hammer pattern.
    """
    body = (close - open_).abs()
    range_ = high - low
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)

    small_body = body <= body_ratio * range_
    long_lower = lower_shadow >= shadow_ratio * body.replace(0, np.nan).fillna(0)
    small_upper = upper_shadow <= body

    return (small_body & long_lower & small_upper).rename("hammer")


def compute_engulfing(open_: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Detect Bullish and Bearish Engulfing patterns.

    Returns a DataFrame with columns 'bullish_engulfing' and 'bearish_engulfing'.
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    bullish = (prev_close < prev_open) & (close > open_) & \
              (open_ < prev_close) & (close > prev_open)
    bearish = (prev_close > prev_open) & (close < open_) & \
              (open_ > prev_close) & (close < prev_open)

    return pd.DataFrame({
        "bullish_engulfing": bullish,
        "bearish_engulfing": bearish,
    })


def compute_pattern_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all candlestick pattern indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close.

    Returns
    -------
    pd.DataFrame
        DataFrame with pattern indicator columns appended.
    """
    result = df.copy()
    result["doji"] = compute_doji(df["open"], df["high"], df["low"], df["close"])
    result["hammer"] = compute_hammer(df["open"], df["high"], df["low"], df["close"])
    engulfing = compute_engulfing(df["open"], df["close"])
    result["bullish_engulfing"] = engulfing["bullish_engulfing"]
    result["bearish_engulfing"] = engulfing["bearish_engulfing"]
    return result
