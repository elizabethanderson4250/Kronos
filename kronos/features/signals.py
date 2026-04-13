"""Trading signal generation based on technical indicators."""

import pandas as pd
import numpy as np
from kronos.features.technical import compute_sma, compute_ema, compute_rsi, compute_macd, compute_bollinger_bands


def signal_sma_crossover(prices: pd.Series, short_window: int = 10, long_window: int = 30) -> pd.Series:
    """Generate buy/sell signals from SMA crossover.

    Returns +1 (buy), -1 (sell), or 0 (hold).
    """
    short_sma = compute_sma(prices, short_window)
    long_sma = compute_sma(prices, long_window)
    signal = pd.Series(0, index=prices.index, dtype=int)
    signal[short_sma > long_sma] = 1
    signal[short_sma < long_sma] = -1
    return signal


def signal_rsi_threshold(prices: pd.Series, period: int = 14,
                         oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    """Generate buy/sell signals from RSI thresholds.

    Returns +1 (buy when oversold), -1 (sell when overbought), or 0 (hold).
    """
    rsi = compute_rsi(prices, period)
    signal = pd.Series(0, index=prices.index, dtype=int)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    return signal


def signal_macd_crossover(prices: pd.Series, fast: int = 12,
                          slow: int = 26, signal_period: int = 9) -> pd.Series:
    """Generate buy/sell signals from MACD line crossing the signal line.

    Returns +1 (buy), -1 (sell), or 0 (hold).
    """
    macd_line, signal_line, _ = compute_macd(prices, fast, slow, signal_period)
    signal = pd.Series(0, index=prices.index, dtype=int)
    signal[macd_line > signal_line] = 1
    signal[macd_line < signal_line] = -1
    return signal


def signal_bollinger_breakout(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Generate buy/sell signals from Bollinger Band breakouts.

    Returns +1 (buy when price below lower band), -1 (sell when above upper band), 0 (hold).
    """
    upper, middle, lower = compute_bollinger_bands(prices, period, num_std)
    signal = pd.Series(0, index=prices.index, dtype=int)
    signal[prices < lower] = 1
    signal[prices > upper] = -1
    return signal


def combine_signals(signals: list[pd.Series], weights: list[float] | None = None) -> pd.Series:
    """Combine multiple signal series into a single consensus signal.

    Uses weighted majority voting. Returns +1, -1, or 0.
    """
    if weights is None:
        weights = [1.0] * len(signals)
    if len(signals) != len(weights):
        raise ValueError("signals and weights must have the same length.")

    combined = pd.Series(0.0, index=signals[0].index)
    for sig, w in zip(signals, weights):
        combined += sig.reindex(combined.index).fillna(0) * w

    result = pd.Series(0, index=combined.index, dtype=int)
    result[combined > 0] = 1
    result[combined < 0] = -1
    return result
