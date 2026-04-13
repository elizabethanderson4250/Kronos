"""Tests for kronos.features.signals module."""

import pytest
import numpy as np
import pandas as pd
from kronos.features.signals import (
    signal_sma_crossover,
    signal_rsi_threshold,
    signal_macd_crossover,
    signal_bollinger_breakout,
    combine_signals,
)


@pytest.fixture
def trending_up_prices():
    """Steadily increasing price series."""
    return pd.Series(np.linspace(100, 200, 100), dtype=float)


@pytest.fixture
def oscillating_prices():
    """Oscillating price series to trigger RSI extremes."""
    t = np.linspace(0, 4 * np.pi, 200)
    return pd.Series(100 + 30 * np.sin(t), dtype=float)


def test_sma_crossover_returns_valid_signals(trending_up_prices):
    sig = signal_sma_crossover(trending_up_prices, short_window=5, long_window=20)
    assert set(sig.unique()).issubset({-1, 0, 1})
    assert len(sig) == len(trending_up_prices)


def test_sma_crossover_uptrend_mostly_buy(trending_up_prices):
    sig = signal_sma_crossover(trending_up_prices, short_window=5, long_window=20)
    # After warmup, an uptrend should produce mostly buy signals
    assert (sig == 1).sum() > (sig == -1).sum()


def test_rsi_signal_returns_valid_signals(oscillating_prices):
    sig = signal_rsi_threshold(oscillating_prices, period=14)
    assert set(sig.unique()).issubset({-1, 0, 1})
    assert len(sig) == len(oscillating_prices)


def test_rsi_signal_detects_extremes(oscillating_prices):
    sig = signal_rsi_threshold(oscillating_prices, period=14, oversold=30, overbought=70)
    # Oscillating series should trigger both buy and sell
    assert (sig == 1).sum() > 0
    assert (sig == -1).sum() > 0


def test_macd_crossover_returns_valid_signals(trending_up_prices):
    sig = signal_macd_crossover(trending_up_prices)
    assert set(sig.unique()).issubset({-1, 0, 1})
    assert len(sig) == len(trending_up_prices)


def test_bollinger_breakout_returns_valid_signals(oscillating_prices):
    sig = signal_bollinger_breakout(oscillating_prices, period=20)
    assert set(sig.unique()).issubset({-1, 0, 1})
    assert len(sig) == len(oscillating_prices)


def test_combine_signals_equal_weights():
    idx = pd.RangeIndex(10)
    s1 = pd.Series([1, 1, -1, 0, 1, -1, 0, 1, -1, 0], index=idx)
    s2 = pd.Series([1, -1, -1, 0, 1, 1, 0, 1, -1, 0], index=idx)
    combined = combine_signals([s1, s2])
    assert set(combined.unique()).issubset({-1, 0, 1})
    # Both agree on index 0 -> buy
    assert combined.iloc[0] == 1
    # Both agree on index 2 -> sell
    assert combined.iloc[2] == -1


def test_combine_signals_mismatched_lengths_raises():
    s1 = pd.Series([1, 0, -1])
    with pytest.raises(ValueError, match="same length"):
        combine_signals([s1], weights=[1.0, 0.5])


def test_combine_signals_custom_weights():
    idx = pd.RangeIndex(5)
    s1 = pd.Series([1, 1, 1, 1, 1], index=idx)
    s2 = pd.Series([-1, -1, -1, -1, -1], index=idx)
    # s1 has much higher weight, should dominate
    combined = combine_signals([s1, s2], weights=[10.0, 1.0])
    assert (combined == 1).all()
