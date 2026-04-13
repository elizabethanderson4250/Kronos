"""Tests for kronos.features.pattern."""

import numpy as np
import pandas as pd
import pytest

from kronos.features.pattern import (
    compute_doji,
    compute_engulfing,
    compute_hammer,
    compute_pattern_indicators,
)


@pytest.fixture
def ohlc() -> pd.DataFrame:
    """Simple OHLC DataFrame for testing."""
    np.random.seed(42)
    n = 60
    close = 100 + np.cumsum(np.random.randn(n))
    open_ = close + np.random.uniform(-0.5, 0.5, n)
    high = np.maximum(open_, close) + np.abs(np.random.randn(n))
    low = np.minimum(open_, close) - np.abs(np.random.randn(n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_doji_length(ohlc):
    result = compute_doji(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    assert len(result) == len(ohlc)


def test_doji_name(ohlc):
    result = compute_doji(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    assert result.name == "doji"


def test_doji_boolean(ohlc):
    result = compute_doji(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    assert result.dtype == bool


def test_doji_detects_exact_doji():
    """A candle where open == close should be a doji."""
    df = pd.DataFrame({"open": [10.0], "high": [12.0], "low": [8.0], "close": [10.0]})
    result = compute_doji(df["open"], df["high"], df["low"], df["close"])
    assert result.iloc[0] is True or result.iloc[0] == True  # noqa: E712


def test_hammer_length(ohlc):
    result = compute_hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    assert len(result) == len(ohlc)


def test_hammer_name(ohlc):
    result = compute_hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    assert result.name == "hammer"


def test_hammer_boolean(ohlc):
    result = compute_hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    assert result.dtype == bool


def test_engulfing_returns_dataframe(ohlc):
    result = compute_engulfing(ohlc["open"], ohlc["close"])
    assert isinstance(result, pd.DataFrame)
    assert "bullish_engulfing" in result.columns
    assert "bearish_engulfing" in result.columns


def test_engulfing_length(ohlc):
    result = compute_engulfing(ohlc["open"], ohlc["close"])
    assert len(result) == len(ohlc)


def test_engulfing_boolean(ohlc):
    result = compute_engulfing(ohlc["open"], ohlc["close"])
    assert result["bullish_engulfing"].dtype == bool
    assert result["bearish_engulfing"].dtype == bool


def test_pattern_indicators_returns_dataframe(ohlc):
    result = compute_pattern_indicators(ohlc)
    assert isinstance(result, pd.DataFrame)


def test_pattern_indicators_has_all_columns(ohlc):
    result = compute_pattern_indicators(ohlc)
    for col in ["doji", "hammer", "bullish_engulfing", "bearish_engulfing"]:
        assert col in result.columns


def test_pattern_indicators_preserves_original_columns(ohlc):
    result = compute_pattern_indicators(ohlc)
    for col in ["open", "high", "low", "close"]:
        assert col in result.columns
