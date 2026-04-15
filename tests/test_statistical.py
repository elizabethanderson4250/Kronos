"""Tests for kronos/features/statistical.py"""

import numpy as np
import pandas as pd
import pytest

from kronos.features.statistical import (
    compute_zscore,
    compute_rolling_skewness,
    compute_rolling_kurtosis,
    compute_rolling_correlation,
    compute_statistical_indicators,
)


@pytest.fixture
def price_series():
    """Simple synthetic price series."""
    np.random.seed(42)
    return pd.Series(np.cumsum(np.random.randn(100)) + 100)


@pytest.fixture
def ohlcv():
    """Synthetic OHLCV DataFrame."""
    np.random.seed(0)
    n = 100
    close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
    volume = pd.Series(np.random.randint(1000, 5000, size=n).astype(float))
    return pd.DataFrame({"close": close, "volume": volume})


# --- compute_zscore ---

def test_zscore_length(price_series):
    result = compute_zscore(price_series)
    assert len(result) == len(price_series)


def test_zscore_name(price_series):
    result = compute_zscore(price_series, period=20)
    assert "zscore" in result.name.lower()


def test_zscore_has_nan_at_start(price_series):
    period = 20
    result = compute_zscore(price_series, period=period)
    assert result.iloc[:period - 1].isna().all()


def test_zscore_values_reasonable(price_series):
    result = compute_zscore(price_series, period=20).dropna()
    # Z-scores for typical price data should be within a reasonable range
    assert (result.abs() < 10).all()


# --- compute_rolling_skewness ---

def test_rolling_skewness_length(price_series):
    result = compute_rolling_skewness(price_series)
    assert len(result) == len(price_series)


def test_rolling_skewness_name(price_series):
    result = compute_rolling_skewness(price_series)
    assert "skew" in result.name.lower()


def test_rolling_skewness_has_nan_at_start(price_series):
    period = 20
    result = compute_rolling_skewness(price_series, period=period)
    assert result.iloc[:period - 1].isna().all()


def test_rolling_skewness_values_finite(price_series):
    result = compute_rolling_skewness(price_series, period=20).dropna()
    assert np.isfinite(result).all()


# --- compute_rolling_kurtosis ---

def test_rolling_kurtosis_length(price_series):
    result = compute_rolling_kurtosis(price_series)
    assert len(result) == len(price_series)


def test_rolling_kurtosis_name(price_series):
    result = compute_rolling_kurtosis(price_series)
    assert "kurt" in result.name.lower()


def test_rolling_kurtosis_has_nan_at_start(price_series):
    period = 20
    result = compute_rolling_kurtosis(price_series, period=period)
    assert result.iloc[:period - 1].isna().all()


def test_rolling_kurtosis_values_finite(price_series):
    result = compute_rolling_kurtosis(price_series, period=20).dropna()
    assert np.isfinite(result).all()


# --- compute_rolling_correlation ---

def test_rolling_correlation_length(ohlcv):
    result = compute_rolling_correlation(ohlcv["close"], ohlcv["volume"])
    assert len(result) == len(ohlcv)


def test_rolling_correlation_name(ohlcv):
    result = compute_rolling_correlation(ohlcv["close"], ohlcv["volume"])
    assert "corr" in result.name.lower()


def test_rolling_correlation_range(ohlcv):
    result = compute_rolling_correlation(ohlcv["close"], ohlcv["volume"], period=20).dropna()
    assert ((result >= -1.0) & (result <= 1.0)).all()


def test_rolling_correlation_has_nan_at_start(ohlcv):
    period = 20
    result = compute_rolling_correlation(ohlcv["close"], ohlcv["volume"], period=period)
    assert result.iloc[:period - 1].isna().all()


# --- compute_statistical_indicators ---

def test_statistical_indicators_returns_dataframe(ohlcv):
    result = compute_statistical_indicators(ohlcv["close"], ohlcv["volume"])
    assert isinstance(result, pd.DataFrame)


def test_statistical_indicators_length(ohlcv):
    result = compute_statistical_indicators(ohlcv["close"], ohlcv["volume"])
    assert len(result) == len(ohlcv)


def test_statistical_indicators_expected_columns(ohlcv):
    result = compute_statistical_indicators(ohlcv["close"], ohlcv["volume"])
    columns_lower = [c.lower() for c in result.columns]
    assert any("zscore" in c for c in columns_lower)
    assert any("skew" in c for c in columns_lower)
    assert any("kurt" in c for c in columns_lower)
    assert any("corr" in c for c in columns_lower)
