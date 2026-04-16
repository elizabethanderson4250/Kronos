"""Tests for kronos.features.trend module."""

import pytest
import numpy as np
import pandas as pd
from kronos.features.trend import compute_adx, compute_parabolic_sar, compute_cci


@pytest.fixture
def ohlc():
    np.random.seed(42)
    n = 100
    close = pd.Series(100 + np.cumsum(np.random.randn(n)))
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    return high, low, close


# --- ADX Tests ---

def test_adx_returns_dataframe(ohlc):
    high, low, close = ohlc
    result = compute_adx(high, low, close)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"ADX", "+DI", "-DI"}


def test_adx_length(ohlc):
    high, low, close = ohlc
    result = compute_adx(high, low, close, period=14)
    assert len(result) == len(close)


def test_adx_values_in_range(ohlc):
    high, low, close = ohlc
    result = compute_adx(high, low, close)
    adx_valid = result["ADX"].dropna()
    assert (adx_valid >= 0).all()
    assert (adx_valid <= 100).all()


def test_adx_di_positive(ohlc):
    high, low, close = ohlc
    result = compute_adx(high, low, close)
    assert (result["+DI"].dropna() >= 0).all()
    assert (result["-DI"].dropna() >= 0).all()


# --- Parabolic SAR Tests ---

def test_sar_length(ohlc):
    high, low, close = ohlc
    result = compute_parabolic_sar(high, low)
    assert len(result) == len(high)


def test_sar_name(ohlc):
    high, low, close = ohlc
    result = compute_parabolic_sar(high, low)
    assert result.name == "ParabolicSAR"


def test_sar_is_series(ohlc):
    high, low, close = ohlc
    result = compute_parabolic_sar(high, low)
    assert isinstance(result, pd.Series)


def test_sar_no_all_nan(ohlc):
    high, low, close = ohlc
    result = compute_parabolic_sar(high, low)
    assert result.notna().sum() > 0


# --- CCI Tests ---

def test_cci_length(ohlc):
    high, low, close = ohlc
    result = compute_cci(high, low, close)
    assert len(result) == len(close)


def test_cci_name(ohlc):
    high, low, close = ohlc
    result = compute_cci(high, low, close)
    assert result.name == "CCI"


def test_cci_nan_prefix(ohlc):
    high, low, close = ohlc
    period = 20
    result = compute_cci(high, low, close, period=period)
    assert result.iloc[:period - 1].isna().all()


def test_cci_is_series(ohlc):
    high, low, close = ohlc
    result = compute_cci(high, low, close)
    assert isinstance(result, pd.Series)


# Note: CCI values beyond +/-100 indicate overbought/oversold conditions.
# This test checks that the non-NaN portion of the output actually varies
# (i.e. the function isn't returning a constant), which would be a sign of
# a broken implementation.
def test_cci_has_variance(ohlc):
    high, low, close = ohlc
    result = compute_cci(high, low, close)
    assert result.dropna().std() > 0


# Personal note: added a sanity check to ensure CCI non-NaN count matches
# expected number of valid rows given the default period of 20.
def test_cci_valid_count(ohlc):
    high, low, close = ohlc
    period = 20
    result = compute_cci(high, low, close, period=period)
    expected_valid = len(close) - (period - 1)
    assert result.notna().sum() == expected_valid
