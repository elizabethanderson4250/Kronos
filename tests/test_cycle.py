"""Tests for kronos.features.cycle module."""

import numpy as np
import pandas as pd
import pytest

from kronos.features.cycle import (
    compute_cmo,
    compute_cycle_indicators,
    compute_dpo,
    compute_ultimate_oscillator,
)


@pytest.fixture
def ohlc():
    np.random.seed(42)
    n = 100
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    return high, low, close


# --- DPO ---

def test_dpo_length(ohlc):
    _, _, close = ohlc
    result = compute_dpo(close, period=20)
    assert len(result) == len(close)


def test_dpo_name(ohlc):
    _, _, close = ohlc
    result = compute_dpo(close, period=20)
    assert result.name == "DPO_20"


def test_dpo_has_nan_at_start(ohlc):
    _, _, close = ohlc
    result = compute_dpo(close, period=20)
    assert result.isna().any()


def test_dpo_custom_period(ohlc):
    _, _, close = ohlc
    result = compute_dpo(close, period=10)
    assert result.name == "DPO_10"
    assert len(result) == len(close)


# --- CMO ---

def test_cmo_length(ohlc):
    _, _, close = ohlc
    result = compute_cmo(close, period=14)
    assert len(result) == len(close)


def test_cmo_name(ohlc):
    _, _, close = ohlc
    result = compute_cmo(close)
    assert result.name == "CMO_14"


def test_cmo_range(ohlc):
    _, _, close = ohlc
    result = compute_cmo(close, period=14).dropna()
    assert (result >= -100).all() and (result <= 100).all()


def test_cmo_constant_raises_or_nan():
    """Constant price series should yield NaN (division by zero)."""
    close = pd.Series([50.0] * 50)
    result = compute_cmo(close, period=14)
    assert result.dropna().empty or result.dropna().eq(0).all()


# --- Ultimate Oscillator ---

def test_uo_length(ohlc):
    high, low, close = ohlc
    result = compute_ultimate_oscillator(high, low, close)
    assert len(result) == len(close)


def test_uo_name(ohlc):
    high, low, close = ohlc
    result = compute_ultimate_oscillator(high, low, close)
    assert result.name == "UO_7_14_28"


def test_uo_range(ohlc):
    high, low, close = ohlc
    result = compute_ultimate_oscillator(high, low, close).dropna()
    assert (result >= 0).all() and (result <= 100).all()


# --- compute_cycle_indicators ---

def test_cycle_indicators_returns_dataframe(ohlc):
    high, low, close = ohlc
    result = compute_cycle_indicators(high, low, close)
    assert isinstance(result, pd.DataFrame)


def test_cycle_indicators_columns(ohlc):
    high, low, close = ohlc
    result = compute_cycle_indicators(high, low, close)
    assert set(result.columns) == {"DPO_20", "CMO_14", "UO_7_14_28"}


def test_cycle_indicators_length(ohlc):
    high, low, close = ohlc
    result = compute_cycle_indicators(high, low, close)
    assert len(result) == len(close)
