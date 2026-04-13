"""Tests for kronos.features.momentum module."""

import pytest
import numpy as np
import pandas as pd

from kronos.features.momentum import (
    compute_roc,
    compute_momentum,
    compute_stochastic,
    compute_williams_r,
)


@pytest.fixture
def ohlc():
    np.random.seed(42)
    n = 60
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    return high, low, close


# --- ROC ---

def test_roc_length(ohlc):
    _, _, close = ohlc
    roc = compute_roc(close, period=10)
    assert len(roc) == len(close)


def test_roc_name(ohlc):
    _, _, close = ohlc
    roc = compute_roc(close, period=5)
    assert roc.name == "ROC_5"


def test_roc_first_values_nan(ohlc):
    _, _, close = ohlc
    roc = compute_roc(close, period=10)
    assert roc.iloc[:10].isna().all()


# --- Momentum ---

def test_momentum_length(ohlc):
    _, _, close = ohlc
    mom = compute_momentum(close, period=10)
    assert len(mom) == len(close)


def test_momentum_name(ohlc):
    _, _, close = ohlc
    mom = compute_momentum(close, period=7)
    assert mom.name == "MOM_7"


def test_momentum_values(ohlc):
    _, _, close = ohlc
    period = 5
    mom = compute_momentum(close, period=period)
    idx = 20
    expected = close.iloc[idx] - close.iloc[idx - period]
    assert pytest.approx(mom.iloc[idx], rel=1e-6) == expected


# --- Stochastic ---

def test_stochastic_columns(ohlc):
    high, low, close = ohlc
    result = compute_stochastic(high, low, close)
    assert "stoch_k" in result.columns
    assert "stoch_d" in result.columns


def test_stochastic_k_range(ohlc):
    high, low, close = ohlc
    result = compute_stochastic(high, low, close)
    valid = result["stoch_k"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_stochastic_length(ohlc):
    high, low, close = ohlc
    result = compute_stochastic(high, low, close)
    assert len(result) == len(close)


# --- Williams %R ---

def test_williams_r_length(ohlc):
    high, low, close = ohlc
    wr = compute_williams_r(high, low, close, period=14)
    assert len(wr) == len(close)


def test_williams_r_range(ohlc):
    high, low, close = ohlc
    wr = compute_williams_r(high, low, close, period=14)
    valid = wr.dropna()
    assert (valid >= -100).all() and (valid <= 0).all()


def test_williams_r_name(ohlc):
    high, low, close = ohlc
    wr = compute_williams_r(high, low, close, period=14)
    assert wr.name == "WILLR_14"
