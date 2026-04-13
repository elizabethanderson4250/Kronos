"""Tests for kronos.features.volatility module."""

import numpy as np
import pandas as pd
import pytest

from kronos.features.volatility import (
    compute_atr,
    compute_historical_volatility,
    compute_keltner_channels,
    compute_chaikin_volatility,
)


@pytest.fixture
def ohlc():
    np.random.seed(42)
    n = 100
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    return high, low, close


def test_atr_length(ohlc):
    high, low, close = ohlc
    atr = compute_atr(high, low, close, period=14)
    assert len(atr) == len(close)


def test_atr_positive(ohlc):
    high, low, close = ohlc
    atr = compute_atr(high, low, close, period=14)
    assert (atr.dropna() > 0).all()


def test_atr_name(ohlc):
    high, low, close = ohlc
    atr = compute_atr(high, low, close, period=14)
    assert atr.name == "ATR_14"


def test_historical_volatility_length(ohlc):
    _, _, close = ohlc
    hv = compute_historical_volatility(close, period=20)
    assert len(hv) == len(close)


def test_historical_volatility_non_negative(ohlc):
    _, _, close = ohlc
    hv = compute_historical_volatility(close, period=20)
    assert (hv.dropna() >= 0).all()


def test_historical_volatility_name(ohlc):
    _, _, close = ohlc
    hv = compute_historical_volatility(close, period=20)
    assert hv.name == "HV_20"


def test_keltner_channels_columns(ohlc):
    high, low, close = ohlc
    kc = compute_keltner_channels(high, low, close)
    assert set(kc.columns) == {"KC_UPPER", "KC_MIDDLE", "KC_LOWER"}


def test_keltner_channels_ordering(ohlc):
    high, low, close = ohlc
    kc = compute_keltner_channels(high, low, close).dropna()
    assert (kc["KC_UPPER"] >= kc["KC_MIDDLE"]).all()
    assert (kc["KC_MIDDLE"] >= kc["KC_LOWER"]).all()


def test_chaikin_volatility_length(ohlc):
    high, low, close = ohlc
    cv = compute_chaikin_volatility(high, low)
    assert len(cv) == len(close)


def test_chaikin_volatility_name(ohlc):
    high, low, _ = ohlc
    cv = compute_chaikin_volatility(high, low, ema_period=10, roc_period=10)
    assert cv.name == "CHAIKIN_VOL_10_10"
