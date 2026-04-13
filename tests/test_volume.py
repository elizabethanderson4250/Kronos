"""Tests for kronos/features/volume.py"""

import pytest
import pandas as pd
import numpy as np
from kronos.features.volume import (
    compute_obv, compute_vwap, compute_mfi, compute_cmf, compute_volume_indicators
)


@pytest.fixture
def ohlcv():
    np.random.seed(42)
    n = 100
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    volume = pd.Series(np.random.randint(1000, 10000, n).astype(float), name="volume")
    df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    return df


def test_obv_length(ohlcv):
    obv = compute_obv(ohlcv["close"], ohlcv["volume"])
    assert len(obv) == len(ohlcv)


def test_obv_name(ohlcv):
    obv = compute_obv(ohlcv["close"], ohlcv["volume"])
    assert obv.name == "OBV"


def test_obv_no_nan(ohlcv):
    obv = compute_obv(ohlcv["close"], ohlcv["volume"])
    assert not obv.isna().any()


def test_vwap_length(ohlcv):
    vwap = compute_vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert len(vwap) == len(ohlcv)


def test_vwap_name(ohlcv):
    vwap = compute_vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert vwap.name == "VWAP"


def(ohlcv):
    vwap = compute_vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert (vwap > 0).all test_mfi_length(ohlcv):
    mfi = compute_mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert len(mfi) == len(ohlcv)


def test_mfi_range(ohlcv):
    mfi = compute_mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    valid = mfi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_mfi_name(ohlcv):
    mfi = compute_mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert mfi.name == "MFI"


def test_cmf_length(ohlcv):
    cmf = compute_cmf(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    assert len(cmf) == len(ohlcv)


def test_cmf_range(ohlcv):
    cmf = compute_cmf(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    valid = cmf.dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


def test_compute_volume_indicators_columns(ohlcv):
    result = compute_volume_indicators(ohlcv)
    assert set(result.columns) == {"OBV", "VWAP", "MFI", "CMF"}
    assert len(result) == len(ohlcv)
