"""Unit tests for kronos.features.technical."""

import numpy as np
import pandas as pd
import pytest

from kronos.features.technical import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    add_technical_features,
)


@pytest.fixture
def price_series() -> pd.Series:
    np.random.seed(42)
    data = 100 + np.cumsum(np.random.randn(60))
    return pd.Series(data, name="close")


@pytest.fixture
def ohlcv_df(price_series) -> pd.DataFrame:
    n = len(price_series)
    return pd.DataFrame(
        {
            "open": price_series.values * 0.99,
            "high": price_series.values * 1.01,
            "low": price_series.values * 0.98,
            "close": price_series.values,
            "volume": np.random.randint(1000, 5000, n).astype(float),
        }
    )


def test_sma_length(price_series):
    result = compute_sma(price_series, window=10)
    assert len(result) == len(price_series)


def test_sma_values(price_series):
    result = compute_sma(price_series, window=5)
    expected = price_series.iloc[:5].mean()
    assert abs(result.iloc[4] - expected) < 1e-9


def test_ema_length(price_series):
    result = compute_ema(price_series, span=12)
    assert len(result) == len(price_series)


def test_rsi_range(price_series):
    rsi = compute_rsi(price_series, period=14)
    assert rsi.between(0, 100).all(), "RSI values must be in [0, 100]"


def test_rsi_constant_series():
    constant = pd.Series([50.0] * 30)
    rsi = compute_rsi(constant, period=14)
    # No gains or losses → falls back to fill value 50
    assert rsi.between(0, 100).all()


def test_macd_columns(price_series):
    result = compute_macd(price_series)
    assert set(result.columns) == {"macd", "signal", "histogram"}
    assert len(result) == len(price_series)


def test_macd_histogram_is_diff(price_series):
    result = compute_macd(price_series)
    diff = (result["macd"] - result["signal"]).round(10)
    assert (diff == result["histogram"].round(10)).all()


def test_bollinger_bands_columns(price_series):
    bb = compute_bollinger_bands(price_series)
    assert set(bb.columns) == {"upper", "middle", "lower"}


def test_bollinger_bands_ordering(price_series):
    bb = compute_bollinger_bands(price_series)
    assert (bb["upper"] >= bb["middle"]).all()
    assert (bb["middle"] >= bb["lower"]).all()


def test_add_technical_features_columns(ohlcv_df):
    result = add_technical_features(ohlcv_df)
    expected_cols = [
        "sma_10", "sma_20", "ema_12", "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "volume_sma_10",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_add_technical_features_no_volume(ohlcv_df):
    df_no_vol = ohlcv_df.drop(columns=["volume"])
    result = add_technical_features(df_no_vol, volume_col=None)
    assert "volume_sma_10" not in result.columns
    assert "rsi_14" in result.columns
