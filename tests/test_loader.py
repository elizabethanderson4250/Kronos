"""Tests for kronos.data.loader."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from kronos.data.loader import load_csv, split_train_test, compute_returns

SAMPLE_CSV = Path("examples/data/XSHG_5min_600977.csv")


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample data not available")
def test_load_csv_basic():
    df = load_csv(str(SAMPLE_CSV), datetime_col="datetime", close_col="close", volume_col="volume")
    assert "close" in df.columns
    assert "volume" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample data not available")
def test_load_csv_no_volume():
    df = load_csv(str(SAMPLE_CSV), datetime_col="datetime", close_col="close", volume_col=None)
    assert "close" in df.columns
    assert "volume" not in df.columns


def test_load_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent.csv")


def test_split_train_test():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"close": np.random.rand(100)}, index=idx)
    train, test = split_train_test(df, test_ratio=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train.index[-1] < test.index[0]


def test_split_train_test_invalid_ratio():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_train_test(df, test_ratio=1.5)


def test_compute_returns_log():
    prices = pd.Series([100.0, 110.0, 121.0])
    returns = compute_returns(prices, log=True)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(np.log(110 / 100))


def test_compute_returns_pct():
    prices = pd.Series([100.0, 110.0, 121.0])
    returns = compute_returns(prices, log=False)
    assert returns.iloc[0] == pytest.approx(0.1)
