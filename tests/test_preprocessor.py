"""Tests for kronos.data.preprocessor."""

import pytest
import numpy as np
from kronos.data.preprocessor import MinMaxScaler, build_sequences


def test_minmax_fit_transform():
    data = np.array([0.0, 5.0, 10.0])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[-1] == pytest.approx(1.0)


def test_minmax_inverse_transform():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, data, atol=1e-8)


def test_minmax_custom_range():
    data = np.array([0.0, 10.0])
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaled = scaler.fit_transform(data)
    assert scaled[0] == pytest.approx(-1.0)
    assert scaled[1] == pytest.approx(1.0)


def test_minmax_not_fitted_raises():
    scaler = MinMaxScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([1.0, 2.0]))


def test_minmax_constant_series():
    data = np.array([5.0, 5.0, 5.0])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    np.testing.assert_array_equal(scaled, np.zeros(3))


def test_build_sequences_shape():
    series = np.arange(20, dtype=float)
    X, y = build_sequences(series, window=5, horizon=2)
    assert X.shape == (13, 5)
    assert y.shape == (13, 2)


def test_build_sequences_values():
    series = np.arange(10, dtype=float)
    X, y = build_sequences(series, window=3, horizon=1)
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    np.testing.assert_array_equal(y[0], [3])


def test_build_sequences_too_short():
    series = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        build_sequences(series, window=3, horizon=2)
