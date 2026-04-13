"""Preprocessing utilities for Kronos model inputs."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class MinMaxScaler:
    """Simple min-max scaler that remembers fit parameters."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_: Optional[float] = None
        self.max_: Optional[float] = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        self.min_ = float(np.min(data))
        self.max_ = float(np.max(data))
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        scale = self.max_ - self.min_
        if scale == 0:
            return np.zeros_like(data, dtype=float)
        lo, hi = self.feature_range
        return lo + (data - self.min_) / scale * (hi - lo)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        lo, hi = self.feature_range
        scale = self.max_ - self.min_
        return self.min_ + (data - lo) / (hi - lo) * scale

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


def build_sequences(
    series: np.ndarray,
    window: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window input/target pairs.

    Args:
        series: 1-D array of values.
        window: Number of past time steps used as input.
        horizon: Number of future steps to predict.

    Returns:
        X of shape (N, window) and y of shape (N, horizon).
    """
    if len(series) < window + horizon:
        raise ValueError("Series too short for the given window and horizon.")
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    return np.array(X), np.array(y)
