"""Data loading utilities for Kronos time series prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_csv(
    filepath: str,
    datetime_col: str = "datetime",
    close_col: str = "close",
    volume_col: Optional[str] = "volume",
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Args:
        filepath: Path to the CSV file.
        datetime_col: Name of the datetime column.
        close_col: Name of the close price column.
        volume_col: Name of the volume column, or None to skip.
        freq: Optional pandas frequency string for resampling.

    Returns:
        DataFrame with DatetimeIndex and normalized column names.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    required = [close_col]
    if volume_col:
        required.append(volume_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map = {close_col: "close"}
    if volume_col:
        rename_map[volume_col] = "volume"
    df = df.rename(columns=rename_map)

    if freq:
        df = df.resample(freq).last().dropna(subset=["close"])

    return df


def split_train_test(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test sets chronologically."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def compute_returns(series: pd.Series, log: bool = True) -> pd.Series:
    """Compute returns from a price series."""
    if log:
        return np.log(series / series.shift(1)).dropna()
    return series.pct_change().dropna()
