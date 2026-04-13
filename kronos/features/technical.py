"""Technical indicator computation for Kronos."""

import numpy as np
import pandas as pd
from typing import Optional


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=series.index,
    )


def compute_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: upper, middle (SMA), and lower bands."""
    middle = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std().fillna(0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower},
        index=series.index,
    )


def add_technical_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: Optional[str] = "volume",
) -> pd.DataFrame:
    """Append common technical indicators to a OHLCV DataFrame."""
    out = df.copy()
    price = out[price_col]

    out["sma_10"] = compute_sma(price, 10)
    out["sma_20"] = compute_sma(price, 20)
    out["ema_12"] = compute_ema(price, 12)
    out["rsi_14"] = compute_rsi(price, 14)

    macd = compute_macd(price)
    out["macd"] = macd["macd"]
    out["macd_signal"] = macd["signal"]
    out["macd_hist"] = macd["histogram"]

    bb = compute_bollinger_bands(price)
    out["bb_upper"] = bb["upper"]
    out["bb_middle"] = bb["middle"]
    out["bb_lower"] = bb["lower"]

    if volume_col and volume_col in out.columns:
        out["volume_sma_10"] = compute_sma(out[volume_col].astype(float), 10)

    return out
