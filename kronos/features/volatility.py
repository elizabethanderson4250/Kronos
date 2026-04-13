"""Volatility indicators for Kronos."""

import pandas as pd
import numpy as np


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr.name = f"ATR_{period}"
    return atr


def compute_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute historical (realized) volatility as annualized std of log returns."""
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(252)
    hv.name = f"HV_{period}"
    return hv


def compute_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Compute Keltner Channels.

    Returns a DataFrame with columns: KC_UPPER, KC_MIDDLE, KC_LOWER.
    """
    middle = close.ewm(span=ema_period, adjust=False).mean()
    atr = compute_atr(high, low, close, period=atr_period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return pd.DataFrame({
        "KC_UPPER": upper,
        "KC_MIDDLE": middle,
        "KC_LOWER": lower,
    })


def compute_chaikin_volatility(
    high: pd.Series, low: pd.Series, ema_period: int = 10, roc_period: int = 10
) -> pd.Series:
    """Compute Chaikin Volatility indicator."""
    hl_diff = high - low
    ema_hl = hl_diff.ewm(span=ema_period, adjust=False).mean()
    chaikin_vol = ((ema_hl - ema_hl.shift(roc_period)) / ema_hl.shift(roc_period)) * 100
    chaikin_vol.name = f"CHAIKIN_VOL_{ema_period}_{roc_period}"
    return chaikin_vol
