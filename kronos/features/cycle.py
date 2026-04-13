"""Cycle and oscillator indicators for Kronos."""

import numpy as np
import pandas as pd


def compute_dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator (DPO).

    Removes the trend from price to identify cycles.

    Args:
        close: Closing price series.
        period: Lookback period.

    Returns:
        DPO series.
    """
    shift = period // 2 + 1
    sma = close.rolling(window=period).mean()
    dpo = close.shift(shift) - sma
    dpo.name = f"DPO_{period}"
    return dpo


def compute_cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (CMO).

    Measures momentum on both up and down days.

    Args:
        close: Closing price series.
        period: Lookback period.

    Returns:
        CMO series ranging from -100 to 100.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    sum_up = up.rolling(window=period).sum()
    sum_down = down.rolling(window=period).sum()

    cmo = 100.0 * (sum_up - sum_down) / (sum_up + sum_down)
    cmo.name = f"CMO_{period}"
    return cmo


def compute_ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator.

    Combines three timeframes to reduce false divergence signals.

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        period1: Short period.
        period2: Medium period.
        period3: Long period.

    Returns:
        Ultimate Oscillator series (0-100).
    """
    prev_close = close.shift(1)
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
    true_high = pd.concat([high, prev_close], axis=1).max(axis=1)
    true_range = true_high - true_low

    buying_pressure = close - true_low

    def _avg(bp, tr, period):
        return bp.rolling(period).sum() / tr.rolling(period).sum()

    avg1 = _avg(buying_pressure, true_range, period1)
    avg2 = _avg(buying_pressure, true_range, period2)
    avg3 = _avg(buying_pressure, true_range, period3)

    uo = 100.0 * (4 * avg1 + 2 * avg2 + avg3) / 7.0
    uo.name = f"UO_{period1}_{period2}_{period3}"
    return uo


def compute_cycle_indicators(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """Compute all cycle indicators and return as a DataFrame."""
    return pd.DataFrame(
        {
            "DPO_20": compute_dpo(close, 20),
            "CMO_14": compute_cmo(close, 14),
            "UO_7_14_28": compute_ultimate_oscillator(high, low, close),
        }
    )
