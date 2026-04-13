"""Trend indicator functions for Kronos."""

import pandas as pd
import numpy as np


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index (ADX) with +DI and -DI."""
    high = high.reset_index(drop=True)
    low = low.reset_index(drop=True)
    close = close.reset_index(drop=True)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()

    return pd.DataFrame({"ADX": adx, "+DI": plus_di, "-DI": minus_di})


def compute_parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Compute Parabolic SAR."""
    high = high.reset_index(drop=True)
    low = low.reset_index(drop=True)
    n = len(high)
    sar = np.full(n, np.nan)
    bull = True
    af = af_start
    ep = low.iloc[0]
    sar[0] = high.iloc[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[max(i - 2, 0)])
            if low.iloc[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[max(i - 2, 0)])
            if high.iloc[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, name="ParabolicSAR")


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Commodity Channel Index (CCI)."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - sma) / (0.015 * mean_dev)
    cci.name = "CCI"
    return cci
