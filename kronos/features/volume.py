"""Volume-based technical indicators for Kronos."""

import pandas as pd
import numpy as np


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV)."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    obv.name = "OBV"
    return obv


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series) -> pd.Series:
    """Compute Volume Weighted Average Price (VWAP)."""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap.name = "VWAP"
    return vwap


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
               volume: pd.Series, period: int = 14) -> pd.Series:
    """Compute Money Flow Index (MFI)."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum()

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    mfi.name = "MFI"
    return mfi


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series,
               volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute Chaikin Money Flow (CMF)."""
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / hl_range
    money_flow_volume = money_flow_multiplier * volume

    cmf = (money_flow_volume.rolling(window=period).sum() /
           volume.rolling(window=period).sum())
    cmf.name = "CMF"
    return cmf


def compute_volume_indicators(df: pd.DataFrame, mfi_period: int = 14,
                              cmf_period: int = 20) -> pd.DataFrame:
    """Compute all volume indicators and return as a DataFrame."""
    result = pd.DataFrame(index=df.index)
    result["OBV"] = compute_obv(df["close"], df["volume"])
    result["VWAP"] = compute_vwap(df["high"], df["low"], df["close"], df["volume"])
    result["MFI"] = compute_mfi(df["high"], df["low"], df["close"],
                                df["volume"], period=mfi_period)
    result["CMF"] = compute_cmf(df["high"], df["low"], df["close"],
                                df["volume"], period=cmf_period)
    return result
