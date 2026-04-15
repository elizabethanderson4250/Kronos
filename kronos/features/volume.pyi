"""Type stubs for kronos/features/volume.py"""

import pandas as pd


def compute_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    Args:
        close: Closing price series.
        volume: Volume series.

    Returns:
        pd.Series: OBV values.
    """
    ...


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute Volume Weighted Average Price (VWAP).

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        volume: Volume series.

    Returns:
        pd.Series: VWAP values.

    Note:
        VWAP is calculated as cumulative(typical_price * volume) / cumulative(volume),
        where typical_price = (high + low + close) / 3.
    """
    ...


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Money Flow Index (MFI).

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        volume: Volume series.
        period: Lookback period (default 14).

    Returns:
        pd.Series: MFI values in range [0, 100].
    """
    ...


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compute Chaikin Money Flow (CMF).

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        volume: Volume series.
        period: Lookback period (default 20).

    Returns:
        pd.Series: CMF values in range [-1, 1].
    """
    ...


def compute_volume_indicators(
    df: pd.DataFrame,
    mfi_period: int = 14,
    cmf_period: int = 20,
) -> pd.DataFrame:
    """Compute all volume indicators and return as a DataFrame.

    Args:
        df: OHLCV DataFrame with columns: high, low, close, volume.
        mfi_period: Period for MFI calculation (default 14).
        cmf_period: Period for CMF calculation (default 20). Some sources
            recommend 21 for monthly data; 20 works well for daily data.

    Returns:
        pd.DataFrame: DataFrame with columns OBV, VWAP, MFI, CMF.
    """
    ...
