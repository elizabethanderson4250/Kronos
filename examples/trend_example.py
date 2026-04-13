"""Example demonstrating trend indicators from Kronos."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from kronos.data.loader import load_csv
from kronos.features.trend import compute_adx, compute_parabolic_sar, compute_cci


def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "XSHG_5min_600977.csv")
    df = load_csv(data_path)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Compute trend indicators
    adx_df = compute_adx(high, low, close, period=14)
    sar = compute_parabolic_sar(high, low)
    cci = compute_cci(high, low, close, period=20)

    tail = 200
    idx = range(tail)
    close_tail = close.iloc[-tail:].reset_index(drop=True)
    sar_tail = sar.iloc[-tail:].reset_index(drop=True)
    adx_tail = adx_df.iloc[-tail:].reset_index(drop=True)
    cci_tail = cci.iloc[-tail:].reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price + Parabolic SAR
    axes[0].plot(idx, close_tail, label="Close", color="steelblue")
    axes[0].scatter(idx, sar_tail, label="Parabolic SAR", color="orange", s=5)
    axes[0].set_title("Price & Parabolic SAR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ADX / +DI / -DI
    axes[1].plot(idx, adx_tail["ADX"], label="ADX", color="purple")
    axes[1].plot(idx, adx_tail["+DI"], label="+DI", color="green", linestyle="--")
    axes[1].plot(idx, adx_tail["-DI"], label="-DI", color="red", linestyle="--")
    axes[1].axhline(25, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_title("ADX / +DI / -DI")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # CCI
    axes[2].plot(idx, cci_tail, label="CCI", color="darkcyan")
    axes[2].axhline(100, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(-100, color="green", linestyle="--", linewidth=0.8)
    axes[2].axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_title("CCI (20)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
