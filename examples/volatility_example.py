"""Example demonstrating volatility indicators from Kronos."""

import matplotlib.pyplot as plt
import pandas as pd

from kronos.data.loader import load_csv
from kronos.features.volatility import (
    compute_atr,
    compute_historical_volatility,
    compute_keltner_channels,
    compute_chaikin_volatility,
)

DATA_PATH = "data/XSHG_5min_600977.csv"

# Increase tail to see more history in the plots
TAIL = 300


def main():
    df = load_csv(DATA_PATH)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = compute_atr(high, low, close, period=14)
    hv = compute_historical_volatility(close, period=20)
    # Using a slightly wider multiplier (2.5) for the Keltner Channels to reduce
    # false breakout signals on this 5-min data
    kc = compute_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.5)
    chaikin = compute_chaikin_volatility(high, low, ema_period=10, roc_period=10)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(close.iloc[-TAIL:].values, label="Close", color="black")
    axes[0].plot(kc["KC_UPPER"].iloc[-TAIL:].values, label="KC Upper", linestyle="--", color="red")
    axes[0].plot(kc["KC_MIDDLE"].iloc[-TAIL:].values, label="KC Middle", linestyle="-", color="blue")
    axes[0].plot(kc["KC_LOWER"].iloc[-TAIL:].values, label="KC Lower", linestyle="--", color="green")
    axes[0].set_title("Keltner Channels (multiplier=2.5)")
    axes[0].legend(fontsize=8)

    axes[1].plot(atr.iloc[-TAIL:].values, color="purple")
    axes[1].set_title("ATR (14)")

    axes[2].plot(hv.iloc[-TAIL:].values, color="darkorange")
    axes[2].set_title("Historical Volatility (20, annualized)")

    axes[3].plot(chaikin.iloc[-TAIL:].values, color="teal")
    axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_title("Chaikin Volatility (10, 10)")

    plt.tight_layout()
    plt.savefig("volatility_indicators.png", dpi=120)
    print("Plot saved to volatility_indicators.png")


if __name__ == "__main__":
    main()
