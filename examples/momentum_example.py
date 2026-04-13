"""Example demonstrating momentum indicators from Kronos."""

import pandas as pd
import matplotlib.pyplot as plt

from kronos.data.loader import load_csv
from kronos.features.momentum import (
    compute_roc,
    compute_momentum,
    compute_stochastic,
    compute_williams_r,
)

DATA_PATH = "examples/data/XSHG_5min_600977.csv"


def main():
    df = load_csv(DATA_PATH)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    roc = compute_roc(close, period=10)
    mom = compute_momentum(close, period=10)
    stoch = compute_stochastic(high, low, close, k_period=14, d_period=3)
    wr = compute_williams_r(high, low, close, period=14)

    tail = 200
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Momentum Indicators — 600977 (5min)", fontsize=14)

    axes[0].plot(close.iloc[-tail:].values, label="Close", color="steelblue")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    axes[1].plot(roc.iloc[-tail:].values, label="ROC(10)", color="darkorange")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("ROC %")
    axes[1].legend()

    axes[2].plot(stoch["stoch_k"].iloc[-tail:].values, label="%K", color="purple")
    axes[2].plot(stoch["stoch_d"].iloc[-tail:].values, label="%D", color="red", linestyle="--")
    axes[2].axhline(80, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axhline(20, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Stochastic")
    axes[2].legend()

    axes[3].plot(wr.iloc[-tail:].values, label="Williams %R(14)", color="teal")
    axes[3].axhline(-20, color="gray", linestyle=":", linewidth=0.8)
    axes[3].axhline(-80, color="gray", linestyle=":", linewidth=0.8)
    axes[3].set_ylabel("Williams %R")
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    print("Last 5 rows of momentum indicators:")
    summary = pd.DataFrame({
        "close": close,
        "ROC_10": roc,
        "MOM_10": mom,
        "stoch_k": stoch["stoch_k"],
        "stoch_d": stoch["stoch_d"],
        "WILLR_14": wr,
    })
    print(summary.tail())


if __name__ == "__main__":
    main()
