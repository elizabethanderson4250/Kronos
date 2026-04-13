"""Example demonstrating volume-based indicators in Kronos."""

import pandas as pd
import matplotlib.pyplot as plt
from kronos.data.loader import load_csv
from kronos.features.volume import (
    compute_obv, compute_vwap, compute_mfi, compute_cmf
)


def main():
    df = load_csv("data/XSHG_5min_600977.csv")
    df = df.tail(200).reset_index(drop=True)

    obv = compute_obv(df["close"], df["volume"])
    vwap = compute_vwap(df["high"], df["low"], df["close"], df["volume"])
    mfi = compute_mfi(df["high"], df["low"], df["close"], df["volume"], period=14)
    cmf = compute_cmf(df["high"], df["low"], df["close"], df["volume"], period=20)

    print("=== Volume Indicators (last 5 rows) ===")
    result = pd.DataFrame({
        "close": df["close"],
        "volume": df["volume"],
        "OBV": obv,
        "VWAP": vwap,
        "MFI": mfi,
        "CMF": cmf,
    })
    print(result.tail())

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(df["close"], label="Close", color="black")
    axes[0].plot(vwap, label="VWAP", color="blue", linestyle="--")
    axes[0].set_title("Price & VWAP")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(obv, label="OBV", color="purple")
    axes[1].set_title("On-Balance Volume (OBV)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(mfi, label="MFI", color="orange")
    axes[2].axhline(80, color="red", linestyle="--", alpha=0.6, label="Overbought (80)")
    axes[2].axhline(20, color="green", linestyle="--", alpha=0.6, label="Oversold (20)")
    axes[2].set_title("Money Flow Index (MFI)")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(cmf, label="CMF", color="teal")
    axes[3].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[3].set_title("Chaikin Money Flow (CMF)")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("volume_indicators.png", dpi=150)
    print("Plot saved to volume_indicators.png")
    plt.show()


if __name__ == "__main__":
    main()
