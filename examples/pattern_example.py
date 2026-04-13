"""Example demonstrating candlestick pattern detection with Kronos."""

import os

import matplotlib.pyplot as plt
import pandas as pd

from kronos.data.loader import load_csv
from kronos.features.pattern import compute_pattern_indicators


def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "XSHG_5min_600977.csv")
    df = load_csv(data_path)

    # Use last 200 rows for clarity
    df = df.tail(200).reset_index(drop=True)

    result = compute_pattern_indicators(df)

    print("Pattern summary (last 200 bars):")
    for col in ["doji", "hammer", "bullish_engulfing", "bearish_engulfing"]:
        count = result[col].sum()
        print(f"  {col:25s}: {count} occurrences")

    # Plot close price with pattern markers
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, result["close"], label="Close", color="steelblue", linewidth=1)

    markers = [
        ("doji", "o", "orange", "Doji"),
        ("hammer", "^", "green", "Hammer"),
        ("bullish_engulfing", "D", "lime", "Bullish Engulfing"),
        ("bearish_engulfing", "v", "red", "Bearish Engulfing"),
    ]
    for col, marker, color, label in markers:
        idx = result.index[result[col]]
        ax.scatter(idx, result.loc[idx, "close"], marker=marker, color=color,
                   label=label, zorder=5, s=60)

    ax.set_title("Candlestick Pattern Detection — 600977 (5 min)")
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
