"""Plot summary CSVs produced by ``main.py``.

Usage:
    python plot_results.py results/summary_full.csv [results/summary_self.csv ...]
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib.pyplot as plt


def load(path: str) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def plot(paths: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for path in paths:
        cols = load(path)
        label = os.path.splitext(os.path.basename(path))[0].replace("summary_", "")
        axes[0].plot(cols["day"], cols["infected"], label=label, marker="o", markersize=3)
        axes[1].plot(cols["day"], cols["new_infections"], label=label, marker="o", markersize=3)

    axes[0].set_title("Active infections (I) over time")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("# infected")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Daily new infections")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("# new infections")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.dirname(paths[0]) or "."
    out = os.path.join(out_dir, "epidemic_curves.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    plot(sys.argv[1:])
