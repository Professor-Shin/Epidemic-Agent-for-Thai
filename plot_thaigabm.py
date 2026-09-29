"""Figures for a ThaiGABM-SEIR results directory.

    python3 plot_thaigabm.py results_thaigabm

Reads every ``timeseries_*.csv`` in the directory and writes:

    fig1_epidemic_curves.png   infectious prevalence per arm
    fig2_reported_cases.png    what the *agents* see (reported incidence)
    fig3_behaviour.png         mobility / mask use / isolation per arm
    fig4_rt.png                effective reproduction number, with Rt = 1
    fig5_strata.png            per-stratum infections (last arm, or --arm)
    fig6_ablation.png          Experiment E effect sizes, if present

Colour identifies the *arm* and is consistent across every figure.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.thaigabm.strata import STRATUM_KEYS

# Validated categorical palette, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
NEUTRAL = "#8a8985"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def load(path: str) -> dict[str, list]:
    cols: dict[str, list] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    val: float | str = float(v)
                except (TypeError, ValueError):
                    val = v
                cols.setdefault(k, []).append(val)
    return cols


def load_dir(d: str) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    for path in sorted(glob.glob(os.path.join(d, "timeseries_*.csv"))):
        name = re.sub(r"^timeseries_|\.csv$", "", os.path.basename(path))
        out[name] = load(path)
    return out


def _style(ax, title: str, ylabel: str, xlabel: str = "Day") -> None:
    ax.set_title(title, loc="left", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.55)
    ax.set_axisbelow(True)


def _millions(y, _pos):
    if y >= 1e6:
        return f"{y / 1e6:.0f}M"
    if y >= 1e3:
        return f"{y / 1e3:.0f}k"
    return f"{y:.0f}"


def _plot_series(ax, runs: dict[str, dict[str, list]], key: str) -> None:
    """One line per arm, colour fixed by arm order and never cycled.

    Identity is carried by the legend (always present) plus comparison.csv as
    the table view, which is the relief the palette's sub-3:1 slots require.
    End-of-line labels are deliberately not used: with several arms converging
    near zero they collide into an unreadable stack.
    """
    for i, (name, cols) in enumerate(runs.items()):
        if key not in cols:
            continue
        ax.plot(cols["day"], cols[key], color=SERIES[i % len(SERIES)],
                linewidth=2.0, label=name, solid_capstyle="round")


def fig_epidemic(runs, out) -> None:
    """Linear and log panels of the same series.

    An unmitigated arm can peak three orders of magnitude above a behaviourally
    suppressed one, so a single linear panel flattens every interesting arm onto
    the axis. Both panels show the same measure on the same colour assignment --
    it is one chart read at two magnitudes, not a second y-scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax in axes:
        _plot_series(ax, runs, "infectious")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    _style(axes[0], "Infectious prevalence (linear)", "People infectious (I)")
    _style(axes[1], "Same series, log scale", "People infectious (I)")
    axes[1].set_yscale("log")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_epidemic_curves.png"), dpi=160)
    plt.close(fig)


def fig_reported(runs, out) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_series(ax, runs, "reported_per_100k")
    _style(ax, "Reported incidence — what the agents actually see",
           "Reported cases per 100,000/day")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_reported_cases.png"), dpi=160)
    plt.close(fig)


def fig_behaviour(runs, out) -> None:
    panels = [("mobility_index", "Mobility index"),
              ("mask", "Mask use"),
              ("isolating", "Fully isolating")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 9), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        _plot_series(ax, runs, key)
        _style(ax, title, "Share of population", xlabel="")
        ax.set_ylim(0, 1.02)
    axes[-1].set_xlabel("Day")
    axes[0].legend(loc="lower left", ncols=min(len(runs), 3))
    fig.suptitle("Behavioural response by arm", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "fig3_behaviour.png"), dpi=160)
    plt.close(fig)


def fig_rt(runs, out) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_series(ax, runs, "Rt")
    ax.axhline(1.0, color=NEUTRAL, linewidth=1.2, linestyle="--", zorder=0)
    ax.annotate("Rt = 1", (0.01, 1.0), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4),
                color=INK_2, fontsize=8)
    _style(ax, "Effective reproduction number under realised behaviour", "Rt")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_rt.png"), dpi=160)
    plt.close(fig)


def fig_strata(runs, out, arm: str | None = None) -> None:
    name = arm or next((n for n in runs if "thai" in n.lower() or n.startswith("D")),
                       list(runs)[-1])
    cols = runs.get(name)
    if not cols:
        return
    keys = [f"{k}__I" for k in STRATUM_KEYS if f"{k}__I" in cols]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(keys):
        label = key.replace("__I", "").replace("_", " ")
        ax.plot(cols["day"], cols[key], color=SERIES[i % len(SERIES)],
                linewidth=2.0, label=label, solid_capstyle="round")
    _style(ax, f"Infections by population stratum — {name}", "People infectious")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_strata.png"), dpi=160)
    plt.close(fig)


def fig_ablation(runs, out) -> None:
    """Effect of removing one behavioural construct, relative to the full model.

    Diverging around zero: blue = removing it *reduced* the epidemic,
    red = removing it made the epidemic worse.
    """
    full = next((n for n in runs if n.endswith("full_model")), None)
    ablated = {n: c for n, c in runs.items() if "ablate" in n}
    if not full or not ablated:
        return
    base = max(runs[full]["infectious"])
    rows = []
    for name, cols in ablated.items():
        peak = max(cols["infectious"])
        rows.append((name.replace("E_ablate_", "").replace("_", " "),
                     100.0 * (peak - base) / base if base else 0.0))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2.4))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = ["#2a78d6" if v < 0 else "#e34948" for v in values]
    bars = ax.barh(labels, values, color=colours, height=0.62)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:+.1f}%", (v, bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points",
                    xytext=(6 if v >= 0 else -6, 0),
                    ha="left" if v >= 0 else "right", va="center",
                    color=INK_2, fontsize=9)
    ax.axvline(0, color=NEUTRAL, linewidth=1.2)
    ax.set_title("Experiment E — effect of removing a behavioural construct",
                 loc="left", pad=10)
    ax.set_xlabel("Change in peak infectious prevalence vs full model (%)")
    ax.grid(True, axis="x", alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    pad = max(abs(min(values)), abs(max(values)), 1.0) * 0.28
    ax.set_xlim(min(values + [0]) - pad, max(values + [0]) + pad)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_ablation.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default="results_thaigabm")
    ap.add_argument("--arm", default=None, help="Arm to use for the strata figure.")
    args = ap.parse_args()

    runs = load_dir(args.results_dir)
    if not runs:
        raise SystemExit(f"No timeseries_*.csv found in {args.results_dir}")

    fig_epidemic(runs, args.results_dir)
    fig_reported(runs, args.results_dir)
    fig_behaviour(runs, args.results_dir)
    fig_rt(runs, args.results_dir)
    fig_strata(runs, args.results_dir, args.arm)
    fig_ablation(runs, args.results_dir)
    print(f"Wrote figures to {os.path.abspath(args.results_dir)}/ "
          f"({len(runs)} arm(s): {', '.join(runs)})")


if __name__ == "__main__":
    main()
