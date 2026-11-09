"""Figures 1 and 2 of report/final_report_TH.md: the 2x2 attack rates and reason diversity.

    python3 scripts/analysis/fig_2x2_and_reasons.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BASE, TUNED = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

# ---------- Figure 1: the 2x2 ----------
fam = ["Llama 3.1 8B", "Qwen 2.5 7B"]
data = {"C": {"base": [0.819, 0.830], "tuned": [0.640, 0.858],
              "bl": [("llama8b", "typhoon8b"), ("qwen7b", "openthaigpt7b")]},
        "D": {"base": [0.601, 0.666], "tuned": [0.268, 0.723],
              "bl": [("llama8b", "typhoon8b"), ("qwen7b", "openthaigpt7b")]}}
REF = {"C": ("A: no behaviour (0.795)", 0.795), "D": ("A: no behaviour (0.795)", 0.795)}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
x, w = np.arange(2), 0.34
for ax, cond in zip(axes, ["C", "D"]):
    d = data[cond]
    for off, key, c, lab in [(-w/2 - 0.012, "base", BASE, "base model"),
                             (+w/2 + 0.012, "tuned", TUNED, "Thai-tuned")]:
        bars = ax.bar(x + off, d[key], w, color=c, label=lab, zorder=3)
        for b, v, names in zip(bars, d[key], d["bl"]):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        for b, names in zip(bars, d["bl"]):
            ax.text(b.get_x() + b.get_width()/2, 0.02, names[0 if key == "base" else 1],
                    ha="center", va="bottom", fontsize=7.5, color="white", rotation=90)
    ax.axhline(REF[cond][1], color=INK2, ls="--", lw=1, zorder=2)
    # effect arrows
    for i in range(2):
        a, b = d["base"][i], d["tuned"][i]
        ax.annotate("", xy=(x[i] + w/2 + 0.012, b), xytext=(x[i] - w/2 - 0.012, a),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1,
                                    connectionstyle="arc3,rad=-0.25"), zorder=4)
        ax.text(x[i], 1.07, f"{b - a:+.3f}", ha="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(fam, color=INK)
    ax.set_title(f"Condition {cond}" + (" — LLM residents" if cond == "C" else " — residents + LLM leader"),
                 fontsize=11, color=INK, pad=10)
    ax.set_ylim(0, 1.16); ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axes[0].set_ylabel("180-day cumulative attack rate", color=INK2)
h1 = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [BASE, TUNED]]
h1.append(plt.Line2D([0], [0], color=INK2, ls="--", lw=1))
fig.legend(h1, ["base model", "Thai-tuned", "condition A: no behavioural response (0.795)"],
           frameon=False, ncol=3, fontsize=9.5, labelcolor=INK2,
           loc="upper center", bbox_to_anchor=(0.5, 0.935))
fig.suptitle("Thai-tuning changes sign between model families", fontsize=13, color=INK, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(str(ROOT / "outputs" / "figures") + "/04_2x2_localisation.png", dpi=200)
print("wrote outputs/figures/04_2x2_localisation.png")

# ---------- Figure 2: reason diversity ----------
models = ["openthaigpt7b", "qwen7b", "seallm7b", "typhoon8b", "llama8b"]
uniq = [17.1, 19.0, 64.8, 65.2, 76.5]
income = [65, 56, 59, 19, 54]
famcol = [TUNED, BASE, "#7a7a76", TUNED, BASE]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
y = np.arange(5)
for ax, vals, title, note in [
        (axes[0], uniq, "Distinct reasons produced", "% of 7,200 decisions with a reason not seen before"),
        (axes[1], income, 'Reasons framed around income', '% of reasons mentioning income / earnings')]:
    bars = ax.barh(y, vals, 0.6, color=famcol, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height()/2, f"{v:.1f}%" if v % 1 else f"{v:.0f}%",
                va="center", fontsize=9, color=INK)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel(note, fontsize=8.5, color=INK2)
    ax.set_xlim(0, 100); ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axes[0].set_yticks(y); axes[0].set_yticklabels(models, color=INK)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [BASE, TUNED, "#7a7a76"]]
fig.legend(h, ["base model", "Thai-tuned", "SEA-multilingual control"], frameon=False,
           ncol=3, fontsize=9.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, 0.925))
fig.suptitle("Qwen-family models reuse a small set of reasons", fontsize=13, color=INK, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(str(ROOT / "outputs" / "figures") + "/04_reason_diversity.png", dpi=200)
print("wrote outputs/figures/04_reason_diversity.png")
