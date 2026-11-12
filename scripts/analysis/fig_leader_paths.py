"""Figure 3 of report/final_report_TH.md: weekly policy level against the case count the leader saw.

    python3 scripts/analysis/fig_leader_paths.py
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
LINE = "#eb6834"
BAND = {"open": "#eef4fd", "limited": "#cde2fb", "strict": "#9ec5f4"}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

def series(run):
    d = ROOT / "outputs" / "runs" / run
    L = [json.loads(l) for l in (d / "leader.jsonl").open()]
    wk = np.arange(len(L))
    cases = [float(re.search(r"cases: ([\d.]+) per 100,000", x["prompt"]).group(1)) for x in L]
    return wk, np.array(cases), [x["choice"] for x in L]

PANELS = [("typhoon8b", "D_typhoon8b_s1_b823e074", "seed 1"),
          ("llama8b", "D_llama8b_s1_02c2a9d6", "seed 1"),
          ("qwen7b", "D_qwen7b_s1_e12d6e4d", "seed 1"),
          ("seallm7b", "D_seallm7b_s1_3ce1a6e3", "seed 1"),
          ("openthaigpt7b", "D_openthaigpt7b_s1_6a8f8b94", "seed 1"),
          ("llama8b", "D_llama8b_s2_054ddc7a", "seed 2 — escalated, 4 weeks")]

fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), sharex=True, sharey=True)
for ax, (model, run, note) in zip(axes.ravel(), PANELS):
    wk, cases, lvl = series(run)
    for i, c in enumerate(lvl):                       # policy as a background band, one per week
        ax.axvspan(i - 0.5, i + 0.5, color=BAND[c], lw=0, zorder=0)
    ax.plot(wk, cases, color=LINE, lw=2, zorder=3)
    for y in (528, 602):                               # the two case loads that ever triggered strict
        ax.axhline(y, color=INK2, ls="--", lw=1, zorder=2)
    ax.set_title(f"{model}  ·  {note}", fontsize=10, color=INK, pad=6, loc="left")
    ax.set_xlim(-0.5, len(wk) - 0.5); ax.set_ylim(0, 1050)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=1)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.text(0.5, 1010, f"peak {cases.max():.0f}", fontsize=9, color=INK, va="top")

for ax in axes[1]: ax.set_xlabel("week", color=INK2)
for ax in axes[:, 0]: ax.set_ylabel("reported cases\nper 100k per day", color=INK2)

h = [plt.Rectangle((0, 0), 1, 1, color=BAND[k]) for k in ["open", "limited", "strict"]]
h.append(plt.Line2D([0], [0], color=LINE, lw=2))
h.append(plt.Line2D([0], [0], color=INK2, ls="--", lw=1))
fig.legend(h, ["policy: open", "policy: limited", "policy: strict", "reported cases",
               "528 and 602 — the only case loads that ever triggered strict"],
           frameon=False, ncol=5, fontsize=9, labelcolor=INK2,
           loc="upper center", bbox_to_anchor=(0.5, 0.945))
fig.suptitle("The LLM leader reports the epidemic correctly and acts the same way regardless",
             fontsize=13, color=INK, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.savefig(str(ROOT / "outputs" / "figures") + "/05_leader_policy_paths.png", dpi=200)
print("wrote outputs/figures/05_leader_policy_paths.png")
