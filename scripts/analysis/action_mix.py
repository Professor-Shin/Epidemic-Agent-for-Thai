"""Figure 3 and the action-mix table of report/report_cumulative_TH.md.

What the agents actually chose, which no table in the report had shown for the full panel:
the mix of the four daily actions per model, and how the mix moves over the epidemic.

    python3 scripts/analysis/action_mix.py

Writes outputs/figures/06_action_mix.png and outputs/analysis/action_mix.csv.

Two shares exist and they are not the same number. daily.csv carries `share_<action>`, which
weights each agent by its persona post-stratification weight -- that is the share q_out is built
from, so it is the one the epidemic actually sees, and the one reported here. decisions.jsonl
gives the unweighted count proportion; the script prints both so the gap stays visible (it is at
most 0.013 across every run and action).
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DAYS = 180
# Ordered by how much out-of-home contact the action carries, so the stack reads low -> high
# exposure left to right and the sequential ramp is doing real work.
ACTS = ["stay_home", "essential_errand", "visit_family", "work_onsite"]
# Labels stay in English like every other figure in the report: matplotlib does no complex-script
# shaping, so Thai vowel and tone marks land in the wrong place. The Thai gloss is in the caption.
LABEL = {"stay_home": "stay home", "essential_errand": "essential errand",
         "visit_family": "visit family", "work_onsite": "work onsite"}
# One hue, light -> dark: a sequential ramp for an ordered quantity, not a categorical set.
RAMP = {"stay_home": "#d6e4f7", "essential_errand": "#8ab2e3",
        "visit_family": "#4a7fc1", "work_onsite": "#1f4677"}
ONDARK = {"stay_home": False, "essential_errand": False, "visit_family": True, "work_onsite": True}

ORDER = ["typhoon8b", "llama8b", "qwen7b", "openthaigpt7b", "seallm7b"]
# Same family colours as figure 1: blue = base model, orange = Thai-tuned, grey = SEA control.
LINE = {"llama8b": ("#2a78d6", "-"), "qwen7b": ("#2a78d6", "--"),
        "typhoon8b": ("#eb6834", "-"), "openthaigpt7b": ("#eb6834", "--"),
        "seallm7b": ("#52514e", ":")}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

def collect():
    rows, paths, peaks = [], {}, {}
    for d in sorted((ROOT / "outputs" / "runs").glob("*")):
        if not (d / "runtime.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if c["tag"] or c["cfg"]["sim"]["days"] != DAYS or c["condition"] not in ("C", "D"):
            continue
        v = pd.read_csv(d / "daily.csv")
        cnt = Counter(json.loads(l)["choice"] for l in open(d / "decisions.jsonl"))
        n = sum(cnt.values())
        r = {"cond": c["condition"], "model": c["model_key"], "seed": c["seed"], "n_decisions": n}
        for a in ACTS:
            r[a] = v[f"share_{a}"].mean()          # persona-weighted, the share q_out uses
            r["unweighted_" + a] = cnt[a] / n
        rows.append(r)
        paths.setdefault((c["condition"], c["model_key"]), []).append(v["share_stay_home"].values)
        peaks.setdefault((c["condition"], c["model_key"]), []).append(int(v.I.idxmax()))
    return pd.DataFrame(rows), paths, peaks


df, paths, peaks = collect()
(ROOT / "outputs" / "analysis").mkdir(parents=True, exist_ok=True)
df.sort_values(["cond", "model", "seed"]).to_csv(ROOT / "outputs" / "analysis" / "action_mix.csv", index=False)

gap = max(abs(df[a] - df["unweighted_" + a]).max() for a in ACTS)
print(f"{len(df)} runs, {int(df.n_decisions.sum()):,d} decisions")
print(f"max |weighted - unweighted| over every run and action: {gap:.4f}\n")
for cond in ("C", "D"):
    m = df[df.cond == cond].groupby("model")[ACTS].mean().reindex(ORDER)
    s = df[df.cond == cond].groupby("model")[ACTS].std().reindex(ORDER)
    print(f"--- condition {cond} (mean, sd over seeds) ---")
    for mod in ORDER:
        print(f"  {mod:14s} " + "  ".join(f"{a}={m.loc[mod, a]:.3f}({s.loc[mod, a]:.3f})" for a in ACTS))
    print()

# ---------------- the figure ----------------
fig = plt.figure(figsize=(9.8, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.08)

for j, cond in enumerate(("C", "D")):
    ax = fig.add_subplot(gs[0, j])
    m = df[df.cond == cond].groupby("model")[ACTS].mean().reindex(ORDER)
    y = np.arange(len(ORDER))[::-1]
    left = np.zeros(len(ORDER))
    for a in ACTS:
        w = m[a].values
        # 2px of surface between segments, taken off the right edge of each one
        ax.barh(y, np.maximum(w - 0.004, 0), left=left, height=0.62, color=RAMP[a], zorder=3)
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= 0.07:
                ax.text(l + wi / 2, y[yi], f"{wi:.2f}"[1:], ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if ONDARK[a] else INK, zorder=4)
        left = left + w
    ax.set_yticks(y, ORDER, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_title(f"condition {cond}", fontsize=10.5, color=INK, pad=8, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if j == 1:
        ax.tick_params(labelleft=False)

handles = [plt.Rectangle((0, 0), 1, 1, color=RAMP[a]) for a in ACTS]
labels = [LABEL[a] for a in ACTS]
labels[ACTS.index("visit_family")] += "  (never above .007)"
fig.legend(handles, labels, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=9, handlelength=1.1)

ax = fig.add_subplot(gs[1, :])
for mod in ORDER:
    if ("C", mod) not in paths:
        continue
    col, ls = LINE[mod]
    y = np.mean(np.vstack(paths[("C", mod)]), axis=0)
    ax.plot(np.arange(len(y)), y, color=col, ls=ls, lw=2, zorder=3)
    ax.text(len(y) + 3, y[-10:].mean(), mod, color=col, fontsize=9, va="center")
    # the day infections peak in this model -- the behaviour curve does not bend at it
    pk = int(round(np.mean(peaks[("C", mod)])))
    ax.plot([pk], [y[max(pk - 2, 0):pk + 3].mean()], marker="o", ms=8, mfc=SURFACE,
            mec=col, mew=2, zorder=4)
ax.set_xlim(0, 205)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175])
ax.set_ylim(0, 0.55)
ax.set_xlabel("day", fontsize=9.5)
ax.set_ylabel("share choosing stay home", fontsize=9.5)
ax.set_title("share choosing stay home over 180 days  —  condition C, mean of 6 seeds"
             "    (○ = day infections peak)",
             fontsize=10.5, color=INK, pad=8, loc="left")
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

out = ROOT / "outputs" / "figures" / "06_action_mix.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
