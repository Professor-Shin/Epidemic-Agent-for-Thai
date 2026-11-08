"""Tornado figure for section 5.8: how far each parameter moves the attack rate on condition C.

    python3 scripts/analysis/fig_sensitivity.py
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BASE, TUNED, CTRL = "#2a78d6", "#eb6834", "#7a7a76"
COLOR = {"llama8b": BASE, "qwen7b": BASE, "typhoon8b": TUNED,
         "openthaigpt7b": TUNED, "seallm7b": CTRL}
LABEL = {"home": "w_home_scale\n(การสัมผัสในครัวเรือน)", "f_errand": "f_errand\n(การสัมผัสตอนทำธุระ)",
         "f_visit": "f_visit\n(การสัมผัสตอนเยี่ยมญาติ)", "wfh": "wfh_scale\n(ความสามารถทำงานจากบ้าน)"}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def main():
    rows = []
    for d in sorted((ROOT / "outputs" / "runs").glob("C_*")):
        if not (d / "runtime.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        tag = c.get("tag") or ""
        if not tag.startswith("sens_") or c["cfg"]["sim"]["days"] != 180:
            continue
        v = pd.read_csv(d / "daily.csv")
        N = v[["S", "E", "I", "R"]].iloc[0].sum()
        rows.append(dict(model=c["model_key"], tag=tag, attack=1 - v.S.iloc[-1] / N))
    if not rows:
        print("ยังไม่มีผล sens บนเงื่อนไข C"); return
    df = pd.DataFrame(rows)
    df["axis"] = df.tag.str.extract(r"sens_(home|f_errand|f_visit|wfh)=")[0]
    g = df.groupby(["model", "axis"]).attack.agg(["min", "max"]).reset_index()
    g["span"] = g["max"] - g["min"]

    axes_order = [a for a in ["home", "f_errand", "f_visit", "wfh"] if a in set(g.axis)]
    models = [m for m in ["llama8b", "typhoon8b", "qwen7b", "openthaigpt7b", "seallm7b"]
              if m in set(g.model)]
    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(axes_order) * len(models) + 2.0))
    y, ticks, labels = 0, [], []
    for a in axes_order:
        block = g[g.axis == a]
        for m in models:
            r = block[block.model == m]
            if r.empty:
                continue
            lo, hi = float(r["min"].iloc[0]), float(r["max"].iloc[0])
            ax.plot([lo, hi], [y, y], color=COLOR[m], lw=7, solid_capstyle="butt", zorder=3)
            ax.text(hi + 0.004, y, f"{hi - lo:.3f}", va="center", fontsize=8.5, color=INK)
            ticks.append(y); labels.append(m); y -= 1
        ax.text(-0.012, y + 0.5 + len(models) / 2, LABEL[a], ha="right", va="center",
                fontsize=9.5, color=INK, transform=ax.get_yaxis_transform(which="grid"))
        y -= 0.6
    ax.set_yticks(ticks); ax.set_yticklabels(labels, color=INK2, fontsize=9)
    ax.set_xlabel("attack rate สะสม 180 วัน (ช่วงที่พารามิเตอร์กวาดได้)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [BASE, TUNED, CTRL]]
    fig.legend(h, ["โมเดลฐาน", "จูนไทย", "ตัวควบคุม SEA"], frameon=False, ncol=3,
               fontsize=9, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("พารามิเตอร์ตัวใดขยับผลการระบาดมากที่สุด เมื่อมี LLM อยู่ในลูป",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=[0.0, 0, 1, 0.92])
    out = ROOT / "outputs" / "figures" / "04_sensitivity_C.png"
    fig.savefig(out, dpi=200); print("wrote", out)


if __name__ == "__main__":
    main()
