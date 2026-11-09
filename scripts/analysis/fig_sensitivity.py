"""Tornado figure for section 5.8: how far each parameter can move the 180-day attack rate.

One bar per (axis, run family) showing max - min across the swept values. Condition B (rule-based, no LLM)
is drawn next to condition C on the three contact axes it was also swept on, because the comparison is the
point: the same parameter that moves B a long way barely moves C.

    python3 scripts/analysis/fig_sensitivity.py
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASE, TUNED, CTRL, NOLLM = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
COLOR = {"llama8b": BASE, "qwen7b": BASE, "typhoon8b": TUNED, "openthaigpt7b": TUNED, "seallm7b": CTRL}
AXIS_LABEL = {"home": "w_home_scale  (น้ำหนักการสัมผัสในครัวเรือน)",
              "f_errand": "f_errand  (การสัมผัสขณะออกไปทำธุระ)",
              "f_visit": "f_visit  (การสัมผัสขณะเยี่ยมญาติ)",
              "wfh": "wfh_scale  (สัดส่วนผู้ที่ทำงานจากบ้านได้)"}
AXIS_ORDER = ["home", "f_errand", "f_visit", "wfh"]
plt.rcParams.update({"font.family": ["Noto Sans Thai", "DejaVu Sans"], "font.size": 9.5,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def attack(run_dir):
    d = pd.read_csv(run_dir / "daily.csv")
    return 1 - d.S.iloc[-1] / d[["S", "E", "I", "R"]].iloc[-1].sum()


def collect():
    rows = []
    for d in sorted((ROOT / "outputs" / "runs").glob("*")):
        if not (d / "runtime.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if c["cfg"]["sim"]["days"] != 180 or not c["tag"].startswith("sens_"):
            continue
        axis, value = c["tag"][5:].split("=")
        rows.append(dict(axis=axis, value=float(value), cond=c["condition"],
                         who=c["model_key"] or "ไม่มี LLM (เงื่อนไข B)", attack=attack(d)))
    df = pd.DataFrame(rows)
    g = df.groupby(["axis", "cond", "who"]).attack.agg(["min", "max", "count"]).reset_index()
    g["swing"] = g["max"] - g["min"]
    return g


def main():
    g = collect()
    bars, y, ticks, labels, heads = [], 0.0, [], [], []
    for axis in AXIS_ORDER:
        sub = g[g.axis == axis]
        if sub.empty:
            continue
        heads.append((y + 0.9, AXIS_LABEL[axis]))
        sub = sub.sort_values(["cond", "swing"], ascending=[True, False])
        for _, r in sub.iterrows():
            bars.append((y, r.swing, r.who, r.cond, r["min"], r["max"]))
            ticks.append(y); labels.append(r.who)
            y -= 1.0
        y -= 1.0

    fig, ax = plt.subplots(figsize=(9.4, 0.46 * len(bars) + 3.1))
    for yy, swing, who, cond, lo, hi in bars:
        col = NOLLM if cond == "B" else COLOR[who]
        ax.barh(yy, swing, height=0.52, color=col, zorder=3)
        ax.text(swing + 0.0025, yy, f"{swing:.3f}", va="center", fontsize=8.5, color=INK, zorder=4)
        ax.text(swing + 0.030, yy, f"({lo:.3f} → {hi:.3f})", va="center", fontsize=8, color=MUTED, zorder=4)
    for yy, text in heads:
        ax.text(-0.001, yy, text, ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")

    ax.set_yticks(ticks); ax.set_yticklabels(labels, color=INK2, fontsize=9)
    ax.set_ylim(min(t for t, *_ in bars) - 0.9, max(h for h, _ in heads) + 0.7)
    ax.set_xlim(0, 0.215)
    ax.set_xlabel("ช่วงที่ attack rate สะสม 180 วัน ขยับได้ (ค่าสูงสุด − ค่าต่ำสุดของแกนนั้น)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BASE, TUNED, CTRL, NOLLM)]
    fig.legend(handles, ["โมเดลฐาน (C)", "จูนไทย (C)", "ตัวควบคุม SEA (C)", "ไม่มี LLM (B)"],
               frameon=False, ncol=4, fontsize=9, labelcolor=INK2, loc="upper center",
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("พารามิเตอร์การสัมผัสขยับผลลัพธ์ของเงื่อนไข B ได้มาก แต่ขยับเงื่อนไข C ได้น้อย",
                 fontsize=12.5, color=INK, y=0.995)
    fig.text(0.5, 0.012, "แกนสามตัวบนรันบน llama8b เท่านั้นเพราะงบ GPU จำกัด · wfh_scale รันครบทุกโมเดล · "
             "ทุกรันใช้ seed 1 และ 180 วัน", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.028, 1.0, 0.925))
    out = ROOT / "outputs" / "figures" / "04_sensitivity_C.png"
    fig.savefig(out, dpi=200); print("wrote", out)


if __name__ == "__main__":
    main()
