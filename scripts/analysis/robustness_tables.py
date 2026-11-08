"""Tables for sections 5.8-5.10 of the final report: parameter sensitivity on condition C,
the baseline_mix check, and the long-horizon run. Reads outputs/runs/ directly so it works
on whatever has finished; rows that are not there yet are simply absent.

    python3 scripts/analysis/robustness_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL = ["llama8b", "typhoon8b", "qwen7b", "openthaigpt7b", "seallm7b"]
BASE_C = {"llama8b": 0.818, "typhoon8b": 0.642, "qwen7b": 0.830,
          "openthaigpt7b": 0.858, "seallm7b": 0.846}      # tag "", seed 1, 180 d


def load(pattern="*"):
    rows = []
    for d in sorted((ROOT / "outputs" / "runs").glob(pattern)):
        if not (d / "runtime.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        v = pd.read_csv(d / "daily.csv")
        N = v[["S", "E", "I", "R"]].iloc[0].sum()
        rows.append(dict(cond=c["condition"], model=c["model_key"] or "-", seed=c["seed"],
                         tag=c.get("tag") or "", days=c["cfg"]["sim"]["days"],
                         attack=1 - v.S.iloc[-1] / N,
                         peak_I=v.I.max() / N, peak_day=int(v.I.idxmax()),
                         q_out=v.q_out.mean() if "q_out" in v else float("nan"),
                         S_end=v.S.iloc[-1] / N, I_end=v.I.iloc[-1] / N,
                         run=d.name))
    return pd.DataFrame(rows)


def fmt(df, **kw):
    return df.to_string(float_format=lambda x: f"{x:.3f}", **kw)


def main():
    df = load()
    pd.set_option("display.width", 240)

    # ---- 5.8 sensitivity on condition C ----
    s = df[(df.cond == "C") & df.tag.str.startswith("sens_") & (df.days == 180)].copy()
    print("=" * 78, "\n5.8  SENSITIVITY ON CONDITION C\n")
    if s.empty:
        print("  (ยังไม่มีผล)")
    else:
        s["axis"] = s.tag.str.extract(r"sens_(home|f_errand|f_visit|wfh)=")[0]
        s["value"] = s.tag.str.extract(r"=([\d.]+)$")[0].astype(float)
        for ax, g in s.groupby("axis"):
            p = g.pivot_table(index="model", columns="value", values="attack")
            p["ช่วงที่เปลี่ยน"] = p.max(axis=1) - p.min(axis=1)
            print(f"  แกน {ax}"); print(fmt(p)); print()
        print("  เทียบกับเงื่อนไข B (ไม่มี LLM) ที่รายงานในหัวข้อ 5.7:")
        b = df[(df.cond == "B") & df.tag.str.startswith("sens_") & (df.days == 180)].copy()
        if not b.empty:
            b["axis"] = b.tag.str.extract(r"sens_(home|f_errand|f_visit|wfh)=")[0]
            print(fmt(b.groupby("axis").attack.agg(["min", "max"]).assign(
                ช่วงที่เปลี่ยน=lambda x: x["max"] - x["min"])))

    # ---- 5.9 baseline_mix ----
    print("\n" + "=" * 78, "\n5.9  BASELINE_MIX CHECK\n")
    bl = df[(df.cond == "C") & df.tag.str.startswith("baseline=") & (df.days == 180)]
    if bl.empty:
        print("  (ยังไม่มีผล)")
    else:
        t = bl.set_index("model")[["attack", "q_out"]].rename(
            columns={"attack": "attack ใหม่", "q_out": "q_out ใหม่"})
        t["attack เดิม"] = pd.Series(BASE_C)
        t["ส่วนต่าง"] = t["attack ใหม่"] - t["attack เดิม"]
        t = t.reindex([m for m in PANEL if m in t.index])
        print(fmt(t[["attack เดิม", "attack ใหม่", "ส่วนต่าง", "q_out ใหม่"]]))
        print("\n  A ที่ 180 วัน = 0.795 — ดูว่า C ยังสูงกว่า A หรือไม่")
        print(f"  โมเดลที่ยังสูงกว่า A: {(t['attack ใหม่'] > 0.795).sum()} จาก {len(t)}")
        print(f"  โมเดลที่ q_out ยังเกิน 1: {(t['q_out ใหม่'] > 1).sum()} จาก {len(t)}")
        have = set(t.index)
        for a, b_, name in [("typhoon8b", "llama8b", "Llama"), ("openthaigpt7b", "qwen7b", "Qwen")]:
            if {a, b_} <= have:
                old = BASE_C[a] - BASE_C[b_]
                new = t.loc[a, "attack ใหม่"] - t.loc[b_, "attack ใหม่"]
                keep = "คงเครื่องหมาย" if old * new > 0 else "** เครื่องหมายพลิก **"
                print(f"  ผลการจูน {name}: เดิม {old:+.3f} -> ใหม่ {new:+.3f}  {keep}")

    # ---- 5.10 long horizon ----
    print("\n" + "=" * 78, "\n5.10  LONG HORIZON\n")
    lh = df[(df.days != 180) & (df.tag == "")]
    if lh.empty:
        print("  (ยังไม่มีผล)")
    else:
        print(fmt(lh[["cond", "model", "seed", "days", "attack", "peak_I", "peak_day",
                      "S_end", "I_end", "run"]], index=False))
        print("\n  เทียบกับค่าสะสม 180 วันของรันเดียวกัน:")
        for _, r in lh.iterrows():
            ref = df[(df.cond == r.cond) & (df.model == r.model) & (df.days == 180)
                     & (df.tag == "") & (df.seed == r.seed)]
            if len(ref):
                print(f"    {r.cond} {r.model} seed {r.seed}: 180 วัน {ref.attack.iloc[0]:.3f}"
                      f" -> {r.days} วัน {r.attack:.3f}  (เพิ่ม {r.attack - ref.attack.iloc[0]:+.3f})")
                print(f"      ยังมีผู้ติดเชื้อที่วันสุดท้าย {r.I_end:.4f} ของประชากร"
                      f" — {'จบแล้ว' if r.I_end < 0.001 else 'ยังไม่จบ'}")

    print("\n" + "=" * 78)
    print(f"รันทั้งหมดที่อ่านได้ {len(df)} · เงื่อนไข {df.cond.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
