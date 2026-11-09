"""Headline 2x2 and baseline tables behind sections 5.1, 5.3 and 5.5 of report/final_report_TH.md.

    python3 scripts/analysis/headline_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import pandas as pd
import numpy as np

def col(v, name):
    return v[name].mean() if name in v.columns else np.nan

rows = []
for d in sorted((ROOT / "outputs" / "runs").iterdir()):
    if not (d / "runtime.json").exists():
        continue
    c = json.loads((d / "config.json").read_text())
    if c["cfg"]["sim"]["days"] != 180 or (c.get("tag") or ""):
        continue
    v = pd.read_csv(d / "daily.csv")
    N = v[["S", "E", "I", "R"]].iloc[0].sum()
    rt = json.loads((d / "runtime.json").read_text())
    rows.append(dict(cond=c["condition"], model=c["model_key"] or "-", seed=c["seed"],
                     attack=1 - v.S.iloc[-1] / N,
                     peak_I=v.I.max() / N, peak_day=int(v.I.idxmax()),
                     q_out=col(v, "q_out"),
                     stay=col(v, "share_stay_home"), work=col(v, "share_work_onsite"),
                     errand=col(v, "share_essential_errand"), visit=col(v, "share_visit_family"),
                     S_end=v.S.iloc[-1]/N, E_end=v.E.iloc[-1]/N, I_end=v.I.iloc[-1]/N,
                     fb=v.n_fallback.sum() if "n_fallback" in v else 0,
                     retry=v.n_retried.sum() if "n_retried" in v else 0,
                     secs=rt.get("seconds"), run=d.name))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
df.to_csv(f"{__import__('os').environ['S']}/headline.csv", index=False) if False else df.to_csv("/tmp/claude-1000/-home-siwakorn-Documents-bioinfo-Epidemic-Agent-for-Thai/42bacb72-0ffa-4590-8f06-ca2a2584cead/scratchpad/headline.csv", index=False)
print("n runs:", len(df), "| conditions:", df.cond.value_counts().to_dict())
print()
g = df.groupby(["cond", "model"]).agg(n=("attack","size"), attack=("attack","mean"), sd=("attack","std"),
    peakI=("peak_I","mean"), peakday=("peak_day","mean"), q_out=("q_out","mean"),
    stay=("stay","mean"), work=("work","mean"), errand=("errand","mean"), visit=("visit","mean"),
    S_end=("S_end","mean"), E_end=("E_end","mean"), I_end=("I_end","mean"), fb=("fb","sum"), retry=("retry","sum"))
print(g.to_string(float_format=lambda x: f"{x:.4f}"))
