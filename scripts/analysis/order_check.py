"""Action-order table behind section 5.7 of report/final_report_TH.md.

    python3 scripts/analysis/order_check.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import pandas as pd
rows = []
for d in sorted((ROOT / "outputs" / "runs").glob("C_*_s1_*")):
    if not (d / "runtime.json").exists():
        continue
    c = json.loads((d / "config.json").read_text())
    if c["cfg"]["sim"]["days"] != 180:
        continue
    tag = c.get("tag") or "order=base"
    if not tag.startswith("order="):
        continue
    v = pd.read_csv(d / "daily.csv")
    N = v[["S", "E", "I", "R"]].iloc[0].sum()
    rows.append(dict(model=c["model_key"], tag=tag,
                     stay=v.share_stay_home.mean(), work=v.share_work_onsite.mean(),
                     errand=v.share_essential_errand.mean(), visit=v.share_visit_family.mean(),
                     q_out=v.q_out.mean(), attack=1 - v.S.iloc[-1] / N, run=d.name[-8:]))
order = {"order=base": 0, "order=rev": 1, "order=visitfirst": 2}
df = pd.DataFrame(rows).assign(o=lambda x: x.tag.map(order)).sort_values(["model", "o"]).drop(columns="o")
pd.set_option("display.width", 220)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()
p = df.pivot(index="model", columns="tag", values="attack")[list(order)]
print("attack rate spread:"); print(p.assign(spread=p.max(1) - p.min(1)).to_string(float_format=lambda x: f"{x:.3f}"))
print()
print("localisation effect (Thai-tuned - base), by order:")
for a, b, name in [("typhoon8b", "llama8b", "Llama"), ("openthaigpt7b", "qwen7b", "Qwen")]:
    print(f"  {name:<6}", {t: round(p.loc[a, t] - p.loc[b, t], 3) for t in order})
df.to_csv(f"{Path(__file__).parent}/order.csv", index=False)
