"""Section 5.3: exact permutation tests for the localisation effect inside each model family.

The effect being tested is the difference in 180-day attack rate between a base model and its Thai-tuned
counterpart. Seeds are the unit of resampling, so the test asks: if the tuned/base label carried no
information, how often would relabelling the runs produce a mean difference at least this extreme?

The test is exact - every one of the C(n_a + n_b, n_a) relabellings is enumerated, not sampled - so the
reported p is not itself a random quantity. Two things about it must be stated wherever it is quoted:

  * p has a floor of 2 / C(n_a + n_b, n_a) for a two-sided test. At 3 vs 3 that floor is 0.10, so no
    amount of evidence can push a 3-seed comparison below it. The floor is a property of the design, not
    of the data, and the script prints it next to every p so the two are never confused.
  * seeds vary the SEIR/health RNG and the LLM sampling seed only. personas are drawn once with
    personas.persona_seed and reused by every run, so this test quantifies sampling noise around ONE
    synthetic population; it says nothing about how the effect would move under a different persona draw.
    Section 5.8.3 is the evidence that that second source is the larger one.

    python3 scripts/analysis/permutation_test.py [--days 180] [--condition C]
"""
import argparse
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PAIRS = [("Llama 3.1 8B", "llama8b", "typhoon8b"), ("Qwen 2.5 7B", "qwen7b", "openthaigpt7b")]


def attack_by_seed(condition, days):
    """{model_key: {seed: attack rate}} over untagged runs of this condition and horizon."""
    out = {}
    for d in sorted((ROOT / "outputs" / "runs").glob(f"{condition}_*")):
        if not (d / "runtime.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if c["tag"] or c["cfg"]["sim"]["days"] != days or c["backend"] == "mock":
            continue
        v = pd.read_csv(d / "daily.csv")
        N = v[["S", "E", "I", "R"]].iloc[0].sum()
        out.setdefault(c["model_key"], {})[c["seed"]] = 1 - v.S.iloc[-1] / N
    return out


def exact_permutation(a, b):
    """Two-sided exact permutation test on the difference of means. Returns (diff, p, floor, n_perm)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool, na = np.concatenate([a, b]), len(a)
    observed = b.mean() - a.mean()
    diffs = []
    for idx in combinations(range(len(pool)), na):
        mask = np.zeros(len(pool), bool)
        mask[list(idx)] = True
        diffs.append(pool[~mask].mean() - pool[mask].mean())
    diffs = np.asarray(diffs)
    n_perm = comb(len(pool), na)
    p = float((np.abs(diffs) >= abs(observed) - 1e-12).sum()) / n_perm
    return observed, p, 2.0 / n_perm, n_perm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--condition", default="C")
    args = ap.parse_args()

    acc = attack_by_seed(args.condition, args.days)
    rows = []
    print(f"condition {args.condition}, {args.days} days, seeds are the resampling unit\n")
    for family, base, tuned in PAIRS:
        if base not in acc or tuned not in acc:
            print(f"  {family}: missing runs for {base if base not in acc else tuned}")
            continue
        sb, st = acc[base], acc[tuned]
        seeds = sorted(set(sb) & set(st))          # only seeds present for BOTH models
        a, b = [sb[s] for s in seeds], [st[s] for s in seeds]
        diff, p, floor, nperm = exact_permutation(a, b)
        rows.append(dict(family=family, base=base, tuned=tuned, n=len(seeds), seeds=",".join(map(str, seeds)),
                         base_mean=np.mean(a), tuned_mean=np.mean(b), effect=diff,
                         base_sd=np.std(a, ddof=1), tuned_sd=np.std(b, ddof=1),
                         p=p, p_floor=floor, n_perm=nperm, at_floor=abs(p - floor) < 1e-12))
        print(f"  {family}")
        print(f"    seeds used          : {seeds}  (n = {len(seeds)} per model)")
        print(f"    {base:<14}      : {np.mean(a):.4f}  sd {np.std(a, ddof=1):.4f}")
        print(f"    {tuned:<14}      : {np.mean(b):.4f}  sd {np.std(b, ddof=1):.4f}")
        print(f"    effect (tuned-base) : {diff:+.4f}")
        pooled = np.std(np.concatenate([a - np.mean(a), b - np.mean(b)]), ddof=1)
        print(f"    effect / pooled sd  : {abs(diff) / pooled:.1f}x")
        print(f"    exact two-sided p   : {p:.4f}   (floor {floor:.4f} over {nperm} relabellings"
              f"{', p IS the floor' if abs(p - floor) < 1e-12 else ''})")
        print()

    if not rows:
        return
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"permutation_{args.condition}_{args.days}d.csv", index=False)
    print(f"wrote {out / f'permutation_{args.condition}_{args.days}d.csv'}")
    if df.at_floor.any():
        print("\nNOTE: a p at its floor reports the number of seeds, not the strength of the evidence.")
        print("      Quote it together with the effect size and the sd, never on its own.")


if __name__ == "__main__":
    main()
