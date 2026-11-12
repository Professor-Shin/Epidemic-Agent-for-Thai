"""Section 5.5.1: split the C - D gap into the policy's mechanical and behavioural channels.

A policy level acts on the epidemic through two separate doors: q_policy multiplies transmission per
contact, and the measure text plus the banned-action list change the prompt residents see, and therefore
what they choose. Reporting only C - D leaves those two added together and invites the objection that the
policy is being counted twice.

Neither channel needs a new run. daily.csv already records q_out for every day of condition C and
q_policy / level for every day of condition D, so the counterfactuals are re-integrations of the recorded
series through the same SEIR step the simulation used:

    mechanical  = attack(C) - attack(C's q_out  under D's q_policy)
    behavioural = attack(C) - attack(D's q_out  with q_policy = 1)

Neither is a clean counterfactual on its own: residents in D did not in fact behave as they did in C.
The check that the split is informative is that the two channels sum to roughly the observed C - D gap;
the residual is the interaction between them.

Both channels are attack-rate deltas, so they also carry the curvature of the final-size curve: a model
sitting lower on that curve converts the same change in transmission into a larger change in attack rate.
d_q_out is the curvature-free behavioural measure - the drop in mean q_out between C and D - and it is the
column to use when comparing how much residents of different models actually responded to the announcement.

    python3 scripts/analysis/policy_decomposition.py [--seed 1] [--days 180]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from gabm import load_config, seir
from gabm.sim import contact_model_from_cfg

MODELS = ["llama8b", "typhoon8b", "qwen7b", "openthaigpt7b", "seallm7b"]


def find_run(cond, model, seed, days):
    """The untagged run of this (condition, model, seed, days), or None."""
    for d in sorted((ROOT / "outputs" / "runs").glob(f"{cond}_{model}_s{seed}_*")):
        if not (d / "runtime.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if c["tag"] or c["cfg"]["sim"]["days"] != days:
            continue
        return d
    return None


def replay(cfg, cm, q_out, q_policy):
    """Re-integrate SEIR over recorded daily (q_out, q_policy) pairs; returns the attack rate."""
    beta0, sigma, gamma = seir.rates(cfg["disease"])
    N = float(cfg["population"]["N"])
    state = seir.initial_state(cfg["population"])
    for q, qp in zip(q_out, q_policy):
        state = seir.step(state, cm.beta_eff(beta0, q, qp), sigma, gamma, N)
    return 1 - state["S"] / N


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    cfg = load_config(ROOT / "config" / "base.yaml")
    cm = contact_model_from_cfg(cfg, ROOT)
    N = float(cfg["population"]["N"])

    rows = []
    for m in MODELS:
        dc, dd = find_run("C", m, args.seed, args.days), find_run("D", m, args.seed, args.days)
        if dc is None or dd is None:
            print(f"  skipping {m}: missing {'C' if dc is None else 'D'} run", file=sys.stderr)
            continue
        C, D = pd.read_csv(dc / "daily.csv"), pd.read_csv(dd / "daily.csv")
        n = min(len(C), len(D))
        aC, aD = 1 - C.S.iloc[n - 1] / N, 1 - D.S.iloc[n - 1] / N
        a_mech = replay(cfg, cm, C.q_out[:n], D.q_policy[:n])
        a_behav = replay(cfg, cm, D.q_out[:n], [1.0] * n)
        rows.append({
            "model": m, "C": aC, "D": aD, "gap": aC - aD,
            "mech": aC - a_mech, "behav": aC - a_behav, "sum": (aC - a_mech) + (aC - a_behav),
            "d_q_out": C.q_out[:n].mean() - D.q_out[:n].mean(),
            "q_out_C": C.q_out[:n].mean(), "q_out_D": D.q_out[:n].mean(),
            "q_policy_mean": D.q_policy[:n].mean(),
            "days_open": int((D.level == "open").sum()),
            "days_limited": int((D.level == "limited").sum()),
            "days_strict": int((D.level == "strict").sum()),
        })

    df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    df["residual"] = df.gap - df["sum"]
    pd.set_option("display.width", 220)
    print(f"seed {args.seed}, {args.days} days\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out = ROOT / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"policy_decomposition_s{args.seed}.csv", index=False)
    print(f"\nwrote {out / f'policy_decomposition_s{args.seed}.csv'}")


if __name__ == "__main__":
    main()
