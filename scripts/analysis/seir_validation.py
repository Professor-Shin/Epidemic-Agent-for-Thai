"""Section 5.1: run condition A to convergence and compare it with the analytic final size.

Condition A calls no LLM, so the long horizons here cost seconds of CPU and need no GPU run on disk.
The comparison the report makes is NOT the textbook final size z = 1 - exp(-R0 z): that assumes a fully
susceptible population, while the model seeds E = I = 0.002, leaving S0/N = 0.996. The matching relation
is s = s0 * exp(-R0 (1 - s)), and it is the one the simulation reproduces exactly.

    python3 scripts/analysis/seir_validation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from gabm import load_config, seir

HORIZONS = (180, 365, 730, 1095)


def converge(cfg, horizons=HORIZONS):
    beta0, sigma, gamma = seir.rates(cfg["disease"])
    N = float(cfg["population"]["N"])
    rows, state, done = [], seir.initial_state(cfg["population"]), 0
    for h in horizons:
        for _ in range(h - done):
            state = seir.step(state, beta0, sigma, gamma, N)
        done = h
        rows.append({"days": h, "attack": 1 - state["S"] / N, "I_frac": state["I"] / N})
    return pd.DataFrame(rows)


def analytic(cfg):
    beta0, _, gamma = seir.rates(cfg["disease"])
    R0 = beta0 / gamma
    p = cfg["population"]
    s0 = 1 - p["initial_exposed_frac"] - p["initial_infectious_frac"] - p["initial_recovered_frac"]
    naive = brentq(lambda z: z - (1 - np.exp(-R0 * z)), 1e-9, 1.0)
    seeded = 1 - brentq(lambda s: s - s0 * np.exp(-R0 * (1 - s)), 1e-12, s0)
    return R0, s0, naive, seeded


def main():
    cfg = load_config(ROOT / "config" / "base.yaml")
    df = converge(cfg)
    R0, s0, naive, seeded = analytic(cfg)
    print(f"R0 = {R0:.1f} | S0/N = {s0:.4f} (E and I seeded at "
          f"{cfg['population']['initial_exposed_frac']} each)\n")
    print("condition A, no behavioural response, no LLM:")
    for _, r in df.iterrows():
        print(f"  {int(r.days):>5}d  attack = {r.attack:.6f}   I = {r.I_frac:.3e}")
    print()
    print(f"  textbook final size, fully susceptible start : {naive:.6f}")
    print(f"  final size matching S0/N = {s0:.4f}            : {seeded:.6f}   <- the right comparison")
    conv = df.attack.iloc[-1]
    print(f"  simulated, converged                         : {conv:.6f}")
    print(f"\n  |simulated - seeded| = {abs(conv - seeded):.2e}")
    out = ROOT / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.assign(analytic_seeded=seeded, analytic_naive=naive).to_csv(out / "seir_validation.csv", index=False)
    print(f"\nwrote {out / 'seir_validation.csv'}")


if __name__ == "__main__":
    main()
