"""Deterministic SEIR advanced one day at a time with a beta that is constant within the day.

Rates are per day: sigma = 1/latent_days, gamma = 1/infectious_days.
Two cumulative counters track flows: cum_inf (S->E, true infections) and cum_rep (E->I, reported cases,
following Aoki & Ghaffarzadegan where reported cases are the flow out of E).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

KEYS = ("S", "E", "I", "R", "cum_inf", "cum_rep")


def rates(disease):
    return disease["beta0"], 1.0 / disease["latent_days"], 1.0 / disease["infectious_days"]


def initial_state(population):
    N = float(population["N"])
    E = N * population["initial_exposed_frac"]
    I = N * population["initial_infectious_frac"]
    R = N * population["initial_recovered_frac"]
    return {"S": N - E - I - R, "E": E, "I": I, "R": R, "cum_inf": 0.0, "cum_rep": 0.0}


def _rhs(t, y, beta, sigma, gamma, N):
    S, E, I, R, _, _ = y
    inf = beta * S * I / N
    prog = sigma * E
    rec = gamma * I
    return [-inf, inf - prog, prog - rec, rec, inf, prog]


def check_conservation(state, N, tol=1e-6):
    total = state["S"] + state["E"] + state["I"] + state["R"]
    assert abs(total - N) <= tol * N, f"S+E+I+R={total:.3f} differs from N={N}"
    assert min(state[k] for k in ("S", "E", "I", "R")) >= -tol * N, f"negative compartment: {state}"


def step(state, beta, sigma, gamma, N, dt=1.0):
    """Advance one day. Returns the new state plus new_infections and new_reported for that day."""
    y0 = [state[k] for k in KEYS]
    sol = solve_ivp(_rhs, (0.0, dt), y0, args=(beta, sigma, gamma, N), method="RK45", rtol=1e-8, atol=1e-6)
    y = sol.y[:, -1]
    new = dict(zip(KEYS, map(float, y)))
    check_conservation(new, N)
    new["new_infections"] = new["cum_inf"] - state["cum_inf"]
    new["new_reported"] = new["cum_rep"] - state["cum_rep"]
    return new


def run(cfg, days, beta_fn=None):
    """Plain SEIR run. beta_fn(day, state) -> beta; default is constant beta0 (condition A)."""
    beta0, sigma, gamma = rates(cfg["disease"])
    N = float(cfg["population"]["N"])
    state = initial_state(cfg["population"])
    rows = []
    for day in range(days):
        beta = beta0 if beta_fn is None else beta_fn(day, state)
        state = step(state, beta, sigma, gamma, N)
        rows.append({"day": day, "beta_eff": beta, **state})
    return rows


def final_size(R0):
    """Final attack rate z from z = 1 - exp(-R0 z) for a fully susceptible population (R0 > 1)."""
    if R0 <= 1:
        return 0.0
    return brentq(lambda z: z - (1 - np.exp(-R0 * z)), 1e-9, 1.0)


def trailing_mean(values, window=7):
    values = list(values)[-window:]
    return float(np.mean(values)) if values else 0.0
