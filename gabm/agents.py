"""Representative resident agents: personas, personal health state, and what each agent observes.

Agents are behavioural representatives of population groups (guideline section 5.1), not one agent per person.
Each persona carries a population weight; action shares are weighted by it.
"""
import numpy as np
import pandas as pd

AGE_TEXT = {"20-29": "20-29", "30-39": "30-39", "40-49": "40-49", "50-64": "50-64"}
INCOME_TEXT = {
    "high": "high (daily or self-employed income, no pay on days you do not work)",
    "medium": "medium (household depends partly on your work)",
    "low": "low (salaried with paid leave, or income does not depend on going out)",
}


def _allocate(shares, n):
    """Largest-remainder allocation of n agents to groups, at least one per group."""
    raw = np.array(shares) * n
    counts = np.maximum(np.floor(raw).astype(int), 1)
    while counts.sum() < n:
        counts[np.argmax(raw - counts)] += 1
    while counts.sum() > n:
        counts[np.argmax(counts - raw)] -= 1
    return counts


def make_personas(marginals, pcfg, n, seed, wfh_scale=1.0):
    """Stratified, seeded persona draw. Returns one row per agent with a population weight."""
    rng = np.random.default_rng(seed)
    m = marginals.copy()
    m["share"] = m["share"] / m["share"].sum()
    counts = _allocate(m["share"].values, n)

    def draw(dist):
        keys = list(dist)
        p = np.array([dist[k] for k in keys], float)
        return keys[rng.choice(len(keys), p=p / p.sum())]

    rows = []
    for (_, g), k in zip(m.iterrows(), counts):
        for _ in range(k):
            u = rng.random()
            rows.append({
                "group": g["group"],
                "occupation": g["label"],
                "age_group": draw(pcfg["age_groups"]),
                "p_can_wfh": g["p_can_wfh"],
                "wfh_u": u,                      # kept so WFH sensitivity can rescale without redrawing others
                "can_wfh": bool(u < min(1.0, g["p_can_wfh"] * wfh_scale)),
                "income_loss": g["income_loss"],
                "essential_worker": str(g["essential_worker"]).lower() == "true",
                "household_size": draw(pcfg["household_size"]),
                "high_risk_member": bool(rng.random() < pcfg["p_high_risk_member"]),
                "risk_perception": draw(pcfg["risk_perception"]),
                "trust_health_info": draw(pcfg["trust_health_info"]),
                "weight": g["share"] / k,
            })
    df = pd.DataFrame(rows)
    df.insert(0, "agent_id", range(len(df)))
    return df


def rescale_wfh(personas, wfh_scale):
    """Same personas, only WFH ability changed: can_wfh = wfh_u < min(1, p_can_wfh * scale)."""
    p = personas.copy()
    p["can_wfh"] = p["wfh_u"] < np.minimum(1.0, p["p_can_wfh"] * wfh_scale)
    return p


def persona_text(p):
    in_work = p["group"] != "notwork"
    lines = [
        f"- Age group: {AGE_TEXT.get(p['age_group'], p['age_group'])}",
        f"- Main activity: {p['occupation']}",
        f"- Can work from home: {'yes' if p['can_wfh'] else 'no'}" if in_work else "- Not in paid work",
        f"- Income loss if you do not go out to work: {INCOME_TEXT[p['income_loss']]}",
        f"- Household size: {p['household_size']} people",
        f"- Lives with an elderly or chronically ill person: {'yes' if p['high_risk_member'] else 'no'}",
        f"- How serious you think this disease is: {p['risk_perception']}",
        f"- Trust in official health information: {p['trust_health_info']}",
    ]
    if p.get("essential_worker"):
        lines.append("- Your work is considered essential")
    return "\n".join(lines)


class AgentHealth:
    """Personal S/E/I/R status per agent, driven by the population force of infection and the agent's own contacts."""

    def __init__(self, n, state0, N, symptomatic_frac, rng):
        p = np.array([state0["S"], state0["E"], state0["I"], state0["R"]]) / N
        self.status = rng.choice(np.array(list("SEIR")), size=n, p=p / p.sum())
        self.symptomatic = (self.status == "I") & (rng.random(n) < symptomatic_frac)
        self.days_sick = np.zeros(n, int)
        self.symptomatic_frac = symptomatic_frac

    def step(self, rng, beta0, I_frac, rel_contacts, sigma, gamma):
        n = len(self.status)
        u = rng.random(n)
        new_status = self.status.copy()
        s, e, i = self.status == "S", self.status == "E", self.status == "I"
        p_inf = 1 - np.exp(-beta0 * I_frac * np.asarray(rel_contacts))
        new_status[s & (u < p_inf)] = "E"
        to_i = e & (u < 1 - np.exp(-sigma))
        new_status[to_i] = "I"
        new_status[i & (u < 1 - np.exp(-gamma))] = "R"
        self.symptomatic = np.where(to_i, rng.random(n) < self.symptomatic_frac, self.symptomatic)
        self.symptomatic &= new_status == "I"
        self.days_sick = np.where(self.symptomatic, self.days_sick + 1, 0)
        self.status = new_status

    def health_text(self, i):
        if self.symptomatic[i]:
            return f"You have had fever and cough for {self.days_sick[i] or 1} day(s)."
        if self.status[i] == "R":
            return "You had this illness earlier and have recovered."
        return "You feel healthy and have no symptoms."

    def to_dict(self):
        return {"status": self.status.tolist(), "symptomatic": self.symptomatic.tolist(),
                "days_sick": self.days_sick.tolist(), "symptomatic_frac": self.symptomatic_frac}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.status = np.array(d["status"])
        obj.symptomatic = np.array(d["symptomatic"], bool)
        obj.days_sick = np.array(d["days_sick"], int)
        obj.symptomatic_frac = d["symptomatic_frac"]
        return obj


def case_report(reported_history, N):
    """What residents see: today's reported cases and 7-day average per 100,000, plus the trend."""
    per100k = 1e5 / N
    today = reported_history[-1] * per100k if reported_history else 0.0
    last7 = np.mean(reported_history[-7:]) * per100k if reported_history else 0.0
    prev7 = np.mean(reported_history[-14:-7]) * per100k if len(reported_history) >= 14 else None
    if prev7 is None or prev7 == 0:
        trend = "unknown" if not reported_history else ("rising" if today > last7 else "stable or falling")
    else:
        ratio = last7 / prev7
        trend = "rising" if ratio > 1.1 else "falling" if ratio < 0.9 else "stable"
    return {"today_per100k": round(float(today), 1), "avg7_per100k": round(float(last7), 1), "trend": trend}
