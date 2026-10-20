"""Action -> contact mapping (guideline section 5.3).

Location contacts per adult per day come from Prem et al. (2021) Thailand matrices (notebook 00).
Each action gets an out-of-home contact level c_out[a]. Home contacts are kept for every action,
so stay_home never means zero contact.

    base_out  = sum_a p0_a * c_out[a]                (normal-day mix, config contacts.baseline_mix)
    w_home    = home / (home + base_out)             (share of normal-day contacts at home)
    q_out(t)  = sum_a p_a(t) * c_out[a] / base_out   (p_a weighted by persona population weights)
    beta_eff  = beta0 * [w_home + (1 - w_home) * q_out * q_policy]

Actions are categories: proportions are computed per action, numeric action codes are never averaged.
"""
from collections import Counter

import pandas as pd


def load_location_contacts(path, column="adult_20_64"):
    df = pd.read_csv(path)
    return dict(zip(df["location"], df[column].astype(float)))


class ContactModel:
    def __init__(self, loc, contacts_cfg, home_scale=1.0):
        p = contacts_cfg
        self.actions = list(p["baseline_mix"])
        self.home = loc["home"] * home_scale
        work_per_worker = (loc["work"] + loc["school"]) / p["employment_rate"]
        self.c_out = {
            "stay_home": 0.0,
            "work_onsite": work_per_worker + p["f_work_other"] * loc["others"],
            "essential_errand": p["f_errand"] * loc["others"],
            "visit_family": p["f_visit"] * loc["others"] + p["f_visit_home"] * self.home,
        }
        assert set(self.c_out) == set(self.actions)
        self.baseline_mix = dict(p["baseline_mix"])
        self.base_out = sum(self.baseline_mix[a] * self.c_out[a] for a in self.actions)
        self.w_home = self.home / (self.home + self.base_out)

    def shares(self, actions, weights=None):
        weights = [1.0] * len(actions) if weights is None else list(weights)
        tot = Counter()
        for a, w in zip(actions, weights):
            tot[a] += w
        s = sum(tot.values())
        return {a: tot[a] / s for a in self.actions}

    def q_out(self, shares):
        return sum(shares[a] * self.c_out[a] for a in self.actions) / self.base_out

    def beta_eff(self, beta0, q_out, q_policy=1.0):
        return beta0 * (self.w_home + (1 - self.w_home) * q_out * q_policy)

    def relative_contact(self, action, q_policy=1.0):
        """Per-agent multiplier on beta0; its baseline-mix average is 1."""
        return self.w_home + (1 - self.w_home) * q_policy * self.c_out[action] / self.base_out

    def table(self):
        rows = [{"action": a, "c_home": self.home, "c_out": self.c_out[a],
                 "relative_contact": self.relative_contact(a), "baseline_share": self.baseline_mix[a]}
                for a in self.actions]
        return pd.DataFrame(rows)
