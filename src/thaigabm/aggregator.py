"""Behavioural aggregator: individual decisions -> per-stratum SEIR inputs.

This module is the formal interface between the two halves of the model and is
where the guideline's beta_t = beta_0 * f(...) actually lives -- except that we
keep it richer than a single scalar: participation is resolved *per setting and
per stratum*, so a street vendor who keeps working but stops visiting their
parents produces a different epidemic than an office worker who does the
reverse.

    phi^s_g(t) = weighted mean over agents i in g of participation_i in setting s
    mask_g(t)  = weighted mean mask use in g

Weights are the agents' population weights, so a stratum sampled with few
agents still contributes its true population share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.thaigabm.actions import BehaviourDecision
from src.thaigabm.information import POLICY_HARD_CAP
from src.thaigabm.persona import Persona
from src.thaigabm.seir import BehaviourInput
from src.thaigabm.strata import N_STRATA, SETTINGS, STRATUM_INDEX, STRATUM_KEYS

# A person staying home does not stop having household contacts -- they have
# more. Home participation therefore rises as outside participation falls.
HOME_COMPENSATION = 0.30


@dataclass
class AggregateBehaviour:
    behaviour: BehaviourInput
    per_stratum: dict[str, dict[str, float]]
    population: dict[str, float]

    def summary_row(self) -> dict:
        return {f"pop_{k}": round(v, 4) for k, v in self.population.items()}


def aggregate(
    people: list[Persona],
    decisions: list[BehaviourDecision],
    policy: str = "none",
    apply_policy_cap: bool = True,
) -> AggregateBehaviour:
    """Collapse N individual decisions into the per-stratum inputs SEIR needs."""
    w_tot = np.zeros(N_STRATA)
    acc = {s: np.zeros(N_STRATA) for s in SETTINGS}
    mask = np.zeros(N_STRATA)
    care = np.zeros(N_STRATA)
    isolating = np.zeros(N_STRATA)
    distancing = np.zeros(N_STRATA)

    for p, d in zip(people, decisions):
        g = STRATUM_INDEX[p.stratum]
        w = p.weight
        w_tot[g] += w
        acc["work"][g] += w * float(d.go_to_work)
        acc["social"][g] += w * float(d.social_activity)
        acc["family_visit"][g] += w * float(d.visit_family)
        acc["home"][g] += w  # everyone is at home some of the day
        mask[g] += w * float(d.wear_mask)
        care[g] += w * float(d.seek_healthcare)
        isolating[g] += w * float(d.is_isolating)
        distancing[g] += w * float(d.distancing)

    w_safe = np.maximum(w_tot, 1e-12)
    phi = {s: acc[s] / w_safe for s in SETTINGS}
    mask /= w_safe
    care /= w_safe
    isolating /= w_safe
    distancing /= w_safe

    # People who withdraw from work and social life spend that time at home.
    outside = 0.5 * (phi["work"] + phi["social"])
    phi["home"] = 1.0 + HOME_COMPENSATION * (1.0 - outside)

    if apply_policy_cap:
        caps = POLICY_HARD_CAP.get(policy, POLICY_HARD_CAP["none"])
        for s, cap in caps.items():
            phi[s] = np.minimum(phi[s], cap)

    per_stratum: dict[str, dict[str, float]] = {}
    for gi, key in enumerate(STRATUM_KEYS):
        if w_tot[gi] <= 0:
            continue
        per_stratum[key] = {
            "phi_work": float(phi["work"][gi]),
            "phi_social": float(phi["social"][gi]),
            "phi_family_visit": float(phi["family_visit"][gi]),
            "phi_home": float(phi["home"][gi]),
            "mask": float(mask[gi]),
            "isolating": float(isolating[gi]),
            "healthcare_seeking": float(care[gi]),
        }

    W = w_tot.sum() or 1.0
    population = {
        "phi_work": float(w_tot @ phi["work"] / W),
        "phi_social": float(w_tot @ phi["social"] / W),
        "phi_family_visit": float(w_tot @ phi["family_visit"] / W),
        "mask": float(w_tot @ mask / W),
        "isolating": float(w_tot @ isolating / W),
        "distancing": float(w_tot @ distancing / W),
        "healthcare_seeking": float(w_tot @ care / W),
    }

    return AggregateBehaviour(
        behaviour=BehaviourInput(phi=phi, mask=mask, healthcare_seeking=care),
        per_stratum=per_stratum,
        population=population,
    )


def mobility_index(agg: AggregateBehaviour) -> float:
    """Single scalar M_t comparable to a mobility report (Google/Apple style)."""
    p = agg.population
    return float(0.55 * p["phi_work"] + 0.30 * p["phi_social"]
                 + 0.15 * p["phi_family_visit"])
