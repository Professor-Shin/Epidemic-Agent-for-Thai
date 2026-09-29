"""Shadow individual health states for the representative agents.

The SEIR model is compartmental, but an agent cannot reason about "I am 3.2%
infected". So each agent carries a *shadow* individual trajectory driven by the
exact same rates as the compartments:

    P(S -> E) per day = 1 - exp(-lambda_g)      (lambda_g from the SEIR model)
    P(E -> I) per day = 1 - exp(-sigma)
    P(I -> R) per day = 1 - exp(-gamma)

In expectation the shadow population reproduces the compartment dynamics, so
nothing is double-counted: the shadow layer supplies *perception* (my symptoms,
my household's symptoms), while the SEIR layer remains the sole authority on
epidemiology. Only the shadow states feed back into prompts, never into
transmission.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.thaigabm.persona import Population
from src.thaigabm.strata import STRATUM_INDEX


class Health(str, Enum):
    SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


SYMPTOM_TEXT = {
    "none": "feel completely normal",
    "mild": "have a light cough and feel a bit tired",
    "severe": "have a fever and a bad cough",
}


@dataclass
class ShadowState:
    health: Health = Health.SUSCEPTIBLE
    days_infectious: int = 0
    symptomatic: bool = False
    tested_positive: bool = False
    days_isolated: int = 0

    @property
    def symptom_level(self) -> str:
        if self.health is not Health.INFECTIOUS or not self.symptomatic:
            return "none"
        d = self.days_infectious
        if d <= 1:
            return "none"
        if d in (2, 5, 6):
            return "mild"
        return "severe"

    def symptom_text(self) -> str:
        return SYMPTOM_TEXT[self.symptom_level]


class ShadowPopulation:
    def __init__(self, pop: Population, rng: random.Random,
                 symptomatic_fraction: float, sigma: float, gamma: float):
        self.pop = pop
        self.rng = rng
        self.symptomatic_fraction = symptomatic_fraction
        self.p_progress = 1.0 - math.exp(-sigma)
        self.p_recover = 1.0 - math.exp(-gamma)
        self.state: dict[int, ShadowState] = {p.id: ShadowState() for p in pop.people}
        self._households = {h.id: h for h in pop.households}

    # -- dynamics ---------------------------------------------------------
    def step(self, lam: np.ndarray, isolating: dict[int, bool] | None = None) -> None:
        """Advance every shadow individual by one day given the SEIR hazards."""
        isolating = isolating or {}
        p_infect = 1.0 - np.exp(-np.clip(lam, 0.0, 50.0))
        for p in self.pop.people:
            s = self.state[p.id]
            if s.health is Health.INFECTIOUS:
                s.days_infectious += 1
                if self.rng.random() < self.p_recover:
                    s.health = Health.RECOVERED
                    s.symptomatic = False
            elif s.health is Health.EXPOSED:
                if self.rng.random() < self.p_progress:
                    s.health = Health.INFECTIOUS
                    s.days_infectious = 0
                    s.symptomatic = self.rng.random() < self.symptomatic_fraction
            elif s.health is Health.SUSCEPTIBLE:
                hazard = float(p_infect[STRATUM_INDEX[p.stratum]])
                # Someone who stayed home today faces a much lower hazard; this
                # only shapes *who* in the shadow layer gets sick, not how many
                # infections the epidemic produces.
                if isolating.get(p.id):
                    hazard *= 0.35
                if self.rng.random() < hazard:
                    s.health = Health.EXPOSED
            if isolating.get(p.id):
                s.days_isolated += 1
            else:
                s.days_isolated = 0

    def seed(self, prevalence: float) -> None:
        """Seed shadow infections at the SEIR model's initial prevalence.

        Seeding a fixed count instead would make agents perceive an outbreak
        that the compartments do not yet have, biasing early behaviour.
        """
        n = int(round(prevalence * len(self.pop.people)))
        for p in self.rng.sample(self.pop.people, min(max(n, 0), len(self.pop.people))):
            s = self.state[p.id]
            s.health = Health.INFECTIOUS
            s.days_infectious = 1
            s.symptomatic = self.rng.random() < self.symptomatic_fraction

    def resync(self, prevalence_by_stratum: np.ndarray) -> None:
        """Nudge shadow prevalence toward the SEIR truth.

        Sampling noise in a few hundred agents can drift away from the
        compartment solution in either direction. Periodically we top up or trim
        infectious shadow agents per stratum so that agents' *perceptions* stay
        consistent with the epidemic the SEIR model reports. Expected-value
        rounding is used so small prevalences still produce the right long-run
        frequency of "I feel sick" days rather than a hard zero.
        """
        by_str = self.pop.by_stratum()
        for key, members in by_str.items():
            if not members:
                continue
            target = float(prevalence_by_stratum[STRATUM_INDEX[key]])
            cur = [m for m in members if self.state[m.id].health is Health.INFECTIOUS]
            expected = target * len(members)
            want = int(expected) + (1 if self.rng.random() < expected % 1 else 0)
            if want > len(cur):
                pool = [m for m in members
                        if self.state[m.id].health is Health.SUSCEPTIBLE]
                for m in self.rng.sample(pool, min(want - len(cur), len(pool))):
                    s = self.state[m.id]
                    s.health = Health.INFECTIOUS
                    s.days_infectious = self.rng.randint(0, 3)
                    s.symptomatic = self.rng.random() < self.symptomatic_fraction
            elif want < len(cur):
                for m in self.rng.sample(cur, len(cur) - want):
                    s = self.state[m.id]
                    s.health = Health.RECOVERED
                    s.symptomatic = False

    # -- perception --------------------------------------------------------
    def household_situation(self, person_id: int) -> dict:
        p = next(x for x in self.pop.people if x.id == person_id)
        hh = self._households[p.household_id]
        others = [m for m in hh.member_ids if m != person_id]
        sick = [m for m in others if self.state[m].symptom_level != "none"]
        return {
            "household_size": hh.size,
            "someone_sick_at_home": bool(sick),
            "n_sick_at_home": len(sick),
            "elderly_at_home": p.elderly_in_household,
            "children_at_home": p.children_in_household,
        }

    def counts(self) -> dict[str, int]:
        out = {h.value: 0 for h in Health}
        for s in self.state.values():
            out[s.health.value] += 1
        return out
