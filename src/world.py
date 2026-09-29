"""Agent-based model of the town's daily contact network.

Every person who decides to go out draws ``contact_rate`` unique other outgoing
people to interact with. We keep contacts as an undirected edge set so that a
pair (A, B) is not counted twice when both sampled each other. For every edge
with exactly one infectious endpoint and one susceptible endpoint, we roll a
transmission attempt with probability ``infectivity``.
"""

from __future__ import annotations

import random
from typing import Iterable

from src.person import HealthState, Person


class World:
    def __init__(
        self,
        people: list[Person],
        contact_rate: int,
        infectivity: float,
        rng: random.Random,
    ):
        self.people = people
        self.contact_rate = contact_rate
        self.infectivity = infectivity
        self.rng = rng

    def compute_new_infections(self, active_ids: Iterable[int]) -> list[int]:
        active_ids = set(active_ids)
        active = [p for p in self.people if p.id in active_ids]
        if len(active) < 2:
            return []

        edges: set[tuple[int, int]] = set()
        for p in active:
            others = [o for o in active if o.id != p.id]
            k = min(self.contact_rate, len(others))
            for c in self.rng.sample(others, k):
                edges.add(tuple(sorted((p.id, c.id))))  # type: ignore[arg-type]

        id_map = {p.id: p for p in self.people}
        infected_today: list[int] = []
        infected_set: set[int] = set()

        for a, b in edges:
            pa, pb = id_map[a], id_map[b]
            susceptible: Person | None = None
            source: Person | None = None
            if pa.health == HealthState.SUSCEPTIBLE and pb.is_infectious:
                susceptible, source = pa, pb
            elif pb.health == HealthState.SUSCEPTIBLE and pa.is_infectious:
                susceptible, source = pb, pa
            if susceptible is None or susceptible.id in infected_set:
                continue
            if self.rng.random() < self.infectivity:
                infected_today.append(susceptible.id)
                infected_set.add(susceptible.id)

        return infected_today
