"""Main simulation loop: orchestrates GAI decisions and ABM spread each day."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
from typing import Union

from src.agent import Decision, DecisionAgent, MockDecisionAgent
from src.config import SimulationConfig
from src.disease import DISEASE_DURATION, infect
from src.person import HealthState, Person
from src.traits import sample_age, sample_name, sample_traits
from src.world import World

AgentLike = Union[DecisionAgent, MockDecisionAgent]


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.people: list[Person] = self._create_population()
        self._seed_initial_infections()
        self.world = World(
            self.people, config.contact_rate, config.infectivity, self.rng
        )
        self.agent: AgentLike = (
            MockDecisionAgent(self.rng)
            if config.mock
            else DecisionAgent(model=config.model)
        )
        self.history: list[dict] = []

    def _create_population(self) -> list[Person]:
        used: set[str] = set()
        return [
            Person(
                id=i,
                name=sample_name(self.rng, used),
                age=sample_age(self.rng),
                traits=sample_traits(self.rng),
            )
            for i in range(self.config.population)
        ]

    def _seed_initial_infections(self) -> None:
        n = min(self.config.initial_infected, len(self.people))
        for pid in self.rng.sample(range(len(self.people)), n):
            self.people[pid].health = HealthState.INFECTED
            self.people[pid].infection_day = 1  # start on their first infectious day

    async def _gather_decisions(
        self, case_pct: float, day: int
    ) -> tuple[set[int], list[Decision]]:
        tasks = [
            self.agent.decide(
                person=p,
                condition=self.config.condition,
                daily_case_pct=case_pct,
                day=day,
                virus_name=self.config.virus_name,
                town_name=self.config.town_name,
            )
            for p in self.people
        ]
        decisions = await asyncio.gather(*tasks)
        active = {p.id for p, d in zip(self.people, decisions) if d.go_to_work}
        return active, decisions

    async def run(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(
            f"Running condition='{self.config.condition}' "
            f"n={self.config.population} days={self.config.days} "
            f"contacts={self.config.contact_rate} "
            f"infectivity={self.config.infectivity} "
            f"model={'mock' if self.config.mock else self.config.model}"
        )
        print("-" * 78)

        prev_day4 = 0
        for day in range(1, self.config.days + 1):
            # Yesterday's newly-symptomatic (day 4) count feeds today's news.
            case_pct = 100.0 * prev_day4 / len(self.people)

            # 1) GAI decision phase.
            active_ids, decisions = await self._gather_decisions(case_pct, day)

            # 2) ABM contact/transmission phase.
            newly_infected = self.world.compute_new_infections(active_ids)
            for pid in newly_infected:
                infect(self.people[pid])  # sets health=INFECTED, infection_day=0

            # 3) Health status update at end of day.
            for p in self.people:
                if p.health == HealthState.INFECTED and p.infection_day is not None:
                    p.infection_day += 1
                    if p.infection_day > DISEASE_DURATION:
                        p.health = HealthState.RECOVERED
                        p.infection_day = None

            # 4) Bookkeeping.
            prev_day4 = sum(
                1
                for p in self.people
                if p.health == HealthState.INFECTED and p.infection_day == 4
            )
            record = self._daily_record(day, active_ids, newly_infected, decisions, case_pct)
            self.history.append(record)
            self._print_daily(record)

            # Early stop if disease is gone.
            if record["infected"] == 0 and day > 1:
                print(f"(no more active infections after day {day}; stopping early)")
                break

        self._save_results()

    def _daily_record(
        self,
        day: int,
        active_ids: set[int],
        newly_infected: list[int],
        decisions: list[Decision],
        case_pct: float,
    ) -> dict:
        return {
            "day": day,
            "case_pct_yesterday": round(case_pct, 3),
            "went_to_work": len(active_ids),
            "stayed_home": len(self.people) - len(active_ids),
            "new_infections": len(newly_infected),
            "susceptible": sum(
                1 for p in self.people if p.health == HealthState.SUSCEPTIBLE
            ),
            "infected": sum(
                1 for p in self.people if p.health == HealthState.INFECTED
            ),
            "recovered": sum(
                1 for p in self.people if p.health == HealthState.RECOVERED
            ),
            "decisions": [
                {
                    "id": p.id,
                    "name": p.name,
                    "age": p.age,
                    "go": d.go_to_work,
                    "reason": d.reason,
                    "health": p.health.value,
                    "infection_day": p.infection_day,
                }
                for p, d in zip(self.people, decisions)
            ],
        }

    def _print_daily(self, r: dict) -> None:
        print(
            f"Day {r['day']:3d} | S={r['susceptible']:3d} "
            f"I={r['infected']:3d} R={r['recovered']:3d} | "
            f"active={r['went_to_work']:3d} | new={r['new_infections']:3d} | "
            f"news={r['case_pct_yesterday']:5.2f}%"
        )

    def _save_results(self) -> None:
        cond = self.config.condition
        out = self.config.output_dir

        summary_path = os.path.join(out, f"summary_{cond}.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "day",
                    "susceptible",
                    "infected",
                    "recovered",
                    "new_infections",
                    "went_to_work",
                    "stayed_home",
                    "case_pct_yesterday",
                ]
            )
            for r in self.history:
                w.writerow(
                    [
                        r["day"],
                        r["susceptible"],
                        r["infected"],
                        r["recovered"],
                        r["new_infections"],
                        r["went_to_work"],
                        r["stayed_home"],
                        r["case_pct_yesterday"],
                    ]
                )

        detail_path = os.path.join(out, f"detail_{cond}.json")
        with open(detail_path, "w") as f:
            json.dump(
                {"config": self.config.to_dict(), "history": self.history},
                f,
                indent=2,
            )

        print("-" * 78)
        print(f"Wrote {summary_path}")
        print(f"Wrote {detail_path}")
