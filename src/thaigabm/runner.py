"""The coupled ThaiGABM-SEIR simulation loop.

One simulated day:

    1. perception   shadow health + household + yesterday's news -> AgentContext
    2. behaviour    BehaviourModel -> one BehaviourDecision per agent   (LLM here)
    3. aggregation  decisions -> phi^s_g, mask_g                        (no LLM)
    4. epidemiology stratified SEIR advances one day                    (no LLM)
    5. perception   SEIR hazards drive the shadow individuals
    6. information  true incidence -> reported cases -> policy -> feedback

The LLM appears in exactly one of these six steps. That separation is the
project's modelling contribution and is what makes the arms comparable.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np

from src.thaigabm.actions import BehaviourDecision
from src.thaigabm.aggregator import aggregate, mobility_index
from src.thaigabm.behaviour import AgentContext, BehaviourModel
from src.thaigabm.config import RunConfig
from src.thaigabm.evidence import evidence_table
from src.thaigabm.information import InformationEnvironment, PolicyRegime
from src.thaigabm.persona import Population
from src.thaigabm.seir import StratifiedSEIR
from src.thaigabm.shadow import ShadowPopulation
from src.thaigabm.strata import STRATUM_KEYS


@dataclass
class RunResult:
    config: dict
    history: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def series(self, key: str) -> list[float]:
        return [float(r[key]) for r in self.history]


class ThaiGABMSimulation:
    def __init__(self, cfg: RunConfig, behaviour_factory=None):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.population = Population(
            n_agents=cfg.n_agents,
            total_population=cfg.total_population,
            rng=self.rng,
            flavour="thai" if cfg.arm.thai_context else "generic",
            ablate=cfg.arm.ablate,
        )
        N = np.array(self.population.stratum_populations())
        self.seir = StratifiedSEIR(
            N, disease=cfg.disease, mask_efficacy=cfg.mask_efficacy,
            family_visit_boost=cfg.family_visit_boost,
        )
        self.seir.seed(cfg.initial_infected)
        self.shadow = ShadowPopulation(
            self.population, random.Random(cfg.seed + 1),
            symptomatic_fraction=cfg.disease.symptomatic_fraction,
            sigma=cfg.disease.sigma, gamma=cfg.disease.gamma,
        )
        self.shadow.seed(cfg.initial_infected / cfg.total_population)
        self.info = InformationEnvironment(
            total_population=cfg.total_population,
            ascertainment=cfg.disease.ascertainment,
            reporting_delay_days=cfg.reporting_delay_days,
            policy=PolicyRegime(enabled=cfg.policy_enabled),
            stigma_suppression=cfg.stigma_suppression,
        )
        self.behaviour: BehaviourModel = (
            behaviour_factory(self) if behaviour_factory else cfg.build_behaviour(self)
        )
        self.result = RunResult(config=cfg.to_dict())
        self._last_decisions: list[BehaviourDecision] | None = None

    # -- one day -----------------------------------------------------------
    def _contexts(self, day: int) -> list[AgentContext]:
        info = self.info.state
        out: list[AgentContext] = []
        for p in self.population.people:
            s = self.shadow.state[p.id]
            out.append(AgentContext(
                person=p, day=day,
                symptom_level=s.symptom_level,
                symptom_text=s.symptom_text(),
                household=self.shadow.household_situation(p.id),
                info=info,
            ))
        return out

    async def _decide(self, day: int, contexts: list[AgentContext]) -> list[BehaviourDecision]:
        k = max(1, self.cfg.decision_interval_days)
        if self._last_decisions is not None and (day - 1) % k != 0:
            return self._last_decisions
        decisions = await self.behaviour.decide_all(contexts)
        self._last_decisions = decisions
        return decisions

    async def run(self, verbose: bool = True) -> RunResult:
        cfg = self.cfg
        t0 = time.time()
        if verbose:
            print(f"[{cfg.name}] {cfg.describe()}")
            print(f"  R0 (behaviour-free) = {self.seir.behaviour_free_rt():.2f}   "
                  f"q = {self.seir.q:.5f}")
            print("-" * 92)

        for day in range(1, cfg.days + 1):
            contexts = self._contexts(day)
            decisions = await self._decide(day, contexts)

            agg = aggregate(
                self.population.people, decisions,
                policy=self.info.state.policy,
                apply_policy_cap=cfg.policy_enabled,
            )
            lam = self.seir.force_of_infection(agg.behaviour)
            rt = self.seir.effective_reproduction_number(agg.behaviour)
            flows = self.seir.step_day(agg.behaviour)

            isolating = {p.id: d.is_isolating
                         for p, d in zip(self.population.people, decisions)}
            self.shadow.step(lam, isolating)
            if cfg.shadow_resync_days and day % cfg.shadow_resync_days == 0:
                self.shadow.resync(self.seir.prevalence())

            mean_stigma = float(np.mean([p.c("stigma_sensitivity")
                                         for p in self.population.people]))
            info_state = self.info.update(
                new_symptomatic=flows["new_symptomatic"],
                symptomatic_fraction=cfg.disease.symptomatic_fraction,
                mean_stigma=mean_stigma,
                prevalence=float(self.seir.state.I.sum() / self.seir.N.sum()),
                day=day,
            )

            record = self._record(day, flows, agg, rt, info_state)
            self.result.history.append(record)
            if cfg.log_decisions:
                self._log_decisions(day, decisions)
            if verbose and (day % cfg.print_every == 0 or day == 1):
                self._print(record)

        self.result.meta = {
            "runtime_seconds": round(time.time() - t0, 2),
            "R0_behaviour_free": round(self.seir.behaviour_free_rt(), 4),
            "q": self.seir.q,
            "attack_rate": round(self.seir.attack_rate(), 4),
            "peak_infectious": max(self.result.series("infectious")),
            "peak_day": int(np.argmax(self.result.series("infectious")) + 1),
            "cache_stats": getattr(getattr(self.behaviour, "cache", None), "stats",
                                   lambda: {})(),
            "llm_errors": getattr(self.behaviour, "n_errors", 0),
            "population_summary": self.population.summary(),
        }
        if verbose:
            print("-" * 92)
            print(f"  attack rate {self.result.meta['attack_rate']:.1%}   "
                  f"peak I {self.result.meta['peak_infectious']:,.0f} on day "
                  f"{self.result.meta['peak_day']}   "
                  f"({self.result.meta['runtime_seconds']}s)")
            if self.result.meta["cache_stats"]:
                print(f"  cache: {self.result.meta['cache_stats']}")
        return self.result

    # -- bookkeeping -------------------------------------------------------
    def _record(self, day, flows, agg, rt, info_state) -> dict:
        S, E, I, R = self.seir.state.totals()
        rec = {
            "day": day,
            "susceptible": S, "exposed": E, "infectious": I, "recovered": R,
            "new_infections": flows["new_infections"],
            "new_symptomatic": flows["new_symptomatic"],
            "cumulative_infections": flows["cumulative_infections"],
            "reported_cases": info_state.reported_new_cases,
            "reported_per_100k": info_state.reported_per_100k,
            "trend": info_state.trend,
            "policy": info_state.policy,
            "Rt": rt,
            "mobility_index": mobility_index(agg),
        }
        rec.update({k: v for k, v in agg.population.items()})
        for key in STRATUM_KEYS:
            st = agg.per_stratum.get(key)
            if st:
                rec[f"{key}__phi_work"] = st["phi_work"]
                rec[f"{key}__isolating"] = st["isolating"]
                rec[f"{key}__mask"] = st["mask"]
        gi = {k: i for i, k in enumerate(STRATUM_KEYS)}
        for key, i in gi.items():
            rec[f"{key}__I"] = float(self.seir.state.I[i])
        return rec

    def _log_decisions(self, day: int, decisions: list[BehaviourDecision]) -> None:
        for p, d in zip(self.population.people, decisions):
            row = {"day": day, "agent_id": p.id, "name": p.name,
                   "stratum": p.stratum, "age": p.age,
                   "health": self.shadow.state[p.id].health.value,
                   "symptom": self.shadow.state[p.id].symptom_level}
            row.update(d.as_row())
            self.result.decisions.append(row)

    def _print(self, r: dict) -> None:
        print(f"Day {r['day']:3d} | I={r['infectious']:>11,.0f} "
              f"R={r['recovered']:>11,.0f} | Rt={r['Rt']:4.2f} | "
              f"work={r['phi_work']:.2f} social={r['phi_social']:.2f} "
              f"mask={r['mask']:.2f} iso={r['isolating']:.2f} | "
              f"reported={r['reported_per_100k']:7.1f}/100k | {r['policy']}")

    # -- output ------------------------------------------------------------
    def save(self, out_dir: str) -> dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        name = self.cfg.name
        paths: dict[str, str] = {}

        summary = os.path.join(out_dir, f"timeseries_{name}.csv")
        if self.result.history:
            with open(summary, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.result.history[0]))
                w.writeheader()
                w.writerows(self.result.history)
            paths["timeseries"] = summary

        if self.result.decisions:
            dpath = os.path.join(out_dir, f"decisions_{name}.csv")
            with open(dpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.result.decisions[0]))
                w.writeheader()
                w.writerows(self.result.decisions)
            paths["decisions"] = dpath

        mpath = os.path.join(out_dir, f"meta_{name}.json")
        with open(mpath, "w") as f:
            json.dump({"config": self.result.config, "meta": self.result.meta},
                      f, indent=2, default=str)
        paths["meta"] = mpath

        epath = os.path.join(out_dir, "PARAMETER_PROVENANCE.md")
        if not os.path.exists(epath):
            with open(epath, "w") as f:
                f.write("# Parameter provenance\n\nEvery behavioural parameter "
                        "used by ThaiGABM-SEIR, with its evidential status.\n"
                        "`assumption` rows are ours and must be sensitivity-"
                        "tested before any substantive claim.\n\n")
                f.write(evidence_table() + "\n")
        paths["provenance"] = epath
        return paths
