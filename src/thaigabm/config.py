"""Run configuration and the experimental arms A-E from the project guideline."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from src.thaigabm.evidence import MASK_EFFICACY, DiseaseParams
from src.thaigabm.prompts import ArmSpec


@dataclass
class RunConfig:
    name: str = "D_thai_gabm"
    arm: ArmSpec = field(default_factory=ArmSpec)
    behaviour: str = "llm"          # static | rule | llm | mock

    # population / epidemiology
    n_agents: int = 200
    total_population: int = 70_000_000
    days: int = 180
    initial_infected: int = 500
    disease: DiseaseParams = field(default_factory=DiseaseParams)
    mask_efficacy: float = MASK_EFFICACY.value
    family_visit_boost: float = 2.0

    # information environment
    policy_enabled: bool = True
    reporting_delay_days: int = 2
    stigma_suppression: float = 0.25

    # LLM / cost control
    model: str = "gemini-2.0-flash"
    cache_mode: str = "sample"      # sample | first | off
    concurrency: int = 8
    decision_interval_days: int = 1

    # bookkeeping
    seed: int = 42
    log_decisions: bool = True
    print_every: int = 5
    shadow_resync_days: int = 7

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arm"] = asdict(self.arm)
        d["disease"] = asdict(self.disease)
        return d

    def describe(self) -> str:
        a = self.arm
        bits = [f"behaviour={self.behaviour}",
                f"context={'thai' if a.thai_context else 'generic'}",
                f"n={self.n_agents}", f"days={self.days}",
                f"R0={self.disease.r0}", f"policy={'on' if self.policy_enabled else 'off'}"]
        if self.behaviour in ("llm", "mock"):
            bits.append(f"model={self.model if self.behaviour == 'llm' else 'mock'}")
            bits.append(f"cache={self.cache_mode}")
        if a.ablate:
            bits.append(f"ablate={'+'.join(a.ablate)}")
        if a.language != "en":
            bits.append(f"lang={a.language}")
        return "  ".join(bits)

    # -- behaviour model factory ------------------------------------------
    def build_behaviour(self, sim):
        from src.thaigabm.behaviour import (
            LLMBehaviour, MockLLMBehaviour, RuleBasedBehaviour, StaticBehaviour,
        )
        from src.thaigabm.cache import DecisionCache

        rng = random.Random(self.seed + 7)
        if self.behaviour == "static":
            return StaticBehaviour()
        if self.behaviour == "rule":
            return RuleBasedBehaviour(rng)
        cache = DecisionCache(mode=self.cache_mode, rng=random.Random(self.seed + 11))
        if self.behaviour == "mock":
            return MockLLMBehaviour(self.arm, self.total_population, rng,
                                    cache=DecisionCache(mode="off"))
        return LLMBehaviour(
            arm=self.arm, model=self.model,
            total_population=self.total_population,
            cache=cache, concurrency=self.concurrency, rng=rng,
        )


# --------------------------------------------------------------------------
# The five arms of the guideline's experimental design (section 10).
# Each is a function so that CLI overrides can be applied on top.
# --------------------------------------------------------------------------
def _cfg(defaults: dict, kw: dict) -> RunConfig:
    """Build a RunConfig from arm defaults, letting caller keyword args win."""
    merged = {**defaults, **kw}
    return RunConfig(**merged)


def arm_A_classical(**kw) -> RunConfig:
    """Classical SEIR: constant beta, no behavioural response at all."""
    return _cfg(dict(
        name="A_classical_seir", behaviour="static", policy_enabled=False,
        arm=ArmSpec(thai_context=False, include_self_health=False,
                    include_household=False, include_community_news=False,
                    include_policy=False),
    ), kw)


def arm_B_rule(**kw) -> RunConfig:
    """Rule-based behavioural SEIR: beta(t) responds to reported cases."""
    return _cfg(dict(name="B_rule_based", behaviour="rule"), kw)


def arm_C_generic_gabm(**kw) -> RunConfig:
    """Generic GABM-SEIR: LLM agents with the source paper's thin personas."""
    return _cfg(dict(name="C_generic_gabm", behaviour="llm",
                     arm=ArmSpec(thai_context=False, include_household=False)), kw)


def arm_D_thai_gabm(**kw) -> RunConfig:
    """Thai-context GABM-SEIR: full empirical behavioural parameterisation."""
    return _cfg(dict(name="D_thai_gabm", behaviour="llm", arm=ArmSpec()), kw)


def arm_E_ablation(construct: str, **kw) -> RunConfig:
    """Thai GABM with one behavioural construct removed (prompt + parameter)."""
    return _cfg(dict(name=f"E_ablate_{construct}", behaviour="llm",
                     arm=ArmSpec(ablate=(construct,))), kw)


ARMS = {
    "A": arm_A_classical,
    "B": arm_B_rule,
    "C": arm_C_generic_gabm,
    "D": arm_D_thai_gabm,
}
