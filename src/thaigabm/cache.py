"""Decision cache -- the main answer to the source paper's cost problem.

Williams et al. report >90 hours for a 1,000-agent run because every agent is
queried every day. But an agent's decision is a function of a small,
*discrete* situation: who am I (roughly), how sick am I, what is my household
like, how bad is the news, what is the policy. Two informal workers with the
same coarse profile in the same situation do not need two API calls.

We therefore key decisions on a coarsened situation vector and reuse them.
Two reuse modes:

  "sample" (default) -- keep every distinct response seen for a key and draw one
      at random. Preserves the LLM's behavioural stochasticity, which the
      Epi-LLM work argues is part of what the LLM contributes.
  "first"  -- deterministic: always reuse the first response. Lowest cost and
      lowest variance; good for internal-validity runs.

Cache statistics are reported so the compute saving is auditable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.thaigabm.actions import BehaviourDecision
from src.thaigabm.information import InformationState
from src.thaigabm.persona import Persona

CACHED_CONSTRUCTS = (
    "economic_pressure", "family_obligation", "perceived_severity",
    "perceived_susceptibility", "institutional_trust", "norm_sensitivity",
    "stigma_sensitivity",
)


def _bucket(v: float, n: int = 3) -> int:
    return min(int(v * n), n - 1)


def _case_bucket(per100k: float) -> int:
    for i, edge in enumerate((0.01, 0.5, 2.0, 5.0, 20.0, 50.0)):
        if per100k < edge:
            return i
    return 6


def situation_key(
    p: Persona,
    arm_signature: str,
    symptom_level: str,
    household: dict,
    info: InformationState,
) -> tuple:
    return (
        arm_signature,
        p.stratum,
        _bucket(p.wfh_ability, 2),
        tuple(_bucket(p.c(c)) for c in CACHED_CONSTRUCTS),
        symptom_level,
        bool(household.get("someone_sick_at_home")),
        bool(household.get("elderly_at_home")),
        _case_bucket(info.reported_per_100k),
        info.trend,
        info.policy,
    )


@dataclass
class DecisionCache:
    mode: str = "sample"          # sample | first | off
    rng: random.Random = field(default_factory=random.Random)
    store: dict[tuple, list[BehaviourDecision]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    max_per_key: int = 4

    def get(self, key: tuple) -> BehaviourDecision | None:
        if self.mode == "off":
            return None
        bucket = self.store.get(key)
        if not bucket:
            return None
        # Keep collecting a few distinct samples per key before reusing, so the
        # cached distribution is not one lucky draw.
        if self.mode == "sample" and len(bucket) < self.max_per_key:
            if self.rng.random() < 0.5:
                return None
        self.hits += 1
        chosen = bucket[0] if self.mode == "first" else self.rng.choice(bucket)
        out = BehaviourDecision(**{**chosen.__dict__, "source": "cache"})
        return out

    def put(self, key: tuple, decision: BehaviourDecision) -> None:
        if self.mode == "off":
            return
        bucket = self.store.setdefault(key, [])
        if len(bucket) < self.max_per_key:
            bucket.append(decision)

    def note_miss(self) -> None:
        self.misses += 1

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "llm_calls": self.misses,
            "cache_hits": self.hits,
            "decisions": total,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "distinct_situations": len(self.store),
        }
