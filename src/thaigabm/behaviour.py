"""Behaviour models: the pluggable "who decides" layer.

Every model implements ``decide_all`` and returns one BehaviourDecision per
agent. Swapping the model is what defines the experimental arms:

    StaticBehaviour      -> Experiment A  (classical SEIR, behaviour frozen)
    RuleBasedBehaviour   -> Experiment B  (adaptive but hand-written rules)
    LLMBehaviour         -> Experiments C / D / E (generative agents)

RuleBasedBehaviour doubles as the offline ``--mock`` model and as the fallback
when an LLM response cannot be parsed, so a degraded run is still a
scientifically defensible run rather than a silent bias toward "go to work".
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import Protocol

from src.thaigabm.actions import BehaviourDecision, parse_decision
from src.thaigabm.information import InformationState
from src.thaigabm.persona import Persona
from src.thaigabm.prompts import ArmSpec, build_prompt, build_prompt_th
from src.thaigabm.cache import DecisionCache, situation_key


@dataclass
class AgentContext:
    person: Persona
    day: int
    symptom_level: str
    symptom_text: str
    household: dict
    info: InformationState


class BehaviourModel(Protocol):
    name: str

    async def decide_all(self, contexts: list[AgentContext]) -> list[BehaviourDecision]:
        ...


# --------------------------------------------------------------------------
# A. Static behaviour -- nobody ever changes anything.
# --------------------------------------------------------------------------
class StaticBehaviour:
    """Experiment A: the classical SEIR baseline (constant beta)."""

    name = "static"

    def __init__(self, work: bool = True, social: bool = True,
                 family: bool = True, mask: bool = False):
        self._d = BehaviourDecision(
            go_to_work=work, social_activity=social, visit_family=family,
            wear_mask=mask, reason="Behaviour is held constant in this arm.",
            source="rule",
        )

    async def decide_all(self, contexts: list[AgentContext]) -> list[BehaviourDecision]:
        return [BehaviourDecision(**self._d.__dict__) for _ in contexts]


# --------------------------------------------------------------------------
# B. Rule-based adaptive behaviour.
# --------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RuleBasedBehaviour:
    """Experiment B: adaptive behaviour from an explicit utility rule.

    Uses the *same* constructs and the *same* information channels the LLM
    sees, so any difference between arms B and D is attributable to the
    reasoning mechanism rather than to the information available. This is the
    control that answers "do we actually need an LLM?".

    Calibration anchors (all checked by tests/test_behaviour_calibration.py):
      * with no epidemic news and no symptoms, life is normal -- ~95% of
        workers work, masks are rare;
      * at full salience under lockdown, mask use approaches the 97% reported
        by Pan-ngum et al. [3] and distancing the reported 94%;
      * a symptomatic informal worker still goes out far more often than a
        symptomatic office worker, which is the economic-vulnerability
        mechanism the Thai survey evidence points to.
    """

    name = "rule"

    def __init__(self, rng: random.Random | None = None, salience_scale: float = 5.0,
                 use_constructs: bool = True, arm: ArmSpec | None = None):
        self.rng = rng or random.Random(0)
        self.salience_scale = salience_scale
        self.use_constructs = use_constructs
        self.arm = arm            # None => every information channel is on

    # -- information gating -------------------------------------------------
    def _channels(self) -> dict[str, bool]:
        a = self.arm
        if a is None:
            return {"health": True, "household": True, "news": True,
                    "policy": True, "thai": True}
        return {"health": a.include_self_health, "household": a.include_household,
                "news": a.include_community_news, "policy": a.include_policy,
                "thai": a.thai_context}

    def _salience(self, info: InformationState) -> float:
        s = 1.0 - math.exp(-info.reported_per_100k / self.salience_scale)
        if info.trend == "rising":
            s = min(1.0, s * 1.25)
        elif info.trend == "falling":
            s *= 0.85
        return s

    def decide(self, ctx: AgentContext) -> BehaviourDecision:
        p, info = ctx.person, ctx.info
        ch = self._channels()
        # A generic (non-Thai) agent reasons from personality, not from the
        # Thai behavioural constructs, so those collapse to neutral.
        thai = self.use_constructs and ch["thai"]
        c = (lambda k: p.c(k)) if thai else (lambda k: 0.5)

        salience = self._salience(info) if ch["news"] else 0.0
        policy_push = ({"none": 0.0, "advisory": 0.4, "restriction": 1.0,
                        "lockdown": 1.8}[info.policy] if ch["policy"] else 0.0)

        risk = c("perceived_severity") * (0.4 + 0.6 * c("perceived_susceptibility"))
        symptom_push = ({"none": 0.0, "mild": 1.8, "severe": 4.0}[ctx.symptom_level]
                        if ch["health"] else 0.0)
        household_push = (1.0 if (ch["household"] and thai
                                  and ctx.household.get("someone_sick_at_home")) else 0.0)
        elder_push = (1.6 * c("family_obligation") * max(salience, 0.3)
                      if (ch["household"] and thai
                          and ctx.household.get("elderly_at_home")) else 0.0)
        norm_push = 1.6 * c("norm_sensitivity") * salience
        trust_push = 1.5 * c("institutional_trust") * policy_push
        econ_resist = 2.2 * c("economic_pressure") * (1.0 - p.wfh_ability)

        # Net pressure to withdraw from public life (>= 0 in normal times).
        pressure = (3.0 * risk * salience + symptom_push + household_push
                    + elder_push + norm_push + trust_push)

        # WORK: the offset -3.0 sets ~95% attendance when nothing is happening.
        p_stay_work = _sigmoid(pressure - econ_resist - 3.0)
        go_to_work = self.rng.random() > p_stay_work

        # SOCIAL: discretionary, so it collapses long before work does.
        base_social = 0.30 + 0.50 * c("social_preference")
        p_social = base_social * _sigmoid(2.0 - 1.2 * pressure)
        social = self.rng.random() < p_social

        # FAMILY VISIT: family obligation pulls people towards relatives in
        # normal times and pushes them away once elders are at risk.
        base_family = 0.10 + 0.30 * c("family_obligation")
        p_family = base_family * _sigmoid(1.5 - 1.0 * pressure)
        if ch["household"] and thai and ctx.household.get("elderly_at_home"):
            p_family *= (1.0 - 0.6 * c("family_obligation") * salience)
        family = self.rng.random() < p_family

        # MASK: essentially unused until the outbreak becomes salient.
        p_mask = _sigmoid(-3.2 + salience * (3.0 + 1.8 * c("norm_sensitivity")
                                             + 1.5 * risk + 0.8 * c("health_literacy"))
                          + 1.0 * policy_push)
        mask = self.rng.random() < p_mask

        # CARE-SEEKING: stigma suppresses it, trust encourages it.
        p_care = 0.0
        if ch["health"] and ctx.symptom_level != "none":
            base = 0.25 if ctx.symptom_level == "mild" else 0.70
            p_care = base * (1.0 - 0.5 * c("stigma_sensitivity")) \
                * (0.5 + 0.5 * c("institutional_trust"))
        care = self.rng.random() < p_care

        reason = (f"rule: risk={risk:.2f} salience={salience:.2f} "
                  f"pressure={pressure:.2f} econ_resist={econ_resist:.2f} "
                  f"p_stay={p_stay_work:.2f}")
        return BehaviourDecision(
            go_to_work=go_to_work, social_activity=social, visit_family=family,
            wear_mask=mask, seek_healthcare=care, reason=reason, source="rule",
        )

    async def decide_all(self, contexts: list[AgentContext]) -> list[BehaviourDecision]:
        return [self.decide(c) for c in contexts]


# --------------------------------------------------------------------------
# C / D / E. LLM behaviour via Google ADK.
# --------------------------------------------------------------------------
class LLMBehaviour:
    """Generative behavioural layer backed by an ADK ``LlmAgent``.

    Cost controls, all auditable in the run's ``cache_stats``:
      * situation-keyed decision cache (see cache.py),
      * bounded concurrency,
      * optional re-decision interval (agents keep yesterday's plan for k days).
    """

    name = "llm"

    def __init__(
        self,
        arm: ArmSpec,
        model: str = "gemini-2.0-flash",
        total_population: int = 70_000_000,
        cache: DecisionCache | None = None,
        concurrency: int = 8,
        rng: random.Random | None = None,
        app_name: str = "thaigabm",
    ):
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner

        self.arm = arm
        self.model = model
        self.total_population = total_population
        self.cache = cache or DecisionCache(rng=rng or random.Random(0))
        self.fallback = RuleBasedBehaviour(rng or random.Random(1))
        self.sem = asyncio.Semaphore(concurrency)
        self.app_name = app_name
        self.n_errors = 0
        # In deterministic cache mode, agents in the *same* batch that share a
        # situation await one call instead of firing N identical requests. In
        # "sample" mode duplicates are allowed through on purpose: that is how
        # the response distribution for a situation gets built.
        self._inflight: dict[tuple, asyncio.Future] = {}

        self.agent = LlmAgent(
            name="thai_resident",
            model=model,
            instruction=arm.system_instruction(),
        )
        self.runner = InMemoryRunner(agent=self.agent, app_name=app_name)

    @property
    def arm_signature(self) -> str:
        a = self.arm
        return (f"{'thai' if a.thai_context else 'generic'}|{a.language}|"
                f"h{int(a.include_self_health)}{int(a.include_household)}"
                f"{int(a.include_community_news)}{int(a.include_policy)}|"
                f"abl:{','.join(sorted(a.ablate))}")

    def build(self, ctx: AgentContext) -> str:
        builder = build_prompt_th if self.arm.language == "th" else build_prompt
        return builder(ctx.person, self.arm, ctx.day, ctx.symptom_text,
                       ctx.household, ctx.info, self.total_population)

    async def _call(self, prompt: str, uid: str, sid: str) -> str:
        from google.genai import types

        await self.runner.session_service.create_session(
            app_name=self.app_name, user_id=uid, session_id=sid
        )
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        chunks: list[str] = []
        async for event in self.runner.run_async(
            user_id=uid, session_id=sid, new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        chunks.append(text)
        return "".join(chunks)

    async def _decide_one(self, ctx: AgentContext) -> BehaviourDecision:
        key = situation_key(ctx.person, self.arm_signature, ctx.symptom_level,
                            ctx.household, ctx.info)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        dedupe = self.cache.mode == "first"
        if dedupe and key in self._inflight:
            shared = await asyncio.shield(self._inflight[key])
            self.cache.hits += 1
            return BehaviourDecision(**{**shared.__dict__, "source": "cache"})

        self.cache.note_miss()
        future: asyncio.Future | None = None
        if dedupe:
            future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future

        rule_default = self.fallback.decide(ctx)
        prompt = self.build(ctx)
        result = rule_default
        try:
            async with self.sem:
                for attempt in range(2):
                    try:
                        raw = await self._call(
                            prompt,
                            uid=f"agent_{ctx.person.id}",
                            sid=f"d{ctx.day}_a{ctx.person.id}_{attempt}",
                        )
                        decision = parse_decision(raw, fallback=rule_default)
                        if decision.source == "llm":
                            self.cache.put(key, decision)
                        result = decision
                        break
                    except Exception as exc:  # network / quota / transport
                        self.n_errors += 1
                        if attempt == 0:
                            await asyncio.sleep(1.5)
                            continue
                        result = BehaviourDecision(**{**rule_default.__dict__,
                                                      "source": "fallback"})
                        result.reason = (f"(LLM error {type(exc).__name__}; "
                                         "rule fallback)")
        finally:
            if future is not None:
                self._inflight.pop(key, None)
                if not future.done():
                    future.set_result(result)
        return result

    async def decide_all(self, contexts: list[AgentContext]) -> list[BehaviourDecision]:
        return list(await asyncio.gather(*(self._decide_one(c) for c in contexts)))


class MockLLMBehaviour(RuleBasedBehaviour):
    """Offline stand-in for LLMBehaviour so the full pipeline runs with no API key.

    It is the rule model plus prompt construction, so ``--mock`` exercises every
    code path except the network call itself (prompts are still rendered and can
    be dumped for inspection).
    """

    name = "mock-llm"

    def __init__(self, arm: ArmSpec, total_population: int,
                 rng: random.Random | None = None,
                 cache: DecisionCache | None = None):
        super().__init__(rng, arm=arm)
        self.arm = arm
        self.total_population = total_population
        self.cache = cache or DecisionCache(mode="off")
        self.n_errors = 0
        self.last_prompt: str = ""

    @property
    def arm_signature(self) -> str:
        return LLMBehaviour.arm_signature.fget(self)  # type: ignore[attr-defined]

    def build(self, ctx: AgentContext) -> str:
        builder = build_prompt_th if self.arm.language == "th" else build_prompt
        return builder(ctx.person, self.arm, ctx.day, ctx.symptom_text,
                       ctx.household, ctx.info, self.total_population)

    async def decide_all(self, contexts: list[AgentContext]) -> list[BehaviourDecision]:
        out: list[BehaviourDecision] = []
        for ctx in contexts:
            # Prompts are still rendered so --mock exercises every code path
            # except the network call, and so prompts can be dumped for review.
            self.last_prompt = self.build(ctx)
            d = self.decide(ctx)
            d.source = "mock"
            out.append(d)
        return out
