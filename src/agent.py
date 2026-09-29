"""Google ADK reasoning agent for the generative-agent-based model.

Each simulated resident's daily decision ("go to work" vs. "stay home") is made
by an :class:`~google.adk.agents.LlmAgent`. We use a single ADK agent and give
each call a fresh session, so the LLM behaves as a stateless reasoner whose
entire context comes from the prompt we build for the current day. This mirrors
how the paper feeds "relevant memories" through the prompt every step.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Optional

from src.config import Condition
from src.person import Person, Symptom

SYSTEM_INSTRUCTION = (
    "You role-play as a specific resident of the town of Dewberry Hollow. "
    "Each turn the user gives you your personal profile and the current "
    "context. Based ONLY on that information, decide whether to go to work "
    "today. Reason like a real person: consider your job, personality, how "
    "you physically feel, and what you know about any disease spreading in "
    "town. "
    "Respond STRICTLY as compact JSON on a single line with two fields: "
    '"decision" (either "go" or "stay") and "reason" (one short first-person '
    "sentence). Do not add markdown, code fences, or any other text."
)


@dataclass
class Decision:
    go_to_work: bool
    reason: str


def build_prompt(
    person: Person,
    condition: Condition,
    daily_case_pct: float,
    virus_name: str,
    town_name: str,
) -> str:
    """Assemble the per-agent daily prompt for the requested experimental condition."""
    lines: list[str] = [
        f"You are {person.name}, {person.age} years old, "
        f"and you live in {town_name}.",
        f"Your personality: {person.trait_description()}.",
        f"{person.name} goes to work to earn money to support "
        f"{person.name}'s self.",
    ]

    if condition in ("self", "full"):
        lines.append(f"Today, physically, {person.name} {person.symptom.value}.")

    if condition == "full":
        lines.append(
            f"{person.name} knows about the {virus_name} virus spreading across "
            "the country. It is an infectious disease that spreads from human "
            "to human contact via an airborne virus. The deadliness of the "
            "virus is unknown. Scientists are warning about a potential "
            f"epidemic. {person.name} checks the newspaper and finds that "
            f"{daily_case_pct:.2f}% of {town_name}'s population caught new "
            f"infections of the {virus_name} virus yesterday."
        )

    lines.append(
        "Decide whether you will go to work today. "
        'Answer STRICTLY as JSON: {"decision": "go" | "stay", "reason": "..."}.'
    )
    return "\n".join(lines)


def parse_decision(raw: str) -> Decision:
    """Best-effort parse of the model's JSON response."""
    text = raw.strip()
    match = re.search(r"\{.*?\}", text, re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        dec = str(data.get("decision", "")).strip().lower()
        go = dec.startswith("go")
        reason = str(data.get("reason", "")).strip() or "(no reason given)"
        return Decision(go_to_work=go, reason=reason)
    except Exception:
        lower = raw.lower()
        stay = "stay" in lower or "isolate" in lower or "quarantine" in lower
        return Decision(go_to_work=not stay, reason=f"(unparsed) {raw.strip()[:140]}")


class DecisionAgent:
    """Google ADK LlmAgent wrapper that returns a :class:`Decision` per call."""

    def __init__(self, model: str = "gemini-2.0-flash", app_name: str = "epidemic"):
        # Import lazily so the module can be imported (e.g. in tests / mock
        # runs) without google-adk being installed.
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner

        self.app_name = app_name
        self.agent = LlmAgent(
            name="dewberry_resident",
            model=model,
            instruction=SYSTEM_INSTRUCTION,
        )
        self.runner = InMemoryRunner(agent=self.agent, app_name=app_name)

    async def decide(
        self,
        person: Person,
        condition: Condition,
        daily_case_pct: float,
        day: int,
        virus_name: str,
        town_name: str,
    ) -> Decision:
        from google.genai import types

        user_id = f"person_{person.id}"
        session_id = f"day_{day}_person_{person.id}"

        await self.runner.session_service.create_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )

        prompt = build_prompt(person, condition, daily_case_pct, virus_name, town_name)
        message = types.Content(role="user", parts=[types.Part(text=prompt)])

        chunks: list[str] = []
        try:
            async for event in self.runner.run_async(
                user_id=user_id, session_id=session_id, new_message=message
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            chunks.append(text)
        except Exception as exc:  # pragma: no cover
            return Decision(
                go_to_work=True,
                reason=f"(LLM error: {type(exc).__name__}) defaulting to go",
            )

        return parse_decision("".join(chunks))


class MockDecisionAgent:
    """Deterministic heuristic stand-in for offline runs and tests.

    Used when ``--mock`` is passed. Applies simple, plausible rules so the
    orchestration end-to-end can be exercised without any LLM calls.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random(0)

    async def decide(
        self,
        person: Person,
        condition: Condition,
        daily_case_pct: float,
        day: int,
        virus_name: str,
        town_name: str,
    ) -> Decision:
        # Health-driven behavior (self / full conditions only)
        if condition in ("self", "full"):
            if person.symptom == Symptom.FEVER:
                return Decision(False, "I have a fever, staying home.")
            if person.symptom == Symptom.LIGHT_COUGH and self.rng.random() < 0.5:
                return Decision(False, "I have a cough, resting today.")

        # News-driven behavior (full condition only)
        if condition == "full" and daily_case_pct > 0:
            worry = "worried" in person.traits.get("neuroticism", "")
            threshold = 2.0 if worry else 8.0
            if daily_case_pct >= threshold and self.rng.random() < 0.6:
                return Decision(False, "Cases are rising, I'll self-isolate.")

        return Decision(True, "Feel fine, going to work.")
