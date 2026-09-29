"""Exercise the real LLMBehaviour code path against a stubbed ADK.

google-adk may not be installed, and even when it is we do not want tests
hitting the network. These stubs stand in for ``google.adk`` and
``google.genai`` so the prompt -> call -> parse -> cache path is genuinely
executed, including the retry and fallback branches.
"""

from __future__ import annotations

import asyncio
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Session:
    async def create_session(self, **kw):
        return types.SimpleNamespace(**kw)


class _Part:
    def __init__(self, text):
        self.text = text


class _Runner:
    """Replays a scripted list of responses; an Exception entry raises."""

    script: list = []
    calls: list = []

    def __init__(self, agent=None, app_name=None):
        self.session_service = _Session()

    async def run_async(self, user_id, session_id, new_message):
        _Runner.calls.append(new_message)
        item = _Runner.script.pop(0) if _Runner.script else '{"go_to_work": true, "reason": "ok"}'
        if isinstance(item, Exception):
            raise item
        yield types.SimpleNamespace(
            content=types.SimpleNamespace(parts=[_Part(item)]))


def _install_stubs():
    google = sys.modules.setdefault("google", types.ModuleType("google"))
    adk = types.ModuleType("google.adk")
    agents = types.ModuleType("google.adk.agents")
    runners = types.ModuleType("google.adk.runners")
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")

    class LlmAgent:
        def __init__(self, **kw):
            self.kw = kw

    agents.LlmAgent = LlmAgent
    runners.InMemoryRunner = _Runner
    gtypes.Content = lambda role, parts: types.SimpleNamespace(role=role, parts=parts)
    gtypes.Part = lambda text: _Part(text)
    genai.types = gtypes
    google.adk = adk
    adk.agents, adk.runners = agents, runners
    sys.modules.update({"google.adk": adk, "google.adk.agents": agents,
                        "google.adk.runners": runners, "google.genai": genai,
                        "google.genai.types": gtypes})


_install_stubs()

from src.thaigabm.behaviour import AgentContext, LLMBehaviour  # noqa: E402
from src.thaigabm.cache import DecisionCache  # noqa: E402
from src.thaigabm.information import InformationState  # noqa: E402
from src.thaigabm.persona import Population  # noqa: E402
from src.thaigabm.prompts import ArmSpec  # noqa: E402


def _contexts(n=3):
    pop = Population(n, 70_000_000, random.Random(0))
    info = InformationState(day=5, reported_per_100k=4.0, trend="rising",
                            policy="advisory")
    return [AgentContext(person=p, day=5, symptom_level="none",
                         symptom_text="feel completely normal",
                         household={"household_size": p.household_size,
                                    "someone_sick_at_home": False,
                                    "n_sick_at_home": 0,
                                    "elderly_at_home": p.elderly_in_household,
                                    "children_at_home": False},
                         info=info)
            for p in pop.people]


def _model(**kw):
    _Runner.script, _Runner.calls = [], []
    return LLMBehaviour(arm=ArmSpec(), total_population=70_000_000,
                        cache=DecisionCache(mode="off"),
                        rng=random.Random(0), **kw)


def test_llm_response_is_parsed_into_decisions():
    m = _model()
    ctxs = _contexts(3)
    _Runner.script = ['{"go_to_work": false, "social_activity": false, '
                      '"visit_family": false, "wear_mask": true, '
                      '"seek_healthcare": false, "reason": "Cases are rising."}'] * 3
    out = asyncio.run(m.decide_all(ctxs))
    assert len(out) == 3
    assert all(d.source == "llm" and d.go_to_work is False and d.wear_mask
               for d in out)


def test_prompt_actually_reaches_the_model():
    m = _model()
    asyncio.run(m.decide_all(_contexts(1)))
    sent = _Runner.calls[0].parts[0].text
    assert "living in Thailand" in sent
    assert "financial pressure to keep earning" in sent
    assert "Respond with ONE line of compact JSON" in sent


def test_transport_error_retries_then_falls_back_to_the_rule():
    m = _model()
    _Runner.script = [RuntimeError("503"), RuntimeError("503")]
    out = asyncio.run(m.decide_all(_contexts(1)))
    assert out[0].source == "fallback"
    assert m.n_errors == 2


def test_retry_succeeds_after_one_failure():
    m = _model()
    _Runner.script = [RuntimeError("timeout"),
                      '{"go_to_work": true, "reason": "fine"}']
    out = asyncio.run(m.decide_all(_contexts(1)))
    assert out[0].source == "llm" and m.n_errors == 1


def test_cache_cuts_api_calls_for_repeated_situations():
    _Runner.script, _Runner.calls = [], []
    m = LLMBehaviour(arm=ArmSpec(), total_population=70_000_000,
                     cache=DecisionCache(mode="first", rng=random.Random(0)),
                     rng=random.Random(0))
    ctxs = _contexts(1) * 6      # identical situation, six times
    asyncio.run(m.decide_all(ctxs))
    stats = m.cache.stats()
    assert stats["llm_calls"] < 6 and stats["cache_hits"] > 0
    assert stats["llm_calls"] == len(_Runner.calls)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as exc:
            bad += 1; print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if bad else 0)
