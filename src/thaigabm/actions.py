"""The behavioural action space shared by every behaviour model.

Guideline section 3: the LLM controls *behaviour*, never biology. Its entire
output surface is this struct. Anything epidemiological (beta, sigma, gamma,
case counts) is computed downstream by the mechanistic model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict


@dataclass
class BehaviourDecision:
    """One agent's plan for one simulated day."""

    go_to_work: bool = True
    social_activity: bool = False
    visit_family: bool = False
    wear_mask: bool = False
    seek_healthcare: bool = False
    reason: str = ""
    source: str = "llm"          # llm | cache | rule | fallback
    raw: str = ""

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d

    @property
    def is_isolating(self) -> bool:
        return not (self.go_to_work or self.social_activity or self.visit_family)

    @property
    def distancing(self) -> bool:
        """Proxy for the survey item 'practised social distancing'."""
        return not self.social_activity or self.wear_mask


SCHEMA_HINT = (
    'Respond with ONE line of compact JSON and nothing else:\n'
    '{"go_to_work": true|false, "social_activity": true|false, '
    '"visit_family": true|false, "wear_mask": true|false, '
    '"seek_healthcare": true|false, "reason": "<one short first-person sentence>"}'
)

_TRUE = {"true", "yes", "y", "1"}


def _as_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in _TRUE
    return default


def parse_decision(raw: str, fallback: "BehaviourDecision | None" = None) -> BehaviourDecision:
    """Best-effort parse of a model response into a :class:`BehaviourDecision`.

    LLMs drift: fenced code, prose preambles, ``True`` instead of ``true``. We
    recover the first JSON object and coerce; if that fails we fall back to the
    supplied rule-based decision so a parse failure degrades to a *defensible*
    behaviour rather than a silent "go to work".
    """
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidate = match.group(0)
        for attempt in (candidate, candidate.replace("True", "true").replace("False", "false")):
            try:
                data = json.loads(attempt)
            except Exception:
                continue
            return BehaviourDecision(
                go_to_work=_as_bool(data.get("go_to_work"), True),
                social_activity=_as_bool(data.get("social_activity")),
                visit_family=_as_bool(data.get("visit_family")),
                wear_mask=_as_bool(data.get("wear_mask")),
                seek_healthcare=_as_bool(data.get("seek_healthcare")),
                reason=str(data.get("reason", "")).strip()[:240] or "(no reason given)",
                source="llm",
                raw=text[:400],
            )
    if fallback is not None:
        out = BehaviourDecision(**{**asdict(fallback), "source": "fallback"})
        out.reason = f"(unparsed LLM output; rule fallback) {fallback.reason}"
        out.raw = text[:400]
        return out
    return BehaviourDecision(reason="(unparsed)", source="fallback", raw=text[:400])
