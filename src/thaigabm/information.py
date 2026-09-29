"""The information environment closing the feedback loop SEIR -> agents.

Agents never see the true state. They see what a person in Thailand would
plausibly see: a *reported* case number (delayed and under-ascertained), the
policy level currently in force, and their own household situation. Stigma
suppresses reporting further -- a mechanism the qualitative Thai literature
reports as concealment of infection [4].
"""

from __future__ import annotations

from dataclasses import dataclass

POLICY_LEVELS = ("none", "advisory", "restriction", "lockdown")


@dataclass
class PolicyRegime:
    """Case-triggered policy escalation (a government reaction function).

    Deliberately mechanical: policy is an *input* to agent reasoning, not
    something the LLM decides, so that arms remain comparable.
    """

    enabled: bool = True
    advisory_per_100k: float = 1.0
    restriction_per_100k: float = 5.0
    lockdown_per_100k: float = 20.0
    min_days_between_changes: int = 7

    def level_for(self, reported_per_100k: float) -> str:
        if not self.enabled:
            return "none"
        if reported_per_100k >= self.lockdown_per_100k:
            return "lockdown"
        if reported_per_100k >= self.restriction_per_100k:
            return "restriction"
        if reported_per_100k >= self.advisory_per_100k:
            return "advisory"
        return "none"


@dataclass
class InformationState:
    day: int = 0
    reported_new_cases: float = 0.0
    reported_per_100k: float = 0.0
    reported_cum: float = 0.0
    trend: str = "steady"              # rising | falling | steady
    policy: str = "none"
    policy_days: int = 0
    true_prevalence: float = 0.0

    def news_line(self, total_population: int) -> str:
        pct = 100.0 * self.reported_new_cases / max(total_population, 1)
        return (f"Yesterday the Ministry of Public Health reported "
                f"{self.reported_new_cases:,.0f} new cases nationwide "
                f"({self.reported_per_100k:.1f} per 100,000 people, "
                f"{pct:.3f}% of the population). Cases are {self.trend}.")


class InformationEnvironment:
    def __init__(
        self,
        total_population: float,
        ascertainment: float,
        reporting_delay_days: int = 2,
        policy: PolicyRegime | None = None,
        stigma_suppression: float = 0.25,
    ):
        self.total_population = float(total_population)
        self.ascertainment = ascertainment
        self.delay = max(0, int(reporting_delay_days))
        self.policy = policy or PolicyRegime()
        self.stigma_suppression = stigma_suppression
        self._pipeline: list[float] = []
        self._history: list[float] = []
        self.state = InformationState()
        self._days_at_level = 0

    def update(self, new_symptomatic: float, symptomatic_fraction: float,
               mean_stigma: float, prevalence: float, day: int) -> InformationState:
        """Push one day of true incidence through detection, delay and policy."""
        detectable = new_symptomatic * symptomatic_fraction
        # Stigma reduces the probability an infected person presents for testing.
        eff_ascertainment = self.ascertainment * (1.0 - self.stigma_suppression * mean_stigma)
        self._pipeline.append(detectable * eff_ascertainment)

        reported = self._pipeline[-(self.delay + 1)] if len(self._pipeline) > self.delay else 0.0
        self._history.append(reported)

        prev_week = self._history[-8:-1]
        avg_prev = sum(prev_week) / len(prev_week) if prev_week else 0.0
        if reported > avg_prev * 1.10:
            trend = "rising"
        elif reported < avg_prev * 0.90:
            trend = "falling"
        else:
            trend = "steady"

        per100k = 1e5 * reported / self.total_population
        new_level = self.policy.level_for(per100k)
        if new_level != self.state.policy and self._days_at_level < self.policy.min_days_between_changes:
            new_level = self.state.policy   # hysteresis: governments react slowly
            self._days_at_level += 1
        elif new_level != self.state.policy:
            self._days_at_level = 0
        else:
            self._days_at_level += 1

        self.state = InformationState(
            day=day,
            reported_new_cases=reported,
            reported_per_100k=per100k,
            reported_cum=self.state.reported_cum + reported,
            trend=trend,
            policy=new_level,
            policy_days=self._days_at_level,
            true_prevalence=prevalence,
        )
        return self.state


POLICY_PROMPT = {
    "none": "There are no special government measures in force.",
    "advisory": ("The government has issued an advisory asking people to avoid "
                 "crowds and wear masks. Nothing is mandatory."),
    "restriction": ("The government has restricted gatherings; venues close early "
                    "and masks are required indoors. Enforcement is uneven."),
    "lockdown": ("The government has declared a lockdown: non-essential "
                 "workplaces are closed and movement is restricted, though "
                 "informal work and markets continue in practice."),
}

# How strongly each policy level *mechanically* constrains a setting, on top of
# whatever the agents choose. Informal work is only weakly constrained: this is
# a documented reality of enforcement in Thailand, and is what stops the model
# from producing an implausibly obedient population.
POLICY_HARD_CAP: dict[str, dict[str, float]] = {
    "none":        {"work": 1.00, "social": 1.00, "family_visit": 1.00},
    "advisory":    {"work": 1.00, "social": 0.95, "family_visit": 1.00},
    "restriction": {"work": 0.90, "social": 0.60, "family_visit": 0.85},
    "lockdown":    {"work": 0.70, "social": 0.30, "family_visit": 0.55},
}
