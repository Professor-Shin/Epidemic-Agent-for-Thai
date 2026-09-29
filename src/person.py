from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HealthState(str, Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"


class Symptom(str, Enum):
    NORMAL = "feels normal"
    LIGHT_COUGH = "has a light cough"
    FEVER = "has a fever and cough"


@dataclass
class Person:
    """A single resident of the simulated town.

    Fields mirror the individual-level information the paper feeds into each
    generative agent: identity, age, Big Five personality traits, and dynamic
    disease state expressed as ``infection_day`` (0 == just infected).
    """

    id: int
    name: str
    age: int
    traits: dict
    health: HealthState = HealthState.SUSCEPTIBLE
    infection_day: Optional[int] = None

    @property
    def symptom(self) -> Symptom:
        if self.health != HealthState.INFECTED or self.infection_day is None:
            return Symptom.NORMAL
        d = self.infection_day
        if d in (1, 2):
            return Symptom.NORMAL
        if d in (3, 6):
            return Symptom.LIGHT_COUGH
        if d in (4, 5):
            return Symptom.FEVER
        return Symptom.NORMAL

    @property
    def is_infectious(self) -> bool:
        return (
            self.health == HealthState.INFECTED
            and self.infection_day is not None
            and 1 <= self.infection_day <= 6
        )

    def trait_description(self) -> str:
        return "; ".join(f"{k}: {v}" for k, v in self.traits.items())
