"""Disease progression rules from the paper.

- Total infectious period: 6 days.
- Day 0            : just infected today, asymptomatic and not yet infectious.
- Day 1-2          : asymptomatic ("feels normal"), infectious.
- Day 3            : light cough, infectious.
- Day 4-5          : fever and cough, infectious.
- Day 6            : light cough, infectious.
- Day 7+           : recovered and immune.
"""

from src.person import HealthState, Person

DISEASE_DURATION = 6


def infect(person: Person) -> None:
    """Move a susceptible person into the infected compartment."""
    if person.health == HealthState.SUSCEPTIBLE:
        person.health = HealthState.INFECTED
        person.infection_day = 0


def advance_day(person: Person) -> None:
    """Advance the clock on an infection by one day, recovering when done."""
    if person.health != HealthState.INFECTED or person.infection_day is None:
        return
    person.infection_day += 1
    if person.infection_day > DISEASE_DURATION:
        person.health = HealthState.RECOVERED
        person.infection_day = None
