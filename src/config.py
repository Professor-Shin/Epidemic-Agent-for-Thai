from dataclasses import dataclass, asdict
from typing import Literal

Condition = Literal["base", "self", "full"]


@dataclass
class SimulationConfig:
    """Runtime configuration for a single simulation run.

    The three conditions mirror the experimental design in the paper:
      - ``base``  : agents receive only their persona (no health or news info).
      - ``self``  : agents also receive their own symptom status each day.
      - ``full``  : agents additionally read a "news bulletin" with yesterday's
                    percentage of new cases in Dewberry Hollow.
    """

    condition: Condition = "full"
    population: int = 20
    days: int = 30
    contact_rate: int = 5
    infectivity: float = 0.1
    initial_infected: int = 1
    model: str = "gemini-2.0-flash"
    output_dir: str = "results"
    seed: int = 42
    mock: bool = False
    virus_name: str = "Catasat"
    town_name: str = "Dewberry Hollow"

    def to_dict(self) -> dict:
        return asdict(self)
