"""Random persona generation.

The paper samples names from the ``names-dataset`` package, ages uniformly from
[18, 64], and Big Five traits with a 50/50 chance of the positive or negative
pole per trait. We replicate that scheme with a bundled name list so the
project has no non-Python data dependency.
"""

import random

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Maya", "Noah", "Olivia", "Paul",
    "Quinn", "Rita", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zack", "Amy", "Ben", "Chloe", "Dan", "Emma", "Finn",
    "Gina", "Hugo", "Ivy", "James", "Kira", "Leo", "Mia", "Nate",
    "Opal", "Pete", "Ruby", "Sean", "Tina", "Uri", "Vera", "Will",
    "Xena", "Yuri", "Zoe", "Anna", "Brian", "Cara", "Derek", "Elena",
    "Fiona", "Gabe", "Hana", "Isaac", "Jill", "Karl", "Lena", "Marco",
    "Nora", "Oscar", "Priya", "Roman", "Sara", "Tom", "Ursula", "Vince",
]

LAST_NAMES = [
    "Smith", "Johnson", "Brown", "Taylor", "Anderson", "Wilson", "Moore",
    "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
    "Green", "Adams", "Baker", "Carter", "Mitchell", "Roberts", "Turner",
]

BIG_FIVE = {
    "openness": (
        "curious and open to new experiences",
        "traditional and prefers familiar routines",
    ),
    "conscientiousness": (
        "organized, disciplined, and careful about rules",
        "spontaneous, flexible, and casual about rules",
    ),
    "extraversion": (
        "outgoing, sociable, and energized by being around people",
        "reserved, introverted, and prefers time alone",
    ),
    "agreeableness": (
        "warm, cooperative, and considerate of others' well-being",
        "competitive, skeptical, and independent-minded",
    ),
    "neuroticism": (
        "sensitive to stress and easily worried about health risks",
        "calm, emotionally stable, and slow to worry",
    ),
}


def sample_name(rng: random.Random, used: set) -> str:
    for _ in range(1000):
        candidate = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    fallback = f"Person_{len(used)}"
    used.add(fallback)
    return fallback


def sample_traits(rng: random.Random) -> dict:
    return {
        trait: descs[0] if rng.random() < 0.5 else descs[1]
        for trait, descs in BIG_FIVE.items()
    }


def sample_age(rng: random.Random) -> int:
    return rng.randint(18, 64)
