"""Synthetic population: personas, behavioural attributes and households.

Two persona flavours exist so that Experiment C (generic GABM) and Experiment D
(Thai-grounded GABM) differ in *substance*, not just in a country label -- the
Epi-LLM 2026 finding is that a geographic label alone does not produce
culturally differentiated behaviour [2].

  GenericPersona : name, age, Big Five, job.  (the source paper's persona)
  ThaiPersona    : demographics + household structure + nine explicitly
                   parameterised behavioural constructs.

Each agent is a *representative* of a stratum, carrying a population weight, so
that a few hundred LLM calls can stand in for millions of people.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from src.thaigabm import evidence
from src.thaigabm.strata import STRATA, STRATUM_INDEX, Stratum, stratum_shares

THAI_GIVEN_NAMES = [
    "Somchai", "Somsak", "Wichai", "Anong", "Malee", "Siriporn", "Nattapong",
    "Kanya", "Prasit", "Sunisa", "Thanakorn", "Pimchanok", "Chalerm", "Duangjai",
    "Weerapong", "Nareerat", "Sarawut", "Ratchanee", "Kittisak", "Waraporn",
    "Apinya", "Boonmee", "Chaiwat", "Darika", "Ekkachai", "Fonthip", "Jirapat",
    "Kulap", "Manop", "Nipha", "Orawan", "Panya", "Rungnapha", "Suchart",
    "Tanit", "Ubon", "Wanida", "Yupin", "Chanya", "Peerapat",
]
THAI_FAMILY_NAMES = [
    "Sirisak", "Wongsawat", "Chaiyaporn", "Rattanakosin", "Suwannarat",
    "Phromsuk", "Thongchai", "Boonsri", "Nakhonsawan", "Intharat",
    "Saetang", "Kittichai", "Charoenkul", "Pattanasiri", "Mongkhon",
    "Aramwong", "Bunnag", "Chanthara", "Damrongsak", "Ekwattana",
]

GENERIC_GIVEN = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Maya", "Noah", "Olivia", "Paul",
]
GENERIC_FAMILY = [
    "Smith", "Johnson", "Brown", "Taylor", "Anderson", "Wilson", "Moore",
    "Martin", "Lee", "Clark", "Lewis", "Walker", "Young", "King",
]

BIG_FIVE = {
    "openness": ("curious and open to new experiences",
                 "traditional and prefers familiar routines"),
    "conscientiousness": ("organized, disciplined, and careful about rules",
                          "spontaneous, flexible, and casual about rules"),
    "extraversion": ("outgoing and energized by being around people",
                     "reserved and prefers time alone"),
    "agreeableness": ("warm, cooperative, considerate of others",
                      "competitive, skeptical, independent-minded"),
    "neuroticism": ("sensitive to stress and easily worried about health risks",
                    "calm, emotionally stable, and slow to worry"),
}


# --------------------------------------------------------------------------
# Beta sampling helper
# --------------------------------------------------------------------------
def _beta(rng: random.Random, mean: float, concentration: float) -> float:
    mean = min(max(mean, 0.02), 0.98)
    a = mean * concentration
    b = (1.0 - mean) * concentration
    return float(min(max(rng.betavariate(a, b), 0.0), 1.0))


@dataclass
class Household:
    id: int
    size: int
    member_ids: list[int] = field(default_factory=list)
    has_elderly: bool = False
    has_children: bool = False


@dataclass
class Persona:
    """One representative agent."""

    id: int
    name: str
    age: int
    stratum: str                      # key into STRATA
    weight: float                     # people represented by this agent
    household_id: int
    household_size: int
    elderly_in_household: bool
    children_in_household: bool
    wfh_ability: float
    income_security: float
    big_five: dict[str, str]
    constructs: dict[str, float] = field(default_factory=dict)
    flavour: str = "thai"             # "thai" | "generic"

    # --- convenience -----------------------------------------------------
    @property
    def stratum_obj(self) -> Stratum:
        return STRATA[STRATUM_INDEX[self.stratum]]

    def c(self, name: str) -> float:
        return self.constructs.get(name, 0.5)

    def to_dict(self) -> dict:
        return asdict(self)


def _sample_age(rng: random.Random, st: Stratum) -> int:
    lo, hi = st.age_range
    return rng.randint(lo, hi)


def _sample_constructs(rng: random.Random, st: Stratum,
                       ablate: tuple[str, ...] = ()) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in evidence.BEHAVIOURAL_CONSTRUCTS:
        if spec.name in ablate:
            out[spec.name] = 0.5           # neutralised for the ablation arm
            continue
        # Stratum shifts are clamped so no construct saturates at 0/1: an
        # informal worker should be under heavy economic pressure, not under
        # *certain* economic pressure.
        mean = min(max(spec.mean + st.construct_shift.get(spec.name, 0.0), 0.10), 0.90)
        out[spec.name] = _beta(rng, mean, spec.concentration)
    return out


def _sample_big_five(rng: random.Random) -> dict[str, str]:
    return {k: (v[0] if rng.random() < 0.5 else v[1]) for k, v in BIG_FIVE.items()}


class Population:
    """A synthetic population of representative agents plus their households."""

    def __init__(
        self,
        n_agents: int,
        total_population: int,
        rng: random.Random,
        flavour: str = "thai",
        ablate: tuple[str, ...] = (),
    ):
        self.rng = rng
        self.flavour = flavour
        self.ablate = tuple(ablate)
        self.total_population = total_population
        self.shares = stratum_shares()
        self.households: list[Household] = []
        self.people: list[Persona] = []
        self._build(n_agents)

    # -- construction -----------------------------------------------------
    def _build(self, n_agents: int) -> None:
        rng = self.rng
        # Allocate agents to strata proportionally (largest-remainder), so the
        # stratum weights stay faithful even for small n.
        raw = [s * n_agents for s in self.shares]
        counts = [int(x) for x in raw]
        for idx in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True):
            if sum(counts) >= n_agents:
                break
            counts[idx] += 1
        while sum(counts) > n_agents:
            counts[max(range(len(counts)), key=lambda i: counts[i])] -= 1

        stratum_of: list[int] = []
        for gi, c in enumerate(counts):
            stratum_of += [gi] * c
        rng.shuffle(stratum_of)

        # Households: draw sizes until every agent has a home.
        hh_sizes: list[int] = []
        sizes = list(evidence.HOUSEHOLD_SIZE_PMF)
        probs = [evidence.HOUSEHOLD_SIZE_PMF[s] for s in sizes]
        remaining = n_agents
        while remaining > 0:
            s = min(rng.choices(sizes, weights=probs)[0], remaining)
            hh_sizes.append(s)
            remaining -= s

        used_names: set[str] = set()
        pid = 0
        cursor = 0
        for hid, size in enumerate(hh_sizes):
            hh = Household(id=hid, size=size)
            members = stratum_of[cursor:cursor + size]
            cursor += size
            # Thai multigenerational structure: with probability
            # P_MULTIGENERATIONAL a household of >=3 gets an elder member.
            if size >= 3 and rng.random() < evidence.P_MULTIGENERATIONAL.value:
                if STRATUM_INDEX["elderly_retired"] not in members:
                    members[rng.randrange(len(members))] = STRATUM_INDEX["elderly_retired"]
            for gi in members:
                st = STRATA[gi]
                p = Persona(
                    id=pid,
                    name=self._name(rng, used_names),
                    age=_sample_age(rng, st),
                    stratum=st.key,
                    weight=0.0,           # filled below
                    household_id=hid,
                    household_size=size,
                    elderly_in_household=False,
                    children_in_household=False,
                    wfh_ability=min(1.0, max(0.0, st.wfh_ability + rng.gauss(0, 0.1))),
                    income_security=st.income_security,
                    big_five=_sample_big_five(rng),
                    constructs=_sample_constructs(rng, st, self.ablate),
                    flavour=self.flavour,
                )
                hh.member_ids.append(pid)
                self.people.append(p)
                pid += 1
            self.households.append(hh)

        by_id = {p.id: p for p in self.people}
        for hh in self.households:
            ages = [by_id[m].age for m in hh.member_ids]
            hh.has_elderly = any(a >= 60 for a in ages)
            hh.has_children = any(a < 15 for a in ages)
            for m in hh.member_ids:
                # "elderly in household" means someone *else* is elderly.
                by_id[m].elderly_in_household = any(
                    by_id[o].age >= 60 for o in hh.member_ids if o != m
                )
                by_id[m].children_in_household = any(
                    by_id[o].age < 15 for o in hh.member_ids if o != m
                )

        # Population weights: each agent stands for N_g / n_g real people.
        for gi, st in enumerate(STRATA):
            members = [p for p in self.people if p.stratum == st.key]
            if not members:
                continue
            n_real = self.shares[gi] * self.total_population
            w = n_real / len(members)
            for p in members:
                p.weight = w

    def _name(self, rng: random.Random, used: set[str]) -> str:
        given, family = ((THAI_GIVEN_NAMES, THAI_FAMILY_NAMES) if self.flavour == "thai"
                         else (GENERIC_GIVEN, GENERIC_FAMILY))
        for _ in range(500):
            cand = f"{rng.choice(given)} {rng.choice(family)}"
            if cand not in used:
                used.add(cand)
                return cand
        cand = f"Person_{len(used)}"
        used.add(cand)
        return cand

    # -- queries ----------------------------------------------------------
    def by_stratum(self) -> dict[str, list[Persona]]:
        out: dict[str, list[Persona]] = {st.key: [] for st in STRATA}
        for p in self.people:
            out[p.stratum].append(p)
        return out

    def stratum_populations(self) -> list[float]:
        """Real-world people per stratum (N_g), from the agent weights."""
        totals = [0.0] * len(STRATA)
        for p in self.people:
            totals[STRATUM_INDEX[p.stratum]] += p.weight
        return totals

    def household_of(self, p: Persona) -> Household:
        return self.households[p.household_id]

    def summary(self) -> str:
        by = self.by_stratum()
        lines = [f"Population: {len(self.people)} agents representing "
                 f"{self.total_population:,} people in {len(self.households)} households",
                 f"Flavour: {self.flavour}" +
                 (f"  (ablated: {', '.join(self.ablate)})" if self.ablate else "")]
        for st in STRATA:
            members = by[st.key]
            if not members:
                continue
            ep = sum(m.c('economic_pressure') for m in members) / len(members)
            fo = sum(m.c('family_obligation') for m in members) / len(members)
            eld = sum(1 for m in members if m.elderly_in_household) / len(members)
            lines.append(
                f"  {st.key:<18} n={len(members):>4}  w={members[0].weight:>10,.0f}  "
                f"econ_pressure={ep:.2f}  family_oblig={fo:.2f}  elder_at_home={eld:.0%}"
            )
        return "\n".join(lines)
