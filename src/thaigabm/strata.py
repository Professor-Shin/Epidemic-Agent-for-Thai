"""Population strata and setting-specific contact matrices.

Why strata at all? Guideline section 4: a single SEIR compartment would make
the agents redundant -- 63% staying home is just a scalar. With *stratified*
SEIR the causal chain

    persona -> individual behaviour -> stratum-specific contact -> transmission

is preserved, and heterogeneous behaviour produces heterogeneous epidemics.

Contact structure is split into four settings, because Thai behaviour differs
per setting: a street vendor cannot stop working but can stop visiting their
parents. Behaviour therefore modulates each setting independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SETTINGS: tuple[str, ...] = ("home", "work", "social", "family_visit")


@dataclass(frozen=True)
class Stratum:
    key: str
    label: str
    share: float                  # fraction of the population
    age_range: tuple[int, int]
    wfh_ability: float            # 0-1, can this job be done from home?
    income_security: float        # 0-1, 1 = salaried & stable
    # Setting-specific baseline daily contacts (pre-behaviour), used to build
    # the contact matrices below.
    activity: dict[str, float]
    # Multiplicative nudges applied to the population-level construct means.
    construct_shift: dict[str, float] = field(default_factory=dict)


# Shares are approximate Thailand figures (labour-force + age structure scale).
# They are calibration placeholders: swap in the exact NSO table before any
# quantitative claim about Thailand.
STRATA: tuple[Stratum, ...] = (
    Stratum(
        "student", "student", 0.18, (6, 22), wfh_ability=0.70, income_security=0.50,
        activity={"home": 1.0, "work": 1.2, "social": 1.0, "family_visit": 0.9},
        construct_shift={"economic_pressure": -0.25, "perceived_severity": -0.10,
                         "social_preference": +0.20, "health_literacy": +0.05},
    ),
    Stratum(
        "office_worker", "office / salaried worker", 0.16, (23, 59),
        wfh_ability=0.75, income_security=0.85,
        activity={"home": 1.0, "work": 1.0, "social": 0.9, "family_visit": 0.8},
        construct_shift={"economic_pressure": -0.20, "health_literacy": +0.10,
                         "institutional_trust": +0.05},
    ),
    Stratum(
        "informal_worker", "informal / daily-wage worker (e.g. street vendor)",
        0.30, (18, 64), wfh_ability=0.05, income_security=0.15,
        activity={"home": 1.0, "work": 1.1, "social": 0.7, "family_visit": 0.9},
        construct_shift={"economic_pressure": +0.18, "institutional_trust": -0.10,
                         "health_literacy": -0.10, "stigma_sensitivity": +0.05},
    ),
    Stratum(
        "service_worker", "service / retail / transport worker", 0.16, (18, 60),
        wfh_ability=0.10, income_security=0.45,
        activity={"home": 1.0, "work": 1.4, "social": 0.8, "family_visit": 0.8},
        construct_shift={"economic_pressure": +0.10, "stigma_sensitivity": +0.05},
    ),
    Stratum(
        "healthcare_worker", "healthcare worker", 0.02, (22, 60),
        wfh_ability=0.05, income_security=0.80,
        activity={"home": 1.0, "work": 1.6, "social": 0.6, "family_visit": 0.6},
        construct_shift={"perceived_severity": +0.20, "perceived_susceptibility": +0.25,
                         "health_literacy": +0.30, "economic_pressure": -0.20},
    ),
    Stratum(
        "elderly_retired", "retired / elderly (60+)", 0.18, (60, 85),
        wfh_ability=1.0, income_security=0.40,
        activity={"home": 1.2, "work": 0.05, "social": 0.5, "family_visit": 1.1},
        construct_shift={"perceived_severity": +0.20, "perceived_susceptibility": +0.25,
                         "economic_pressure": -0.10, "social_preference": -0.05},
    ),
)

STRATUM_KEYS: tuple[str, ...] = tuple(s.key for s in STRATA)
STRATUM_INDEX: dict[str, int] = {s.key: i for i, s in enumerate(STRATA)}
N_STRATA = len(STRATA)

# Mean daily contacts per setting for an average person, pre-behaviour.
# Total ~15/day is the usual POLYMOD-scale order of magnitude.
SETTING_TOTALS: dict[str, float] = {
    "home": 4.0, "work": 6.0, "social": 4.0, "family_visit": 1.0,
}

# Within-stratum assortativity per setting (people mix with their own kind).
ASSORTATIVITY: dict[str, float] = {
    "home": 0.2, "work": 0.8, "social": 0.6, "family_visit": 0.1,
}


def stratum_shares() -> np.ndarray:
    s = np.array([st.share for st in STRATA], dtype=float)
    return s / s.sum()


def _raw_matrix(setting: str, shares: np.ndarray) -> np.ndarray:
    """Reciprocity-preserving contact matrix for one setting.

    C[g, h] = number of contacts a member of g has with members of h per day.

    Built as an activity-proportional mixing term plus a diagonal assortative
    term. Both satisfy the reciprocity requirement N_g C[g,h] == N_h C[h,g],
    which the SEIR force-of-infection relies on.
    """
    a = np.array([st.activity[setting] for st in STRATA], dtype=float)
    prop = np.outer(a, a * shares)                    # N_g*C = a_g a_h N_g N_h : symmetric
    assort = np.diag(ASSORTATIVITY[setting] * a * a * shares)
    C = prop + assort
    # Scale so that the population-average total contacts in this setting match.
    mean_contacts = float(shares @ C.sum(axis=1))
    if mean_contacts > 0:
        C *= SETTING_TOTALS[setting] / mean_contacts
    return C


def contact_matrices(shares: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Baseline (behaviour-free) contact matrix per setting."""
    if shares is None:
        shares = stratum_shares()
    return {s: _raw_matrix(s, shares) for s in SETTINGS}


def family_visit_bridge(C: np.ndarray, boost: float = 2.0) -> np.ndarray:
    """Up-weight intergenerational contact in the family-visit setting.

    The Thai mechanism we care about: working-age people carry infection home
    to co-resident or nearby elders. Kept reciprocity-preserving by boosting
    the (g, elderly) and (elderly, g) entries as a symmetric outer adjustment.
    """
    e = STRATUM_INDEX["elderly_retired"]
    out = C.copy()
    out[:, e] *= boost
    out[e, :] *= boost
    out[e, e] /= boost  # applied twice above
    return out


def check_reciprocity(C: np.ndarray, shares: np.ndarray, tol: float = 1e-9) -> bool:
    M = shares[:, None] * C
    return bool(np.allclose(M, M.T, atol=tol))
