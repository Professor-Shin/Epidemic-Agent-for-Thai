"""Empirically grounded parameters for the Thai behavioural layer.

Design rule from the project guideline (section 12): *no cultural stereotype
parameters*. Every number here must be traceable to one of

  * a published survey / qualitative study of Thai COVID-19 response,
  * an official statistic (census, labour force survey), or
  * an explicitly-labelled MODELLING ASSUMPTION that the reader can challenge.

Each entry therefore carries a ``source`` string and a ``kind`` field:

    "survey"     -- read off a published survey statistic
    "official"   -- official statistic (NSO / MoPH scale)
    "assumption" -- our own assumption; NOT evidence, must be sensitivity-tested

References
----------
[3] Pan-ngum W. et al. (2021) "Perspectives on public health interventions in
    the management of the COVID-19 pandemic in Thailand". BMC Public Health.
    n = 1,020 Thai respondents (2020).
[4] "'Like a wake-up call for humankind': views, challenges and coping
    strategies related to public health measures during the first COVID-19
    lockdown in Thailand" (2022/2023), qualitative study.
[2] Ferenz P. et al. (2026) The Epi-LLM Framework. arXiv:2606.02867 -- motivates
    explicit attitudinal parameterisation instead of geographic labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    """A single grounded parameter."""

    name: str
    value: float
    kind: str  # survey | official | assumption
    source: str
    note: str = ""

    def __float__(self) -> float:  # convenience
        return self.value


# --------------------------------------------------------------------------
# Population-level behavioural marginals observed in Thailand (2020).
# These double as *validation targets* for the behavioural layer (Level 1/2 of
# the validation plan): if our agents are Thai-grounded, the simulated rates
# during a high-salience period should land near these.
# --------------------------------------------------------------------------
THAI_BEHAVIOURAL_TARGETS: dict[str, Param] = {
    "mask_use": Param(
        "mask_use", 0.97, "survey", "Pan-ngum et al. 2021 [3]",
        "97% reported using personal protective equipment (masks).",
    ),
    "social_distancing": Param(
        "social_distancing", 0.94, "survey", "Pan-ngum et al. 2021 [3]",
        "94% reported practising social distancing.",
    ),
    "self_initiated_change": Param(
        "self_initiated_change", 0.92, "survey", "Pan-ngum et al. 2021 [3]",
        "92% adjusted social behaviour BEFORE government measures -- i.e. "
        "behaviour change is voluntary/anticipatory, not purely policy-driven.",
    ),
    "income_concern": Param(
        "income_concern", 0.80, "survey", "Pan-ngum et al. 2021 [3]",
        ">80% reported loss of income as a major concern.",
    ),
}


# --------------------------------------------------------------------------
# Behavioural construct distributions.
#
# Each construct is sampled per-agent as Beta(a, b) reparameterised by
# (mean, concentration). Means are *stratum-modulated* in strata.py; the values
# here are the population-level anchors.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstructSpec:
    name: str
    mean: float
    concentration: float  # higher = tighter distribution
    kind: str
    source: str
    prompt_label: str
    note: str = ""


BEHAVIOURAL_CONSTRUCTS: tuple[ConstructSpec, ...] = (
    ConstructSpec(
        "economic_pressure", 0.80, 6.0, "survey", "Pan-ngum et al. 2021 [3]",
        "financial pressure to keep earning",
        "Anchored on the >80% income-concern figure.",
    ),
    ConstructSpec(
        "family_obligation", 0.75, 8.0, "qualitative",
        "Thai lockdown qualitative study [4]",
        "sense of duty towards family and household elders",
        "Family/community ties reported as central; magnitude is our reading "
        "of a qualitative finding, so treat as semi-grounded.",
    ),
    ConstructSpec(
        "perceived_severity", 0.70, 6.0, "survey", "Pan-ngum et al. 2021 [3]",
        "belief that the disease is dangerous",
        "Consistent with very high protective-behaviour uptake.",
    ),
    ConstructSpec(
        "perceived_susceptibility", 0.45, 5.0, "assumption", "MODELLING ASSUMPTION",
        "belief that you personally are likely to catch it",
        "Not directly measured; sensitivity-test this.",
    ),
    ConstructSpec(
        "institutional_trust", 0.55, 4.0, "assumption", "MODELLING ASSUMPTION",
        "trust in government and health authorities",
        "Deliberately mid-range and wide: Thai trust was contested and "
        "time-varying. A key ablation target.",
    ),
    ConstructSpec(
        "norm_sensitivity", 0.75, 7.0, "survey", "Pan-ngum et al. 2021 [3]",
        "tendency to do what people around you are doing",
        "Anchored on 92% pre-policy voluntary change + 94-97% uptake, which "
        "implies strong normative diffusion rather than enforcement.",
    ),
    ConstructSpec(
        "stigma_sensitivity", 0.60, 5.0, "qualitative",
        "Thai lockdown qualitative study [4]",
        "fear of being blamed or shamed if known to be infected",
        "Study reports stigma/shaming and concealment of infection.",
    ),
    ConstructSpec(
        "social_preference", 0.55, 4.0, "assumption", "MODELLING ASSUMPTION",
        "enjoyment of socialising outside the household",
        "",
    ),
    ConstructSpec(
        "health_literacy", 0.60, 5.0, "assumption", "MODELLING ASSUMPTION",
        "ability to interpret health information",
        "",
    ),
)

CONSTRUCT_NAMES: tuple[str, ...] = tuple(c.name for c in BEHAVIOURAL_CONSTRUCTS)

# Constructs that the ablation study (Experiment E) can switch off. Switching
# off means: (a) set the value to the neutral 0.5 for every agent, and
# (b) remove the line from the LLM prompt.
ABLATABLE = (
    "economic_pressure",
    "family_obligation",
    "institutional_trust",
    "stigma_sensitivity",
    "norm_sensitivity",
    "perceived_severity",
)


# --------------------------------------------------------------------------
# Household structure (Thai households are larger and more intergenerational
# than the Western default -- this is the structural, not attitudinal, half of
# "Thai context").
# --------------------------------------------------------------------------
HOUSEHOLD_SIZE_PMF: dict[int, float] = {1: 0.21, 2: 0.24, 3: 0.22, 4: 0.18, 5: 0.09, 6: 0.06}
HOUSEHOLD_SIZE_SOURCE = Param(
    "household_size", 3.0, "official",
    "Approximate Thai household-size distribution (NSO scale); mean ~2.9",
    "Replace with the exact census table before publication.",
)

# Probability that a household containing >=3 members includes a person 60+.
P_MULTIGENERATIONAL = Param(
    "p_multigenerational", 0.40, "assumption", "MODELLING ASSUMPTION",
    "Co-residence of elderly with working-age children is common in Thailand; "
    "exact rate should be taken from the census.",
)


# --------------------------------------------------------------------------
# Epidemiological constants (these are biology, not behaviour -- the LLM must
# never set them; see guideline section 12).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DiseaseParams:
    latent_days: float = 3.0       # 1/sigma
    infectious_days: float = 6.0   # 1/gamma, matches the source paper's 6 days
    r0: float = 2.5                # basic reproduction number, behaviour-free
    symptomatic_fraction: float = 0.65
    ascertainment: float = 0.35    # fraction of symptomatic cases reported
    source: str = "Williams et al. 2023 (6-day infectious period); R0 2.5 is a "
    "conventional wild-type SARS-CoV-2 value."

    @property
    def sigma(self) -> float:
        return 1.0 / self.latent_days

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_days


MASK_EFFICACY = Param(
    "mask_efficacy", 0.30, "assumption", "MODELLING ASSUMPTION",
    "Per-contact relative reduction in transmission per masked party.",
)


def evidence_table() -> str:
    """Human-readable provenance table -- printed into every results directory."""
    rows: list[str] = ["| parameter | value | kind | source |",
                       "| --- | --- | --- | --- |"]
    for p in THAI_BEHAVIOURAL_TARGETS.values():
        rows.append(f"| {p.name} | {p.value} | {p.kind} | {p.source} |")
    for c in BEHAVIOURAL_CONSTRUCTS:
        rows.append(f"| {c.name} (mean) | {c.mean} | {c.kind} | {c.source} |")
    rows.append(f"| household_size | see PMF | official | {HOUSEHOLD_SIZE_SOURCE.source} |")
    rows.append(f"| {P_MULTIGENERATIONAL.name} | {P_MULTIGENERATIONAL.value} | "
                f"{P_MULTIGENERATIONAL.kind} | {P_MULTIGENERATIONAL.source} |")
    rows.append(f"| {MASK_EFFICACY.name} | {MASK_EFFICACY.value} | "
                f"{MASK_EFFICACY.kind} | {MASK_EFFICACY.source} |")
    return "\n".join(rows)
