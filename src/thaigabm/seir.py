"""Mechanistic stratified SEIR model.

This is the half of the system the LLM is *not* allowed to touch. It receives
per-stratum, per-setting behavioural multipliers and returns epidemic state.

    dS_g/dt = -lambda_g S_g
    dE_g/dt =  lambda_g S_g - sigma E_g
    dI_g/dt =  sigma E_g     - gamma I_g
    dR_g/dt =  gamma I_g

    lambda_g(t) = sum_h q * C_gh(t) * m_g(t) * m_h(t) * I_h / N_h

    C_gh(t) = sum_settings C^s_gh * phi^s_g(t) * phi^s_h(t)
    m_g(t)  = 1 - mask_efficacy * mask_coverage_g(t)

``q`` (per-contact transmission probability) is calibrated once from a target
R0 via the next-generation matrix of the *baseline* contact structure, so that
"R0 = 2.5" means the same thing across every experimental arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.thaigabm.evidence import MASK_EFFICACY, DiseaseParams
from src.thaigabm.strata import (
    N_STRATA, SETTINGS, STRATUM_KEYS, contact_matrices, family_visit_bridge,
)


@dataclass
class BehaviourInput:
    """What the aggregator hands the SEIR model each day."""

    phi: dict[str, np.ndarray]    # setting -> per-stratum participation in [0,1]
    mask: np.ndarray              # per-stratum mask coverage in [0,1]
    healthcare_seeking: np.ndarray = field(
        default_factory=lambda: np.zeros(N_STRATA))

    @staticmethod
    def neutral() -> "BehaviourInput":
        return BehaviourInput(
            phi={s: np.ones(N_STRATA) for s in SETTINGS},
            mask=np.zeros(N_STRATA),
        )


@dataclass
class EpiState:
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray

    def totals(self) -> tuple[float, float, float, float]:
        return (float(self.S.sum()), float(self.E.sum()),
                float(self.I.sum()), float(self.R.sum()))


class StratifiedSEIR:
    def __init__(
        self,
        N: np.ndarray,
        disease: DiseaseParams | None = None,
        mask_efficacy: float = MASK_EFFICACY.value,
        family_visit_boost: float = 2.0,
        dt: float = 0.25,
    ):
        self.N = np.asarray(N, dtype=float)
        self.disease = disease or DiseaseParams()
        self.mask_efficacy = mask_efficacy
        self.dt = dt

        shares = self.N / self.N.sum()
        C = contact_matrices(shares)
        C["family_visit"] = family_visit_bridge(C["family_visit"], family_visit_boost)
        self.C_base: dict[str, np.ndarray] = C
        self.q = self._calibrate_q(self.disease.r0)

        self.state = EpiState(
            S=self.N.copy(), E=np.zeros(N_STRATA),
            I=np.zeros(N_STRATA), R=np.zeros(N_STRATA),
        )
        self._cum_infections = 0.0

    # -- calibration ------------------------------------------------------
    def total_contact_matrix(self, b: BehaviourInput | None = None) -> np.ndarray:
        if b is None:
            return sum(self.C_base.values())
        out = np.zeros((N_STRATA, N_STRATA))
        for s, C in self.C_base.items():
            phi = np.clip(b.phi.get(s, np.ones(N_STRATA)), 0.0, 1.5)
            out += C * np.outer(phi, phi)
        return out

    def next_generation_matrix(self, C: np.ndarray, q: float | None = None,
                               S_frac: np.ndarray | None = None) -> np.ndarray:
        """K[g, h] = expected infections in g caused by one infectious person in h.

        An infectious person in h makes C[h, g] * (N_g/N_h)... no: with the
        reciprocity convention C[g,h] = contacts a g-person has with h-people,
        an h-person has C[h,g] contacts with g-people per day, infects a
        susceptible fraction S_g/N_g of them with probability q, for 1/gamma days.
        """
        q = self.q if q is None else q
        S_frac = np.ones(N_STRATA) if S_frac is None else S_frac
        return q * C.T * S_frac[:, None] / self.disease.gamma

    def _calibrate_q(self, target_r0: float) -> float:
        C = self.total_contact_matrix()
        K1 = self.next_generation_matrix(C, q=1.0)
        rho = float(np.max(np.abs(np.linalg.eigvals(K1))))
        return target_r0 / rho if rho > 0 else 0.0

    def effective_reproduction_number(self, b: BehaviourInput | None = None) -> float:
        C = self.total_contact_matrix(b)
        m = self._mask_factor(b)
        K = self.next_generation_matrix(C * np.outer(m, m),
                                        S_frac=self.state.S / np.maximum(self.N, 1e-9))
        return float(np.max(np.abs(np.linalg.eigvals(K))))

    def behaviour_free_rt(self) -> float:
        return self.effective_reproduction_number(BehaviourInput.neutral())

    # -- dynamics ---------------------------------------------------------
    def _mask_factor(self, b: BehaviourInput | None) -> np.ndarray:
        if b is None:
            return np.ones(N_STRATA)
        return 1.0 - self.mask_efficacy * np.clip(b.mask, 0.0, 1.0)

    def force_of_infection(self, b: BehaviourInput | None) -> np.ndarray:
        C = self.total_contact_matrix(b)
        m = self._mask_factor(b)
        prev = self.state.I / np.maximum(self.N, 1e-9)
        return self.q * (C * np.outer(m, m)) @ prev

    def step_day(self, b: BehaviourInput | None = None) -> dict:
        """Advance one simulated day with fixed behaviour, returning day flows."""
        sigma, gamma = self.disease.sigma, self.disease.gamma
        new_infections = 0.0
        new_symptomatic = 0.0
        st = self.state
        for _ in range(int(round(1.0 / self.dt))):
            lam = self.force_of_infection(b)
            inf = lam * st.S
            prog = sigma * st.E
            rec = gamma * st.I
            st.S = np.maximum(st.S - self.dt * inf, 0.0)
            st.E = np.maximum(st.E + self.dt * (inf - prog), 0.0)
            st.I = np.maximum(st.I + self.dt * (prog - rec), 0.0)
            st.R = np.maximum(st.R + self.dt * rec, 0.0)
            new_infections += float(self.dt * inf.sum())
            new_symptomatic += float(self.dt * prog.sum())
        self._cum_infections += new_infections
        return {
            "new_infections": new_infections,
            "new_symptomatic": new_symptomatic,
            "cumulative_infections": self._cum_infections,
        }

    # -- seeding / queries -------------------------------------------------
    def seed(self, n_infected: float, stratum: str | None = None) -> None:
        idx = (list(range(N_STRATA)) if stratum is None
               else [STRATUM_KEYS.index(stratum)])
        share = self.N[idx] / self.N[idx].sum()
        add = n_infected * share
        self.state.S[idx] -= add
        self.state.I[idx] += add

    def prevalence(self) -> np.ndarray:
        """Infectious prevalence per stratum (fraction)."""
        return self.state.I / np.maximum(self.N, 1e-9)

    def attack_rate(self) -> float:
        return float((self.state.R.sum() + self.state.I.sum() + self.state.E.sum())
                     / self.N.sum())
