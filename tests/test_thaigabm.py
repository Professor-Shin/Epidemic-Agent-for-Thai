"""Invariants that must hold for ThaiGABM-SEIR results to mean anything.

    python3 -m pytest tests/ -q      (or: python3 tests/test_thaigabm.py)
"""

from __future__ import annotations

import asyncio
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.thaigabm.actions import BehaviourDecision, parse_decision
from src.thaigabm.aggregator import aggregate
from src.thaigabm.behaviour import AgentContext, RuleBasedBehaviour
from src.thaigabm.config import arm_A_classical, arm_D_thai_gabm
from src.thaigabm.evidence import THAI_BEHAVIOURAL_TARGETS
from src.thaigabm.information import InformationState
from src.thaigabm.persona import Population
from src.thaigabm.prompts import ArmSpec, build_prompt
from src.thaigabm.runner import ThaiGABMSimulation
from src.thaigabm.seir import BehaviourInput, StratifiedSEIR
from src.thaigabm.strata import (
    N_STRATA, check_reciprocity, contact_matrices, family_visit_bridge,
    stratum_shares,
)


def _pop(n=120, seed=3, **kw):
    return Population(n, 70_000_000, random.Random(seed), **kw)


def _seir(pop):
    return StratifiedSEIR(np.array(pop.stratum_populations()))


# --------------------------------------------------------------------------
# Contact structure
# --------------------------------------------------------------------------
def test_contact_matrices_are_reciprocal():
    """N_g C_gh must equal N_h C_hg, or the force of infection is unphysical."""
    shares = stratum_shares()
    for setting, C in contact_matrices(shares).items():
        assert check_reciprocity(C, shares), f"{setting} violates reciprocity"


def test_family_visit_bridge_preserves_reciprocity():
    shares = stratum_shares()
    C = family_visit_bridge(contact_matrices(shares)["family_visit"], 2.0)
    assert check_reciprocity(C, shares)


# --------------------------------------------------------------------------
# SEIR
# --------------------------------------------------------------------------
def test_r0_is_calibrated_exactly():
    m = _seir(_pop())
    assert abs(m.behaviour_free_rt() - m.disease.r0) < 1e-8


def test_unmitigated_final_size_matches_theory():
    """Arm A must reproduce the analytic SEIR final size for its R0."""
    m = _seir(_pop())
    m.seed(1000)
    for _ in range(400):
        m.step_day(BehaviourInput.neutral())
    r0 = m.disease.r0
    # Final-size relation: z = 1 - exp(-R0 z)
    z = 0.5
    for _ in range(500):
        z = 1.0 - math.exp(-r0 * z)
    assert abs(m.attack_rate() - z) < 0.03, (m.attack_rate(), z)


def test_population_is_conserved():
    m = _seir(_pop())
    m.seed(5000)
    total0 = sum(m.state.totals())
    for _ in range(120):
        m.step_day(BehaviourInput.neutral())
    assert abs(sum(m.state.totals()) - total0) / total0 < 1e-9


def test_behaviour_reduces_transmission_monotonically():
    m = _seir(_pop())
    prev = m.effective_reproduction_number(BehaviourInput.neutral())
    for level in (0.75, 0.5, 0.25):
        b = BehaviourInput(phi={s: np.full(N_STRATA, level)
                                for s in ("home", "work", "social", "family_visit")},
                           mask=np.zeros(N_STRATA))
        rt = m.effective_reproduction_number(b)
        assert rt < prev
        prev = rt


def test_masks_reduce_rt():
    m = _seir(_pop())
    bare = BehaviourInput.neutral()
    masked = BehaviourInput(phi=bare.phi, mask=np.ones(N_STRATA))
    assert m.effective_reproduction_number(masked) < m.effective_reproduction_number(bare)


# --------------------------------------------------------------------------
# Decision parsing
# --------------------------------------------------------------------------
def test_parse_decision_handles_llm_drift():
    variants = [
        '{"go_to_work": true, "social_activity": false, "visit_family": false, '
        '"wear_mask": true, "seek_healthcare": false, "reason": "I need the money."}',
        '```json\n{"go_to_work": True, "wear_mask": True, "reason": "x"}\n```',
        'Sure! Here is my answer:\n{"go_to_work": "yes", "wear_mask": "no", "reason": "x"}',
    ]
    for raw in variants:
        d = parse_decision(raw)
        assert d.source == "llm", raw
        assert d.go_to_work is True


def test_parse_failure_falls_back_to_the_rule_decision():
    """A garbled response must not silently become 'went to work'."""
    fallback = BehaviourDecision(go_to_work=False, wear_mask=True, reason="rule")
    d = parse_decision("I'm sorry, I cannot help with that.", fallback=fallback)
    assert d.source == "fallback"
    assert d.go_to_work is False and d.wear_mask is True


# --------------------------------------------------------------------------
# Behavioural calibration anchors
# --------------------------------------------------------------------------
def _ctx(person, *, symptom="none", per100k=0.0, policy="none", trend="steady",
         sick_at_home=False, elderly=False, day=10):
    return AgentContext(
        person=person, day=day, symptom_level=symptom,
        symptom_text=symptom, household={
            "household_size": person.household_size,
            "someone_sick_at_home": sick_at_home,
            "n_sick_at_home": int(sick_at_home),
            "elderly_at_home": elderly, "children_at_home": False},
        info=InformationState(day=day, reported_per_100k=per100k, trend=trend,
                              policy=policy),
    )


def _rates(model, people, n=40, **ctx_kw):
    work = mask = dist = 0
    total = len(people) * n
    for _ in range(n):
        for p in people:
            d = model.decide(_ctx(p, **ctx_kw))
            work += d.go_to_work
            mask += d.wear_mask
            dist += d.distancing
    return {"work": work / total, "mask": mask / total, "distancing": dist / total}


def test_quiet_period_looks_like_normal_life():
    """With no outbreak news and no symptoms, people work and do not mask."""
    pop = _pop()
    workers = [p for p in pop.people if p.stratum != "elderly_retired"]
    r = _rates(RuleBasedBehaviour(random.Random(0)), workers)
    assert r["work"] > 0.90, r
    assert r["mask"] < 0.15, r


def test_peak_salience_reproduces_survey_uptake():
    """At full salience under lockdown, mask use and distancing approach the
    rates reported by Pan-ngum et al. (97% / 94%)."""
    pop = _pop()
    model = RuleBasedBehaviour(random.Random(0))
    r = _rates(model, pop.people, per100k=200.0, policy="lockdown", trend="rising")
    assert r["mask"] > THAI_BEHAVIOURAL_TARGETS["mask_use"].value - 0.15, r
    assert r["distancing"] > THAI_BEHAVIOURAL_TARGETS["social_distancing"].value - 0.15, r


def test_economic_vulnerability_keeps_informal_workers_working():
    """The central Thai mechanism, in the regime where the trade-off is live.

    With a heavy outbreak under way, an informal worker who cannot work from
    home keeps going out far more than an office worker who can. The gap has
    to be large in absolute terms, not merely in the right direction.
    """
    pop = _pop(400)
    model = RuleBasedBehaviour(random.Random(1))
    informal = [p for p in pop.people if p.stratum == "informal_worker"]
    office = [p for p in pop.people if p.stratum == "office_worker"]
    for kw in (dict(per100k=50.0, policy="restriction"),
               dict(symptom="mild", per100k=50.0, policy="restriction"),
               dict(per100k=200.0, policy="lockdown")):
        ri = _rates(model, informal, n=20, **kw)
        ro = _rates(model, office, n=20, **kw)
        assert ri["work"] - ro["work"] > 0.20, (kw, ri, ro)


def test_severe_illness_overrides_economic_pressure_but_not_completely():
    """A fever should stop most people working -- yet the economically exposed
    stratum still works several times more often than the protected one, which
    is the concealment/keep-earning pattern the Thai literature describes."""
    pop = _pop(400)
    model = RuleBasedBehaviour(random.Random(1))
    kw = dict(symptom="severe", per100k=50.0, policy="restriction")
    ri = _rates(model, [p for p in pop.people if p.stratum == "informal_worker"],
                n=20, **kw)
    ro = _rates(model, [p for p in pop.people if p.stratum == "office_worker"],
                n=20, **kw)
    assert ri["work"] < 0.25, ri            # illness dominates
    assert ri["work"] > 2.0 * ro["work"], (ri, ro)


def test_elderly_at_home_suppresses_family_visits():
    pop = _pop(400)
    model = RuleBasedBehaviour(random.Random(2))
    people = [p for p in pop.people if p.stratum == "office_worker"]
    kw = dict(per100k=50.0, trend="rising")
    with_elder = sum(model.decide(_ctx(p, elderly=True, **kw)).visit_family
                     for p in people for _ in range(20))
    without = sum(model.decide(_ctx(p, elderly=False, **kw)).visit_family
                  for p in people for _ in range(20))
    assert with_elder < without, (with_elder, without)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def test_aggregate_uses_population_weights_not_agent_counts():
    """A stratum sampled with few agents must still count for its real size."""
    pop = _pop(200)
    people, decisions = [], []
    for p in pop.people:
        people.append(p)
        decisions.append(BehaviourDecision(
            go_to_work=(p.stratum != "healthcare_worker"), wear_mask=True))
    agg = aggregate(people, decisions)
    assert agg.per_stratum["healthcare_worker"]["phi_work"] == 0.0
    expected = 1.0 - sum(p.weight for p in people if p.stratum == "healthcare_worker") \
        / sum(p.weight for p in people)
    assert abs(agg.population["phi_work"] - expected) < 1e-9


def test_policy_cap_bounds_participation():
    pop = _pop(100)
    decisions = [BehaviourDecision(go_to_work=True, social_activity=True,
                                   visit_family=True) for _ in pop.people]
    free = aggregate(pop.people, decisions, policy="none")
    locked = aggregate(pop.people, decisions, policy="lockdown")
    assert locked.population["phi_social"] < free.population["phi_social"]
    assert locked.population["phi_work"] <= 0.70 + 1e-9


# --------------------------------------------------------------------------
# Arms and ablation
# --------------------------------------------------------------------------
def test_ablation_neutralises_the_construct_and_removes_it_from_the_prompt():
    arm = ArmSpec(ablate=("economic_pressure",))
    pop = _pop(120, ablate=("economic_pressure",))
    assert all(p.c("economic_pressure") == 0.5 for p in pop.people)
    text = build_prompt(pop.people[0], arm, 10, "feel completely normal",
                        {"someone_sick_at_home": False, "elderly_at_home": False},
                        InformationState(), 70_000_000)
    assert "financial pressure to keep earning" not in text
    assert "sense of duty towards family" in text  # other constructs survive


def test_generic_arm_omits_thai_parameterisation():
    pop = _pop(60, flavour="generic")
    text = build_prompt(pop.people[0], ArmSpec(thai_context=False,
                                               include_household=False),
                        10, "feel completely normal",
                        {"someone_sick_at_home": True, "elderly_at_home": True},
                        InformationState(), 70_000_000)
    assert "living in Thailand" not in text
    assert "financial pressure to keep earning" not in text
    assert "elderly relative" not in text


def test_arm_A_stays_a_clean_baseline_under_mock():
    """--mock must not give the classical-SEIR baseline behaviour or policy."""
    from src.thaigabm.experiments import apply_mock

    cfg = apply_mock(arm_A_classical(days=20, n_agents=40, print_every=999), True)
    assert cfg.behaviour == "static" and cfg.policy_enabled is False
    result = asyncio.run(ThaiGABMSimulation(cfg).run(verbose=False))
    assert all(r["mobility_index"] == 1.0 for r in result.history)
    assert all(r["policy"] == "none" for r in result.history)


def test_behaviour_arms_flatten_the_curve_relative_to_arm_A():
    """The whole point: adaptive behaviour must lower the peak."""
    kw = dict(days=140, n_agents=80, print_every=999, log_decisions=False)
    a = asyncio.run(ThaiGABMSimulation(arm_A_classical(**kw)).run(verbose=False))
    d = asyncio.run(ThaiGABMSimulation(
        arm_D_thai_gabm(behaviour="mock", **kw)).run(verbose=False))
    assert max(d.series("infectious")) < 0.5 * max(a.series("infectious"))


def test_runs_are_reproducible_under_a_seed():
    kw = dict(days=40, n_agents=60, print_every=999, log_decisions=False, seed=11)
    r1 = asyncio.run(ThaiGABMSimulation(arm_D_thai_gabm(behaviour="mock", **kw))
                     .run(verbose=False))
    r2 = asyncio.run(ThaiGABMSimulation(arm_D_thai_gabm(behaviour="mock", **kw))
                     .run(verbose=False))
    assert r1.series("infectious") == r2.series("infectious")


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def test_cache_reuses_identical_situations():
    from src.thaigabm.cache import DecisionCache, situation_key

    pop = _pop(60)
    cache = DecisionCache(mode="first", rng=random.Random(0))
    p = pop.people[0]
    info = InformationState(reported_per_100k=3.0, trend="rising", policy="advisory")
    hh = {"someone_sick_at_home": False, "elderly_at_home": True}
    key = situation_key(p, "sig", "none", hh, info)
    assert cache.get(key) is None
    cache.put(key, BehaviourDecision(go_to_work=False, reason="stay"))
    hit = cache.get(key)
    assert hit is not None and hit.source == "cache" and hit.go_to_work is False
    assert cache.stats()["cache_hits"] == 1


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
