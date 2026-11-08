"""Sensitivity variants and action-order checks shared by notebook 04 and scripts/run_server.py.

Both must build exactly the same (cfg, contact model, personas) for each tag, so the run ids match and a run
made on Colab is skipped on the server (and the other way round).
"""
import copy

from .agents import rescale_wfh
from .sim import contact_model_from_cfg


SENS_AXES = ["w_home_scale", "f_errand", "f_visit", "wfh_scale"]


def sensitivity_variants(cfg, root, personas, cm, axes=None):
    """One-at-a-time changes: (tag, cfg, contact model, personas).

    `axes` keeps a subset of SENS_AXES, for when the GPU budget does not cover all four on every model.
    `wfh_scale` is the one worth keeping if only one fits: it is the only axis that edits the prompt itself
    (the "Can work from home" line of the persona), so it asks whether the agent acts on its own attributes.
    The other three change how an action maps to contacts, and reach the agent only through the reported
    case count it sees the next day.
    """
    S = cfg["sensitivity"]
    want = set(axes or SENS_AXES)
    unknown = want - set(SENS_AXES)
    if unknown:
        raise ValueError(f"unknown sensitivity axes {sorted(unknown)}; choose from {SENS_AXES}")
    if "w_home_scale" in want:
        for v in S["w_home_scale"]:
            yield f"sens_home={v}", cfg, contact_model_from_cfg(cfg, root, home_scale=v), personas
    for key in ["f_errand", "f_visit"]:
        if key in want:
            for v in S[key]:
                c2 = copy.deepcopy(cfg); c2["contacts"][key] = v
                yield f"sens_{key}={v}", c2, contact_model_from_cfg(c2, root), personas
    if "wfh_scale" in want:
        for v in S["wfh_scale"]:
            yield f"sens_wfh={v}", cfg, cm, rescale_wfh(personas, v)    # only WFH ability changes


def baseline_mix_variants(cfg, root):
    """(tag, cfg, contact model) with contacts.baseline_mix set to the alternative the report calls for.

    q_out is measured against baseline_mix, the activity split of a normal day, which is an `assumption` row in
    data/parameter_sources.csv with no source behind it. Four of five models produce q_out > 1 under the default
    work_onsite = 0.55, i.e. residents going out MORE than the baseline during an epidemic, so the finding rests
    on that one number. This variant raises work_onsite to contacts.employment_rate, the defensible upper bound
    (everyone employed works on-site on a normal day), and rescales the other three in proportion. If the 2x2
    effect keeps its sign here, the headline result does not depend on the assumption.
    """
    mix = cfg["contacts"]["baseline_mix"]
    work = cfg["contacts"]["employment_rate"]
    others = [a for a in mix if a != "work_onsite"]
    rest = sum(mix[a] for a in others)
    c2 = copy.deepcopy(cfg)
    c2["contacts"]["baseline_mix"] = {**{a: mix[a] / rest * (1 - work) for a in others}, "work_onsite": work}
    yield f"baseline=work{work}", c2, contact_model_from_cfg(c2, root)


def action_order_variants(cfg):
    """(tag, cfg) with the resident action list reordered, to check that choices are not a prompt-order artefact."""
    base = list(cfg["actions"])
    orders = {"order=rev": base[::-1],
              "order=visitfirst": ["visit_family"] + [a for a in base if a != "visit_family"]}
    for tag, order in orders.items():
        c2 = copy.deepcopy(cfg); c2["actions"] = {a: cfg["actions"][a] for a in order}
        yield tag, c2
