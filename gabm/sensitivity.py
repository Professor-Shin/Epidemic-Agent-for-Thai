"""Sensitivity variants and action-order checks shared by notebook 04 and scripts/run_server.py.

Both must build exactly the same (cfg, contact model, personas) for each tag, so the run ids match and a run
made on Colab is skipped on the server (and the other way round).
"""
import copy

from .agents import rescale_wfh
from .sim import contact_model_from_cfg


def sensitivity_variants(cfg, root, personas, cm):
    """One-at-a-time changes: (tag, cfg, contact model, personas)."""
    S = cfg["sensitivity"]
    for v in S["w_home_scale"]:
        yield f"sens_home={v}", cfg, contact_model_from_cfg(cfg, root, home_scale=v), personas
    for key in ["f_errand", "f_visit"]:
        for v in S[key]:
            c2 = copy.deepcopy(cfg); c2["contacts"][key] = v
            yield f"sens_{key}={v}", c2, contact_model_from_cfg(c2, root), personas
    for v in S["wfh_scale"]:
        yield f"sens_wfh={v}", cfg, cm, rescale_wfh(personas, v)    # only WFH ability changes


def action_order_variants(cfg):
    """(tag, cfg) with the resident action list reordered, to check that choices are not a prompt-order artefact."""
    base = list(cfg["actions"])
    orders = {"order=rev": base[::-1],
              "order=visitfirst": ["visit_family"] + [a for a in base if a != "visit_family"]}
    for tag, order in orders.items():
        c2 = copy.deepcopy(cfg); c2["actions"] = {a: cfg["actions"][a] for a in order}
        yield tag, c2
