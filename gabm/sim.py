"""One simulation run for conditions A-D (guideline section 8.1).

A  SEIR baseline            q_out = 1
B  rule-based behaviour     q_out = m(t) = 1 / (1 + alpha * cases7)
C  LLM residents            q_out from the residents' weighted action shares
D  LLM residents + leader   C plus a weekly leader level (q_policy, banned actions, measure text)

Daily order: residents observe yesterday's reports -> (leader, every 7 days) -> residents choose ->
aggregate to q_out -> beta_eff -> SEIR one day -> agents' personal health update -> log -> checkpoint.
"""
import time

import numpy as np

from . import ROOT, seir
from .agents import AgentHealth, case_report, persona_text
from .contacts import ContactModel, load_location_contacts
from .io import RunStore, config_hash, environment_info, make_run_id
from .llm import build_resident_prompt, decide_batch
from .policy import allowed_actions, build_leader_prompt, q_policy, rule_based_m

NO_POLICY_TEXT = "None. Normal activities are allowed."


def contact_model_from_cfg(cfg, root=ROOT, home_scale=1.0):
    loc = load_location_contacts(root / cfg["contacts"]["file"], cfg["contacts"]["column"])
    return ContactModel(loc, cfg["contacts"], home_scale=home_scale)


def default_action(p):
    return "work_onsite" if p["group"] != "notwork" and not p["can_wfh"] else "stay_home"


def run_simulation(cfg, condition, seed, personas=None, backend=None, model_key=None,
                   cm=None, root=ROOT, tag="", resume=True, verbose=True):
    assert condition in "ABCD"
    uses_llm = condition in "CD"
    if uses_llm:
        assert personas is not None and backend is not None, "conditions C/D need personas and a backend"
    N = float(cfg["population"]["N"])
    beta0, sigma, gamma = seir.rates(cfg["disease"])
    days = cfg["sim"]["days"]
    cm = cm or contact_model_from_cfg(cfg, root)
    pol = cfg["policy"]
    actions = cfg["actions"]

    settings = {"condition": condition, "model_key": model_key, "seed": seed, "tag": tag,
                "backend": getattr(backend, "name", None), "cfg": cfg,
                "contacts": {"c_out": cm.c_out, "home": cm.home, "w_home": cm.w_home},
                "personas": config_hash(personas.to_dict("records")) if personas is not None else None}
    store = RunStore(root, make_run_id(condition, model_key, seed, settings))
    if resume and store.is_complete():
        if verbose:
            print(f"[{store.run_id}] already complete, skipping")
        return store

    ckpt = store.load_json("state.json") if resume else None
    if ckpt:
        store.truncate_after(ckpt["day"])
        state, hist, level, prev = ckpt["seir"], ckpt["reported_hist"], ckpt["level"], ckpt["prev_actions"]
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng"]
        health = AgentHealth.from_dict(ckpt["health"]) if ckpt["health"] else None
        day0, elapsed_total = ckpt["day"], ckpt["elapsed"]
        if verbose:
            print(f"[{store.run_id}] resuming at day {day0}")
    else:
        store.reset()
        store.save_json("config.json", settings)
        state, hist, level = seir.initial_state(cfg["population"]), [], pol["initial_level"]
        rng = np.random.default_rng(seed)
        health = AgentHealth(len(personas), state, N, cfg["disease"]["symptomatic_frac"], rng) if uses_llm else None
        prev = [default_action(p) for p in personas.to_dict("records")] if uses_llm else None
        day0, elapsed_total = 0, 0.0

    recs_p = personas.to_dict("records") if uses_llm else []
    ptexts = [persona_text(p) for p in recs_p]
    weights = personas["weight"].tolist() if uses_llm else None

    for day in range(day0, days):
        t_day = time.time()
        report = case_report(hist, N)
        row = {"day": day}

        if condition == "D" and day % pol["leader_every_days"] == 0:
            work_share = np.mean([a == "work_onsite" for a in prev]) if prev else 0.0
            prompt = build_leader_prompt(day, N, report, work_share, level, pol)
            recs, sec = decide_batch(backend, [prompt], list(pol["levels"]), [level], cfg["llm"],
                                     seed=seed * 7919 + day, key="level")
            level = recs[0]["choice"]
            store.append_jsonl("leader.jsonl", [{"day": day, "prompt": prompt, **recs[0], "seconds": sec}])
        qp = q_policy(level, pol) if condition == "D" else 1.0
        policy_text = pol["levels"][level]["text"] if condition == "D" else NO_POLICY_TEXT

        if uses_llm:
            allowed = allowed_actions(actions, level, pol) if condition == "D" else actions
            prompts = [build_resident_prompt(ptexts[i], day, health.health_text(i), report, policy_text, allowed)
                       for i in range(len(recs_p))]
            fallbacks = [a if a in allowed else "stay_home" for a in prev]
            recs, sec = decide_batch(backend, prompts, list(allowed), fallbacks, cfg["llm"],
                                     seed=seed * 100003 + day * 1009)
            acts = [r["choice"] for r in recs]
            shares = cm.shares(acts, weights)
            q_out = cm.q_out(shares)
            row.update({f"share_{a}": shares[a] for a in cm.actions})
            row.update(n_retried=sum(r["retried"] for r in recs), n_fallback=sum(r["fallback"] for r in recs),
                       llm_seconds=sec)
        elif condition == "B":
            q_out = rule_based_m(np.mean(hist[-7:]) if hist else 0.0, pol["alpha"])
        else:
            q_out = 1.0

        beta = cm.beta_eff(beta0, q_out, qp)
        I_frac = state["I"] / N
        new = seir.step(state, beta, sigma, gamma, N)

        if uses_llm:
            health.step(rng, beta0, I_frac, [cm.relative_contact(a, qp) for a in acts], sigma, gamma)
            store.append_jsonl("decisions.jsonl", [
                {"day": day, "agent_id": recs_p[i]["agent_id"], "group": recs_p[i]["group"],
                 "health": health.health_text(i), "prompt": prompts[i], **r}
                for i, r in enumerate(recs)])
            row["agents_symptomatic"] = int(health.symptomatic.sum())

        row.update(q_out=q_out, q_policy=qp, level=level if condition == "D" else None, beta_eff=beta,
                   **{k: new[k] for k in ("S", "E", "I", "R", "cum_inf", "cum_rep", "new_infections", "new_reported")},
                   reported_per100k=new["new_reported"] * 1e5 / N, seconds=time.time() - t_day)
        store.append_daily(row)
        hist.append(new["new_reported"])
        state = {k: new[k] for k in seir.KEYS}
        prev = acts if uses_llm else None
        elapsed_total += time.time() - t_day
        store.save_json("state.json", {"day": day + 1, "seir": state, "reported_hist": hist, "level": level,
                                       "prev_actions": prev, "health": health.to_dict() if health else None,
                                       "rng": rng.bit_generator.state, "elapsed": elapsed_total})
        if verbose and uses_llm:
            print(f"[{store.run_id}] day {day:>3}  q_out={q_out:.3f}  beta={beta:.4f}  "
                  f"fallback={row['n_fallback']}  {row['seconds']:.1f}s")

    store.save_json("runtime.json", {"total_seconds": elapsed_total, "days": days,
                                     "n_agents": len(recs_p), "env": environment_info()})
    if verbose:
        print(f"[{store.run_id}] done in {elapsed_total:.0f}s -> {store.dir}")
    return store
