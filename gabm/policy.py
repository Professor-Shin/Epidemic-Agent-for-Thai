"""Policy effects: rule-based behaviour (condition B) and the weekly LLM leader (condition D).

The leader chooses a bounded level. A level sets q_policy = 1 - k*d (per-contact transmission reduction),
the measure text residents see, and which actions are banned. It never touches the SEIR state directly.
"""

LEADER_TEMPLATE = """You are the provincial governor in a synthetic epidemic scenario set in a province in Thailand (population {N:,}). Each week you set the level of public-health measures for the next 7 days. You must balance health and the livelihoods of residents, many of whom lose income when they cannot work.

Weekly report (day {day}):
- 7-day average of newly reported cases: {avg7_per100k} per 100,000 people per day
- Trend compared with the previous week: {trend}
- Share of residents who went out to work on-site last week: {work_share:.0%}
- Current level: {current}

Available levels:
{level_list}

Reply with JSON only, no other text:
{{"level": "<one of: {level_names}>", "reason": "<at most 20 words>"}}"""


def rule_based_m(cases7, alpha):
    """Aoki & Ghaffarzadegan behavioural modifier m = 1 / (1 + alpha * weekly cases), used as q_out in condition B."""
    return 1.0 / (1.0 + alpha * cases7)


def q_policy(level, policy_cfg):
    return 1.0 - policy_cfg["k"] * policy_cfg["levels"][level]["d"]


def allowed_actions(actions, level, policy_cfg):
    banned = set(policy_cfg["levels"][level].get("banned", []))
    return {a: d for a, d in actions.items() if a not in banned}


def build_leader_prompt(day, N, report, work_share, current, policy_cfg):
    levels = policy_cfg["levels"]
    level_list = "\n".join(f"- {k}: {v['text']}" for k, v in levels.items())
    return LEADER_TEMPLATE.format(N=int(N), day=day, work_share=work_share, current=current,
                                  level_list=level_list, level_names=", ".join(levels), **report)
