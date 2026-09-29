"""Prompt construction for the LLM behavioural layer.

Three things are deliberately separated so that experimental arms differ in
exactly one thing at a time:

  * WHO the agent is        -> persona block (generic vs Thai-parameterised)
  * WHAT the agent knows    -> context block (health / household / news / policy)
  * WHAT the agent must do  -> the fixed action schema in actions.py

The Thai persona block states behavioural constructs as *numbers rendered into
words*, not as an ethnic label, which is the correction the Epi-LLM 2026 work
argues for: a geographic label alone does not produce differentiated behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.thaigabm.actions import SCHEMA_HINT
from src.thaigabm.evidence import BEHAVIOURAL_CONSTRUCTS
from src.thaigabm.information import POLICY_PROMPT, InformationState
from src.thaigabm.persona import Persona

CONSTRUCT_LABEL = {c.name: c.prompt_label for c in BEHAVIOURAL_CONSTRUCTS}

SYSTEM_GENERIC = (
    "You role-play one ordinary resident of a country experiencing a "
    "respiratory disease outbreak. You are given your profile and today's "
    "situation. Decide how you will actually spend today. Reason like a real "
    "person weighing money, health, and obligations -- not like a public "
    "health official. Never mention that you are an AI or a simulation. "
    "Output only the JSON object requested."
)

SYSTEM_THAI = (
    "You role-play one ordinary resident of Thailand during a respiratory "
    "disease outbreak. You are given your profile, including how strongly you "
    "personally feel a set of pressures, and today's situation. Decide how you "
    "will actually spend today. Weigh income against health, and take your "
    "household -- especially any elderly relatives you live with -- into "
    "account. Behave like this specific person, not like an ideal citizen: "
    "people who cannot afford to stop working usually do not stop working. "
    "Never mention that you are an AI or a simulation. Output only the JSON "
    "object requested."
)


def _level_word(v: float) -> str:
    if v >= 0.80:
        return "very high"
    if v >= 0.60:
        return "high"
    if v >= 0.40:
        return "moderate"
    if v >= 0.20:
        return "low"
    return "very low"


@dataclass
class ArmSpec:
    """Which information channels an experimental arm switches on."""

    thai_context: bool = True
    include_self_health: bool = True
    include_household: bool = True
    include_community_news: bool = True
    include_policy: bool = True
    ablate: tuple[str, ...] = ()
    language: str = "en"          # "en" | "th"

    def system_instruction(self) -> str:
        return SYSTEM_THAI if self.thai_context else SYSTEM_GENERIC


# --------------------------------------------------------------------------
# Persona blocks
# --------------------------------------------------------------------------
def persona_block(p: Persona, arm: ArmSpec) -> list[str]:
    if not arm.thai_context:
        # The source paper's persona: identity + Big Five + a job.
        traits = "; ".join(f"{k}: {v}" for k, v in p.big_five.items())
        return [
            f"You are {p.name}, {p.age} years old.",
            f"Your personality: {traits}.",
            f"You work as a {p.stratum_obj.label} and you go to work to earn money.",
        ]

    st = p.stratum_obj
    lines = [
        f"You are {p.name}, {p.age} years old, living in Thailand.",
        f"Work: {st.label}. "
        + ("You can do your job from home." if p.wfh_ability > 0.6
           else "Your work cannot be done from home; if you do not go out, you earn nothing."
           if p.wfh_ability < 0.25 else
           "Some of your work can be done from home, but not all of it."),
        f"Household: you live with {p.household_size - 1} other people."
        + (" One of them is over 60 and lives under your roof." if p.elderly_in_household else "")
        + (" There are young children at home." if p.children_in_household else ""),
    ]
    feelings = []
    for name, label in CONSTRUCT_LABEL.items():
        if name in arm.ablate:
            continue
        feelings.append(f"- {label}: {_level_word(p.c(name))}")
    if feelings:
        lines.append("How you personally feel about things right now:")
        lines.extend(feelings)
    return lines


# --------------------------------------------------------------------------
# Context block
# --------------------------------------------------------------------------
def context_block(
    p: Persona,
    arm: ArmSpec,
    day: int,
    symptom_text: str,
    household: dict,
    info: InformationState,
    total_population: int,
) -> list[str]:
    lines = [f"Today is day {day} of the outbreak."]

    if arm.include_self_health:
        lines.append(f"Physically, today you {symptom_text}.")

    if arm.include_household and arm.thai_context:
        if household.get("someone_sick_at_home"):
            n = household["n_sick_at_home"]
            lines.append(
                f"{n} person in your household is coughing or feverish."
                if n == 1 else
                f"{n} people in your household are coughing or feverish."
            )
        if household.get("elderly_at_home"):
            lines.append(
                "Your elderly relative at home would be in serious danger if you "
                "brought the infection back."
            )

    if arm.include_community_news:
        lines.append(info.news_line(total_population))

    if arm.include_policy:
        lines.append(POLICY_PROMPT.get(info.policy, POLICY_PROMPT["none"]))

    return lines


DECISION_QUESTION = (
    "Decide, for today only:\n"
    "  - will you go out to work?\n"
    "  - will you meet friends or go to a social gathering?\n"
    "  - will you visit relatives outside your household?\n"
    "  - will you wear a mask when you are around other people?\n"
    "  - will you go to a clinic or hospital to get checked?\n"
)


def build_prompt(
    p: Persona,
    arm: ArmSpec,
    day: int,
    symptom_text: str,
    household: dict,
    info: InformationState,
    total_population: int,
) -> str:
    parts = persona_block(p, arm)
    parts += [""]
    parts += context_block(p, arm, day, symptom_text, household, info, total_population)
    parts += ["", DECISION_QUESTION, SCHEMA_HINT]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Thai-language variant (for the RQ4 prompt-robustness experiment: same
# content, different language, so any difference is a language effect).
# --------------------------------------------------------------------------
SYSTEM_THAI_TH = (
    "คุณกำลังสวมบทบาทเป็นคนไทยคนหนึ่งในช่วงที่มีโรคระบาดทางเดินหายใจ "
    "คุณจะได้รับข้อมูลส่วนตัวของคุณและสถานการณ์วันนี้ "
    "ให้ตัดสินใจว่าวันนี้คุณจะใช้ชีวิตอย่างไรจริง ๆ โดยชั่งน้ำหนักระหว่างรายได้ สุขภาพ "
    "และภาระหน้าที่ต่อครอบครัว โดยเฉพาะผู้สูงอายุที่อยู่บ้านเดียวกัน "
    "ให้ตอบเป็น JSON ตามรูปแบบที่กำหนดเท่านั้น"
)

_TH_LEVEL = {"very high": "สูงมาก", "high": "ค่อนข้างสูง", "moderate": "ปานกลาง",
             "low": "ค่อนข้างต่ำ", "very low": "ต่ำมาก"}
_TH_LABEL = {
    "economic_pressure": "แรงกดดันทางการเงินที่ต้องหารายได้",
    "family_obligation": "ความรู้สึกรับผิดชอบต่อครอบครัวและผู้ใหญ่ในบ้าน",
    "perceived_severity": "ความเชื่อว่าโรคนี้อันตราย",
    "perceived_susceptibility": "ความเชื่อว่าตัวเองมีโอกาสติดเชื้อ",
    "institutional_trust": "ความไว้วางใจต่อรัฐและหน่วยงานสาธารณสุข",
    "norm_sensitivity": "ความใส่ใจว่าคนรอบข้างทำอย่างไร",
    "stigma_sensitivity": "ความกลัวถูกตำหนิหรือถูกรังเกียจถ้าติดเชื้อ",
    "social_preference": "ความชอบพบปะสังสรรค์นอกบ้าน",
    "health_literacy": "ความสามารถในการเข้าใจข้อมูลสุขภาพ",
}


def build_prompt_th(p: Persona, arm: ArmSpec, day: int, symptom_text_en: str,
                    household: dict, info: InformationState,
                    total_population: int) -> str:
    st = p.stratum_obj
    lines = [
        f"คุณคือ {p.name} อายุ {p.age} ปี อาศัยอยู่ในประเทศไทย",
        f"อาชีพ: {st.label}"
        + ("  ทำงานจากบ้านได้" if p.wfh_ability > 0.6
           else "  ทำงานจากบ้านไม่ได้ ถ้าไม่ออกไปก็ไม่มีรายได้" if p.wfh_ability < 0.25
           else "  ทำงานจากบ้านได้บางส่วน"),
        f"ที่บ้านมีคนอื่นอีก {p.household_size - 1} คน"
        + ("  และมีผู้สูงอายุ 60 ปีขึ้นไปอยู่ด้วย" if p.elderly_in_household else ""),
    ]
    feelings = [f"- {_TH_LABEL[n]}: {_TH_LEVEL[_level_word(p.c(n))]}"
                for n in _TH_LABEL if n not in arm.ablate]
    if feelings:
        lines.append("ความรู้สึกของคุณตอนนี้:")
        lines += feelings
    lines.append("")
    lines.append(f"วันนี้เป็นวันที่ {day} ของการระบาด")
    if arm.include_self_health:
        lines.append(f"วันนี้ร่างกายคุณ: {symptom_text_en}")
    if arm.include_household and household.get("someone_sick_at_home"):
        lines.append(f"มีคนในบ้าน {household['n_sick_at_home']} คนที่ไอหรือมีไข้")
    if arm.include_community_news:
        lines.append(info.news_line(total_population))
    if arm.include_policy:
        lines.append(POLICY_PROMPT.get(info.policy, POLICY_PROMPT["none"]))
    lines += ["", "ตัดสินใจสำหรับวันนี้: จะออกไปทำงานไหม จะไปสังสรรค์ไหม "
              "จะไปเยี่ยมญาติต่างบ้านไหม จะใส่หน้ากากไหม และจะไปหาหมอไหม", SCHEMA_HINT]
    return "\n".join(lines)
