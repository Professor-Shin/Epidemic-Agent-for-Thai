# แนวทางการต่อยอดงาน Epidemic Modeling with Generative Agents สู่ Thai-context GABM-SEIR

อ่านแล้วครับ และผมคิดว่าไอเดียนี้ **ต่อยอดได้ดีมาก** แต่มีจุดสำคัญอย่างหนึ่งที่อยากแก้กรอบตั้งแต่ต้น:

> งานต้นฉบับไม่ได้เป็น “GABM + mechanistic SEIR” โดยตรง แต่เป็น **GABM ที่ใช้ LLM ตัดสินใจ + ABM ที่จำลองการแพร่เชื้อในระดับบุคคล**  
> ดังนั้นโปรเจคของเราสามารถสร้าง contribution ใหม่โดย **แยก behavioral model ออกจาก disease model แล้ว coupling เข้ากับ mechanistic SEIR อย่างชัดเจน** แทน [1]

และผมคิดว่า “ประเทศไทย” ไม่ควรเป็นแค่การเปลี่ยนชื่อเมืองจาก Dewberry Hollow เป็น Bangkok/Thailand เพราะงานล่าสุดในสาย Epi-LLM พบว่า **ใส่ geographic label อย่างเดียวไม่ได้ทำให้เกิดพฤติกรรมที่แตกต่างทางวัฒนธรรมอย่างมีนัยสำคัญ** ต้อง parameterize attitudes/norms ให้ชัดเจนด้วย [2]

---

# 1. ก่อนอื่น มาดูว่า paper ต้นฉบับทำอะไรจริง ๆ

งานของ Williams et al. (2023) มี architecture ประมาณนี้:

```text
             ┌──────────────────────────┐
             │       LLM / GAI          │
             │ Persona + health + news  │
             └────────────┬─────────────┘
                          │
                   stay home / go out
                          │
                          ▼
             ┌──────────────────────────┐
             │      Agent-Based Model   │
             │ contacts + transmission  │
             └────────────┬─────────────┘
                          │
                    infection state
                          │
                          ▼
                feedback to LLM
```

แต่ละ agent มีชื่อ อายุ Big Five personality biography และ memory จากนั้นทุกวันให้ LLM ตัดสินใจว่าจะอยู่บ้านหรือออกไปทำงาน โดย input อาจรวมอาการของตัวเองและจำนวนผู้ติดเชื้อในเมืองด้วย [1]

ตัว disease model ของเขาเป็น agent-based จริง ๆ เช่น คนที่ออกจากบ้านจะถูกสุ่มให้เจอกับคนอื่นตาม contact rate แล้วจึงคำนวณ transmission จาก infected → susceptible [1]

เขาทดลอง 3 condition:

```text
1. Base
   Persona
      ↓
   LLM decision

2. Self-health feedback
   Persona + symptoms
          ↓
       LLM decision

3. Full feedback
   Persona + symptoms + community cases/news
                  ↓
             LLM decision
```

ผลที่น่าสนใจคือ เมื่อ agents ได้ข้อมูลเกี่ยวกับสุขภาพและจำนวนผู้ติดเชื้อ พวกเขามีแนวโน้มลด mobility ทำให้ epidemic peak ลดลง และเกิดหลายระลอกได้จาก feedback loop ระหว่าง disease กับ behavior [1]

แต่จุดอ่อนที่ชัดมากคือ computational cost — งานเดิมรายงานว่า simulation 1,000 agents ใช้เวลามากกว่า 90 ชั่วโมง เพราะต้องยิง API หลายหมื่นครั้ง [1]

---

# 2. เพราะฉะนั้น โปรเจคของเราควร “เปลี่ยนตรงไหน”

ผมจะแนะนำให้เปลี่ยน **4 ชั้นหลัก**

## Layer 1 — Disease dynamics

จาก

> Agent-based transmission model

เป็น

> **Mechanistic SEIR model**

เช่น

\[
\frac{dS}{dt}=-\beta_t\frac{SI}{N}
\]

\[
\frac{dE}{dt}=\beta_t\frac{SI}{N}-\sigma E
\]

\[
\frac{dI}{dt}=\sigma E-\gamma I
\]

\[
\frac{dR}{dt}=\gamma I
\]

ตรงนี้คือส่วนที่ **ไม่ต้องให้ LLM คิด**

LLM ไม่ควรเป็นคนบอกว่า “วันนี้มีคนติดเชื้อ 120 คน เพราะฉะนั้น infection rate = ...”

ให้ mechanistic model เป็นคนจัดการ epidemiology ทั้งหมด

---

# 3. แล้ว LLM agents จะส่งอะไรให้ SEIR?

นี่คือส่วนที่ผมคิดว่าเป็นหัวใจของ thesis/project

แต่ละ agent \(i\) จะมี decision

\[
a_{i,t}
\]

เช่น

```text
0 = stay home
1 = go to work
2 = social activity
3 = visit family
4 = seek healthcare
```

หรือถ้าต้องการเริ่มง่ายมาก:

```text
0 = isolate
1 = normal activity
```

จาก agents ทั้งหมด เราคำนวณ **population-level behavioral effect**

เช่น

\[
M_t = \frac{1}{N}\sum_i a_{i,t}
\]

แล้วเอาไป modify transmission:

\[
\beta_t = \beta_0 M_t
\]

หรือดีกว่านั้น:

\[
\beta_t = \beta_0 f(M_t, mask_t, distancing_t, workplace_t, social_t)
\]

ดังนั้น architecture จะกลายเป็น:

```text
                  ┌────────────────────────┐
                  │      Agent Persona     │
                  │ age / job / family /   │
                  │ risk / social norms    │
                  └───────────┬────────────┘
                              │
                    + disease information
                    + social information
                    + economic pressure
                              │
                              ▼
                  ┌────────────────────────┐
                  │          LLM           │
                  │ behavioral reasoning   │
                  └───────────┬────────────┘
                              │
                        agent decisions
                    a1, a2, a3, ..., aN
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Behavioral Aggregator  │
                  │ effective mobility /   │
                  │ contact / compliance   │
                  └───────────┬────────────┘
                              │
                          β(t), etc.
                              │
                              ▼
                  ┌────────────────────────┐
                  │      SEIR Model        │
                  │ S → E → I → R          │
                  └───────────┬────────────┘
                              │
                       epidemic state
                              │
                              └─────── feedback ────► LLM
```

**อันนี้แตกต่างจาก paper ต้นฉบับอย่างมีสาระ** เพราะเราให้ LLM เป็น behavioral submodel และ SEIR เป็น mechanistic epidemiological submodel อย่างชัดเจน [1]

---

# 4. แต่มีปัญหาใหญ่หนึ่งข้อ: ถ้าเป็น SEIR ธรรมดา แล้ว agents มีไว้ทำไม?

นี่เป็นคำถามที่อาจโดนอาจารย์ถามแน่นอน

เพราะ classic SEIR ไม่ได้สนใจ “นาย A ไปทำงาน นาย B อยู่บ้าน”

มันสนใจแค่ aggregate population:

\[
S,E,I,R
\]

ดังนั้นถ้าเราทำ

```text
1,000 LLM agents
       ↓
     63% stay home
       ↓
SEIR beta × 0.37
```

จริง ๆ เราสามารถลดเหลือ 10 agents หรือแม้แต่ rule-based function ก็ได้

ดังนั้นผมแนะนำให้ใช้ **stratified SEIR** แทน SEIR เดียว ถ้าต้องการให้ project มีความลึกทาง research มากขึ้น

ตัวอย่างแบ่ง population เป็นกลุ่ม:

```text
              Thailand Population

       ┌──────────┬──────────┬──────────┐
       │ Students │ Workers  │ Elderly  │
       └────┬─────┴────┬─────┴────┬─────┘
            │          │          │
         SEIR-1     SEIR-2     SEIR-3
```

และแต่ละ subgroup มี behavioral modifier จาก agents

```text
Students    → β_student(t)
Workers     → β_worker(t)
Elderly     → β_elderly(t)
```

หรือแบ่งตาม lifestyle:

```text
office worker
informal worker
student
retired
healthcare worker
service worker
```

แบบนี้ **persona → behavior → subgroup transmission → total epidemic** มี causal chain ที่ชัดกว่ามาก

---

# 5. จุดที่สำคัญที่สุด: “Thai context” ต้องไม่ใช่แค่ prompt ภาษาไทย

อันนี้ผมอยากเน้นมาก

งาน Epi-LLM ปี 2026 ที่ออกมาหลัง paper ที่คุณส่งมาทำการทดลองเรื่องนี้โดยตรง และพบว่า geographic labels เช่น “China / Iraq / Kenya / UK” อย่างเดียว **ไม่สามารถทำให้ LLM agents แสดง culturally differentiated behavior ได้อย่างน่าเชื่อถือ** แต่ explicit attitudinal parameterisation ช่วยได้ [2]

เพราะฉะนั้นแบบนี้:

```text
You are a Thai person.
You live in Bangkok.
You care about your family.
```

ผมจะ **ไม่ถือว่าเป็น cultural modeling**

ต้องสร้างตัวแปร behavioral/cultural ที่สามารถ operationalize ได้จริง

เช่น

```text
Family obligation
Economic vulnerability
Trust in government
Trust in health authorities
Perceived severity
Perceived susceptibility
Social responsibility
Stigma sensitivity
Preference for social interaction
Work flexibility
Ability to work from home
Risk tolerance
Mask adherence
Norm conformity
```

แล้วแต่ละ agent สุ่ม/parameterize ตัวแปรเหล่านี้

ตัวอย่าง:

```python
Agent(
    age=42,
    occupation="street_food_vendor",
    income_level="low",
    family_dependents=3,
    work_from_home=False,

    perceived_severity=0.72,
    perceived_susceptibility=0.40,
    government_trust=0.55,
    social_norm_sensitivity=0.84,
    family_obligation=0.91,
    economic_pressure=0.95,
)
```

แล้วจึงเอาค่าเหล่านี้เข้า LLM prompt

---

# 6. และตรงนี้มีข้อมูลไทยที่เอามา ground model ได้จริง

ตรงนี้ผมคิดว่า project จะดีขึ้นมาก เพราะเราไม่จำเป็นต้อง “เดาว่าคนไทยเป็นอย่างไร”

มี survey ของประเทศไทยที่ศึกษาการตอบสนองต่อ isolation, quarantine และ social distancing โดยตรง เช่น survey 1,020 คนในปี 2020 พบว่ามากกว่า 80% รายงานว่า **loss of income** เป็น concern สำคัญ และ 92% ระบุว่าปรับพฤติกรรมทางสังคมก่อนมาตรการรัฐบาล ขณะที่ 94% รายงาน social distancing และ 97% ใช้ PPE [3]

อีกงานหนึ่งชี้ถึงบทบาทของ **family/community ties** และ social interaction ในบริบทไทย รวมถึงปัญหา stigma และการ conceal infection ที่อาจกระทบการตรวจและการรายงานโรค [4]

นั่นหมายความว่าเราสามารถสร้าง persona จากข้อมูลที่มีหลักฐาน เช่น

```text
economic constraint
family dependency
social-contact preference
risk perception
compliance tendency
stigma sensitivity
```

แทนการใส่ stereotype ของ “คนไทย” เข้าไปตรง ๆ

---

# 7. Persona ของ agent ควรเปลี่ยนจาก paper เดิมมาก

Paper เดิมใช้ประมาณ:

```text
Name
Age
Big Five
Biography
```

แต่สำหรับไทย ผมเสนอ:

### Static attributes

```text
age
gender
occupation
income group
urban/rural
household size
children
elderly in household
work-from-home ability
region
```

### Behavioral attributes

```text
risk perception
economic pressure
family obligation
social interaction preference
trust in authority
health literacy
stigma sensitivity
norm sensitivity
```

### Dynamic state

```text
health status
known exposure
symptoms
local case count
family infection
policy restrictions
recent news
financial loss
days in isolation
```

นี่จะทำให้ agent ไม่ได้แค่มี “personality” แต่มี **decision-relevant state**

---

# 8. ผมจะเพิ่ม “ครอบครัว” เข้าไปด้วย

นี่อาจเป็นหนึ่งในจุด novelty ที่ดีมาก

Paper ต้นฉบับค่อนข้าง individualistic:

```text
Person → Should I go to work?
```

แต่เราสามารถ model:

```text
                 Household
                    │
       ┌────────────┼────────────┐
       │            │            │
    Father        Mother       Child
       │            │            │
       └────── shared risk ──────┘
```

เช่น agent รู้ว่า

> “แม่อายุ 72 ปีอยู่บ้านเดียวกัน”

การตัดสินใจจึงเปลี่ยนจาก

> “ฉันเสี่ยงติดเชื้อ”

เป็น

> “ถ้าฉันติด อาจนำเชื้อกลับไปให้แม่”

ตรงนี้ทำให้ **social structure → agent reasoning → disease dynamics** มีความน่าสนใจมาก

งานเชิงคุณภาพในประเทศไทยเกี่ยวกับมาตรการ COVID-19 ก็รายงานความสำคัญของความสัมพันธ์ในครอบครัวและชุมชน ตลอดจนผลกระทบของการลดการปฏิสัมพันธ์ทางสังคมและ stigma/shaming ซึ่งสนับสนุนว่าการ model household/social structure อาจเป็นองค์ประกอบที่มีประโยชน์สำหรับบริบทไทย [4]

---

# 9. อีกอย่างที่ควรเพิ่มคือ “economic trade-off”

อันนี้สำคัญมากใน context ไทย และมี empirical support ว่า economic loss เป็น concern สำคัญใน survey ไทย [3]

แทนที่จะถามแค่

> Stay home or go to work?

ให้ agent มี utility conflict:

```text
health risk
        ↕
economic survival
        ↕
family responsibility
```

ตัวอย่าง:

```text
Agent A
office worker
→ WFH available
→ high health concern
→ stay home

Agent B
street vendor
→ WFH impossible
→ high economic pressure
→ go outside

Agent C
caregiver
→ elderly parent at home
→ high family obligation
→ avoid crowded locations
```

LLM จะ reasoning บนข้อมูลที่ต่างกัน

นี่ทำให้ heterogeneity มีความหมายมากกว่าแค่ random Big Five

---

# 10. ผมแนะนำ experimental design แบบนี้

อย่าเริ่มด้วย “Thai vs American” อย่างเดียว เพราะมันจะกลายเป็น comparison ที่ตีความยาก

ใช้ **ablation study** ดีกว่า

### Experiment A — Classical SEIR

```text
SEIR
β = constant
```

เป็น baseline

### Experiment B — Rule-based behavioral SEIR

```text
SEIR
β(t) = function(case_count)
```

เพื่อดูว่า adaptive behavior แบบ mathematical rule ทำอะไรได้

### Experiment C — Generic GABM-SEIR

```text
LLM agents
generic personas
       ↓
behavior
       ↓
SEIR
```

### Experiment D — Thai-context GABM-SEIR

```text
LLM agents
+
Thai empirical behavioral parameters
       ↓
behavior
       ↓
SEIR
```

### Experiment E — Thai contextual factors ablation

เช่นเอาทีละ factor ออก:

```text
Full model

- family obligation
- economic pressure
- government trust
- stigma
```

แล้วดูว่า epidemic outcome เปลี่ยนอย่างไร

อันนี้จะทำให้ตอบคำถามทางวิทยาศาสตร์ได้ ไม่ใช่แค่ “LLM จำลองได้เหมือนคน”

---

# 11. Research Question ที่ผมคิดว่าแข็งแรง

ผมจะไม่ตั้งว่า

> “Can LLM simulate Thai people's behavior?”

เพราะกว้างเกินไปและพิสูจน์ยากมาก

ผมชอบประมาณนี้มากกว่า:

### RQ1
**How does LLM-driven adaptive human behavior affect epidemic dynamics when coupled with a mechanistic SEIR model?**

### RQ2
**Does explicit Thai behavioral and socio-cultural parameterization produce epidemic and behavioral dynamics that better match empirical Thai observations than generic LLM agents?**

### RQ3
**Which behavioral factors have the greatest influence on voluntary isolation, mobility, and epidemic outcomes in a Thai population?**

และถ้าจะให้เป็น thesis ที่มี ML/AI flavor เพิ่ม:

### RQ4
**How robust are simulated behavioral decisions across different LLM architectures and prompt configurations?**

อันนี้สำคัญมากเพราะ LLM แต่ละ model อาจให้ behavior ต่างกัน งาน Epi-LLM ปี 2026 พบว่า architecture ของ LLM เองมีผลต่อ simulated epidemic dynamics อย่างมีนัยสำคัญ และเสนอว่า low-variance models เหมาะกับ internal validity ขณะที่ high-variance models อาจสะท้อน stochasticity ของมนุษย์ได้มากกว่า [2]

---

# 12. สิ่งที่ผม “ไม่แนะนำ” ให้ทำ

## อย่าให้ LLM คุม SEIR โดยตรง

ไม่เอาแบบ:

```text
LLM:
"วันนี้ผู้ติดเชื้อเพิ่มขึ้น ดังนั้น beta = 0.13"
```

เพราะจะสูญเสีย mechanistic interpretability

LLM ควร control **behavior**

ไม่ใช่ control biology

---

## อย่าใช้ LLM 1 agent = 1 คนจริงทั้งประเทศ

สมมติ Thailand population 70 ล้านคนแล้วทำ 70 ล้าน agents เป็นไปไม่ได้

ควรทำ:

```text
Synthetic population
      ↓
representative agents
      ↓
behavioral distribution
      ↓
population-level SEIR
```

เช่น simulation 500–5,000 agents แล้ว calibrate distribution ให้ represent population strata

---

## อย่าใช้ cultural stereotype เป็น parameter

เช่น

```text
Thai = collectivist = 0.9
American = individualist = 0.2
```

แบบนี้ reviewer น่าจะยิงทันที

ต้องทำ

```text
literature / survey
        ↓
behavioral construct
        ↓
parameter distribution
        ↓
agents
```

และ recent Epi-LLM work ก็ชี้ตรง ๆ ว่าการใส่ geographic label อย่างเดียวไม่พอ [2]

---

# 13. Validation จะเป็นหัวใจของงานนี้เลย

นี่คือจุดที่ผมคิดว่า **project ของคุณสามารถเหนือกว่า paper ต้นฉบับได้**

Paper 2023 แสดงว่า agent ให้ behavior ที่ “plausible” และได้ epidemic patterns ที่น่าสนใจ แต่ยังไม่ได้ ground behavioral model กับข้อมูลพฤติกรรมของประชากรไทยอย่างจริงจัง [1]

และ literature ล่าสุดก็วิจารณ์ตรง ๆ ว่า **validation เป็นปัญหาหลักของ generative ABM** เพราะ LLM มี stochasticity และอาจมี cultural bias; การแสดงว่า behavior “ดูเหมือนมนุษย์” อย่างเดียวไม่เพียงพอสำหรับ operational validity [5]

ดังนั้นของเราควรมีอย่างน้อย 3 ระดับ:

```text
Level 1 — Behavioral validation
Agent decision vs Thai survey data

Level 2 — Aggregate validation
Simulated mobility/compliance vs observed data

Level 3 — Epidemiological validation
Simulated cases vs actual Thai epidemic curves
```

เช่น

\[
MAE_{cases}
\]

\[
RMSE_{cases}
\]

หรือ distribution similarity สำหรับ behavioral outcomes

---

# 14. ภาพรวมโปรเจคที่ผมคิดว่า “ใช่”

ผมจะวางชื่อชั่วคราวประมาณ:

> **ThaiGABM-SEIR: A Generative Agent-Based Behavioral Model Coupled with a Mechanistic SEIR Model for Culturally Grounded Epidemic Simulation in Thailand**

Architecture:

```text
                    ┌───────────────────────┐
                    │ Thai Population Data  │
                    │ surveys / census /    │
                    │ mobility / behavior   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Synthetic Population  │
                    │ persona + demographics │
                    │ behavioral attributes  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      LLM Agents       │
                    │                       │
                    │ "go work?"            │
                    │ "isolate?"             │
                    │ "visit family?"       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Behavioral Aggregator  │
                    │                       │
                    │ mobility              │
                    │ contact reduction     │
                    │ compliance             │
                    └───────────┬───────────┘
                                │
                                ▼
                     β(t), contact modifier
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Mechanistic SEIR   │
                    │                       │
                    │ S → E → I → R         │
                    └───────────┬───────────┘
                                │
                                │ disease state
                                ▼
                    ┌───────────────────────┐
                    │ Information Environment│
                    │ cases / news / policy │
                    └───────────┬───────────┘
                                │
                                └──────► Agents
```

และในปี 2026 มีงานที่ใกล้กับแนวคิดนี้ขึ้นเรื่อย ๆ เช่น Aoki & Ghaffarzadegan ใช้ generative AI agent ในการตัดสินใจเชิงนโยบายโดยฝัง agent ไว้ใน **SEIR environment** ซึ่งแสดงให้เห็นว่า coupling ระหว่าง generative agent กับ mechanistic epidemic model เป็นทิศทางที่กำลังพัฒนาอยู่แล้ว [6]

---

# 15. ที่จริงผมคิดว่า contribution ของเราควรเป็น 3 อย่าง

ไม่ใช่แค่

> “เอา paper นี้มาทำกับคนไทย”

แต่ควร claim ว่า:

**Contribution 1 — Modeling**

สร้าง framework ที่แยก

> LLM behavioral model ↔ mechanistic SEIR

อย่างชัดเจน

**Contribution 2 — Cultural grounding**

สร้าง Thai-specific behavioral parameterization จาก empirical evidence แทนการพึ่ง cultural stereotype หรือ geographic prompt เพียงอย่างเดียว

**Contribution 3 — Validation**

ทดสอบว่า

> generic LLM → Thai-grounded LLM → empirical data

ส่งผลต่อทั้ง **individual behavior และ epidemic trajectory** อย่างไร

สามข้อนี้ทำให้มันเริ่มกลายเป็น research project จริง ๆ มากกว่าการ replicate paper

---

## และผมว่ามีอีกจุดที่น่าสนใจมากสำหรับคุณ

จาก background ที่คุณทำ ML/Data Science + งาน bioinformatics อยู่ ผม **ไม่แนะนำให้เริ่มจาก “สร้างระบบ LLM agents ใหญ่ ๆ” ก่อน**

ควรเริ่มจาก:

```text
Phase 1
Classical SEIR
       ↓
Phase 2
Rule-based behavioral SEIR
       ↓
Phase 3
LLM behavioral layer
       ↓
Phase 4
Thai behavioral parameterization
       ↓
Phase 5
Calibration + validation
       ↓
Phase 6
Policy experiments
```

จะทำให้รู้ชัดว่า performance ที่ดีขึ้นมาจาก **LLM จริง ๆ** หรือแค่เพิ่ม behavioral feedback เข้าไป

และที่สำคัญคือ computational cost จะต่ำกว่าการทำแบบ paper เดิมมาก เพราะเราไม่จำเป็นต้องให้ LLM control ทุก component ของ disease simulation [1]

**สรุปแบบสั้นที่สุด:** ผมคิดว่าโปรเจคนี้ควร evolve จาก

> **“LLM agents simulate an epidemic”**

ไปเป็น

> **“Empirically grounded Thai LLM agents provide a dynamic behavioral layer for a mechanistic SEIR model.”**

อันหลังมี research question, novelty และ validation story ที่แข็งแรงกว่ามาก

อีกอย่างที่สำคัญคือ ตอนนี้มี **Epi-LLM (2026)** ที่ออกมาหลัง paper ต้นฉบับและแตะประเด็น SEIR baseline + empirical behavioral grounding + cultural parameterization โดยตรงแล้ว ดังนั้นเราควรอ่านงานนั้นเป็น “related work ที่ใกล้ที่สุด” ด้วย ไม่อย่างนั้น proposal อาจดูเหมือนข้าม literature ช่วง 2023–2026 ไป [2]

---

# References

[1] Williams, R., Hosseinichimeh, N., Majumdar, A., & Ghaffarzadegan, N. (2023). *Epidemic Modeling with Generative Agents*. arXiv:2307.04986. https://arxiv.org/abs/2307.04986

[2] Ferenz, P., Keeling, A., O'Keefe, T., Stigliano, L., Di Lauro, F., Colubri, A., & Panovska-Griffiths, J. (2026). *The Epi-LLM Framework: probing LLM behavioral priors through epidemiological agent-based models*. arXiv:2606.02867. https://arxiv.org/abs/2606.02867

[3] Pan-ngum, W., Pooma, W., Parnell, S., et al. (2021). *Perspectives on public health interventions in the management of the COVID-19 pandemic in Thailand*. BMC Public Health. https://pmc.ncbi.nlm.nih.gov/articles/PMC8283548/

[4] *“Like a wake-up call for humankind”: Views, challenges, and coping strategies related to public health measures during the first COVID-19 lockdown in Thailand*. (2022/2023). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC10021331/

[5] Larooij, M., & Törnberg, P. (2026). *Validation is the central challenge for generative social simulation: a critical review of LLMs in agent-based modeling*. Artificial Intelligence Review, 59, 15. https://pmc.ncbi.nlm.nih.gov/articles/PMC12627210/

[6] Aoki, G., & Ghaffarzadegan, N. (2026). *AI Agents as Policymakers in Simulated Epidemics*. arXiv:2601.04245. https://arxiv.org/abs/2601.04245
