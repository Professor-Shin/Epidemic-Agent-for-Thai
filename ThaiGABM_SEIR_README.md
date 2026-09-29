# ThaiGABM-SEIR

**A generative agent-based behavioural model coupled to a mechanistic stratified SEIR model for culturally grounded epidemic simulation in Thailand.**

โปรเจคนี้ต่อยอดจากการ replicate เปเปอร์ Williams et al. (2023) *Epidemic Modeling with Generative Agents* (`2307.04986.pdf`, โค้ดเดิมอยู่ที่ `main.py` + `src/*.py` และเอกสารที่ [document.md](document.md)) ไปเป็นสถาปัตยกรรมตาม [ThaiGABM_SEIR_project_guideline.md](ThaiGABM_SEIR_project_guideline.md)

ความต่างหลักจากงานต้นฉบับ คือ **แยก behavioral model ออกจาก disease model อย่างชัดเจน**

| | Williams et al. 2023 | ThaiGABM-SEIR (โปรเจคนี้) |
|---|---|---|
| Disease dynamics | agent-based transmission | **mechanistic stratified SEIR** (6 strata × 4 settings) |
| หน้าที่ของ LLM | ตัดสินใจ + ขับ transmission ทางอ้อม | **ตัดสินใจพฤติกรรมอย่างเดียว** ไม่แตะ epidemiology |
| Persona | name, age, Big Five | demographics + household + **9 behavioural constructs ที่ ground กับ survey ไทย** |
| Action space | go / stay | work / social / family visit / mask / healthcare |
| ต้นทุน | ~90 ชม. ต่อ 1,000 agents | situation-keyed cache + representative agents (ดู `cache_stats` ในผลลัพธ์) |
| Validation | behaviour ดู plausible | **3 ระดับ** พร้อม metric ที่ตรวจสอบได้ |

---

## 1. สถาปัตยกรรม

```text
     Thai evidence (survey / census)        <- src/thaigabm/evidence.py
                  |
     Synthetic population + households      <- persona.py, strata.py
                  |
     ┌────────────▼────────────┐
     │  Behavioural model      │            <- behaviour.py, prompts.py
     │  LLM agents / rules     │               *** จุดเดียวที่ LLM ทำงาน ***
     └────────────┬────────────┘
          a_i,t (5 การตัดสินใจต่อคนต่อวัน)
                  |
     ┌────────────▼────────────┐
     │ Behavioural aggregator  │            <- aggregator.py
     │  phi^s_g(t), mask_g(t)  │
     └────────────┬────────────┘
                  |
     ┌────────────▼────────────┐
     │ Mechanistic SEIR        │            <- seir.py
     │  S -> E -> I -> R       │
     └────────────┬────────────┘
                  |
     ┌────────────▼────────────┐
     │ Information environment │            <- information.py
     │ reported cases / policy │
     └────────────┬────────────┘
                  └──── feedback ────► agents
```

หนึ่งวันของ simulation มี 6 ขั้น (`runner.py`) และ **LLM ปรากฏในขั้นเดียวเท่านั้น** — นี่คือ Contribution 1 ของโปรเจค

### สมการหลัก

Force of infection ของ stratum `g`:

```text
lambda_g(t) = sum_h  q * C_gh(t) * m_g(t) * m_h(t) * I_h / N_h

C_gh(t) = sum_s  C^s_gh * phi^s_g(t) * phi^s_h(t)      s ∈ {home, work, social, family_visit}
m_g(t)  = 1 - mask_efficacy * mask_coverage_g(t)
```

`phi` และ `mask` มาจาก agent ทั้งหมด ส่วน `q` calibrate ครั้งเดียวจาก target R0 ผ่าน **next-generation matrix** ของ contact structure ตอน baseline ดังนั้น "R0 = 2.5" มีความหมายเดียวกันทุก arm (ตรวจสอบใน `tests/test_thaigabm.py::test_r0_is_calibrated_exactly`)

Contact matrix ทุก setting รักษาเงื่อนไข reciprocity `N_g C_gh = N_h C_hg` (ทดสอบไว้แล้ว) ถ้าไม่รักษา force of infection จะไม่มีความหมายทางฟิสิกส์

---

## 2. เริ่มใช้งาน

```bash
pip install -r requirements.txt

# ทดสอบทั้ง pipeline แบบไม่เรียก API เลย
python3 run_thaigabm.py --arms ABCD --mock --agents 150 --days 200 --output results_thaigabm
python3 plot_thaigabm.py results_thaigabm

# ดูว่า prompt ที่ส่งให้ LLM หน้าตาเป็นอย่างไร (ถูกที่สุด และควรทำก่อนเสมอ)
python3 run_thaigabm.py --dump-prompt

# รันจริงด้วย Gemini ผ่าน ADK (ต้องมี GOOGLE_API_KEY ใน .env)
python3 run_thaigabm.py --arms D --agents 200 --days 180
```

รันเทสต์: `python3 -m pytest tests/ -q` (27 tests; ไม่ต้องใช้ network — มี ADK stub ให้)

---

## 3. Experimental design (guideline §10)

| Arm | คำสั่ง | คืออะไร |
|---|---|---|
| **A** | `--arms A` | Classical SEIR, beta คงที่, ไม่มี behaviour, ไม่มี policy — baseline |
| **B** | `--arms B` | Rule-based behavioural SEIR: beta(t) ตอบสนอง reported cases ด้วยสูตรที่เขียนมือ |
| **C** | `--arms C` | Generic GABM-SEIR: LLM agents ที่ใช้ persona บาง ๆ แบบเปเปอร์เดิม |
| **D** | `--arms D` | Thai-context GABM-SEIR: persona + 9 constructs + household |
| **E** | `--ablation` | ตัด construct ทีละตัวออกจาก **ทั้ง prompt และ parameter** |

**Arm B คือ control ที่สำคัญที่สุด** เพราะใช้ construct และ information channel ชุดเดียวกับ D ต่างกันแค่กลไกการให้เหตุผล ถ้า D ไม่ต่างจาก B อย่างมีนัยสำคัญ แปลว่า LLM ไม่ได้เพิ่มอะไร — ซึ่งเป็นคำตอบที่ต้องรายงานตามตรง

Robustness สำหรับ RQ4:

```bash
python3 run_thaigabm.py --model-sweep gemini-2.0-flash,gemini-2.5-flash   # ต่าง architecture
python3 run_thaigabm.py --language-sweep                                   # prompt EN vs TH
python3 run_thaigabm.py --arms D --seeds 1,2,3,4,5                         # stochastic replication
```

---

## 4. Thai context ไม่ใช่แค่เปลี่ยนชื่อเมือง

ตาม guideline §5 และข้อค้นพบของ Epi-LLM (2026) ว่า geographic label อย่างเดียวไม่พอ โปรเจคนี้ parameterize **9 behavioural constructs** ต่อ agent (`evidence.py`):

`economic_pressure`, `family_obligation`, `perceived_severity`, `perceived_susceptibility`, `institutional_trust`, `norm_sensitivity`, `stigma_sensitivity`, `social_preference`, `health_literacy`

แต่ละตัวสุ่มจาก Beta distribution ที่ค่ากลาง anchor กับหลักฐาน แล้ว shift ตาม stratum และ **ทุกตัวมี field `kind` บอกสถานะหลักฐาน**

- `survey` — อ่านจากสถิติ survey ที่ตีพิมพ์ (Pan-ngum et al. 2021, n=1,020)
- `qualitative` — ตีความจากงาน qualitative ไทย
- `assumption` — **สมมติฐานของเราเอง ไม่ใช่หลักฐาน** ต้องทำ sensitivity analysis

ทุกครั้งที่รัน ระบบจะเขียน `PARAMETER_PROVENANCE.md` ลง output directory เพื่อให้ตรวจสอบได้ว่าเลขไหนมาจากไหน — อันนี้ตอบข้อกังวลใน guideline §12 เรื่อง cultural stereotype โดยตรง

**โครงสร้างครัวเรือน** ก็เป็นส่วนหนึ่งของ Thai context ไม่ใช่แค่ attitude: agent อยู่ในครัวเรือนจริง มี intergenerational co-residence (`P_MULTIGENERATIONAL`) และรู้ว่ามีใครป่วยอยู่บ้านไหม ทำให้เกิด reasoning แบบ "ถ้าฉันติด อาจเอาเชื้อกลับไปให้แม่" ตาม guideline §8

---

## 5. Shadow individual states — จุดที่ต้องอธิบายกับอาจารย์

SEIR เป็น compartmental แต่ agent ต้องตอบว่า "วันนี้ฉันมีไข้ไหม" จึงมี **shadow layer** (`shadow.py`): agent แต่ละตัวมี trajectory ส่วนตัวที่ขับด้วย rate ชุดเดียวกับ compartment

```text
P(S -> E) ต่อวัน = 1 - exp(-lambda_g)      lambda_g มาจาก SEIR
P(E -> I) ต่อวัน = 1 - exp(-sigma)
P(I -> R) ต่อวัน = 1 - exp(-gamma)
```

ในเชิงค่าคาดหวัง shadow population จะสร้าง dynamics เดียวกับ compartment จึง **ไม่มีการนับซ้ำ**: shadow ให้ *การรับรู้* เท่านั้น (อาการของฉัน อาการของคนในบ้าน) ส่วน SEIR ยังเป็นผู้ตัดสินระบาดวิทยาแต่ผู้เดียว มี `resync` เป็นระยะเพื่อกัน sampling drift ของ agent จำนวนไม่กี่ร้อยตัว

---

## 6. Validation 3 ระดับ (guideline §13)

`validation.py` เขียนรายงานลง `validation.json` ทุกครั้งที่รัน

| Level | เทียบอะไร | ต้องใช้ข้อมูลเพิ่มไหม |
|---|---|---|
| 1 behavioural | mask use / social distancing ที่ simulate ได้ เทียบ 97% / 94% จาก survey ไทย | ไม่ต้อง (target ฝังใน `evidence.py`) |
| 2 aggregate | mobility index เทียบ mobility report จริง | ต้องมี `--observed-mobility file.csv` |
| 3 epidemiological | reported cases เทียบ epidemic curve ไทยจริง (MAE / RMSE / MAPE / r / peak error) | ต้องมี `--observed-cases file.csv` |

Level 1 เทียบเฉพาะ **ช่วงที่โรคระบาดเด่นชัด** (reported ≥ 2/100k) เพราะ survey เก็บช่วง first wave ไม่ใช่ทั้ง run ถ้าไม่มีวันไหนเข้าเงื่อนไข ระบบจะรายงานว่า "ไม่มี counterpart" แทนที่จะเปรียบเทียบมั่ว ๆ

Level 2/3 ถ้าไม่ให้ไฟล์ ระบบจะรายงาน `skipped` — **ไม่สร้างข้อมูลเทียบขึ้นมาเอง**

```bash
python3 run_thaigabm.py --arms BD --mock \
    --observed-cases data/thailand_cases.csv --cases-column cases
```

---

## 7. การควบคุมต้นทุน

ปัญหาใหญ่ของเปเปอร์เดิมคือ >90 ชม. ต่อ 1,000 agents โปรเจคนี้ลดด้วย 4 กลไก ที่ตรวจสอบได้จาก `cache_stats` ในไฟล์ `meta_*.json`

1. **Representative agents** — agent 1 ตัวถือ `weight = N_g / n_g` แทนคนจริงหลายแสนคน (aggregator ใช้ weight ไม่ใช่จำนวน agent)
2. **Situation-keyed cache** — key จาก (persona แบบหยาบ, อาการ, สถานการณ์ในบ้าน, bucket ของ case count, trend, policy) โหมด `sample` เก็บหลายคำตอบต่อ key แล้วสุ่มใช้เพื่อรักษา stochasticity ของ LLM ส่วน `first` ตัดความแปรปรวนทิ้งเพื่อ internal validity
3. **In-flight deduplication** — ใน `--cache first` agent ที่สถานการณ์ตรงกันในวันเดียวกันรอ call เดียว
4. **`--decision-interval k`** — ให้ agent ใช้แผนเดิม k วัน ลดต้นทุนเชิงเส้น

---

## 8. ไฟล์ผลลัพธ์

| ไฟล์ | เนื้อหา |
|---|---|
| `timeseries_<arm>.csv` | รายวัน: S/E/I/R, new infections, reported cases, Rt, phi ต่อ setting, mask, isolating, และแยกราย stratum |
| `decisions_<arm>.csv` | การตัดสินใจรายคนรายวัน พร้อม **เหตุผลภาษาธรรมชาติ** — ใช้ทำ qualitative analysis ของ reasoning ได้ |
| `meta_<arm>.json` | config เต็ม, R0, q, attack rate, peak, cache stats, จำนวน LLM error |
| `comparison.csv` | ตารางเทียบทุก arm |
| `validation.json` | รายงาน validation 3 ระดับ |
| `PARAMETER_PROVENANCE.md` | ที่มาของทุก parameter |
| `fig1..fig6 *.png` | รูป (สร้างด้วย `plot_thaigabm.py`) |

---

## 9. ข้อจำกัดที่ต้องระบุในรายงาน

1. **`--mock` ไม่ได้จำลอง LLM** — mode นี้ใช้ rule model แทน ดังนั้น arm D กับ arm B จะได้ผลเหมือนกันเป๊ะเมื่อรันแบบ mock ความต่าง B vs D จะเกิดขึ้นก็ต่อเมื่อรันด้วย LLM จริง ใช้ `--mock` เพื่อทดสอบ pipeline เท่านั้น ห้ามใช้เป็นผลการทดลอง
2. **สัดส่วน stratum และ household size** ยังเป็นตัวเลขประมาณระดับประเทศ ต้องแทนที่ด้วยตาราง NSO จริงก่อนอ้างเชิงปริมาณเกี่ยวกับประเทศไทย
3. **`salience_scale` ของ rule model** (ค่าเริ่มต้น 5.0 per 100k) เป็น free parameter ที่ยังไม่ calibrate — เป็นงาน Phase 5
4. **Level 1 validation ปัจจุบันยังไม่ผ่าน** สำหรับ rule model เพราะ mask use ขึ้นไปไม่ถึง 97% ในระบอบที่ epidemic ถูกกดไว้ตั้งแต่ต้น อันนี้เป็นผลลัพธ์จริงที่ต้องรายงาน ไม่ใช่บั๊ก และเป็นเหตุผลที่ต้องทำ calibration
5. **ค่าที่ติดป้าย `assumption`** (perceived_susceptibility, institutional_trust, social_preference, health_literacy, mask_efficacy, P_MULTIGENERATIONAL) ยังไม่มีหลักฐานไทยรองรับโดยตรง

---

## 10. Phase ถัดไป (guideline §15)

- [x] Phase 1 — Classical SEIR (arm A)
- [x] Phase 2 — Rule-based behavioural SEIR (arm B)
- [x] Phase 3 — LLM behavioural layer (arm C/D)
- [x] Phase 4 — Thai behavioural parameterization (`evidence.py`)
- [ ] Phase 5 — Calibration + validation ด้วยข้อมูลไทยจริง (โครงพร้อมแล้วใน `validation.py` เหลือหาไฟล์ข้อมูล)
- [ ] Phase 6 — Policy experiments (ปรับ `PolicyRegime` / `POLICY_HARD_CAP` ใน `information.py`)

## References

[1] Williams et al. (2023) *Epidemic Modeling with Generative Agents*. arXiv:2307.04986
[2] Ferenz et al. (2026) *The Epi-LLM Framework*. arXiv:2606.02867
[3] Pan-ngum et al. (2021) *Perspectives on public health interventions in the management of the COVID-19 pandemic in Thailand*. BMC Public Health
[4] *"Like a wake-up call for humankind"* (2022/2023) — Thai first-lockdown qualitative study
[5] Larooij & Törnberg (2026) *Validation is the central challenge for generative social simulation*. AI Review 59:15
[6] Aoki & Ghaffarzadegan (2026) *AI Agents as Policymakers in Simulated Epidemics*. arXiv:2601.04245
