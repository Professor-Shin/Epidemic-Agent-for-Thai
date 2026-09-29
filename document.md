# Epidemic Modeling with Generative Agents — Python + Google ADK Implementation

> **หมายเหตุ:** เอกสารนี้อธิบายการ replicate เปเปอร์ต้นฉบับ (`main.py` + `src/*.py`)
> โปรเจคได้ต่อยอดเป็น **ThaiGABM-SEIR** แล้ว — ดู [ThaiGABM_SEIR_README.md](ThaiGABM_SEIR_README.md)
> โค้ดเดิมยังใช้งานได้ตามปกติ และถูกเก็บไว้เป็น reference implementation ของงานต้นฉบับ

An implementation of the Generative Agent-Based Model (GABM) proposed in

> Williams, R., Hosseinichimeh, N., Majumdar, A., & Ghaffarzadegan, N. (2023).
> **Epidemic Modeling with Generative Agents.** arXiv:2307.04986.

The original paper uses `Mesa` + OpenAI `gpt-3.5-turbo`. This project reproduces the
same paradigm using **Python** and **Google's Agent Development Kit (ADK)**, driving
each generative agent's reasoning with a Gemini model (default: `gemini-2.0-flash`).

---

## 1. Overview

### 1.1 What the paper proposes

Classic Agent-Based Models (ABMs) of epidemics hard-code behavioral rules
("if symptomatic, stay home with probability p"). The paper argues that this
makes it impossible to represent the real, endogenous coupling between an
epidemic's state and human decision-making. Their contribution is to replace
the hand-written rule with an **LLM prompt**:

- Each agent has a persona (name, age, Big Five personality traits, job).
- Each simulated day, the agent is fed a prompt containing its persona plus
  whatever "relevant memories" the experimental condition allows (its own
  symptoms; the town-level news bulletin).
- The LLM answers: *do I go to work today?* + a natural-language reason.
- People who go out get sampled into contact pairs; a susceptible person
  meeting an infectious one is transmitted with probability `infectivity`.
- The disease then progresses in each infected agent by one day, and the
  loop repeats.

Three experimental conditions ablate what information the agents get:

| Condition | Persona | Own symptoms | News (yesterday's new-case %) |
| --------- | :-----: | :----------: | :---------------------------: |
| `base`    |    ✓    |              |                               |
| `self`    |    ✓    |      ✓       |                               |
| `full`    |    ✓    |      ✓       |               ✓               |

The paper shows that `base` reproduces a textbook SIR outbreak, `self`
flattens the curve because sick people self-quarantine, and `full` further
flattens it because *risk-aware* agents self-isolate as case counts rise —
even producing multiple waves and an endemic tail without any behavioral
rule being hard-coded.

### 1.2 How this repo implements it

Two layers run per simulated day:

1. **GAI (Google ADK reasoning layer).** A single ADK `LlmAgent`, wrapped by
   `DecisionAgent`, is invoked once per person in parallel via
   `asyncio.gather`. Each call opens a fresh ADK session so the LLM is
   stateless — all "memory" flows through the prompt, exactly as in the
   paper. The model returns compact JSON `{"decision": "go"|"stay", "reason": "..."}`.
2. **ABM (spread layer).** `World.compute_new_infections` samples, for each
   person who chose to go out, `contact_rate` unique other outgoing people,
   deduplicates the edges, and rolls a Bernoulli(`infectivity`) transmission
   for each susceptible-meets-infectious pair.

After the ABM step, we advance every infected agent's disease clock and
recover anyone past day 6.

---

## 2. Repository layout

```
epidemic_agent/
├── main.py                 # CLI entry point
├── plot_results.py         # Optional: plots I(t) and new-infections curves
├── requirements.txt
├── .env.example
├── document.md             # (this file)
├── 2307.04986v1.pdf        # The original paper
└── src/
    ├── config.py           # SimulationConfig dataclass
    ├── person.py           # Person, HealthState, Symptom
    ├── traits.py           # Name & Big Five sampling
    ├── disease.py          # Disease progression (infect / advance_day)
    ├── world.py            # ABM contact-network + transmission
    ├── agent.py            # Google ADK LlmAgent wrapper + MockDecisionAgent
    └── runner.py           # Simulation orchestrator (main daily loop)
```

---

## 3. How the model works (details)

### 3.1 Persona generation (`src/traits.py`)

At initialisation, each of `N` residents is assigned:

- **Name**: `first_name last_name` sampled without replacement from bundled
  lists (the paper uses `names-dataset 3.1.0`; we bundle a smaller list to
  keep the project dependency-light).
- **Age**: uniform integer in `[18, 64]` — matches the paper.
- **Big Five traits**: for each of `openness`, `conscientiousness`,
  `extraversion`, `agreeableness`, `neuroticism` we flip a fair coin between
  a "positive" and "negative" natural-language description. These strings
  are fed directly into the LLM prompt so the model reasons about a
  concrete personality rather than a five-tuple of numbers.

### 3.2 Disease dynamics (`src/disease.py`, `src/person.py`)

Each infected person carries an `infection_day` counter:

| `infection_day` |     Symptom shown to LLM     | Infectious? |
| :-------------: | :--------------------------: | :---------: |
|        0        |         feels normal         |     no      |
|      1, 2       |         feels normal         |     yes     |
|        3        |     has a light cough        |     yes     |
|      4, 5       |     has a fever and cough    |     yes     |
|        6        |     has a light cough        |     yes     |
|       7+        |     recovered (immune)       |     —       |

`Person.symptom` and `Person.is_infectious` derive from this counter, so the
prompt-building code never has to branch on state manually.

### 3.3 The daily loop (`src/runner.py`, `Simulation.run`)

For each simulation day (following the paper's ordering):

1. **Read yesterday's news.** `case_pct = 100 * prev_day4 / N`, where
   `prev_day4` is the number of people who reached `infection_day == 4` at
   the end of yesterday. This mirrors the paper's rule: *"New daily active
   cases are counted by the number of agents who are in day 4 of being
   infected."*
2. **GAI decision phase.** Build a prompt for every resident and dispatch
   them all in parallel with `asyncio.gather`. The set of people who choose
   `"go"` becomes `active_ids`.
3. **ABM contact phase.** `World.compute_new_infections(active_ids)`:
   - For each active person, uniformly sample up to `contact_rate` distinct
     other active people to interact with.
   - Store contacts as an undirected edge set (so a mutual pick counts
     once).
   - For every edge with exactly one infectious endpoint and one
     susceptible endpoint, roll `Bernoulli(infectivity)`; on success, mark
     the susceptible for infection.
4. **Apply infections.** New infections start at `infection_day = 0`
   (asymptomatic and *not yet* infectious that same day — avoiding
   physically-implausible zero-latency super-spread).
5. **Advance the clock.** Every infected person's `infection_day` increments
   by 1; anyone whose counter passes 6 recovers and becomes immune.
6. **Record & log.** A per-day row is appended to `self.history` and
   printed. If active infections ever hit 0 after day 1, we stop early.

### 3.4 Google ADK integration (`src/agent.py`)

```python
class DecisionAgent:
    def __init__(self, model="gemini-2.0-flash", app_name="epidemic"):
        self.agent = LlmAgent(
            name="dewberry_resident",
            model=model,
            instruction=SYSTEM_INSTRUCTION,
        )
        self.runner = InMemoryRunner(agent=self.agent, app_name=app_name)
```

- **Single agent, many sessions.** We instantiate one `LlmAgent` and one
  `InMemoryRunner`, then create a fresh session per (person, day) with
  `session_service.create_session(...)`. Because the session has no prior
  turns, the LLM sees only that day's prompt — matching the paper's
  stateless prompt-with-memories approach.
- **System instruction.** Tells the model: role-play as a resident, respond
  in single-line JSON only.
- **Prompt template.** `build_prompt(person, condition, case_pct, ...)`
  assembles the exact strings from the paper's Coding section:
  - Base run adds: *"[Name] goes to work to earn money to support [Name]'s
    self."*
  - Self-health feedback appends: *"Today, physically, [Name] feels
    normal / has a light cough / has a fever and cough."*
  - Full feedback appends the news paragraph including
    *"[X]% of Dewberry Hollow's population caught new infections of the
    Catasat virus yesterday."*
- **Response parsing.** `parse_decision(raw)` extracts the JSON object with
  a permissive regex and falls back to keyword-scanning (`"stay"`,
  `"isolate"`, `"quarantine"`) if the model deviates from the format.

### 3.5 Offline / mock mode

`--mock` swaps `DecisionAgent` for `MockDecisionAgent`, which applies a
hand-written heuristic (fever → stay; cough → 50% stay; high news + anxious
personality → self-isolate). This lets you exercise the whole pipeline —
including the three-condition ablation — without any API calls, and it is
what the smoke-test in section 5 uses.

---

## 4. Setup

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Google AI Studio API key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=... (get one at https://aistudio.google.com/apikey)
```

`GOOGLE_GENAI_USE_VERTEXAI=FALSE` tells ADK to hit the AI Studio endpoint
rather than Vertex AI. If you have a Vertex AI project, set that to `TRUE`
and provide the usual `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` vars
instead of `GOOGLE_API_KEY`.

---

## 5. Running

```bash
# Quick offline smoke test (no API calls, uses heuristic agent)
python main.py --mock --population 20 --days 30

# Single condition (LLM required)
python main.py --condition full --population 30 --days 40

# All three conditions back-to-back (base → self → full)
python main.py --all --population 20 --days 30

# Plot the summary CSVs for comparison
python plot_results.py results/summary_base.csv results/summary_self.csv results/summary_full.csv
```

### 5.1 CLI flags

| Flag                  | Default              | Meaning                                                              |
| --------------------- | -------------------- | -------------------------------------------------------------------- |
| `--condition`         | `full`               | Experimental condition: `base` / `self` / `full`.                    |
| `--all`               | off                  | Run all three conditions sequentially (overrides `--condition`).     |
| `--population, -n`    | `20`                 | Number of residents.                                                 |
| `--days, -d`          | `30`                 | Number of simulation days.                                           |
| `--contact-rate`      | `5`                  | Unique contacts per person per day (paper default).                  |
| `--infectivity`       | `0.1`                | Per-contact transmission probability (paper default; ~R0=3).         |
| `--initial-infected`  | `1`                  | Seeded infections at day 0.                                          |
| `--model`             | `gemini-2.0-flash`   | Google GenAI model id.                                               |
| `--output`            | `results`            | Output directory for CSV and JSON.                                   |
| `--seed`              | `42`                 | RNG seed for reproducible personas, contacts, and transmissions.     |
| `--mock`              | off                  | Skip LLM calls and use `MockDecisionAgent`.                          |

### 5.2 Output artefacts

For each run, two files are written into `--output`:

- `summary_<condition>.csv` — one row per day with SIR counts, active
  workers, new infections, and yesterday's news percentage. Consumed by
  `plot_results.py`.
- `detail_<condition>.json` — the same day-level counts *plus* every
  individual agent's decision, first-person reason, and health state each
  day. This is what you'd inspect to reproduce the paper's qualitative
  analysis of *why* agents chose to stay home.

### 5.3 A note on runtime cost

The paper's headline runs use `n=1000` over ~80 hours per configuration.
This implementation parallelises calls with `asyncio.gather`, but you are
still bound by the Gemini rate limit for your key (15 RPM on the free tier
for `gemini-2.0-flash`). A single `--population 30 --days 40` run =
1,200 LLM calls; budget accordingly, or set a smaller `n` for
demonstrations. `--mock` is instantaneous.

---

## 6. Reading the output

The mock-mode smoke test shows the expected qualitative pattern from the
paper (n=25, seed=7):

- **base**: infection rips through the whole town — 25/25 recovered by day 20.
- **self**: sick people stay home → outbreak dies at just 1 case.
- **full**: same, plus a visible drop in "active workers" the day news
  breaks (25 → 16 in one step in our smoke test).

When you switch to a real LLM (`--condition full`), you should observe the
paper's central finding: agents who are anxious (high neuroticism), older,
or reading alarming news pre-emptively self-isolate, producing a flattened
curve and — on some seeds — the multi-wave / endemic pattern illustrated in
Figure 2 of the paper.

---

## 7. Mapping to the paper

| Paper concept                                 | This repo                                          |
| --------------------------------------------- | -------------------------------------------------- |
| Generative agent with persona + prompt        | `DecisionAgent` + `build_prompt` (`src/agent.py`)  |
| ChatGPT `gpt-3.5-turbo-0301` API              | Google ADK `LlmAgent` on `gemini-2.0-flash`        |
| Mesa ABM                                      | `World.compute_new_infections` (`src/world.py`)    |
| Big Five personality (50/50 pos/neg)          | `sample_traits` (`src/traits.py`)                  |
| Age uniform 18–64                             | `sample_age` (`src/traits.py`)                     |
| Disease timeline (6 days, symptoms)           | `Person.symptom` (`src/person.py`)                 |
| "New daily active cases = day-4 infections"   | `prev_day4` bookkeeping in `Simulation.run`        |
| Base / self-health / full-feedback conditions | `--condition base|self|full` and `build_prompt`    |
| Contact rate = 5, infectivity = 0.1 (R0 ≈ 3)  | CLI defaults in `main.py`                          |

---

## 8. Extending

Sensible next steps aligned with the paper's follow-up work:

- **Age-stratified risk.** The paper notes the LLM implicitly weighs age.
  Adding `--sensitivity age` that varies age distributions and comparing
  self-isolation rates would reproduce their sensitivity analysis.
- **Vaccination / masking prompts.** Add optional prompt fragments (e.g.
  *"[Name] has been offered a vaccine"*) as new condition flags.
- **Memory across days.** Instead of a fresh session per call, keep one
  session per person so the LLM can reason about its own history (*"I
  stayed home yesterday because…"*). ADK's `session_service` already
  supports this — just stop calling `create_session` with a day-scoped id.
- **Batching.** For large `n`, batch prompts through the Gemini `generateContent`
  endpoint directly (bypassing per-call session overhead) rather than through
  the ADK runner.
