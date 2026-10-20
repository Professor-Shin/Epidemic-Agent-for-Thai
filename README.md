# Thai-context GABM-SEIR · Colab notebooks

Implementation of `../ThaiGABM_SEIR_project_guideline_updated.md`: SEIR baseline → open-weight LLM residents (4 actions) → model comparison → weekly LLM leader.

## Run on Google Colab

1. Upload the whole `abm/` folder to Google Drive as **`MyDrive/ThaiGABM/`**.
   You can skip `data/raw/` (137 MB); notebook 00 downloads it again.
2. Open a notebook from Drive in Colab. For 02–05, choose **Runtime → Change runtime type → T4 GPU**.
3. For `llama8b` (gated): accept the licence at huggingface.co/meta-llama/Llama-3.1-8B-Instruct, then add
   a Colab secret named `HF_TOKEN` (key icon in the left bar) and allow notebook access.
4. Run the notebooks in order. Each one mounts Drive, installs `requirements-colab.txt`, and loads `config/base.yaml`.

| Notebook | GPU | What it does | Main outputs |
|---|---|---|---|
| `00_data_prep` | no | Downloads Thai public data and builds contact levels, personas, provenance | `data/processed/*`, `config/personas.csv`, `data/PROVENANCE.json`, `data/parameter_sources.csv` |
| `01_seir_baseline` | no | Conditions A and B, conservation, β₀=0, and final-size checks | `outputs/runs/A_*`, `B_*`, figures |
| `02_llm_feasibility` | T4 | One model per run: VRAM, speed, schema validity, behaviour, consistency | `outputs/feasibility/feasibility.csv` |
| `03_resident_prototype` | T4 | Condition C (40 agents × 30 days × 3 seeds) vs A/B | `outputs/runs/C_*` |
| `04_model_comparison` | T4 | C for each model + sensitivity sweeps; analysis runs without a GPU | `outputs/figures/04_*` |
| `05_leader_agent` | T4 | Condition D (weekly leader) and the D vs C ablation, gated on the §12 criteria | `outputs/runs/D_*` |

**Switching models:** set `MODEL_KEY` → *Runtime → Restart session* → Run all. A T4 cannot hold two 7–8B models at once.
Available keys: `typhoon8b` (Thai-tuned, SCB10X, based on Llama 3.1 8B), `qwen7b`, `llama8b` (gated), `mistral7b`, `qwen1_5b` (fast debug).
`typhoon8b` is the best fit for Thai resident reasoning; on HPC, swap its `hf_id` for `scb10x/llama3.1-typhoon2-70b-instruct` and set `backend: vllm`.

**Colab disconnected?** Reopen the notebook and Run all. Each run continues from its last saved day, and finished runs are skipped.
The run id is a hash of the full config, persona set, model, seed and tag, so changing any of these starts a new run.

**Testing without a GPU:** set `BACKEND = "mock"`. A rule-based stand-in replaces the LLM, about 5% of its replies are malformed on purpose, and its runs are excluded from analysis by default.

## Model mechanics (see `gabm/`)

- `contacts.py`: c_out per action from Prem et al. 2021 Thai location contacts. β_eff = β₀·[w_home + (1−w_home)·q_out·q_policy]. Home contacts are always kept.
- `llm.py`: one prompt template for all models. JSON reply → allowed-action check → one retry → otherwise keep the previous action, flagged `fallback`. Raw output is always logged.
- `sim.py`: the daily loop for conditions A–D, with a checkpoint after every day.
- `seir.py`: SEIR with rates per day. Reported cases = the E→I flow.

## Data status

Every parameter is listed in `data/parameter_sources.csv` with status `data`, `literature` or `assumption`.
Occupation shares and WFH ability in `config/persona_marginals.csv` are **assumptions** until they are replaced with NSO Labour Force Survey tables. After editing them, rerun notebook 00.
Thai OxCGRT cases are a qualitative reference only. This is a scenario study, not a calibrated forecast.
