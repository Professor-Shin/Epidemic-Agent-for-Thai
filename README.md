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
| `04_model_comparison` | T4 | C for each model + sensitivity sweeps + 2×2 table + action-order check; analysis runs without a GPU | `outputs/figures/04_*`, `outputs/metrics.csv` |
| `05_leader_agent` | T4 | Condition D (weekly leader) and the D vs C ablation, gated on the §12 criteria | `outputs/runs/D_*` |

**Switching models:** set `MODEL_KEY` → *Runtime → Restart session* → Run all. A T4 cannot hold two 7–8B models at once.
Available keys: `typhoon8b` (Thai-tuned, SCB10X, Llama 3.1 8B base), `qwen7b`, `llama8b` (gated), `openthaigpt7b` (Thai-tuned, Qwen 2.5 7B base), `seallm7b` (SEA-multilingual, Qwen 2.5 7B base), `mistral7b` (negative control), `qwen1_5b` (fast debug).
`typhoon8b` is the best fit for Thai resident reasoning; on HPC, swap its `hf_id` for `scb10x/llama3.1-typhoon2-70b-instruct` and set `backend: vllm`.

**Model panel (2×2 design).** The comparison is built as *family × localisation* so the effect of Thai-tuning can be separated from the model family:

| | base (no Thai tuning) | Thai-tuned | SEA-multilingual |
|---|---|---|---|
| **Llama-3.1-8B** | `llama8b` (gated) | `typhoon8b` | — |
| **Qwen2.5-7B** | `qwen7b` | `openthaigpt7b` | `seallm7b` |

Run the four core keys (`llama8b`, `typhoon8b`, `qwen7b`, `openthaigpt7b`) through notebooks 04 and 05 to fill the 2×2; `seallm7b` and `mistral7b` are optional extras. All keys fit a T4 at 4-bit; `openthaigpt7b` and `seallm7b` are ungated (no `HF_TOKEN` needed).

**Run horizon (`DAYS`).** `config/base.yaml` keeps `days: 30` as the pilot. Notebooks 04 and 05 default `DAYS = 180` (set near `MODEL_KEY`) so the comparison covers the full epidemic (past the peak); set it back to 30 to reproduce the pilot. 180 days is ≈6× the LLM cost per model (≈1 h/model/condition on a T4). Notebook 03 keeps the 30-day pilot by default. The analysis cells filter on `days == DAYS`, so 30- and 180-day runs can coexist in `outputs/runs/` without mixing.

**Extra robustness in notebook 04.** `RUN_SENSITIVITY_LLM = True` adds the LLM sensitivity sweeps including the `wfh_scale` axis (the one that changes the resident prompt). `RUN_ORDER_CHECK = True` re-runs condition C with the action list reordered, to confirm `visit_family ≈ 0%` is a real behaviour and not a prompt-ordering artefact. Every analysis call now de-duplicates repeated runs and `export_metrics()` writes `outputs/metrics.csv` for the report.

**Running the full panel on a SLURM GPU server** (Colab is too slow for 5 models × 180 days): see
[scripts/server/README.md](scripts/server/README.md), or the Thai step-by-step guide for CU-AIM, [CUAIM_GUIDE_TH.md](CUAIM_GUIDE_TH.md). Each model gets its own GPU job, all in parallel, and the run ids match
Colab's, so the notebooks analyse server results unchanged. The parameter cells of notebooks 02/04/05 are tagged for
papermill, and notebook 05 has a `RUN_LLM` flag (False = analysis only, no GPU).

**Colab disconnected?** Reopen the notebook and Run all. Each run continues from its last saved day, and finished runs are skipped.
The run id is a hash of the full config (including the model registry and `DAYS`), persona set, model, seed and tag, so adding a model to `config/base.yaml` or changing `DAYS` gives new run ids — previous runs are not overwritten, but they are no longer auto-skipped, so a fresh batch re-runs from scratch.

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
