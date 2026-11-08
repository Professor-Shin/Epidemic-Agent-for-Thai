# Running on CU-AIM (SLURM)

A step-by-step guide in Thai, from `git clone` to downloading the results, is in [CUAIM_GUIDE_TH.md](../../CUAIM_GUIDE_TH.md).

Runs the model panel (notebooks 02, 04, 05) headless on the CU-AIM cluster: one GPU job per model, then one
CPU job that executes the analysis cells. `scripts/run_server.py` makes the same `run_simulation` calls as the
notebooks, so **run ids are identical to Colab**: the notebooks read the server results unchanged, and runs
already finished on Colab are skipped on the server.

The software matches the Colab runs: python 3.13, torch 2.11.0, transformers 5.20.1, bitsandbytes 0.50.2,
4-bit nf4. These are the versions the Colab runs recorded in `runtime.json`.

`scripts/server/server.env` is preset from the *CU-AIM Server User Manual 2026*: partition `gpu`, QoS `gpu40g`,
one full A100-40GB (`gpu:7g.40gb:1`), and a Singularity container (manual chapter 6). The container is needed
because CU-AIM's system libraries are older than current torch wheels require. That is the same reason new
VS Code versions cannot connect to it.

## Steps

Before you start, follow manual chapters 1–2: connect to the Chula VPN, and get your SSH key authorized.
Then run everything below on the **login node**.

```bash
# 1. Clone into your project storage (manual 3.1), not /data/home, which has a small quota
cd /data/project/<your-group>/<your-folder>
git clone https://github.com/Professor-Shin/Epidemic-Agent-for-Thai.git gabm && cd gabm

# 2. Check the presets in scripts/server/server.env against your account
check_qos                     # QoS names and limits; QOS and CPU_QOS must be listed here, and
                              # TIME_GPU/TIME_CPU must be <= the MaxWall column of QOS/CPU_QOS
ldd --version | head -1       # glibc. If >= 2.28 you may use ENV_KIND="conda" instead of the container
sinfo -o "%20P %5a %10l %12G %N"   # partition names and their gres; PARTITION must be listed here

# Optional: the driver version decides TORCH_INDEX (cu128 needs "CUDA Version" >= 12.8). The smoke job in
# step 6 prints it too, so skip this if the interactive queue is busy; cancel with scancel <jobid>.
srun -p gpu --qos=interactive-gpu --gres=gpu:1g.5gb:1 -c2 --mem=8G --pty nvidia-smi

# 3. Optional: copy the Colab results from your PC so finished work is skipped
#    outputs/feasibility/feasibility.csv   -> the 5 Colab models skip feasibility; only new ones run it
#    outputs/runs/*                         -> any run with the same config is skipped

# 4. Software environment (once; pulls python:3.13-slim and installs into .venv-gabm)
bash -l scripts/server/setup_env.sh

# 5. Model weights into .hf_cache (~80 GB for the 5 default models)
#    llama8b is gated: accept its licence on huggingface.co, then store your token where huggingface_hub
#    finds it (.hf_cache is gitignored; never put the token in server.env, which is committed)
mkdir -p .hf_cache && echo "hf_xxxxxxxx" > .hf_cache/token && chmod 600 .hf_cache/token
bash -l scripts/server/download.sh

# 6. Smoke test: env, GPU, cached weights, and a 2-day run of the first model
bash scripts/server/submit_all.sh smoke
tail -f logs/gabm-smoke_<jobid>.out      # ends with "smoke test OK"

# 7. Full run
bash scripts/server/submit_all.sh
squeue -u $USER                          # manual 4.5; cancel with: scancel -f <jobid>
```

## What runs

| Job | Resources | Does |
|---|---|---|
| `run_model.sbatch` (array, 1 task per model) | `gpu40g`, 1× A100-40GB | `check` → `feasibility` (only for models missing from `feasibility.csv`) → C × 3 seeds → D × 3 seeds, plus `order`/`sens` if they are listed in `STAGES` |
| `analyze.sbatch` (starts after the array ends) | `cpu40h`, no GPU | notebook 04 (`RUN_LLM=False`) and notebook 05 for each model, via papermill |

- `MAX_PARALLEL=2`: at most 2 models hold a GPU at once, because CU-AIM is shared by the lab. Raise it if the queue is free.
- `SEEDS_IN_PARALLEL=1`: the 3 seeds of a condition run as 3 processes on one A100-40GB, about 30 GB in total.
  If you switch to a smaller MIG slice (`gpu:3g.20gb:1` with its QoS), set it to 0.
- D starts only if C finished on every seed with fallback < 5%. This is the same gate as notebook 05.

## Outputs

- `outputs/runs/<run_id>/`: the same layout as on Colab
- `outputs/executed/*.ipynb`: notebooks 02/04/05 with every table and figure rendered. Download them and open locally.
- `outputs/figures/`, `outputs/metrics.csv`
- `logs/`: one log per job (`gabm-model_<array>_<task>.out`, `gabm-analyze_<id>.out`)

Copy the results back with `scp -r`, `rsync`, or Cyberduck (manual 1.4).

## Runtime and timeouts

A 180-day run took about 1 h at T4 speed. The A100 is faster, and the 3 seeds run at the same time. The smoke
log prints seconds per simulated day; multiply by 180 × 2 (C + D) for a per-model estimate.
`TIME_GPU` cannot exceed the QoS MaxWall: `gpu40g` caps at 8 h, `gpu40g64` (64 CPUs, 128 GB) at 12 h, and
`gpu10gh` at 4 days but only on a 10 GB slice, which needs `SEEDS_IN_PARALLEL=0`.
Every run saves a checkpoint after each day. If a job hits `TIME_GPU`, run `submit_all.sh` again: finished runs
are skipped and unfinished ones resume. Then run `submit_all.sh analyze` if the analysis ran too early.

## Known limits

- In MIG mode, `nvidia-smi` may not report per-slice memory. The VRAM columns of `feasibility.csv` can then be
  empty, but speed, schema-validity and consistency numbers are unaffected. The `gpu` column records the
  hardware, so A100 rows can be told apart from Colab's T4 rows.
- `srun: Requested partition configuration not available now` means the partition exists but no node in it
  currently offers the requested MIG profile (busy, drained, or that profile is not configured there). The job
  just waits. Check what the partition really has with `sinfo -o "%20P %10l %12G %N"` and
  `scontrol show node <node> | grep -i gres`, then pick a profile it lists.
- If `singularity` is not found, set `MODULE_LOAD` in `server.env` (try `module avail` / `module spider singularity`).
- If the analysis job is rejected, check `CPU_PARTITION`/`CPU_QOS` against `sinfo` and `check_qos`.
