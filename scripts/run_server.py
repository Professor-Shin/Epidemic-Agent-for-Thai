"""Headless runner for a GPU server (SLURM). Makes the same run_simulation calls as notebooks 04 and 05 (and runs
notebook 02 through papermill), so the run ids match the notebooks: the analysis cells pick the results up
unchanged, and runs already finished on Colab are skipped here.

  python scripts/run_server.py --model typhoon8b --stage check           # env, GPU, model in the local cache
  python scripts/run_server.py --model typhoon8b --stage smoke           # 2-day C run, tag "smoke" (not analysed)
  python scripts/run_server.py --model typhoon8b --stage feasibility     # notebook 02 via papermill
  python scripts/run_server.py --model typhoon8b --stage C --seeds 1     # condition C, one seed
  python scripts/run_server.py --model typhoon8b --stage D               # condition D, all seeds (after C)
  python scripts/run_server.py --model typhoon8b --stage order           # action-order check (notebook 04)
  python scripts/run_server.py --model typhoon8b --stage sens            # LLM sensitivity sweeps (notebook 04)
  python scripts/run_server.py --model typhoon8b --stage baseline         # C under the alternative baseline_mix

Normally called by scripts/server/run_model.sbatch; see scripts/server/README.md.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from gabm import load_config  # noqa: E402

STAGES = ["check", "smoke", "feasibility", "C", "D", "order", "sens", "baseline"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model key from config/base.yaml (llm.models)")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--seeds", type=int, nargs="*", help="default: sim.seeds from base.yaml")
    ap.add_argument("--days", type=int, default=180, help="must equal DAYS in notebooks 04/05 (default 180)")
    ap.add_argument("--force", action="store_true",
                    help="feasibility: rerun even if the model is already in feasibility.csv; D: skip the C gate")
    ap.add_argument("--kernel", default=os.environ.get("PAPERMILL_KERNEL", "python3"), help="Jupyter kernel for papermill")
    ap.add_argument("--backend", choices=["transformers", "vllm", "mock"],
                    help="default: llm.backend from base.yaml; 'mock' tests the pipeline on CPU (runs excluded from analysis)")
    return ap.parse_args()


def check(cfg, model):
    """Fail early (non-zero exit) if the GPU, the packages or the cached model weights are not usable."""
    import torch
    import transformers
    import bitsandbytes
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | bitsandbytes {bitsandbytes.__version__}")
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: check the GPU request (--gres) and the torch build (TORCH_INDEX)")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.0f} GiB")
    from huggingface_hub import snapshot_download
    hf_id = cfg["llm"]["models"][model]["hf_id"]
    try:
        # Check only what transformers loads. download_models.py deliberately skips LICENSE, README and the
        # native-format weight duplicates (original/, consolidated*), so asking for a complete snapshot here
        # would report a correct download as incomplete.
        path = Path(snapshot_download(hf_id, local_files_only=True, allow_patterns=["config.json"]))
        weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
        tokenizer = sorted(path.glob("tokenizer*")) or sorted(path.glob("*.model"))
        if not weights or not tokenizer:
            raise FileNotFoundError(
                f"the cached snapshot at {path} has no {'weight' if not weights else 'tokenizer'} files")
        print(f"{model}: cached at {path} ({len(weights)} weight files)")
    except Exception as e:
        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        msg = f"{model} ({hf_id}) is not in the local cache {os.environ.get('HF_HOME', '~/.cache/huggingface')}"
        if offline:
            sys.exit(f"{msg} and HF_HUB_OFFLINE=1: run scripts/server/download.sh on the login node first ({e})")
        print(f"warning: {msg}; it will be downloaded on first load")


def feasibility(args):
    """Notebook 02 via papermill. It appends to feasibility.csv, so array tasks take turns (file lock)."""
    import fcntl
    import papermill as pm
    feas = ROOT / "outputs" / "feasibility"
    feas.mkdir(parents=True, exist_ok=True)
    f = feas / "feasibility.csv"
    if not args.force and f.exists() and args.model in set(pd.read_csv(f).model_key):
        print(f"{args.model} already in {f}; skipping (use --force to rerun)")
        return
    out = ROOT / "outputs" / "executed" / f"02_feasibility_{args.model}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(feas / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pm.execute_notebook(str(ROOT / "notebooks" / "02_llm_feasibility.ipynb"), str(out),
                            parameters={"MODEL_KEY": args.model}, kernel_name=args.kernel,
                            cwd=str(ROOT / "notebooks"))
    print(f"feasibility done -> {out}")


def d_gate(cfg, args):
    """Notebook 05 criteria: C finished on every seed with fallback rate < 5%."""
    from gabm.analysis import load_runs
    runs = load_runs(ROOT, pattern=f"C_{args.model}_*")
    C = runs[(runs.tag == "") & (runs.days == args.days)] if len(runs) else runs
    seeds = set(cfg["sim"]["seeds"])
    missing = seeds - set(C.seed) if len(C) else seeds
    if missing:
        return f"condition C missing for seeds {sorted(missing)}"
    if C.fallback_rate.max() >= 0.05:
        return f"fallback rate {C.fallback_rate.max():.1%} >= 5% in condition C"
    return None


def main():
    args = parse_args()
    cfg = load_config(ROOT / "config" / "base.yaml")
    if args.model not in cfg["llm"]["models"]:
        sys.exit(f"unknown model {args.model!r}; choose from {list(cfg['llm']['models'])}")
    cfg["sim"]["days"] = args.days        # same override as the DAYS cell in notebooks 04/05 -> same run ids
    seeds = args.seeds or cfg["sim"]["seeds"]
    sens_seed = cfg["sim"]["seeds"][0]    # SENS_SEED in notebook 04

    if args.stage == "check":
        return check(cfg, args.model)
    if args.stage == "feasibility":
        return feasibility(args)
    if args.stage == "D" and not args.force:
        problem = d_gate(cfg, args)
        if problem:
            sys.exit(f"D gate not met for {args.model}: {problem} (run stage C first, or pass --force)")

    from gabm.llm import load_backend
    from gabm.sensitivity import action_order_variants, baseline_mix_variants, sensitivity_variants
    from gabm.sim import contact_model_from_cfg, run_simulation

    personas = pd.read_csv(ROOT / cfg["personas"]["file"])
    cm = contact_model_from_cfg(cfg, ROOT)
    backend = load_backend(cfg["llm"], args.model, args.backend)   # default: base.yaml (transformers 4-bit), as on Colab
    common = dict(personas=personas, backend=backend, model_key=args.model, cm=cm, root=ROOT)

    if args.stage == "smoke":
        c2 = copy.deepcopy(cfg); c2["sim"]["days"] = 2
        run_simulation(c2, "C", seed=seeds[0], tag="smoke", resume=False, **common)
    elif args.stage in ("C", "D"):
        for s in seeds:
            run_simulation(cfg, args.stage, seed=s, **common)
    elif args.stage == "order":
        for tag, c2 in action_order_variants(cfg):
            run_simulation(c2, "C", seed=sens_seed, tag=tag, **common)
    elif args.stage == "sens":
        for tag, c2, cm2, p2 in sensitivity_variants(cfg, ROOT, personas, cm):
            run_simulation(c2, "C", seed=sens_seed, tag=tag, **{**common, "cm": cm2, "personas": p2})
    elif args.stage == "baseline":
        for tag, c2, cm2 in baseline_mix_variants(cfg, ROOT):
            run_simulation(c2, "C", seed=sens_seed, tag=tag, **{**common, "cm": cm2})
    backend.unload()


if __name__ == "__main__":
    main()
