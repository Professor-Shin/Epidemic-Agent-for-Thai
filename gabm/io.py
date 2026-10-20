"""Run folders, config snapshots, JSONL logs, and per-day checkpoints so runs survive Colab disconnects.

outputs/runs/<run_id>/
    config.json      full config + run settings (reproducibility)
    daily.csv        one row per simulated day
    decisions.jsonl  one record per agent per day (raw output, validated choice, flags)
    leader.jsonl     leader decisions (condition D)
    state.json       checkpoint written after each completed day
    runtime.json     hardware, versions, timing (written at the end)
"""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import pandas as pd


def config_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def make_run_id(condition, model_key, seed, settings):
    return f"{condition}_{model_key or 'none'}_s{seed}_{config_hash(settings)}"


class RunStore:
    def __init__(self, root, run_id):
        self.dir = Path(root) / "outputs" / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def path(self, name):
        return self.dir / name

    def save_json(self, name, obj):
        tmp = self.path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, self.path(name))

    def load_json(self, name):
        p = self.path(name)
        return json.load(open(p, encoding="utf-8")) if p.exists() else None

    def append_jsonl(self, name, records):
        with open(self.path(name), "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def append_daily(self, row):
        p = self.path("daily.csv")
        pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)

    def truncate_after(self, day):
        """Drop log rows for days >= day (they were written after the last checkpoint)."""
        p = self.path("daily.csv")
        if p.exists():
            df = pd.read_csv(p)
            df[df["day"] < day].to_csv(p, index=False)
        for name in ("decisions.jsonl", "leader.jsonl"):
            p = self.path(name)
            if p.exists():
                keep = [l for l in open(p, encoding="utf-8") if json.loads(l)["day"] < day]
                open(p, "w", encoding="utf-8").writelines(keep)

    def reset(self):
        for f in self.dir.iterdir():
            f.unlink()

    def is_complete(self):
        return self.path("runtime.json").exists()


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "torch", "transformers", "bitsandbytes", "vllm"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        info["gpu"] = None
    return info


def gpu_memory_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0])
    except Exception:
        return None
