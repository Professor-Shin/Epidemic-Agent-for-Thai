"""Thai-context GABM-SEIR shared code. Notebooks add the project root to sys.path and import from here."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "base.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mix = cfg["contacts"]["baseline_mix"]
    assert set(mix) == set(cfg["actions"]), "baseline_mix must list every action"
    assert abs(sum(mix.values()) - 1) < 1e-9, "baseline_mix must sum to 1"
    return cfg
