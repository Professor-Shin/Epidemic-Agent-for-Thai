"""Metrics and plots shared by notebooks 01, 03, 04 and 05 (guideline section 8.3)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import ROOT

ACTION_COLORS = {"stay_home": "#4C78A8", "work_onsite": "#F58518", "essential_errand": "#54A24B", "visit_family": "#B279A2"}
CONDITION_COLORS = {"A": "#7f7f7f", "B": "#4C78A8", "C": "#F58518", "D": "#54A24B"}


def epi_metrics(daily, N):
    d = daily.sort_values("day")
    return {
        "attack_rate_window": d["cum_inf"].iloc[-1] / N,
        "peak_I": d["I"].max(),
        "peak_day": int(d.loc[d["I"].idxmax(), "day"]),
        "peak_reported_per100k": d["reported_per100k"].max(),
        "mean_beta_eff": d["beta_eff"].mean(),
        "mean_q_out": d["q_out"].mean(),
    }


def decision_metrics(run_dir):
    p = Path(run_dir) / "decisions.jsonl"
    if not p.exists():
        return {}
    dec = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])
    out = {"n_decisions": len(dec),
           "valid_first_try": 1 - dec["retried"].mean(),
           "fallback_rate": dec["fallback"].mean()}
    for a, s in dec["choice"].value_counts(normalize=True).items():
        out[f"unweighted_{a}"] = s
    return out


DEDUPE_KEYS = ["condition", "model_key", "seed", "tag", "days", "backend"]


def load_runs(root=ROOT, pattern="*", include_mock=False, dedupe=True):
    """One row per completed run with metadata, epidemic metrics, decision metrics and runtime.
    Runs made with the mock backend are pipeline tests and are left out unless include_mock=True.
    dedupe=True drops runs that repeat the same (condition, model_key, seed, tag, days, backend) —
    these appear when Colab is run twice or the config registry changes the run-id hash without
    changing the result. The most recently finished run is kept (by runtime.json mtime), so re-running
    a config always supersedes an older copy in the analysis; nothing on disk is overwritten."""
    rows = []
    for d in sorted((Path(root) / "outputs" / "runs").glob(pattern)):
        if not (d / "runtime.json").exists():
            continue
        conf = json.load(open(d / "config.json", encoding="utf-8"))
        if conf["backend"] == "mock" and not include_mock:
            continue
        rt = json.load(open(d / "runtime.json", encoding="utf-8"))
        daily = pd.read_csv(d / "daily.csv")
        N = conf["cfg"]["population"]["N"]
        rows.append({"run_id": d.name, "condition": conf["condition"], "model_key": conf["model_key"],
                     "seed": conf["seed"], "tag": conf["tag"], "backend": conf["backend"],
                     "days": rt.get("days", len(daily)), "finished_at": (d / "runtime.json").stat().st_mtime,
                     **epi_metrics(daily, N), **decision_metrics(d),
                     "runtime_s": rt["total_seconds"], "gpu": rt["env"].get("gpu")})
    df = pd.DataFrame(rows)
    if dedupe and not df.empty:
        df = (df.sort_values("finished_at").drop_duplicates(DEDUPE_KEYS, keep="last")
                .drop(columns="finished_at").reset_index(drop=True))
    elif not df.empty:
        df = df.drop(columns="finished_at")
    return df


def export_metrics(root=ROOT, path=None, **kw):
    """Write the deduped per-run summary table to outputs/metrics.csv so it can be attached to a report."""
    runs = load_runs(root, **kw)
    path = Path(path) if path else (Path(root) / "outputs" / "metrics.csv")
    runs.to_csv(path, index=False)
    return runs, path


def load_daily(run_ids, root=ROOT):
    frames = []
    for rid in run_ids:
        d = Path(root) / "outputs" / "runs" / rid
        conf = json.load(open(d / "config.json", encoding="utf-8"))
        df = pd.read_csv(d / "daily.csv")
        df["run_id"], df["condition"], df["model_key"], df["seed"], df["tag"] = (
            rid, conf["condition"], conf["model_key"], conf["seed"], conf["tag"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_epidemic(daily, group="condition", ax=None, y="I", colors=CONDITION_COLORS):
    ax = ax or plt.gca()
    for key, g in daily.groupby(group):
        for i, (_, run) in enumerate(g.groupby("run_id")):
            ax.plot(run["day"], run[y], color=colors.get(key), alpha=0.8, lw=1.5, label=str(key) if i == 0 else None)
    ax.set_xlabel("day")
    ax.set_ylabel(y)
    ax.legend(title=group, frameon=False)
    return ax


def plot_action_shares(daily, ax=None, title=None):
    """Mean weighted action shares across runs (stacked area)."""
    ax = ax or plt.gca()
    cols = [c for c in daily.columns if c.startswith("share_")]
    m = daily.groupby("day")[cols].mean()
    ax.stackplot(m.index, m.T.values, labels=[c[6:] for c in cols],
                 colors=[ACTION_COLORS.get(c[6:]) for c in cols], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("day")
    ax.set_ylabel("population-weighted share")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    if title:
        ax.set_title(title)
    return ax
