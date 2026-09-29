"""Experiment drivers: the A-E design plus the RQ4 robustness sweeps."""

from __future__ import annotations

import json
import os
from dataclasses import replace

from src.thaigabm.config import ARMS, RunConfig, arm_D_thai_gabm, arm_E_ablation
from src.thaigabm.evidence import ABLATABLE
from src.thaigabm.runner import RunResult, ThaiGABMSimulation
from src.thaigabm.validation import (
    compare_arms, format_validation, validate, write_comparison,
)


def apply_mock(cfg: RunConfig, mock: bool) -> RunConfig:
    """Swap the LLM for the offline rule model -- and ONLY for LLM arms.

    Arms A and B never call an LLM, so ``--mock`` must leave them exactly as
    defined; otherwise the classical-SEIR baseline silently acquires behaviour
    and stops being a baseline.
    """
    if mock and cfg.behaviour == "llm":
        cfg.behaviour = "mock"
    return cfg


async def run_config(cfg: RunConfig, out_dir: str, verbose: bool = True) -> RunResult:
    sim = ThaiGABMSimulation(cfg)
    result = await sim.run(verbose=verbose)
    sim.save(out_dir)
    return result


async def run_arms(names: list[str], out_dir: str, overrides: dict,
                   verbose: bool = True, mock: bool = False) -> dict[str, RunResult]:
    """Run any subset of arms A-D back to back."""
    results: dict[str, RunResult] = {}
    for key in names:
        factory = ARMS[key.upper()]
        cfg = apply_mock(factory(**overrides), mock)
        results[cfg.name] = await run_config(cfg, out_dir, verbose)
        if verbose:
            print()
    return results


async def run_ablation(out_dir: str, overrides: dict,
                       constructs: tuple[str, ...] = ABLATABLE,
                       verbose: bool = True, mock: bool = False) -> dict[str, RunResult]:
    """Experiment E: full Thai model, then one construct removed at a time.

    Removing a construct means both neutralising the parameter (every agent
    gets 0.5) and deleting the line from the prompt, so the agent genuinely
    cannot reason about it.
    """
    results: dict[str, RunResult] = {}
    full = apply_mock(arm_D_thai_gabm(**overrides), mock)
    full.name = "E_full_model"
    results[full.name] = await run_config(full, out_dir, verbose)
    for construct in constructs:
        cfg = apply_mock(arm_E_ablation(construct, **overrides), mock)
        results[cfg.name] = await run_config(cfg, out_dir, verbose)
        if verbose:
            print()
    return results


async def run_seed_sweep(base: RunConfig, seeds: list[int], out_dir: str,
                         verbose: bool = False) -> dict[str, RunResult]:
    """Stochastic replication: same arm, different seeds."""
    results: dict[str, RunResult] = {}
    for s in seeds:
        cfg = replace(base, seed=s, name=f"{base.name}_seed{s}")
        results[cfg.name] = await run_config(cfg, out_dir, verbose)
    return results


async def run_model_sweep(models: list[str], out_dir: str, overrides: dict,
                          verbose: bool = True) -> dict[str, RunResult]:
    """RQ4: are behavioural decisions robust across LLM architectures?"""
    results: dict[str, RunResult] = {}
    for m in models:
        cfg = arm_D_thai_gabm(model=m, **overrides)
        cfg.name = f"RQ4_{m.replace('/', '_').replace('.', '_')}"
        results[cfg.name] = await run_config(cfg, out_dir, verbose)
        if verbose:
            print()
    return results


async def run_language_sweep(out_dir: str, overrides: dict, verbose: bool = True,
                             mock: bool = False) -> dict[str, RunResult]:
    """RQ4: same content, English vs Thai prompt -- isolates a language effect."""
    results: dict[str, RunResult] = {}
    for lang in ("en", "th"):
        cfg = apply_mock(arm_D_thai_gabm(**overrides), mock)
        cfg.arm = replace(cfg.arm, language=lang)
        cfg.name = f"RQ4_lang_{lang}"
        results[cfg.name] = await run_config(cfg, out_dir, verbose)
        if verbose:
            print()
    return results


def finalise(results: dict[str, RunResult], out_dir: str,
             observed_cases: str | None = None,
             observed_mobility: str | None = None,
             verbose: bool = True) -> dict:
    """Write the cross-arm comparison table and per-arm validation reports."""
    os.makedirs(out_dir, exist_ok=True)
    rows = compare_arms(results)
    comparison_path = os.path.join(out_dir, "comparison.csv")
    write_comparison(rows, comparison_path)

    reports = {name: validate(res, observed_cases, observed_mobility)
               for name, res in results.items()}
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(reports, f, indent=2)

    if verbose and rows:
        print("=" * 92)
        print("Arm comparison")
        headers = ["arm", "attack_rate", "peak_infectious", "peak_day",
                   "mean_mobility", "mean_mask", "min_Rt", "llm_calls"]
        print("  ".join(h.rjust(16) for h in headers))
        for r in rows:
            print("  ".join(str(r.get(h, "")).rjust(16) for h in headers))
        print()
        for name, rep in reports.items():
            print(f"--- {name} ---")
            print(format_validation(rep))
            print()
        print(f"Wrote {comparison_path}")
    return {"comparison": rows, "validation": reports}
