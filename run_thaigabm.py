"""ThaiGABM-SEIR command line.

Offline smoke test of the whole design (no API key needed):

    python3 run_thaigabm.py --arms ABCD --mock --days 150 --agents 100

One real Thai-context GABM run:

    python3 run_thaigabm.py --arms D --agents 200 --days 180

The ablation study (Experiment E):

    python3 run_thaigabm.py --ablation --mock

Robustness sweeps for RQ4:

    python3 run_thaigabm.py --model-sweep gemini-2.0-flash,gemini-2.5-flash
    python3 run_thaigabm.py --language-sweep
    python3 run_thaigabm.py --arms D --seeds 1,2,3,4,5

Validating against real data:

    python3 run_thaigabm.py --arms BD --mock \\
        --observed-cases data/thailand_cases.csv --cases-column cases
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.thaigabm.evidence import DiseaseParams
from src.thaigabm.experiments import (
    finalise, run_ablation, run_arms, run_language_sweep, run_model_sweep,
    run_seed_sweep,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ThaiGABM-SEIR: LLM behavioural layer coupled to a "
                    "mechanistic stratified SEIR model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arms", default="D",
                   help="Any subset of ABCD, e.g. 'ABCD' (default: D).")
    p.add_argument("--ablation", action="store_true",
                   help="Run Experiment E: drop one behavioural construct at a time.")
    p.add_argument("--model-sweep", default=None,
                   help="Comma-separated model ids for the RQ4 architecture sweep.")
    p.add_argument("--language-sweep", action="store_true",
                   help="Run the Thai arm with English and Thai prompts.")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds to replicate the chosen arm over.")

    p.add_argument("--agents", "-n", type=int, default=200)
    p.add_argument("--days", "-d", type=int, default=180)
    p.add_argument("--total-population", type=int, default=70_000_000)
    p.add_argument("--initial-infected", type=int, default=500)
    p.add_argument("--r0", type=float, default=2.5)
    p.add_argument("--latent-days", type=float, default=3.0)
    p.add_argument("--infectious-days", type=float, default=6.0)
    p.add_argument("--ascertainment", type=float, default=0.35)
    p.add_argument("--no-policy", action="store_true",
                   help="Disable the government reaction function.")

    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--mock", action="store_true",
                   help="No API calls: LLM arms fall back to the rule model "
                        "while still rendering every prompt.")
    p.add_argument("--cache", choices=["sample", "first", "off"], default="sample")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--decision-interval", type=int, default=1,
                   help="Re-query agents every k days (k>1 cuts cost linearly).")

    p.add_argument("--observed-cases", default=None,
                   help="CSV of observed reported cases for Level 3 validation.")
    p.add_argument("--cases-column", default="cases")
    p.add_argument("--observed-mobility", default=None,
                   help="CSV of observed mobility for Level 2 validation.")
    p.add_argument("--mobility-column", default="mobility")

    p.add_argument("--output", "-o", default="results_thaigabm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dump-prompt", action="store_true",
                   help="Print one example prompt per arm and exit.")
    return p


def user_specified(args: argparse.Namespace, ap: argparse.ArgumentParser) -> set[str]:
    """Which options the user actually typed (vs. left at their default).

    Needed because some arms *define themselves* by a setting -- arm A is the
    no-policy, no-behaviour baseline -- and a CLI default must not silently
    overwrite that.
    """
    return {dest for dest, val in vars(args).items()
            if val != ap.get_default(dest)}


def overrides_from(args: argparse.Namespace, provided: set[str]) -> dict:
    disease = DiseaseParams(
        latent_days=args.latent_days,
        infectious_days=args.infectious_days,
        r0=args.r0,
        ascertainment=args.ascertainment,
    )
    ov = {
        "n_agents": args.agents,
        "days": args.days,
        "total_population": args.total_population,
        "initial_infected": args.initial_infected,
        "disease": disease,
        "policy_enabled": not args.no_policy,
        "model": args.model,
        "cache_mode": args.cache,
        "concurrency": args.concurrency,
        "decision_interval_days": args.decision_interval,
        "seed": args.seed,
        "print_every": args.print_every,
    }
    # Arm-owned settings: only override when the user asked for it explicitly.
    if "no_policy" not in provided:
        ov.pop("policy_enabled")
    return ov


async def dump_prompt(args: argparse.Namespace) -> None:
    """Render one prompt for eyeballing -- the cheapest possible sanity check."""
    import random

    from src.thaigabm.information import InformationState
    from src.thaigabm.persona import Population
    from src.thaigabm.prompts import ArmSpec, build_prompt, build_prompt_th

    for label, arm in (("GENERIC (arm C)", ArmSpec(thai_context=False,
                                                   include_household=False)),
                       ("THAI (arm D)", ArmSpec()),
                       ("THAI, Thai language", ArmSpec(language="th"))):
        pop = Population(30, args.total_population, random.Random(args.seed),
                         flavour="thai" if arm.thai_context else "generic")
        person = next(p for p in pop.people if p.stratum == "informal_worker")
        info = InformationState(day=42, reported_new_cases=8200,
                                reported_per_100k=11.7, trend="rising",
                                policy="restriction")
        household = {"household_size": person.household_size,
                     "someone_sick_at_home": True, "n_sick_at_home": 1,
                     "elderly_at_home": True, "children_at_home": False}
        builder = build_prompt_th if arm.language == "th" else build_prompt
        print("=" * 92)
        print(label)
        print("=" * 92)
        print("[system]\n" + arm.system_instruction() + "\n")
        print("[user]\n" + builder(person, arm, 42, "have a light cough and feel a bit tired",
                                   household, info, args.total_population))
        print()


async def main_async(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    if args.dump_prompt:
        await dump_prompt(args)
        return

    verbose = not args.quiet
    out = args.output
    overrides = overrides_from(args, user_specified(args, ap))
    results = {}

    if args.mock and verbose:
        print("MOCK MODE: no API calls. LLM arms are driven by the rule model, "
              "so arms C/D differ only through the information channels their "
              "prompts expose -- not through LLM reasoning.\n")

    if args.ablation:
        results = await run_ablation(out, overrides, verbose=verbose, mock=args.mock)
    elif args.model_sweep:
        models = [m.strip() for m in args.model_sweep.split(",") if m.strip()]
        results = await run_model_sweep(models, out, overrides, verbose=verbose)
    elif args.language_sweep:
        results = await run_language_sweep(out, overrides, verbose=verbose,
                                           mock=args.mock)
    elif args.seeds:
        from src.thaigabm.config import ARMS
        from src.thaigabm.experiments import apply_mock

        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        key = args.arms.upper()[0] if args.arms else "D"
        base = apply_mock(ARMS[key](**overrides), args.mock)
        results = await run_seed_sweep(base, seeds, out, verbose=verbose)
    else:
        arms = list(args.arms.upper())
        bad = [a for a in arms if a not in "ABCD"]
        if bad:
            print(f"Unknown arm(s): {bad}. Choose from A, B, C, D.", file=sys.stderr)
            sys.exit(2)
        # Arms A and B never call an LLM, so mock only affects C and D.
        results = await run_arms(arms, out, overrides, verbose=verbose,
                                 mock=args.mock)

    finalise(results, out,
             observed_cases=args.observed_cases,
             observed_mobility=args.observed_mobility,
             verbose=verbose)
    if verbose:
        print(f"All outputs in {os.path.abspath(out)}/")
        print(f"Plot them with:  python3 plot_thaigabm.py {out}")


def main() -> None:
    ap = parser()
    args = ap.parse_args()
    needs_llm = (not args.mock) and (
        args.model_sweep or args.language_sweep
        or any(a in args.arms.upper() for a in "CD"))
    if needs_llm and not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY is not set. Export it, put it in .env, "
              "or pass --mock to run the whole pipeline offline.\n", file=sys.stderr)
    asyncio.run(main_async(args, ap))


if __name__ == "__main__":
    main()
