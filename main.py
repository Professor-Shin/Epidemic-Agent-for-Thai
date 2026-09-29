"""CLI entry point for the generative-agent epidemic simulation.

Examples
--------
Quick offline smoke test (no API calls, uses heuristic MockDecisionAgent):

    python main.py --mock --population 20 --days 30

One "full feedback" run with the LLM (requires GOOGLE_API_KEY in .env):

    python main.py --condition full --population 30 --days 40

Run all three experimental conditions from the paper back-to-back:

    python main.py --all --population 20 --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

from src.config import SimulationConfig
from src.runner import Simulation


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generative agent-based epidemic model (Williams et al., 2023)."
    )
    p.add_argument(
        "--condition",
        choices=["base", "self", "full"],
        default="full",
        help="Experimental condition (default: full).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all three conditions sequentially instead of just --condition.",
    )
    p.add_argument("--population", "-n", type=int, default=20)
    p.add_argument("--days", "-d", type=int, default=30)
    p.add_argument("--contact-rate", type=int, default=5)
    p.add_argument(
        "--infectivity",
        type=float,
        default=0.1,
        help="Per-contact transmission probability (paper default: 0.1).",
    )
    p.add_argument("--initial-infected", type=int, default=1)
    p.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Google GenAI model id (default: gemini-2.0-flash).",
    )
    p.add_argument("--output", default="results", help="Output directory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mock",
        action="store_true",
        help="Skip LLM calls; use a deterministic heuristic agent instead.",
    )
    return p


def _make_config(args: argparse.Namespace, condition: str) -> SimulationConfig:
    return SimulationConfig(
        condition=condition,  # type: ignore[arg-type]
        population=args.population,
        days=args.days,
        contact_rate=args.contact_rate,
        infectivity=args.infectivity,
        initial_infected=args.initial_infected,
        model=args.model,
        output_dir=args.output,
        seed=args.seed,
        mock=args.mock,
    )


async def _run_conditions(args: argparse.Namespace) -> None:
    conditions = ["base", "self", "full"] if args.all else [args.condition]
    for cond in conditions:
        cfg = _make_config(args, cond)
        sim = Simulation(cfg)
        await sim.run()
        print()


def main() -> None:
    args = _parser().parse_args()

    if not args.mock and not os.getenv("GOOGLE_API_KEY"):
        print(
            "WARNING: GOOGLE_API_KEY is not set. Either export it, put it in "
            ".env, or run with --mock to use the offline heuristic agent."
        )

    asyncio.run(_run_conditions(args))


if __name__ == "__main__":
    main()
