from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SCENARIOS, SimulationParameters
from .simulation import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the morality-happiness simulation.")
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario id S1-S6, comma separated ids, or 'all'.",
    )
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--food-scale", type=float, default=None)
    parser.add_argument("--lambda-c", type=float, default=None)
    parser.add_argument("--lambda-g", type=float, default=None)
    parser.add_argument("--env-volatility", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scenario == "all":
        scenario_ids = list(SCENARIOS)
    else:
        scenario_ids = [item.strip().upper() for item in args.scenario.split(",") if item.strip()]
    overrides = {}
    if args.ticks is not None:
        overrides["ticks"] = args.ticks
    if args.population is not None:
        overrides["population_size"] = args.population
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.food_scale is not None:
        overrides["food_scale"] = args.food_scale
    if args.lambda_c is not None:
        overrides["lambda_c"] = args.lambda_c
    if args.lambda_g is not None:
        overrides["lambda_g"] = args.lambda_g
    if args.env_volatility is not None:
        overrides["environment_volatility"] = args.env_volatility
    params = SimulationParameters.from_dict(overrides)
    result = run_batch(scenario_ids, params)
    print(_summary_table(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON result to {args.out}")


def _summary_table(batch: dict[str, object]) -> str:
    rows = batch["comparison"]  # type: ignore[index]
    header = (
        "Scenario | Avg food | Avg wellbeing | Gini | Survival | Total food\n"
        "---------|----------|---------------|------|----------|-----------"
    )
    lines = [header]
    for row in rows:  # type: ignore[assignment]
        lines.append(
            f"{row['scenario_id']} {row['scenario_name']} | "
            f"{row['avg_food_allocated']:.3f} | "
            f"{row['avg_wellbeing']:.2f} | "
            f"{row['social_gini_index']:.3f} | "
            f"{row['minimum_survival_rate']:.3f} | "
            f"{row['total_social_food']:.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
