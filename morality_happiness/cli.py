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
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--food-scale", type=float, default=None)
    parser.add_argument("--scarcity-resource-multiplier", type=float, default=None)
    parser.add_argument("--abundance-resource-multiplier", type=float, default=None)
    parser.add_argument("--relief-source", choices=("external", "public", "none"), default=None)
    parser.add_argument("--public-food-reserve", type=float, default=None)
    parser.add_argument("--lambda-c", type=float, default=None)
    parser.add_argument("--lambda-g", type=float, default=None)
    parser.add_argument("--env-volatility", type=float, default=None)
    parser.add_argument("--relationship-memory", type=float, default=None)
    parser.add_argument("--moral-adaptation-rate", type=float, default=None)
    parser.add_argument("--promotion-period", type=int, default=None)
    parser.add_argument("--chieftain-slots", type=int, default=None)
    parser.add_argument("--hunters-per-chieftain", type=int, default=None)
    parser.add_argument("--disable-trade", action="store_true")
    parser.add_argument("--disable-ranks", action="store_true")
    parser.add_argument("--disable-rank-split", action="store_true")
    parser.add_argument(
        "--disable-participation-choice",
        action="store_true",
        help="Disable per-agent participation decisions and force participation to 1.0.",
    )
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
    if args.replications is not None:
        overrides["replications"] = args.replications
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.food_scale is not None:
        overrides["food_scale"] = args.food_scale
    if args.scarcity_resource_multiplier is not None:
        overrides["scarcity_resource_multiplier"] = args.scarcity_resource_multiplier
    if args.abundance_resource_multiplier is not None:
        overrides["abundance_resource_multiplier"] = args.abundance_resource_multiplier
    if args.relief_source is not None:
        overrides["relief_source"] = args.relief_source
    if args.public_food_reserve is not None:
        overrides["public_food_reserve"] = args.public_food_reserve
    if args.lambda_c is not None:
        overrides["lambda_c"] = args.lambda_c
    if args.lambda_g is not None:
        overrides["lambda_g"] = args.lambda_g
    if args.env_volatility is not None:
        overrides["environment_volatility"] = args.env_volatility
    if args.relationship_memory is not None:
        overrides["relationship_memory_weight"] = args.relationship_memory
    if args.moral_adaptation_rate is not None:
        overrides["moral_adaptation_rate"] = args.moral_adaptation_rate
    if args.promotion_period is not None:
        overrides["promotion_period"] = args.promotion_period
    if args.chieftain_slots is not None:
        overrides["chieftain_slots"] = args.chieftain_slots
    if args.hunters_per_chieftain is not None:
        overrides["hunters_per_chieftain"] = args.hunters_per_chieftain
    if args.disable_trade:
        overrides["trade_enabled"] = False
    if args.disable_ranks:
        overrides["rank_enabled"] = False
        overrides["promotion_enabled"] = False
    if args.disable_rank_split:
        overrides["rank_split_enabled"] = False
    if args.disable_participation_choice:
        overrides["enable_participation_choice"] = False
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
