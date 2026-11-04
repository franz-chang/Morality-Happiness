from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morality_happiness.config import SimulationParameters
from morality_happiness.simulation import Simulation


def measure_supply_ratio(
    *, scenario_id: str, multiplier: float, population: int, ticks: int, seed: int
) -> float:
    scarcity = scenario_id == "S3"
    overrides: dict[str, object] = {
        "population_size": population,
        "ticks": ticks,
        "replications": 1,
        "seed": seed,
        "trade_enabled": False,
        "rank_enabled": False,
        "rank_split_enabled": False,
        "promotion_enabled": False,
        "relief_source": "external",
        "scarcity_resource_multiplier": multiplier if scarcity else 1.0,
        "abundance_resource_multiplier": 1.0 if scarcity else multiplier,
    }
    result = Simulation(scenario_id, SimulationParameters.from_dict(overrides)).run()
    return float(result["society_metrics"]["realized_supply_ratio"])


def calibrate(
    *, scenario_id: str, target: float, population: int, ticks: int, seed: int, iterations: int
) -> float:
    low, high = 0.0, 5.0
    low_ratio = measure_supply_ratio(
        scenario_id=scenario_id, multiplier=low, population=population, ticks=ticks, seed=seed
    )
    high_ratio = measure_supply_ratio(
        scenario_id=scenario_id, multiplier=high, population=population, ticks=ticks, seed=seed
    )
    if not low_ratio <= target <= high_ratio:
        raise RuntimeError(
            f"{scenario_id} target {target:.3f} is outside pilot range "
            f"[{low_ratio:.3f}, {high_ratio:.3f}] at multipliers [0, 5]."
        )

    best = (abs(low_ratio - target), low)
    for iteration in range(iterations):
        midpoint = (low + high) / 2.0
        ratio = measure_supply_ratio(
            scenario_id=scenario_id,
            multiplier=midpoint,
            population=population,
            ticks=ticks,
            seed=seed,
        )
        print(
            f"{scenario_id} calibration {iteration + 1}/{iterations}: "
            f"multiplier={midpoint:.5f}, realized={ratio:.5f}, target={target:.5f}",
            flush=True,
        )
        if abs(ratio - target) < best[0]:
            best = (abs(ratio - target), midpoint)
        if ratio < target:
            low = midpoint
        else:
            high = midpoint
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate scarcity/abundance multipliers on the S3/S6 no-trade baseline."
    )
    parser.add_argument("--scarcity-target", type=float, default=0.80)
    parser.add_argument("--abundance-target", type=float, default=1.025)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--ticks", type=int, default=560)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    scarcity = calibrate(
        scenario_id="S3",
        target=args.scarcity_target,
        population=args.population,
        ticks=args.ticks,
        seed=args.seed,
        iterations=args.iterations,
    )
    abundance = calibrate(
        scenario_id="S6",
        target=args.abundance_target,
        population=args.population,
        ticks=args.ticks,
        seed=args.seed,
        iterations=args.iterations,
    )
    print("\nRecommended config values:")
    print(f"  scarcity_resource_multiplier: {scarcity:.5f}")
    print(f"  abundance_resource_multiplier: {abundance:.5f}")


if __name__ == "__main__":
    main()
