from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morality_happiness import SimulationParameters, run_batch


def main() -> None:
    params = SimulationParameters(population_size=120, ticks=40, seed=7).normalized()
    result = run_batch(parameters=params)
    validate(result)
    out = Path("outputs/smoke.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("smoke ok")
    for row in result["comparison"]:
        print(
            row["scenario_id"],
            f"food={row['avg_food_allocated']:.3f}",
            f"wellbeing={row['avg_wellbeing']:.2f}",
            f"gini={row['social_gini_index']:.3f}",
            f"survival={row['minimum_survival_rate']:.3f}",
        )


def validate(value: object, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            validate(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            validate(item, f"{path}[{idx}]")
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite value at {path}: {value}"
        if path.endswith("avg_wellbeing") or path.endswith("wellbeing"):
            assert 0.0 <= value <= 100.0, f"wellbeing out of range at {path}: {value}"
        if path.endswith("social_gini_index") or path.endswith("group_gini_index"):
            assert 0.0 <= value <= 1.0, f"gini out of range at {path}: {value}"
        if "food" in path or "reserve" in path:
            assert value >= -1e-9, f"negative food value at {path}: {value}"


if __name__ == "__main__":
    main()
