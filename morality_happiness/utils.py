from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    if not vals:
        return default
    return sum(vals) / len(vals)


def stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals)


def finite(value: float, default: float = 0.0) -> float:
    if math.isfinite(value):
        return value
    return default


def gini(values: Sequence[float]) -> float:
    vals = sorted(max(0.0, float(v)) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 1e-12:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(vals))
    return clip((2.0 * weighted) / (n * total) - (n + 1.0) / n, 0.0, 1.0)


def weighted_choice(items: Sequence[T], weights: Sequence[float], rng: random.Random) -> T:
    total = sum(max(0.0, w) for w in weights)
    if total <= 1e-12:
        return items[rng.randrange(len(items))]
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights, strict=True):
        cumulative += max(0.0, weight)
        if cumulative >= threshold:
            return item
    return items[-1]


def normalize_dict(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 1e-12:
        even = 1.0 / max(1, len(values))
        return {key: even for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}

