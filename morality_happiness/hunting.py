from __future__ import annotations

import random
from dataclasses import dataclass

from .utils import clip, weighted_choice


@dataclass(frozen=True)
class PreyTier:
    name: str
    spawn_probability: float
    combat_power: float
    meat: float


PREY_TIERS: tuple[PreyTier, ...] = (
    PreyTier("Small", 0.60, 5.0, 10.0),
    PreyTier("Medium", 0.30, 20.0, 50.0),
    PreyTier("Large", 0.10, 50.0, 150.0),
)


@dataclass
class HuntOutcome:
    meat: float
    successes: int
    failures: int
    events: int
    expected_meat: float
    prey_counts: dict[str, int]

    @property
    def success_rate(self) -> float:
        if self.events == 0:
            return 0.0
        return self.successes / self.events


class HuntingEngine:
    """Discrete prey encounter model from the hunting-system supplement."""

    def __init__(self, meat_scale: float = 0.18) -> None:
        self.meat_scale = max(0.0, meat_scale)

    def resolve_hunt(
        self,
        *,
        team_power: float,
        rng: random.Random,
        events: int = 1,
        encounter_rate: float = 1.0,
    ) -> HuntOutcome:
        events = max(0, int(events))
        team_power = max(0.0, team_power)
        encounter_rate = clip(encounter_rate, 0.0, 1.0)
        meat = 0.0
        expected = 0.0
        successes = 0
        failures = 0
        prey_counts = {tier.name: 0 for tier in PREY_TIERS}
        weights = [tier.spawn_probability for tier in PREY_TIERS]

        for _ in range(events):
            if rng.random() > encounter_rate:
                continue
            prey = weighted_choice(PREY_TIERS, weights, rng)
            prey_counts[prey.name] += 1
            success_probability = team_power / (team_power + prey.combat_power)
            expected += success_probability * prey.meat * self.meat_scale
            if rng.random() < success_probability:
                successes += 1
                meat += prey.meat * self.meat_scale
            else:
                failures += 1

        return HuntOutcome(
            meat=meat,
            successes=successes,
            failures=failures,
            events=max(1, events),
            expected_meat=expected,
            prey_counts=prey_counts,
        )

