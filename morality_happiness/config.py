from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import ActionMode, Scenario
from .settings import normalize_provider, simulation_defaults


SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario(
        id="S1",
        name="匮乏强组织总体行动",
        resource_label="高度匮乏",
        mode=ActionMode.ORGANIZED,
        base_resource=0.35,
        scarcity_pressure=0.65,
        opportunity=0.35,
        collectivization=1.00,
        individualization=0.00,
        moral_fit_weights={
            "self_dev": 0.0,
            "self_rest": 0.3,
            "alt_dev": 0.2,
            "alt_rest": 0.5,
        },
    ),
    "S2": Scenario(
        id="S2",
        name="匮乏弱组织抱团行动",
        resource_label="高度匮乏",
        mode=ActionMode.GROUP,
        base_resource=0.35,
        scarcity_pressure=0.65,
        opportunity=0.35,
        collectivization=0.50,
        individualization=0.50,
        moral_fit_weights={
            "self_dev": 0.1,
            "self_rest": 0.2,
            "alt_dev": 0.3,
            "alt_rest": 0.4,
        },
    ),
    "S3": Scenario(
        id="S3",
        name="匮乏弱组织各自行动",
        resource_label="高度匮乏",
        mode=ActionMode.INDIVIDUAL,
        base_resource=0.35,
        scarcity_pressure=0.65,
        opportunity=0.35,
        collectivization=0.00,
        individualization=1.00,
        moral_fit_weights={
            "self_dev": 0.2,
            "self_rest": 0.6,
            "alt_dev": 0.1,
            "alt_rest": 0.1,
        },
    ),
    "S4": Scenario(
        id="S4",
        name="丰裕强组织总体行动",
        resource_label="基础丰裕",
        mode=ActionMode.ORGANIZED,
        base_resource=0.75,
        scarcity_pressure=0.25,
        opportunity=0.75,
        collectivization=1.00,
        individualization=0.00,
        moral_fit_weights={
            "self_dev": 0.2,
            "self_rest": 0.1,
            "alt_dev": 0.5,
            "alt_rest": 0.2,
        },
    ),
    "S5": Scenario(
        id="S5",
        name="丰裕弱组织抱团行动",
        resource_label="基础丰裕",
        mode=ActionMode.GROUP,
        base_resource=0.75,
        scarcity_pressure=0.25,
        opportunity=0.75,
        collectivization=0.50,
        individualization=0.50,
        moral_fit_weights={
            "self_dev": 0.4,
            "self_rest": 0.0,
            "alt_dev": 0.4,
            "alt_rest": 0.2,
        },
    ),
    "S6": Scenario(
        id="S6",
        name="丰裕弱组织各自行动",
        resource_label="基础丰裕",
        mode=ActionMode.INDIVIDUAL,
        base_resource=0.75,
        scarcity_pressure=0.25,
        opportunity=0.75,
        collectivization=0.00,
        individualization=1.00,
        moral_fit_weights={
            "self_dev": 0.7,
            "self_rest": 0.1,
            "alt_dev": 0.1,
            "alt_rest": 0.1,
        },
    ),
}


@dataclass
class SimulationParameters:
    population_size: int = 120
    ticks: int = 200
    seed: int = 42
    need_base: float = 1.5
    food_scale: float = 1.6
    alpha: float = 1.15
    alpha_individual: float = 1.35
    lambda_c: float = 0.20
    lambda_g: float = 0.25
    org_cost_per_team: float = 0.75
    group_cost_per_team: float = 0.45
    base_combat_power: float = 10.0
    prey_meat_scale: float = 0.18
    environment_volatility: float = 0.10
    organized_equal_weight: float = 0.80
    share_norm_alt_weight: float = 0.40
    share_norm_restraint_weight: float = 0.20
    restraint_need_discount: float = 0.12
    reserve_cap_days: float = 7.0
    typical_team_size: int = 10
    target_group_size: int = 8
    min_group_size: int = 3
    max_group_size: int = 12
    regroup_interval: int = 20
    individual_transfer_strength: float = 0.18
    group_need_memory_weight: float = 0.30
    moral_consistency_weight: float = 0.20
    autonomy_weight: float = 0.10
    llm_provider: str = "rule"
    llm_sample_rate: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, object] | None) -> "SimulationParameters":
        merged: dict[str, object] = {
            key: value
            for key, value in simulation_defaults().items()
            if key in cls.__dataclass_fields__
        }
        if values:
            merged.update(values)
        allowed = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in merged.items() if key in allowed}
        params = cls(**clean)
        return params.normalized()

    def normalized(self) -> "SimulationParameters":
        self.population_size = max(4, int(self.population_size))
        self.ticks = max(1, int(self.ticks))
        self.seed = int(self.seed)
        self.need_base = max(0.1, float(self.need_base))
        self.food_scale = max(0.0, float(self.food_scale))
        self.alpha = max(0.0, float(self.alpha))
        self.alpha_individual = max(0.0, float(self.alpha_individual))
        self.lambda_c = max(0.0, float(self.lambda_c))
        self.lambda_g = max(0.0, float(self.lambda_g))
        self.org_cost_per_team = max(0.0, float(self.org_cost_per_team))
        self.group_cost_per_team = max(0.0, float(self.group_cost_per_team))
        self.base_combat_power = max(0.1, float(self.base_combat_power))
        self.prey_meat_scale = max(0.0, float(self.prey_meat_scale))
        self.environment_volatility = max(0.0, min(0.5, float(self.environment_volatility)))
        self.organized_equal_weight = max(0.0, min(1.0, float(self.organized_equal_weight)))
        self.share_norm_alt_weight = max(0.0, min(1.0, float(self.share_norm_alt_weight)))
        self.share_norm_restraint_weight = max(
            0.0, min(1.0, float(self.share_norm_restraint_weight))
        )
        self.restraint_need_discount = max(0.0, min(0.35, float(self.restraint_need_discount)))
        self.reserve_cap_days = max(0.0, min(30.0, float(self.reserve_cap_days)))
        self.typical_team_size = max(1, int(self.typical_team_size))
        self.target_group_size = max(2, int(self.target_group_size))
        self.min_group_size = max(1, int(self.min_group_size))
        self.max_group_size = max(self.min_group_size, int(self.max_group_size))
        self.regroup_interval = max(1, int(self.regroup_interval))
        self.individual_transfer_strength = max(
            0.0, min(0.8, float(self.individual_transfer_strength))
        )
        self.group_need_memory_weight = max(0.0, min(1.0, float(self.group_need_memory_weight)))
        self.moral_consistency_weight = max(
            0.0, min(0.5, float(self.moral_consistency_weight))
        )
        self.autonomy_weight = max(0.0, min(0.5, float(self.autonomy_weight)))
        self.llm_provider = normalize_provider(str(self.llm_provider))
        self.llm_sample_rate = max(0.0, min(1.0, float(self.llm_sample_rate)))
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
