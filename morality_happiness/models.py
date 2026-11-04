from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


MORAL_COMPONENTS = ("self_dev", "self_rest", "alt_dev", "alt_rest")


class MoralType(str, Enum):
    SELF_DEVELOPMENT = "self_development"
    SELF_RESTRAINT = "self_restraint"
    ALTRUISTIC_DEVELOPMENT = "altruistic_development"
    ALTRUISTIC_RESTRAINT = "altruistic_restraint"

    @property
    def label(self) -> str:
        return {
            MoralType.SELF_DEVELOPMENT: "自我发展",
            MoralType.SELF_RESTRAINT: "自我克制",
            MoralType.ALTRUISTIC_DEVELOPMENT: "发展型利他",
            MoralType.ALTRUISTIC_RESTRAINT: "克制型利他",
        }[self]


MORAL_TYPE_WEIGHTS: dict[MoralType, dict[str, float]] = {
    MoralType.SELF_DEVELOPMENT: {
        "self_dev": 0.70,
        "self_rest": 0.10,
        "alt_dev": 0.15,
        "alt_rest": 0.05,
    },
    MoralType.SELF_RESTRAINT: {
        "self_dev": 0.10,
        "self_rest": 0.70,
        "alt_dev": 0.05,
        "alt_rest": 0.15,
    },
    MoralType.ALTRUISTIC_DEVELOPMENT: {
        "self_dev": 0.15,
        "self_rest": 0.05,
        "alt_dev": 0.70,
        "alt_rest": 0.10,
    },
    MoralType.ALTRUISTIC_RESTRAINT: {
        "self_dev": 0.05,
        "self_rest": 0.15,
        "alt_dev": 0.10,
        "alt_rest": 0.70,
    },
}


GROUP_CHOICE_WEIGHTS: dict[MoralType, dict[str, float]] = {
    MoralType.SELF_DEVELOPMENT: {
        "expected_food": 0.55,
        "fairness": 0.10,
        "similarity": 0.15,
        "stability": 0.20,
    },
    MoralType.SELF_RESTRAINT: {
        "expected_food": 0.30,
        "fairness": 0.20,
        "similarity": 0.15,
        "stability": 0.35,
    },
    MoralType.ALTRUISTIC_DEVELOPMENT: {
        "expected_food": 0.30,
        "fairness": 0.25,
        "similarity": 0.25,
        "stability": 0.20,
    },
    MoralType.ALTRUISTIC_RESTRAINT: {
        "expected_food": 0.20,
        "fairness": 0.40,
        "similarity": 0.20,
        "stability": 0.20,
    },
}


class AgeGroup(str, Enum):
    YOUNG = "young"
    MIDDLE = "middle"
    OLD = "old"

    @property
    def label(self) -> str:
        return {
            AgeGroup.YOUNG: "青年",
            AgeGroup.MIDDLE: "中年",
            AgeGroup.OLD: "老年",
        }[self]


AGE_PROFILES: dict[AgeGroup, dict[str, float]] = {
    AgeGroup.YOUNG: {
        "physical": 0.75,
        "exploration": 0.70,
        "coordination": 0.45,
        "need_multiplier": 1.05,
    },
    AgeGroup.MIDDLE: {
        "physical": 0.70,
        "exploration": 0.55,
        "coordination": 0.65,
        "need_multiplier": 1.00,
    },
    AgeGroup.OLD: {
        "physical": 0.45,
        "exploration": 0.40,
        "coordination": 0.75,
        "need_multiplier": 0.90,
    },
}


class Status(str, Enum):
    NORMAL = "Normal"
    EXTREME_FATIGUE = "Extreme_Fatigue"


class Role(str, Enum):
    CHIEF = "chief"
    CHIEFTAIN = "chieftain"
    HUNTER = "hunter"

    @property
    def label(self) -> str:
        return {
            Role.CHIEF: "首领",
            Role.CHIEFTAIN: "酋长",
            Role.HUNTER: "猎人",
        }[self]


class ActionMode(str, Enum):
    ORGANIZED = "organized"
    GROUP = "group"
    INDIVIDUAL = "individual"

    @property
    def label(self) -> str:
        return {
            ActionMode.ORGANIZED: "全体高度有组织行动",
            ActionMode.GROUP: "自由抱团行动",
            ActionMode.INDIVIDUAL: "各自行动",
        }[self]


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    resource_label: str
    mode: ActionMode
    base_resource: float
    scarcity_pressure: float
    opportunity: float
    collectivization: float
    individualization: float
    moral_fit_weights: dict[str, float]
    target_supply_ratio: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["mode_label"] = self.mode.label
        return data


@dataclass
class AgentState:
    id: int
    age_group: AgeGroup
    sex: str
    moral_type: MoralType
    moral_weights: dict[str, float]
    physical: float
    exploration: float
    coordination: float
    compliance: float
    need: float
    wellbeing: float
    food_reserve: float = 0.0
    pending_relief: float = 0.0
    fatigue: float = 0.0
    status: Status = Status.NORMAL
    role: Role = Role.HUNTER
    role_since_tick: int = 0
    superior_id: int | None = None
    shell_balance: float = 0.0
    shell_earned_from_trade: float = 0.0
    shell_spent_on_food: float = 0.0
    shell_earned_from_hunting: float = 0.0
    food_bought: float = 0.0
    food_sold: float = 0.0
    trade_count: int = 0
    group_id: int | None = None
    previous_group_id: int | None = None
    group_affinity: dict[int, float] = field(default_factory=dict)
    cumulative_food: float = 0.0
    cumulative_food_consumed: float = 0.0
    cumulative_unmet_food: float = 0.0
    cumulative_relief_received: float = 0.0
    hunting_contribution: float = 0.0
    period_hunting_contribution: float = 0.0
    extreme_triggers: int = 0
    allocated_history: list[float] = field(default_factory=list)
    food_consumed_history: list[float] = field(default_factory=list)
    unmet_food_history: list[float] = field(default_factory=list)
    relief_received_history: list[float] = field(default_factory=list)
    extreme_trigger_history: list[int] = field(default_factory=list)
    wellbeing_history: list[float] = field(default_factory=list)
    moral_consistency_history: list[float] = field(default_factory=list)

    @property
    def alt_orientation(self) -> float:
        return self.moral_weights["alt_dev"] + self.moral_weights["alt_rest"]

    @property
    def restraint_orientation(self) -> float:
        return self.moral_weights["self_rest"] + self.moral_weights["alt_rest"]

    @property
    def development_orientation(self) -> float:
        return self.moral_weights["self_dev"] + self.moral_weights["alt_dev"]

    @property
    def self_orientation(self) -> float:
        return self.moral_weights["self_dev"] + self.moral_weights["self_rest"]

    def to_snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "age_group": self.age_group.value,
            "age_label": self.age_group.label,
            "sex": self.sex,
            "moral_type": self.moral_type.value,
            "moral_label": self.moral_type.label,
            "physical": self.physical,
            "exploration": self.exploration,
            "coordination": self.coordination,
            "need": self.need,
            "wellbeing": self.wellbeing,
            "food_reserve": self.food_reserve,
            "pending_relief": self.pending_relief,
            "fatigue": self.fatigue,
            "status": self.status.value,
            "role": self.role.value,
            "role_since_tick": self.role_since_tick,
            "role_label": self.role.label,
            "superior_id": self.superior_id,
            "shell_balance": self.shell_balance,
            "shell_earned_from_trade": self.shell_earned_from_trade,
            "shell_spent_on_food": self.shell_spent_on_food,
            "shell_earned_from_hunting": self.shell_earned_from_hunting,
            "food_bought": self.food_bought,
            "food_sold": self.food_sold,
            "trade_count": self.trade_count,
            "group_id": self.group_id,
            "group_affinity": dict(self.group_affinity),
            "cumulative_food": self.cumulative_food,
            "cumulative_food_consumed": self.cumulative_food_consumed,
            "cumulative_unmet_food": self.cumulative_unmet_food,
            "cumulative_relief_received": self.cumulative_relief_received,
            "hunting_contribution": self.hunting_contribution,
            "extreme_triggers": self.extreme_triggers,
        }


@dataclass
class AgentDecision:
    moral_fit: float
    effort: float
    compliance_action: float
    share_rate: float
    restraint_rate: float
    participation_action: float = 1.0
    decision_source: str = "rule"
    decision_reason: str = ""


@dataclass
class WellbeingScores:
    eval_score: float
    peer_score: float
    self_score: float
    composite: float
    material_score: float
    moral_consistency: float
    perceived_fairness: float
    autonomy: float
    social_belonging: float
    stress: float
    eval_reason: str = ""
    peer_reason: str = ""
    self_reason: str = ""
