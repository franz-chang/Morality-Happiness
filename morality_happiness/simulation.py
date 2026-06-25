from __future__ import annotations

import math
import json
import random
from typing import Any, Callable

from .config import SCENARIOS, SimulationParameters
from .hunting import HuntingEngine
from .llm import LLMProviderConfig, build_llm_provider, provider_status
from .metrics import aggregate_group_metrics, summarize_by_moral_type, summarize_society
from .models import (
    AGE_PROFILES,
    GROUP_CHOICE_WEIGHTS,
    MORAL_COMPONENTS,
    MORAL_TYPE_WEIGHTS,
    ActionMode,
    AgeGroup,
    AgentDecision,
    AgentState,
    MoralType,
    Scenario,
    Status,
)
from .utils import clip, gini, mean, safe_div
from .wellbeing import evaluate_wellbeing


StreamCallback = Callable[[dict[str, Any]], None]
StopChecker = Callable[[], bool]


class Simulation:
    def __init__(
        self,
        scenario_id: str,
        parameters: SimulationParameters | None = None,
        event_callback: StreamCallback | None = None,
        stop_checker: StopChecker | None = None,
    ) -> None:
        if scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
        self.scenario: Scenario = SCENARIOS[scenario_id]
        self.params = (
            parameters.normalized()
            if parameters is not None
            else SimulationParameters.from_dict({})
        )
        scenario_offset = int(self.scenario.id[1:]) * 100_000
        self.rng = random.Random(self.params.seed + scenario_offset)
        self.hunting = HuntingEngine(meat_scale=self.params.prey_meat_scale)
        self.agents = self._initialize_agents()
        self.groups: dict[int, list[int]] = {}
        self.group_memory: dict[int, dict[str, float]] = {}
        self.group_records: list[dict[str, Any]] = []
        self.time_series: list[dict[str, Any]] = []
        self.agent_logs: list[dict[str, Any]] = []
        self.system_logs: list[dict[str, Any]] = []
        self.event_callback = event_callback
        self.stop_checker = stop_checker
        self._log_sequence = 0
        self.total_food_by_tick: list[float] = []
        self.survival_rates: list[float] = []
        self.last_underfed_rate = self.scenario.scarcity_pressure
        self.model_status = provider_status(self.params.llm_provider).to_dict()
        self.llm_provider = None
        self.llm_config: LLMProviderConfig | None = None
        if self.params.llm_provider != "rule":
            self._activate_llm_decision_driver()

    def _activate_llm_decision_driver(self) -> None:
        if self.model_status.get("using_fallback") or not self.model_status.get("active"):
            raise RuntimeError(
                f"{self.params.llm_provider} 模型行为驱动未就绪: {self.model_status.get('reason', '')}"
            )
        self.llm_config = LLMProviderConfig.from_config(self.params.llm_provider)
        try:
            self.llm_provider = build_llm_provider(self.llm_config, allow_fallback=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{self.params.llm_provider} 模型行为驱动初始化失败: {exc}"
            ) from exc

    def run(self) -> dict[str, Any]:
        self._append_system_log(
            {
                "tick": 0,
                "scenario_id": self.scenario.id,
                "level": "info",
                "event": "model_provider",
                "message": (
                    f"模型驱动: {self.model_status['framework']} / "
                    f"{self.model_status['provider']} / {self.model_status['model']}. "
                    f"{self.model_status['reason']}"
                ),
            }
        )
        if self.llm_provider is not None:
            self._append_system_log(
                {
                    "tick": 0,
                    "scenario_id": self.scenario.id,
                    "level": "info",
                    "event": "llm_agent_driver",
                    "message": (
                        f"LLM 行为驱动已启用: {self.model_status['provider']} / "
                        f"{self.model_status['model']}。每一轮按 Agent id 顺序进行独立行为判断，"
                        "不使用本地规则决策公式。"
                    ),
                }
            )
        for tick in range(1, self.params.ticks + 1):
            self._run_tick(tick)
        return self._result()

    def _append_system_log(self, record: dict[str, Any]) -> None:
        self._log_sequence += 1
        record.setdefault("sequence", self._log_sequence)
        self.system_logs.append(record)
        self._emit({"type": "log", "target": "system_logs", "scenario_id": self.scenario.id, "log": record})

    def _append_agent_log(self, record: dict[str, Any]) -> None:
        self._log_sequence += 1
        record.setdefault("sequence", self._log_sequence)
        self.agent_logs.append(record)
        self._emit({"type": "log", "target": "agent_logs", "scenario_id": self.scenario.id, "log": record})

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def _raise_if_stopped(self) -> None:
        if self.stop_checker is not None and self.stop_checker():
            raise RuntimeError("simulation stopped by user")

    def _initialize_agents(self) -> list[AgentState]:
        population = self.params.population_size
        moral_cycle = list(MoralType)
        moral_types = [moral_cycle[idx % len(moral_cycle)] for idx in range(population)]
        self.rng.shuffle(moral_types)
        agents: list[AgentState] = []
        for agent_id, moral_type in enumerate(moral_types):
            age_group = self._sample_age_group()
            profile = AGE_PROFILES[age_group]
            moral_weights = dict(MORAL_TYPE_WEIGHTS[moral_type])
            physical = self._ability(profile["physical"])
            exploration = self._ability(profile["exploration"])
            coordination = self._ability(profile["coordination"])
            restraint = moral_weights["self_rest"] + moral_weights["alt_rest"]
            alt = moral_weights["alt_dev"] + moral_weights["alt_rest"]
            compliance = clip(
                0.35 + 0.35 * coordination + 0.15 * restraint + 0.10 * alt + self.rng.gauss(0, 0.05),
                0.0,
                1.0,
            )
            need = self.params.need_base * profile["need_multiplier"]
            wellbeing = clip(self.rng.gauss(60.0, 10.0), 35.0, 82.0)
            agents.append(
                AgentState(
                    id=agent_id,
                    age_group=age_group,
                    sex="F" if self.rng.random() < 0.5 else "M",
                    moral_type=moral_type,
                    moral_weights=moral_weights,
                    physical=physical,
                    exploration=exploration,
                    coordination=coordination,
                    compliance=compliance,
                    need=need,
                    wellbeing=wellbeing,
                )
            )
        return agents

    def _sample_age_group(self) -> AgeGroup:
        roll = self.rng.random()
        if roll < 0.35:
            return AgeGroup.YOUNG
        if roll < 0.80:
            return AgeGroup.MIDDLE
        return AgeGroup.OLD

    def _ability(self, center: float) -> float:
        return clip(self.rng.gauss(center, 0.08), 0.15, 0.95)

    def _run_tick(self, tick: int) -> None:
        self._raise_if_stopped()
        self._recover_rescued_agents()
        pressure_index = clip(
            (1.0 - self.params.group_need_memory_weight) * self.scenario.scarcity_pressure
            + self.params.group_need_memory_weight * self.last_underfed_rate,
            0.0,
            1.0,
        )
        if self.scenario.mode is ActionMode.GROUP:
            self._ensure_groups_for_tick(tick)

        decisions: dict[int, AgentDecision] = {}
        prior_decisions: list[dict[str, Any]] = []
        for agent in self.agents:
            self._raise_if_stopped()
            decision = self._decide(
                agent,
                pressure_index,
                tick=tick,
                prior_decisions=prior_decisions,
            )
            self._raise_if_stopped()
            decisions[agent.id] = decision
            if decision.decision_source != "rule":
                self._append_agent_log(
                    self._agent_decision_log_record(
                        tick=tick,
                        agent=agent,
                        decision=decision,
                    )
                )
            prior_decisions.append(
                {
                    "agent_id": agent.id,
                    "moral_type": agent.moral_type.value,
                    "effort": round(decision.effort, 3),
                    "share_rate": round(decision.share_rate, 3),
                    "restraint_rate": round(decision.restraint_rate, 3),
                }
            )
        if self.scenario.mode is ActionMode.ORGANIZED:
            allocations, contexts, production = self._organized_tick(tick, decisions)
        elif self.scenario.mode is ActionMode.GROUP:
            allocations, contexts, production = self._group_tick(tick, decisions)
        else:
            allocations, contexts, production = self._individual_tick(tick, decisions)

        starved_count = 0
        for agent in self.agents:
            self._raise_if_stopped()
            decision = decisions[agent.id]
            allocated = max(0.0, allocations.get(agent.id, 0.0))
            context = contexts.get(agent.id, {})
            reserve_before = agent.food_reserve
            fatigue_before = agent.fatigue
            wellbeing_before = agent.wellbeing
            status_before = agent.status
            effective_need = max(
                0.65 * agent.need,
                agent.need * (1.0 - self.params.restraint_need_discount * decision.restraint_rate),
            )
            available = agent.food_reserve + allocated
            starved = available + 1e-9 < effective_need

            if starved:
                starved_count += 1
                agent.extreme_triggers += 1
                agent.status = Status.EXTREME_FATIGUE
                agent.food_reserve = agent.need
                agent.fatigue = max(0.0, agent.fatigue - 0.1)
            else:
                agent.status = Status.NORMAL
                reserve_cap = agent.need * self.params.reserve_cap_days
                agent.food_reserve = min(reserve_cap, max(0.0, available - effective_need))

            if decision.effort <= 1e-9 or status_before is Status.EXTREME_FATIGUE:
                agent.fatigue = max(0.0, agent.fatigue - 0.1)
            elif starved or context.get("hunt_failed", False):
                agent.fatigue = min(2.0, agent.fatigue + 0.2)
            elif decision.effort > 0.82:
                agent.fatigue = min(2.0, agent.fatigue + 0.03)

            scores = evaluate_wellbeing(
                agent=agent,
                decision=decision,
                scenario=self.scenario,
                allocated_food=allocated,
                effective_need=effective_need,
                perceived_fairness=float(context.get("fairness", 0.5)),
                autonomy=float(context.get("autonomy", self.scenario.individualization)),
                social_belonging=float(context.get("social_belonging", 0.4)),
                contribution_signal=float(context.get("contribution_signal", 0.5)),
                starved=starved,
                moral_consistency_weight=self.params.moral_consistency_weight,
                autonomy_weight=self.params.autonomy_weight,
            )
            agent.wellbeing = scores.composite
            agent.cumulative_food += allocated
            agent.allocated_history.append(allocated)
            agent.wellbeing_history.append(agent.wellbeing)
            agent.moral_consistency_history.append(scores.moral_consistency)
            self._append_agent_log(
                self._agent_log_record(
                    tick=tick,
                    agent=agent,
                    decision=decision,
                    context=context,
                    allocated=allocated,
                    effective_need=effective_need,
                    reserve_before=reserve_before,
                    fatigue_before=fatigue_before,
                    wellbeing_before=wellbeing_before,
                    status_before=status_before,
                    starved=starved,
                    scores={
                        "material_score": scores.material_score,
                        "moral_consistency": scores.moral_consistency,
                        "perceived_fairness": scores.perceived_fairness,
                        "autonomy": scores.autonomy,
                        "social_belonging": scores.social_belonging,
                        "stress": scores.stress,
                    },
                )
            )

        survival_rate = 1.0 - starved_count / max(1, len(self.agents))
        self.last_underfed_rate = 1.0 - survival_rate
        self.survival_rates.append(survival_rate)
        self.total_food_by_tick.append(production["total_food"])
        self.time_series.append(self._tick_record(tick, production, allocations, survival_rate))
        self._append_system_log(
            {
                "tick": tick,
                "scenario_id": self.scenario.id,
                "level": "info",
                "event": "tick_summary",
                "message": (
                    f"{self.scenario.id} 第 {tick} 轮: 总产出 {production['total_food']:.2f}, "
                    f"平均幸福 {mean(agent.wellbeing for agent in self.agents):.2f}, "
                    f"最低保障率 {survival_rate:.2f}, 极度疲劳 {starved_count} 人。"
                ),
                "total_food": production["total_food"],
                "avg_wellbeing": mean(agent.wellbeing for agent in self.agents),
                "minimum_survival_rate": survival_rate,
                "extreme_fatigue_count": starved_count,
            }
        )

    def _recover_rescued_agents(self) -> None:
        for agent in self.agents:
            if agent.status is Status.EXTREME_FATIGUE and agent.food_reserve >= agent.need:
                agent.status = Status.NORMAL

    def _decide(
        self,
        agent: AgentState,
        pressure_index: float,
        *,
        tick: int,
        prior_decisions: list[dict[str, Any]],
    ) -> AgentDecision:
        moral_fit = sum(
            agent.moral_weights[key] * self.scenario.moral_fit_weights[key]
            for key in MORAL_COMPONENTS
        )
        if self.llm_provider is not None:
            return self._llm_decide(
                agent=agent,
                pressure_index=pressure_index,
                moral_fit=moral_fit,
                tick=tick,
                prior_decisions=prior_decisions,
            )
        return self._rule_decide(agent, pressure_index, moral_fit)

    def _rule_decide(
        self,
        agent: AgentState,
        pressure_index: float,
        moral_fit: float,
    ) -> AgentDecision:
        if agent.status is Status.EXTREME_FATIGUE:
            effort = 0.0
        else:
            effort = clip(
                0.30
                + 0.30 * agent.physical
                + 0.20 * moral_fit
                + 0.10 * (agent.wellbeing / 100.0)
                - 0.10 * agent.fatigue,
                0.0,
                1.0,
            )
        compliance_action = clip(
            0.20
            + 0.30 * agent.compliance
            + 0.25 * agent.alt_orientation
            + 0.15 * agent.restraint_orientation
            + 0.10 * self.scenario.collectivization,
            0.0,
            1.0,
        )
        share_rate = clip(
            0.05
            + 0.35 * agent.alt_orientation
            + 0.20 * self.scenario.collectivization
            + 0.15 * pressure_index
            + 0.10 * moral_fit,
            0.0,
            1.0,
        )
        age_restraint = {
            AgeGroup.YOUNG: 0.25,
            AgeGroup.MIDDLE: 0.50,
            AgeGroup.OLD: 0.75,
        }[agent.age_group]
        restraint_rate = clip(
            0.05
            + 0.35 * agent.restraint_orientation
            + 0.25 * self.scenario.scarcity_pressure
            + 0.10 * age_restraint,
            0.0,
            1.0,
        )
        return AgentDecision(
            moral_fit=moral_fit,
            effort=effort,
            compliance_action=compliance_action,
            share_rate=share_rate,
            restraint_rate=restraint_rate,
            decision_source="rule",
            decision_reason="本地规则公式",
        )

    def _llm_decide(
        self,
        *,
        agent: AgentState,
        pressure_index: float,
        moral_fit: float,
        tick: int,
        prior_decisions: list[dict[str, Any]],
    ) -> AgentDecision:
        assert self.llm_provider is not None
        prompt = self._agent_decision_prompt(
            agent=agent,
            pressure_index=pressure_index,
            moral_fit=moral_fit,
            tick=tick,
            prior_decisions=prior_decisions,
        )
        try:
            raw = self.llm_provider.complete(prompt)
            payload = self._parse_llm_decision(raw)
        except Exception as exc:
            raise RuntimeError(
                f"{self.model_status.get('model')} 第 {tick} 轮 Agent {agent.id} 行为决策失败: {exc}"
            ) from exc

        effort = self._required_unit_float(payload, "effort")
        if agent.status is Status.EXTREME_FATIGUE:
            effort = 0.0
        return AgentDecision(
            moral_fit=moral_fit,
            effort=effort,
            compliance_action=self._required_unit_float(payload, "compliance_action"),
            share_rate=self._required_unit_float(payload, "share_rate"),
            restraint_rate=self._required_unit_float(payload, "restraint_rate"),
            decision_source=f"llm:{self.model_status.get('provider')}",
            decision_reason=str(payload.get("reason", "")).strip()[:280],
        )

    def _agent_decision_prompt(
        self,
        *,
        agent: AgentState,
        pressure_index: float,
        moral_fit: float,
        tick: int,
        prior_decisions: list[dict[str, Any]],
    ) -> str:
        observed_prior = prior_decisions[-12:]
        payload = {
            "task": "decide_one_agent_action",
            "output_schema": {
                "effort": "float 0..1; hunting/production effort for this tick",
                "compliance_action": "float 0..1; willingness to follow collective or group coordination",
                "share_rate": "float 0..1; willingness to share surplus or accept redistribution",
                "restraint_rate": "float 0..1; willingness to restrain own need/consumption under pressure",
                "reason": "brief Chinese explanation, <= 60 chars",
            },
            "scenario": {
                "id": self.scenario.id,
                "name": self.scenario.name,
                "mode": self.scenario.mode.value,
                "resource_label": self.scenario.resource_label,
                "scarcity_pressure": round(self.scenario.scarcity_pressure, 4),
                "opportunity": round(self.scenario.opportunity, 4),
                "collectivization": round(self.scenario.collectivization, 4),
                "individualization": round(self.scenario.individualization, 4),
            },
            "tick_context": {
                "tick": tick,
                "population_size": len(self.agents),
                "pressure_index": round(pressure_index, 4),
                "last_underfed_rate": round(self.last_underfed_rate, 4),
                "same_tick_decision_order": "Agents decide sequentially by id; previous same-tick decisions are observable only as summaries.",
                "prior_decisions_in_this_tick": observed_prior,
                "prior_decision_count": len(prior_decisions),
            },
            "agent": {
                "id": agent.id,
                "age_group": agent.age_group.value,
                "age_label": agent.age_group.label,
                "sex": agent.sex,
                "moral_type": agent.moral_type.value,
                "moral_label": agent.moral_type.label,
                "moral_weights": {key: round(value, 4) for key, value in agent.moral_weights.items()},
                "moral_fit_to_scenario": round(moral_fit, 4),
                "physical": round(agent.physical, 4),
                "exploration": round(agent.exploration, 4),
                "coordination": round(agent.coordination, 4),
                "compliance_trait": round(agent.compliance, 4),
                "need": round(agent.need, 4),
                "wellbeing": round(agent.wellbeing, 4),
                "food_reserve": round(agent.food_reserve, 4),
                "fatigue": round(agent.fatigue, 4),
                "status": agent.status.value,
                "group_id": agent.group_id,
                "previous_group_id": agent.previous_group_id,
                "cumulative_food": round(agent.cumulative_food, 4),
            },
            "hard_constraints": [
                "Return JSON object only; no markdown, no prose outside JSON.",
                "All four numeric action fields must be finite numbers in [0, 1].",
                "If status is Extreme_Fatigue, set effort to 0 and prefer high restraint_rate.",
                "Do not optimize only food. Balance material survival, moral consistency, autonomy, fairness, belonging, and fatigue.",
                "Treat this Agent as an independent behavior subject, not as an aggregate statistic.",
            ],
        }
        return (
            "你是《道德幸福 Agent》Multi-Agent 仿真中的单个 Agent 行为决策器。\n"
            "设计原则: Agent 的幸福由物质保障、道德一致性、自主感、社会归属、公平感、压力与疲劳共同构成；"
            "不同道德类型会在自我发展、自我克制、利他发展、利他克制之间形成不同权重。\n"
            "请根据以下 JSON 状态，为当前 Agent 在当前轮次做一次独立行为判断。\n"
            "同一轮中 Agent 按 id 顺序判断，当前 Agent 可以参考此前同轮 Agent 的摘要，但不能改写它们。\n"
            "只返回符合 schema 的紧凑 JSON。\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _parse_llm_decision(raw: str) -> dict[str, Any]:
        text = str(raw).strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"未返回 JSON 对象: {text[:160]}")
        data = json.loads(text[start:end + 1])
        if isinstance(data, dict) and isinstance(data.get("decision"), dict):
            data = data["decision"]
        if not isinstance(data, dict):
            raise ValueError("JSON 顶层必须是对象")
        return data

    @staticmethod
    def _required_unit_float(payload: dict[str, Any], key: str) -> float:
        if key not in payload:
            raise ValueError(f"缺少字段 {key}")
        try:
            value = float(payload[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是数字") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值")
        return clip(value, 0.0, 1.0)

    def _agent_ability(self, agent: AgentState, mode: ActionMode | None = None) -> float:
        mode = mode or self.scenario.mode
        if mode is ActionMode.INDIVIDUAL:
            return 0.5 * agent.physical + 0.5 * agent.exploration
        return 0.5 * agent.physical + 0.5 * agent.coordination

    def _environment_shock(self) -> float:
        spread = self.params.environment_volatility
        return self.rng.uniform(1.0 - spread, 1.0 + spread)

    def _organized_tick(
        self, tick: int, decisions: dict[int, AgentDecision]
    ) -> tuple[dict[int, float], dict[int, dict[str, float]], dict[str, float]]:
        base_food = (
            len(self.agents)
            * self.scenario.base_resource
            * self.params.food_scale
            * self._environment_shock()
        )
        contributions = {
            agent.id: decisions[agent.id].effort * self._agent_ability(agent, ActionMode.ORGANIZED)
            for agent in self.agents
        }
        total_contribution = sum(contributions.values())
        avg_compliance = mean(decisions[agent.id].compliance_action for agent in self.agents)
        team_count = max(1, math.ceil(len(self.agents) / self.params.typical_team_size))
        continuous_food = (
            self.params.alpha
            * total_contribution
            * (1.0 + self.params.lambda_c * self.scenario.collectivization * avg_compliance)
            - self.params.org_cost_per_team * team_count
        )
        team_power = (
            self.params.base_combat_power
            * sum(decisions[a.id].effort * (0.5 + 0.5 * a.physical) for a in self.agents)
            * (1.0 + self.params.lambda_c * avg_compliance)
        )
        hunt = self.hunting.resolve_hunt(
            team_power=team_power,
            rng=self.rng,
            events=team_count,
            encounter_rate=0.85 + 0.15 * self.scenario.base_resource,
        )
        produced_food = max(0.0, continuous_food) + hunt.meat
        total_food = max(0.0, base_food + produced_food)
        allocations = self._need_contribution_allocation(
            total_food=total_food,
            agents=self.agents,
            contributions=contributions,
            equal_weight=self.params.organized_equal_weight,
        )
        social_gini = gini(list(allocations.values()))
        contexts = {}
        avg_need_share = total_food / max(1, len(self.agents))
        for agent in self.agents:
            fairness = 1.0 - min(
                1.0, abs(allocations[agent.id] - avg_need_share) / max(0.1, avg_need_share)
            )
            contexts[agent.id] = {
                "fairness": clip(0.65 * fairness + 0.35 * (1.0 - social_gini), 0.0, 1.0),
                "autonomy": 0.22 + 0.10 * self.scenario.individualization,
                "social_belonging": clip(0.62 + 0.28 * avg_compliance, 0.0, 1.0),
                "contribution_signal": clip(
                    safe_div(contributions[agent.id], mean(contributions.values()), 0.0),
                    0.0,
                    1.0,
                ),
                "hunt_failed": total_food < sum(a.need for a in self.agents) * 0.60,
            }
        return allocations, contexts, {
            "base_food": base_food,
            "produced_food": produced_food,
            "hunt_food": hunt.meat,
            "total_food": total_food,
            "avg_compliance": avg_compliance,
            "hunt_success_rate": hunt.success_rate,
        }

    def _group_tick(
        self, tick: int, decisions: dict[int, AgentDecision]
    ) -> tuple[dict[int, float], dict[int, dict[str, float]], dict[str, float]]:
        allocations: dict[int, float] = {}
        contexts: dict[int, dict[str, float]] = {}
        total_base = 0.0
        total_produced = 0.0
        total_hunt = 0.0
        stability_values: list[float] = []

        for group_id, member_ids in sorted(self.groups.items()):
            members = [self.agents[agent_id] for agent_id in member_ids]
            if not members:
                continue
            n = len(members)
            base_food = (
                n * self.scenario.base_resource * self.params.food_scale * self._environment_shock()
            )
            contributions = {
                agent.id: decisions[agent.id].effort * self._agent_ability(agent, ActionMode.GROUP)
                for agent in members
            }
            total_contribution = sum(contributions.values())
            group_similarity = self._group_moral_similarity(members)
            changed = sum(1 for agent in members if agent.previous_group_id not in (None, group_id))
            stability = clip(1.0 - changed / max(1, n), 0.0, 1.0)
            stability_values.append(stability)
            avg_coordination = mean(agent.coordination for agent in members)
            group_coordination = clip(
                0.40 * avg_coordination + 0.30 * group_similarity + 0.30 * stability,
                0.0,
                1.0,
            )
            continuous_food = (
                self.params.alpha
                * total_contribution
                * (1.0 + self.params.lambda_g * group_coordination)
                - self.params.group_cost_per_team * max(1.0, n / self.params.typical_team_size)
            )
            team_power = (
                self.params.base_combat_power
                * sum(decisions[a.id].effort * (0.5 + 0.5 * a.physical) for a in members)
                * (1.0 + self.params.lambda_g * group_coordination)
            )
            hunt = self.hunting.resolve_hunt(
                team_power=team_power,
                rng=self.rng,
                events=max(1, round(n / self.params.typical_team_size)),
                encounter_rate=0.80 + 0.15 * self.scenario.base_resource,
            )
            group_food = max(0.0, base_food + max(0.0, continuous_food) + hunt.meat)
            total_base += base_food
            total_produced += max(0.0, continuous_food)
            total_hunt += hunt.meat

            avg_alt = mean(agent.alt_orientation for agent in members)
            avg_restraint = mean(agent.restraint_orientation for agent in members)
            share_norm = clip(
                0.20
                + self.params.share_norm_alt_weight * avg_alt
                + self.params.share_norm_restraint_weight * avg_restraint,
                0.0,
                1.0,
            )
            group_allocations = self._need_contribution_allocation(
                total_food=group_food,
                agents=members,
                contributions=contributions,
                equal_weight=share_norm,
            )
            group_gini = gini(list(group_allocations.values()))
            group_avg_food = mean(group_allocations.values())
            for agent in members:
                allocations[agent.id] = group_allocations[agent.id]
                contexts[agent.id] = {
                    "fairness": clip(1.0 - group_gini, 0.0, 1.0),
                    "autonomy": 0.62 + 0.18 * self.scenario.individualization,
                    "social_belonging": clip(0.48 + 0.25 * stability + 0.22 * group_coordination, 0.0, 1.0),
                    "contribution_signal": clip(
                        safe_div(contributions[agent.id], mean(contributions.values()), 0.0),
                        0.0,
                        1.0,
                    ),
                    "hunt_failed": group_food < sum(a.need for a in members) * 0.55,
                }

            self.group_memory[group_id] = {
                "expected_food": group_avg_food,
                "fairness": 1.0 - group_gini,
                "similarity": group_similarity,
                "stability": stability,
                "coordination": group_coordination,
            }
            self.group_records.append(
                {
                    "tick": tick,
                    "group_id": group_id,
                    "size": n,
                    "group_total_food": group_food,
                    "group_gini_index": group_gini,
                    "group_stability": stability,
                    "group_coordination": group_coordination,
                    "share_norm": share_norm,
                    "dominant_moral_type": self._dominant_moral_type(members),
                }
            )

        for agent in self.agents:
            allocations.setdefault(agent.id, 0.0)
            contexts.setdefault(
                agent.id,
                {
                    "fairness": 0.3,
                    "autonomy": 0.7,
                    "social_belonging": 0.2,
                    "contribution_signal": 0.0,
                    "hunt_failed": True,
                },
            )
        total_food = sum(allocations.values())
        return allocations, contexts, {
            "base_food": total_base,
            "produced_food": total_produced + total_hunt,
            "hunt_food": total_hunt,
            "total_food": total_food,
            "group_stability": mean(stability_values, 1.0),
            "hunt_success_rate": 0.0,
        }

    def _ensure_groups_for_tick(self, tick: int) -> None:
        if not self.groups or tick == 1 or (tick - 1) % self.params.regroup_interval == 0:
            self._regroup(tick)

    def _individual_tick(
        self, tick: int, decisions: dict[int, AgentDecision]
    ) -> tuple[dict[int, float], dict[int, dict[str, float]], dict[str, float]]:
        raw_food: dict[int, float] = {}
        contexts: dict[int, dict[str, float]] = {}
        total_base = 0.0
        total_produced = 0.0
        total_hunt = 0.0
        successes = 0
        events = 0

        for agent in self.agents:
            decision = decisions[agent.id]
            base_food = self.scenario.base_resource * self.params.food_scale * self._environment_shock()
            ability = self._agent_ability(agent, ActionMode.INDIVIDUAL)
            risk_mean = (
                1.0
                + 0.25 * agent.development_orientation * self.scenario.opportunity
                - 0.12 * agent.development_orientation * self.scenario.scarcity_pressure
                - 0.08 * agent.restraint_orientation
            )
            risk_sd = (
                0.12
                + 0.20 * agent.development_orientation
                + 0.14 * self.scenario.scarcity_pressure
                - 0.08 * agent.restraint_orientation
            )
            risk_return = max(0.0, self.rng.gauss(risk_mean, max(0.03, risk_sd)))
            continuous_food = self.params.alpha_individual * decision.effort * ability * risk_return
            encounter_rate = clip(0.15 + 0.70 * decision.effort + 0.10 * agent.exploration, 0.0, 0.95)
            team_power = (
                self.params.base_combat_power
                * decision.effort
                * (0.55 + 0.45 * agent.physical)
                * (0.85 + 0.25 * agent.exploration)
            )
            hunt = self.hunting.resolve_hunt(
                team_power=team_power,
                rng=self.rng,
                events=1,
                encounter_rate=encounter_rate,
            )
            events += hunt.events
            successes += hunt.successes
            total_base += base_food
            total_produced += continuous_food
            total_hunt += hunt.meat
            raw_food[agent.id] = max(0.0, base_food + continuous_food + hunt.meat)
            contexts[agent.id] = {
                "fairness": 0.42,
                "autonomy": 0.90 + 0.08 * self.scenario.individualization,
                "social_belonging": 0.24 + 0.10 * agent.alt_orientation,
                "contribution_signal": clip(decision.effort * ability / 0.75, 0.0, 1.0),
                "hunt_failed": decision.effort > 0.05 and hunt.successes == 0 and continuous_food < agent.need * 0.25,
            }

        allocations = dict(raw_food)
        recipients = [
            agent for agent in self.agents
            if raw_food[agent.id] + agent.food_reserve < agent.need
        ]
        total_gap = sum(max(0.0, agent.need - raw_food[agent.id] - agent.food_reserve) for agent in recipients)
        if total_gap > 1e-9:
            for donor in self.agents:
                surplus = max(0.0, raw_food[donor.id] - donor.need)
                if surplus <= 0:
                    continue
                transfer_pool = (
                    self.params.individual_transfer_strength
                    * decisions[donor.id].share_rate
                    * surplus
                )
                transfer_pool = min(transfer_pool, allocations[donor.id])
                if transfer_pool <= 0:
                    continue
                allocations[donor.id] -= transfer_pool
                contexts[donor.id]["social_belonging"] = clip(
                    contexts[donor.id]["social_belonging"] + 0.12,
                    0.0,
                    1.0,
                )
                for recipient in recipients:
                    gap = max(0.0, recipient.need - raw_food[recipient.id] - recipient.food_reserve)
                    if gap <= 0:
                        continue
                    received = transfer_pool * gap / total_gap
                    allocations[recipient.id] += received
                    contexts[recipient.id]["social_belonging"] = clip(
                        contexts[recipient.id]["social_belonging"] + 0.16,
                        0.0,
                        1.0,
                    )
                    contexts[recipient.id]["fairness"] = clip(
                        contexts[recipient.id]["fairness"] + 0.12,
                        0.0,
                        1.0,
                    )

        social_gini = gini(list(allocations.values()))
        for context in contexts.values():
            context["fairness"] = clip(0.50 * context["fairness"] + 0.50 * (1.0 - social_gini), 0.0, 1.0)
        return allocations, contexts, {
            "base_food": total_base,
            "produced_food": total_produced + total_hunt,
            "hunt_food": total_hunt,
            "total_food": sum(allocations.values()),
            "hunt_success_rate": successes / max(1, events),
        }

    def _need_contribution_allocation(
        self,
        *,
        total_food: float,
        agents: list[AgentState],
        contributions: dict[int, float],
        equal_weight: float,
    ) -> dict[int, float]:
        total_food = max(0.0, total_food)
        equal_weight = clip(equal_weight, 0.0, 1.0)
        total_need = sum(agent.need for agent in agents)
        total_contribution = sum(max(0.0, value) for value in contributions.values())
        allocations: dict[int, float] = {}
        for agent in agents:
            need_share = total_food * safe_div(agent.need, total_need, 1.0 / max(1, len(agents)))
            contribution_share = total_food * safe_div(
                max(0.0, contributions.get(agent.id, 0.0)),
                total_contribution,
                1.0 / max(1, len(agents)),
            )
            allocations[agent.id] = max(
                0.0,
                equal_weight * need_share + (1.0 - equal_weight) * contribution_share,
            )
        return allocations

    def _regroup(self, tick: int) -> None:
        group_count = max(1, math.ceil(len(self.agents) / self.params.target_group_size))
        if not self.groups:
            ordered = sorted(
                self.agents,
                key=lambda agent: (agent.moral_type.value, self.rng.random()),
            )
            self.groups = {group_id: [] for group_id in range(group_count)}
            for idx, agent in enumerate(ordered):
                group_id = idx % group_count
                self.groups[group_id].append(agent.id)
                agent.previous_group_id = agent.group_id
                agent.group_id = group_id
            return

        for agent in self.agents:
            agent.previous_group_id = agent.group_id

        new_groups = {group_id: [] for group_id in range(group_count)}
        memories = self._normalized_group_memory(group_count)
        agents = list(self.agents)
        self.rng.shuffle(agents)
        for agent in agents:
            weights = GROUP_CHOICE_WEIGHTS[agent.moral_type]
            best_group = None
            best_score = -1.0
            for group_id in range(group_count):
                if len(new_groups[group_id]) >= self.params.max_group_size:
                    continue
                memory = memories[group_id]
                similarity = self._similarity_to_group_memory(agent, memory)
                score = (
                    weights["expected_food"] * memory["expected_food"]
                    + weights["fairness"] * memory["fairness"]
                    + weights["similarity"] * similarity
                    + weights["stability"] * memory["stability"]
                    + self.rng.uniform(-0.02, 0.02)
                )
                if score > best_score:
                    best_score = score
                    best_group = group_id
            if best_group is None:
                best_group = min(new_groups, key=lambda gid: len(new_groups[gid]))
            new_groups[best_group].append(agent.id)
            agent.group_id = best_group

        self.groups = self._repair_group_sizes(new_groups)

    def _normalized_group_memory(self, group_count: int) -> dict[int, dict[str, float]]:
        expected_values = [
            self.group_memory.get(group_id, {}).get("expected_food", self.params.need_base)
            for group_id in range(group_count)
        ]
        max_expected = max(expected_values) if expected_values else self.params.need_base
        max_expected = max(0.1, max_expected)
        memories: dict[int, dict[str, float]] = {}
        for group_id in range(group_count):
            raw = self.group_memory.get(group_id, {})
            memories[group_id] = {
                "expected_food": clip(raw.get("expected_food", self.params.need_base) / max_expected, 0.0, 1.0),
                "fairness": clip(raw.get("fairness", 0.65), 0.0, 1.0),
                "similarity": clip(raw.get("similarity", 0.55), 0.0, 1.0),
                "stability": clip(raw.get("stability", 0.80), 0.0, 1.0),
            }
        return memories

    def _repair_group_sizes(self, groups: dict[int, list[int]]) -> dict[int, list[int]]:
        small_groups = [
            group_id for group_id, members in groups.items()
            if 0 < len(members) < self.params.min_group_size
        ]
        for group_id in small_groups:
            members = groups[group_id][:]
            groups[group_id] = []
            for agent_id in members:
                target = min(
                    (gid for gid in groups if gid != group_id),
                    key=lambda gid: len(groups[gid]),
                )
                groups[target].append(agent_id)
                self.agents[agent_id].group_id = target
        return {gid: members for gid, members in groups.items() if members}

    def _similarity_to_group_memory(self, agent: AgentState, memory: dict[str, float]) -> float:
        # Memory stores only scalar similarity, so preserve group identity through a mild same-group prior.
        same_group_prior = 0.08 if agent.group_id is not None else 0.0
        return clip(memory.get("similarity", 0.55) + same_group_prior, 0.0, 1.0)

    def _group_moral_similarity(self, members: list[AgentState]) -> float:
        if len(members) < 2:
            return 1.0
        centroid = {
            key: mean(agent.moral_weights[key] for agent in members)
            for key in MORAL_COMPONENTS
        }
        distances = []
        for agent in members:
            distances.append(sum(abs(agent.moral_weights[key] - centroid[key]) for key in MORAL_COMPONENTS) / 2.0)
        return clip(1.0 - mean(distances), 0.0, 1.0)

    def _dominant_moral_type(self, members: list[AgentState]) -> str:
        counts: dict[MoralType, int] = {}
        for agent in members:
            counts[agent.moral_type] = counts.get(agent.moral_type, 0) + 1
        if not counts:
            return ""
        return max(counts, key=counts.get).value

    def _agent_log_record(
        self,
        *,
        tick: int,
        agent: AgentState,
        decision: AgentDecision,
        context: dict[str, float],
        allocated: float,
        effective_need: float,
        reserve_before: float,
        fatigue_before: float,
        wellbeing_before: float,
        status_before: Status,
        starved: bool,
        scores: dict[str, float],
    ) -> dict[str, Any]:
        action = self._action_label(agent, decision, status_before)
        if starved:
            result = "触发极度疲劳救济"
        elif context.get("hunt_failed", False):
            result = "狩猎受挫并完成结算"
        else:
            result = "完成狩猎并完成结算"
        group_text = "" if agent.group_id is None else f", 群体 {agent.group_id}"
        driver_text = (
            ""
            if decision.decision_source == "rule"
            else f" 模型判断: {decision.decision_reason or decision.decision_source};"
        )
        message = (
            f"{self.scenario.id} 第 {tick} 轮 Agent {agent.id}({agent.moral_type.label})"
            f"{group_text}: {action}; 努力 {decision.effort:.2f}, "
            f"分得肉量 {allocated:.2f}/{effective_need:.2f}, "
            f"储备 {reserve_before:.2f}->{agent.food_reserve:.2f}, "
            f"疲劳 {fatigue_before:.2f}->{agent.fatigue:.2f}, "
            f"幸福 {wellbeing_before:.1f}->{agent.wellbeing:.1f};{driver_text} {result}。"
        )
        return {
            "tick": tick,
            "scenario_id": self.scenario.id,
            "scenario_name": self.scenario.name,
            "agent_id": agent.id,
            "age_group": agent.age_group.value,
            "age_label": agent.age_group.label,
            "sex": agent.sex,
            "moral_type": agent.moral_type.value,
            "moral_label": agent.moral_type.label,
            "group_id": agent.group_id,
            "action_mode": self.scenario.mode.value,
            "action": action,
            "effort": round(decision.effort, 4),
            "compliance": round(decision.compliance_action, 4),
            "share_rate": round(decision.share_rate, 4),
            "restraint_rate": round(decision.restraint_rate, 4),
            "moral_fit": round(decision.moral_fit, 4),
            "decision_source": decision.decision_source,
            "decision_reason": decision.decision_reason,
            "allocated_food": round(allocated, 4),
            "effective_need": round(effective_need, 4),
            "food_reserve_before": round(reserve_before, 4),
            "food_reserve_after": round(agent.food_reserve, 4),
            "fatigue_before": round(fatigue_before, 4),
            "fatigue_after": round(agent.fatigue, 4),
            "status_before": status_before.value,
            "status_after": agent.status.value,
            "wellbeing_before": round(wellbeing_before, 4),
            "wellbeing_after": round(agent.wellbeing, 4),
            "fairness": round(float(context.get("fairness", 0.0)), 4),
            "autonomy": round(float(context.get("autonomy", 0.0)), 4),
            "social_belonging": round(float(context.get("social_belonging", 0.0)), 4),
            "hunt_failed": bool(context.get("hunt_failed", False)),
            "starved": starved,
            "result": result,
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "message": message,
        }

    def _agent_decision_log_record(
        self,
        *,
        tick: int,
        agent: AgentState,
        decision: AgentDecision,
    ) -> dict[str, Any]:
        action = self._action_label(agent, decision, agent.status)
        message = (
            f"{self.scenario.id} 第 {tick} 轮 Agent {agent.id}({agent.moral_type.label})"
            f" 完成模型行为判断: {action}; 努力 {decision.effort:.2f}, "
            f"服从 {decision.compliance_action:.2f}, 分享 {decision.share_rate:.2f}, "
            f"节制 {decision.restraint_rate:.2f}; {decision.decision_reason or '无额外理由'}。"
        )
        return {
            "tick": tick,
            "scenario_id": self.scenario.id,
            "scenario_name": self.scenario.name,
            "level": "info",
            "event": "agent_decision",
            "agent_id": agent.id,
            "age_group": agent.age_group.value,
            "age_label": agent.age_group.label,
            "sex": agent.sex,
            "moral_type": agent.moral_type.value,
            "moral_label": agent.moral_type.label,
            "group_id": agent.group_id,
            "action_mode": self.scenario.mode.value,
            "action": action,
            "effort": round(decision.effort, 4),
            "compliance": round(decision.compliance_action, 4),
            "share_rate": round(decision.share_rate, 4),
            "restraint_rate": round(decision.restraint_rate, 4),
            "moral_fit": round(decision.moral_fit, 4),
            "decision_source": decision.decision_source,
            "decision_reason": decision.decision_reason,
            "status_before": agent.status.value,
            "status_after": agent.status.value,
            "result": "已完成行为选择",
            "message": message,
        }

    def _action_label(self, agent: AgentState, decision: AgentDecision, status_before: Status) -> str:
        if status_before is Status.EXTREME_FATIGUE or decision.effort <= 1e-9:
            return "休息恢复/等待救济"
        if self.scenario.mode is ActionMode.ORGANIZED:
            return "服从强组织分工并参与集体狩猎"
        if self.scenario.mode is ActionMode.GROUP:
            group_id = "未分组" if agent.group_id is None else str(agent.group_id)
            return f"加入群体 {group_id} 进行抱团狩猎"
        return "独自探索并个人狩猎"

    def _tick_record(
        self,
        tick: int,
        production: dict[str, float],
        allocations: dict[int, float],
        survival_rate: float,
    ) -> dict[str, Any]:
        by_moral = {}
        for moral_type in MoralType:
            members = [agent for agent in self.agents if agent.moral_type is moral_type]
            by_moral[moral_type.value] = {
                "label": moral_type.label,
                "avg_food_allocated": mean(allocations.get(agent.id, 0.0) for agent in members),
                "avg_wellbeing": mean(agent.wellbeing for agent in members),
                "avg_fatigue": mean(agent.fatigue for agent in members),
                "moral_fit": mean(
                    sum(
                        agent.moral_weights[key] * self.scenario.moral_fit_weights[key]
                        for key in MORAL_COMPONENTS
                    )
                    for agent in members
                ),
            }
        record = {
            "tick": tick,
            "avg_food_allocated": mean(allocations.values()),
            "avg_wellbeing": mean(agent.wellbeing for agent in self.agents),
            "social_gini_index": gini(list(allocations.values())),
            "minimum_survival_rate": survival_rate,
            "base_food": production["base_food"],
            "produced_food": production["produced_food"],
            "hunt_food": production.get("hunt_food", 0.0),
            "total_food": production["total_food"],
            "by_moral_type": by_moral,
        }
        for key, value in production.items():
            if key not in record:
                record[key] = value
        return record

    def _result(self) -> dict[str, Any]:
        society = summarize_society(
            agents=self.agents,
            total_ticks=self.params.ticks,
            total_food_by_tick=self.total_food_by_tick,
            survival_rates=self.survival_rates,
        )
        result = {
            "scenario": self.scenario.to_dict(),
            "parameters": self.params.to_dict(),
            "model_status": self.model_status,
            "society_metrics": society,
            "moral_type_metrics": summarize_by_moral_type(self.agents, self.params.ticks),
            "group_metrics": aggregate_group_metrics(self.group_records),
            "time_series": self.time_series,
            "system_logs": self.system_logs,
            "agent_logs": self.agent_logs,
            "final_agents": [agent.to_snapshot() for agent in self.agents],
            "group_records": self.group_records[-200:],
        }
        self._assert_numeric_result(result)
        return result

    def _assert_numeric_result(self, value: Any, path: str = "result") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                self._assert_numeric_result(item, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                self._assert_numeric_result(item, f"{path}[{idx}]")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite numeric value at {path}: {value}")


def run_batch(
    scenario_ids: list[str] | None = None,
    parameters: SimulationParameters | dict[str, object] | None = None,
    event_callback: StreamCallback | None = None,
    stop_checker: StopChecker | None = None,
) -> dict[str, Any]:
    params = (
        parameters
        if isinstance(parameters, SimulationParameters)
        else SimulationParameters.from_dict(parameters)
    )
    ids = scenario_ids or list(SCENARIOS)
    if event_callback is not None:
        event_callback(
            {
                "type": "batch_start",
                "parameters": params.to_dict(),
                "scenarios": [SCENARIOS[scenario_id].to_dict() for scenario_id in ids],
                "model_status": provider_status(params.llm_provider).to_dict(),
            }
        )
    results = []
    for scenario_id in ids:
        if stop_checker is not None and stop_checker():
            raise RuntimeError("simulation stopped by user")
        if event_callback is not None:
            event_callback({"type": "scenario_start", "scenario": SCENARIOS[scenario_id].to_dict()})
        simulation = Simulation(
            scenario_id,
            params,
            event_callback=event_callback,
            stop_checker=stop_checker,
        )
        scenario_result = simulation.run()
        results.append(scenario_result)
        if event_callback is not None:
            event_callback({"type": "scenario_result", "result": scenario_result})
    result = {
        "parameters": params.to_dict(),
        "scenarios": [SCENARIOS[scenario_id].to_dict() for scenario_id in ids],
        "results": results,
        "comparison": _comparison(results),
        "model_status": provider_status(params.llm_provider).to_dict(),
    }
    if event_callback is not None:
        event_callback({"type": "complete", "result": result})
    return result


def _comparison(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        scenario = result["scenario"]
        society = result["society_metrics"]
        rows.append(
            {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "mode": scenario["mode"],
                "avg_food_allocated": society["avg_food_allocated"],
                "avg_wellbeing": society["avg_wellbeing"],
                "social_gini_index": society["social_gini_index"],
                "minimum_survival_rate": society["minimum_survival_rate"],
                "total_social_food": society["total_social_food"],
            }
        )
    return rows
