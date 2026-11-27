from __future__ import annotations

import math
import json
import random
import time
from typing import Any, Callable

from .config import SCENARIOS, SimulationParameters
from .hunting import HuntingEngine
from .llm import LLMProviderConfig, build_llm_provider, provider_status
from .metrics import (
    aggregate_group_metrics,
    summarize_by_moral_type,
    summarize_role_metrics,
    summarize_society,
    summarize_trade_metrics,
)
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
    Role,
    Scenario,
    Status,
    WellbeingScores,
)
from .settings import llm_provider_settings
from .utils import clip, gini, mean, safe_div, stdev
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
        collect_agent_logs: bool = True,
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
        self.scenario_seed = self.params.seed + scenario_offset
        # Pair starting populations and exogenous streams; keep endogenous grouping scenario-seeded.
        self.init_rng = random.Random(self.params.seed)
        self.environment_rng = random.Random(self.params.seed + 0xE001)
        self.hunting_rng = random.Random(self.params.seed + 0xB007)
        self.individual_rng = random.Random(self.params.seed + 0x1D1D)
        self.group_rng = random.Random(self.scenario_seed + 0x6A0B)
        self.hunting = HuntingEngine(meat_scale=self.params.prey_meat_scale)
        self.agents = self._initialize_agents()
        self.initial_shell_total = sum(agent.shell_balance for agent in self.agents)
        self.promotion_tiebreakers = {
            agent.id: self.init_rng.random() for agent in self.agents
        }
        self.groups: dict[int, list[int]] = {}
        self.chief_id: int | None = None
        self.chieftain_ids: list[int] = []
        self.hunter_assignments: dict[int, list[int]] = {}
        self._initialize_hierarchy()
        self.group_memory: dict[int, dict[str, float]] = {}
        self.group_records: list[dict[str, Any]] = []
        self.time_series: list[dict[str, Any]] = []
        self.agent_logs: list[dict[str, Any]] = []
        self.system_logs: list[dict[str, Any]] = []
        self.trade_records: list[dict[str, Any]] = []
        self.wellbeing_evaluations: list[dict[str, Any]] = []
        self.promotion_events: list[dict[str, Any]] = []
        self.promotion_reviews: list[dict[str, Any]] = []
        self.event_callback = event_callback
        self.stop_checker = stop_checker
        self.collect_agent_logs = collect_agent_logs
        self._log_sequence = 0
        self.total_food_by_tick: list[float] = []
        self.food_consumed_by_tick: list[float] = []
        self.unmet_food_by_tick: list[float] = []
        self.external_relief_by_tick: list[float] = []
        self.public_relief_by_tick: list[float] = []
        self.food_loss_by_tick: list[float] = []
        self.food_ledger_records: list[dict[str, float | int]] = []
        self.survival_rates: list[float] = []
        self.shell_totals_by_tick: list[float] = []
        self.initial_public_food_reserve = self.params.public_food_reserve
        self.public_food_reserve = self.params.public_food_reserve
        self.external_relief_total = 0.0
        self.public_relief_total = 0.0
        self.total_food_consumed = 0.0
        self.total_food_loss = 0.0
        self.last_underfed_rate = self.scenario.scarcity_pressure
        self.model_status = provider_status(self.params.llm_provider).to_dict()
        self.llm_provider = None
        self.llm_config: LLMProviderConfig | None = None
        self.llm_decision_retries = 0
        if self.params.llm_provider != "rule":
            provider_settings = llm_provider_settings(self.params.llm_provider)
            self.llm_decision_retries = max(0, int(provider_settings.get("decision_retries", 2)))
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
        if not self.collect_agent_logs:
            return
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
        self.init_rng.shuffle(moral_types)
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
                0.35 + 0.35 * coordination + 0.15 * restraint + 0.10 * alt + self.init_rng.gauss(0, 0.05),
                0.0,
                1.0,
            )
            need = self.params.need_base * profile["need_multiplier"]
            wellbeing = clip(self.init_rng.gauss(60.0, 10.0), 35.0, 82.0)
            agents.append(
                AgentState(
                    id=agent_id,
                    age_group=age_group,
                    sex="F" if self.init_rng.random() < 0.5 else "M",
                    moral_type=moral_type,
                    moral_weights=moral_weights,
                    physical=physical,
                    exploration=exploration,
                    coordination=coordination,
                    compliance=compliance,
                    need=need,
                    wellbeing=wellbeing,
                    shell_balance=(
                        self.init_rng.uniform(
                            self.params.initial_shells_min,
                            self.params.initial_shells_max,
                        )
                        if self.params.initial_shells_distribution == "uniform"
                        else self.params.initial_shells
                    ),
                )
            )
        return agents

    def _initialize_hierarchy(self) -> None:
        """Create the initial fixed-capacity hierarchy before the first tick."""
        if not self.params.rank_enabled or not self.agents:
            return
        buckets: dict[tuple[MoralType, AgeGroup], list[AgentState]] = {}
        for agent in self.agents:
            buckets.setdefault((agent.moral_type, agent.age_group), []).append(agent)
        for bucket in buckets.values():
            self.init_rng.shuffle(bucket)
        selected_counts = {key: 0 for key in buckets}
        stratum_sizes = {key: len(bucket) for key, bucket in buckets.items()}

        def select_balanced() -> AgentState:
            min_share = min(
                selected_counts[key] / stratum_sizes[key]
                for key, bucket in buckets.items()
                if bucket
            )
            eligible = [
                key for key, bucket in buckets.items()
                if bucket
                and abs(selected_counts[key] / stratum_sizes[key] - min_share) < 1e-12
            ]
            key = self.init_rng.choice(eligible)
            selected_counts[key] += 1
            return buckets[key].pop()

        self.chief_id = select_balanced().id
        chieftain_count = min(self.params.chieftain_slots, max(1, len(self.agents) - 1))
        self.chieftain_ids = [select_balanced().id for _ in range(chieftain_count)]
        self._rebuild_rank_groups(initial=True)

    def _rebuild_rank_groups(self, *, initial: bool = False) -> None:
        if not self.params.rank_enabled or self.chief_id is None:
            return
        old_groups = self.groups if hasattr(self, "groups") else {}
        for agent in self.agents:
            if not initial:
                agent.previous_group_id = agent.group_id
            agent.group_id = None

        chief = self.agents[self.chief_id]
        chief.role = Role.CHIEF
        chief.superior_id = None
        hunter_ids = [agent.id for agent in self.agents if agent.id != self.chief_id]
        chieftain_id_set = set(self.chieftain_ids)
        for agent in self.agents:
            if agent.id in chieftain_id_set:
                agent.role = Role.CHIEFTAIN
            elif agent.id != self.chief_id:
                agent.role = Role.HUNTER
                agent.superior_id = None

        hunters = [agent_id for agent_id in hunter_ids if agent_id not in chieftain_id_set]
        group_count = max(1, len(self.chieftain_ids))
        # Preserve as much of the old attachment as possible, then balance quotas.
        old_order = {
            agent_id: index
            for index, group_id in enumerate(sorted(old_groups))
            for agent_id in old_groups[group_id]
        }
        hunters.sort(key=lambda agent_id: (old_order.get(agent_id, 10_000), agent_id))
        base_quota, extra = divmod(len(hunters), group_count)
        assignments: dict[int, list[int]] = {}
        cursor = 0
        for index, chieftain_id in enumerate(self.chieftain_ids):
            quota = base_quota + (1 if index < extra else 0)
            assignments[chieftain_id] = hunters[cursor : cursor + quota]
            cursor += quota
        self.hunter_assignments = assignments
        self.groups = {}
        for chieftain_id, assigned_hunters in assignments.items():
            members = [chieftain_id, *assigned_hunters]
            self.groups[chieftain_id] = members
            chieftain = self.agents[chieftain_id]
            chieftain.role = Role.CHIEFTAIN
            chieftain.superior_id = self.chief_id
            chieftain.group_id = chieftain_id
            for hunter_id in assigned_hunters:
                hunter = self.agents[hunter_id]
                hunter.role = Role.HUNTER
                hunter.superior_id = chieftain_id
                hunter.group_id = chieftain_id
        chief.group_id = None

    def _sample_age_group(self) -> AgeGroup:
        roll = self.init_rng.random()
        if roll < 0.35:
            return AgeGroup.YOUNG
        if roll < 0.80:
            return AgeGroup.MIDDLE
        return AgeGroup.OLD

    def _ability(self, center: float) -> float:
        return clip(self.init_rng.gauss(center, 0.08), 0.15, 0.95)

    def _run_tick(self, tick: int) -> None:
        self._raise_if_stopped()
        food_stock_start = self.public_food_reserve + sum(agent.food_reserve for agent in self.agents)
        external_relief_before = self.external_relief_total
        public_relief_before = self.public_relief_total
        relief_received_by_agent = self._recover_rescued_agents()
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
                    "id": agent.id,
                    "m": agent.moral_type.value,
                    "e": round(decision.effort, 3),
                    "s": round(decision.share_rate, 3),
                    "r": round(decision.restraint_rate, 3),
                }
            )
        if self.scenario.mode is ActionMode.ORGANIZED:
            allocations, contexts, production = self._organized_tick(tick, decisions)
        elif self.scenario.mode is ActionMode.GROUP:
            allocations, contexts, production = self._group_tick(tick, decisions)
        else:
            allocations, contexts, production = self._individual_tick(tick, decisions)

        effective_needs = {
            agent.id: max(
                0.65 * agent.need,
                agent.need * (1.0 - self.params.restraint_need_discount * decisions[agent.id].restraint_rate),
            )
            for agent in self.agents
        }
        if self.params.trade_enabled:
            trade = self._run_market(tick, allocations, effective_needs, contexts)
        else:
            trade = self._empty_trade_record(tick)

        for agent in self.agents:
            agent.hunting_contribution += float(contexts.get(agent.id, {}).get("hunting_contribution", 0.0))
            agent.period_hunting_contribution += float(
                contexts.get(agent.id, {}).get("hunting_contribution", 0.0)
            )

        starved_count = 0
        tick_consumed = 0.0
        tick_unmet = 0.0
        tick_food_loss = 0.0
        for agent in self.agents:
            self._raise_if_stopped()
            decision = decisions[agent.id]
            allocated = max(0.0, allocations.get(agent.id, 0.0))
            context = contexts.get(agent.id, {})
            reserve_before = agent.food_reserve
            fatigue_before = agent.fatigue
            wellbeing_before = agent.wellbeing
            status_before = agent.status
            effective_need = effective_needs[agent.id]
            available = agent.food_reserve + allocated
            food_consumed = min(available, effective_need)
            unmet_food = max(0.0, effective_need - food_consumed)
            starved = unmet_food > 1e-9
            surplus = max(0.0, available - food_consumed)
            reserve_cap = agent.need * self.params.reserve_cap_days
            food_loss = max(0.0, surplus - reserve_cap)
            agent.food_reserve = min(surplus, reserve_cap)

            if starved:
                starved_count += 1
                agent.extreme_triggers += 1
                agent.status = Status.EXTREME_FATIGUE
                agent.pending_relief = agent.need if self.params.relief_source != "none" else 0.0
                agent.fatigue = max(0.0, agent.fatigue - 0.1)
            else:
                agent.status = Status.NORMAL
                agent.pending_relief = 0.0

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
                food_consumed=food_consumed,
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
            agent.cumulative_food_consumed += food_consumed
            agent.cumulative_unmet_food += unmet_food
            agent.cumulative_relief_received += relief_received_by_agent.get(agent.id, 0.0)
            agent.allocated_history.append(allocated)
            agent.food_consumed_history.append(food_consumed)
            agent.unmet_food_history.append(unmet_food)
            agent.relief_received_history.append(relief_received_by_agent.get(agent.id, 0.0))
            agent.extreme_trigger_history.append(1 if starved else 0)
            agent.wellbeing_history.append(agent.wellbeing)
            agent.moral_consistency_history.append(scores.moral_consistency)
            self._adapt_moral_weights(
                agent=agent,
                decision=decision,
                scores=scores,
                context=context,
                starved=starved,
            )
            self._append_agent_log(
                self._agent_log_record(
                    tick=tick,
                    agent=agent,
                    decision=decision,
                    context=context,
                    allocated=allocated,
                    food_consumed=food_consumed,
                    unmet_food=unmet_food,
                    relief_received=relief_received_by_agent.get(agent.id, 0.0),
                    effective_need=effective_need,
                    reserve_before=reserve_before,
                    fatigue_before=fatigue_before,
                    wellbeing_before=wellbeing_before,
                    status_before=status_before,
                    starved=starved,
                    scores={
                        "eval_score": scores.eval_score,
                        "peer_score": scores.peer_score,
                        "self_score": scores.self_score,
                        "material_score": scores.material_score,
                        "moral_consistency": scores.moral_consistency,
                        "perceived_fairness": scores.perceived_fairness,
                        "autonomy": scores.autonomy,
                        "social_belonging": scores.social_belonging,
                        "stress": scores.stress,
                        "eval_reason": scores.eval_reason,
                        "peer_reason": scores.peer_reason,
                        "self_reason": scores.self_reason,
                    },
                )
            )
            tick_consumed += food_consumed
            tick_unmet += unmet_food
            tick_food_loss += food_loss

        self._evaluate_additional_wellbeing(tick, contexts, effective_needs)
        survival_rate = 1.0 - starved_count / max(1, len(self.agents))
        self.last_underfed_rate = 1.0 - survival_rate
        self.survival_rates.append(survival_rate)
        self.total_food_by_tick.append(production["total_food"])
        self.food_consumed_by_tick.append(tick_consumed)
        self.unmet_food_by_tick.append(tick_unmet)
        external_relief = self.external_relief_total - external_relief_before
        public_relief = self.public_relief_total - public_relief_before
        self.external_relief_by_tick.append(external_relief)
        self.public_relief_by_tick.append(public_relief)
        self.food_loss_by_tick.append(tick_food_loss)
        self.total_food_consumed += tick_consumed
        self.total_food_loss += tick_food_loss
        food_stock_end = self.public_food_reserve + sum(agent.food_reserve for agent in self.agents)
        food_ledger_error = (
            food_stock_start
            + production["total_food"]
            + external_relief
            - tick_consumed
            - tick_food_loss
            - food_stock_end
        )
        self.food_ledger_records.append(
            {
                "tick": tick,
                "opening_stock": food_stock_start,
                "production": production["total_food"],
                "external_relief": external_relief,
                "consumption": tick_consumed,
                "loss": tick_food_loss,
                "closing_stock": food_stock_end,
                "ledger_error": food_ledger_error,
            }
        )
        self.shell_totals_by_tick.append(sum(agent.shell_balance for agent in self.agents))
        self.time_series.append(
            self._tick_record(
                tick,
                production,
                allocations,
                survival_rate,
                trade,
                food_opening_stock=food_stock_start,
                food_closing_stock=food_stock_end,
                consumed=tick_consumed,
                unmet=tick_unmet,
                external_relief=external_relief,
                public_relief=public_relief,
                food_loss=tick_food_loss,
                food_ledger_error=food_ledger_error,
            )
        )
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
                "trade_food": trade["food_volume"],
                "trade_shells": trade["shell_volume"],
                "market_price": trade["clearing_price"],
                "external_relief": external_relief,
                "public_relief": public_relief,
                "food_ledger_error": food_ledger_error,
            }
        )
        if (
            self.params.rank_enabled
            and self.params.promotion_enabled
            and self._promotion_due(tick)
        ):
            self._evaluate_promotions(tick)

    def _evaluate_additional_wellbeing(self, tick: int, contexts: dict, needs: dict) -> None:
        """Fixed, scenario-independent scoring scales; peer scores use observable team activity."""
        role_scores = {Role.CHIEF: 100.0, Role.CHIEFTAIN: 75.0, Role.HUNTER: 50.0}
        for agent in self.agents:
            ratio = agent.food_consumed_history[-1] / max(needs[agent.id], 1e-9)
            food = 100.0 * clip(ratio, 0.0, 1.0) * (1.0 - 0.25 * clip(agent.fatigue / 2.0, 0.0, 1.0))
            if agent.status is Status.EXTREME_FATIGUE:
                food *= 0.5
            contribution = max(0.0, float(contexts.get(agent.id, {}).get("hunting_contribution", 0.0)))
            production = 100.0 * min(1.0, contribution / max(agent.need, 1e-9))
            role = role_scores[agent.role] if self.params.rank_enabled else 50.0
            wealth = 100.0 * min(1.0, math.log1p(agent.shell_balance) / math.log1p(40.0))
            agent.objective_wellbeing = 0.4 * food + 0.2 * production + 0.2 * role + 0.2 * wealth

            if self.scenario.mode is ActionMode.ORGANIZED:
                observers = [other for other in self.agents if other.id != agent.id]
            elif self.scenario.mode is ActionMode.GROUP and agent.group_id is not None:
                observers = [other for other in self.agents if other.id != agent.id and other.group_id == agent.group_id]
            else:
                observers = []
            team_ids = {other.id for other in observers}
            partners = contexts.get(agent.id, {}).get("trade_partners", set())
            observers.extend(other for other in self.agents if other.id in partners and other.id not in team_ids)
            # At most three rotating team observers keep long runs tractable and reproducible.
            observers.sort(key=lambda other: other.id)
            if observers:
                start = (tick + agent.id) % len(observers)
                observers = (observers[start:] + observers[:start])[:3]
            ratings = []
            for observer in observers:
                # Role and contribution are visible to a hunting teammate; private food and wealth are not assumed visible.
                same_team = self.scenario.mode is ActionMode.ORGANIZED or (self.scenario.mode is ActionMode.GROUP and agent.group_id is not None and agent.group_id == observer.group_id)
                rating = 0.5 * role + 0.5 * production if same_team else role
                ratings.append(rating)
                observed = {"role": agent.role.value if self.params.rank_enabled else None}
                if same_team:
                    observed["hunting_contribution"] = contribution
                self.wellbeing_evaluations.append({"tick": tick, "observer_id": observer.id, "agent_id": agent.id, "score": rating, "observed": observed})
            agent.peer_perceived_wellbeing = mean(ratings) if ratings else None
            agent.wellbeing_perception_gap = agent.peer_perceived_wellbeing - agent.objective_wellbeing if ratings else None
            agent.objective_wellbeing_history.append(agent.objective_wellbeing)
            agent.peer_perceived_wellbeing_history.append(agent.peer_perceived_wellbeing)

    def _promotion_due(self, tick: int) -> bool:
        """Schedule the first review at burn-in completion, then every full period."""
        first_review_tick = self.params.burn_in_ticks or self.params.promotion_period
        return tick >= first_review_tick and (
            (tick - first_review_tick) % self.params.promotion_period == 0
        )

    def _empty_trade_record(self, tick: int) -> dict[str, float | int | None]:
        return {
            "tick": tick,
            "food_volume": 0.0,
            "food_volume_meals": 0.0,
            "shell_volume": 0.0,
            "clearing_price": None,
            "normalized_price": None,
            "price_anchor": 1.0,
            "price_rule": self.params.market_price_rule,
            "buyer_budget_fraction": self.params.buyer_budget_fraction,
            "buyers": 0,
            "sellers": 0,
            "transactions": 0,
            "participant_ids": [],
            "unfunded_demand": 0.0,
            "total_demand_meals": 0.0,
        }

    def _run_market(
        self,
        tick: int,
        allocations: dict[int, float],
        effective_needs: dict[int, float],
        contexts: dict[int, dict[str, float]],
    ) -> dict[str, float | int | None]:
        """Clear one centralized food--shell market without creating either asset."""
        target = {
            agent.id: effective_needs[agent.id] * self.params.trade_safety_buffer
            for agent in self.agents
        }
        sellers = [
            {
                "agent": agent,
                "food": max(
                    0.0,
                    agent.food_reserve + allocations.get(agent.id, 0.0) - target[agent.id],
                ),
            }
            for agent in self.agents
        ]
        sellers = [item for item in sellers if item["food"] > 1e-9]
        buyers = [
            {
                "agent": agent,
                "food": max(
                    0.0,
                    target[agent.id] - agent.food_reserve - allocations.get(agent.id, 0.0),
                ),
            }
            for agent in self.agents
        ]
        potential_buyers = [item for item in buyers if item["food"] > 1e-9]
        supply = sum(float(item["food"]) for item in sellers)
        demand = sum(float(item["food"]) for item in potential_buyers)
        if supply > 1e-9 and demand > 1e-9:
            clearing_price = (
                self.params.shell_base_price
                if self.params.market_price_rule == "fixed"
                else clip(demand / supply, self.params.shell_min_price, self.params.shell_max_price)
            )
        else:
            clearing_price = self.params.shell_base_price
        for item in potential_buyers:
            budget = item["agent"].shell_balance * self.params.buyer_budget_fraction
            item["budget"] = budget
            item["effective_food"] = min(float(item["food"]), budget / max(1e-12, clearing_price))
        effective_demand = sum(float(item["effective_food"]) for item in potential_buyers)
        buyers = [item for item in potential_buyers if float(item["effective_food"]) > 1e-9]
        unfunded_demand = max(0.0, demand - effective_demand)
        if supply <= 1e-9 or effective_demand <= 1e-9:
            record = self._empty_trade_record(tick)
            record.update(
                {
                    "buyers": len(buyers),
                    "sellers": len(sellers),
                    "unfunded_demand": unfunded_demand,
                    "total_demand_meals": demand / max(1e-12, self.params.need_base),
                }
            )
            self.trade_records.append(record)
            return record

        traded_food = min(supply, effective_demand)
        for item in sellers:
            item["remaining_food"] = traded_food * float(item["food"]) / supply
        for item in buyers:
            item["remaining_food"] = traded_food * float(item["effective_food"]) / effective_demand
        sellers.sort(key=lambda item: (item["agent"].shell_balance, item["agent"].id))
        buyers.sort(key=lambda item: item["agent"].id)
        food_volume = 0.0
        shell_volume = 0.0
        transactions = 0
        participant_ids: set[int] = set()
        seller_index = 0
        for buyer_item in buyers:
            buyer = buyer_item["agent"]
            remaining = float(buyer_item["remaining_food"])
            for seller_item in sellers[seller_index:]:
                if remaining <= 1e-9:
                    break
                seller = seller_item["agent"]
                available = float(seller_item["remaining_food"])
                if available <= 1e-9:
                    seller_index += 1
                    continue
                amount_food = min(remaining, available)
                if amount_food <= 1e-9:
                    break
                shells = amount_food * clearing_price
                from_allocation = min(amount_food, max(0.0, allocations.get(seller.id, 0.0)))
                allocations[seller.id] = max(0.0, allocations.get(seller.id, 0.0) - from_allocation)
                from_reserve = amount_food - from_allocation
                if from_reserve > 1e-9:
                    seller.food_reserve = max(0.0, seller.food_reserve - from_reserve)
                allocations[buyer.id] = allocations.get(buyer.id, 0.0) + amount_food
                seller.shell_balance += shells
                buyer.shell_balance -= shells
                seller.food_sold += amount_food
                buyer.food_bought += amount_food
                seller.shell_earned_from_trade += shells
                buyer.shell_spent_on_food += shells
                seller.trade_count += 1
                buyer.trade_count += 1
                participant_ids.update({seller.id, buyer.id})
                contexts.setdefault(seller.id, {}).setdefault("trade_partners", set()).add(buyer.id)
                contexts.setdefault(buyer.id, {}).setdefault("trade_partners", set()).add(seller.id)
                contexts.setdefault(seller.id, {})["trade_food_sold"] = (
                    float(contexts.get(seller.id, {}).get("trade_food_sold", 0.0)) + amount_food
                )
                contexts.setdefault(buyer.id, {})["trade_food_bought"] = (
                    float(contexts.get(buyer.id, {}).get("trade_food_bought", 0.0)) + amount_food
                )
                seller_item["remaining_food"] = available - amount_food
                remaining -= amount_food
                food_volume += amount_food
                shell_volume += shells
                transactions += 1
                if seller_item["remaining_food"] <= 1e-9:
                    seller_index += 1
            if seller_index >= len(sellers):
                break

        record = {
            "tick": tick,
            "food_volume": food_volume,
            "food_volume_meals": food_volume / max(1e-12, self.params.need_base),
            "shell_volume": shell_volume,
            "clearing_price": clearing_price if transactions else None,
            "normalized_price": clearing_price if transactions else None,
            "price_anchor": 1.0,
            "price_rule": self.params.market_price_rule,
            "buyer_budget_fraction": self.params.buyer_budget_fraction,
            "buyers": len(buyers),
            "sellers": len(sellers),
            "transactions": transactions,
            "participant_ids": sorted(participant_ids),
            "unfunded_demand": unfunded_demand,
            "total_demand_meals": demand / max(1e-12, self.params.need_base),
        }
        self.trade_records.append(record)
        return record

    def _role_weight(self, role: Role) -> float:
        return {
            Role.CHIEF: self.params.chief_share_weight,
            Role.CHIEFTAIN: self.params.chieftain_share_weight,
            Role.HUNTER: self.params.hunter_share_weight,
        }[role]

    def _role_weight_allocation(
        self, total_food: float, agents: list[AgentState]
    ) -> dict[int, float]:
        weights = {agent.id: max(0.0, self._role_weight(agent.role)) for agent in agents}
        total_weight = sum(weights.values())
        if total_weight <= 1e-12:
            return self._need_contribution_allocation(
                total_food=total_food,
                agents=agents,
                contributions={agent.id: 0.0 for agent in agents},
                equal_weight=1.0,
            )
        return {
            agent_id: max(0.0, total_food) * weight / total_weight
            for agent_id, weight in weights.items()
        }

    def _apply_idiosyncratic_allocation_noise(
        self,
        allocations: dict[int, float],
        members: list[AgentState],
    ) -> dict[int, float]:
        """Perturb post-allocation individual realizations while preserving the pool."""
        total_food = sum(max(0.0, allocations.get(agent.id, 0.0)) for agent in members)
        if self.params.idio_sigma <= 1e-12 or len(members) < 2 or total_food <= 1e-12:
            return allocations
        realized = {
            agent.id: max(0.0, allocations.get(agent.id, 0.0))
            * math.exp(self.individual_rng.gauss(0.0, self.params.idio_sigma))
            for agent in members
        }
        realized_total = sum(realized.values())
        if realized_total <= 1e-12:
            return allocations
        scale = total_food / realized_total
        return {agent.id: realized[agent.id] * scale for agent in members}

    @staticmethod
    def _z_scores(values: list[float]) -> list[float]:
        average = mean(values)
        spread = stdev(values)
        if spread <= 1e-12:
            return [0.0 for _ in values]
        return [(value - average) / spread for value in values]

    def _evaluate_promotions(self, tick: int) -> None:
        if not self.agents or not self.params.rank_enabled:
            return
        shell_values = [agent.shell_balance for agent in self.agents]
        hunt_values = [agent.period_hunting_contribution for agent in self.agents]
        shell_z = self._z_scores([math.log1p(max(0.0, value)) for value in shell_values])
        hunt_z = self._z_scores(hunt_values)
        scores = {
            agent.id: self.params.promotion_shell_weight * shell_z[index]
            + self.params.promotion_hunting_weight * hunt_z[index]
            for index, agent in enumerate(self.agents)
        }
        ranked = sorted(
            self.agents,
            key=lambda agent: (
                scores[agent.id],
                agent.shell_balance,
                agent.period_hunting_contribution,
                -agent.extreme_triggers,
                self.promotion_tiebreakers[agent.id],
            ),
            reverse=True,
        )
        minimum_tenure_ticks = self.params.min_tenure_periods * self.params.promotion_period
        eligible_ids = {
            agent.id
            for agent in self.agents
            if tick - agent.role_since_tick >= minimum_tenure_ticks
        }
        chief_is_locked = self.chief_id not in eligible_ids
        chief_id = (
            self.chief_id
            if chief_is_locked
            else next(agent.id for agent in ranked if agent.id in eligible_ids)
        )
        locked_chieftains = [
            agent_id for agent_id in self.chieftain_ids if agent_id not in eligible_ids
        ]
        chieftain_slots_left = max(0, self.params.chieftain_slots - len(locked_chieftains))
        newly_selected_chieftains = [
            agent.id
            for agent in ranked
            if agent.id != chief_id
            and agent.id not in locked_chieftains
            and agent.id in eligible_ids
        ][:chieftain_slots_left]
        chieftain_ids = [*locked_chieftains, *newly_selected_chieftains]
        self.promotion_reviews.append(
            {
                "tick": tick,
                "evaluation_period": tick // self.params.promotion_period,
                "warmup_period": tick == self.params.burn_in_ticks,
                "score_mode": "zlog",
                "min_tenure_periods": self.params.min_tenure_periods,
                "top_ranked_chief_id": ranked[0].id,
                "chief_id": chief_id,
                "chieftain_ids": list(chieftain_ids),
                "shell_balances_tied_at_review": len(set(shell_values)) == 1,
            }
        )
        target_roles = {
            agent.id: (
                Role.CHIEF
                if agent.id == chief_id
                else Role.CHIEFTAIN
                if agent.id in set(chieftain_ids)
                else Role.HUNTER
            )
            for agent in self.agents
        }
        old_roles = {agent.id: agent.role for agent in self.agents}
        old_chief = self.chief_id
        self.chief_id = chief_id
        self.chieftain_ids = chieftain_ids
        for agent in self.agents:
            new_role = target_roles[agent.id]
            if old_roles[agent.id] is not new_role:
                agent.role_since_tick = tick + 1
                event = {
                    "tick": tick,
                    "effective_from_tick": tick + 1,
                    "evaluation_period": tick // self.params.promotion_period,
                    "warmup_period": tick == self.params.burn_in_ticks,
                    "agent_id": agent.id,
                    "old_role": old_roles[agent.id].value,
                    "old_role_label": old_roles[agent.id].label,
                    "new_role": new_role.value,
                    "new_role_label": new_role.label,
                    "shell_balance": agent.shell_balance,
                    "role_since_tick": agent.role_since_tick,
                    "hunting_contribution": agent.period_hunting_contribution,
                    "promotion_score": scores[agent.id],
                }
                self.promotion_events.append(event)
                self._append_system_log(
                    {
                        "tick": tick,
                        "scenario_id": self.scenario.id,
                        "level": "info",
                        "event": "role_change",
                        "agent_id": agent.id,
                        "message": (
                            f"第 {tick} 轮身份评选：Agent {agent.id} "
                            f"{old_roles[agent.id].label} → {new_role.label}，"
                            f"贝壳 {agent.shell_balance:.2f}，评选分 {scores[agent.id]:.3f}。"
                        ),
                        **event,
                    }
                )
            agent.period_hunting_contribution = 0.0
        self._rebuild_rank_groups()
        if old_chief != self.chief_id:
            self._append_system_log(
                {
                    "tick": tick,
                    "scenario_id": self.scenario.id,
                    "level": "info",
                    "event": "chief_elected",
                    "message": f"第 {tick} 轮评选完成：Agent {self.chief_id} 成为新首领。",
                    "agent_id": self.chief_id,
                }
            )

    def _recover_rescued_agents(self) -> dict[int, float]:
        received: dict[int, float] = {}
        for agent in self.agents:
            requested = agent.pending_relief
            if requested > 1e-9:
                if self.params.relief_source == "external":
                    delivered = requested
                    self.external_relief_total += delivered
                elif self.params.relief_source == "public":
                    delivered = min(requested, self.public_food_reserve)
                    self.public_food_reserve -= delivered
                    self.public_relief_total += delivered
                else:
                    delivered = 0.0
                agent.food_reserve += delivered
                agent.pending_relief = max(0.0, requested - delivered)
                received[agent.id] = delivered
            if agent.status is Status.EXTREME_FATIGUE and agent.food_reserve >= agent.need:
                agent.status = Status.NORMAL
                agent.pending_relief = 0.0
        return received

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
        participation_action = self._participation_action(agent, pressure_index, moral_fit)
        if self.params.enable_participation_choice:
            effort = 0.0 if participation_action < 0.15 else effort * (0.35 + 0.65 * participation_action)
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
            participation_action=participation_action,
            decision_source="rule",
            decision_reason="本地规则公式",
        )

    def _participation_action(
        self,
        agent: AgentState,
        pressure_index: float,
        moral_fit: float,
    ) -> float:
        if not self.params.enable_participation_choice:
            return 1.0
        social_pull = 0.12 * self.scenario.collectivization * agent.alt_orientation
        autonomy_pull = 0.10 * self.scenario.individualization * agent.development_orientation
        participation = (
            0.30
            + 0.24 * moral_fit
            + 0.16 * agent.compliance
            + 0.12 * (agent.wellbeing / 100.0)
            + 0.10 * pressure_index * agent.restraint_orientation
            + social_pull
            + autonomy_pull
            - 0.18 * agent.fatigue
        )
        if agent.status is Status.EXTREME_FATIGUE:
            participation -= 0.45
        return clip(participation, 0.0, 1.0)

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
        total_attempts = 1 + self.llm_decision_retries
        last_error: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            raw = ""
            try:
                raw = self.llm_provider.complete(prompt)
                payload = self._parse_llm_decision(raw)
                rule_default = self._rule_decide(agent, pressure_index, moral_fit)
                repairs: list[str] = []
                effort = self._unit_float_or_default(payload, "effort", rule_default.effort, repairs)
                if agent.status is Status.EXTREME_FATIGUE:
                    effort = 0.0
                compliance_action = self._unit_float_or_default(
                    payload,
                    "compliance_action",
                    rule_default.compliance_action,
                    repairs,
                )
                share_rate = self._unit_float_or_default(payload, "share_rate", rule_default.share_rate, repairs)
                restraint_rate = self._unit_float_or_default(
                    payload,
                    "restraint_rate",
                    rule_default.restraint_rate,
                    repairs,
                )
                if len(repairs) >= 4:
                    raise ValueError(f"LLM JSON 数值字段全部不可用: {repairs}")
                reason = self._normalize_decision_reason(payload.get("reason", ""))
                participation_action = (
                    self._unit_float_or_default(
                        payload,
                        "participation_action",
                        rule_default.participation_action,
                        repairs,
                    )
                    if "participation_action" in payload
                    else rule_default.participation_action
                )
                decision = AgentDecision(
                    moral_fit=moral_fit,
                    effort=effort,
                    compliance_action=compliance_action,
                    share_rate=share_rate,
                    restraint_rate=restraint_rate,
                    participation_action=participation_action,
                    decision_source=f"llm:{self.model_status.get('provider')}",
                    decision_reason=reason,
                )
                if repairs:
                    self._append_system_log(
                        {
                            "tick": tick,
                            "scenario_id": self.scenario.id,
                            "level": "warning",
                            "event": "llm_decision_repaired",
                            "message": (
                                f"{self.model_status.get('model')} 第 {tick} 轮 Agent {agent.id} "
                                f"返回 JSON 字段不完整，已用规则基准修复: {', '.join(repairs)}"
                            ),
                            "agent_id": agent.id,
                            "repairs": repairs,
                        }
                    )
                if attempt > 1:
                    self._append_system_log(
                        {
                            "tick": tick,
                            "scenario_id": self.scenario.id,
                            "level": "info",
                            "event": "llm_decision_retry_success",
                            "message": (
                                f"{self.model_status.get('model')} 第 {tick} 轮 Agent {agent.id} "
                                f"在第 {attempt}/{total_attempts} 次尝试时完成行为决策。"
                            ),
                            "agent_id": agent.id,
                            "attempt": attempt,
                            "total_attempts": total_attempts,
                        }
                    )
                return decision
            except Exception as exc:
                last_error = exc
                detail = str(exc)
                raw_excerpt = self._compact_response_excerpt(raw)
                if raw_excerpt:
                    detail = f"{detail}；响应片段: {raw_excerpt}"
                if attempt < total_attempts:
                    self._append_system_log(
                        {
                            "tick": tick,
                            "scenario_id": self.scenario.id,
                            "level": "warning",
                            "event": "llm_decision_retry",
                            "message": (
                                f"{self.model_status.get('model')} 第 {tick} 轮 Agent {agent.id} "
                                f"行为决策第 {attempt}/{total_attempts} 次失败，正在重试: {detail}"
                            ),
                            "agent_id": agent.id,
                            "attempt": attempt,
                            "total_attempts": total_attempts,
                        }
                    )
                    time.sleep(min(0.6 * attempt, 1.5))
                    continue
                break

        fallback = self._rule_decide(agent, pressure_index, moral_fit)
        fallback.decision_source = f"llm-fallback:{self.model_status.get('provider')}"
        fallback.decision_reason = f"LLM多次失败后安全兜底: {str(last_error)[:120]}"
        self._append_system_log(
            {
                "tick": tick,
                "scenario_id": self.scenario.id,
                "level": "warning",
                "event": "llm_decision_fallback",
                "message": (
                    f"{self.model_status.get('model')} 第 {tick} 轮 Agent {agent.id} "
                    f"重试 {total_attempts} 次后仍失败，已使用本地安全兜底完成本步: {last_error}"
                ),
                "agent_id": agent.id,
                "attempt": total_attempts,
                "total_attempts": total_attempts,
            }
        )
        return fallback

    def _agent_decision_prompt(
        self,
        *,
        agent: AgentState,
        pressure_index: float,
        moral_fit: float,
        tick: int,
        prior_decisions: list[dict[str, Any]],
    ) -> str:
        if self.model_status.get("provider") == "local-deepseek":
            return self._compact_agent_decision_prompt(
                agent=agent,
                pressure_index=pressure_index,
                moral_fit=moral_fit,
                tick=tick,
                prior_decisions=prior_decisions,
            )

        observed_prior = prior_decisions[-12:]
        payload = {
            "sc": {
                "id": self.scenario.id,
                "m": self.scenario.mode.value,
                "sp": round(self.scenario.scarcity_pressure, 4),
                "op": round(self.scenario.opportunity, 4),
                "c": round(self.scenario.collectivization, 4),
                "i": round(self.scenario.individualization, 4),
            },
            "ctx": {
                "t": tick,
                "n": len(self.agents),
                "p": round(pressure_index, 4),
                "u": round(self.last_underfed_rate, 4),
                "prior": observed_prior,
            },
            "ag": {
                "id": agent.id,
                "age": agent.age_group.value,
                "sex": agent.sex,
                "m": agent.moral_type.value,
                "mw": {key: round(value, 3) for key, value in agent.moral_weights.items()},
                "fit": round(moral_fit, 4),
                "phy": round(agent.physical, 4),
                "exp": round(agent.exploration, 4),
                "coor": round(agent.coordination, 4),
                "comp": round(agent.compliance, 4),
                "need": round(agent.need, 4),
                "well": round(agent.wellbeing, 4),
                "res": round(agent.food_reserve, 4),
                "fat": round(agent.fatigue, 4),
                "role": agent.role.value,
                "shells": round(agent.shell_balance, 4),
                "st": agent.status.value,
                "gid": agent.group_id,
            },
        }
        return (
            "Return exactly one compact JSON object. No markdown, no prose, no extra keys. "
            'Template: {"participation_action":0.9,"effort":0.6,"compliance_action":0.7,'
            '"share_rate":0.5,"restraint_rate":0.2,"reason":"fit"}. '
            "participation_action, effort, compliance_action, share_rate, restraint_rate must be numbers from 0 to 1. "
            "compliance_action means obedience score; never output an action word. "
            "reason must be one short code: survival, fit, fair, fatigue, reserve, or belong. "
            "If ag.st is Extreme_Fatigue, set effort=0 and restraint_rate>=0.7. "
            "Balance survival, moral fit, autonomy, belonging, fairness, and fatigue. "
            "Data:"
            f"{json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}"
        )

    def _compact_agent_decision_prompt(
        self,
        *,
        agent: AgentState,
        pressure_index: float,
        moral_fit: float,
        tick: int,
        prior_decisions: list[dict[str, Any]],
    ) -> str:
        moral_code = {
            MoralType.SELF_DEVELOPMENT: "sd",
            MoralType.SELF_RESTRAINT: "sr",
            MoralType.ALTRUISTIC_DEVELOPMENT: "ad",
            MoralType.ALTRUISTIC_RESTRAINT: "ar",
        }[agent.moral_type]
        mode_code = {
            ActionMode.ORGANIZED: "org",
            ActionMode.GROUP: "grp",
            ActionMode.INDIVIDUAL: "ind",
        }[self.scenario.mode]
        payload = {
            "sc": {
                "id": self.scenario.id,
                "m": mode_code,
                "scar": round(self.scenario.scarcity_pressure, 2),
                "opp": round(self.scenario.opportunity, 2),
                "coll": round(self.scenario.collectivization, 2),
                "ind": round(self.scenario.individualization, 2),
            },
            "ctx": {
                "t": tick,
                "p": round(pressure_index, 2),
                "u": round(self.last_underfed_rate, 2),
                "prior": prior_decisions[-4:],
            },
            "ag": {
                "id": agent.id,
                "age": agent.age_group.value[0],
                "m": moral_code,
                "fit": round(moral_fit, 2),
                "phy": round(agent.physical, 2),
                "comp": round(agent.compliance, 2),
                "need": round(agent.need, 2),
                "well": round(agent.wellbeing, 1),
                "res": round(agent.food_reserve, 2),
                "fat": round(agent.fatigue, 2),
                "role": agent.role.value,
                "shells": round(agent.shell_balance, 2),
                "st": "xf" if agent.status is Status.EXTREME_FATIGUE else "n",
            },
        }
        return (
            "JSON only. Keys exactly participation_action,effort,compliance_action,share_rate,restraint_rate,reason. "
            "First five are numbers 0..1; compliance_action is obedience score. "
            "reason one code: survival,fit,fair,fatigue,reserve,belong. "
            "If ag.st=xf then effort=0 and restraint_rate>=0.7. "
            f"Data:{json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}"
        )

    def _adapt_moral_weights(
        self,
        *,
        agent: AgentState,
        decision: AgentDecision,
        scores: WellbeingScores,
        context: dict[str, Any],
        starved: bool,
    ) -> None:
        rate = self.params.moral_adaptation_rate
        if rate <= 1e-12:
            return
        action_profile = self._action_moral_profile(
            decision=decision,
            perceived_fairness=scores.perceived_fairness,
            social_belonging=scores.social_belonging,
            contribution_signal=float(context.get("contribution_signal", 0.5)),
        )
        scenario_profile = {
            "self_dev": self.scenario.opportunity * (0.55 + 0.45 * self.scenario.individualization),
            "self_rest": self.scenario.scarcity_pressure * (0.55 + 0.45 * self.scenario.individualization),
            "alt_dev": self.scenario.opportunity * (0.45 + 0.55 * self.scenario.collectivization),
            "alt_rest": self.scenario.scarcity_pressure * (0.45 + 0.55 * self.scenario.collectivization),
        }
        if starved:
            scenario_profile["self_rest"] += 0.25
            scenario_profile["alt_rest"] += 0.15
        reward = clip(scores.composite / 100.0, 0.0, 1.0)
        target = {
            key: 0.45 * self._normalized_profile(scenario_profile)[key]
            + 0.35 * action_profile[key]
            + 0.20 * self.scenario.moral_fit_weights[key]
            for key in MORAL_COMPONENTS
        }
        effective_rate = rate * (0.35 + 0.65 * reward)
        updated = {
            key: (1.0 - effective_rate) * agent.moral_weights[key] + effective_rate * target[key]
            for key in MORAL_COMPONENTS
        }
        agent.moral_weights = self._normalized_profile(updated)

    @staticmethod
    def _normalized_profile(profile: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, float(profile.get(key, 0.0))) for key in MORAL_COMPONENTS)
        if total <= 1e-12:
            return {key: 1.0 / len(MORAL_COMPONENTS) for key in MORAL_COMPONENTS}
        return {
            key: max(0.0, float(profile.get(key, 0.0))) / total
            for key in MORAL_COMPONENTS
        }

    @staticmethod
    def _action_moral_profile(
        *,
        decision: AgentDecision,
        perceived_fairness: float,
        social_belonging: float,
        contribution_signal: float,
    ) -> dict[str, float]:
        action = {
            "self_dev": (
                0.35 * decision.effort
                + 0.25 * decision.participation_action
                + 0.25 * contribution_signal
                + 0.15 * (1.0 - decision.restraint_rate)
            ),
            "self_rest": (
                0.55 * decision.restraint_rate
                + 0.25 * (1.0 - min(1.0, decision.effort))
                + 0.20 * perceived_fairness
            ),
            "alt_dev": (
                0.35 * decision.share_rate
                + 0.25 * decision.compliance_action
                + 0.20 * social_belonging
                + 0.20 * contribution_signal
            ),
            "alt_rest": (
                0.35 * decision.share_rate
                + 0.35 * decision.restraint_rate
                + 0.20 * perceived_fairness
                + 0.10 * decision.compliance_action
            ),
        }
        total = sum(action.values())
        if total <= 1e-12:
            return {key: 1.0 / len(MORAL_COMPONENTS) for key in MORAL_COMPONENTS}
        return {key: clip(value / total, 0.0, 1.0) for key, value in action.items()}

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
        if isinstance(data, dict) and isinstance(data.get("out"), dict):
            data = data["out"]
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

    @staticmethod
    def _unit_float_or_default(
        payload: dict[str, Any],
        key: str,
        default: float,
        repairs: list[str],
    ) -> float:
        try:
            if key not in payload:
                raise ValueError("missing")
            value = payload[key]
            if isinstance(value, str):
                value = value.strip()
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("non-finite")
            return clip(number, 0.0, 1.0)
        except (TypeError, ValueError):
            repairs.append(f"{key}->{default:.2f}")
            return clip(default, 0.0, 1.0)

    @staticmethod
    def _normalize_decision_reason(raw: Any) -> str:
        text = str(raw or "").strip().replace("\n", " ")
        if not text:
            return "fit"
        translations = {
            "survival": "生存优先",
            "fit": "道德适配",
            "fair": "公平分配",
            "fatigue": "控制疲劳",
            "reserve": "保留储备",
            "belong": "群体归属",
        }
        return translations.get(text.lower(), text[:80])

    @staticmethod
    def _compact_response_excerpt(raw: str, limit: int = 160) -> str:
        text = str(raw or "").strip().replace("\n", " ")
        if not text:
            return ""
        return text[:limit]

    def _agent_ability(self, agent: AgentState, mode: ActionMode | None = None) -> float:
        mode = mode or self.scenario.mode
        if mode is ActionMode.INDIVIDUAL:
            return 0.5 * agent.physical + 0.5 * agent.exploration
        return 0.5 * agent.physical + 0.5 * agent.coordination

    def _environment_shock(self) -> float:
        spread = self.params.environment_volatility
        return self.environment_rng.uniform(1.0 - spread, 1.0 + spread)

    def _resource_multiplier(self) -> float:
        if self.scenario.target_supply_ratio < 1.0:
            return self.params.scarcity_resource_multiplier
        return self.params.abundance_resource_multiplier

    def _organized_tick(
        self, tick: int, decisions: dict[int, AgentDecision]
    ) -> tuple[dict[int, float], dict[int, dict[str, float]], dict[str, float]]:
        resource_multiplier = self._resource_multiplier()
        base_food = (
            len(self.agents)
            * self.scenario.base_resource
            * self.params.food_scale
            * self._environment_shock()
            * resource_multiplier
        )
        contributions = {
            agent.id: decisions[agent.id].effort * self._agent_ability(agent, ActionMode.ORGANIZED)
            for agent in self.agents
        }
        total_contribution = sum(contributions.values())
        avg_compliance = mean(decisions[agent.id].compliance_action for agent in self.agents)
        team_count = max(1, math.ceil(len(self.agents) / self.params.typical_team_size))
        organization_cost = self.params.org_cost_per_team * team_count
        continuous_food = max(0.0, (
            self.params.alpha
            * total_contribution
            * (1.0 + self.params.lambda_c * self.scenario.collectivization * avg_compliance)
            * resource_multiplier
            - organization_cost
        ))
        team_power = (
            self.params.base_combat_power
            * sum(decisions[a.id].effort * (0.5 + 0.5 * a.physical) for a in self.agents)
            * (1.0 + self.params.lambda_c * avg_compliance)
        )
        hunt = self.hunting.resolve_hunt(
            team_power=team_power,
            rng=self.hunting_rng,
            events=team_count,
            encounter_rate=0.85 + 0.15 * self.scenario.base_resource,
        )
        hunt_food = hunt.meat * resource_multiplier
        produced_food = continuous_food + hunt_food
        total_food = max(0.0, base_food + produced_food)
        if self.params.rank_enabled and self.params.rank_split_enabled:
            allocations = self._role_weight_allocation(total_food, self.agents)
        else:
            allocations = self._need_contribution_allocation(
                total_food=total_food,
                agents=self.agents,
                contributions=contributions,
                equal_weight=self.params.organized_equal_weight,
            )
        allocations = self._apply_idiosyncratic_allocation_noise(allocations, self.agents)
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
                "hunting_contribution": contributions[agent.id],
            }
        return allocations, contexts, {
            "base_food": base_food,
            "produced_food": produced_food,
            "hunt_food": hunt_food,
            "organization_cost": organization_cost,
            "team_count": float(team_count),
            "total_food": total_food,
            "target_supply_ratio": self.scenario.target_supply_ratio,
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
        total_organization_cost = 0.0
        stability_values: list[float] = []

        for group_id, member_ids in sorted(self.groups.items()):
            members = [self.agents[agent_id] for agent_id in member_ids]
            if not members:
                continue
            n = len(members)
            resource_multiplier = self._resource_multiplier()
            base_food = (
                n * self.scenario.base_resource * self.params.food_scale * self._environment_shock()
                * resource_multiplier
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
            organization_cost = self.params.group_cost_per_team * max(
                1.0, n / self.params.typical_team_size
            )
            continuous_food = max(0.0, (
                self.params.alpha
                * total_contribution
                * (1.0 + self.params.lambda_g * group_coordination)
                * resource_multiplier
                - organization_cost
            ))
            team_power = (
                self.params.base_combat_power
                * sum(decisions[a.id].effort * (0.5 + 0.5 * a.physical) for a in members)
                * (1.0 + self.params.lambda_g * group_coordination)
            )
            hunt = self.hunting.resolve_hunt(
                team_power=team_power,
                rng=self.hunting_rng,
                events=max(1, round(n / self.params.typical_team_size)),
                encounter_rate=0.80 + 0.15 * self.scenario.base_resource,
            )
            group_hunt_food = hunt.meat * resource_multiplier
            group_food = max(0.0, base_food + continuous_food + group_hunt_food)
            total_base += base_food
            total_produced += continuous_food
            total_hunt += group_hunt_food
            total_organization_cost += organization_cost

            avg_alt = mean(agent.alt_orientation for agent in members)
            avg_restraint = mean(agent.restraint_orientation for agent in members)
            share_norm = clip(
                0.20
                + self.params.share_norm_alt_weight * avg_alt
                + self.params.share_norm_restraint_weight * avg_restraint,
                0.0,
                1.0,
            )
            if self.params.rank_enabled and self.params.rank_split_enabled:
                leadership_food = group_food * self.params.chief_leadership_share
                group_allocations = self._role_weight_allocation(
                    group_food - leadership_food,
                    members,
                )
                if self.chief_id is not None:
                    allocations[self.chief_id] = allocations.get(self.chief_id, 0.0) + (
                        leadership_food
                    )
            else:
                group_allocations = self._need_contribution_allocation(
                    total_food=group_food,
                    agents=members,
                    contributions=contributions,
                    equal_weight=share_norm,
                )
            group_allocations = self._apply_idiosyncratic_allocation_noise(
                group_allocations,
                members,
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
                    "hunting_contribution": contributions[agent.id],
                }
                self._update_group_affinity(
                    agent=agent,
                    group_id=group_id,
                    group_avg_food=group_avg_food,
                    fairness=float(contexts[agent.id]["fairness"]),
                    social_belonging=float(contexts[agent.id]["social_belonging"]),
                )

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
                    "hunting_contribution": 0.0,
                },
            )
        if self.chief_id is not None and self.params.rank_enabled and self.params.rank_split_enabled:
            chief = self.agents[self.chief_id]
            contexts[chief.id] = {
                "fairness": clip(1.0 - gini(list(allocations.values())), 0.0, 1.0),
                "autonomy": 0.35,
                "social_belonging": 0.85,
                "contribution_signal": 0.5,
                "hunt_failed": allocations.get(chief.id, 0.0) < chief.need * 0.55,
                "hunting_contribution": 0.0,
            }
        total_food = sum(allocations.values())
        return allocations, contexts, {
            "base_food": total_base,
            "produced_food": total_produced + total_hunt,
            "hunt_food": total_hunt,
            "organization_cost": total_organization_cost,
            "team_count": float(len(self.groups)),
            "total_food": total_food,
            "target_supply_ratio": self.scenario.target_supply_ratio,
            "group_stability": mean(stability_values, 1.0),
            "hunt_success_rate": 0.0,
        }

    def _update_group_affinity(
        self,
        *,
        agent: AgentState,
        group_id: int,
        group_avg_food: float,
        fairness: float,
        social_belonging: float,
    ) -> None:
        weight = self.params.relationship_memory_weight
        if weight <= 1e-12:
            return
        food_signal = clip(group_avg_food / max(0.1, agent.need), 0.0, 1.0)
        signal = clip(0.35 * fairness + 0.35 * social_belonging + 0.30 * food_signal, 0.0, 1.0)
        current = clip(agent.group_affinity.get(group_id, 0.0), 0.0, 1.0)
        agent.group_affinity[group_id] = clip((1.0 - weight) * current + weight * signal, 0.0, 1.0)
        decay = 1.0 - min(0.08, 0.25 * weight)
        for other_group_id in list(agent.group_affinity):
            if other_group_id != group_id:
                agent.group_affinity[other_group_id] = clip(agent.group_affinity[other_group_id] * decay, 0.0, 1.0)

    def _ensure_groups_for_tick(self, tick: int) -> None:
        if self.params.rank_enabled:
            if not self.groups:
                self._rebuild_rank_groups(initial=True)
            return
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
        resource_multiplier = self._resource_multiplier()
        successes = 0
        events = 0

        for agent in self.agents:
            decision = decisions[agent.id]
            base_food = (
                self.scenario.base_resource
                * self.params.food_scale
                * self._environment_shock()
                * resource_multiplier
            )
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
            risk_return = max(0.0, self.individual_rng.gauss(risk_mean, max(0.03, risk_sd)))
            continuous_food = (
                self.params.alpha_individual * decision.effort * ability * risk_return
                * resource_multiplier
            )
            encounter_rate = clip(0.15 + 0.70 * decision.effort + 0.10 * agent.exploration, 0.0, 0.95)
            team_power = (
                self.params.base_combat_power
                * decision.effort
                * (0.55 + 0.45 * agent.physical)
                * (0.85 + 0.25 * agent.exploration)
            )
            hunt = self.hunting.resolve_hunt(
                team_power=team_power,
                rng=self.hunting_rng,
                events=1,
                encounter_rate=encounter_rate,
            )
            events += hunt.events
            successes += hunt.successes
            total_base += base_food
            total_produced += continuous_food
            scaled_hunt_food = hunt.meat * resource_multiplier
            total_hunt += scaled_hunt_food
            raw_food[agent.id] = max(0.0, base_food + continuous_food + scaled_hunt_food)
            contexts[agent.id] = {
                "fairness": 0.42,
                "autonomy": 0.90 + 0.08 * self.scenario.individualization,
                "social_belonging": 0.24 + 0.10 * agent.alt_orientation,
                "contribution_signal": clip(decision.effort * ability / 0.75, 0.0, 1.0),
                "hunt_failed": decision.effort > 0.05 and hunt.successes == 0 and continuous_food < agent.need * 0.25,
                "hunting_contribution": max(0.0, continuous_food + scaled_hunt_food),
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
            "organization_cost": 0.0,
            "team_count": float(len(self.agents)),
            "total_food": sum(allocations.values()),
            "target_supply_ratio": self.scenario.target_supply_ratio,
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
                key=lambda agent: (agent.moral_type.value, self.group_rng.random()),
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
        self.group_rng.shuffle(agents)
        for agent in agents:
            weights = GROUP_CHOICE_WEIGHTS[agent.moral_type]
            best_group = None
            best_score = -1.0
            for group_id in range(group_count):
                if len(new_groups[group_id]) >= self.params.max_group_size:
                    continue
                memory = memories[group_id]
                similarity = self._similarity_to_group_memory(agent, memory)
                prior_relation = clip(agent.group_affinity.get(group_id, 0.0), 0.0, 1.0)
                score = (
                    weights["expected_food"] * memory["expected_food"]
                    + weights["fairness"] * memory["fairness"]
                    + weights["similarity"] * similarity
                    + weights["stability"] * memory["stability"]
                    + self.params.relationship_memory_weight * prior_relation
                    + self.group_rng.uniform(-0.02, 0.02)
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
        food_consumed: float,
        unmet_food: float,
        relief_received: float,
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
        trade_food_bought = float(context.get("trade_food_bought", 0.0))
        trade_food_sold = float(context.get("trade_food_sold", 0.0))
        driver_text = (
            ""
            if decision.decision_source == "rule"
            else f" 模型判断: {decision.decision_reason or decision.decision_source};"
        )
        message = (
            f"{self.scenario.id} 第 {tick} 轮 Agent {agent.id}({agent.moral_type.label})"
            f"{group_text}: {action}; 努力 {decision.effort:.2f}, "
            f"分得 {allocated:.2f}, 实际消费 {food_consumed:.2f}/{effective_need:.2f}, "
            f"缺口 {unmet_food:.2f}, 救济 {relief_received:.2f}, "
            f"储备 {reserve_before:.2f}->{agent.food_reserve:.2f}, "
            f"疲劳 {fatigue_before:.2f}->{agent.fatigue:.2f}, "
            f"幸福 {wellbeing_before:.1f}->{agent.wellbeing:.1f}, "
            f"身份 {agent.role.label}, 贝壳 {agent.shell_balance:.2f}"
            f"（买入 {trade_food_bought:.2f}/卖出 {trade_food_sold:.2f}）;{driver_text} {result}。"
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
            "participation": round(decision.participation_action, 4),
            "effort": round(decision.effort, 4),
            "compliance": round(decision.compliance_action, 4),
            "share_rate": round(decision.share_rate, 4),
            "restraint_rate": round(decision.restraint_rate, 4),
            "moral_fit": round(decision.moral_fit, 4),
            "decision_source": decision.decision_source,
            "decision_reason": decision.decision_reason,
            "allocated_food": round(allocated, 4),
            "food_consumed": round(food_consumed, 4),
            "unmet_food": round(unmet_food, 4),
            "relief_received": round(relief_received, 4),
            "effective_need": round(effective_need, 4),
            "food_reserve_before": round(reserve_before, 4),
            "food_reserve_after": round(agent.food_reserve, 4),
            "fatigue_before": round(fatigue_before, 4),
            "fatigue_after": round(agent.fatigue, 4),
            "status_before": status_before.value,
            "status_after": agent.status.value,
            "role": agent.role.value,
            "role_label": agent.role.label,
            "superior_id": agent.superior_id,
            "shell_balance": round(agent.shell_balance, 4),
            "trade_food_bought": round(trade_food_bought, 4),
            "trade_food_sold": round(trade_food_sold, 4),
            "wellbeing_before": round(wellbeing_before, 4),
            "wellbeing_after": round(agent.wellbeing, 4),
            "fairness": round(float(context.get("fairness", 0.0)), 4),
            "autonomy": round(float(context.get("autonomy", 0.0)), 4),
            "social_belonging": round(float(context.get("social_belonging", 0.0)), 4),
            "hunt_failed": bool(context.get("hunt_failed", False)),
            "starved": starved,
            "result": result,
            "scores": {
                key: round(value, 4) if isinstance(value, (int, float)) else value
                for key, value in scores.items()
            },
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
            f" 完成模型行为判断: {action}; 参与 {decision.participation_action:.2f}, 努力 {decision.effort:.2f}, "
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
            "participation": round(decision.participation_action, 4),
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
        trade: dict[str, float | int | None],
        *,
        food_opening_stock: float,
        food_closing_stock: float,
        consumed: float,
        unmet: float,
        external_relief: float,
        public_relief: float,
        food_loss: float,
        food_ledger_error: float,
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
            "avg_objective_wellbeing": mean(agent.objective_wellbeing for agent in self.agents if agent.objective_wellbeing is not None),
            "avg_peer_perceived_wellbeing": (mean(a.peer_perceived_wellbeing for a in self.agents if a.peer_perceived_wellbeing is not None) if any(a.peer_perceived_wellbeing is not None for a in self.agents) else None),
            "social_gini_index": gini(list(allocations.values())),
            "minimum_survival_rate": survival_rate,
            "base_food": production["base_food"],
            "produced_food": production["produced_food"],
            "hunt_food": production.get("hunt_food", 0.0),
            "total_food": production["total_food"],
            "food_consumed": consumed,
            "unmet_food": unmet,
            "food_opening_stock": food_opening_stock,
            "food_closing_stock": food_closing_stock,
            "external_relief": external_relief,
            "public_relief": public_relief,
            "public_food_reserve": self.public_food_reserve,
            "food_loss": food_loss,
            "food_ledger_error": food_ledger_error,
            "realized_supply_ratio": production["total_food"]
            / max(1e-9, sum(agent.need for agent in self.agents)),
            "trade_food": float(trade.get("food_volume", 0.0)),
            "trade_shells": float(trade.get("shell_volume", 0.0)),
            "market_price": trade.get("clearing_price"),
            "normalized_market_price": trade.get("normalized_price"),
            "unfunded_demand": float(trade.get("unfunded_demand", 0.0)),
            "avg_shell_balance": mean(agent.shell_balance for agent in self.agents),
            "shell_gini_index": gini([agent.shell_balance for agent in self.agents]),
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
            burn_in_ticks=self.params.burn_in_ticks,
        )
        trade_metrics = summarize_trade_metrics(
            agents=self.agents,
            trade_records=self.trade_records,
            burn_in_ticks=self.params.burn_in_ticks,
        )
        final_food_inventory = self.public_food_reserve + sum(
            agent.food_reserve for agent in self.agents
        )
        initial_food_inventory = self.initial_public_food_reserve
        total_production = sum(self.total_food_by_tick)
        food_balance_error = (
            initial_food_inventory
            + total_production
            + self.external_relief_total
            - self.total_food_consumed
            - self.total_food_loss
            - final_food_inventory
        )
        initial_shell_total = self.initial_shell_total
        final_shell_total = sum(agent.shell_balance for agent in self.agents)
        shell_balance_error = final_shell_total - initial_shell_total
        max_tick_ledger_error = max(
            (abs(float(record["ledger_error"])) for record in self.food_ledger_records),
            default=0.0,
        )
        if abs(food_balance_error) > 1e-6 or max_tick_ledger_error > 1e-6:
            raise RuntimeError(
                "Food inventory ledger failed conservation check: "
                f"final error={food_balance_error:.3g}, max tick error={max_tick_ledger_error:.3g}"
            )
        if abs(shell_balance_error) > 1e-6:
            raise RuntimeError(
                f"Shell ledger failed conservation check: error={shell_balance_error:.3g}"
            )
        society.update(
            {
                "avg_objective_wellbeing": (mean(v for a in self.agents for v in a.objective_wellbeing_history[self.params.burn_in_ticks:]) if any(a.objective_wellbeing_history[self.params.burn_in_ticks:] for a in self.agents) else None),
                "avg_peer_perceived_wellbeing": (mean(v for a in self.agents for v in a.peer_perceived_wellbeing_history[self.params.burn_in_ticks:] if v is not None) if any(v is not None for a in self.agents for v in a.peer_perceived_wellbeing_history[self.params.burn_in_ticks:]) else None),
                "avg_shell_balance": trade_metrics["avg_shell_balance"],
                "shell_gini_index": trade_metrics["shell_gini_index"],
                "total_trade_volume_food": trade_metrics["total_trade_volume_food"],
                "total_trade_volume_meals": trade_metrics["total_trade_volume_meals"],
                "total_trade_volume_shells": trade_metrics["total_trade_volume_shells"],
                "trade_participation_rate": trade_metrics["trade_participation_rate"],
                "unfunded_demand_meals": trade_metrics["unfunded_demand_meals"],
            }
        )
        role_metrics = summarize_role_metrics(
            self.agents,
            self.params.ticks,
            self.params.burn_in_ticks,
        )
        result = {
            "scenario": self.scenario.to_dict(),
            "parameters": self.params.to_dict(),
            "model_status": self.model_status,
            "society_metrics": society,
            "moral_type_metrics": summarize_by_moral_type(
                self.agents,
                self.params.ticks,
                self.params.burn_in_ticks,
            ),
            "group_metrics": aggregate_group_metrics(self.group_records),
            "time_series": self.time_series,
            "system_logs": self.system_logs,
            "agent_logs": self.agent_logs,
            "wellbeing_evaluations": self.wellbeing_evaluations,
            "final_agents": [agent.to_snapshot() for agent in self.agents],
            "group_records": self.group_records[-200:],
            "trade_metrics": trade_metrics,
            "role_metrics": role_metrics,
            "hierarchy_metrics": self._hierarchy_metrics(role_metrics),
            "trade_records": self.trade_records[-200:],
            "promotion_events": self.promotion_events[-200:],
            "promotion_reviews": self.promotion_reviews[-100:],
            "food_accounting": {
                "initial_public_food_reserve": self.initial_public_food_reserve,
                "final_public_food_reserve": self.public_food_reserve,
                "external_relief_total": self.external_relief_total,
                "public_relief_total": self.public_relief_total,
                "total_production": total_production,
                "total_consumed": self.total_food_consumed,
                "total_loss": self.total_food_loss,
                "final_inventory": final_food_inventory,
                "balance_error": food_balance_error,
                "max_abs_tick_error": max_tick_ledger_error,
                "relief_source": self.params.relief_source,
            },
            "shell_accounting": {
                "initial_shell_total": initial_shell_total,
                "final_shell_total": final_shell_total,
                "balance_error": shell_balance_error,
            },
            "feasibility_checks": {
                "food_ledger_conserved": abs(food_balance_error) <= 1e-6,
                "shell_ledger_conserved": abs(shell_balance_error) <= 1e-6,
                "abundance_extreme_fatigue_below_50pct": (
                    society["extreme_fatigue_rate"] < 0.5
                )
                if self.scenario.target_supply_ratio > 1.0
                else None,
                "market_effective": (
                    bool(trade_metrics["market_effective"])
                    if self.params.trade_enabled
                    else None
                ),
            },
            "experiment_protocol": {
                "ticks": self.params.ticks,
                "replications": self.params.replications,
                "seed_base": self.params.seed,
                "replication_seed_step": 100_003,
                "scenario_seed_offset_step": 100_000,
                "scenario_seed": self.scenario_seed,
                "paired_exogenous_streams": True,
                "promotion_period": self.params.promotion_period,
                "burn_in_ticks": self.params.burn_in_ticks,
                "analysis_ticks": max(0, self.params.ticks - self.params.burn_in_ticks),
                "initial_shells_distribution": self.params.initial_shells_distribution,
                "market_price_rule": self.params.market_price_rule,
                "idio_sigma": self.params.idio_sigma,
                "detailed_agent_logs_retained": self.collect_agent_logs,
            },
            "hierarchy": {
                "chief_id": self.chief_id,
                "chieftain_ids": list(self.chieftain_ids),
                "hunter_assignments": {
                    str(chieftain_id): list(hunters)
                    for chieftain_id, hunters in self.hunter_assignments.items()
                },
                "chieftain_slots": self.params.chieftain_slots,
                "hunters_per_chieftain": self.params.hunters_per_chieftain,
            },
        }
        self._assert_numeric_result(result)
        return result

    def _hierarchy_metrics(self, role_metrics: dict[str, dict[str, float | str]]) -> dict[str, float]:
        reviews = [
            review
            for review in self.promotion_reviews
            if int(review.get("tick", 0)) >= self.params.burn_in_ticks
        ]
        role_rank = {Role.HUNTER.value: 0, Role.CHIEFTAIN.value: 1, Role.CHIEF.value: 2}
        events = [
            event
            for event in self.promotion_events
            if int(event.get("tick", 0)) >= self.params.burn_in_ticks
        ]
        transitions = max(1, len(reviews) - 1)
        chief_retained = sum(
            1
            for previous, current in zip(reviews, reviews[1:], strict=False)
            if previous.get("chief_id") == current.get("chief_id")
        )
        chieftain_retained = sum(
            len(set(previous.get("chieftain_ids", [])) & set(current.get("chieftain_ids", [])))
            / max(1, self.params.chieftain_slots)
            for previous, current in zip(reviews, reviews[1:], strict=False)
        )
        upward = sum(
            1
            for event in events
            if role_rank[str(event["new_role"])] > role_rank[str(event["old_role"])]
        )
        downward = sum(
            1
            for event in events
            if role_rank[str(event["new_role"])] < role_rank[str(event["old_role"])]
        )
        role_shells = [
            float(role_metrics[role.value]["avg_shell_balance"])
            for role in Role
        ]
        role_food = [
            float(role_metrics[role.value]["avg_food_allocated"])
            for role in Role
        ]
        return {
            "promotion_rate": len(events) / max(1, len(self.agents) * max(1, len(reviews))),
            "chief_retention_rate": chief_retained / transitions,
            "chieftain_retention_rate": chieftain_retained / transitions,
            "upward_mobility_rate": upward / max(1, len(self.agents) * max(1, len(reviews))),
            "downward_mobility_rate": downward / max(1, len(self.agents) * max(1, len(reviews))),
            "role_hierarchy_shell_gini": gini(role_shells),
            "role_hierarchy_food_gini": gini(role_food),
        }

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
    if params.replications > 1:
        return _run_replicated_batch(
            ids,
            params,
            event_callback=event_callback,
            stop_checker=stop_checker,
        )
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
            event_callback(
                {
                    "type": "scenario_start",
                    "scenario": SCENARIOS[scenario_id].to_dict(),
                    "scenario_seed": params.seed + int(scenario_id[1:]) * 100_000,
                }
            )
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
    result["hypothesis_checks"] = _hypothesis_checks(result["comparison"])
    if event_callback is not None:
        event_callback({"type": "complete", "result": result})
    return result


def _run_replicated_batch(
    ids: list[str],
    params: SimulationParameters,
    *,
    event_callback: StreamCallback | None = None,
    stop_checker: StopChecker | None = None,
) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = {scenario_id: [] for scenario_id in ids}
    if event_callback is not None:
        event_callback(
            {
                "type": "batch_start",
                "parameters": params.to_dict(),
                "scenarios": [SCENARIOS[scenario_id].to_dict() for scenario_id in ids],
                "model_status": provider_status(params.llm_provider).to_dict(),
            }
        )
    for replication in range(params.replications):
        rep_params = SimulationParameters.from_dict(params.to_dict())
        rep_params.replications = 1
        rep_params.seed = params.seed + replication * 100_003
        for scenario_id in ids:
            if stop_checker is not None and stop_checker():
                raise RuntimeError("simulation stopped by user")
            if event_callback is not None:
                event_callback(
                    {
                        "type": "scenario_start",
                        "scenario": SCENARIOS[scenario_id].to_dict(),
                        "replication": replication + 1,
                        "seed": rep_params.seed,
                        "scenario_seed": rep_params.seed + int(scenario_id[1:]) * 100_000,
                    }
                )

            def emit_replication_event(event: dict[str, Any]) -> None:
                if event_callback is None:
                    return
                event_callback(
                    {
                        **event,
                        "replication": replication + 1,
                        "seed": rep_params.seed,
                        "scenario_seed": rep_params.seed + int(scenario_id[1:]) * 100_000,
                    }
                )

            simulation = Simulation(
                scenario_id,
                rep_params,
                event_callback=emit_replication_event if event_callback is not None else None,
                stop_checker=stop_checker,
                collect_agent_logs=False,
            )
            scenario_result = simulation.run()
            scenario_result["replication"] = replication + 1
            scenario_result["seed"] = rep_params.seed
            by_scenario[scenario_id].append(scenario_result)

    results = [
        _aggregate_replicate_results(scenario_id, scenario_results, params)
        for scenario_id, scenario_results in by_scenario.items()
    ]
    comparison = _comparison(results)
    result = {
        "parameters": params.to_dict(),
        "scenarios": [SCENARIOS[scenario_id].to_dict() for scenario_id in ids],
        "results": results,
        "comparison": comparison,
        "model_status": provider_status(params.llm_provider).to_dict(),
        "replication_count": params.replications,
        "hypothesis_checks": _hypothesis_checks(comparison),
    }
    if event_callback is not None:
        event_callback({"type": "complete", "result": result})
    return result


def _aggregate_replicate_results(
    scenario_id: str,
    scenario_results: list[dict[str, Any]],
    params: SimulationParameters,
) -> dict[str, Any]:
    first = scenario_results[0]
    abundance_checks = [
        bool(result["feasibility_checks"]["abundance_extreme_fatigue_below_50pct"])
        for result in scenario_results
        if result.get("feasibility_checks", {}).get(
            "abundance_extreme_fatigue_below_50pct"
        ) is not None
    ]
    market_checks = [
        bool(result["feasibility_checks"]["market_effective"])
        for result in scenario_results
        if result.get("feasibility_checks", {}).get("market_effective") is not None
    ]
    food_accounting = _mean_metric_dict(
        [result.get("food_accounting", {}) for result in scenario_results]
    )
    food_accounting["max_abs_tick_error"] = max(
        float(result.get("food_accounting", {}).get("max_abs_tick_error", 0.0))
        for result in scenario_results
    )
    food_accounting["max_abs_balance_error"] = max(
        abs(float(result.get("food_accounting", {}).get("balance_error", 0.0)))
        for result in scenario_results
    )
    shell_accounting = _mean_metric_dict(
        [result.get("shell_accounting", {}) for result in scenario_results]
    )
    shell_accounting["max_abs_balance_error"] = max(
        abs(float(result.get("shell_accounting", {}).get("balance_error", 0.0)))
        for result in scenario_results
    )
    result = {
        "scenario": SCENARIOS[scenario_id].to_dict(),
        "parameters": params.to_dict(),
        "model_status": first["model_status"],
        "society_metrics": _mean_metric_dict([result["society_metrics"] for result in scenario_results]),
        "moral_type_metrics": _mean_nested_metric_dict(
            [result["moral_type_metrics"] for result in scenario_results]
        ),
        "group_metrics": _mean_metric_dict([result["group_metrics"] for result in scenario_results]),
        "trade_metrics": _mean_metric_dict([result.get("trade_metrics", {}) for result in scenario_results]),
        "role_metrics": _mean_nested_metric_dict([result.get("role_metrics", {}) for result in scenario_results]),
        "hierarchy_metrics": _mean_metric_dict(
            [result.get("hierarchy_metrics", {}) for result in scenario_results]
        ),
        "food_accounting": food_accounting,
        "shell_accounting": shell_accounting,
        "feasibility_checks": {
            "food_ledger_conserved": all(
                result.get("feasibility_checks", {}).get("food_ledger_conserved", False)
                for result in scenario_results
            ),
            "shell_ledger_conserved": all(
                result.get("feasibility_checks", {}).get("shell_ledger_conserved", False)
                for result in scenario_results
            ),
            "abundance_extreme_fatigue_pass_rate": (
                mean(float(value) for value in abundance_checks) if abundance_checks else None
            ),
            "market_effective_pass_rate": (
                mean(float(value) for value in market_checks) if market_checks else None
            ),
        },
        "experiment_protocol": first.get("experiment_protocol", {}),
        "time_series": _mean_time_series([result["time_series"] for result in scenario_results]),
        "system_logs": first["system_logs"][:200],
        "agent_logs": first["agent_logs"][:300],
        "wellbeing_evaluations": [{**evaluation, "replication": index + 1} for index, result in enumerate(scenario_results) for evaluation in result.get("wellbeing_evaluations", [])],
        "final_agents": first["final_agents"],
        "group_records": first["group_records"],
        "trade_records": first.get("trade_records", []),
        "promotion_events": first.get("promotion_events", []),
        "promotion_reviews": first.get("promotion_reviews", []),
        "hierarchy": first.get("hierarchy", {}),
        "replication_count": len(scenario_results),
        "replicate_summaries": [
            {
                "replication": result.get("replication", index + 1),
                "seed": result.get("seed"),
                "society_metrics": result["society_metrics"],
                "moral_type_metrics": result["moral_type_metrics"],
                "food_accounting": result.get("food_accounting", {}),
                "shell_accounting": result.get("shell_accounting", {}),
                "feasibility_checks": result.get("feasibility_checks", {}),
            }
            for index, result in enumerate(scenario_results)
        ],
    }
    return result


def _mean_metric_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    keys = set().union(*(item.keys() for item in items))
    result: dict[str, Any] = {}
    for key in keys:
        values = [item.get(key) for item in items if key in item]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if numeric:
            result[key] = mean(numeric)
        elif all(value is None for value in values):
            result[key] = None
        elif values:
            result[key] = values[0]
    return result


def _mean_nested_metric_dict(items: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    keys = set().union(*(item.keys() for item in items))
    return {
        key: _mean_metric_dict([item[key] for item in items if key in item])
        for key in keys
    }


def _mean_time_series(series_items: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not series_items:
        return []
    min_len = min(len(series) for series in series_items)
    rows: list[dict[str, Any]] = []
    for index in range(min_len):
        row_items = [series[index] for series in series_items]
        row = _mean_metric_dict([
            {key: value for key, value in item.items() if key != "by_moral_type"}
            for item in row_items
        ])
        row["by_moral_type"] = _mean_nested_metric_dict([
            item.get("by_moral_type", {}) for item in row_items
        ])
        rows.append(row)
    return rows


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
                "avg_food_consumed": society["avg_food_consumed"],
                "avg_unmet_food": society["avg_unmet_food"],
                "realized_supply_ratio": society["realized_supply_ratio"],
                "avg_wellbeing": society["avg_wellbeing"],
                "social_gini_index": society["social_gini_index"],
                "minimum_survival_rate": society["minimum_survival_rate"],
                "total_social_food": society["total_social_food"],
                "avg_shell_balance": society.get("avg_shell_balance", 0.0),
                "shell_gini_index": society.get("shell_gini_index", 0.0),
                "total_trade_volume_food": society.get("total_trade_volume_food", 0.0),
                "total_trade_volume_meals": society.get("total_trade_volume_meals", 0.0),
                "trade_participation_rate": society.get("trade_participation_rate", 0.0),
                "moral_type_metrics": result["moral_type_metrics"],
                "group_metrics": result.get("group_metrics", {}),
                "trade_metrics": result.get("trade_metrics", {}),
                "role_metrics": result.get("role_metrics", {}),
            }
        )
    return rows


def _hypothesis_checks(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["scenario_id"]): row for row in comparison}

    def metric(scenario_id: str, moral_type: str, key: str, default: float = 0.0) -> float:
        return float(
            by_id.get(scenario_id, {})
            .get("moral_type_metrics", {})
            .get(moral_type, {})
            .get(key, default)
        )

    def avg(scenario_ids: list[str], moral_types: list[str], key: str) -> float:
        values = [metric(scenario_id, moral_type, key) for scenario_id in scenario_ids for moral_type in moral_types]
        return mean(values)

    def scenario_metric(scenario_id: str, key: str) -> float:
        return float(by_id.get(scenario_id, {}).get(key, 0.0))

    checks = []
    restraint = ["self_restraint", "altruistic_restraint"]
    development = ["self_development", "altruistic_development"]
    self_types = ["self_development", "self_restraint"]
    alt_types = ["altruistic_development", "altruistic_restraint"]
    scarce = ["S1", "S2", "S3"]
    affluent = ["S4", "S5", "S6"]

    restraint_fatigue = avg(scarce, restraint, "extreme_fatigue_rate")
    development_fatigue = avg(scarce, development, "extreme_fatigue_rate")
    checks.append({
        "id": "H1",
        "passed": restraint_fatigue <= development_fatigue,
        "evidence": f"匮乏情境克制类极疲劳率 {restraint_fatigue:.3f} vs 发展类 {development_fatigue:.3f}",
    })

    dev_food = avg(affluent, development, "avg_food_allocated")
    rest_food = avg(affluent, restraint, "avg_food_allocated")
    checks.append({
        "id": "H2",
        "passed": dev_food >= rest_food,
        "evidence": f"丰裕情境发展类食物 {dev_food:.3f} vs 克制类 {rest_food:.3f}",
    })

    organized_alt = avg(["S1", "S4"], alt_types, "avg_wellbeing")
    organized_self = avg(["S1", "S4"], self_types, "avg_wellbeing")
    checks.append({
        "id": "H3",
        "passed": organized_alt >= organized_self,
        "evidence": f"强组织利他类幸福 {organized_alt:.3f} vs 自我类 {organized_self:.3f}",
    })

    individual_self_food = avg(["S3", "S6"], self_types, "avg_food_allocated")
    individual_alt_food = avg(["S3", "S6"], alt_types, "avg_food_allocated")
    checks.append({
        "id": "H4",
        "passed": individual_self_food >= individual_alt_food,
        "evidence": f"各自行动自我类食物 {individual_self_food:.3f} vs 利他类 {individual_alt_food:.3f}",
    })

    group_count = mean([
        float(by_id.get(scenario_id, {}).get("group_metrics", {}).get("groups_per_tick", 0.0))
        for scenario_id in ["S2", "S5"]
    ])
    checks.append({
        "id": "H5",
        "passed": group_count > 1.0,
        "evidence": f"抱团情境平均群体数 {group_count:.3f}",
    })

    s1_consistency = {
        moral_type: metric("S1", moral_type, "avg_moral_consistency")
        for moral_type in [*self_types, *alt_types]
    }
    best_s1 = max(s1_consistency, key=s1_consistency.get) if s1_consistency else ""
    checks.append({
        "id": "H6",
        "passed": best_s1 == "altruistic_restraint",
        "evidence": f"S1 道德一致性最高: {best_s1 or 'NA'}",
    })

    s6_food = {
        moral_type: metric("S6", moral_type, "avg_food_allocated")
        for moral_type in [*self_types, *alt_types]
    }
    best_s6 = max(s6_food, key=s6_food.get) if s6_food else ""
    checks.append({
        "id": "H7",
        "passed": best_s6 == "self_development",
        "evidence": f"S6 食物最高: {best_s6 or 'NA'}",
    })

    s5_wellbeing = scenario_metric("S5", "avg_wellbeing")
    s4_wellbeing = scenario_metric("S4", "avg_wellbeing")
    checks.append({
        "id": "H8",
        "passed": s5_wellbeing >= s4_wellbeing,
        "evidence": f"S5 平均幸福 {s5_wellbeing:.3f} vs S4 {s4_wellbeing:.3f}",
    })
    return checks
