from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import AgentState, MoralType, Role
from .utils import gini, mean, stdev


def summarize_by_moral_type(agents: list[AgentState], total_ticks: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for moral_type in MoralType:
        members = [agent for agent in agents if agent.moral_type == moral_type]
        if not members:
            continue
        all_wellbeing = [value for agent in members for value in agent.wellbeing_history]
        all_food = [value for agent in members for value in agent.allocated_history]
        all_consistency = [value for agent in members for value in agent.moral_consistency_history]
        result[moral_type.value] = {
            "label": moral_type.label,
            "avg_food_allocated": mean(all_food),
            "avg_wellbeing": mean(all_wellbeing),
            "wellbeing_volatility": mean(stdev(agent.wellbeing_history) for agent in members),
            "extreme_fatigue_rate": sum(agent.extreme_triggers for agent in members)
            / max(1, total_ticks * len(members)),
            "avg_moral_consistency": mean(all_consistency),
            "avg_final_reserve": mean(agent.food_reserve for agent in members),
            "population": float(len(members)),
        }
    return result


def summarize_society(
    *,
    agents: list[AgentState],
    total_ticks: int,
    total_food_by_tick: list[float],
    survival_rates: list[float],
) -> dict[str, float]:
    all_food = [value for agent in agents for value in agent.allocated_history]
    all_wellbeing = [value for agent in agents for value in agent.wellbeing_history]
    return {
        "total_social_food": sum(total_food_by_tick),
        "avg_tick_social_food": mean(total_food_by_tick),
        "avg_food_allocated": mean(all_food),
        "avg_wellbeing": mean(all_wellbeing),
        "wellbeing_volatility": mean(stdev(agent.wellbeing_history) for agent in agents),
        "social_gini_index": gini([agent.cumulative_food for agent in agents]),
        "minimum_survival_rate": mean(survival_rates),
        "extreme_fatigue_rate": sum(agent.extreme_triggers for agent in agents)
        / max(1, total_ticks * len(agents)),
        "avg_final_reserve": mean(agent.food_reserve for agent in agents),
    }


def aggregate_group_metrics(group_records: list[dict[str, Any]]) -> dict[str, float]:
    if not group_records:
        return {}
    by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in group_records:
        by_tick[int(record["tick"])].append(record)
    return {
        "group_total_food": sum(float(record["group_total_food"]) for record in group_records),
        "avg_group_total_food": mean(float(record["group_total_food"]) for record in group_records),
        "group_gini_index": mean(float(record["group_gini_index"]) for record in group_records),
        "group_stability": mean(float(record["group_stability"]) for record in group_records),
        "avg_group_size": mean(float(record["size"]) for record in group_records),
        "groups_per_tick": mean(len(records) for records in by_tick.values()),
    }


def summarize_trade_metrics(
    *,
    agents: list[AgentState],
    trade_records: list[dict[str, Any]],
) -> dict[str, float]:
    prices = [float(record.get("clearing_price", 0.0)) for record in trade_records]
    food_volume = sum(float(record.get("food_volume", 0.0)) for record in trade_records)
    shell_volume = sum(float(record.get("shell_volume", 0.0)) for record in trade_records)
    participating = sum(1 for agent in agents if agent.trade_count > 0)
    return {
        "avg_shell_balance": mean(agent.shell_balance for agent in agents),
        "shell_gini_index": gini([agent.shell_balance for agent in agents]),
        "total_trade_volume_food": food_volume,
        "total_trade_volume_shells": shell_volume,
        "avg_market_clearing_price": mean(prices),
        "trade_participation_rate": participating / max(1, len(agents)),
        "food_self_sufficiency_rate": sum(
            1 for agent in agents if agent.food_bought <= 1e-9
        ) / max(1, len(agents)),
        "trade_rounds": float(len(trade_records)),
    }


def summarize_role_metrics(agents: list[AgentState]) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    for role in Role:
        members = [agent for agent in agents if agent.role is role]
        result[role.value] = {
            "label": role.label,
            "population": float(len(members)),
            "avg_shell_balance": mean(agent.shell_balance for agent in members),
            "avg_food_allocated": mean(
                value for agent in members for value in agent.allocated_history
            ),
            "avg_wellbeing": mean(
                value for agent in members for value in agent.wellbeing_history
            ),
            "avg_hunting_contribution": mean(
                agent.hunting_contribution for agent in members
            ),
            "avg_food_bought": mean(agent.food_bought for agent in members),
            "avg_food_sold": mean(agent.food_sold for agent in members),
        }
    return result
