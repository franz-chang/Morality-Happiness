from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import AgentState, MoralType, Role
from .utils import gini, mean, stdev


def summarize_by_moral_type(
    agents: list[AgentState], total_ticks: int, burn_in_ticks: int = 0
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    analysis_ticks = max(1, total_ticks - burn_in_ticks)
    for moral_type in MoralType:
        members = [agent for agent in agents if agent.moral_type == moral_type]
        if not members:
            continue
        all_wellbeing = [value for agent in members for value in agent.wellbeing_history[burn_in_ticks:]]
        all_food = [value for agent in members for value in agent.allocated_history[burn_in_ticks:]]
        all_consistency = [
            value for agent in members for value in agent.moral_consistency_history[burn_in_ticks:]
        ]
        result[moral_type.value] = {
            "label": moral_type.label,
            "avg_food_allocated": mean(all_food),
            "avg_food_consumed": sum(sum(agent.food_consumed_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
            "avg_unmet_food": sum(sum(agent.unmet_food_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
            "avg_relief_received": sum(sum(agent.relief_received_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
            "avg_wellbeing": mean(all_wellbeing),
            "wellbeing_volatility": mean(stdev(agent.wellbeing_history) for agent in members),
            "extreme_fatigue_rate": sum(sum(agent.extreme_trigger_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
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
    burn_in_ticks: int = 0,
) -> dict[str, float]:
    analysis_ticks = max(1, total_ticks - burn_in_ticks)
    all_food = [value for agent in agents for value in agent.allocated_history[burn_in_ticks:]]
    all_wellbeing = [value for agent in agents for value in agent.wellbeing_history[burn_in_ticks:]]
    return {
        "total_social_food": sum(total_food_by_tick[burn_in_ticks:]),
        "avg_tick_social_food": mean(total_food_by_tick[burn_in_ticks:]),
        "avg_food_allocated": mean(all_food),
        "avg_food_consumed": sum(sum(agent.food_consumed_history[burn_in_ticks:]) for agent in agents)
        / max(1, analysis_ticks * len(agents)),
        "avg_unmet_food": sum(sum(agent.unmet_food_history[burn_in_ticks:]) for agent in agents)
        / max(1, analysis_ticks * len(agents)),
        "total_food_consumed": sum(sum(agent.food_consumed_history[burn_in_ticks:]) for agent in agents),
        "total_unmet_food": sum(sum(agent.unmet_food_history[burn_in_ticks:]) for agent in agents),
        "total_relief_received": sum(sum(agent.relief_received_history[burn_in_ticks:]) for agent in agents),
        "realized_supply_ratio": mean(total_food_by_tick[burn_in_ticks:])
        / max(1e-9, sum(agent.need for agent in agents)),
        "avg_wellbeing": mean(all_wellbeing),
        "wellbeing_volatility": mean(stdev(agent.wellbeing_history) for agent in agents),
        "social_gini_index": gini([agent.cumulative_food for agent in agents]),
        "minimum_survival_rate": mean(survival_rates[burn_in_ticks:]),
        "extreme_fatigue_rate": sum(sum(agent.extreme_trigger_history[burn_in_ticks:]) for agent in agents)
        / max(1, analysis_ticks * len(agents)),
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
    burn_in_ticks: int = 0,
) -> dict[str, float | bool | None]:
    trade_records = [record for record in trade_records if int(record.get("tick", 0)) > burn_in_ticks]
    prices = [
        float(record["clearing_price"])
        for record in trade_records
        if isinstance(record.get("clearing_price"), (int, float))
    ]
    normalized_prices = [
        float(record["normalized_price"])
        for record in trade_records
        if isinstance(record.get("normalized_price"), (int, float))
    ]
    food_volume = sum(float(record.get("food_volume", 0.0)) for record in trade_records)
    food_volume_meals = sum(
        float(record.get("food_volume_meals", 0.0)) for record in trade_records
    )
    shell_volume = sum(float(record.get("shell_volume", 0.0)) for record in trade_records)
    participating_ids = {
        int(agent_id)
        for record in trade_records
        for agent_id in record.get("participant_ids", [])
    }
    total_demand_meals = sum(
        float(record.get("total_demand_meals", 0.0)) for record in trade_records
    )
    unfunded_demand_meals = sum(
        float(record.get("unfunded_demand", 0.0)) for record in trade_records
    )
    trade_participation_rate = len(participating_ids) / max(1, len(agents))
    return {
        "avg_shell_balance": mean(agent.shell_balance for agent in agents),
        "shell_gini_index": gini([agent.shell_balance for agent in agents]),
        "total_trade_volume_food": food_volume,
        "total_trade_volume_meals": food_volume_meals,
        "total_trade_volume_shells": shell_volume,
        "avg_market_clearing_price": mean(prices) if prices else None,
        "avg_normalized_market_price": mean(normalized_prices) if normalized_prices else None,
        "price_anchor": next(
            (
                float(record["price_anchor"])
                for record in trade_records
                if isinstance(record.get("price_anchor"), (int, float))
            ),
            None,
        ),
        "unfunded_demand_meals": unfunded_demand_meals,
        "total_demand_meals": total_demand_meals,
        "unfunded_demand_share": unfunded_demand_meals / max(1e-9, total_demand_meals),
        "trade_participation_rate": trade_participation_rate,
        "market_effective": trade_participation_rate > 0.30,
        "food_self_sufficiency_rate": sum(
            1 for agent in agents if agent.food_bought <= 1e-9
        ) / max(1, len(agents)),
        "trade_rounds": float(len(trade_records)),
    }


def summarize_role_metrics(
    agents: list[AgentState], total_ticks: int, burn_in_ticks: int = 0
) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    analysis_ticks = max(1, total_ticks - burn_in_ticks)
    for role in Role:
        members = [agent for agent in agents if agent.role is role]
        result[role.value] = {
            "label": role.label,
            "population": float(len(members)),
            "avg_shell_balance": mean(agent.shell_balance for agent in members),
            "avg_food_allocated": mean(
                value for agent in members for value in agent.allocated_history[burn_in_ticks:]
            ),
            "avg_food_consumed": sum(sum(agent.food_consumed_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
            "avg_unmet_food": sum(sum(agent.unmet_food_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
            "avg_relief_received": sum(sum(agent.relief_received_history[burn_in_ticks:]) for agent in members)
            / max(1, analysis_ticks * len(members)),
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
