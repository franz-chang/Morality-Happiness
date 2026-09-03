from __future__ import annotations

from .models import AgentDecision, AgentState, Scenario, WellbeingScores
from .utils import clip


def _reason(label: str, values: dict[str, float]) -> str:
    strongest = max(values, key=lambda key: abs(values[key])) if values else label
    return f"{label}:{strongest}"


def material_wellbeing_score(food_ratio: float) -> float:
    """Piecewise inverted-U score: hungry hurts, enough peaks, surplus decays."""
    ratio = max(0.0, food_ratio)
    if ratio < 1.0:
        score = 80.0 * ratio
    elif ratio <= 1.5:
        score = 80.0 + 40.0 * (ratio - 1.0)
    else:
        score = 100.0 - 16.0 * (ratio - 1.5)
    return clip(score, 0.0, 100.0)


def moral_consistency(
    *,
    agent: AgentState,
    decision: AgentDecision,
    scenario: Scenario,
    perceived_fairness: float,
    social_belonging: float,
    contribution_signal: float,
) -> float:
    action = {
        "self_dev": (
            0.35 * decision.effort
            + 0.35 * scenario.individualization
            + 0.20 * contribution_signal
            + 0.10 * (1.0 - decision.restraint_rate)
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
        return 0.0
    action = {key: value / total for key, value in action.items()}
    distance = sum(abs(action[key] - agent.moral_weights[key]) for key in action)
    return clip(1.0 - distance / 2.0, 0.0, 1.0)


def evaluate_wellbeing(
    *,
    agent: AgentState,
    decision: AgentDecision,
    scenario: Scenario,
    allocated_food: float,
    effective_need: float,
    perceived_fairness: float,
    autonomy: float,
    social_belonging: float,
    contribution_signal: float,
    starved: bool,
    moral_consistency_weight: float,
    autonomy_weight: float,
) -> WellbeingScores:
    food_ratio = allocated_food / max(0.1, effective_need)
    material = material_wellbeing_score(food_ratio)
    fairness = clip(perceived_fairness, 0.0, 1.0)
    autonomy = clip(autonomy, 0.0, 1.0)
    social = clip(social_belonging, 0.0, 1.0)
    consistency = moral_consistency(
        agent=agent,
        decision=decision,
        scenario=scenario,
        perceived_fairness=fairness,
        social_belonging=social,
        contribution_signal=contribution_signal,
    )
    stress = clip(0.45 * (agent.fatigue / 2.0) + 0.35 * scenario.scarcity_pressure + (0.20 if starved else 0.0), 0.0, 1.0)

    moral_weight = clip(moral_consistency_weight, 0.05, 0.45)
    auto_weight = clip(autonomy_weight, 0.05, 0.35)
    fairness_weight = 0.15
    social_weight = 0.12
    material_weight = max(0.25, 0.72 - moral_weight - auto_weight - fairness_weight - social_weight)

    eval_terms = {
        "material": material_weight * material,
        "moral": moral_weight * consistency * 100.0,
        "fairness": fairness_weight * fairness * 100.0,
        "autonomy": auto_weight * autonomy * 100.0,
        "social": social_weight * social * 100.0,
        "stress": -14.0 * stress,
    }
    eval_score = (
        eval_terms["material"]
        + eval_terms["moral"]
        + eval_terms["fairness"]
        + eval_terms["autonomy"]
        + eval_terms["social"]
        + eval_terms["stress"]
    )

    self_moral_bias = 0.10 * agent.self_orientation * autonomy * 100.0
    self_social_bias = 0.10 * agent.alt_orientation * social * 100.0
    self_terms = {
        "material": 0.42 * material,
        "moral": 0.24 * consistency * 100.0,
        "fairness": 0.12 * fairness * 100.0,
        "autonomy": 0.12 * autonomy * 100.0 + self_moral_bias,
        "social": self_social_bias,
        "stress": -16.0 * stress,
    }
    self_score = (
        self_terms["material"]
        + self_terms["moral"]
        + self_terms["fairness"]
        + self_terms["autonomy"]
        + self_terms["social"]
        + self_terms["stress"]
    )

    peer_terms = {
        "material": 0.48 * material,
        "fairness": 0.18 * fairness * 100.0,
        "social": 0.20 * social * 100.0,
        "moral": 0.14 * consistency * 100.0,
        "stress": -10.0 * stress,
    }
    peer_score = (
        peer_terms["material"]
        + peer_terms["fairness"]
        + peer_terms["social"]
        + peer_terms["moral"]
        + peer_terms["stress"]
    )
    eval_score = clip(eval_score, 0.0, 100.0)
    self_score = clip(self_score, 0.0, 100.0)
    peer_score = clip(peer_score, 0.0, 100.0)
    composite = clip((eval_score + self_score + peer_score) / 3.0, 0.0, 100.0)

    return WellbeingScores(
        eval_score=eval_score,
        peer_score=peer_score,
        self_score=self_score,
        composite=composite,
        material_score=material,
        moral_consistency=consistency,
        perceived_fairness=fairness,
        autonomy=autonomy,
        social_belonging=social,
        stress=stress,
        eval_reason=_reason("eval", eval_terms),
        peer_reason=_reason("peer", peer_terms),
        self_reason=_reason("self", self_terms),
    )
