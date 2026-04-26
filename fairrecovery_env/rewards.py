"""
FairRecovery++ — Reward Engine and Grader.

Computes dense rewards and final terminal scores (The Honest Truth).
"""

from __future__ import annotations
import structlog
from typing import List, Optional
from .constants import (
    WEIGHT_UTILITY, WEIGHT_FAIRNESS, WEIGHT_SAFETY,
    GRADER_SCORE_MIN, GRADER_SCORE_MAX,
    PENALTY_SAFETY_VIOLATION, PENALTY_REPEATED_ACTION,
    REWARD_STABILITY
)
from .models import FairRecoveryAction, FairRecoveryState, ZoneState
from .tasks import TaskDefinition

logger = structlog.get_logger(__name__)

class RewardEngine:
    """Stateful reward calculator for a FairRecovery episode."""

    def __init__(self, task: TaskDefinition) -> None:
        self._task = task
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._action_history: list[str] = []

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    def compute_reward(self, action: FairRecoveryAction, state: FairRecoveryState) -> tuple[float, str]:
        """Compute reward for a single agent action based on environmental state."""
        reward = 0.0
        feedback_parts: list[str] = []
        self._step_count += 1

        # 1. Action Type History (Penalty for repetition)
        action_repr = action.action_type.value
        recent = self._action_history[-3:]
        if len(recent) >= 3 and all(a == action_repr for a in recent):
            reward += PENALTY_REPEATED_ACTION
            feedback_parts.append("System is stagnating. Try a different phase.")
        self._action_history.append(action_repr)

        # 2. Safety Violations (Adversarial Penalties)
        # Check if agent ignored critical zones identified in 'analyze'
        if action.action_type == "allocate":
            # (Simplified check: did they allocate anything?)
            if not action.allocations:
                reward += PENALTY_SAFETY_VIOLATION
                feedback_parts.append("Budget is unspent while zones suffer.")

        # 3. Stability Reward (Small positive signal for keeping things from getting worse)
        avg_damage = sum(z.damage for z in state.zones) / len(state.zones)
        if avg_damage < 0.5:
            reward += REWARD_STABILITY
        
        self._cumulative_reward += reward
        feedback = " ".join(feedback_parts) if feedback_parts else "Action processed."

        logger.info(
            "reward_computed",
            action_type=action.action_type.value,
            reward=round(reward, 4),
            cumulative=round(self._cumulative_reward, 4),
        )

        return reward, feedback

    def get_final_score(self, state: FairRecoveryState) -> float:
        """Compute the 'Honest Truth' terminal score in [0, 1]."""
        
        # 1. Utility (Inverse of final damage)
        avg_damage = sum(z.damage for z in state.zones) / len(state.zones)
        utility = 1.0 - avg_damage
        
        # 2. Fairness (MAD-based Equity)
        services = [z.service_level for z in state.zones]
        mean_svc = sum(services) / len(services)
        mad = sum(abs(s - mean_svc) for s in services) / len(services)
        fairness = 1.0 - mad
        
        # 3. Safety
        # Penalize for cumulative violations during the episode
        safety = max(0.0, 1.0 - (state.violations_total / 10.0))

        # The Composite Formula
        final_score = (
            WEIGHT_UTILITY * utility +
            WEIGHT_FAIRNESS * fairness +
            WEIGHT_SAFETY * safety
        )
        
        clamped = max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, final_score))
        return round(clamped, 4)


class TaskGrader:
    """Grader for evaluating trained models."""
    
    def __init__(self, task_id: str) -> None:
        from .tasks import TaskID, get_task
        self._task = get_task(TaskID(task_id))
        self._engine = RewardEngine(self._task)

    def grade(self, state: FairRecoveryState) -> float:
        return self._engine.get_final_score(state)
