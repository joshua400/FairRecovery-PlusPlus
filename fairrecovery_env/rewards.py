"""
FairRecovery++ — Reward Engine and Grader.

Computes dense rewards and final terminal scores (The Honest Truth).
Includes fixes for the persistent_ignore_vulnerable bug.
"""

from __future__ import annotations
import structlog
from typing import List, Optional, Dict
from .constants import (
    WEIGHT_UTILITY, WEIGHT_FAIRNESS, WEIGHT_SAFETY,
    GRADER_SCORE_MIN, GRADER_SCORE_MAX,
    PENALTY_SAFETY_VIOLATION, PENALTY_REPEATED_ACTION,
    REWARD_STABILITY, REWARD_WEIGHTS
)
from .models import FairRecoveryAction, FairRecoveryState, ZoneState
from .tasks import TaskDefinition

logger = structlog.get_logger(__name__)

class RewardComponents:
    def __init__(self, R_total: float, R_exec: float, R_fair: float, R_safe: float, feedback: str):
        self.R_total = R_total
        self.R_exec = R_exec
        self.R_fair = R_fair
        self.R_safe = R_safe
        self.feedback = feedback

class RewardEngine:
    """Stateful reward calculator for a FairRecovery episode."""

    def __init__(self, task: TaskDefinition) -> None:
        self._task = task
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._action_history: list[str] = []
        self._vulnerable_ignored_days: int = 0

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    def compute_reward(self, action: FairRecoveryAction, state: FairRecoveryState) -> tuple[float, str]:
        """Compute reward for a single agent action."""
        # This is a generic wrapper that handles analyze/allocate. 
        # For 'execute', we call compute_execute_step separately.
        
        reward = 0.0
        feedback_parts: list[str] = []
        self._step_count += 1

        # 1. Action Type History (Penalty for repetition)
        action_repr = action.action_type.value
        recent = self._action_history[-3:]
        if len(recent) >= 3 and all(a == action_repr for a in recent):
            reward += PENALTY_REPEATED_ACTION
            feedback_parts.append("System is stagnating.")
        self._action_history.append(action_repr)

        # 2. Heuristic Penalties
        if action.action_type == "allocate" and not action.allocations:
            reward += PENALTY_SAFETY_VIOLATION
            feedback_parts.append("Budget is unspent.")

        # 3. Phase Specific Reward Signal
        if action.action_type == "analyze":
            # Small positive signal for analyzing critical zones
            reward += 0.05
        
        self._cumulative_reward += reward
        feedback = " ".join(feedback_parts) if feedback_parts else "Action processed."
        return float(reward), feedback

    def compute_execute_step(
        self,
        state: FairRecoveryState,
        violations: List[str],
        allocated_zone_ids: frozenset = frozenset(),
    ) -> RewardComponents:
        """Detailed execution reward with fairness trap detection."""
        
        # 1. Utility (Execution Reward)
        # Reward for reducing damage across all zones
        R_exec = 0.0
        for zone in state.zones:
            R_exec += (1.0 - zone.damage)
        R_exec /= len(state.zones)

        # 2. Fairness (Persistent Ignore Logic)
        vuln_zone_ids = {z.zone_id for z in state.zones if z.vulnerable_ratio > 0.6}
        
        # Check if any vulnerable zone was served this step
        zone_served = bool(vuln_zone_ids & allocated_zone_ids)
        
        if vuln_zone_ids:
            if not zone_served and state.day > 1:
                self._vulnerable_ignored_days += 1
                if self._vulnerable_ignored_days >= 2:
                    violations.append(f"persistent_ignore_vulnerable:{list(vuln_zone_ids)}")
            else:
                self._vulnerable_ignored_days = max(0, self._vulnerable_ignored_days - 1)

        R_fair = self._compute_equity(state.zones)
        
        # 3. Safety (Violations)
        R_safe = 1.0 - (len(violations) * 0.1)
        R_safe = max(-1.0, R_safe)

        # Composite Total
        w = REWARD_WEIGHTS
        R_total = (
            w["exec"] * R_exec +
            w["fair"] * R_fair +
            w["safe"] * R_safe
        )
        R_total = float(max(-1.0, min(1.0, R_total)))
        self._cumulative_reward += R_total
        
        feedback = f"Exec Score: {R_exec:.2f}, Fairness: {R_fair:.2f}"
        if self._vulnerable_ignored_days >= 2:
            feedback += " | ⚠️ PERSISTENT NEGLECT WARNING"

        return RewardComponents(R_total, R_exec, R_fair, R_safe, feedback)

    def _compute_equity(self, zones: List[ZoneState]) -> float:
        services = [z.service for z in zones]
        mean_svc = sum(services) / len(services)
        if mean_svc == 0: return 0.5
        mad = sum(abs(s - mean_svc) for s in services) / len(services)
        return 1.0 - (mad / mean_svc if mean_svc > 0 else 0)

    def get_final_score(self, state: FairRecoveryState) -> float:
        """Compute the 'Honest Truth' terminal score."""
        avg_damage = sum(z.damage for z in state.zones) / len(state.zones)
        utility = 1.0 - avg_damage
        
        equity = self._compute_equity(state.zones)
        safety = max(0.0, 1.0 - (state.violations_total / 10.0))

        final_score = (
            WEIGHT_UTILITY * utility +
            WEIGHT_FAIRNESS * equity +
            WEIGHT_SAFETY * safety
        )
        return round(max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, final_score)), 4)

def compute_fairness_reward(zones: List[ZoneState]) -> float:
    # Helper for external calls
    services = [z.service_level for z in zones]
    mean_svc = sum(services) / len(services)
    if mean_svc == 0: return 0.5
    mad = sum(abs(s - mean_svc) for s in services) / len(services)
    return 1.0 - mad
