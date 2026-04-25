"""
FairRecovery++ - Reward Engine (Fair-GRPO-RLVR).

Computes dense, verifiable, formula-based rewards - no learned reward model.
Implements the Fair-GRPO-RLVR multi-objective reinforcement learning framework.

R_total = 0.4*Utility + 0.4*Fairness + 0.2*Safety
"""

from __future__ import annotations
import structlog
from dataclasses import dataclass, field
from typing import List
from .constants import (GRADER_SCORE_MAX, GRADER_SCORE_MIN, MAX_DAYS)
from .state import CityState, ZoneState
from .tasks import ScenarioConfig

logger = structlog.get_logger(__name__)


def compute_exec_reward(zones: List[ZoneState]) -> float:
    """Utility: Mean service level [0, 1]."""
    if not zones:
        return 0.0
    services = [z.service for z in zones]
    return float(sum(services) / len(services))


def compute_fairness_reward(zones: List[ZoneState]) -> float:
    """
    Equity: 1 - Mean Absolute Deviation.
    Higher value means more equitable distribution of services.
    """
    if not zones:
        return 0.0
    services = [z.service for z in zones]
    mean_svc = sum(services) / len(services)
    if not services: return 0.0
    
    disparity = sum(abs(s - mean_svc) for s in services) / len(services)
    # Fairness index in [0, 1]
    return float(max(0.0, 1.0 - disparity))


def compute_safety_reward(violations: List[str]) -> float:
    """Safety: Normalized violation count [0, 1]."""
    return float(max(0.0, 1.0 - len(violations) / 10.0))


def compute_analysis_reward(chosen_zones: List[int], zones: List[ZoneState]) -> float:
    """Partial reward for correctly identifying critical zones."""
    if not zones or not chosen_zones:
        return 0.0
    k = max(1, len(zones) // 2)
    ranked = sorted(range(len(zones)),
                    key=lambda i: zones[i].damage * zones[i].vulnerable_ratio, reverse=True)
    top_k = set(ranked[:k])
    return float(len(top_k & set(chosen_zones)) / k)


@dataclass
class RewardComponents:
    """Named reward components for a single step."""
    R_exec: float = 0.0
    R_fair: float = 0.0
    R_safe: float = 0.0
    R_analysis: float = 0.0
    R_total: float = 0.0
    violations: List[str] = field(default_factory=list)
    feedback: str = ""

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in {"R_exec": self.R_exec, "R_fair": self.R_fair,
                             "R_safe": self.R_safe, "R_total": self.R_total,
                             "violations": self.violations}.items()}


class RewardEngine:
    """Stateful reward calculator for a single episode."""

    def __init__(self, task: ScenarioConfig) -> None:
        self._task = task
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._action_history: List[str] = []

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    def compute_analysis_step(self, chosen_zones: List[int], city: CityState) -> RewardComponents:
        self._step_count += 1
        R_analysis = compute_analysis_reward(chosen_zones, city.zones)
        # Analysis provides a small progress signal
        R_total = 0.05 * R_analysis 
        self._cumulative_reward += R_total
        return RewardComponents(
            R_analysis=R_analysis, R_total=R_total,
            feedback=f"Analysis: {R_total:+.3f} ({int(R_analysis * max(1, len(city.zones)//2))}"
                     f"/{max(1, len(city.zones)//2)} critical zones correct)")

    def compute_execute_step(self, city: CityState, violations: List[str]) -> RewardComponents:
        """Main dense reward after execute step using Fair-GRPO-RLVR formula."""
        self._step_count += 1

        utility = compute_exec_reward(city.zones)
        fairness = compute_fairness_reward(city.zones)
        safety = compute_safety_reward(violations)

        # Truth Formula: 0.4*Utility + 0.4*Fairness + 0.2*Safety
        R_total = 0.4 * utility + 0.4 * fairness + 0.2 * safety
        R_total = float(max(0.0, min(1.0, R_total)))
        
        # Note: In interactive mode, we track cumulative, but train.ipynb uses final state.
        self._cumulative_reward += R_total

        feedback = (f"Utility={utility:.3f} | Fairness={fairness:.3f} | Safety={safety:.3f} → R_step={R_total:.3f}")
        if violations:
            feedback += f" | Violations: {violations}"

        return RewardComponents(R_exec=utility, R_fair=fairness, R_safe=safety,
                                R_total=R_total, violations=violations, feedback=feedback)

    def compute_submit_reward(self, city: CityState) -> RewardComponents:
        """Final submission reward (matches Truth Formula)."""
        self._step_count += 1
        utility = compute_exec_reward(city.zones)
        fairness = compute_fairness_reward(city.zones)
        safety = compute_safety_reward([]) # Assume no new violations on submit
        
        terminal = 0.4 * utility + 0.4 * fairness + 0.2 * safety
        terminal = float(max(0.0, min(1.0, terminal)))
        self._cumulative_reward += terminal
        
        return RewardComponents(
            R_fair=fairness, R_exec=utility, R_total=terminal,
            feedback=f"Terminal Score={terminal:.3f} (Utility={utility:.3f}, Fairness={fairness:.3f})")

    def get_final_grader_score(self, city: CityState) -> float:
        """Normalised score in (GRADER_SCORE_MIN, GRADER_SCORE_MAX)."""
        utility = compute_exec_reward(city.zones)
        fairness = compute_fairness_reward(city.zones)
        safety = compute_safety_reward([])
        
        normalised = 0.4 * utility + 0.4 * fairness + 0.2 * safety
        return round(float(max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, normalised))), 4)
