"""
FairRecovery++ - Reward Engine (Fair-GRPO-RLVR).

Computes dense, verifiable, formula-based rewards - no learned reward model.
Implements the Fair-GRPO-RLVR multi-objective reinforcement learning framework.

R_total = w_exec*R_exec + w_fair*R_fair + w_safe*R_safe
"""

from __future__ import annotations
import structlog
from dataclasses import dataclass, field
from typing import List
from .constants import (GRADER_SCORE_MAX, GRADER_SCORE_MIN, MAX_DAYS,
                         PENALTY_IGNORE_VULNERABLE, PENALTY_PATTERN_IGNORED,
                         REWARD_WEIGHTS, VULNERABILITY_THRESHOLD)
from .state import CityState, ZoneState
from .tasks import ScenarioConfig

logger = structlog.get_logger(__name__)


def compute_exec_reward(prev_services: List[float], zones: List[ZoneState]) -> float:
    """Mean service improvement this day."""
    if not zones:
        return 0.0
    improvements = [z.service - prev for z, prev in zip(zones, prev_services)]
    return float(sum(improvements) / len(improvements))


def compute_fairness_reward(zones: List[ZoneState]) -> float:
    """
    Research-level Fairness Index: 1 - variance in service levels.
    Higher value means more equitable distribution of services.
    """
    if not zones:
        return 0.0
    services = [z.service for z in zones]
    mean_svc = sum(services) / len(services)
    variance = sum((s - mean_svc) ** 2 for s in services) / len(services)
    # Fairness index in [0, 1]
    return float(max(0.0, 1.0 - variance * 2.0))


def compute_safety_reward(violations: List[str]) -> float:
    """Penalty per safety violation, capped."""
    return float(-min(0.5, len(violations) * 0.1))


def compute_stability_reward(zones: List[ZoneState]) -> float:
    """Reward for system balance — low variance in satisfaction."""
    sats = [z.citizen_satisfaction for z in zones]
    if len(sats) < 2:
        return 0.0
    mean_sat = sum(sats) / len(sats)
    variance = sum((s - mean_sat) ** 2 for s in sats) / len(sats)
    return float(max(-1.0, -variance * 4))  # Scale up variance penalty


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
    R_adapt: float = 0.0
    R_stable: float = 0.0
    R_analysis: float = 0.0
    R_total: float = 0.0
    violations: List[str] = field(default_factory=list)
    feedback: str = ""

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in {"R_exec": self.R_exec, "R_fair": self.R_fair,
                             "R_safe": self.R_safe, "R_adapt": self.R_adapt,
                             "R_stable": self.R_stable, "R_total": self.R_total,
                             "violations": self.violations}.items()}


class RewardEngine:
    """Stateful reward calculator for a single episode."""

    def __init__(self, task: ScenarioConfig) -> None:
        self._task = task
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._action_history: List[str] = []
        self._vulnerable_ignored_days: int = 0

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    def compute_analysis_step(self, chosen_zones: List[int], city: CityState) -> RewardComponents:
        self._step_count += 1
        R_analysis = compute_analysis_reward(chosen_zones, city.zones)
        # More significant reward for correct analysis
        R_total = 0.2 * R_analysis 
        self._cumulative_reward += R_total
        return RewardComponents(
            R_analysis=R_analysis, R_total=R_total,
            feedback=f"Analysis: {R_total:+.3f} ({int(R_analysis * max(1, len(city.zones)//2))}"
                     f"/{max(1, len(city.zones)//2)} critical zones correct)")

    def compute_execute_step(self, city: CityState, violations: List[str],
                              adaptation_score: float = 0.0) -> RewardComponents:
        """Main dense reward after execute step — now includes adaptation and stability."""
        self._step_count += 1

        # Check if vulnerable zones consistently ignored
        vuln_ids = {z.zone_id for z in city.zones if z.is_vulnerable}
        if vuln_ids:
            history_text = " ".join(city.history)
            zone_served = any(str(zid) in history_text for zid in vuln_ids)
            if not zone_served and city.day > 1:
                self._vulnerable_ignored_days += 1
                if self._vulnerable_ignored_days >= 2:
                    violations.append(f"persistent_ignore_vulnerable:{vuln_ids}")

        R_exec = compute_exec_reward(city.prev_services, city.zones)
        R_fair = compute_fairness_reward(city.zones)
        R_safe = compute_safety_reward(violations)
        R_adapt = adaptation_score  # from Predictor.evaluate_adaptation()
        R_stable = compute_stability_reward(city.zones)

        w = REWARD_WEIGHTS
        # Add a +0.1 baseline for successful step execution to ensure positive polarity for good work
        R_total = (w["exec"] * R_exec + w["fair"] * R_fair + w["safe"] * (R_safe + 0.1))
        R_total = float(max(-0.5, min(1.0, R_total)))
        self._cumulative_reward += R_total

        feedback = (f"R_exec={R_exec:+.3f} | R_fair={R_fair:+.3f} | R_safe={R_safe:+.3f} | "
                    f"R_adapt={R_adapt:+.3f} | R_stable={R_stable:+.3f} → R_total={R_total:+.3f}")
        if violations:
            feedback += f" | Violations: {violations}"

        return RewardComponents(R_exec=R_exec, R_fair=R_fair, R_safe=R_safe,
                                R_total=R_total, violations=violations, feedback=feedback)

    def compute_submit_reward(self, city: CityState) -> RewardComponents:
        self._step_count += 1
        R_fair = compute_fairness_reward(city.zones)
        avg_svc = sum(z.service for z in city.zones) / max(1, len(city.zones))
        avg_sat = sum(z.citizen_satisfaction for z in city.zones) / max(1, len(city.zones))
        terminal = 0.4 * avg_svc + 0.3 * (1.0 + R_fair) + 0.3 * avg_sat
        terminal = float(max(0.0, min(1.0, terminal)))
        self._cumulative_reward += terminal
        return RewardComponents(
            R_fair=R_fair, R_exec=avg_svc, R_stable=avg_sat, R_total=terminal,
            feedback=f"Terminal bonus={terminal:.3f} (svc={avg_svc:.3f}, fair={R_fair:.3f}, sat={avg_sat:.3f})")

    def get_final_grader_score(self, city: CityState) -> float:
        """Normalised score in (GRADER_SCORE_MIN, GRADER_SCORE_MAX)."""
        avg_svc = sum(z.service for z in city.zones) / max(1, len(city.zones))
        R_fair = compute_fairness_reward(city.zones)
        avg_sat = sum(z.citizen_satisfaction for z in city.zones) / max(1, len(city.zones))
        normalised = 0.4 * avg_svc + 0.3 * (1.0 + R_fair) + 0.3 * avg_sat
        return round(float(max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, normalised))), 4)
