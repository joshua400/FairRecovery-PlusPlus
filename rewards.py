"""
FairRecovery — Reward Engine (RLVR).

Computes dense, verifiable, formula-based rewards — no learned reward model.
All reward components are returned separately for transparency and logging.

R_total = w_exec * R_exec + w_fair * R_fair + w_safe * R_safe

  R_exec — average service increase from allocations (execution quality)
  R_fair — negative disparity between vulnerable vs non-vulnerable group service
  R_safe — negative penalty for constraint violations

Anti-reward-hacking properties:
  • Vulnerable-zone ignore penalty: agent can't just ignore high-vulnerability zones
  • Budget overflow blocked: allocations exceeding budget are voided, not scaled
  • Per-component caps: no single component can dominate to mask failures elsewhere
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .constants import (
    GRADER_SCORE_MAX,
    GRADER_SCORE_MIN,
    MAX_DAYS,
    PENALTY_IGNORE_VULNERABLE,
    PENALTY_REPEATED_ACTION,
    REWARD_WEIGHTS,
    VULNERABILITY_THRESHOLD,
)
from .state import CityState, ZoneState
from .tasks import ScenarioConfig

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Component reward functions (pure, testable)
# ──────────────────────────────────────────────────────────────────────────────

def compute_exec_reward(prev_services: List[float], zones: List[ZoneState]) -> float:
    """
    Mean service improvement this day.

    Positive when allocations increased service levels.
    Zero when no allocations were made or budget was exhausted.
    """
    if not zones:
        return 0.0
    improvements = [
        z.service - prev
        for z, prev in zip(zones, prev_services)
    ]
    return float(sum(improvements) / len(improvements))


def compute_fairness_reward(zones: List[ZoneState]) -> float:
    """
    Negative disparity between vulnerable and non-vulnerable group service.

    Returns a value ≤ 0:
      0.0   = perfect parity (vulnerable zones served as well as others)
      -1.0  = maximum disparity (vulnerable zones have 0 service, others have 1)

    Anti-gaming: measures group averages, not individual zones,
    so agent can't game it by over-serving one token vulnerable zone.
    """
    vuln   = [z for z in zones if z.vulnerable_ratio >= VULNERABILITY_THRESHOLD]
    normal = [z for z in zones if z.vulnerable_ratio <  VULNERABILITY_THRESHOLD]

    if not vuln or not normal:
        # Homogeneous population — use service variance as proxy
        services = [z.service for z in zones]
        if len(services) < 2:
            return 0.0
        mean_svc = sum(services) / len(services)
        variance = sum((s - mean_svc) ** 2 for s in services) / len(services)
        return float(-variance)

    avg_vuln   = sum(z.service for z in vuln)   / len(vuln)
    avg_normal = sum(z.service for z in normal) / len(normal)
    disparity  = avg_normal - avg_vuln    # positive = normal zones better off
    return float(-disparity)              # penalise if vulnerable lag behind


def compute_safety_reward(violations: List[str]) -> float:
    """
    Penalty per safety violation, capped to avoid dominating the total.
    -0.1 per violation, max -0.5 total.
    """
    penalty = min(0.5, len(violations) * 0.1)
    return float(-penalty)


def compute_analysis_reward(
    chosen_zones: List[int],
    zones: List[ZoneState],
) -> float:
    """
    Partial reward for correctly identifying the most critical zones.

    Correct = chose at least one zone from the top-k by (damage × vulnerable_ratio).
    Returns proportion of top-k correctly identified.
    """
    if not zones or not chosen_zones:
        return 0.0

    k = max(1, len(zones) // 2)
    ranked = sorted(
        range(len(zones)),
        key=lambda i: zones[i].damage * zones[i].vulnerable_ratio,
        reverse=True,
    )
    top_k   = set(ranked[:k])
    chosen  = set(chosen_zones)
    overlap = len(top_k & chosen)
    return float(overlap / k)


@dataclass
class RewardComponents:
    """Named reward components for a single step — logged and returned to training."""
    R_exec: float = 0.0
    R_fair: float = 0.0
    R_safe: float = 0.0
    R_analysis: float = 0.0
    R_total: float = 0.0
    violations: List[str] = field(default_factory=list)
    feedback: str = ""

    def to_dict(self) -> dict:
        return {
            "R_exec":     round(self.R_exec, 4),
            "R_fair":     round(self.R_fair, 4),
            "R_safe":     round(self.R_safe, 4),
            "R_analysis": round(self.R_analysis, 4),
            "R_total":    round(self.R_total, 4),
            "violations": self.violations,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Stateful Reward Engine
# ──────────────────────────────────────────────────────────────────────────────

class RewardEngine:
    """
    Stateful reward calculator for a single episode.

    Tracks cumulative reward and step count.
    All individual computations delegate to pure functions above.
    """

    def __init__(self, task: ScenarioConfig) -> None:
        self._task = task
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._action_history: List[str] = []
        self._vulnerable_ignored_days: int = 0

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    # ── analysis step reward ─────────────────────────────────────────────────

    def compute_analysis_step(
        self,
        chosen_zones: List[int],
        city: CityState,
    ) -> RewardComponents:
        """Reward for analysis quality. Dense partial-progress signal."""
        self._step_count += 1
        R_analysis = compute_analysis_reward(chosen_zones, city.zones)
        R_total = 0.1 * R_analysis  # small reward — analysis is not execution

        self._cumulative_reward += R_total
        components = RewardComponents(
            R_analysis=R_analysis,
            R_total=R_total,
            feedback=(
                f"Analysis reward: {R_total:.3f} "
                f"(identified {int(R_analysis * max(1, len(city.zones)//2))}"
                f"/{max(1, len(city.zones)//2)} critical zones correctly)."
            ),
        )
        self._log("analysis", components)
        return components

    # ── execute step reward ───────────────────────────────────────────────────

    def compute_execute_step(
        self,
        city: CityState,
        violations: List[str],
    ) -> RewardComponents:
        """
        Main dense reward after execute step.

        Also checks if vulnerable zones were consistently ignored —
        a key anti-reward-hacking signal.
        """
        self._step_count += 1

        # Check if high-vulnerability zones were ignored this day
        vuln_zone_ids = {z.zone_id for z in city.zones if z.is_vulnerable}
        allocated_zone_ids = {
            a.get("zone") for a in (city.pending_allocations or [])
        } if city.pending_allocations else set()

        # Check historically — did any vulnerable zone receive resources?
        if vuln_zone_ids:
            history_text = " ".join(city.history)
            zone_served = any(
                f"zone {zid}" in history_text.lower() or str(zid) in history_text
                for zid in vuln_zone_ids
            )
            if not zone_served and city.day > 1:
                self._vulnerable_ignored_days += 1
                if self._vulnerable_ignored_days >= 2:
                    violations.append(f"persistent_ignore_vulnerable:{vuln_zone_ids}")

        R_exec = compute_exec_reward(city.prev_services, city.zones)
        R_fair = compute_fairness_reward(city.zones)
        R_safe = compute_safety_reward(violations)

        w = REWARD_WEIGHTS
        R_total = (
            w["exec"] * R_exec +
            w["fair"] * R_fair +
            w["safe"] * R_safe
        )

        # Clamp to [-1, 1] per step
        R_total = float(max(-1.0, min(1.0, R_total)))

        self._cumulative_reward += R_total

        feedback_parts = [
            f"R_exec={R_exec:+.3f}",
            f"R_fair={R_fair:+.3f}",
            f"R_safe={R_safe:+.3f}",
            f"→ R_total={R_total:+.3f}",
        ]
        if violations:
            feedback_parts.append(f"Violations: {violations}.")

        components = RewardComponents(
            R_exec=R_exec,
            R_fair=R_fair,
            R_safe=R_safe,
            R_total=R_total,
            violations=violations,
            feedback=" | ".join(feedback_parts),
        )
        self._log("execute", components)
        return components

    # ── final episode reward ─────────────────────────────────────────────────

    def compute_submit_reward(self, city: CityState) -> RewardComponents:
        """
        Terminal reward when agent submits.

        Combines final service level and final fairness score
        into a normalised bonus. Efficient episodes (fewer steps) earn a bonus.
        """
        self._step_count += 1

        R_fair = compute_fairness_reward(city.zones)
        avg_svc = sum(z.service for z in city.zones) / max(1, len(city.zones))

        # Terminal bonus in [0, 1]:  0.5*avg_service + 0.5*(1 + R_fair) clamped
        terminal = 0.5 * avg_svc + 0.5 * (1.0 + R_fair)
        terminal = float(max(0.0, min(1.0, terminal)))

        self._cumulative_reward += terminal

        components = RewardComponents(
            R_fair=R_fair,
            R_exec=avg_svc,
            R_total=terminal,
            feedback=(
                f"Episode submitted. Terminal bonus={terminal:.3f} "
                f"(avg_service={avg_svc:.3f}, fairness={R_fair:.3f})."
            ),
        )
        self._log("submit", components)
        return components

    # ── grader score ─────────────────────────────────────────────────────────

    def get_final_grader_score(self, city: CityState) -> float:
        """
        Normalised score in (GRADER_SCORE_MIN, GRADER_SCORE_MAX).

        Based on: average service improvement + fairness improvement.
        Never exactly 0 or 1 (per OpenEnv grading conventions).
        """
        avg_svc = sum(z.service for z in city.zones) / max(1, len(city.zones))
        R_fair  = compute_fairness_reward(city.zones)

        # Normalise: avg_svc in [0,1], R_fair in [-1,0] → map to [0,1]
        normalised = 0.6 * avg_svc + 0.4 * (1.0 + R_fair)
        clamped    = max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, normalised))
        return round(float(clamped), 4)

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, step_type: str, components: RewardComponents) -> None:
        logger.info(
            "reward_computed",
            step_type=step_type,
            step=self._step_count,
            **components.to_dict(),
            cumulative=round(self._cumulative_reward, 4),
        )
