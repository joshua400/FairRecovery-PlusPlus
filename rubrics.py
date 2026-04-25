"""
FairRecovery — Composable Rubric System.

Implements OpenEnv RFC 004 rubric pattern.
Composable rubrics > monolithic scoring — each rubric scores one dimension.

FairnessRubric  — outcome: reduction in service disparity across episode
UtilityRubric   — outcome: average service level at end of episode
AnalysisRubric  — process: quality of zone identification at analyze steps

Wire them in the environment's __init__ and call rubric.forward(action, obs)
at the end of each step() call.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

try:
    from openenv.core.rubrics.base import Rubric as BaseRubric
except ImportError:
    # Fallback stub matching OpenEnv Rubric interface
    class BaseRubric:  # type: ignore
        def forward(self, action: Any, observation: Any) -> float:
            return 0.0
        def reset(self) -> None:
            pass

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fairness Rubric
# ──────────────────────────────────────────────────────────────────────────────
class FairnessRubric(BaseRubric):
    """
    Outcome rubric — scores reduction in service disparity across the episode.

    Only fires at episode end (when observation.done == True).
    Measures improvement in fairness_score from start to finish.

    Score is positive if the agent improved fairness over the episode,
    negative if fairness got worse.
    """

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight
        self._initial_fairness: Optional[float] = None

    def set_initial_fairness(self, score: float) -> None:
        """Call once at reset() with the baseline fairness score."""
        self._initial_fairness = score
        logger.debug("fairness_rubric_init", initial_fairness=score)

    def forward(self, action: Any, observation: Any) -> float:
        """
        Score fairness improvement. Only active at episode end.

        Args:
            action: The action that led to this observation (unused here).
            observation: FairRecoveryObservation with done and fairness_score.

        Returns:
            Weighted fairness improvement score, or 0.0 if not done.
        """
        if not getattr(observation, "done", False):
            return 0.0

        final_fairness = getattr(observation, "fairness_score", 0.0)

        if self._initial_fairness is None:
            score = self.weight * final_fairness
        else:
            improvement = final_fairness - self._initial_fairness
            score = self.weight * improvement

        logger.info(
            "fairness_rubric_score",
            initial=self._initial_fairness,
            final=final_fairness,
            score=round(score, 4),
        )
        return float(score)

    def reset(self) -> None:
        self._initial_fairness = None


# ──────────────────────────────────────────────────────────────────────────────
# Utility Rubric
# ──────────────────────────────────────────────────────────────────────────────
class UtilityRubric(BaseRubric):
    """
    Outcome rubric — scores average service level at end of episode.

    Rewards agents that restore essential services to as many zones as possible.
    Only fires at episode end.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def forward(self, action: Any, observation: Any) -> float:
        """
        Score average service restoration. Only active at episode end.

        Args:
            action: Unused.
            observation: FairRecoveryObservation with done and zones.

        Returns:
            Weighted average service level, or 0.0 if not done.
        """
        if not getattr(observation, "done", False):
            return 0.0

        zones = getattr(observation, "zones", [])
        if not zones:
            return 0.0

        avg_service = sum(getattr(z, "service", 0.0) for z in zones) / len(zones)
        score = self.weight * avg_service

        logger.info(
            "utility_rubric_score",
            avg_service=round(avg_service, 4),
            score=round(score, 4),
        )
        return float(score)

    def reset(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Analysis Quality Rubric (process rubric — fires at each analyze step)
# ──────────────────────────────────────────────────────────────────────────────
class AnalysisRubric(BaseRubric):
    """
    Process rubric — scores the quality of zone identification at analyze steps.

    Fires at every analyze step (not just episode end), providing dense
    intermediate signal on whether the agent correctly identified critical zones.
    """

    def __init__(self, weight: float = 0.1) -> None:
        self.weight = weight
        self._scores: list[float] = []

    def forward(self, action: Any, observation: Any) -> float:
        """
        Score analysis quality for analyze actions.

        Args:
            action: FairRecoveryAction — only active for action_type='analyze'.
            observation: FairRecoveryObservation with zones.

        Returns:
            Weighted analysis quality score, or 0.0 for non-analyze actions.
        """
        action_type = getattr(action, "action_type", None)
        if str(action_type) not in ("analyze", "ActionType.ANALYZE"):
            return 0.0

        # We can't run the full compute_analysis_reward here without importing
        # state — instead we check if critical_zones includes the observation's
        # highest-priority zones by inspection.
        zones = getattr(observation, "zones", [])
        critical_zones = getattr(action, "critical_zones", None) or []

        if not zones or not critical_zones:
            return 0.0

        k = max(1, len(zones) // 2)
        ranked = sorted(
            range(len(zones)),
            key=lambda i: (
                getattr(zones[i], "damage", 0.0) *
                getattr(zones[i], "vulnerable_ratio", 0.0)
            ),
            reverse=True,
        )
        top_k   = set(ranked[:k])
        chosen  = set(critical_zones)
        overlap = len(top_k & chosen)
        quality = overlap / k

        score = self.weight * quality
        self._scores.append(score)

        logger.debug(
            "analysis_rubric_score",
            chosen=list(chosen),
            top_k=list(top_k),
            quality=round(quality, 4),
            score=round(score, 4),
        )
        return float(score)

    def reset(self) -> None:
        self._scores.clear()

    @property
    def mean_analysis_quality(self) -> float:
        """Average analysis quality across the episode."""
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)


# ──────────────────────────────────────────────────────────────────────────────
# Composite Rubric (convenience wrapper — runs all rubrics, sums scores)
# ──────────────────────────────────────────────────────────────────────────────
class CompositeRubric:
    """Runs FairnessRubric + UtilityRubric + AnalysisRubric and sums their scores."""

    def __init__(self) -> None:
        self.fairness  = FairnessRubric(weight=0.5)
        self.utility   = UtilityRubric(weight=1.0)
        self.analysis  = AnalysisRubric(weight=0.1)

    def forward(self, action: Any, observation: Any) -> float:
        scores = [
            self.fairness.forward(action, observation),
            self.utility.forward(action, observation),
            self.analysis.forward(action, observation),
        ]
        return sum(scores)

    def reset(self) -> None:
        self.fairness.reset()
        self.utility.reset()
        self.analysis.reset()
