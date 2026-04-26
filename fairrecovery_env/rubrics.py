"""
FairRecovery++ — Composable Rubric System (RFC 004).
"""

from __future__ import annotations
from typing import Any, Optional
import structlog

try:
    from openenv.core.rubrics.base import Rubric as BaseRubric
except ImportError:
    class BaseRubric:
        def forward(self, action: Any, observation: Any) -> float: return 0.0
        def reset(self) -> None: pass

logger = structlog.get_logger(__name__)


class FairnessRubric(BaseRubric):
    """Outcome rubric — scores reduction in service disparity across episode."""
    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight
        self._initial_fairness: Optional[float] = None

    def set_initial_fairness(self, score: float) -> None:
        self._initial_fairness = score

    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False): return 0.0
        final = getattr(observation, "fairness_score", 0.0)
        if self._initial_fairness is None: return self.weight * final
        return float(self.weight * (final - self._initial_fairness))

    def reset(self) -> None: self._initial_fairness = None


class UtilityRubric(BaseRubric):
    """Outcome rubric — scores average service level at episode end."""
    def __init__(self, weight: float = 1.0) -> None: self.weight = weight
    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False): return 0.0
        zones = getattr(observation, "zones", [])
        if not zones: return 0.0
        # FIX: service_level -> service
        avg = sum(getattr(z, "service", 0.0) for z in zones) / len(zones)
        return float(self.weight * avg)
    def reset(self) -> None: pass


class AnalysisRubric(BaseRubric):
    """Process rubric — scores zone identification quality at analyze steps."""
    def __init__(self, weight: float = 0.1) -> None:
        self.weight = weight
        self._scores: list[float] = []

    def forward(self, action: Any, observation: Any) -> float:
        at = str(getattr(action, "action_type", ""))
        if at not in ("analyze", "ActionType.ANALYZE"): return 0.0
        zones = getattr(observation, "zones", [])
        critical = getattr(action, "critical_zones", None) or []
        if not zones or not critical: return 0.0
        k = max(1, len(zones) // 2)
        ranked = sorted(range(len(zones)),
                        key=lambda i: getattr(zones[i], "damage", 0.0) * getattr(zones[i], "vulnerable_ratio", 0.0),
                        reverse=True)
        quality = len(set(ranked[:k]) & set(critical)) / k
        score = self.weight * quality
        self._scores.append(score)
        return float(score)

    def reset(self) -> None: self._scores.clear()

    @property
    def mean_analysis_quality(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.0


class AdaptationRubric(BaseRubric):
    """Outcome rubric — scores how well agent adapted to multi-agent dynamics."""
    def __init__(self, weight: float = 0.3) -> None: self.weight = weight
    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False): return 0.0
        return float(self.weight * getattr(observation, "r_adapt", 0.0))
    def reset(self) -> None: pass


class CompositeRubric:
    """Runs all rubrics and sums their scores."""
    def __init__(self) -> None:
        self.fairness = FairnessRubric(weight=0.5)
        self.utility = UtilityRubric(weight=1.0)
        self.analysis = AnalysisRubric(weight=0.1)
        self.adaptation = AdaptationRubric(weight=0.3)

    def forward(self, action: Any, observation: Any) -> float:
        return sum([self.fairness.forward(action, observation),
                    self.utility.forward(action, observation),
                    self.analysis.forward(action, observation),
                    self.adaptation.forward(action, observation)])

    def reset(self) -> None:
        self.fairness.reset(); self.utility.reset()
        self.analysis.reset(); self.adaptation.reset()
