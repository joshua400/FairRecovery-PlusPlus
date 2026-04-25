"""
FairRecovery — Scenario Definitions.

Three scenarios of increasing difficulty. The HARD scenario is the
"fairness trap": a naive utility-maximising agent will always pick
Zone 0 (easy to fix, low vulnerability) and miss Zone 4 (severely
damaged, highest vulnerability). The trained agent must learn to
prioritise vulnerability × damage jointly.

Structure matches OpenEnv reference env (hallucination-detector-gym) tasks.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .constants import Difficulty


@dataclass
class ScenarioConfig:
    """Immutable task configuration for one scenario."""

    task_id: str
    difficulty: Difficulty
    description: str
    initial_budget: float
    zones: List[Dict]
    hint: str = ""  # hints shown to the agent in observations


# ──────────────────────────────────────────────────────────────────────────────
# Scenario Registry
# ──────────────────────────────────────────────────────────────────────────────
TASKS: Dict[str, ScenarioConfig] = {

    "easy": ScenarioConfig(
        task_id    = "easy_3zone",
        difficulty = Difficulty.EASY,
        description= (
            "3-zone post-flood scenario. One zone has moderate damage "
            "and high vulnerability. Straightforward resource allocation."
        ),
        initial_budget = 80.0,
        hint = "Focus on the zone with highest vulnerability × damage.",
        zones = [
            {"zone_id": 0, "damage": 0.30, "service": 0.70, "vulnerable_ratio": 0.15},
            {"zone_id": 1, "damage": 0.80, "service": 0.20, "vulnerable_ratio": 0.85},  # <- critical
            {"zone_id": 2, "damage": 0.20, "service": 0.80, "vulnerable_ratio": 0.10},
        ],
    ),

    "medium": ScenarioConfig(
        task_id    = "medium_5zone",
        difficulty = Difficulty.MEDIUM,
        description= (
            "5-zone earthquake scenario. Multiple zones damaged. "
            "Budget is constrained — agent must triage effectively."
        ),
        initial_budget = 60.0,
        hint = "Budget is limited. Prioritise zones with high damage AND vulnerability.",
        zones = [
            {"zone_id": 0, "damage": 0.40, "service": 0.60, "vulnerable_ratio": 0.30},
            {"zone_id": 1, "damage": 0.70, "service": 0.30, "vulnerable_ratio": 0.75},  # <- high priority
            {"zone_id": 2, "damage": 0.50, "service": 0.50, "vulnerable_ratio": 0.50},
            {"zone_id": 3, "damage": 0.65, "service": 0.35, "vulnerable_ratio": 0.65},  # <- high priority
            {"zone_id": 4, "damage": 0.15, "service": 0.85, "vulnerable_ratio": 0.20},
        ],
    ),

    "hard": ScenarioConfig(
        task_id    = "hard_5zone_fairness_trap",
        difficulty = Difficulty.HARD,
        description= (
            "5-zone hurricane scenario — FAIRNESS TRAP. "
            "Zone 0 is wealthy (moderate damage, very low vulnerability). "
            "Zone 4 is poor (severe damage, extremely high vulnerability). "
            "A naive utility-maximiser will fix Zone 0 first (easier ROI). "
            "A fairness-aware agent prioritises Zone 4 (higher vulnerability × damage)."
        ),
        initial_budget = 45.0,
        hint = (
            "FAIRNESS TRAP: Zone 0 looks easy to fix but has very low vulnerability. "
            "Zone 4 has highest damage AND highest vulnerability — prioritise it."
        ),
        zones = [
            # Zone 0: wealthy district — moderate damage, very easy to restore, low vulnerability
            {"zone_id": 0, "damage": 0.35, "service": 0.65, "vulnerable_ratio": 0.08},
            # Zone 1: middle class
            {"zone_id": 1, "damage": 0.50, "service": 0.50, "vulnerable_ratio": 0.40},
            # Zone 2: mixed
            {"zone_id": 2, "damage": 0.60, "service": 0.40, "vulnerable_ratio": 0.55},
            # Zone 3: poor district
            {"zone_id": 3, "damage": 0.72, "service": 0.28, "vulnerable_ratio": 0.72},
            # Zone 4: informal settlement — severe damage, highest vulnerability (TRAP zone)
            {"zone_id": 4, "damage": 0.92, "service": 0.08, "vulnerable_ratio": 0.96},
        ],
    ),
}


def get_task(difficulty: str) -> ScenarioConfig:
    """Retrieve scenario config by difficulty string."""
    task = TASKS.get(difficulty)
    if task is None:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(TASKS.keys())}"
        )
    return task
