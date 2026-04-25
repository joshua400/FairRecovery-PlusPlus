"""
FairRecovery++ — Scenario Definitions.

Three scenarios of increasing difficulty with fairness traps.
"""

from __future__ import annotations
from dataclasses import dataclass
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
    hint: str = ""


TASKS: Dict[str, ScenarioConfig] = {
    "easy": ScenarioConfig(
        task_id="easy_3zone", difficulty=Difficulty.EASY,
        description="3-zone post-flood scenario. One zone has moderate damage and high vulnerability.",
        initial_budget=80.0,
        hint="Focus on the zone with highest vulnerability × damage.",
        zones=[
            {"zone_id": 0, "damage": 0.30, "service": 0.70, "vulnerable_ratio": 0.15},
            {"zone_id": 1, "damage": 0.80, "service": 0.20, "vulnerable_ratio": 0.85},
            {"zone_id": 2, "damage": 0.20, "service": 0.80, "vulnerable_ratio": 0.10},
        ],
    ),
    "medium": ScenarioConfig(
        task_id="medium_5zone", difficulty=Difficulty.MEDIUM,
        description="5-zone earthquake scenario. Multiple zones damaged. Budget is constrained.",
        initial_budget=60.0,
        hint="Budget is limited. Prioritise zones with high damage AND vulnerability.",
        zones=[
            {"zone_id": 0, "damage": 0.40, "service": 0.60, "vulnerable_ratio": 0.30},
            {"zone_id": 1, "damage": 0.70, "service": 0.30, "vulnerable_ratio": 0.75},
            {"zone_id": 2, "damage": 0.50, "service": 0.50, "vulnerable_ratio": 0.50},
            {"zone_id": 3, "damage": 0.65, "service": 0.35, "vulnerable_ratio": 0.65},
            {"zone_id": 4, "damage": 0.15, "service": 0.85, "vulnerable_ratio": 0.20},
        ],
    ),
    "hard": ScenarioConfig(
        task_id="hard_5zone_fairness_trap", difficulty=Difficulty.HARD,
        description=(
            "5-zone hurricane — FAIRNESS TRAP + ADVERSARIAL AGENTS. "
            "Zone 0 is wealthy (low vulnerability). Zone 4 is an informal settlement "
            "(highest vulnerability × damage). Multi-agent dynamics add disruptions."
        ),
        initial_budget=45.0,
        hint=(
            "FAIRNESS TRAP: Zone 0 looks easy but has low vulnerability. "
            "Zone 4 has highest damage AND vulnerability — prioritise it. "
            "Watch for adversarial disruptions and adapt your strategy."
        ),
        zones=[
            {"zone_id": 0, "damage": 0.35, "service": 0.65, "vulnerable_ratio": 0.08},
            {"zone_id": 1, "damage": 0.50, "service": 0.50, "vulnerable_ratio": 0.40},
            {"zone_id": 2, "damage": 0.60, "service": 0.40, "vulnerable_ratio": 0.55},
            {"zone_id": 3, "damage": 0.72, "service": 0.28, "vulnerable_ratio": 0.72},
            {"zone_id": 4, "damage": 0.92, "service": 0.08, "vulnerable_ratio": 0.96},
        ],
    ),
}


def get_task(difficulty: str) -> ScenarioConfig:
    """Retrieve scenario config by difficulty string."""
    task = TASKS.get(difficulty)
    if task is None:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from: {list(TASKS.keys())}")
    return task
