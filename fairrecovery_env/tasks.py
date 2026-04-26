"""
FairRecovery++ — Task Definitions.

Pre-configured disaster scenarios for training and evaluation.
"""

from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel
from .constants import TaskID, Difficulty, NUM_ZONES
from .models import ZoneState


class TaskDefinition(BaseModel):
    """Configuration for a specific disaster scenario."""
    task_id: TaskID
    difficulty: Difficulty
    title: str
    description: str
    initial_zones: List[ZoneState]
    budget_limit: float = 1.0


def get_task(task_id: TaskID) -> TaskDefinition:
    """Retrieve a pre-configured task definition."""
    
    if task_id == TaskID.FLOOD_EASY:
        return TaskDefinition(
            task_id=task_id,
            difficulty=Difficulty.EASY,
            title="Monsoon Flash Flood",
            description="Moderate damage in urban zones. Clear priorities.",
            initial_zones=[
                ZoneState(zone_id=0, damage=0.2, vulnerable_ratio=0.1),
                ZoneState(zone_id=1, damage=0.3, vulnerable_ratio=0.2),
                ZoneState(zone_id=2, damage=0.2, vulnerable_ratio=0.15),
                ZoneState(zone_id=3, damage=0.4, vulnerable_ratio=0.3),
                ZoneState(zone_id=4, damage=0.5, vulnerable_ratio=0.5),
            ]
        )
    
    elif task_id == TaskID.EARTHQUAKE_MEDIUM:
        return TaskDefinition(
            task_id=task_id,
            difficulty=Difficulty.MEDIUM,
            title="7.2 Magnitude Earthquake",
            description="Heavy damage across central districts. Power grid failure.",
            initial_zones=[
                ZoneState(zone_id=0, damage=0.4, vulnerable_ratio=0.1),
                ZoneState(zone_id=1, damage=0.6, vulnerable_ratio=0.4),
                ZoneState(zone_id=2, damage=0.5, vulnerable_ratio=0.3),
                ZoneState(zone_id=3, damage=0.7, vulnerable_ratio=0.6),
                ZoneState(zone_id=4, damage=0.8, vulnerable_ratio=0.8),
            ]
        )
    
    else:  # MULTI_DISASTER_HARD
        return TaskDefinition(
            task_id=TaskID.MULTI_DISASTER_HARD,
            difficulty=Difficulty.HARD,
            title="The Fairness Trap: Urban Cyclone",
            description="Zone 4 is critical but ignored by greedy planners.",
            initial_zones=[
                ZoneState(zone_id=0, damage=0.15, vulnerable_ratio=0.08),
                ZoneState(zone_id=1, damage=0.35, vulnerable_ratio=0.40),
                ZoneState(zone_id=2, damage=0.42, vulnerable_ratio=0.55),
                ZoneState(zone_id=3, damage=0.72, vulnerable_ratio=0.72),
                ZoneState(zone_id=4, damage=0.92, vulnerable_ratio=0.96), # The Fairness Trap
            ]
        )
