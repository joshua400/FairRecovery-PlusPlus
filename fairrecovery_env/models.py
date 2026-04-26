"""
FairRecovery++ — Domain Models.

Strict Pydantic models for Actions, Observations, and internal State.
Updated to match bug_fixes.py requirements.
"""

from __future__ import annotations
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from .constants import ResourceType, ActionType, Difficulty, TaskID


class ZoneState(BaseModel):
    """Current state of a specific geographic zone."""
    zone_id: int
    damage: float = Field(..., ge=0.0, le=1.0)
    vulnerable_ratio: float = Field(..., ge=0.0, le=1.0)
    service: float = Field(0.0, ge=0.0, le=1.0) # Renamed from service_level
    history: List[float] = []


class ResourceAllocation(BaseModel):
    """Specific resource assignment to a zone."""
    resource: ResourceType
    zone: int
    amount: float = 1.0


class FairRecoveryAction(BaseModel):
    """Action taken by the agent."""
    action_type: ActionType
    critical_zones: Optional[List[int]] = None
    allocations: Optional[List[ResourceAllocation]] = None
    reasoning: Optional[str] = None


class FairRecoveryObservation(BaseModel):
    """Observation returned to the agent."""
    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: float = Field(default=0.0, description="Step reward received from the last action.")
    day: int
    budget_left: float
    zones: List[ZoneState]
    fairness_score: float
    step_stage: str
    steps_remaining: int
    cumulative_reward: float
    action_history: List[str]
    agent_events: List[str] = Field(default_factory=list, description="Events emitted this step.")
    grader_score: Optional[float] = None
    step_feedback: str = ""
    metadata: Dict[str, Any] = {}


class FairRecoveryState(BaseModel):
    """Internal state of the environment."""
    episode_id: str
    step_count: int = 0
    day: int = 1
    budget_remaining: float = 1.0
    zones: List[ZoneState]
    violations_total: int = 0
    is_done: bool = False
    task_id: TaskID = TaskID.FLOOD_EASY
    difficulty: Difficulty = Difficulty.EASY
    cumulative_reward: float = 0.0
