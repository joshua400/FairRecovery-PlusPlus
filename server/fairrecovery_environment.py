"""
FairRecovery++ — Core Environment.

Updated with critical bug fixes from bug_fixes.py.
"""

from __future__ import annotations
import uuid
from typing import Any, Optional, List
import structlog

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from fairrecovery_env.constants import (
    MAX_STEPS_PER_EPISODE, MAX_DAYS, TaskID, ActionType, ResourceType,
    COST_MEDICAL, COST_WATER, COST_POWER
)
from fairrecovery_env.models import (
    FairRecoveryAction, FairRecoveryObservation, FairRecoveryState, ZoneState
)
from fairrecovery_env.rewards import RewardEngine
from fairrecovery_env.tasks import get_task, TaskDefinition
from fairrecovery_env.rubrics import CompositeRubric

logger = structlog.get_logger(__name__)

class FairRecoveryEnvironment(Environment):
    """OpenEnv environment for fair disaster recovery."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[FairRecoveryState] = None
        self._task: Optional[TaskDefinition] = None
        self._reward_engine: Optional[RewardEngine] = None
        self._rubrics = CompositeRubric()
        self._action_history: list[str] = []
        self._pending_violations: List[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FairRecoveryObservation:
        """Reset the environment."""
        resolved_task_id = TaskID(task_id) if task_id else TaskID.FLOOD_EASY
        self._task = get_task(resolved_task_id)
        self._reward_engine = RewardEngine(self._task)
        self._rubrics.reset()
        self._action_history = []
        self._pending_violations = []

        ep_id = episode_id or str(uuid.uuid4())
        zones = [ZoneState(**z.model_dump()) for z in self._task.initial_zones]
        
        self._state = FairRecoveryState(
            episode_id=ep_id,
            step_count=0,
            day=1,
            budget_remaining=self._task.budget_limit,
            zones=zones,
            task_id=resolved_task_id,
            difficulty=self._task.difficulty,
        )

        return self._build_observation("Environment reset.")

    def step(self, action: Action, **kwargs: Any) -> FairRecoveryObservation:
        """Execute a recovery step."""
        if self._state is None or self._reward_engine is None:
            return FairRecoveryObservation(done=True, reward=0.0, step_feedback="Reset first.")

        try:
            if isinstance(action, FairRecoveryAction):
                typed_action = action
            elif isinstance(action, dict):
                typed_action = FairRecoveryAction(**action)
            else:
                typed_action = FairRecoveryAction(**action.model_dump())
        except Exception as e:
            typed_action = FairRecoveryAction(action_type=ActionType.NOOP, reasoning=str(e))

        self._state.step_count += 1
        self._action_history.append(typed_action.action_type.value)
        
        reward = 0.0
        feedback = ""

        if typed_action.action_type == ActionType.EXECUTE:
            allocated_ids = frozenset()
            if typed_action.allocations:
                allocated_ids = frozenset(a.zone for a in typed_action.allocations)
            
            self._execute_phase(typed_action)
            
            components = self._reward_engine.compute_execute_step(
                state=self._state,
                violations=self._pending_violations,
                allocated_zone_ids=allocated_ids
            )
            reward = components.R_total
            feedback = components.feedback
            self._state.day += 1
            self._pending_violations = []
            
        else:
            if typed_action.action_type == ActionType.ALLOCATE:
                self._allocate_phase(typed_action)
            reward, feedback = self._reward_engine.compute_reward(typed_action, self._state)

        self._state.cumulative_reward = self._reward_engine.cumulative_reward

        is_done = (
            typed_action.action_type == ActionType.SUBMIT or
            self._state.day > MAX_DAYS or
            self._state.step_count >= MAX_STEPS_PER_EPISODE
        )
        self._state.is_done = is_done

        obs = self._build_observation(feedback)
        
        rubric_score = self._rubrics.forward(typed_action, obs)
        if rubric_score != 0.0:
            self._state.cumulative_reward += rubric_score
            obs.cumulative_reward = self._state.cumulative_reward
            obs.reward = round(float(reward + rubric_score), 4)
        else:
            obs.reward = round(float(reward), 4)

        return obs

    def _allocate_phase(self, action: FairRecoveryAction):
        if not action.allocations: return
        for alloc in action.allocations:
            try:
                zone_id = int(alloc.zone)
                cost = {
                    ResourceType.MEDICAL: COST_MEDICAL,
                    ResourceType.WATER: COST_WATER,
                    ResourceType.POWER: COST_POWER
                }.get(alloc.resource, 0.0)

                if self._state.budget_remaining >= cost and 0 <= zone_id < len(self._state.zones):
                    self._state.budget_remaining -= cost
                    zone = self._state.zones[zone_id]
                    zone.damage = max(0.0, zone.damage - 0.05)
                    zone.service = min(1.0, zone.service + 0.1) # FIX: service_level -> service
                else:
                    self._state.violations_total += 1
                    self._pending_violations.append(f"invalid_allocation:{zone_id}")
            except:
                self._pending_violations.append("malformed_allocation")

    def _execute_phase(self, action: FairRecoveryAction):
        for zone in self._state.zones:
            if zone.service < 0.2: # FIX: service_level -> service
                zone.damage = min(1.0, zone.damage + 0.02)
            zone.service = max(0.0, zone.service - 0.05) # FIX: service_level -> service

    def _build_observation(self, feedback: str) -> FairRecoveryObservation:
        services = [z.service for z in self._state.zones] # FIX: service_level -> service
        avg_svc = sum(services) / len(services)
        mad = sum(abs(s - avg_svc) for s in services) / len(services)
        equity = 1.0 - mad

        grader_score = self._reward_engine.get_final_score(self._state) if self._state.is_done else None

        return FairRecoveryObservation(
            done=self._state.is_done,
            reward=0.0, 
            day=self._state.day,
            budget_left=self._state.budget_remaining,
            zones=self._state.zones,
            fairness_score=equity,
            step_stage="dynamic",
            steps_remaining=MAX_STEPS_PER_EPISODE - self._state.step_count,
            cumulative_reward=self._state.cumulative_reward,
            action_history=list(self._action_history),
            grader_score=grader_score,
            step_feedback=feedback,
            metadata={"grader_score": grader_score}
        )

    def state(self) -> FairRecoveryState:
        return self._state
