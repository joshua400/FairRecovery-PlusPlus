"""
FairRecovery — Core Environment Implementation.

Implements the OpenEnv Environment base class with step(), reset(), state().
Mirrors hallucination_environment.py structure from the reference gym exactly.

Design:
  • Multi-step protocol: analyze → allocate → execute → (repeat) → submit
  • Dense rewards at every step (not just terminal)
  • Per-component reward logging (R_exec, R_fair, R_safe) for training transparency
  • Composable rubrics wired via RFC 004 pattern
  • Safety shield validates before state mutation
  • Timeout enforcement via MAX_STEPS_SAFETY_CAP
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Environment:  # type: ignore
        pass
    Action = object  # type: ignore
    Observation = object  # type: ignore
    State = object  # type: ignore

from fairrecovery_env.constants import (
    ActionType,
    MAX_DAYS,
    MAX_STEPS_SAFETY_CAP,
    PENALTY_INVALID_ACTION,
    PENALTY_WRONG_STAGE,
)
from fairrecovery_env.models import (
    AllocationItem,
    FairRecoveryAction,
    FairRecoveryObservation,
    FairRecoveryState,
    ZoneObservation,
)
from fairrecovery_env.rewards import (
    RewardEngine,
    compute_fairness_reward,
)
from fairrecovery_env.rubrics import CompositeRubric
from fairrecovery_env.shield import validate, check_timeout
from fairrecovery_env.state import CityState
from fairrecovery_env.tasks import ScenarioConfig, get_task

logger = structlog.get_logger(__name__)


class FairRecoveryEnvironment(Environment):
    """
    Post-disaster city recovery RL environment.

    Primary Theme:   3.1 — Real-World Professional Tasks
    Secondary Theme: 2   — Long-Horizon Planning

    The agent must:
      1. Analyse zone status (damage, service, vulnerability)
      2. Queue resource allocations respecting budget constraints
      3. Execute allocations and receive dense rewards
      4. Repeat for MAX_DAYS days, then submit for final score

    Rewards balance:
      R_exec — service improvement (utility)
      R_fair — service parity between vulnerable and non-vulnerable zones
      R_safe — constraint satisfaction (no invalid actions or budget overflows)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._city: Optional[CityState]         = None
        self._task: Optional[ScenarioConfig]    = None
        self._reward_engine: Optional[RewardEngine] = None
        self._rubrics: CompositeRubric          = CompositeRubric()
        self._episode_id: Optional[str]         = None
        self._step_count: int                   = 0
        self._action_history: list[str]         = []
        self._current_difficulty: str           = "medium"

    # ──────────────────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "medium",
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FairRecoveryObservation:
        """
        Reset environment. Returns initial observation.

        Args:
            seed:       Unused (scenarios are deterministic).
            difficulty: 'easy' | 'medium' | 'hard'
            episode_id: Optional custom episode ID.

        Returns:
            Initial FairRecoveryObservation with full zone state.
        """
        self._task              = get_task(difficulty)
        self._current_difficulty= difficulty
        self._city              = CityState(self._task.__dict__)
        self._reward_engine     = RewardEngine(self._task)
        self._episode_id        = episode_id or str(uuid.uuid4())
        self._step_count        = 0
        self._action_history    = []
        self._city.step_stage   = "analyze"

        # Rubrics reset and initialise baseline fairness
        self._rubrics.reset()
        initial_fairness = compute_fairness_reward(self._city.zones)
        self._rubrics.fairness.set_initial_fairness(initial_fairness)

        logger.info(
            "environment_reset",
            episode_id=self._episode_id,
            difficulty=difficulty,
            num_zones=len(self._city.zones),
            initial_budget=self._city.budget_left,
            initial_fairness=round(initial_fairness, 4),
        )

        obs = self._build_observation(reward=0.0, done=False, r_exec=0.0, r_fair=0.0, r_safe=0.0)
        obs.step_feedback = (
            f"Episode started. Difficulty: {difficulty}. "
            f"Zones: {len(self._city.zones)}. Budget: {self._city.budget_left}. "
            f"{self._task.hint}"
        )
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────────
    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FairRecoveryObservation:
        """
        Execute one agent action. Returns updated observation with reward.

        Args:
            action:    FairRecoveryAction from the agent.
            timeout_s: Unused (env is synchronous).

        Returns:
            FairRecoveryObservation with reward components and feedback.
        """
        if self._city is None or self._reward_engine is None:
            return self._error_obs("Call reset() before step().")

        # Force-terminate on timeout
        if check_timeout(self._step_count):
            return self._build_observation(
                reward=PENALTY_INVALID_ACTION,
                done=True,
                r_exec=0.0, r_fair=0.0, r_safe=-0.5,
                feedback="Episode terminated: safety step cap exceeded.",
            )

        # Parse action
        try:
            if isinstance(action, FairRecoveryAction):
                typed_action = action
            elif isinstance(action, dict):
                typed_action = FairRecoveryAction(**action)
            else:
                typed_action = FairRecoveryAction(**action.model_dump())
        except Exception as exc:
            logger.warning("malformed_action", error=str(exc))
            typed_action = FairRecoveryAction(action_type="noop", reasoning=f"Parse error: {exc}")

        self._step_count += 1
        city = self._city
        action_type = str(typed_action.action_type).replace("ActionType.", "").lower()

        # Retrieve allocations for shield
        alloc_dicts = None
        if typed_action.allocations:
            alloc_dicts = [a.model_dump() for a in typed_action.allocations]

        # Shield validation
        is_valid, violations = validate(
            action_type=action_type,
            current_stage=city.step_stage,
            step_count=self._step_count,
            city=city,
            allocations=alloc_dicts,
        )

        # Dispatch by action type
        reward   = 0.0
        r_exec   = 0.0
        r_fair   = 0.0
        r_safe   = 0.0
        done     = False
        feedback = ""

        if action_type == "analyze":
            components = self._reward_engine.compute_analysis_step(
                chosen_zones=typed_action.critical_zones or [],
                city=city,
            )
            reward   = components.R_total
            r_exec   = components.R_analysis
            feedback = components.feedback
            city.record(f"analyzed zones={typed_action.critical_zones} | {typed_action.reasoning or ''}")
            city.step_stage = "allocate"

        elif action_type == "allocate":
            # Queue allocations for execute step
            city.pending_allocations = alloc_dicts or []
            city.record(f"queued {len(city.pending_allocations)} allocations")
            city.step_stage = "execute"
            feedback = (
                f"Queued {len(city.pending_allocations)} allocation(s). "
                f"Budget remaining: {city.budget_left}. Call execute next."
            )

        elif action_type == "execute":
            city.snapshot_services()
            exec_violations = city.apply_allocations()
            all_violations  = violations + exec_violations

            components = self._reward_engine.compute_execute_step(
                city=city,
                violations=all_violations,
            )
            reward   = components.R_total
            r_exec   = components.R_exec
            r_fair   = components.R_fair
            r_safe   = components.R_safe
            feedback = components.feedback
            city.record(f"executed | {feedback}")
            city.step_stage = "analyze"

            # Episode termination check
            if city.day >= MAX_DAYS or city.budget_left <= 0:
                done = True

        elif action_type == "submit":
            components = self._reward_engine.compute_submit_reward(city=city)
            reward   = components.R_total
            r_fair   = components.R_fair
            feedback = components.feedback
            done     = True

        elif action_type == "noop":
            reward   = -0.02  # mild penalty for wasting steps
            feedback = "No action taken (noop). This wastes a step."

        else:
            # Unknown action type — shield should have caught this
            reward   = PENALTY_INVALID_ACTION
            feedback = f"Unknown action_type '{action_type}'. Valid: analyze|allocate|execute|submit|noop."
            violations.append(f"unknown_action_type:{action_type}")

        # Stage violations add safety penalty
        if violations and not is_valid:
            reward  += PENALTY_WRONG_STAGE
            feedback += f" | Stage violation: {violations}"

        # Rubric scoring (RFC 004 pattern)
        obs = self._build_observation(
            reward=reward,
            done=done,
            r_exec=r_exec,
            r_fair=r_fair,
            r_safe=r_safe,
            feedback=feedback,
        )
        rubric_score = self._rubrics.forward(typed_action, obs)
        if rubric_score != 0.0:
            obs.cumulative_reward += rubric_score

        # Action history
        action_summary = (
            f"Step {self._step_count}: {action_type} "
            f"(reward={reward:+.3f})"
        )
        self._action_history.append(action_summary)
        obs.action_history = list(self._action_history[-8:])

        # Final grader score
        if done:
            obs.grader_score = self._reward_engine.get_final_grader_score(city)

        logger.info(
            "step_executed",
            episode_id=self._episode_id,
            step=self._step_count,
            action_type=action_type,
            stage=city.step_stage,
            day=city.day,
            reward=round(reward, 4),
            cumulative=round(self._reward_engine.cumulative_reward, 4),
            done=done,
            r_exec=round(r_exec, 4),
            r_fair=round(r_fair, 4),
            r_safe=round(r_safe, 4),
        )

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # state property
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def state(self) -> FairRecoveryState:
        """Internal state exposed via GET /state."""
        if self._city is None:
            return FairRecoveryState()

        city = self._city
        zones_obs = [
            ZoneObservation(
                zone_id=z.zone_id,
                damage=round(z.damage, 3),
                service=round(z.service, 3),
                vulnerable_ratio=round(z.vulnerable_ratio, 3),
            )
            for z in city.zones
        ]
        fairness = compute_fairness_reward(city.zones)

        return FairRecoveryState(
            episode_id=self._episode_id,
            difficulty=self._current_difficulty,
            day=city.day,
            budget_left=round(city.budget_left, 2),
            step_stage=city.step_stage,
            step_count=self._step_count,
            cumulative_reward=round(
                self._reward_engine.cumulative_reward if self._reward_engine else 0.0, 4
            ),
            fairness_score=round(fairness, 4),
            is_done=city.day >= MAX_DAYS or city.budget_left <= 0,
            violations_total=city.violations_total,
            zones=zones_obs,
        )

    def close(self) -> None:
        self._city          = None
        self._task          = None
        self._reward_engine = None
        self._action_history.clear()
        logger.info("environment_closed", episode_id=self._episode_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _build_observation(
        self,
        reward: float,
        done: bool,
        r_exec: float,
        r_fair: float,
        r_safe: float,
        feedback: str = "",
    ) -> FairRecoveryObservation:
        city = self._city

        zones_obs = [
            ZoneObservation(
                zone_id=z.zone_id,
                damage=round(z.damage, 3),
                service=round(z.service, 3),
                vulnerable_ratio=round(z.vulnerable_ratio, 3),
            )
            for z in city.zones
        ] if city else []

        fairness = compute_fairness_reward(city.zones) if city else 0.0
        steps_remaining = (
            (MAX_DAYS - city.day) * 3  # approx: 3 steps/day (analyze+allocate+execute)
            if city else 0
        )
        cumulative = (
            self._reward_engine.cumulative_reward
            if self._reward_engine else 0.0
        )

        return FairRecoveryObservation(
            zones=zones_obs,
            day=city.day if city else 0,
            budget_left=round(city.budget_left, 2) if city else 0.0,
            step_stage=city.step_stage if city else "analyze",
            fairness_score=round(fairness, 4),
            step_feedback=feedback,
            steps_remaining=max(0, steps_remaining),
            cumulative_reward=round(cumulative, 4),
            r_exec=round(r_exec, 4),
            r_fair=round(r_fair, 4),
            r_safe=round(r_safe, 4),
            done=done,
            reward=round(reward, 4),
        )

    def _error_obs(self, msg: str) -> FairRecoveryObservation:
        return FairRecoveryObservation(
            zones=[],
            day=0,
            budget_left=0.0,
            step_stage="analyze",
            fairness_score=0.0,
            step_feedback=f"Error: {msg}",
            done=False,
            reward=PENALTY_INVALID_ACTION,
        )
