"""
FairRecovery++ — Core Environment Implementation.

Adaptive multi-agent OpenEnv environment for post-disaster recovery.
Implements Environment base class with step(), reset(), state().

Design:
  * Multi-step protocol: analyze -> allocate -> execute -> adapt -> submit
  * Multi-agent system: citizens, NGOs, adversaries interact each step
  * Behavior analysis: patterns extracted from interaction logs
  * Predictive engine: forecasts events for proactive planning
  * Dense rewards at every step with 5-component breakdown
  * Composable rubrics wired via RFC 004 pattern
  * Safety shield validates before state mutation

Themes Hit:
  * Theme 3.1 — Real-World Professional Tasks (core)
  * Theme 2   — Long-Horizon Planning (strong)
  * Theme 1   — Multi-Agent Interaction (added)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

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
    ActionType, MAX_DAYS, MAX_STEPS_SAFETY_CAP,
    MIN_STEPS, CURRICULUM_MAX_STEPS, EARLY_SUBMIT_PENALTY,
    FINAL_BONUS_WEIGHT_UTILITY, FINAL_BONUS_WEIGHT_FAIRNESS,
    PENALTY_INVALID_ACTION, PENALTY_WRONG_STAGE,
    PENALTY_PATTERN_IGNORED, PENALTY_ADVERSARIAL_FAILURE,
)
from fairrecovery_env.models import (
    AgentEvent, FairRecoveryAction, FairRecoveryObservation,
    FairRecoveryState, ZoneObservation,
)
from fairrecovery_env.rewards import RewardEngine, compute_fairness_reward
from fairrecovery_env.rubrics import CompositeRubric
from fairrecovery_env.shield import validate, check_timeout
from fairrecovery_env.state import CityState
from fairrecovery_env.tasks import ScenarioConfig, get_task
from fairrecovery_env.agents import MultiAgentManager
from fairrecovery_env.behavior_analyzer import BehaviorAnalyzer
from fairrecovery_env.predictor import Predictor, Prediction

logger = structlog.get_logger(__name__)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _signed_to_unit(value: float) -> float:
    # Convert [-1, 1] style component into [0, 1] for transparent dashboards.
    return _clip01((value + 1.0) / 2.0)


class FairRecoveryEnvironment(Environment):
    """
    Post-disaster city recovery RL environment with multi-agent dynamics.

    The agent must:
      1. Analyse zone status (damage, service, vulnerability, agent events)
      2. Queue resource allocations respecting budget constraints
      3. Execute allocations and receive dense rewards
      4. Adapt strategy based on predictions and agent behavior
      5. Repeat for MAX_DAYS days, then submit for final score

    Rewards balance 5 components:
      R_exec   — service improvement (utility)
      R_fair   — service parity between vulnerable/non-vulnerable zones
      R_adapt  — success against predicted events
      R_stable — system balance (citizen satisfaction variance)
      R_safe   — constraint satisfaction
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._city: Optional[CityState] = None
        self._task: Optional[ScenarioConfig] = None
        self._reward_engine: Optional[RewardEngine] = None
        self._rubrics: CompositeRubric = CompositeRubric()
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._action_history: list[str] = []
        self._current_difficulty: str = "medium"

        # Multi-agent system
        self._agent_manager: Optional[MultiAgentManager] = None
        self._behavior_analyzer: Optional[BehaviorAnalyzer] = None
        self._predictor: Optional[Predictor] = None
        self._last_prediction: Optional[Prediction] = None
        self._last_events: List[AgentEvent] = []
        self._adaptation_scores: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, difficulty: str = "medium",
              episode_id: Optional[str] = None, **kwargs: Any) -> FairRecoveryObservation:
        """Reset environment with multi-agent system initialization."""
        self._task = get_task(difficulty)
        self._current_difficulty = difficulty
        self._city = CityState(self._task.__dict__)
        self._reward_engine = RewardEngine(self._task)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._action_history = []
        self._city.step_stage = "analyze"
        self._adaptation_scores = []

        # Initialize multi-agent system
        vuln_ratios = [z.vulnerable_ratio for z in self._city.zones]
        actual_seed = seed or 42
        self._agent_manager = MultiAgentManager(
            num_zones=len(self._city.zones),
            zone_vulnerabilities=vuln_ratios,
            seed=actual_seed,
        )
        self._behavior_analyzer = BehaviorAnalyzer(num_zones=len(self._city.zones))
        self._predictor = Predictor(self._behavior_analyzer)
        self._last_prediction = None
        self._last_events = []

        # Rubrics reset
        self._rubrics.reset()
        initial_fairness = compute_fairness_reward(self._city.zones)
        self._rubrics.fairness.set_initial_fairness(initial_fairness)

        logger.info("environment_reset", episode_id=self._episode_id,
                     difficulty=difficulty, num_zones=len(self._city.zones),
                     initial_budget=self._city.budget_left,
                     initial_fairness=round(initial_fairness, 4))

        obs = self._build_observation(reward=0.0, done=False, r_exec=0.0, r_fair=0.0, r_safe=0.0)
        obs.step_feedback = (
            f"Episode started. Difficulty: {difficulty}. "
            f"Zones: {len(self._city.zones)}. Budget: {self._city.budget_left}. "
            f"Active agents: {self._agent_manager.get_active_agent_count()}. "
            f"{self._task.hint}"
        )
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: Action, timeout_s: Optional[float] = None,
             **kwargs: Any) -> FairRecoveryObservation:
        """Execute one agent action with multi-agent dynamics."""
        if self._city is None or self._reward_engine is None:
            return self._error_obs("Call reset() before step().")

        if check_timeout(self._step_count):
            return self._build_observation(
                reward=PENALTY_INVALID_ACTION, done=True,
                r_exec=0.0, r_fair=0.0, r_safe=-0.5,
                feedback="Episode terminated: safety step cap exceeded.")

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
            typed_action = FairRecoveryAction(action_type="noop",
                                               reasoning=f"Parse error: {exc}")

        self._step_count += 1
        city = self._city
        action_type = str(typed_action.action_type).replace("ActionType.", "").lower()

        # ── HARD GATE: kill early-submit exploitation (structural, not band-aid) ──
        # If the agent tries to submit before MIN_STEPS, we apply a small penalty,
        # do NOT end the episode, and return immediately with a clear feedback msg.
        # This removes the "submit on step 1 to exit fast" exploit entirely.
        if action_type == "submit" and self._step_count < MIN_STEPS:
            obs = self._build_observation(
                reward=-EARLY_SUBMIT_PENALTY, done=False,
                r_exec=0.0, r_fair=0.0, r_safe=-EARLY_SUBMIT_PENALTY,
                r_adapt=0.0, r_stable=0.0,
                feedback=(
                    f"early_submit_blocked: step {self._step_count} < MIN_STEPS={MIN_STEPS}. "
                    f"Penalty {EARLY_SUBMIT_PENALTY}. Continue with analyze/allocate/execute."
                ),
            )
            self._action_history.append(
                f"Step {self._step_count}: submit_blocked (penalty={-EARLY_SUBMIT_PENALTY:+.3f})"
            )
            obs.action_history = list(self._action_history[-8:])
            logger.info("early_submit_blocked", episode_id=self._episode_id,
                        step=self._step_count, min_steps=MIN_STEPS)
            return obs

        # Retrieve allocations for shield
        alloc_dicts = None
        if typed_action.allocations:
            alloc_dicts = [a.model_dump() for a in typed_action.allocations]

        # Shield validation
        is_valid, violations = validate(
            action_type=action_type, current_stage=city.step_stage,
            step_count=self._step_count, city=city, allocations=alloc_dicts)

        # Run multi-agent system to generate events
        agent_events = self._run_agents()

        # Generate prediction for next step
        prediction = self._generate_prediction()

        # Dispatch by action type
        reward = r_exec = r_fair = r_safe = r_adapt = r_stable = 0.0
        done = False
        feedback = ""

        if action_type == "analyze":
            components = self._reward_engine.compute_analysis_step(
                chosen_zones=typed_action.critical_zones or [], city=city)
            reward = components.R_total
            r_exec = components.R_analysis
            feedback = components.feedback
            city.record(f"analyzed zones={typed_action.critical_zones}")
            city.step_stage = "allocate"

        elif action_type == "allocate":
            city.pending_allocations = alloc_dicts or []
            city.record(f"queued {len(city.pending_allocations)} allocations")
            city.step_stage = "execute"
            feedback = (f"Queued {len(city.pending_allocations)} allocation(s). "
                        f"Budget: {city.budget_left}. Call execute next.")

        elif action_type == "execute":
            city.snapshot_services()
            exec_violations = city.apply_allocations()
            all_violations = violations + exec_violations

            # Evaluate adaptation to predictions
            adapt_score = self._evaluate_adaptation(typed_action)

            # Apply multi-agent effects
            self._apply_agent_effects()

            components = self._reward_engine.compute_execute_step(
                city=city, violations=all_violations)
            reward = components.R_total
            r_exec = components.R_exec
            r_fair = components.R_fair
            r_safe = components.R_safe
            feedback = components.feedback
            city.record(f"executed | {feedback}")
            city.step_stage = "analyze"

            if self._should_end_episode(action_type=action_type):
                done = True

        elif action_type == "adapt":
            # Adaptation step: agent explicitly responds to predictions/events
            adapt_score = self._evaluate_adaptation(typed_action)
            self._adaptation_scores.append(adapt_score)
            reward = 0.05 * adapt_score  # Small reward for adaptation
            r_adapt = adapt_score
            feedback = (f"Adaptation score: {adapt_score:.3f}. "
                        f"Strategy: {typed_action.adaptation_strategy or 'none'}")
            city.record(f"adapted | {feedback}")
            city.step_stage = "analyze"

        elif action_type == "submit":
            components = self._reward_engine.compute_submit_reward(city=city)
            reward = components.R_total
            r_fair = components.R_fair
            feedback = components.feedback
            done = True

        elif action_type == "noop":
            reward = -0.02
            feedback = "No action taken (noop). This wastes a step."

        else:
            reward = PENALTY_INVALID_ACTION
            feedback = f"Unknown action_type '{action_type}'."
            violations.append(f"unknown_action_type:{action_type}")

        # Stage violations
        if violations and not is_valid:
            reward += PENALTY_WRONG_STAGE
            feedback += f" | Violations: {violations}"

        # Global termination check (covers analyze/allocate/adapt/noop paths too).
        # NOTE: submit pre-MIN_STEPS already short-circuited earlier.
        if not done and self._should_end_episode(action_type=action_type):
            done = True
            feedback += " | episode_end_reached"

        # Build observation with multi-agent data
        obs = self._build_observation(
            reward=reward, done=done, r_exec=r_exec, r_fair=r_fair,
            r_safe=r_safe, feedback=feedback, agent_events=self._last_events,
            predictions=prediction.to_dict() if prediction else None)

        # Rubric scoring (RFC 004)
        rubric_score = self._rubrics.forward(typed_action, obs)
        if rubric_score != 0.0:
            obs.cumulative_reward += rubric_score

        # Action history
        action_summary = f"Step {self._step_count}: {action_type} (reward={reward:+.3f})"
        self._action_history.append(action_summary)
        obs.action_history = list(self._action_history[-8:])

        # Final grader score
        if done:
            obs.grader_score = self._reward_engine.get_final_grader_score(city)

        logger.info("step_executed", episode_id=self._episode_id,
                     step=self._step_count, action_type=action_type,
                     day=city.day, reward=round(reward, 4), done=done)

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # state property
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def state(self) -> FairRecoveryState:
        if self._city is None:
            return FairRecoveryState()
        city = self._city
        zones_obs = [
            ZoneObservation(
                zone_id=z.zone_id, damage=round(z.damage, 3),
                service=round(z.service, 3),
                vulnerable_ratio=round(z.vulnerable_ratio, 3),
                citizen_satisfaction=round(z.citizen_satisfaction, 3),
                risk_level=0.0)
            for z in city.zones
        ]
        fairness = compute_fairness_reward(city.zones)
        adapt_avg = (sum(self._adaptation_scores) / len(self._adaptation_scores)
                     if self._adaptation_scores else 0.0)
        return FairRecoveryState(
            episode_id=self._episode_id, difficulty=self._current_difficulty,
            day=city.day, budget_left=round(city.budget_left, 2),
            step_stage=city.step_stage, step_count=self._step_count,
            cumulative_reward=round(self._reward_engine.cumulative_reward if self._reward_engine else 0.0, 4),
            fairness_score=round(fairness, 4),
            is_done=(
                city.day >= MAX_DAYS
                or self._average_recovery() >= 0.95
                or city.budget_left <= 0
            ),
            violations_total=city.violations_total, zones=zones_obs,
            active_agents=self._agent_manager.get_active_agent_count() if self._agent_manager else 0,
            adversarial_events=self._agent_manager.get_adversarial_event_count() if self._agent_manager else 0,
            adaptation_score=round(adapt_avg, 4))

    def close(self) -> None:
        self._city = None
        self._task = None
        self._reward_engine = None
        self._agent_manager = None
        self._behavior_analyzer = None
        self._predictor = None
        self._action_history.clear()
        logger.info("environment_closed", episode_id=self._episode_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-Agent Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _run_agents(self) -> List[AgentEvent]:
        """Run all agents and collect events."""
        if not self._agent_manager or not self._city:
            return []
        city = self._city
        events = self._agent_manager.step(
            zone_services=city.current_services,
            zone_damages=city.current_damages,
            zone_vulnerabilities=city.current_vulnerabilities,
            planner_target_zones=city.planner_target_zones,
            day=city.day)

        # Ingest into behavior analyzer
        if self._behavior_analyzer:
            self._behavior_analyzer.ingest(events)

        # Convert to AgentEvent models for observation
        self._last_events = [
            AgentEvent(agent_type=e.agent_type, event_type=e.event_type,
                       zone_id=e.zone_id, intensity=e.intensity,
                       message=e.message, timestamp=e.timestamp)
            for e in events
        ]
        return self._last_events

    def _generate_prediction(self) -> Optional[Prediction]:
        """Generate prediction for next events."""
        if not self._predictor or not self._city:
            return None
        city = self._city
        prediction = self._predictor.predict_next(
            zone_services=city.current_services,
            zone_damages=city.current_damages,
            zone_vulnerabilities=city.current_vulnerabilities,
            day=city.day)
        self._last_prediction = prediction
        return prediction

    def _evaluate_adaptation(self, action: FairRecoveryAction) -> float:
        """Evaluate how well the agent adapted to predictions."""
        if not self._predictor or not self._last_prediction:
            return 0.5
        planner_zones = []
        if action.critical_zones:
            planner_zones = action.critical_zones
        elif action.allocations:
            planner_zones = [a.zone for a in action.allocations]
        return self._predictor.evaluate_adaptation(
            prediction=self._last_prediction,
            actual_events=self._agent_manager.interaction_log[-5:] if self._agent_manager else [],
            planner_actions=planner_zones)

    def _apply_agent_effects(self) -> None:
        """Apply multi-agent effects to the world state."""
        if not self._agent_manager or not self._city:
            return
        city = self._city
        # Apply disruptions from adversarial agents
        for event in self._last_events:
            if event.event_type == "disruption":
                idx = event.zone_id
                if 0 <= idx < len(city.zones):
                    city.zones[idx].apply_disruption(event.intensity)
            elif event.event_type in ("cooperation", "aid_delivery"):
                idx = event.zone_id
                if 0 <= idx < len(city.zones):
                    city.zones[idx].service = min(1.0, city.zones[idx].service + 0.05)
        # Update citizen satisfaction
        for citizen in self._agent_manager.citizens:
            idx = citizen.zone_id
            if 0 <= idx < len(city.zones):
                city.zones[idx].citizen_satisfaction = citizen.satisfaction

    # ──────────────────────────────────────────────────────────────────────────
    # Observation Builders
    # ──────────────────────────────────────────────────────────────────────────
    def _build_observation(self, reward: float, done: bool, r_exec: float,
                           r_fair: float, r_safe: float, feedback: str = "",
                           agent_events: Optional[List[AgentEvent]] = None,
                           predictions: Optional[dict] = None) -> FairRecoveryObservation:
        city = self._city
        zones_obs = [
            ZoneObservation(
                zone_id=z.zone_id, damage=round(z.damage, 3),
                service=round(z.service, 3),
                vulnerable_ratio=round(z.vulnerable_ratio, 3),
                citizen_satisfaction=round(z.citizen_satisfaction, 3),
                risk_level=0.0)
            for z in city.zones
        ] if city else []

        fairness = compute_fairness_reward(city.zones) if city else 0.0
        steps_remaining = (MAX_DAYS - city.day) * 3 if city else 0
        cumulative = self._reward_engine.cumulative_reward if self._reward_engine else 0.0

        # Attach risk levels from behavior analyzer
        if self._behavior_analyzer and zones_obs:
            risk_levels = self._behavior_analyzer.get_risk_levels()
            for i, zo in enumerate(zones_obs):
                if i < len(risk_levels):
                    zo.risk_level = round(risk_levels[i], 3)

        info = self._build_reward_info(
            reward=reward,
            r_exec=r_exec,
            r_fair=r_fair,
            r_safe=r_safe,
            done=done,
            fairness_score=fairness,
        )

        return FairRecoveryObservation(
            zones=zones_obs, day=city.day if city else 0,
            budget_left=round(city.budget_left, 2) if city else 0.0,
            step_stage=city.step_stage if city else "analyze",
            fairness_score=round(fairness, 4), step_feedback=feedback,
            steps_remaining=max(0, steps_remaining),
            cumulative_reward=round(cumulative, 4),
            r_exec=round(r_exec, 4), r_fair=round(r_fair, 4),
            r_safe=round(r_safe, 4),
            done=done, reward=round(reward, 4),
            info=info,
            agent_events=agent_events or [], predictions=predictions)

    def _error_obs(self, msg: str) -> FairRecoveryObservation:
        return FairRecoveryObservation(
            step_feedback=f"Error: {msg}", done=False,
            reward=PENALTY_INVALID_ACTION,
            info=self._build_reward_info(
                reward=PENALTY_INVALID_ACTION,
                r_exec=0.0,
                r_fair=0.0,
                r_safe=-1.0,
            ),
        )

    def _build_reward_info(
        self,
        reward: float,
        r_exec: float,
        r_fair: float,
        r_safe: float,
        done: bool = False,
        fairness_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Curriculum-weighted reward in [0,1] + trajectory-level final bonus.

        Curriculum:
          progress = min(step_count, CURRICULUM_MAX_STEPS) / CURRICULUM_MAX_STEPS
          step_r   = (0.6 + 0.4*p)*utility + (0.2 + 0.3*p)*fairness + 0.2*safety

        => Early steps reward utility; late steps reward fairness.
        => Prevents "fair but useless" policies and "instant-submit" exploits.

        Final bonus (only when done=True) teaches long-horizon planning:
          bonus = 0.5*final_utility + 0.5*final_fairness
        """
        utility = _signed_to_unit(r_exec)
        fairness = _signed_to_unit(r_fair)
        safety = _signed_to_unit(r_safe)
        raw = _signed_to_unit(reward)

        progress = min(self._step_count, CURRICULUM_MAX_STEPS) / float(CURRICULUM_MAX_STEPS)
        w_u = 0.6 + 0.4 * progress
        w_f = 0.2 + 0.3 * progress
        w_s = 0.2

        step_r = w_u * utility + w_f * fairness + w_s * safety
        # Normalize back to [0,1] (max possible weights ≈ 1.0+0.5+0.2 = 1.7)
        step_r = _clip01(step_r / (w_u + w_f + w_s))

        # Trajectory-level long-horizon bonus on episode end.
        bonus = 0.0
        if done:
            final_utility = self._average_recovery()
            final_fairness = _signed_to_unit(fairness_score)
            bonus = (
                FINAL_BONUS_WEIGHT_UTILITY * final_utility
                + FINAL_BONUS_WEIGHT_FAIRNESS * final_fairness
            )
            bonus = _clip01(bonus)

        # Blended reward: 0.7 * curriculum step reward + 0.3 * final bonus (if any).
        # If not done, blend = step_r alone.
        if done:
            blended = _clip01(0.6 * step_r + 0.4 * bonus)
        else:
            blended = step_r

        return {
            "reward": round(blended, 4),
            "reward_step": round(step_r, 4),
            "reward_raw": round(raw, 4),
            "final_bonus": round(bonus, 4),
            "progress": round(progress, 4),
            "utility": round(utility, 4),
            "fairness": round(fairness, 4),
            "safety": round(safety, 4),
        }

    def _should_end_episode(self, action_type: str) -> bool:
        """Episode-end logic with structural trajectory shaping.

        done = (step_count >= MIN_STEPS AND action == "submit")
            OR step_count >= CURRICULUM_MAX_STEPS
            OR day >= MAX_DAYS
            OR recovery >= 0.95
            OR budget_left <= 0
        """
        if self._city is None:
            return False
        recovery = self._average_recovery()
        valid_submit = action_type == "submit" and self._step_count >= MIN_STEPS
        hit_max_steps = self._step_count >= CURRICULUM_MAX_STEPS
        return (
            self._city.day >= MAX_DAYS
            or recovery >= 0.95
            or valid_submit
            or hit_max_steps
            or self._city.budget_left <= 0
        )

    def _average_recovery(self) -> float:
        if self._city is None or not self._city.zones:
            return 0.0
        return float(sum(z.service for z in self._city.zones) / len(self._city.zones))
