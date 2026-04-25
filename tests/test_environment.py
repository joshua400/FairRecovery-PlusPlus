"""
Tests for FairRecovery++ environment.

Covers: models, reward engine, multi-agent system, environment lifecycle,
task registry, anti-exploitation, and grader score bounds.
Mirrors hallucination-detector-gym test_environment.py pattern exactly.
"""

from __future__ import annotations
import pytest

from fairrecovery_env.constants import ActionType, Difficulty, REWARD_WEIGHTS
from fairrecovery_env.models import (
    AllocationItem, FairRecoveryAction, FairRecoveryObservation,
    FairRecoveryState, ZoneObservation, AgentEvent,
)
from fairrecovery_env.rewards import (
    RewardEngine, compute_exec_reward, compute_fairness_reward,
    compute_safety_reward, compute_stability_reward, compute_analysis_reward,
)
from fairrecovery_env.state import CityState, ZoneState
from fairrecovery_env.tasks import TASKS, get_task
from fairrecovery_env.agents import (
    CitizenAgent, NGOAgent, AdversarialAgent, MultiAgentManager,
)
from fairrecovery_env.behavior_analyzer import BehaviorAnalyzer
from fairrecovery_env.predictor import Predictor, Prediction


# ──────────────────────────────────────────────────────────────────────────────
# Task Registry Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestTaskRegistry:
    def test_registry_has_three_tasks(self) -> None:
        assert len(TASKS) >= 3

    def test_tasks_cover_all_difficulties(self) -> None:
        difficulties = {t.difficulty for t in TASKS.values()}
        assert Difficulty.EASY in difficulties
        assert Difficulty.MEDIUM in difficulties
        assert Difficulty.HARD in difficulties

    def test_get_task_returns_correct(self) -> None:
        task = get_task("easy")
        assert task.difficulty == Difficulty.EASY

    def test_get_task_raises_on_invalid(self) -> None:
        with pytest.raises(ValueError):
            get_task("nonexistent")

    def test_each_task_has_zones(self) -> None:
        for task in TASKS.values():
            assert len(task.zones) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Model Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestModels:
    def test_action_creation_analyze(self) -> None:
        action = FairRecoveryAction(action_type="analyze", critical_zones=[1, 2])
        assert action.critical_zones == [1, 2]

    def test_action_creation_allocate(self) -> None:
        action = FairRecoveryAction(
            action_type="allocate",
            allocations=[AllocationItem(zone=0, resource="medical")])
        assert len(action.allocations) == 1

    def test_action_creation_noop(self) -> None:
        action = FairRecoveryAction(action_type="noop")
        assert action.critical_zones is None

    def test_observation_defaults(self) -> None:
        obs = FairRecoveryObservation()
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.action_history == []

    def test_state_defaults(self) -> None:
        state = FairRecoveryState()
        assert state.step_count == 0
        assert state.is_done is False


# ──────────────────────────────────────────────────────────────────────────────
# Reward Engine Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestRewardEngine:
    def test_exec_reward_positive_on_improvement(self) -> None:
        zones = [ZoneState(0, 0.5, 0.6, 0.5), ZoneState(1, 0.3, 0.8, 0.3)]
        prev = [0.4, 0.5]
        assert compute_exec_reward(prev, zones) > 0.0

    def test_exec_reward_zero_no_change(self) -> None:
        zones = [ZoneState(0, 0.5, 0.5, 0.5)]
        prev = [0.5]
        assert compute_exec_reward(prev, zones) == 0.0

    def test_fairness_negative_disparity(self) -> None:
        zones = [ZoneState(0, 0.5, 0.9, 0.1), ZoneState(1, 0.8, 0.2, 0.9)]
        score = compute_fairness_reward(zones)
        assert score < 0.0  # vulnerable zone has lower service

    def test_fairness_zero_parity(self) -> None:
        zones = [ZoneState(0, 0.5, 0.5, 0.1), ZoneState(1, 0.5, 0.5, 0.9)]
        score = compute_fairness_reward(zones)
        assert abs(score) < 0.01

    def test_safety_penalty_per_violation(self) -> None:
        assert compute_safety_reward([]) == 0.0
        assert compute_safety_reward(["v1"]) < 0.0
        assert compute_safety_reward(["v1", "v2"]) <= compute_safety_reward(["v1"])

    def test_stability_low_variance(self) -> None:
        zones = [ZoneState(0, 0.5, 0.5, 0.5), ZoneState(1, 0.5, 0.5, 0.5)]
        zones[0].citizen_satisfaction = 0.5
        zones[1].citizen_satisfaction = 0.5
        score = compute_stability_reward(zones)
        assert score >= -0.01  # Low variance = good

    def test_analysis_correct_identification(self) -> None:
        zones = [ZoneState(0, 0.1, 0.9, 0.1), ZoneState(1, 0.9, 0.1, 0.9)]
        score = compute_analysis_reward([1], zones)
        assert score > 0.0

    def test_cumulative_reward_tracks(self) -> None:
        task = get_task("easy")
        engine = RewardEngine(task)
        city = CityState(task.__dict__)
        engine.compute_analysis_step([1], city)
        assert engine.cumulative_reward != 0.0

    def test_final_score_in_range(self) -> None:
        task = get_task("easy")
        engine = RewardEngine(task)
        city = CityState(task.__dict__)
        score = engine.get_final_grader_score(city)
        assert 0.0 < score < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Agent Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestMultiAgentSystem:
    def test_citizen_generates_events(self) -> None:
        citizen = CitizenAgent(zone_id=0, vulnerability=0.9, seed=42)
        events = citizen.step(service=0.1, damage=0.9, day=1)
        # High vulnerability + low service should generate complaints
        assert isinstance(events, list)

    def test_adversary_targets_weakest(self) -> None:
        adv = AdversarialAgent(seed=42)
        events = adv.step([0.1, 0.9], [0.9, 0.1], [0.9, 0.1], day=1)
        if events:
            assert events[0].zone_id == 0  # Zone 0 is weakest

    def test_adversary_deactivation(self) -> None:
        adv = AdversarialAgent(seed=42)
        adv.deactivate()
        events = adv.step([0.1], [0.9], [0.9], day=1)
        assert events == []

    def test_ngo_cooperation(self) -> None:
        ngo = NGOAgent(focus_zones=[1], seed=42)
        events = ngo.step([0.5, 0.1], [0.5, 0.9], planner_target_zones=[1], day=1)
        assert isinstance(events, list)

    def test_manager_coordinates_agents(self) -> None:
        mgr = MultiAgentManager(num_zones=3, zone_vulnerabilities=[0.1, 0.9, 0.5], seed=42)
        events = mgr.step([0.5, 0.1, 0.5], [0.5, 0.9, 0.5], [0.1, 0.9, 0.5], [1], day=1)
        assert isinstance(events, list)
        assert mgr.get_active_agent_count() >= 3


# ──────────────────────────────────────────────────────────────────────────────
# Behavior Analyzer Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestBehaviorAnalyzer:
    def test_ingest_and_analyze(self) -> None:
        from fairrecovery_env.agents import InteractionLog
        analyzer = BehaviorAnalyzer(num_zones=3)
        analyzer.ingest([
            InteractionLog("citizen", "complaint", 0, 0.8, 1),
            InteractionLog("citizen", "complaint", 0, 0.7, 2),
            InteractionLog("adversary", "disruption", 1, 0.9, 1),
        ])
        profile = analyzer.analyze_zone(0)
        assert profile.complaint_rate > 0.0
        pattern = analyzer.analyze_system()
        assert pattern.adversarial_active is True

    def test_risk_levels(self) -> None:
        analyzer = BehaviorAnalyzer(num_zones=2)
        assert len(analyzer.get_risk_levels()) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Predictor Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestPredictor:
    def test_predict_returns_prediction(self) -> None:
        analyzer = BehaviorAnalyzer(num_zones=3)
        predictor = Predictor(analyzer)
        pred = predictor.predict_next([0.5, 0.1, 0.5], [0.5, 0.9, 0.5], [0.1, 0.9, 0.5], day=1)
        assert isinstance(pred, Prediction)

    def test_adaptation_evaluation(self) -> None:
        analyzer = BehaviorAnalyzer(num_zones=3)
        predictor = Predictor(analyzer)
        pred = Prediction(likely_zone=1, risk="high", confidence=0.7)
        score = predictor.evaluate_adaptation(pred, [], [1])
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Environment Lifecycle Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestEnvironmentLifecycle:
    def test_reset_produces_clean_state(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        obs = env.reset(difficulty="easy")
        assert obs.done is False
        assert obs.reward == 0.0
        assert len(obs.zones) == 3
        assert env.state.step_count == 0

    def test_step_without_reset_returns_error(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        obs = env.step(FairRecoveryAction(action_type="noop"))
        assert "Error" in (obs.step_feedback or "")

    def test_full_episode_lifecycle(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        env.reset(difficulty="easy")

        obs = env.step(FairRecoveryAction(
            action_type="analyze", critical_zones=[1],
            reasoning="Zone 1 has highest vulnerability."))
        assert obs.done is False

        obs = env.step(FairRecoveryAction(
            action_type="allocate",
            allocations=[AllocationItem(zone=1, resource="medical")]))
        assert obs.done is False

        obs = env.step(FairRecoveryAction(action_type="execute"))
        assert obs.done is False
        assert obs.r_exec != 0.0 or obs.r_fair != 0.0

        obs = env.step(FairRecoveryAction(action_type="submit"))
        assert obs.done is True

    def test_grader_score_strictly_in_0_1(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        for diff in ["easy", "medium", "hard"]:
            env = FairRecoveryEnvironment()
            env.reset(difficulty=diff)
            obs = env.step(FairRecoveryAction(action_type="submit"))
            assert obs.done is True
            assert obs.grader_score is not None
            assert 0.0 < obs.grader_score < 1.0
            formatted = f"{obs.grader_score:.2f}"
            assert formatted != "0.00" and formatted != "1.00"

    def test_multi_agent_events_in_observation(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        env.reset(difficulty="hard")
        obs = env.step(FairRecoveryAction(
            action_type="analyze", critical_zones=[4, 3]))
        assert isinstance(obs.agent_events, list)

    def test_five_day_episode(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        env.reset(difficulty="medium")
        for _ in range(5):
            env.step(FairRecoveryAction(action_type="analyze", critical_zones=[1, 3]))
            env.step(FairRecoveryAction(action_type="allocate",
                     allocations=[AllocationItem(zone=1, resource="power")]))
            obs = env.step(FairRecoveryAction(action_type="execute"))
            if obs.done:
                break
        if not obs.done:
            obs = env.step(FairRecoveryAction(action_type="submit"))
        assert obs.done is True
        assert obs.grader_score is not None
        assert 0.0 < obs.grader_score < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Anti-Exploitation Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestAntiExploitation:
    def test_repeated_noop_penalised(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        env.reset(difficulty="easy")
        obs1 = env.step(FairRecoveryAction(action_type="noop"))
        obs2 = env.step(FairRecoveryAction(action_type="noop"))
        assert obs1.reward < 0.0
        assert obs2.reward < 0.0

    def test_invalid_zone_penalised(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        env = FairRecoveryEnvironment()
        env.reset(difficulty="easy")
        env.step(FairRecoveryAction(action_type="analyze", critical_zones=[1]))
        obs = env.step(FairRecoveryAction(
            action_type="allocate",
            allocations=[AllocationItem(zone=99, resource="power")]))
        # Should have violation feedback
        assert obs.step_feedback is not None

    def test_score_never_exactly_zero_or_one(self) -> None:
        from server.fairrecovery_environment import FairRecoveryEnvironment
        for diff in ["easy", "medium", "hard"]:
            env = FairRecoveryEnvironment()
            env.reset(difficulty=diff)
            obs = env.step(FairRecoveryAction(action_type="submit"))
            assert 0.0 < obs.grader_score < 1.0
