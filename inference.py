"""
FairRecovery++ — Baseline Inference Script.

Updated with Phase-Aware policies to correctly advance days.
"""

from __future__ import annotations
import random
from fairrecovery_env.models import ResourceAllocation, FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.constants import ActionType, ResourceType

def _get_phase_action(obs: FairRecoveryObservation) -> ActionType:
    """Determine the correct action type based on the 3-phase cycle."""
    num_steps = len(obs.action_history)
    cycle_pos = num_steps % 3
    if cycle_pos == 0: return ActionType.ANALYZE
    if cycle_pos == 1: return ActionType.ALLOCATE
    return ActionType.EXECUTE

def greedy_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Greedy: Target zone 0 (easiest/wealthiest) regardless of vulnerability."""
    if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)
    
    action_type = _get_phase_action(obs)
    
    if action_type == ActionType.ANALYZE:
        return FairRecoveryAction(action_type=action_type, critical_zones=[0], reasoning="Greedy focus on Zone 0.")
    
    if action_type == ActionType.ALLOCATE:
        return FairRecoveryAction(
            action_type=action_type,
            allocations=[ResourceAllocation(zone=0, resource=ResourceType.MEDICAL)],
            reasoning="Maximizing utility in Zone 0."
        )
    
    return FairRecoveryAction(action_type=ActionType.EXECUTE)

def fairness_aware_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Fairness-Aware: Prioritizes Zone 4 (highest vulnerability)."""
    if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)
    
    action_type = _get_phase_action(obs)
    
    # Identify most vulnerable zone (usually Zone 4 in hard scenario)
    v_zone = sorted(range(len(obs.zones)), key=lambda i: obs.zones[i].vulnerable_ratio, reverse=True)[0]
    
    if action_type == ActionType.ANALYZE:
        return FairRecoveryAction(action_type=action_type, critical_zones=[v_zone], reasoning=f"Prioritizing high-vulnerability Zone {v_zone}.")
    
    if action_type == ActionType.ALLOCATE:
        return FairRecoveryAction(
            action_type=action_type,
            allocations=[ResourceAllocation(zone=v_zone, resource=ResourceType.MEDICAL)],
            reasoning=f"Protecting vulnerable population in Zone {v_zone}."
        )
    
    return FairRecoveryAction(action_type=ActionType.EXECUTE)
