"""
FairRecovery++ — Baseline Inference Script.

Updated to match the refactored project structure.
"""

from __future__ import annotations
import argparse
import json
import random
import numpy as np
from typing import Dict, List

from fairrecovery_env.models import ResourceAllocation, FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.constants import ActionType, ResourceType, COST_MEDICAL, COST_WATER, COST_POWER

# Local dummy client for testing if server is not running
class LocalInference:
    def __init__(self, base_url: str):
        self.base_url = base_url

def random_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Completely random policy."""
    return FairRecoveryAction(
        action_type=random.choice([ActionType.ANALYZE, ActionType.ALLOCATE, ActionType.EXECUTE]),
        reasoning="Random strategy."
    )

def greedy_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Utility-maximising greedy — ignores vulnerability (WRONG policy)."""
    if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)
    
    # Simple heuristic for this example
    return FairRecoveryAction(
        action_type=ActionType.ALLOCATE,
        allocations=[
            ResourceAllocation(zone=0, resource=ResourceType.MEDICAL)
        ],
        reasoning="Greedy: targeting zone 0 first."
    )

def fairness_aware_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Fairness-aware heuristic (CORRECT policy)."""
    if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)
    
    # Prioritize the most vulnerable zone
    vulnerable_zone = sorted(range(len(obs.zones)), key=lambda i: obs.zones[i].vulnerable_ratio, reverse=True)[0]
    
    return FairRecoveryAction(
        action_type=ActionType.ALLOCATE,
        allocations=[
            ResourceAllocation(zone=vulnerable_zone, resource=ResourceType.MEDICAL)
        ],
        reasoning=f"Fair: prioritizing zone {vulnerable_zone} due to vulnerability."
    )
