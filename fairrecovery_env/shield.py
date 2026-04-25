"""
FairRecovery++ — Safety Shield.

Validates actions BEFORE state mutation to prevent reward hacking.
"""

from __future__ import annotations
from typing import List, Tuple
from .constants import (ActionType, MAX_ACTIONS_PER_DAY, MAX_STEPS_SAFETY_CAP,
                         RESOURCE_COSTS, STAGE_SEQUENCE)
from .state import CityState

VALID_ACTION_TYPES = {a.value for a in ActionType}
VALID_RESOURCES = set(RESOURCE_COSTS.keys())

STAGE_TRANSITIONS = {
    "analyze":  {"allocate", "submit", "noop"},
    "allocate": {"execute", "submit", "noop"},
    "execute":  {"analyze", "adapt", "submit", "noop"},
    "adapt":    {"analyze", "submit", "noop"},
    "submit":   set(),
    "noop":     VALID_ACTION_TYPES,
}


def validate(action_type: str, current_stage: str, step_count: int,
             city: CityState, allocations: list | None = None) -> Tuple[bool, List[str]]:
    """Validate an action before it mutates state."""
    violations: List[str] = []

    if action_type not in VALID_ACTION_TYPES:
        violations.append(f"invalid_action_type:{action_type}")
        return False, violations

    if step_count >= MAX_STEPS_SAFETY_CAP:
        violations.append(f"safety_cap_exceeded:step_count={step_count}")
        return False, violations

    if action_type not in ("submit", "noop"):
        allowed = STAGE_TRANSITIONS.get(current_stage, set())
        if action_type not in allowed:
            violations.append(f"wrong_stage:expected_{allowed}_got_{action_type}")

    if action_type == "allocate" and allocations:
        if len(allocations) > MAX_ACTIONS_PER_DAY:
            violations.append(f"too_many_allocations:{len(allocations)}_max:{MAX_ACTIONS_PER_DAY}")
        for alloc in allocations:
            zone_id = alloc.get("zone")
            resource = alloc.get("resource")
            if zone_id is None or not (0 <= int(zone_id) < len(city.zones)):
                violations.append(f"invalid_zone:{zone_id}")
            if resource not in VALID_RESOURCES:
                violations.append(f"invalid_resource:{resource}")
            elif zone_id is not None and 0 <= int(zone_id) < len(city.zones):
                cost = RESOURCE_COSTS.get(resource, 0)
                if city.budget_left < cost:
                    violations.append(f"budget_exceeded:zone{zone_id}:{resource}")

    return len(violations) == 0, violations


def check_timeout(step_count: int) -> bool:
    """Return True if the episode should be force-terminated."""
    return step_count >= MAX_STEPS_SAFETY_CAP
