"""
FairRecovery — Safety Shield.

Validates actions BEFORE state mutation to prevent reward hacking.
Follows anti-reward-hacking guidance from OpenEnv Hackathon FAQs §13.

All validation is performed by validate() which returns (is_valid, violations).
Violations are passed to the reward engine's R_safe component.

Design principles:
  • Block invalid stages before they mutate state
  • Block over-budget allocations before they're applied
  • Detect and flag persistent vulnerable-zone ignoring
  • Enforce step timeout: if step_count > MAX_STEPS_SAFETY_CAP, force termination
"""

from __future__ import annotations

from typing import List, Tuple

from .constants import (
    ActionType,
    MAX_ACTIONS_PER_DAY,
    MAX_STEPS_SAFETY_CAP,
    RESOURCE_COSTS,
    STAGE_SEQUENCE,
)
from .state import CityState


VALID_ACTION_TYPES  = {a.value for a in ActionType}
VALID_RESOURCES     = set(RESOURCE_COSTS.keys())

# Stage → valid next stages
STAGE_TRANSITIONS = {
    "analyze":  {"allocate", "submit", "noop"},
    "allocate": {"execute", "submit", "noop"},
    "execute":  {"analyze", "submit", "noop"},
    "submit":   set(),  # terminal
    "noop":     VALID_ACTION_TYPES,  # noop is always valid but penalised
}


def validate(
    action_type: str,
    current_stage: str,
    step_count: int,
    city: CityState,
    allocations: list | None = None,
) -> Tuple[bool, List[str]]:
    """
    Validate an action before it mutates state.

    Args:
        action_type:    String action type from the agent.
        current_stage:  Current step_stage in CityState.
        step_count:     Total steps taken this episode.
        city:           Current CityState for budget/zone checks.
        allocations:    List of {zone, resource} dicts (for allocate actions).

    Returns:
        (is_valid, violations): violations is empty when is_valid=True.
    """
    violations: List[str] = []

    # ── 1. Action type valid ─────────────────────────────────────────────────
    if action_type not in VALID_ACTION_TYPES:
        violations.append(f"invalid_action_type:{action_type}")
        return False, violations

    # ── 2. Safety cap — hard stop to prevent infinite loops ──────────────────
    if step_count >= MAX_STEPS_SAFETY_CAP:
        violations.append(f"safety_cap_exceeded:step_count={step_count}")
        return False, violations

    # ── 3. Stage ordering ────────────────────────────────────────────────────
    if action_type not in ("submit", "noop"):
        allowed = STAGE_TRANSITIONS.get(current_stage, set())
        if action_type not in allowed:
            violations.append(
                f"wrong_stage:expected_one_of_{allowed}_got_{action_type}"
            )
            # Wrong stage is a soft violation — penalised but not blocked
            # (allows partial recovery if agent gets confused)

    # ── 4. Allocate-specific: zone and resource checks ───────────────────────
    if action_type == "allocate" and allocations:
        if len(allocations) > MAX_ACTIONS_PER_DAY:
            violations.append(
                f"too_many_allocations:{len(allocations)}_max:{MAX_ACTIONS_PER_DAY}"
            )

        for alloc in allocations:
            zone_id  = alloc.get("zone")
            resource = alloc.get("resource")

            if zone_id is None or not (0 <= int(zone_id) < len(city.zones)):
                violations.append(f"invalid_zone:{zone_id}")

            if resource not in VALID_RESOURCES:
                violations.append(f"invalid_resource:{resource}")
            elif zone_id is not None and 0 <= int(zone_id) < len(city.zones):
                cost = RESOURCE_COSTS.get(resource, 0)
                if city.budget_left < cost:
                    violations.append(
                        f"budget_exceeded:zone{zone_id}:{resource}:"
                        f"cost={cost}_budget={city.budget_left}"
                    )

    is_valid = len(violations) == 0
    return is_valid, violations


def check_timeout(step_count: int) -> bool:
    """Return True if the episode should be force-terminated."""
    return step_count >= MAX_STEPS_SAFETY_CAP
