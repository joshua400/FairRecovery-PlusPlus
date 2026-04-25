"""
FairRecovery — Pydantic Models.

Typed Action, Observation, and State models conforming to the OpenEnv spec.
Uses Literal types for enum fields so the OpenEnv Gradio web interface
renders them as dropdown selectors instead of free-text inputs.

Mirrors the exact pattern from the reference hallucination-detector-gym.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

try:
    from openenv.core.env_server.types import (
        Action as BaseAction,
        Observation as BaseObservation,
        State as BaseState,
    )
except ImportError:
    # Fallback for environments without openenv installed
    from pydantic import BaseModel as BaseAction  # type: ignore
    from pydantic import BaseModel as BaseObservation  # type: ignore
    from pydantic import BaseModel as BaseState  # type: ignore

from .constants import ActionType, Difficulty, ResourceType


# ──────────────────────────────────────────────────────────────────────────────
# Literal type aliases — renders as dropdowns in OpenEnv Gradio UI
# ──────────────────────────────────────────────────────────────────────────────
ActionTypeLiteral  = Literal["analyze", "allocate", "execute", "submit", "noop"]
ResourceTypeLiteral = Literal["power", "water", "medical"]
DifficultyLiteral  = Literal["easy", "medium", "hard"]


def _flatten_enum_from_anyof(schema: Dict[str, Any]) -> None:
    """Post-process JSON schema for OpenEnv Gradio UI compatibility.

    Pydantic v2 wraps Optional[...] in anyOf and uses $ref for enums,
    hiding the 'enum' key the Gradio UI needs to render dropdowns.
    This promotes them to top-level so widgets render correctly.
    """
    defs = schema.get("$defs", {})

    for _name, prop in schema.get("properties", {}).items():
        # Resolve $ref → inline enum
        if "$ref" in prop and "enum" not in prop:
            ref_name = prop["$ref"].rsplit("/", 1)[-1]
            ref_def = defs.get(ref_name, {})
            if "enum" in ref_def:
                prop["enum"] = ref_def["enum"]

        # Lift enum + type from anyOf (for Optional[Literal[...]])
        if "anyOf" in prop:
            for variant in prop["anyOf"]:
                if variant.get("type") == "null":
                    continue
                if "enum" not in prop and "enum" in variant:
                    prop["enum"] = variant["enum"]
                if "type" not in prop and "type" in variant:
                    prop["type"] = variant["type"]
                if "maxLength" not in prop and "maxLength" in variant:
                    prop["maxLength"] = variant["maxLength"]
                if "enum" not in prop and "$ref" in variant:
                    ref_name = variant["$ref"].rsplit("/", 1)[-1]
                    ref_def = defs.get(ref_name, {})
                    if "enum" in ref_def:
                        prop["enum"] = ref_def["enum"]

    # Reorder properties for logical action-building flow
    desired_order = [
        "action_type",
        "difficulty",
        "critical_zones",
        "reasoning",
        "allocations",
        "metadata",
    ]
    props = schema.get("properties", {})
    ordered: Dict[str, Any] = {}
    for key in desired_order:
        if key in props:
            ordered[key] = props[key]
    for key, val in props.items():
        if key not in ordered:
            ordered[key] = val
    schema["properties"] = ordered


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────
class AllocationItem(BaseAction if BaseAction.__name__ != "BaseModel" else object):  # type: ignore
    """Single resource allocation to a zone."""

    # Avoid MRO issues with fallback BaseModel
    model_config = ConfigDict(populate_by_name=True)

    zone: int = Field(
        ...,
        ge=0,
        title="Zone Index",
        description="Zone index (0-based). Must be a valid zone in the current scenario.",
    )
    resource: ResourceTypeLiteral = Field(
        ...,
        title="Resource Type",
        description="Resource to deploy: 'power' (cost 10), 'water' (cost 15), 'medical' (cost 20).",
    )


class ZoneObservation(BaseObservation if BaseObservation.__name__ != "BaseModel" else object):  # type: ignore
    """Per-zone state visible to the agent."""

    model_config = ConfigDict(populate_by_name=True)

    zone_id: int         = Field(..., description="Zone identifier (0-based).")
    damage: float        = Field(..., ge=0.0, le=1.0, description="Damage level: 1.0=destroyed, 0.0=intact.")
    service: float       = Field(..., ge=0.0, le=1.0, description="Service availability: 1.0=full, 0.0=none.")
    vulnerable_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of population that is vulnerable (elderly, low-income, disabled).")


# Use plain Pydantic for sub-models to avoid MRO issues
from pydantic import BaseModel as _BaseModel

class AllocationItem(_BaseModel):
    zone: int = Field(..., ge=0, title="Zone Index",
                      description="Zone index (0-based).")
    resource: ResourceTypeLiteral = Field(..., title="Resource Type",
                                          description="'power'(cost 10), 'water'(cost 15), 'medical'(cost 20).")


class ZoneObservation(_BaseModel):
    zone_id: int            = Field(..., description="Zone identifier (0-based).")
    damage: float           = Field(..., ge=0.0, le=1.0)
    service: float          = Field(..., ge=0.0, le=1.0)
    vulnerable_ratio: float = Field(..., ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Action
# ──────────────────────────────────────────────────────────────────────────────
class FairRecoveryAction(BaseAction):
    """
    Action the agent submits each step.

    Protocol (repeat MAX_DAYS times, then submit):
      1. analyze  — identify critical zones + reasoning
      2. allocate — queue resource allocations
      3. execute  — commit allocations, receive reward
      4. submit   — terminate episode, receive final score
    """

    model_config = ConfigDict(json_schema_extra=_flatten_enum_from_anyof)

    action_type: ActionTypeLiteral = Field(
        default="noop",
        title="Action Type",
        description=(
            "Protocol stage: analyze → allocate → execute → (repeat) → submit. "
            "Skipping stages incurs penalties."
        ),
    )
    difficulty: Optional[DifficultyLiteral] = Field(
        default=None,
        title="Difficulty (reset only)",
        description="Scenario difficulty. Only used when action_type='reset'.",
    )
    critical_zones: Optional[List[int]] = Field(
        default=None,
        title="Critical Zones",
        description="Zone indices identified as critical. Used during 'analyze' step.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        title="Reasoning",
        max_length=2000,
        description="Chain-of-thought reasoning. Not scored but helpful for training.",
    )
    allocations: Optional[List[AllocationItem]] = Field(
        default=None,
        title="Allocations",
        description="List of {zone, resource} pairs to deploy. Used during 'allocate' step.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        title="Metadata",
        description="Optional metadata for debugging.",
    )

    @model_validator(mode="after")
    def _coerce_action_type(self) -> "FairRecoveryAction":
        """Coerce string action_type to ActionType enum."""
        if isinstance(self.action_type, str):
            object.__setattr__(self, "action_type", ActionType(self.action_type))
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────────────────────────────────────
class FairRecoveryObservation(BaseObservation):
    """
    Observation returned to the agent after each step / reset.

    Contains full zone state, budget, day counter, step feedback,
    and component reward breakdown for training transparency.
    """

    zones: List[ZoneObservation] = Field(
        default_factory=list,
        description="Current state of all recovery zones.",
    )
    day: int = Field(
        default=0,
        description="Current day number (0-indexed). Episode ends at MAX_DAYS.",
    )
    budget_left: float = Field(
        default=0.0,
        description="Remaining resource budget.",
    )
    step_stage: str = Field(
        default="analyze",
        description="Expected action type for next step.",
    )
    fairness_score: float = Field(
        default=0.0,
        description="Current fairness score (0=equal service, negative=disparity exists).",
    )
    step_feedback: Optional[str] = Field(
        default=None,
        description="Textual feedback from the last action.",
    )
    steps_remaining: int = Field(
        default=0,
        description="Steps remaining in the episode.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Running total reward accumulated this episode.",
    )
    r_exec: float = Field(
        default=0.0,
        description="Last step execution reward component (service improvement).",
    )
    r_fair: float = Field(
        default=0.0,
        description="Last step fairness reward component (disparity reduction).",
    )
    r_safe: float = Field(
        default=0.0,
        description="Last step safety reward component (constraint satisfaction).",
    )
    action_history: List[str] = Field(
        default_factory=list,
        description="Summary of actions taken so far.",
    )
    grader_score: Optional[float] = Field(
        default=None,
        description="Normalised task score in (0.01, 0.99). Set when done=True.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# State (internal — exposed via GET /state)
# ──────────────────────────────────────────────────────────────────────────────
class FairRecoveryState(BaseState):
    """Internal environment state exposed via state() endpoint."""

    episode_id: Optional[str] = Field(default=None, description="Unique episode identifier.")
    difficulty: Optional[str] = Field(default=None, description="Current scenario difficulty.")
    day: int                  = Field(default=0,    description="Current day.")
    budget_left: float        = Field(default=0.0,  description="Remaining budget.")
    step_stage: str           = Field(default="analyze", description="Next expected action.")
    step_count: int           = Field(default=0,    description="Total steps taken.")
    cumulative_reward: float  = Field(default=0.0,  description="Total reward this episode.")
    fairness_score: float     = Field(default=0.0,  description="Current fairness score.")
    is_done: bool             = Field(default=False, description="Whether episode has ended.")
    violations_total: int     = Field(default=0,    description="Total safety violations.")
    zones: List[ZoneObservation] = Field(default_factory=list, description="Zone snapshots.")
