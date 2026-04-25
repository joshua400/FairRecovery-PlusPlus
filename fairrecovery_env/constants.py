"""
FairRecovery++ — Constants and Configuration.

All configurable values are centralised here.
No hardcoded magic numbers anywhere else in the codebase.

Multi-agent adaptive environment for post-disaster recovery.
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Final


# ──────────────────────────────────────────────────────────────────────────────
# Environment Metadata
# ──────────────────────────────────────────────────────────────────────────────
ENV_NAME: Final[str] = "fairrecovery"
ENV_VERSION: Final[str] = "2.0.0"
ENV_DESCRIPTION: Final[str] = (
    "An adaptive multi-agent OpenEnv environment where an AI planner coordinates "
    "post-disaster recovery while interacting with dynamic agents (citizens, NGOs, "
    "adversaries). Learns to optimise fairness and efficiency while adapting to "
    "evolving behavioral patterns. Designed for RLVR training via TRL/GRPO with Unsloth."
)

# ──────────────────────────────────────────────────────────────────────────────
# Episode Configuration
# ──────────────────────────────────────────────────────────────────────────────
MAX_DAYS: Final[int] = 5              # episode length (days)
MAX_ACTIONS_PER_DAY: Final[int] = 3   # max allocations per execute step
MAX_STEPS_SAFETY_CAP: Final[int] = 50 # hard cap on total steps
DEFAULT_SEED: Final[int] = 42

# ──────────────────────────────────────────────────────────────────────────────
# Resource Definitions
# ──────────────────────────────────────────────────────────────────────────────
RESOURCE_COSTS: Final[dict] = {
    "power":   10,
    "water":   15,
    "medical": 20,
}

RESOURCE_EFFECTS: Final[dict] = {
    "power":   {"service": 0.20, "damage": -0.10},
    "water":   {"service": 0.30, "damage": -0.15},
    "medical": {"service": 0.40, "damage": -0.20},
}

# ──────────────────────────────────────────────────────────────────────────────
# Reward Weights (RLVR — fully deterministic, no learned model)
# ──────────────────────────────────────────────────────────────────────────────
REWARD_WEIGHTS: Final[dict] = {
    "exec":    0.30,   # service improvement (utility)
    "fair":    0.25,   # fairness (disparity reduction)
    "adapt":   0.20,   # adaptation to predicted events
    "stable":  0.15,   # system stability
    "safe":    0.10,   # constraint satisfaction
}

# ──────────────────────────────────────────────────────────────────────────────
# Penalties
# ──────────────────────────────────────────────────────────────────────────────
PENALTY_INVALID_ACTION: Final[float] = -0.5
PENALTY_INVALID_ZONE: Final[float] = -0.3
PENALTY_INVALID_RESOURCE: Final[float] = -0.3
PENALTY_BUDGET_EXCEEDED: Final[float] = -0.2
PENALTY_IGNORE_VULNERABLE: Final[float] = -0.3
PENALTY_WRONG_STAGE: Final[float] = -0.1
PENALTY_REPEATED_ACTION: Final[float] = -0.05
PENALTY_PATTERN_IGNORED: Final[float] = -0.15
PENALTY_ADVERSARIAL_FAILURE: Final[float] = -0.2

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────
VULNERABILITY_THRESHOLD: Final[float] = 0.6
SPAN_OVERLAP_THRESHOLD: Final[float] = 0.3
COMPLAINT_RATE_HIGH: Final[float] = 0.7
RISK_THRESHOLD: Final[float] = 0.6

# Grader score bounds — strict open interval (never exactly 0 or 1)
GRADER_SCORE_MIN: Final[float] = 0.01
GRADER_SCORE_MAX: Final[float] = 0.99

# ──────────────────────────────────────────────────────────────────────────────
# Multi-Agent Configuration
# ──────────────────────────────────────────────────────────────────────────────
CITIZEN_SATISFACTION_DECAY: Final[float] = 0.05
NGO_RESOURCE_MULTIPLIER: Final[float] = 1.2
ADVERSARY_DISRUPTION_CHANCE: Final[float] = 0.3
MAX_INTERACTION_LOG_SIZE: Final[int] = 200

# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────
@unique
class Difficulty(str, Enum):
    """Task difficulty levels."""
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@unique
class ResourceType(str, Enum):
    """Available disaster-recovery resources."""
    POWER   = "power"
    WATER   = "water"
    MEDICAL = "medical"


@unique
class ActionType(str, Enum):
    """
    Agent action types — multi-step protocol:
      analyze → prioritize → allocate → execute → adapt → submit
    """
    ANALYZE  = "analyze"
    ALLOCATE = "allocate"
    EXECUTE  = "execute"
    ADAPT    = "adapt"
    SUBMIT   = "submit"
    NOOP     = "noop"


@unique
class AgentType(str, Enum):
    """Types of agents in the multi-agent system."""
    PLANNER     = "planner"
    CITIZEN     = "citizen"
    NGO         = "ngo"
    ADVERSARY   = "adversary"


@unique
class EventType(str, Enum):
    """Types of events that agents can generate."""
    COMPLAINT       = "complaint"
    RESOURCE_OFFER  = "resource_offer"
    DISRUPTION      = "disruption"
    COOPERATION     = "cooperation"
    PROTEST         = "protest"
    AID_DELIVERY    = "aid_delivery"


# Stage ordering for protocol enforcement
STAGE_SEQUENCE: Final[list] = [
    ActionType.ANALYZE,
    ActionType.ALLOCATE,
    ActionType.EXECUTE,
]
