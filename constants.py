"""
FairRecovery — Constants and Configuration.

All configurable values are centralised here.
No hardcoded magic numbers anywhere else in the codebase.
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Final


# ──────────────────────────────────────────────────────────────────────────────
# Environment Metadata
# ──────────────────────────────────────────────────────────────────────────────
ENV_NAME: Final[str] = "fairrecovery"
ENV_VERSION: Final[str] = "1.0.0"
ENV_DESCRIPTION: Final[str] = (
    "A post-disaster city recovery RL environment where an LLM agent must "
    "allocate limited resources across zones, optimising both efficiency and "
    "fairness for vulnerable populations. Designed for RLVR training via "
    "TRL/GRPO with Unsloth."
)

# ──────────────────────────────────────────────────────────────────────────────
# Episode Configuration
# ──────────────────────────────────────────────────────────────────────────────
MAX_DAYS: Final[int] = 5          # episode length (days)
MAX_ACTIONS_PER_DAY: Final[int] = 3   # max allocations per execute step
MAX_STEPS_SAFETY_CAP: Final[int] = 50  # hard cap on total steps to prevent infinite loops
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
# Reward Weights
# ──────────────────────────────────────────────────────────────────────────────
REWARD_WEIGHTS: Final[dict] = {
    "exec": 1.0,   # service improvement
    "fair": 0.7,   # fairness (disparity reduction)
    "safe": 0.3,   # constraint satisfaction
}

# ──────────────────────────────────────────────────────────────────────────────
# Penalties
# ──────────────────────────────────────────────────────────────────────────────
PENALTY_INVALID_ZONE: Final[float] = -0.3
PENALTY_INVALID_RESOURCE: Final[float] = -0.3
PENALTY_BUDGET_EXCEEDED: Final[float] = -0.2
PENALTY_IGNORE_VULNERABLE: Final[float] = -0.3
PENALTY_WRONG_STAGE: Final[float] = -0.1
PENALTY_REPEATED_ACTION: Final[float] = -0.05

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────
VULNERABILITY_THRESHOLD: Final[float] = 0.6  # zones above this are "vulnerable"
SPAN_OVERLAP_THRESHOLD: Final[float] = 0.3

# Grader score bounds — strict open interval (never exactly 0 or 1)
GRADER_SCORE_MIN: Final[float] = 0.01
GRADER_SCORE_MAX: Final[float] = 0.99


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
      analyze  → allocate → execute  → (repeat MAX_DAYS times) → submit
    """
    ANALYZE  = "analyze"
    ALLOCATE = "allocate"
    EXECUTE  = "execute"
    SUBMIT   = "submit"
    NOOP     = "noop"


# Stage ordering for protocol enforcement
STAGE_SEQUENCE: Final[list] = [
    ActionType.ANALYZE,
    ActionType.ALLOCATE,
    ActionType.EXECUTE,
]
