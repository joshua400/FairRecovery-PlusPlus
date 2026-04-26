"""
FairRecovery++ — Constants and Configuration.

All configurable values are centralized here. No hardcoded magic numbers elsewhere.
"""

from enum import Enum, unique
from typing import Final


# ──────────────────────────────────────────────────────────────────────────────
# Environment Metadata
# ──────────────────────────────────────────────────────────────────────────────
ENV_NAME: Final[str] = "fair_recovery_gym"
ENV_VERSION: Final[str] = "2.0.0"
ENV_DESCRIPTION: Final[str] = (
    "An OpenEnv environment for fair disaster recovery. Agents must allocate "
    "scarce resources across zones while balancing efficiency and social equity."
)

# ──────────────────────────────────────────────────────────────────────────────
# Episode Configuration
# ──────────────────────────────────────────────────────────────────────────────
MAX_STEPS_PER_EPISODE: Final[int] = 30  # 10 days * 3 phases/day
MAX_DAYS: Final[int] = 10
BUDGET_INITIAL: Final[float] = 1.0  # Normalized total budget
NUM_ZONES: Final[int] = 5

# ──────────────────────────────────────────────────────────────────────────────
# Reward Weighting (The Honest Truth Formula)
# ──────────────────────────────────────────────────────────────────────────────
WEIGHT_UTILITY: Final[float] = 0.4
WEIGHT_FAIRNESS: Final[float] = 0.4
WEIGHT_SAFETY: Final[float] = 0.2

# Grader score bounds (strictly between 0 and 1)
GRADER_SCORE_MIN: Final[float] = 0.01
GRADER_SCORE_MAX: Final[float] = 0.99

# ──────────────────────────────────────────────────────────────────────────────
# Penalties and Rewards
# ──────────────────────────────────────────────────────────────────────────────
PENALTY_SAFETY_VIOLATION: Final[float] = -0.05
PENALTY_REPEATED_ACTION: Final[float] = -0.02
PENALTY_BUDGET_OVERRUN: Final[float] = -0.10

REWARD_STABILITY: Final[float] = 0.01  # Per step for avoiding deterioration

# ──────────────────────────────────────────────────────────────────────────────
# Resource Costs (Normalized)
# ──────────────────────────────────────────────────────────────────────────────
COST_MEDICAL: Final[float] = 0.05
COST_WATER: Final[float] = 0.03
COST_POWER: Final[float] = 0.07

# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────
@unique
class ResourceType(str, Enum):
    """Available recovery resources."""
    MEDICAL = "medical"
    WATER = "water"
    POWER = "power"
    NONE = "none"

@unique
class ActionType(str, Enum):
    """Available agent actions."""
    ANALYZE = "analyze"
    ALLOCATE = "allocate"
    EXECUTE = "execute"
    SUBMIT = "submit"
    NOOP = "noop"

@unique
class Difficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@unique
class TaskID(str, Enum):
    """Identifiers for the scenarios."""
    FLOOD_EASY = "flood_easy"
    EARTHQUAKE_MEDIUM = "earthquake_medium"
    MULTI_DISASTER_HARD = "multi_disaster_hard"
