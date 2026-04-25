"""
FairRecovery++ — World State.

CityState is the mutable simulation world for a single episode.
ZoneState tracks per-zone metrics. Neither imports from server/.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional

from .constants import RESOURCE_COSTS, RESOURCE_EFFECTS, VULNERABILITY_THRESHOLD


class ZoneState:
    """Single disaster zone — mutable during an episode."""
    __slots__ = ("zone_id", "damage", "service", "vulnerable_ratio", "citizen_satisfaction")

    def __init__(self, zone_id: int, damage: float, service: float,
                 vulnerable_ratio: float, citizen_satisfaction: float = 0.5) -> None:
        self.zone_id = zone_id
        self.damage = float(damage)
        self.service = float(service)
        self.vulnerable_ratio = float(vulnerable_ratio)
        self.citizen_satisfaction = float(citizen_satisfaction)

    def apply_resource(self, resource: str) -> None:
        """Apply resource effects, clamping to [0, 1]."""
        effects = RESOURCE_EFFECTS.get(resource, {})
        self.service = float(min(1.0, max(0.0, self.service + effects.get("service", 0.0))))
        self.damage = float(min(1.0, max(0.0, self.damage + effects.get("damage", 0.0))))

    def apply_disruption(self, intensity: float) -> None:
        """Apply adversarial disruption effects."""
        self.service = float(max(0.0, self.service - intensity * 0.1))
        self.damage = float(min(1.0, self.damage + intensity * 0.05))

    @property
    def is_vulnerable(self) -> bool:
        return self.vulnerable_ratio >= VULNERABILITY_THRESHOLD

    @property
    def recovery_priority(self) -> float:
        return self.damage * self.vulnerable_ratio

    def to_dict(self) -> Dict:
        return {"zone_id": self.zone_id, "damage": round(self.damage, 3),
                "service": round(self.service, 3), "vulnerable_ratio": round(self.vulnerable_ratio, 3),
                "citizen_satisfaction": round(self.citizen_satisfaction, 3)}

    def __repr__(self) -> str:
        return (f"Zone({self.zone_id}: dmg={self.damage:.2f}, svc={self.service:.2f}, "
                f"vuln={self.vulnerable_ratio:.2f}, sat={self.citizen_satisfaction:.2f})")


class CityState:
    """Full episode world state."""

    def __init__(self, task_config: Dict) -> None:
        zones_data = task_config.get("zones", [])
        self.zones: List[ZoneState] = [ZoneState(**z) for z in zones_data]
        self.initial_budget: float = float(task_config.get("initial_budget", 100.0))
        self.budget_left: float = self.initial_budget
        self.day: int = 0
        self.step_stage: str = "analyze"
        self.history: List[str] = []
        self.pending_allocations: List[Dict] = []
        self.violations_total: int = 0
        self._prev_services: List[float] = [z.service for z in self.zones]
        self._planner_target_zones: List[int] = []

    def snapshot_services(self) -> None:
        self._prev_services = [z.service for z in self.zones]

    def apply_allocations(self) -> List[str]:
        violations: List[str] = []
        for alloc in self.pending_allocations:
            zone_id = alloc.get("zone")
            resource = alloc.get("resource")
            if zone_id is None or not (0 <= zone_id < len(self.zones)):
                violations.append(f"invalid_zone:{zone_id}")
                self.violations_total += 1
                continue
            if resource not in RESOURCE_COSTS:
                violations.append(f"invalid_resource:{resource}")
                self.violations_total += 1
                continue
            cost = RESOURCE_COSTS[resource]
            if self.budget_left < cost:
                violations.append(f"budget_exceeded:zone{zone_id}:{resource}")
                self.violations_total += 1
                continue
            self.budget_left -= cost
            self.zones[zone_id].apply_resource(resource)
            self._planner_target_zones.append(zone_id)
        self.pending_allocations = []
        self.day += 1
        return violations

    def record(self, msg: str) -> None:
        self.history.append(f"Day {self.day}: {msg}")

    @property
    def prev_services(self) -> List[float]:
        return list(self._prev_services)

    @property
    def current_services(self) -> List[float]:
        return [z.service for z in self.zones]

    @property
    def current_damages(self) -> List[float]:
        return [z.damage for z in self.zones]

    @property
    def current_vulnerabilities(self) -> List[float]:
        return [z.vulnerable_ratio for z in self.zones]

    @property
    def planner_target_zones(self) -> List[int]:
        return list(set(self._planner_target_zones))

    @property
    def vulnerable_zones(self) -> List[ZoneState]:
        return [z for z in self.zones if z.is_vulnerable]

    @property
    def non_vulnerable_zones(self) -> List[ZoneState]:
        return [z for z in self.zones if not z.is_vulnerable]

    def to_dict(self) -> Dict:
        return {"zones": [z.to_dict() for z in self.zones], "day": self.day,
                "budget_left": round(self.budget_left, 2), "step_stage": self.step_stage,
                "history": self.history[-5:]}
