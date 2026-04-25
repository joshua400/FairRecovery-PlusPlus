"""
FairRecovery++ — Multi-Agent System.

Dynamic agents that interact with the environment and create
non-static challenges for the LLM planner to adapt to.

Agent Types:
  - CitizenAgent:     Reacts to recovery progress, generates complaints/satisfaction
  - NGOAgent:         Provides resources, may conflict with planner priorities
  - AdversarialAgent: Exploits weaknesses, disrupts recovery, creates unfair outcomes
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import (
    ADVERSARY_DISRUPTION_CHANCE,
    CITIZEN_SATISFACTION_DECAY,
    NGO_RESOURCE_MULTIPLIER,
    VULNERABILITY_THRESHOLD,
    AgentType,
    EventType,
)


@dataclass
class InteractionLog:
    """Single interaction event from an agent."""
    agent_type: str
    event_type: str
    zone_id: int
    intensity: float
    timestamp: int
    message: str = ""


@dataclass
class AgentState:
    """Base state for any agent."""
    agent_type: AgentType
    zone_id: int
    active: bool = True
    history: List[InteractionLog] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Citizen Agent
# ──────────────────────────────────────────────────────────────────────────────
class CitizenAgent:
    """
    Citizen agents react to recovery progress and generate complaints/satisfaction.

    Behaviors:
      - Satisfaction decays if zone is neglected
      - Complaints increase in vulnerable zones with low service
      - Protests trigger when satisfaction drops below threshold
    """

    def __init__(self, zone_id: int, vulnerability: float, seed: int = 42) -> None:
        self.zone_id = zone_id
        self.vulnerability = vulnerability
        self.satisfaction = 0.5
        self.complaint_count = 0
        self._rng = random.Random(seed + zone_id)

    def step(self, service: float, damage: float, day: int) -> List[InteractionLog]:
        """Generate citizen events based on current zone state."""
        events: List[InteractionLog] = []

        # Service improvement → satisfaction increases
        service_delta = service - (1.0 - damage)
        if service > 0.5:
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
        else:
            self.satisfaction = max(0.0, self.satisfaction - CITIZEN_SATISFACTION_DECAY)

        # Complaints when service is low + vulnerability is high
        complaint_probability = (1.0 - service) * self.vulnerability
        if self._rng.random() < complaint_probability:
            self.complaint_count += 1
            intensity = min(1.0, complaint_probability * 1.5)
            events.append(InteractionLog(
                agent_type=AgentType.CITIZEN.value,
                event_type=EventType.COMPLAINT.value,
                zone_id=self.zone_id,
                intensity=intensity,
                timestamp=day,
                message=f"Citizens in zone {self.zone_id} complain about inadequate "
                        f"service (service={service:.2f}, satisfaction={self.satisfaction:.2f})",
            ))

        # Protests when satisfaction drops very low
        if self.satisfaction < 0.2 and self._rng.random() < 0.4:
            events.append(InteractionLog(
                agent_type=AgentType.CITIZEN.value,
                event_type=EventType.PROTEST.value,
                zone_id=self.zone_id,
                intensity=0.8,
                timestamp=day,
                message=f"Protest in zone {self.zone_id}! "
                        f"Satisfaction critically low ({self.satisfaction:.2f})",
            ))

        return events

    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "vulnerability": round(self.vulnerability, 3),
            "satisfaction": round(self.satisfaction, 3),
            "complaint_count": self.complaint_count,
        }


# ──────────────────────────────────────────────────────────────────────────────
# NGO Agent
# ──────────────────────────────────────────────────────────────────────────────
class NGOAgent:
    """
    NGO agents provide supplementary resources but may conflict with planner priorities.

    Behaviors:
      - Offer resources to high-vulnerability zones
      - May conflict by targeting zones the planner is neglecting
      - Cooperation bonus when aligned with planner strategy
    """

    def __init__(self, focus_zones: List[int], seed: int = 42) -> None:
        self.focus_zones = focus_zones
        self.resources_delivered = 0
        self.conflicts = 0
        self._rng = random.Random(seed + 1000)

    def step(
        self,
        zone_services: List[float],
        zone_vulnerabilities: List[float],
        planner_target_zones: List[int],
        day: int,
    ) -> List[InteractionLog]:
        """Generate NGO events based on zone states and planner actions."""
        events: List[InteractionLog] = []

        # Find most neglected vulnerable zone
        worst_zone = -1
        worst_score = float("inf")
        for i, (svc, vuln) in enumerate(zip(zone_services, zone_vulnerabilities)):
            if vuln >= VULNERABILITY_THRESHOLD:
                score = svc - vuln  # lower = more neglected
                if score < worst_score:
                    worst_score = score
                    worst_zone = i

        if worst_zone >= 0 and self._rng.random() < 0.6:
            # Check if aligned with planner
            is_aligned = worst_zone in planner_target_zones
            if is_aligned:
                events.append(InteractionLog(
                    agent_type=AgentType.NGO.value,
                    event_type=EventType.COOPERATION.value,
                    zone_id=worst_zone,
                    intensity=0.6,
                    timestamp=day,
                    message=f"NGO cooperates: delivering aid to zone {worst_zone} "
                            f"(aligned with planner)",
                ))
                self.resources_delivered += 1
            else:
                # Conflict — NGO targets a zone the planner is ignoring
                events.append(InteractionLog(
                    agent_type=AgentType.NGO.value,
                    event_type=EventType.AID_DELIVERY.value,
                    zone_id=worst_zone,
                    intensity=0.5,
                    timestamp=day,
                    message=f"NGO independently aids zone {worst_zone} "
                            f"(conflict: planner targeting {planner_target_zones})",
                ))
                self.conflicts += 1

        return events

    def to_dict(self) -> Dict:
        return {
            "focus_zones": self.focus_zones,
            "resources_delivered": self.resources_delivered,
            "conflicts": self.conflicts,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial Agent
# ──────────────────────────────────────────────────────────────────────────────
class AdversarialAgent:
    """
    Adversarial agent that exploits weaknesses and disrupts recovery.

    Behaviors:
      - Targets low-service zones to cause further disruption
      - Can corrupt resource deliveries (reduce effectiveness)
      - Creates unfair outcomes by exacerbating inequalities
    """

    def __init__(self, seed: int = 42) -> None:
        self.disruptions_caused = 0
        self.total_damage = 0.0
        self._rng = random.Random(seed + 2000)
        self._active = True

    def step(
        self,
        zone_services: List[float],
        zone_damages: List[float],
        zone_vulnerabilities: List[float],
        day: int,
    ) -> List[InteractionLog]:
        """Generate adversarial events targeting the weakest zones."""
        events: List[InteractionLog] = []

        if not self._active:
            return events

        # Target the most vulnerable zone with lowest service
        target_zone = -1
        target_score = -1.0
        for i, (svc, dmg, vuln) in enumerate(
            zip(zone_services, zone_damages, zone_vulnerabilities)
        ):
            # Score: high vulnerability + low service = attractive target
            exploit_score = vuln * (1.0 - svc) * dmg
            if exploit_score > target_score:
                target_score = exploit_score
                target_zone = i

        if target_zone >= 0 and self._rng.random() < ADVERSARY_DISRUPTION_CHANCE:
            self.disruptions_caused += 1
            intensity = min(1.0, target_score * 1.5)
            self.total_damage += intensity * 0.1

            events.append(InteractionLog(
                agent_type=AgentType.ADVERSARY.value,
                event_type=EventType.DISRUPTION.value,
                zone_id=target_zone,
                intensity=intensity,
                timestamp=day,
                message=f"Adversarial disruption in zone {target_zone}! "
                        f"Service delivery compromised (intensity={intensity:.2f})",
            ))

        return events

    def deactivate(self) -> None:
        """Deactivate the adversary (e.g., when detected by planner)."""
        self._active = False

    def to_dict(self) -> Dict:
        return {
            "active": self._active,
            "disruptions_caused": self.disruptions_caused,
            "total_damage": round(self.total_damage, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Agent Manager
# ──────────────────────────────────────────────────────────────────────────────
class MultiAgentManager:
    """
    Coordinates all agents in the system.

    Manages citizen, NGO, and adversarial agents. Collects interaction logs.
    Applies agent effects to the world state.
    """

    def __init__(self, num_zones: int, zone_vulnerabilities: List[float], seed: int = 42) -> None:
        self.citizens: List[CitizenAgent] = [
            CitizenAgent(zone_id=i, vulnerability=vuln, seed=seed)
            for i, vuln in enumerate(zone_vulnerabilities)
        ]

        # NGO focuses on vulnerable zones
        vuln_zones = [
            i for i, v in enumerate(zone_vulnerabilities)
            if v >= VULNERABILITY_THRESHOLD
        ]
        self.ngo = NGOAgent(focus_zones=vuln_zones or [0], seed=seed)
        self.adversary = AdversarialAgent(seed=seed)

        self.interaction_log: List[InteractionLog] = []
        self._seed = seed

    def step(
        self,
        zone_services: List[float],
        zone_damages: List[float],
        zone_vulnerabilities: List[float],
        planner_target_zones: List[int],
        day: int,
    ) -> List[InteractionLog]:
        """Run all agents for one step and collect events."""
        all_events: List[InteractionLog] = []

        # Citizens react to their zone state
        for citizen in self.citizens:
            idx = citizen.zone_id
            if idx < len(zone_services):
                events = citizen.step(
                    service=zone_services[idx],
                    damage=zone_damages[idx],
                    day=day,
                )
                all_events.extend(events)

        # NGO responds to neglected zones
        ngo_events = self.ngo.step(
            zone_services=zone_services,
            zone_vulnerabilities=zone_vulnerabilities,
            planner_target_zones=planner_target_zones,
            day=day,
        )
        all_events.extend(ngo_events)

        # Adversary targets weak points
        adv_events = self.adversary.step(
            zone_services=zone_services,
            zone_damages=zone_damages,
            zone_vulnerabilities=zone_vulnerabilities,
            day=day,
        )
        all_events.extend(adv_events)

        # Store in interaction log
        self.interaction_log.extend(all_events)

        return all_events

    def apply_disruptions(
        self,
        zone_services: List[float],
        zone_damages: List[float],
    ) -> List[float]:
        """Apply adversarial disruption effects to zone services."""
        modified_services = list(zone_services)
        for log in self.interaction_log:
            if log.event_type == EventType.DISRUPTION.value:
                idx = log.zone_id
                if 0 <= idx < len(modified_services):
                    # Reduce service by disruption intensity * 0.1
                    reduction = log.intensity * 0.1
                    modified_services[idx] = max(0.0, modified_services[idx] - reduction)
        return modified_services

    def apply_ngo_aid(
        self,
        zone_services: List[float],
    ) -> List[float]:
        """Apply NGO aid effects (cooperation boosts service)."""
        modified_services = list(zone_services)
        for log in self.interaction_log:
            if log.event_type in (EventType.COOPERATION.value, EventType.AID_DELIVERY.value):
                idx = log.zone_id
                if 0 <= idx < len(modified_services):
                    boost = 0.05 * NGO_RESOURCE_MULTIPLIER
                    modified_services[idx] = min(1.0, modified_services[idx] + boost)
        return modified_services

    def get_citizen_satisfactions(self) -> List[float]:
        """Return satisfaction levels for all citizen agents."""
        return [c.satisfaction for c in self.citizens]

    def get_complaint_rate(self) -> float:
        """Average complaint rate across all zones."""
        if not self.citizens:
            return 0.0
        total = sum(c.complaint_count for c in self.citizens)
        return total / len(self.citizens)

    def get_active_agent_count(self) -> int:
        """Count of all active agents."""
        count = len(self.citizens) + 1  # citizens + NGO
        if self.adversary._active:
            count += 1
        return count

    def get_adversarial_event_count(self) -> int:
        """Count adversarial events in the log."""
        return sum(
            1 for log in self.interaction_log
            if log.agent_type == AgentType.ADVERSARY.value
        )

    def to_dict(self) -> Dict:
        return {
            "citizens": [c.to_dict() for c in self.citizens],
            "ngo": self.ngo.to_dict(),
            "adversary": self.adversary.to_dict(),
            "interaction_log_size": len(self.interaction_log),
        }
