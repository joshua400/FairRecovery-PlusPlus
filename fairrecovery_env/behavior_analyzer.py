"""
FairRecovery++ — Behavior Pattern Analyzer.

Analyzes interaction logs from multi-agent system to extract
behavioral patterns, detect anomalies, and compute risk levels.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from .constants import COMPLAINT_RATE_HIGH, RISK_THRESHOLD, EventType


@dataclass
class ZoneBehaviorProfile:
    """Behavioral profile for a single zone."""
    zone_id: int
    complaint_rate: float = 0.0
    disruption_count: int = 0
    cooperation_count: int = 0
    protest_count: int = 0
    stress_level: float = 0.0
    risk_level: str = "low"
    is_neglected: bool = False
    dominant_issue: str = "none"

    def to_dict(self) -> Dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in {
            "zone_id": self.zone_id, "complaint_rate": self.complaint_rate,
            "disruption_count": self.disruption_count, "stress_level": self.stress_level,
            "risk_level": self.risk_level, "is_neglected": self.is_neglected,
        }.items()}


@dataclass
class SystemPattern:
    """System-wide behavioral pattern."""
    frequent_zone: int = -1
    overall_complaint_rate: float = 0.0
    adversarial_active: bool = False
    stressed_zones: List[int] = field(default_factory=list)
    neglected_zones: List[int] = field(default_factory=list)
    cooperation_zones: List[int] = field(default_factory=list)
    pattern_summary: str = ""

    def to_dict(self) -> Dict:
        return {"frequent_zone": self.frequent_zone, "adversarial_active": self.adversarial_active,
                "stressed_zones": self.stressed_zones, "neglected_zones": self.neglected_zones,
                "overall_complaint_rate": round(self.overall_complaint_rate, 3),
                "pattern_summary": self.pattern_summary}


class BehaviorAnalyzer:
    """Analyzes interaction logs to extract behavioral patterns."""

    def __init__(self, num_zones: int) -> None:
        self.num_zones = num_zones
        self._zone_complaints: Dict[int, int] = defaultdict(int)
        self._zone_disruptions: Dict[int, int] = defaultdict(int)
        self._zone_cooperations: Dict[int, int] = defaultdict(int)
        self._zone_protests: Dict[int, int] = defaultdict(int)
        self._zone_event_counts: Dict[int, int] = defaultdict(int)
        self._total_events = 0
        self._adversarial_detected = False

    def ingest(self, events: List) -> None:
        """Ingest a batch of interaction log entries."""
        for event in events:
            zone_id = getattr(event, "zone_id", -1)
            event_type = getattr(event, "event_type", "")
            if zone_id < 0 or zone_id >= self.num_zones:
                continue
            self._total_events += 1
            self._zone_event_counts[zone_id] += 1
            if event_type == EventType.COMPLAINT.value:
                self._zone_complaints[zone_id] += 1
            elif event_type == EventType.DISRUPTION.value:
                self._zone_disruptions[zone_id] += 1
                self._adversarial_detected = True
            elif event_type in (EventType.COOPERATION.value, EventType.AID_DELIVERY.value):
                self._zone_cooperations[zone_id] += 1
            elif event_type == EventType.PROTEST.value:
                self._zone_protests[zone_id] += 1

    def analyze_zone(self, zone_id: int) -> ZoneBehaviorProfile:
        """Generate behavioral profile for a single zone."""
        total = self._zone_event_counts.get(zone_id, 0)
        complaints = self._zone_complaints.get(zone_id, 0)
        disruptions = self._zone_disruptions.get(zone_id, 0)
        cooperations = self._zone_cooperations.get(zone_id, 0)
        protests = self._zone_protests.get(zone_id, 0)
        rate = complaints / max(1, total) if total > 0 else 0.0
        stress = min(1.0, (complaints * 0.3 + disruptions * 0.5 + protests * 0.4) / max(1, total + 1))
        risk = "high" if stress >= RISK_THRESHOLD else ("medium" if stress >= RISK_THRESHOLD * 0.5 else "low")
        neglected = rate >= COMPLAINT_RATE_HIGH and cooperations == 0
        issues = {"complaints": complaints, "disruptions": disruptions, "protests": protests}
        dominant = max(issues, key=issues.get) if any(issues.values()) else "none"
        return ZoneBehaviorProfile(zone_id=zone_id, complaint_rate=rate, disruption_count=disruptions,
                                   cooperation_count=cooperations, protest_count=protests, stress_level=stress,
                                   risk_level=risk, is_neglected=neglected, dominant_issue=dominant)

    def analyze_system(self) -> SystemPattern:
        """Generate system-wide behavioral pattern analysis."""
        profiles = [self.analyze_zone(i) for i in range(self.num_zones)]
        complaints_map = {z.zone_id: z.complaint_rate for z in profiles}
        freq_zone = max(complaints_map, key=complaints_map.get) if complaints_map else -1
        total_comp = sum(self._zone_complaints.values())
        overall = total_comp / max(1, self._total_events)
        stressed = [z.zone_id for z in profiles if z.risk_level == "high"]
        neglected = [z.zone_id for z in profiles if z.is_neglected]
        coop = [z.zone_id for z in profiles if z.cooperation_count > 0]
        parts = []
        if stressed: parts.append(f"High-stress: {stressed}")
        if neglected: parts.append(f"Neglected: {neglected}")
        if self._adversarial_detected: parts.append("ADVERSARIAL DETECTED")
        return SystemPattern(frequent_zone=freq_zone, overall_complaint_rate=overall,
                             adversarial_active=self._adversarial_detected, stressed_zones=stressed,
                             neglected_zones=neglected, cooperation_zones=coop,
                             pattern_summary=" | ".join(parts) if parts else "Stable")

    def get_risk_levels(self) -> List[float]:
        """Return risk levels for all zones as floats [0, 1]."""
        return [self.analyze_zone(i).stress_level for i in range(self.num_zones)]

    def reset(self) -> None:
        self._zone_complaints.clear()
        self._zone_disruptions.clear()
        self._zone_cooperations.clear()
        self._zone_protests.clear()
        self._zone_event_counts.clear()
        self._total_events = 0
        self._adversarial_detected = False
