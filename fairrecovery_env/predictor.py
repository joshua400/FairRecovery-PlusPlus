"""
FairRecovery++ — Predictive Response Engine.

Predicts next events, risk levels, and enables proactive planning.
Uses behavioral patterns to forecast likely disruptions and needs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from .behavior_analyzer import BehaviorAnalyzer, SystemPattern


@dataclass
class Prediction:
    """Prediction for the next step/day."""
    likely_zone: int = -1
    risk: str = "low"
    predicted_event: str = "none"
    confidence: float = 0.0
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {"likely_zone": self.likely_zone, "risk": self.risk,
                "predicted_event": self.predicted_event,
                "confidence": round(self.confidence, 3),
                "recommendation": self.recommendation}


class Predictor:
    """
    Predicts next events based on behavioral patterns.
    
    Enables proactive planning by the LLM agent — the agent can
    anticipate disruptions and pre-allocate resources.
    """

    def __init__(self, analyzer: BehaviorAnalyzer) -> None:
        self._analyzer = analyzer

    def predict_next(self, zone_services: List[float], zone_damages: List[float],
                     zone_vulnerabilities: List[float], day: int) -> Prediction:
        """Predict the most likely next event and its zone."""
        pattern = self._analyzer.analyze_system()
        
        # Find highest-risk zone
        risk_levels = self._analyzer.get_risk_levels()
        if not risk_levels:
            return Prediction()
        
        max_risk_zone = max(range(len(risk_levels)), key=lambda i: risk_levels[i])
        max_risk = risk_levels[max_risk_zone]
        
        # Predict event type based on pattern
        if pattern.adversarial_active:
            predicted_event = "disruption"
            confidence = 0.7
            recommendation = f"Shield zone {max_risk_zone} against adversarial disruption"
        elif pattern.neglected_zones:
            target = pattern.neglected_zones[0]
            predicted_event = "protest"
            confidence = 0.6
            recommendation = f"Prioritize neglected zone {target} to prevent protest"
            max_risk_zone = target
        elif pattern.stressed_zones:
            target = pattern.stressed_zones[0]
            predicted_event = "complaint_surge"
            confidence = 0.5
            recommendation = f"Increase service to stressed zone {target}"
            max_risk_zone = target
        else:
            predicted_event = "stable"
            confidence = 0.4
            recommendation = "Continue current strategy"

        risk_label = "high" if max_risk > 0.6 else ("medium" if max_risk > 0.3 else "low")
        
        return Prediction(likely_zone=max_risk_zone, risk=risk_label,
                          predicted_event=predicted_event, confidence=confidence,
                          recommendation=recommendation)

    def predict_all_zones(self, zone_services: List[float], zone_damages: List[float],
                          zone_vulnerabilities: List[float]) -> List[Dict]:
        """Generate per-zone risk predictions."""
        risk_levels = self._analyzer.get_risk_levels()
        results = []
        for i in range(len(zone_services)):
            risk = risk_levels[i] if i < len(risk_levels) else 0.0
            # Combine behavioral risk with state-based risk
            state_risk = zone_damages[i] * zone_vulnerabilities[i] * (1.0 - zone_services[i])
            combined = 0.6 * risk + 0.4 * state_risk
            results.append({"zone_id": i, "behavioral_risk": round(risk, 3),
                           "state_risk": round(state_risk, 3),
                           "combined_risk": round(combined, 3)})
        return results

    def evaluate_adaptation(self, prediction: Prediction, actual_events: List,
                            planner_actions: List[int]) -> float:
        """Score how well the planner adapted to predicted events. Returns [0, 1]."""
        if prediction.likely_zone < 0:
            return 0.5
        
        # Did the planner act on the prediction?
        acted_on_predicted = prediction.likely_zone in planner_actions
        
        # Was the prediction accurate?
        actual_zones = {getattr(e, "zone_id", -1) for e in actual_events}
        prediction_accurate = prediction.likely_zone in actual_zones
        
        score = 0.0
        if acted_on_predicted and prediction_accurate:
            score = 1.0  # Proactive + correct
        elif acted_on_predicted and not prediction_accurate:
            score = 0.6  # Proactive but prediction wrong
        elif not acted_on_predicted and prediction_accurate:
            score = 0.2  # Missed predicted event
        else:
            score = 0.5  # Neither acted nor needed
        
        return score * prediction.confidence
