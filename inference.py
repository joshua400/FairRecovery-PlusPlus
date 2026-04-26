"""
FairRecovery++ — Advanced Inference & LLM Connectivity.

Updated with Phase-Aware policies and 'service' field compatibility.
"""

from __future__ import annotations
import os
import json
import time
from typing import Optional, List
from huggingface_hub import InferenceClient
from fairrecovery_env.models import ResourceAllocation, FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.constants import ActionType, ResourceType

class TrainingLogger:
    """Logs state-action trajectories for future RL training."""
    def __init__(self, log_dir: str = "training_data"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_session = f"session_{int(time.time())}.jsonl"

    def log_step(self, obs: FairRecoveryObservation, action: FairRecoveryAction, reward: float):
        entry = {
            "observation": obs.model_dump(),
            "action": action.model_dump(),
            "reward": reward,
            "timestamp": time.time()
        }
        with open(os.path.join(self.log_dir, self.current_session), "a") as f:
            f.write(json.dumps(entry) + "\n")

class HFInferencePolicy:
    """Live LLM Agent using Hugging Face Inference API."""
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct", token: Optional[str] = None):
        self.client = InferenceClient(model=model_id, token=token)

    def __call__(self, obs: FairRecoveryObservation) -> FairRecoveryAction:
        if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)

        prompt = self._build_prompt(obs)
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            content = response.choices[0].message.content
            return self._parse_response(content, obs)
        except Exception as e:
            return FairRecoveryAction(action_type=ActionType.ANALYZE, reasoning=f"API Error: {str(e)}")

    def _build_prompt(self, obs: FairRecoveryObservation) -> str:
        # FIX: z.service_level -> z.service
        zones_info = "\n".join([f"Zone {z.zone_id}: Damage={z.damage:.2f}, Vulnerability={z.vulnerable_ratio:.2f}, Svc={z.service:.2f}" for z in obs.zones])
        return f"""
You are an Emergency Recovery Agent. 
Environment: {zones_info}
Day: {obs.day}
Budget Left: {obs.budget_left:.2f}

Goal: Maximize Utility AND Fairness.
Format your response as valid JSON:
{{"action_type": "analyze"|"allocate"|"execute", "zone": <int>, "reasoning": "<str>"}}
"""

    def _parse_response(self, content: str, obs: FairRecoveryObservation) -> FairRecoveryAction:
        try:
            match = __import__("re").search(r"\{.*\}", content, __import__("re").DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(content)
            a_type = ActionType(data["action_type"].lower())
            allocs = None
            if a_type == ActionType.ALLOCATE:
                allocs = [ResourceAllocation(zone=data.get("zone", 0), resource=ResourceType.MEDICAL)]
            return FairRecoveryAction(
                action_type=a_type,
                critical_zones=[data.get("zone", 0)] if a_type == ActionType.ANALYZE else None,
                allocations=allocs,
                reasoning=data.get("reasoning", "LLM decision.")
            )
        except:
            return FairRecoveryAction(action_type=ActionType.EXECUTE, reasoning="Parse failed.")

def _get_phase_action(obs: FairRecoveryObservation) -> ActionType:
    num_steps = len(obs.action_history)
    cycle_pos = num_steps % 3
    if cycle_pos == 0: return ActionType.ANALYZE
    if cycle_pos == 1: return ActionType.ALLOCATE
    return ActionType.EXECUTE

def greedy_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)
    a_type = _get_phase_action(obs)
    if a_type == ActionType.ANALYZE: return FairRecoveryAction(action_type=a_type, critical_zones=[0])
    if a_type == ActionType.ALLOCATE: return FairRecoveryAction(action_type=a_type, allocations=[ResourceAllocation(zone=0, resource=ResourceType.MEDICAL)])
    return FairRecoveryAction(action_type=ActionType.EXECUTE)

def fairness_aware_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    if obs.day > 10: return FairRecoveryAction(action_type=ActionType.SUBMIT)
    a_type = _get_phase_action(obs)
    v_zone = sorted(range(len(obs.zones)), key=lambda i: obs.zones[i].vulnerable_ratio, reverse=True)[0]
    if a_type == ActionType.ANALYZE: return FairRecoveryAction(action_type=a_type, critical_zones=[v_zone])
    if a_type == ActionType.ALLOCATE: return FairRecoveryAction(action_type=a_type, allocations=[ResourceAllocation(zone=v_zone, resource=ResourceType.MEDICAL)])
    return FairRecoveryAction(action_type=ActionType.EXECUTE)
