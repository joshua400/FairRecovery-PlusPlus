"""
FairRecovery++ — Python Client.

Typed interface for interacting with the FairRecovery++ environment server.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import requests

from fairrecovery_env.models import FairRecoveryAction, FairRecoveryObservation


class FairRecoveryEnv:
    """HTTP client for FairRecovery++ environment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    def __enter__(self) -> "FairRecoveryEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._session.close()

    def close(self) -> None:
        self._session.close()

    def reset(self, difficulty: str = "medium",
              episode_id: Optional[str] = None) -> FairRecoveryObservation:
        params: Dict[str, str] = {"difficulty": difficulty}
        if episode_id:
            params["episode_id"] = episode_id
        resp = self._session.post(f"{self._base_url}/reset", params=params,
                                   timeout=self._timeout)
        resp.raise_for_status()
        return FairRecoveryObservation(**resp.json())

    def step(self, action: FairRecoveryAction) -> FairRecoveryObservation:
        resp = self._session.post(f"{self._base_url}/step",
                                   data=action.model_dump_json(),
                                   timeout=self._timeout)
        resp.raise_for_status()
        return FairRecoveryObservation(**resp.json())

    def state(self) -> Dict:
        resp = self._session.get(f"{self._base_url}/state", timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict:
        resp = self._session.get(f"{self._base_url}/health", timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def run_episode(self, policy_fn: Callable, difficulty: str = "hard",
                    max_days: int = 5, verbose: bool = False) -> Dict:
        """Run a full episode with a policy function."""
        obs = self.reset(difficulty=difficulty)
        total_reward = 0.0
        steps = 0

        for day in range(max_days):
            for stage in ["analyze", "allocate", "execute"]:
                action = policy_fn(obs)
                obs = self.step(action)
                total_reward += obs.reward
                steps += 1
                if verbose and stage == "execute":
                    print(f"    Day {obs.day}: reward={obs.reward:+.3f} "
                          f"fair={obs.fairness_score:.3f} budget={obs.budget_left}")
                if obs.done:
                    break
            if obs.done:
                break

        if not obs.done:
            obs = self.step(FairRecoveryAction(action_type="submit"))
            total_reward += obs.reward

        return {
            "total_reward": total_reward,
            "final_fairness": obs.fairness_score,
            "grader_score": obs.grader_score,
            "steps": steps,
            "agent_events_seen": len(obs.agent_events),
        }
