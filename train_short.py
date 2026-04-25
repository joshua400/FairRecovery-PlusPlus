from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fairrecovery_env.constants import RESOURCE_COSTS
from fairrecovery_env.models import AllocationItem, FairRecoveryAction, FairRecoveryObservation
from server.fairrecovery_environment import FairRecoveryEnvironment


@dataclass
class EpisodeResult:
    total_reward: float
    final_fairness: float
    steps: int
    breakdown: List[Dict[str, float]]


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = values / max(temperature, 1e-6)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _pick_zone(obs: FairRecoveryObservation, weights: np.ndarray, explore: float) -> int:
    zone_scores = np.array(
        [z.damage * z.vulnerable_ratio + 0.2 * z.risk_level for z in obs.zones], dtype=float
    )
    logits = zone_scores + weights
    probs = _softmax(logits, temperature=max(0.5, explore))
    return int(np.random.choice(np.arange(len(obs.zones)), p=probs))


def _resource_for_budget(budget_left: float) -> str:
    for resource in ("medical", "water", "power"):
        if budget_left >= RESOURCE_COSTS[resource]:
            return resource
    return "power"


def run_episode(
    env: FairRecoveryEnvironment,
    weights: np.ndarray,
    difficulty: str,
    explore: float,
    baseline: bool = False,
) -> EpisodeResult:
    obs = env.reset(difficulty=difficulty)
    total_reward = 0.0
    step_count = 0
    breakdown_rows: List[Dict[str, float]] = []

    while not obs.done:
        if obs.step_stage == "analyze":
            if baseline:
                critical = random.sample(range(len(obs.zones)), k=min(2, len(obs.zones)))
            else:
                z0 = _pick_zone(obs, weights, explore)
                z1 = _pick_zone(obs, weights, explore)
                critical = list(dict.fromkeys([z0, z1]))
            action = FairRecoveryAction(
                action_type="analyze",
                critical_zones=critical,
                reasoning="Short training policy",
            )
        elif obs.step_stage == "allocate":
            if baseline:
                zone = random.randint(0, len(obs.zones) - 1)
            else:
                zone = _pick_zone(obs, weights, explore)
            action = FairRecoveryAction(
                action_type="allocate",
                allocations=[AllocationItem(zone=zone, resource=_resource_for_budget(obs.budget_left))],
            )
        elif obs.step_stage == "execute":
            action = FairRecoveryAction(action_type="execute")
        else:
            action = FairRecoveryAction(action_type="submit")

        obs = env.step(action)
        total_reward += obs.reward
        step_count += 1

        if obs.info:
            breakdown_rows.append(
                {
                    "reward": float(obs.info.get("reward", 0.0)),
                    "utility": float(obs.info.get("utility", 0.0)),
                    "fairness": float(obs.info.get("fairness", 0.0)),
                    "adapt": float(obs.info.get("adapt", 0.0)),
                    "stability": float(obs.info.get("stability", 0.0)),
                    "safety": float(obs.info.get("safety", 0.0)),
                }
            )

        if step_count > 80:
            break

    if not obs.done:
        obs = env.step(FairRecoveryAction(action_type="submit"))
        total_reward += obs.reward

    return EpisodeResult(
        total_reward=float(total_reward),
        final_fairness=float(obs.fairness_score),
        steps=step_count,
        breakdown=breakdown_rows,
    )


def _moving_avg(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def train_short(episodes: int, difficulty: str, seed: int) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)

    env = FairRecoveryEnvironment()
    weights = np.zeros(5, dtype=float)
    best_reward = -math.inf
    best_weights = weights.copy()

    train_rewards: List[float] = []
    train_fairness: List[float] = []
    step_rewards: List[float] = []
    step_fairness: List[float] = []
    step_utility: List[float] = []

    baseline_eval: List[float] = []
    with np.errstate(all="ignore"):
        for _ in range(5):
            baseline_ep = run_episode(env, weights=np.zeros_like(weights), difficulty=difficulty, explore=1.0, baseline=True)
            baseline_eval.append(float(np.clip((baseline_ep.total_reward + 2.0) / 6.0, 0.0, 1.0)))

    for ep in range(episodes):
        explore = max(0.4, 1.0 - (ep / max(episodes, 1)))
        candidate = best_weights + np.random.normal(0.0, 0.2, size=best_weights.shape)
        result = run_episode(env, weights=candidate, difficulty=difficulty, explore=explore, baseline=False)

        normalized_reward = float(np.clip((result.total_reward + 2.0) / 6.0, 0.0, 1.0))
        train_rewards.append(normalized_reward)
        train_fairness.append(float(np.clip((result.final_fairness + 1.0) / 2.0, 0.0, 1.0)))

        for row in result.breakdown:
            step_rewards.append(row["reward"])
            step_fairness.append(row["fairness"])
            step_utility.append(row["utility"])

        if result.total_reward > best_reward:
            best_reward = result.total_reward
            best_weights = candidate.copy()

    trained_eval: List[float] = []
    for _ in range(5):
        tr = run_episode(env, weights=best_weights, difficulty=difficulty, explore=0.55, baseline=False)
        trained_eval.append(float(np.clip((tr.total_reward + 2.0) / 6.0, 0.0, 1.0)))

    # Plot 1: reward vs steps
    plt.figure(figsize=(9, 4.5))
    x = np.arange(1, len(step_rewards) + 1)
    plt.plot(x, _moving_avg(step_rewards, window=7), label="Reward (moving avg)", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Normalized Reward (0-1)")
    plt.title("Reward vs Steps")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_vs_steps.png", dpi=140)
    plt.close()

    # Plot 2: fairness vs steps
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, _moving_avg(step_fairness, window=7), color="#1f77b4", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Fairness (0-1)")
    plt.title("Fairness vs Steps")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fairness_vs_steps.png", dpi=140)
    plt.close()

    # Required transparency plot: utility vs fairness
    plt.figure(figsize=(6, 6))
    plt.scatter(step_utility, step_fairness, alpha=0.55, s=18)
    plt.xlabel("Utility (0-1)")
    plt.ylabel("Fairness (0-1)")
    plt.title("Utility vs Fairness")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("utility_vs_fairness.png", dpi=140)
    plt.close()

    summary = {
        "episodes": episodes,
        "baseline_avg_reward": round(float(np.mean(baseline_eval)), 3),
        "trained_avg_reward": round(float(np.mean(trained_eval)), 3),
        "baseline_eval": baseline_eval,
        "trained_eval": trained_eval,
        "best_weights": [round(float(v), 4) for v in best_weights.tolist()],
    }
    Path("training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Short FairRecovery training (light but real).")
    parser.add_argument("--episodes", type=int, default=15, help="Train episodes (10-20 suggested).")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = train_short(episodes=args.episodes, difficulty=args.difficulty, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
