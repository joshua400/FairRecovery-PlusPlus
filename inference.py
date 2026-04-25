"""
FairRecovery++ — Baseline Inference Script.

Run BEFORE training to capture pre-training behaviour.
Run AFTER training to compare trained vs baseline.

Policies:
  random  — random zone + resource selection
  greedy  — utility-maximising heuristic (ignores fairness — the WRONG policy)
  fair    — fairness-aware heuristic (the CORRECT policy baseline)

Usage:
    python inference.py --difficulty hard --episodes 5 --policy all
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List

import numpy as np

from client import FairRecoveryEnv
from fairrecovery_env.models import AllocationItem, FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.constants import RESOURCE_COSTS

BASE_URL = "http://localhost:8000"


def random_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Completely random policy — ignores all observations."""
    stage = obs.step_stage
    if stage == "analyze":
        n = len(obs.zones)
        return FairRecoveryAction(action_type="analyze",
            critical_zones=random.sample(range(n), min(2, n)), reasoning="Random.")
    elif stage == "allocate":
        n = len(obs.zones)
        return FairRecoveryAction(action_type="allocate", allocations=[
            AllocationItem(zone=random.randint(0, n-1),
                          resource=random.choice(list(RESOURCE_COSTS.keys())))])
    elif stage == "execute":
        return FairRecoveryAction(action_type="execute")
    return FairRecoveryAction(action_type="submit")


def greedy_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Utility-maximising greedy — ignores vulnerability (WRONG policy)."""
    stage, zones, budget = obs.step_stage, obs.zones, obs.budget_left
    if stage == "analyze":
        ranked = sorted(range(len(zones)), key=lambda i: zones[i].damage, reverse=True)
        return FairRecoveryAction(action_type="analyze", critical_zones=ranked[:2],
            reasoning=f"Greedy: top by damage={ranked[:2]}")
    elif stage == "allocate":
        ranked = sorted(range(len(zones)), key=lambda i: zones[i].damage, reverse=True)
        allocs = []
        for zid in ranked:
            for res, cost in sorted(RESOURCE_COSTS.items(), key=lambda x: x[1]):
                if budget >= cost:
                    allocs.append(AllocationItem(zone=zid, resource=res))
                    budget -= cost
                    break
            if allocs: break
        return FairRecoveryAction(action_type="allocate",
            allocations=allocs or [AllocationItem(zone=0, resource="power")])
    elif stage == "execute":
        return FairRecoveryAction(action_type="execute")
    return FairRecoveryAction(action_type="submit")


def fairness_aware_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Fairness-aware heuristic — prioritises damage × vulnerability (CORRECT)."""
    stage, zones, budget = obs.step_stage, obs.zones, obs.budget_left
    if stage == "analyze":
        scores = [(i, zones[i].damage * zones[i].vulnerable_ratio) for i in range(len(zones))]
        scores.sort(key=lambda x: -x[1])
        return FairRecoveryAction(action_type="analyze",
            critical_zones=[s[0] for s in scores[:2]],
            reasoning=f"Fair: critical={[s[0] for s in scores[:2]]}")
    elif stage == "allocate":
        scores = [(i, zones[i].damage * zones[i].vulnerable_ratio) for i in range(len(zones))]
        scores.sort(key=lambda x: -x[1])
        allocs = []
        for zid, sc in scores:
            if sc <= 0: break
            for res in ["medical", "water", "power"]:
                if budget >= RESOURCE_COSTS[res]:
                    allocs.append(AllocationItem(zone=zid, resource=res))
                    budget -= RESOURCE_COSTS[res]
                    break
            if allocs: break
        return FairRecoveryAction(action_type="allocate",
            allocations=allocs or [AllocationItem(zone=scores[0][0], resource="power")])
    elif stage == "execute":
        return FairRecoveryAction(action_type="execute")
    return FairRecoveryAction(action_type="submit")


POLICIES = {"random": random_policy, "greedy": greedy_policy, "fair": fairness_aware_policy}


def evaluate_policy(policy_name: str, difficulty: str = "hard",
                    n_episodes: int = 5, base_url: str = BASE_URL) -> Dict:
    policy_fn = POLICIES[policy_name]
    rewards, fairness, scores = [], [], []

    print(f"\n{'='*60}")
    print(f"Policy: {policy_name.upper()} | Difficulty: {difficulty} | Episodes: {n_episodes}")
    print(f"{'='*60}")

    with FairRecoveryEnv(base_url=base_url) as env:
        for ep in range(n_episodes):
            result = env.run_episode(policy_fn=policy_fn, difficulty=difficulty, verbose=True)
            rewards.append(result["total_reward"])
            fairness.append(result["final_fairness"])
            scores.append(result["grader_score"] or 0.0)
            print(f"  Episode {ep+1}: reward={result['total_reward']:.3f}, "
                  f"fairness={result['final_fairness']:.3f}, grader={result['grader_score']:.3f}")

    summary = {"policy": policy_name, "difficulty": difficulty,
               "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
               "mean_fairness": float(np.mean(fairness)), "mean_score": float(np.mean(scores))}
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="FairRecovery++ Baseline Inference")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", default=3, type=int)
    parser.add_argument("--policy", default="all", choices=["all", "random", "greedy", "fair"])
    parser.add_argument("--url", default=BASE_URL)
    args = parser.parse_args()

    policies = list(POLICIES.keys()) if args.policy == "all" else [args.policy]
    results = [evaluate_policy(p, args.difficulty, args.episodes, args.url) for p in policies]

    print(f"\n{'='*60}\nCOMPARISON TABLE\n{'='*60}")
    print(f"{'Policy':<12} {'Mean Reward':>12} {'Mean Fairness':>14} {'Mean Score':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['policy']:<12} {r['mean_reward']:>12.4f} {r['mean_fairness']:>14.4f} {r['mean_score']:>12.4f}")
    print("="*60)
    print("\nKey: 'greedy' has higher utility but LOWER fairness than 'fair'.")
    print("The trained agent should match/exceed 'fair' policy fairness score.")


if __name__ == "__main__":
    main()
