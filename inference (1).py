"""
FairRecovery — Baseline Inference Script.

Run BEFORE training to capture pre-training behaviour.
Run AFTER training to compare trained vs baseline.

Two baseline policies:
  random  — random zone + resource selection
  greedy  — utility-maximising heuristic (ignores fairness — the "wrong" policy)
  fair    — fairness-aware heuristic (the "correct" policy, baseline for comparison)

Usage:
    # Server must be running: uvicorn server.app:app --host 0.0.0.0 --port 8000
    python inference.py --difficulty hard --episodes 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Dict, List

import numpy as np

from client import FairRecoveryEnv
from fairrecovery_env.models import AllocationItem, FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.constants import RESOURCE_COSTS

BASE_URL = "http://localhost:8000"


# ──────────────────────────────────────────────────────────────────────────────
# Policy functions — must follow the multi-step protocol
# ──────────────────────────────────────────────────────────────────────────────

def random_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Completely random policy — ignores all observations."""
    stage = obs.step_stage

    if stage == "analyze":
        n_zones = len(obs.zones)
        chosen  = random.sample(range(n_zones), min(2, n_zones))
        return FairRecoveryAction(
            action_type="analyze",
            critical_zones=chosen,
            reasoning="Random zone selection.",
        )

    elif stage == "allocate":
        resources = list(RESOURCE_COSTS.keys())
        n_zones   = len(obs.zones)
        allocs    = [
            AllocationItem(
                zone=random.randint(0, n_zones - 1),
                resource=random.choice(resources),
            )
        ]
        return FairRecoveryAction(action_type="allocate", allocations=allocs)

    elif stage == "execute":
        return FairRecoveryAction(action_type="execute")

    else:
        return FairRecoveryAction(action_type="submit")


def greedy_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """
    Utility-maximising greedy — prioritises highest-damage zone regardless of vulnerability.
    This is the WRONG policy that the trained agent should outperform on fairness.
    """
    stage  = obs.step_stage
    zones  = obs.zones
    budget = obs.budget_left

    if stage == "analyze":
        # Sort by damage only (ignores vulnerability — biased baseline)
        ranked = sorted(range(len(zones)), key=lambda i: zones[i].damage, reverse=True)
        return FairRecoveryAction(
            action_type="analyze",
            critical_zones=ranked[:2],
            reasoning=f"Greedy: top zones by damage = {ranked[:2]}.",
        )

    elif stage == "allocate":
        # Pick highest-damage zone with best service-per-cost resource
        ranked = sorted(range(len(zones)), key=lambda i: zones[i].damage, reverse=True)
        allocs = []
        for zone_id in ranked:
            for resource, cost in sorted(RESOURCE_COSTS.items(), key=lambda x: x[1]):
                if budget >= cost:
                    allocs.append(AllocationItem(zone=zone_id, resource=resource))
                    budget -= cost
                    break
            if allocs:
                break
        if not allocs:
            allocs = [AllocationItem(zone=0, resource="power")]
        return FairRecoveryAction(action_type="allocate", allocations=allocs)

    elif stage == "execute":
        return FairRecoveryAction(action_type="execute")

    else:
        return FairRecoveryAction(action_type="submit")


def fairness_aware_policy(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """
    Fairness-aware heuristic — prioritises highest (damage × vulnerability) score.
    This is the CORRECT policy. The trained agent should learn to match or exceed it.
    """
    stage  = obs.step_stage
    zones  = obs.zones
    budget = obs.budget_left

    if stage == "analyze":
        # Rank by damage × vulnerable_ratio
        scores = [
            (i, zones[i].damage * zones[i].vulnerable_ratio)
            for i in range(len(zones))
        ]
        scores.sort(key=lambda x: -x[1])
        critical = [s[0] for s in scores[:2]]
        return FairRecoveryAction(
            action_type="analyze",
            critical_zones=critical,
            reasoning=(
                f"Fair heuristic: critical zones {critical} "
                f"(highest damage × vulnerability)."
            ),
        )

    elif stage == "allocate":
        # Pick zone with highest vulnerability × damage score
        scores = [
            (i, zones[i].damage * zones[i].vulnerable_ratio)
            for i in range(len(zones))
        ]
        scores.sort(key=lambda x: -x[1])
        allocs = []
        for zone_id, score in scores:
            if score <= 0:
                break
            # Use most effective resource within budget
            for resource in ["medical", "water", "power"]:
                cost = RESOURCE_COSTS[resource]
                if budget >= cost:
                    allocs.append(AllocationItem(zone=zone_id, resource=resource))
                    budget -= cost
                    break
            if allocs:
                break
        if not allocs:
            allocs = [AllocationItem(zone=scores[0][0], resource="power")]
        return FairRecoveryAction(action_type="allocate", allocations=allocs)

    elif stage == "execute":
        return FairRecoveryAction(action_type="execute")

    else:
        return FairRecoveryAction(action_type="submit")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

POLICIES = {
    "random":  random_policy,
    "greedy":  greedy_policy,
    "fair":    fairness_aware_policy,
}


def evaluate_policy(
    policy_name: str,
    difficulty: str = "hard",
    n_episodes: int = 5,
    base_url: str = BASE_URL,
) -> Dict:
    """
    Run multiple episodes with a policy and compute aggregate metrics.

    Returns dict with mean/std of total_reward, final_fairness, grader_score.
    """
    policy_fn = POLICIES[policy_name]

    rewards:   List[float] = []
    fairness:  List[float] = []
    scores:    List[float] = []

    print(f"\n{'='*60}")
    print(f"Policy: {policy_name.upper()} | Difficulty: {difficulty} | Episodes: {n_episodes}")
    print(f"{'='*60}")

    with FairRecoveryEnv(base_url=base_url) as env:
        for ep in range(n_episodes):
            result = env.run_episode(
                policy_fn=policy_fn,
                difficulty=difficulty,
                verbose=True,
            )
            rewards.append(result["total_reward"])
            fairness.append(result["final_fairness"])
            scores.append(result["grader_score"] or 0.0)

            print(
                f"  Episode {ep+1}: reward={result['total_reward']:.3f}, "
                f"fairness={result['final_fairness']:.3f}, "
                f"grader={result['grader_score']:.3f}, "
                f"steps={result['steps']}"
            )

    summary = {
        "policy":         policy_name,
        "difficulty":     difficulty,
        "n_episodes":     n_episodes,
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "mean_fairness":  float(np.mean(fairness)),
        "std_fairness":   float(np.std(fairness)),
        "mean_score":     float(np.mean(scores)),
        "std_score":      float(np.std(scores)),
    }

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="FairRecovery Baseline Inference")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes",   default=3, type=int)
    parser.add_argument("--policy",     default="all", choices=["all", "random", "greedy", "fair"])
    parser.add_argument("--url",        default=BASE_URL)
    args = parser.parse_args()

    policies = list(POLICIES.keys()) if args.policy == "all" else [args.policy]

    all_results = []
    for policy in policies:
        result = evaluate_policy(
            policy_name=policy,
            difficulty=args.difficulty,
            n_episodes=args.episodes,
            base_url=args.url,
        )
        all_results.append(result)

    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Policy':<12} {'Mean Reward':>12} {'Mean Fairness':>14} {'Mean Score':>12}")
    print("-"*60)
    for r in all_results:
        print(
            f"{r['policy']:<12} "
            f"{r['mean_reward']:>12.4f} "
            f"{r['mean_fairness']:>14.4f} "
            f"{r['mean_score']:>12.4f}"
        )
    print("="*60)
    print("\nKey insight: 'greedy' has higher utility but LOWER fairness than 'fair'.")
    print("The trained agent should match/exceed 'fair' policy fairness score.")


if __name__ == "__main__":
    main()
