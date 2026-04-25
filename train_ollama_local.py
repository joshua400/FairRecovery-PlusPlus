from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests

from fairrecovery_env.models import AllocationItem, FairRecoveryAction, FairRecoveryObservation
from server.fairrecovery_environment import FairRecoveryEnvironment

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class RolloutResult:
    total_reward: float
    transparent_reward: float
    final_fairness: float
    breakdown: List[Dict[str, float]]


def _safe_json_extract(text: str) -> Dict:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"action_type": "noop"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"action_type": "noop"}


def _build_prompt(obs: FairRecoveryObservation, strategy_hint: str) -> str:
    zones = [
        {
            "zone_id": z.zone_id,
            "damage": round(z.damage, 3),
            "service": round(z.service, 3),
            "vulnerable_ratio": round(z.vulnerable_ratio, 3),
            "risk_level": round(z.risk_level, 3),
        }
        for z in obs.zones
    ]
    state = {
        "day": obs.day,
        "step_stage": obs.step_stage,
        "budget_left": round(obs.budget_left, 2),
        "fairness_score": round(obs.fairness_score, 4),
        "zones": zones,
    }
    return (
        "You are a disaster recovery planner.\n"
        "Return only one JSON object. No markdown.\n"
        "Valid action_type: analyze, allocate, execute, adapt, submit, noop.\n"
        "Rules:\n"
        "- Always follow stage protocol analyze->allocate->execute.\n"
        "- Avoid noop unless absolutely necessary.\n"
        "- Favor vulnerable high-damage zones fairly.\n"
        f"- Strategy hint: {strategy_hint}\n\n"
        f"Observation:\n{json.dumps(state, indent=2)}\n\n"
        "Output JSON action:"
    )


def _ask_ollama(model: str, prompt: str, temperature: float) -> Dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    txt = r.json().get("response", "")
    return _safe_json_extract(txt)


def _to_action(payload: Dict, obs: FairRecoveryObservation) -> FairRecoveryAction:
    try:
        if payload.get("action_type") == "allocate":
            allocs = payload.get("allocations") or []
            if not allocs:
                payload["allocations"] = [{"zone": 0, "resource": "power"}]
            clean = []
            for a in payload["allocations"][:1]:
                zone = max(0, min(int(a.get("zone", 0)), len(obs.zones) - 1))
                resource = a.get("resource", "power")
                if resource not in {"power", "water", "medical"}:
                    resource = "power"
                clean.append(AllocationItem(zone=zone, resource=resource))
            return FairRecoveryAction(action_type="allocate", allocations=clean)
        return FairRecoveryAction(**payload)
    except Exception:
        if obs.step_stage == "analyze":
            return FairRecoveryAction(action_type="analyze", critical_zones=[0, 1], reasoning="fallback")
        if obs.step_stage == "allocate":
            return FairRecoveryAction(action_type="allocate", allocations=[AllocationItem(zone=0, resource="power")])
        if obs.step_stage == "execute":
            return FairRecoveryAction(action_type="execute")
        return FairRecoveryAction(action_type="submit")


def run_episode(env: FairRecoveryEnvironment, model: str, strategy_hint: str, temperature: float) -> RolloutResult:
    obs = env.reset(difficulty="hard")
    total_reward = 0.0
    rows: List[Dict[str, float]] = []
    noop_streak = 0
    step_cap = 24

    for _ in range(step_cap):
        prompt = _build_prompt(obs, strategy_hint)
        payload = _ask_ollama(model=model, prompt=prompt, temperature=temperature)
        action = _to_action(payload, obs)
        if action.action_type == "noop":
            noop_streak += 1
        else:
            noop_streak = 0
        if noop_streak >= 4:
            action = FairRecoveryAction(action_type="submit")

        obs = env.step(action)
        total_reward += obs.reward
        if obs.info:
            rows.append(
                {
                    "reward": float(obs.info.get("reward", 0.0)),
                    "utility": float(obs.info.get("utility", 0.0)),
                    "fairness": float(obs.info.get("fairness", 0.0)),
                }
            )
        if obs.done:
            break

    if not obs.done:
        obs = env.step(FairRecoveryAction(action_type="submit"))
        total_reward += obs.reward
        if obs.info:
            rows.append(
                {
                    "reward": float(obs.info.get("reward", 0.0)),
                    "utility": float(obs.info.get("utility", 0.0)),
                    "fairness": float(obs.info.get("fairness", 0.0)),
                }
            )

    avg_transparent = float(np.mean([r["reward"] for r in rows])) if rows else 0.0
    return RolloutResult(
        total_reward=float(total_reward),
        transparent_reward=avg_transparent,
        final_fairness=float(obs.fairness_score),
        breakdown=rows,
    )


def _moving_avg(xs: List[float], w: int = 6) -> List[float]:
    out = []
    for i in range(len(xs)):
        s = max(0, i - w + 1)
        out.append(float(np.mean(xs[s : i + 1])))
    return out


def train_local(model: str, episodes: int, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    env = FairRecoveryEnvironment()
    strategies = [
        "strict fairness: prioritize vulnerable zones first",
        "balanced fairness + utility with low-noop behavior",
        "utility-first but keep fairness parity each day",
    ]
    scores = np.zeros(len(strategies), dtype=float)
    counts = np.zeros(len(strategies), dtype=float)

    baseline_hint = "generic planning with no fairness emphasis"
    baseline_runs = [run_episode(env, model, baseline_hint, 0.1) for _ in range(4)]
    baseline_avg = float(np.mean([r.transparent_reward for r in baseline_runs]))

    step_rewards: List[float] = []
    step_fairness: List[float] = []
    step_utility: List[float] = []

    # Lightweight bandit tuning over prompting strategy.
    for ep in range(episodes):
        eps = max(0.15, 0.6 - (ep / max(1, episodes)))
        if random.random() < eps or np.all(counts == 0):
            idx = random.randint(0, len(strategies) - 1)
        else:
            ucb = np.where(
                counts > 0,
                scores / np.maximum(counts, 1.0) + np.sqrt(2 * np.log(ep + 2) / np.maximum(counts, 1.0)),
                1e9,
            )
            idx = int(np.argmax(ucb))

        rr = run_episode(env, model, strategies[idx], 0.15)
        reward = rr.transparent_reward
        scores[idx] += reward
        counts[idx] += 1
        for row in rr.breakdown:
            step_rewards.append(row["reward"])
            step_fairness.append(row["fairness"])
            step_utility.append(row["utility"])

    best_idx = int(np.argmax(np.where(counts > 0, scores / np.maximum(counts, 1.0), -1e9)))
    best_hint = strategies[best_idx]
    trained_runs = [run_episode(env, model, best_hint, 0.1) for _ in range(4)]
    trained_avg = float(np.mean([r.transparent_reward for r in trained_runs]))

    x = np.arange(1, len(step_rewards) + 1)
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, _moving_avg(step_rewards), linewidth=2)
    plt.title("Reward vs Steps (Ollama qwen2.5)")
    plt.xlabel("Step")
    plt.ylabel("Reward (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("reward_vs_steps.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, _moving_avg(step_fairness), linewidth=2, color="#1f77b4")
    plt.title("Fairness vs Steps (Ollama qwen2.5)")
    plt.xlabel("Step")
    plt.ylabel("Fairness (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fairness_vs_steps.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(step_utility, step_fairness, alpha=0.5, s=16)
    plt.title("Utility vs Fairness")
    plt.xlabel("Utility (0-1)")
    plt.ylabel("Fairness (0-1)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("utility_vs_fairness.png", dpi=140)
    plt.close()

    summary = {
        "engine": "ollama",
        "model": model,
        "episodes": episodes,
        "baseline_avg_reward": round(baseline_avg, 3),
        "trained_avg_reward": round(trained_avg, 3),
        "best_strategy": best_hint,
        "strategy_counts": counts.tolist(),
    }
    Path("ollama_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = train_local(model=args.model, episodes=args.episodes, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

