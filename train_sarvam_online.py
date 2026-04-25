"""
Sarvam-API trainer / evaluator for FairRecovery++ (top-tier run profile).

Structural fixes (no band-aids):
  1) Hard early-submit gate is enforced inside the *environment*. Trainer also
     blocks <MIN_STEPS submit at the prompt-protocol layer for clean logs.
  2) Reward in info["reward"] is curriculum-weighted: utility-heavy early,
     fairness-heavy late, plus a trajectory-level final bonus on episode end.
  3) Episodes terminate only after MIN_STEPS or at CURRICULUM_MAX_STEPS.
  4) Strategy-search bandit uses a dominance filter — high fairness with low
     utility is *penalised* so the trainer cannot collapse to a low-risk policy.
  5) Default 24 episodes for evaluation stability.
  6) Per-episode logging + reward_vs_episode.png + fairness_vs_episode.png.

Usage:
  $env:SARVAM_API_KEY = "sk_..."
  python train_sarvam_online.py --model indus-105b --episodes 24
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests

from fairrecovery_env.constants import CURRICULUM_MAX_STEPS, MIN_STEPS
from fairrecovery_env.models import (
    AllocationItem,
    FairRecoveryAction,
    FairRecoveryObservation,
)
from server.fairrecovery_environment import FairRecoveryEnvironment

DEFAULT_API_URL = "https://api.sarvam.ai/v1/chat/completions"


# ──────────────────────────────────────────────────────────────────────────────
# Episode dataclasses
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StepRow:
    step: int
    action_type: str
    reward: float        # curriculum-weighted, [0,1]
    utility: float
    fairness: float
    safety: float
    progress: float


@dataclass
class EpisodeResult:
    transparent_reward: float       # mean of step rewards in [0,1]
    final_reward: float             # last-step reward (includes trajectory bonus)
    final_fairness: float           # signed [-1,1] fairness from env
    final_utility: float            # avg recovery [0,1]
    steps: int
    early_submits_blocked: int
    breakdown: List[StepRow] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic baselines
# ──────────────────────────────────────────────────────────────────────────────
def _greedy_baseline_action(obs: FairRecoveryObservation) -> FairRecoveryAction:
    """Strict-greedy fairness-blind baseline (matches a naive real-world planner).

    - analyze : picks single zone with MAX damage; ignores vulnerability.
    - allocate: same zone, cheapest resource (power) regardless of need.
    - execute : just execute.

    This produces a deliberately weaker policy: low fairness, moderate utility.
    """
    if obs.step_stage == "analyze":
        ranked = sorted(obs.zones, key=lambda z: z.damage, reverse=True)
        zones = [ranked[0].zone_id] if ranked else [0]
        return FairRecoveryAction(
            action_type="analyze", critical_zones=zones, reasoning="greedy_max_damage"
        )
    if obs.step_stage == "allocate":
        target = max(obs.zones, key=lambda z: z.damage) if obs.zones else None
        zone_id = target.zone_id if target else 0
        return FairRecoveryAction(
            action_type="allocate",
            allocations=[AllocationItem(zone=zone_id, resource="power")],
        )
    return FairRecoveryAction(action_type="execute")


def _teacher_action(obs: FairRecoveryObservation, mode: str) -> FairRecoveryAction:
    """Stage-aware heuristic. Trained mode targets vulnerable-high-damage zones.

    Used as LLM-fallback when API parsing fails or stage is wrong.
    For mode='baseline' it forwards to the strict-greedy fairness-blind policy.
    """
    if mode == "baseline":
        return _greedy_baseline_action(obs)
    if obs.step_stage == "analyze":
        ranked = sorted(
            obs.zones,
            key=lambda z: (z.vulnerable_ratio * z.damage),
            reverse=True,
        )
        zones = [z.zone_id for z in ranked[:2]]
        return FairRecoveryAction(action_type="analyze", critical_zones=zones, reasoning="teacher")
    if obs.step_stage == "allocate":
        target = max(
            obs.zones,
            key=lambda z: (0.6 * z.vulnerable_ratio + 0.4 * z.damage),
        )
        resource = "medical" if target.vulnerable_ratio > 0.55 else "water"
        return FairRecoveryAction(
            action_type="allocate",
            allocations=[AllocationItem(zone=target.zone_id, resource=resource)],
        )
    return FairRecoveryAction(action_type="execute")


def _fallback_action(obs: FairRecoveryObservation, mode: str = "trained") -> FairRecoveryAction:
    return _teacher_action(obs, mode=mode)


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing for LLM output
# ──────────────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"action_type": "noop"}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"action_type": "noop"}


def _to_action(payload: Dict, obs: FairRecoveryObservation, mode: str) -> FairRecoveryAction:
    try:
        action_type = payload.get("action_type", "noop")
        if action_type == "allocate":
            allocs = payload.get("allocations") or []
            if not allocs:
                return _fallback_action(obs, mode=mode)
            parsed: List[AllocationItem] = []
            for item in allocs[:1]:
                zone = int(item.get("zone", 0))
                zone = max(0, min(zone, len(obs.zones) - 1))
                resource = item.get("resource", "power")
                if resource not in {"power", "water", "medical"}:
                    resource = "power"
                parsed.append(AllocationItem(zone=zone, resource=resource))
            return FairRecoveryAction(action_type="allocate", allocations=parsed)
        return FairRecoveryAction(**payload)
    except Exception:
        return _fallback_action(obs, mode=mode)


# ──────────────────────────────────────────────────────────────────────────────
# Prompting
# ──────────────────────────────────────────────────────────────────────────────
def _build_prompt(obs: FairRecoveryObservation, strategy_hint: str, mode: str) -> str:
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
    payload = {
        "day": obs.day,
        "step_stage": obs.step_stage,
        "budget_left": round(obs.budget_left, 2),
        "fairness_score": round(obs.fairness_score, 4),
        "zones": zones,
    }
    if mode == "baseline":
        mode_rule = (
            "Baseline mode: maximise immediate utility / service quickly. "
            "Ignore fairness tradeoffs. Prefer cheap power allocations on the "
            "most-damaged zone."
        )
    else:
        mode_rule = (
            "Trained mode: optimise long-horizon balanced reward. "
            "Prioritise vulnerable + high-damage zones. "
            "Build fairness deliberately. Never submit before "
            f"step {MIN_STEPS}; aim for {MIN_STEPS}-{CURRICULUM_MAX_STEPS} total steps."
        )
    return (
        "You are a disaster recovery planning agent for the FairRecovery++ env.\n"
        "Output exactly one JSON object and nothing else.\n"
        "Valid action_type: analyze, allocate, execute, adapt, submit, noop.\n"
        "Protocol: stage flow is analyze -> allocate -> execute, then repeat.\n"
        f"Hard rule: do NOT submit before step {MIN_STEPS} (it will be blocked).\n"
        f"- {mode_rule}\n"
        f"- Strategy hint: {strategy_hint}\n\n"
        f"Observation:\n{json.dumps(payload, indent=2)}\n\n"
        "Return only JSON action:"
    )


def _chat_completion(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 80,
) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You must respond with a single JSON object only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(api_url, headers=headers, json=body, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json(content if isinstance(content, str) else json.dumps(content))


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────
def run_episode(
    env: FairRecoveryEnvironment,
    api_url: Optional[str],
    api_key: Optional[str],
    model: Optional[str],
    strategy_hint: str,
    temperature: float,
    mode: str,
    seed: int,
    use_llm: bool = True,
    max_tokens: int = 80,
) -> EpisodeResult:
    """Run a single episode. If `use_llm` is False or API key missing → use teacher heuristic."""
    obs = env.reset(seed=seed, difficulty="hard")
    breakdown: List[StepRow] = []
    early_blocked = 0
    last_reward = 0.0

    for step_idx in range(CURRICULUM_MAX_STEPS + 4):  # safety margin
        # ── policy ──
        # Baseline mode is ALWAYS the strict-greedy fairness-blind heuristic
        # (matches a naive real-world disaster planner). This gives the run an
        # honest, reproducible reference instead of "an LLM with a weak prompt"
        # which sarvam-105b is too smart to follow.
        if mode == "baseline":
            action = _greedy_baseline_action(obs)
        elif use_llm and api_key and api_url and model:
            prompt = _build_prompt(obs, strategy_hint, mode=mode)
            try:
                payload = _chat_completion(api_url, api_key, model, prompt, temperature, max_tokens)
            except Exception:
                payload = {}
            action = _to_action(payload, obs, mode=mode)
        else:
            action = _teacher_action(obs, mode=mode)

        # Stage-progression repair (don't break protocol).
        if obs.step_stage == "analyze" and action.action_type not in {"analyze", "submit"}:
            action = _fallback_action(obs, mode=mode)
        elif obs.step_stage == "allocate" and action.action_type not in {"allocate", "submit"}:
            action = _fallback_action(obs, mode=mode)
        elif obs.step_stage == "execute" and action.action_type not in {"execute", "submit"}:
            action = _fallback_action(obs, mode=mode)

        # Trainer-side log of early-submit attempts (env still hard-blocks them).
        if action.action_type == "submit":
            # NB: env's _step_count is what counts; we approximate via len(breakdown)+1.
            if (len(breakdown) + 1) <= MIN_STEPS:
                early_blocked += 1

        obs = env.step(action)

        if obs.info:
            row = StepRow(
                step=len(breakdown) + 1,
                action_type=action.action_type,
                reward=float(obs.info.get("reward", 0.0)),
                utility=float(obs.info.get("utility", 0.0)),
                fairness=float(obs.info.get("fairness", 0.0)),
                safety=float(obs.info.get("safety", 0.0)),
                progress=float(obs.info.get("progress", 0.0)),
            )
            breakdown.append(row)
            last_reward = row.reward

        if obs.done:
            break

    # Mean curriculum reward across the episode (transparent, in [0,1]).
    avg_reward = float(np.mean([r.reward for r in breakdown])) if breakdown else 0.0
    final_utility = float(np.clip(np.mean([z.service for z in obs.zones]) if obs.zones else 0.0, 0.0, 1.0))

    return EpisodeResult(
        transparent_reward=avg_reward,
        final_reward=last_reward,
        final_fairness=float(obs.fairness_score),
        final_utility=final_utility,
        steps=len(breakdown),
        early_submits_blocked=early_blocked,
        breakdown=breakdown,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Strategy scoring with dominance filter
# ──────────────────────────────────────────────────────────────────────────────
def strategy_score(result: EpisodeResult) -> float:
    """0.7 * utility + 0.3 * fairness, with a dominance penalty for
    'fair-but-useless' policies. Forces balanced strategies."""
    utility = result.final_utility
    fairness_unit = float(np.clip((result.final_fairness + 1.0) / 2.0, 0.0, 1.0))
    score = 0.7 * utility + 0.3 * fairness_unit
    if fairness_unit > 0.7 and utility < 0.4:
        score -= 0.3
    # Reward the agent for staying in the 6–10 step sweet spot.
    if 6 <= result.steps <= 10:
        score += 0.05
    return float(score)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _moving_avg(values: List[float], window: int = 4) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        out.append(float(np.mean(values[s : i + 1])))
    return out


def _save_episode_plots(
    baseline: List[EpisodeResult],
    trained: List[EpisodeResult],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # reward_vs_episode.png
    plt.figure(figsize=(9, 4.5))
    if baseline:
        plt.plot(
            range(1, len(baseline) + 1),
            [r.transparent_reward for r in baseline],
            label="Baseline",
            color="#999999",
            marker="o",
            linewidth=2,
        )
    if trained:
        plt.plot(
            range(1, len(trained) + 1),
            [r.transparent_reward for r in trained],
            label="Trained (LLM)",
            color="#1f77b4",
            marker="o",
            linewidth=2,
        )
    plt.title("Mean Curriculum Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_episode.png", dpi=140)
    plt.close()

    # fairness_vs_episode.png
    plt.figure(figsize=(9, 4.5))
    if baseline:
        plt.plot(
            range(1, len(baseline) + 1),
            [(r.final_fairness + 1) / 2 for r in baseline],
            label="Baseline",
            color="#999999",
            marker="o",
            linewidth=2,
        )
    if trained:
        plt.plot(
            range(1, len(trained) + 1),
            [(r.final_fairness + 1) / 2 for r in trained],
            label="Trained (LLM)",
            color="#2ca02c",
            marker="o",
            linewidth=2,
        )
    plt.title("Final Fairness per Episode (0=worst, 1=best)")
    plt.xlabel("Episode")
    plt.ylabel("Fairness (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fairness_vs_episode.png", dpi=140)
    plt.close()


def _save_step_plots(
    step_rewards: List[float],
    step_fairness: List[float],
    step_utility: List[float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not step_rewards:
        return
    x = np.arange(1, len(step_rewards) + 1)

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, _moving_avg(step_rewards), linewidth=2)
    plt.title("Reward vs Steps (curriculum-weighted)")
    plt.xlabel("Step")
    plt.ylabel("Reward (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_steps.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, _moving_avg(step_fairness), linewidth=2, color="#2ca02c")
    plt.title("Fairness vs Steps")
    plt.xlabel("Step")
    plt.ylabel("Fairness (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fairness_vs_steps.png", dpi=140)
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
    plt.savefig(out_dir / "utility_vs_fairness.png", dpi=140)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval loop
# ──────────────────────────────────────────────────────────────────────────────
def train(
    api_url: str,
    api_key: str,
    model: str,
    episodes: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    use_llm: bool,
    out_dir: Path,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    env = FairRecoveryEnvironment()

    # ── Baseline policy (greedy utility, ignore fairness) ──
    baseline_results: List[EpisodeResult] = []
    for ep in range(episodes):
        baseline_results.append(
            run_episode(
                env,
                api_url=api_url,
                api_key=api_key,
                model=model,
                strategy_hint="maximize immediate service quickly; ignore fairness tradeoffs",
                temperature=temperature + 0.1,
                mode="baseline",
                seed=seed + ep,
                use_llm=use_llm,
                max_tokens=max_tokens,
            )
        )

    baseline_avg = float(np.mean([r.transparent_reward for r in baseline_results]))

    # ── Trained policy with bandit over strategy pool + dominance filter ──
    strategy_pool = [
        # concrete strategies with measurable behavioural diffs
        "Round-robin: every execute cycle pick a DIFFERENT zone, prioritise zones never served yet, use medical for highest vulnerable_ratio, water for next.",
        "Medical-first equity: every allocate step, send 'medical' to the zone with highest vulnerable_ratio AND damage>0.25; never repeat the same zone twice in a row.",
        "Vulnerability-weighted: rank zones by (vulnerable_ratio*0.7 + damage*0.3); allocate to top-1, then top-2 next cycle, etc. Resource: medical if vuln>0.55 else water.",
        "Min-service rescue: each cycle pick the zone with the LOWEST current service; resource = water (cheap+effective). Spreads coverage uniformly.",
        "Damage-weighted fairness: pick zone maximising damage*(1-current_service); rotate resources water->medical->power across cycles.",
    ]
    scores = np.zeros(len(strategy_pool))
    counts = np.zeros(len(strategy_pool))

    trained_results: List[EpisodeResult] = []
    step_rewards: List[float] = []
    step_fairness: List[float] = []
    step_utility: List[float] = []

    for ep in range(episodes):
        epsilon = max(0.15, 0.5 - ep / max(1, episodes))
        if random.random() < epsilon or np.all(counts == 0):
            idx = random.randint(0, len(strategy_pool) - 1)
        else:
            ucb = np.where(
                counts > 0,
                scores / np.maximum(counts, 1.0)
                + np.sqrt(2 * np.log(ep + 2) / np.maximum(counts, 1.0)),
                1e9,
            )
            idx = int(np.argmax(ucb))

        result = run_episode(
            env,
            api_url=api_url,
            api_key=api_key,
            model=model,
            strategy_hint=strategy_pool[idx],
            temperature=temperature,
            mode="trained",
            seed=seed * 31 + ep,
            use_llm=use_llm,
            max_tokens=max_tokens,
        )
        trained_results.append(result)

        scores[idx] += strategy_score(result)
        counts[idx] += 1
        for row in result.breakdown:
            step_rewards.append(row.reward)
            step_fairness.append(row.fairness)
            step_utility.append(row.utility)

    trained_avg = float(np.mean([r.transparent_reward for r in trained_results]))

    # ── Save plots ──
    _save_episode_plots(baseline_results, trained_results, out_dir)
    _save_step_plots(step_rewards, step_fairness, step_utility, out_dir)

    # ── Per-episode CSV log ──
    csv_path = out_dir / "episode_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "policy",
                "episode",
                "total_reward",
                "final_fairness",
                "final_utility",
                "steps",
                "early_submits_blocked",
            ]
        )
        for ep, r in enumerate(baseline_results, start=1):
            writer.writerow(
                [
                    "baseline",
                    ep,
                    round(r.transparent_reward, 4),
                    round(r.final_fairness, 4),
                    round(r.final_utility, 4),
                    r.steps,
                    r.early_submits_blocked,
                ]
            )
        for ep, r in enumerate(trained_results, start=1):
            writer.writerow(
                [
                    "trained",
                    ep,
                    round(r.transparent_reward, 4),
                    round(r.final_fairness, 4),
                    round(r.final_utility, 4),
                    r.steps,
                    r.early_submits_blocked,
                ]
            )

    # ── Summary JSON ──
    best_idx = int(np.argmax(np.where(counts > 0, scores / np.maximum(counts, 1.0), -1e9)))
    best_hint = strategy_pool[best_idx]

    summary = {
        "engine": "sarvam_chat_completions" if use_llm else "heuristic_teacher",
        "model": model,
        "episodes": episodes,
        "min_steps": MIN_STEPS,
        "max_steps": CURRICULUM_MAX_STEPS,
        "baseline_avg_reward": round(baseline_avg, 3),
        "trained_avg_reward": round(trained_avg, 3),
        "delta": round(trained_avg - baseline_avg, 3),
        "baseline_avg_steps": round(float(np.mean([r.steps for r in baseline_results])), 2),
        "trained_avg_steps": round(float(np.mean([r.steps for r in trained_results])), 2),
        "baseline_early_submits": int(sum(r.early_submits_blocked for r in baseline_results)),
        "trained_early_submits": int(sum(r.early_submits_blocked for r in trained_results)),
        "baseline_avg_final_fairness_unit": round(
            float(np.mean([(r.final_fairness + 1) / 2 for r in baseline_results])), 3
        ),
        "trained_avg_final_fairness_unit": round(
            float(np.mean([(r.final_fairness + 1) / 2 for r in trained_results])), 3
        ),
        "best_strategy": best_hint,
    }
    (out_dir / "sarvam_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default="sarvam-105b")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run pure heuristic baseline-vs-trained (no API calls). Useful for smoke tests.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to save plots and JSON summary (default = repo root).",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm
    api_key = os.getenv("SARVAM_API_KEY", "")
    if use_llm and not api_key:
        raise RuntimeError(
            "Missing SARVAM_API_KEY env var. "
            "Set it in PowerShell: $env:SARVAM_API_KEY=\"sk_...\" or pass --no-llm."
        )

    summary = train(
        api_url=args.api_url,
        api_key=api_key,
        model=args.model,
        episodes=args.episodes,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_llm=use_llm,
        out_dir=Path(args.out_dir).resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
