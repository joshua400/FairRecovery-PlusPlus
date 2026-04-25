"""
Real LLM training pipeline (Colab/HF GPU) for FairRecovery++.

Uses:
  - Unsloth (LoRA / 4-bit load)
  - TRL SFTTrainer (lightweight real fine-tuning)
  - Environment rollouts for evaluation + plots

Run on Colab T4/A10:
  pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install -U trl transformers accelerate peft bitsandbytes datasets matplotlib
  python train_real_llm_colab.py --episodes 15 --train-samples 220
"""

from __future__ import annotations

import argparse
import json
import random
import re
import inspect
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from fairrecovery_env.constants import RESOURCE_COSTS
from fairrecovery_env.models import AllocationItem, FairRecoveryAction, FairRecoveryObservation
from server.fairrecovery_environment import FairRecoveryEnvironment


SYSTEM_PROMPT = (
    "You are a disaster recovery planner. Output exactly one JSON object for the next action. "
    "Valid action_type: analyze, allocate, execute, adapt, submit, noop. "
    "Be fairness-aware and prioritize vulnerable high-damage zones."
)


def _obs_to_prompt(obs: FairRecoveryObservation) -> str:
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
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Observation:\n{json.dumps(payload, indent=2)}\n\n"
        f"Return only JSON action:"
    )


def _teacher_action(obs: FairRecoveryObservation) -> Dict:
    if obs.step_stage == "analyze":
        scores = sorted(
            range(len(obs.zones)),
            key=lambda i: obs.zones[i].damage * obs.zones[i].vulnerable_ratio + 0.2 * obs.zones[i].risk_level,
            reverse=True,
        )
        return {"action_type": "analyze", "critical_zones": scores[:2], "reasoning": "prioritize vulnerable damage"}
    if obs.step_stage == "allocate":
        scores = sorted(
            range(len(obs.zones)),
            key=lambda i: obs.zones[i].damage * obs.zones[i].vulnerable_ratio + 0.2 * obs.zones[i].risk_level,
            reverse=True,
        )
        resource = "medical" if obs.budget_left >= RESOURCE_COSTS["medical"] else "power"
        return {"action_type": "allocate", "allocations": [{"zone": scores[0], "resource": resource}]}
    if obs.step_stage == "execute":
        return {"action_type": "execute"}
    return {"action_type": "submit"}


def build_train_dataset(env: FairRecoveryEnvironment, n_samples: int, difficulty: str) -> Dataset:
    rows: List[Dict[str, str]] = []
    while len(rows) < n_samples:
        obs = env.reset(difficulty=difficulty)
        done = False
        local_steps = 0
        while not done and len(rows) < n_samples and local_steps < 50:
            action = _teacher_action(obs)
            prompt = _obs_to_prompt(obs)
            completion = json.dumps(action, ensure_ascii=True)
            rows.append({"text": f"{prompt}\n{completion}"})
            obs = env.step(FairRecoveryAction(**action))
            done = obs.done
            local_steps += 1
        if not done:
            env.step(FairRecoveryAction(action_type="submit"))
    return Dataset.from_list(rows)


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"action_type": "noop"}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"action_type": "noop"}


def _safe_action(payload: Dict, obs: FairRecoveryObservation) -> FairRecoveryAction:
    try:
        if payload.get("action_type") == "allocate" and payload.get("allocations"):
            clean_allocs = []
            for a in payload["allocations"][:1]:
                z = int(a.get("zone", 0))
                z = max(0, min(z, len(obs.zones) - 1))
                r = a.get("resource", "power")
                if r not in RESOURCE_COSTS:
                    r = "power"
                clean_allocs.append(AllocationItem(zone=z, resource=r))
            return FairRecoveryAction(action_type="allocate", allocations=clean_allocs)
        return FairRecoveryAction(**payload)
    except Exception:
        if obs.step_stage == "execute":
            return FairRecoveryAction(action_type="execute")
        if obs.step_stage == "allocate":
            return FairRecoveryAction(action_type="allocate", allocations=[AllocationItem(zone=0, resource="power")])
        if obs.step_stage == "analyze":
            return FairRecoveryAction(action_type="analyze", critical_zones=[0, 1], reasoning="fallback")
        return FairRecoveryAction(action_type="submit")


@torch.inference_mode()
def run_llm_episode(env: FairRecoveryEnvironment, model, tokenizer, difficulty: str, max_new_tokens: int = 120):
    obs = env.reset(difficulty=difficulty)
    total = 0.0
    breakdown: List[Dict[str, float]] = []
    steps = 0

    while not obs.done and steps < 60:
        prompt = _obs_to_prompt(obs)
        batch = tokenizer([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        action_payload = _extract_json(text[len(prompt):] if len(text) > len(prompt) else text)
        action = _safe_action(action_payload, obs)
        obs = env.step(action)
        total += obs.reward
        steps += 1
        if obs.info:
            breakdown.append(
                {
                    "reward": float(obs.info.get("reward", 0.0)),
                    "utility": float(obs.info.get("utility", 0.0)),
                    "fairness": float(obs.info.get("fairness", 0.0)),
                }
            )
    if not obs.done:
        obs = env.step(FairRecoveryAction(action_type="submit"))
        total += obs.reward
    return total, obs, breakdown


def _moving_avg(x: List[float], k: int = 5) -> List[float]:
    if not x:
        return []
    y = []
    for i in range(len(x)):
        s = max(0, i - k + 1)
        y.append(float(np.mean(x[s : i + 1])))
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--train-samples", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="real_llm_output")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = FairRecoveryEnvironment()
    ds = build_train_dataset(env, n_samples=args.train_samples, difficulty=args.difficulty)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=1536,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
    )

    sft_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "num_train_epochs": 1,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": "none",
    }
    # TRL has changed argument names across versions.
    config_sig = inspect.signature(SFTConfig).parameters
    if "max_seq_length" in config_sig:
        sft_kwargs["max_seq_length"] = 1536
    elif "max_length" in config_sig:
        sft_kwargs["max_length"] = 1536
    if "dataset_text_field" in config_sig:
        sft_kwargs["dataset_text_field"] = "text"
    if "packing" in config_sig:
        sft_kwargs["packing"] = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(**sft_kwargs),
    )
    train_metrics = trainer.train()

    # Evaluate baseline vs trained model in environment
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=args.model_name, max_seq_length=1536, load_in_4bit=True
    )
    FastLanguageModel.for_inference(base_model)
    FastLanguageModel.for_inference(model)

    baseline_rewards = []
    trained_rewards = []
    step_rewards = []
    step_fairness = []
    step_utility = []
    for _ in range(args.episodes):
        b_total, _, _ = run_llm_episode(env, base_model, base_tok, difficulty=args.difficulty)
        t_total, _, br = run_llm_episode(env, model, tokenizer, difficulty=args.difficulty)
        baseline_rewards.append(float(np.clip((b_total + 2.0) / 6.0, 0.0, 1.0)))
        trained_rewards.append(float(np.clip((t_total + 2.0) / 6.0, 0.0, 1.0)))
        for row in br:
            step_rewards.append(row["reward"])
            step_fairness.append(row["fairness"])
            step_utility.append(row["utility"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.5))
    plt.plot(np.arange(1, len(step_rewards) + 1), _moving_avg(step_rewards, 7), linewidth=2)
    plt.title("Reward vs Steps (Real LLM)")
    plt.xlabel("Training Step")
    plt.ylabel("Reward (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("reward_vs_steps.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(np.arange(1, len(step_fairness) + 1), _moving_avg(step_fairness, 7), color="#1f77b4", linewidth=2)
    plt.title("Fairness vs Steps (Real LLM)")
    plt.xlabel("Training Step")
    plt.ylabel("Fairness (0-1)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("fairness_vs_steps.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(step_utility, step_fairness, alpha=0.5, s=18)
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
        "model_name": args.model_name,
        "episodes": args.episodes,
        "baseline_avg_reward": round(float(np.mean(baseline_rewards)), 3),
        "trained_avg_reward": round(float(np.mean(trained_rewards)), 3),
        "baseline_rewards": baseline_rewards,
        "trained_rewards": trained_rewards,
        "train_runtime_sec": float(getattr(train_metrics, "metrics", {}).get("train_runtime", 0.0)),
    }
    Path("real_llm_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.output_dir, "real_llm_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # Optional: save adapter
    model.save_pretrained(Path(args.output_dir, "adapter").as_posix())
    tokenizer.save_pretrained(Path(args.output_dir, "adapter").as_posix())


if __name__ == "__main__":
    main()

