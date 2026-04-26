---
title: FairRecovery++
emoji: 🏙️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# 🏙️ FairRecovery++ — Post-Disaster City Recovery RL Environment

[![Llama-1B](https://img.shields.io/badge/Model-Llama--3.2--1B-orange)](https://huggingface.co/Joshua1702/fairrecovery-Llama-3.2-1B)
[![Qwen-7B](https://img.shields.io/badge/Model-Qwen--2.5--7B-purple)](https://huggingface.co/Joshua1702/fairrecovery-Qwen2.5-7B-GRPO)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Theme](https://img.shields.io/badge/Theme-3.1%20%7C%202-orange)](https://huggingface.co/openenv)
[![Space](https://img.shields.io/badge/🤗%20Space-Live-green)](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)

> **Train an LLM to make fair disaster recovery decisions — where helping wealthy zones first systematically abandons the most vulnerable people.**

---

## 🌊 The Problem

After disasters like the 2022 Bengaluru floods and 2023 Chennai floods, city authorities must allocate scarce resources (medical units, power crews, water tankers) across many damaged neighborhoods simultaneously — under tight budgets and time pressure.

The trap every naive AI falls into: **optimising for speed means fixing the easiest zones first**, which are almost always the wealthiest. Zone 0 (wealthy district, moderate damage) is faster to restore than Zone 4 (informal settlement, 92% damage, 96% vulnerable population). A greedy agent picks Zone 0 every time — and Zone 4 stays dark for days.

FairRecovery++ is an OpenEnv RL environment that teaches LLM agents to escape this trap: learn to jointly optimize service restoration *and* equitable distribution across vulnerable populations.

**Primary Theme: 3.1 — Real-World Professional Tasks**  
**Secondary Theme: 2 — Long-Horizon Planning & Instruction Following**

---

## 🎯 The Fairness Trap (Hard Scenario)

| Zone | Damage | Service | Vulnerable Pop | Priority? |
|------|--------|---------|----------------|-----------|
| Zone 0 (wealthy) | 35% | 65% | 8% | ❌ Easy but low need |
| Zone 1 | 50% | 50% | 40% | Medium |
| Zone 2 | 60% | 40% | 55% | Medium |
| Zone 3 (poor) | 72% | 28% | 72% | High |
| **Zone 4 (informal)** | **92%** | **8%** | **96%** | ✅ **Must prioritize** |

A greedy agent always picks Zone 0 (quick ROI, easy reward). A fairness-aware agent learns to prioritize Zone 4 despite lower immediate returns — because that's where 96% of the population is vulnerable.

---

## 🏗️ Architecture

```
LLM Agent (GRPO trained)
        │
        ▼ FairRecoveryAction
┌───────────────────────────────┐
│  Safety Shield (shield.py)    │  ← blocks invalid actions before mutation
│  Stage validator              │
│  Budget enforcer              │
└──────────────┬────────────────┘
               │ valid action
        ▼
┌───────────────────────────────┐
│  FairRecoveryEnvironment      │  ← core OpenEnv Environment class
│  Multi-step protocol:         │
│  analyze → allocate →         │
│  execute → (×MAX_DAYS) →      │
│  submit                       │
└──────────────┬────────────────┘
               │ updated CityState
        ▼
┌───────────────────────────────┐
│  Reward Engine (RLVR)         │  ← no learned reward model
│  R_exec  (service improvement)│
│  R_fair  (disparity reduction)│
│  R_safe  (constraint penalty) │
└──────────────┬────────────────┘
               │ per-step reward
        ▼
┌───────────────────────────────┐
│  Composable Rubrics (RFC 004) │  ← FairnessRubric + UtilityRubric
│  Terminal episode scoring     │     + AnalysisRubric
│  Grader score ∈ (0.01, 0.99)  │
└───────────────────────────────┘
```

---

## 🎮 What the Agent Sees, Does, and Gets Rewarded For

### Observation (per step)
```json
{
  "zones": [
    {"zone_id": 4, "damage": 0.92, "service": 0.08, "vulnerable_ratio": 0.96}
  ],
  "day": 2,
  "budget_left": 25.0,
  "step_stage": "allocate",
  "fairness_score": -0.61,
  "cumulative_reward": 0.142
}
```

### Action (multi-step protocol — not just one choice)
```json
// Step 1: analyze
{"action_type": "analyze", "critical_zones": [3, 4], "reasoning": "highest damage × vulnerability"}

// Step 2: allocate
{"action_type": "allocate", "allocations": [
  {"zone": 4, "resource": "medical"},
  {"zone": 3, "resource": "power"}
]}

// Step 3: execute (commits allocations, receives dense reward)
{"action_type": "execute"}

// After MAX_DAYS: submit (receives terminal bonus)
{"action_type": "submit"}
```

### Reward System (RLVR — all verifiable, no learned model)

| Component | Formula | Weight | What it teaches |
|-----------|---------|--------|-----------------|
| `R_exec` | Avg service improvement this step | 0.5 | Restore services efficiently |
| `R_fair` | −(avg_service_normal − avg_service_vulnerable) | 1.0 | Don't leave vulnerable zones behind |
| `R_safe` | −0.1 × violations | 0.5 | Respect constraints |
| `R_analysis` | Overlap(chosen, top-k by damage×vuln) | 0.1 | Correctly identify critical zones |
| **Terminal bonus** | 0.5×avg_svc + 0.5×(1+R_fair) | — | Long-horizon outcome |

**Grader score: `0.6 × avg_service + 0.4 × (1 + fairness)` clamped to (0.01, 0.99)**

### Anti-Reward-Hacking Measures
- Stage ordering enforced (can't skip analyze → go straight to execute)
- Budget overflow: allocations rejected + penalty, state NOT mutated
- Persistent ignore penalty: if vulnerable zones receive 0 resources for 2+ consecutive days
- Early-submit blocked until MIN_STEPS reached
- Step cap: force-terminate at MAX_STEPS_SAFETY_CAP

---

## 📊 Training Results

### Reward: Baseline vs Trained Agent

![Training Results](asset_final/plots/training_results.png)

*Bar chart: Avg Curriculum Reward, Avg Final Utility, Avg Final Fairness — baseline (grey) vs Qwen-7B trained (blue) across 32 episodes.*

### Per-Episode Reward Heatmap

![Score Heatmap](asset_final/plots/score_heatmap.png)

*Figure 2: Heatmap showing per-episode rewards. The bottom row (trained agent) shows higher sustained rewards in the critical middle-to-late days of recovery compared to the baseline.*

### Reward Curve Over Training

![Training Loss](asset_final/plots/training_loss.png)

*Figure 3: 4-episode moving average. The Qwen-7B agent steadily learns to capture both service restoration and fairness bonuses, outperforming the heuristic greedy baseline after ~20 iterations.*

### Utility vs Fairness Trade-off

![Utility vs Fairness](asset_final/plots/utility_vs_fairness.png)

*Figure 4: Intersectional analysis showing the agent's progress. Unlike greedy agents that cluster in the high-utility/low-fairness quadrant, our trained agent successfully moves towards the 'balanced' zone.*

### Fairness Progress

![Fairness Improvement](asset_final/plots/fairness_vs_episode.png)

*Figure 5: Total Fairness Score across episodes. The training successfully pushed the agent to consider vulnerable zones, resulting in a consistent upward trend in equity achievement.*

### Reward Component Breakdown

![Component Rewards](asset_final/plots/component_rewards.png)

*Figure 6: Decomposed reward components (Utility vs Fairness). This plot highlights how the agent learned to sacrifice small amounts of utility early on to gain significant fairness bonuses, eventually maximizing both by the end of training.*

### 🏆 Dual-Model Performance Comparison

We trained two variants of the FairRecovery agent: an efficient **Llama-1B** and a high-reasoning **Qwen-7B**.

![Model Comparison](asset_final/plots/model_comparison.png)

*Figure 1: Cross-model comparison showing Equity Index (Fairness), Reward, and Utility. Larger models like Qwen-7B achieve near-perfect equity by better interpreting the complex trade-offs between speed and fairness.*

### Key Numbers

| Metric | Greedy Baseline | Llama-3.2-1B | Qwen-2.5-7B-GRPO |
|--------|----------------|--------------|------------------|
| Avg Episode Reward | 0.548 | 0.785 | **0.864** (+57%) |
| Avg Final Fairness | 0.732 | 0.840 | **0.912** (+24%) |
| Avg Final Utility  | 0.545 | 0.712 | **0.808** (+48%) |
| Strategy discovered | Always Zone 0 | Vulnerable-first | Strategic-equitable |

---

## 🚀 Quick Start

```bash
git clone https://github.com/joshua400/FairRecovery-PlusPlus
cd FairRecovery-PlusPlus
pip install -r requirements.txt
uvicorn server.app:app --reload
```

```bash
# Verify environment
curl http://localhost:8000/health

# Run a full episode
python inference.py --difficulty hard --episodes 3 --policy fairness_aware
```

### Use as OpenEnv client
```python
from client import FairRecoveryEnv

env = FairRecoveryEnv(base_url="https://Joshua1702-FairRecovery-PlusPlus.hf.space")
obs = env.reset(difficulty="hard")

for day in range(5):
    action = your_policy(obs)      # analyze → allocate → execute
    obs = env.step(action)
    print(f"Day {obs.day}: reward={obs.reward:+.3f} fair={obs.fairness_score:.3f}")
```

### Run training (Colab)
Open `train_COMPLETE.ipynb` — runs on free Colab T4 in ~10 minutes.

---

## 📁 Project Structure

```
fairrecovery_env/
├── constants.py          # REWARD_WEIGHTS, RESOURCE_COSTS, thresholds
├── models.py             # Pydantic v2 Action / Observation / State
├── state.py              # CityState + ZoneState (mutable world model)
├── tasks.py              # 3 scenarios: easy / medium / hard (fairness trap)
├── rewards.py            # 5-component RLVR reward engine (pure functions)
├── rubrics.py            # RFC 004 composable rubrics
├── shield.py             # Safety validator (blocks before mutation)
└── __init__.py           # Package exports

server/
├── fairrecovery_environment.py   # OpenEnv Environment class
└── app.py                        # FastAPI + OpenEnv integration + Gradio UI

client.py                 # Typed HTTP client
inference.py              # Baseline policies (greedy, fairness-aware, random, HF LLM)
train_COMPLETE.ipynb      # GRPO training notebook (TRL + Unsloth)
generate_summary_plots.py # Reproduce all plots from episode_log.csv
docs/                     # Final project documentation and blog post

---

## 🤗 Models
We provide two pre-trained models for this environment, both trained using GRPO (TRL + Unsloth).

1. **Premium Agent (Best):** [Joshua1702/fairrecovery-Qwen2.5-7B-GRPO](https://huggingface.co/Joshua1702/fairrecovery-Qwen2.5-7B-GRPO)
   - Highest equity index (0.912).
   - Superior reasoning for long-horizon resource management.
2. **Efficient Agent:** [Joshua1702/fairrecovery-Llama-3.2-1B](https://huggingface.co/Joshua1702/fairrecovery-Llama-3.2-1B)
   - Excellent performance for its size (0.840 equity).
   - Low latency, suitable for real-time edge deployment.

---

## 🔗 Materials

| Resource | Link |
|----------|------|
| 🤗 **Live Environment (HF Space)** | [Joshua1702/FairRecovery-PlusPlus](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus) |
| 💻 **GitHub Repository** | [joshua400/FairRecovery-PlusPlus](https://github.com/joshua400/FairRecovery-PlusPlus) |
| 📓 **Llama 1B Training (Final)** | [Notebook](docs/train_llama_final.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/docs/train_llama_final.ipynb) |
| 📓 **Qwen 7B Training (Final)** | [Notebook](docs/train_qwen_final.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/docs/train_qwen_final.ipynb) |
| 📝 **HF Blog Post** | [HF_blog_post.md](docs/HF_blog_post.md) |

---

## 🌍 Why It Matters

Post-disaster recovery planning is a $200B/year global challenge. AI systems that optimize only for speed or total utility **systematically disadvantage the most vulnerable populations** — the elderly, disabled, and low-income communities who live in the hardest-hit zones.

FairRecovery++ is the first OpenEnv environment to encode intersectional fairness as a verifiable, first-class RL objective, making it a research-grade benchmark for safe and fair LLM agent training.

---

## OpenEnv Compliance Checklist

- ✅ `openenv.yaml` manifest present
- ✅ `Environment` base class used with try/import fallback
- ✅ `reset()` / `step()` / `state()` standard API
- ✅ Pydantic v2 typed `Action` / `Observation` / `State`
- ✅ Hosted on HF Spaces (Docker)
- ✅ GRPO training with TRL + Unsloth (see `docs/train_qwen_final.ipynb` and `docs/train_llama_final.ipynb`)
- ✅ Multi-Model evidence: Llama-1B & Qwen-7B plots in `asset_final/plots/`
- ✅ Episode data in `episode_log.csv`
- ✅ Composable rubrics (OpenEnv RFC 004)
- ✅ Anti-reward-hacking: stage gates + persistent ignore penalty

---

*Built for the Meta PyTorch OpenEnv Hackathon India 2026.*
